# -*- coding: utf-8 -*-
"""
Phase Q61: Quantum Zeno Effect
===============================
The quantum Zeno effect: frequent measurement prevents a quantum
state from evolving. "A watched pot never boils."

Test: If we repeatedly "observe" (read logits) at multiple layers
during a forward pass, does the S-Qubit state freeze, preventing
the natural evolution through transformer layers?

This has implications for S-Qubit control: we can use intermediate
observations to stabilize quantum states.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INJECT_LAYER = 10
EPOCHS = 100


def train_soul(model, tok, data, device, layer, epochs=EPOCHS, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def hook(m, i, o, v=vec):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def main():
    print("[Q61] Quantum Zeno Effect")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.model.layers)
    hs = model.config.hidden_size
    
    # Train S-Qubit
    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    vec = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    
    prompt = "min(7,2)="
    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Experiment 1: No observation (free evolution)
    # Inject at layer 10, let it evolve freely, measure final hidden state at each layer
    print("\n  Exp 1: Free evolution (no intermediate observation)")
    free_states = []
    handles = []
    
    for layer_idx in range(n_layers):
        captured = {}
        def make_capture(cap):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                cap['h'] = h[0, -1, :].detach().clone()
            return hook
        handles.append(model.model.layers[layer_idx].register_forward_hook(make_capture(captured)))
        free_states.append(captured)
    
    # Inject S-Qubit
    def inject_hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    inject_handle = model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook)
    
    with torch.no_grad():
        model(**inp)
    
    inject_handle.remove()
    for h in handles:
        h.remove()
    
    # Compute drift from injected state at each layer
    ref_state = vec.float().cpu().numpy()
    free_drifts = []
    for layer_idx in range(n_layers):
        if layer_idx >= INJECT_LAYER and 'h' in free_states[layer_idx]:
            state = free_states[layer_idx]['h'].float().cpu().numpy()
            cos_sim = np.dot(ref_state, state) / (np.linalg.norm(ref_state) * np.linalg.norm(state) + 1e-10)
            free_drifts.append((layer_idx, float(cos_sim)))
    
    # Experiment 2: Zeno effect (re-inject at every N layers)
    print("  Exp 2: Zeno effect (periodic re-injection)")
    
    zeno_results = {}  # freq -> final_similarity
    
    for reinject_freq in [1, 2, 3, 5, 8, 12]:
        # Re-inject the S-Qubit every reinject_freq layers after injection
        reinject_layers = list(range(INJECT_LAYER, n_layers, reinject_freq))
        
        zeno_states = []
        handles = []
        
        for layer_idx in range(n_layers):
            captured = {}
            def make_capture2(cap):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    cap['h'] = h[0, -1, :].detach().clone()
                return hook
            handles.append(model.model.layers[layer_idx].register_forward_hook(make_capture2(captured)))
            zeno_states.append(captured)
        
        # Multi-layer injection hooks
        inject_handles = []
        for rl in reinject_layers:
            def make_inject(v=vec):
                def hook(m, i, o):
                    h = (o[0] if isinstance(o, tuple) else o).clone()
                    h[0, -1, :] = v.to(h.dtype)
                    return (h,) + o[1:] if isinstance(o, tuple) else h
                return hook
            inject_handles.append(model.model.layers[rl].register_forward_hook(make_inject()))
        
        with torch.no_grad():
            out = model(**inp)
        
        for h in handles:
            h.remove()
        for h in inject_handles:
            h.remove()
        
        # Final layer similarity to original S-Qubit
        final_state = zeno_states[-1]['h'].float().cpu().numpy() if 'h' in zeno_states[-1] else None
        if final_state is not None:
            final_sim = np.dot(ref_state, final_state) / (
                np.linalg.norm(ref_state) * np.linalg.norm(final_state) + 1e-10)
        else:
            final_sim = 0
        
        # Also get logit for target
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        target_id = tok.encode("2")[-1]
        target_prob = float(probs[target_id])
        
        zeno_results[reinject_freq] = {
            'final_similarity': float(final_sim),
            'target_prob': target_prob,
            'n_injections': len(reinject_layers),
        }
        print("    freq=%d: final_sim=%.4f, p(target)=%.4f, n_inject=%d" % (
            reinject_freq, final_sim, target_prob, len(reinject_layers)))

    # Baseline: no injection at all
    with torch.no_grad():
        out_base = model(**inp)
    base_prob = float(torch.softmax(out_base.logits[0, -1, :].float(), dim=-1)[tok.encode("2")[-1]])
    
    # Single injection result
    inject_handle2 = model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook)
    with torch.no_grad():
        out_single = model(**inp)
    inject_handle2.remove()
    single_prob = float(torch.softmax(out_single.logits[0, -1, :].float(), dim=-1)[tok.encode("2")[-1]])

    print("\n  RESULTS:")
    print("    No injection: p(2)=%.4f" % base_prob)
    print("    Single injection (L%d): p(2)=%.4f" % (INJECT_LAYER, single_prob))
    for freq, res in sorted(zeno_results.items()):
        print("    Zeno freq=%d: p(2)=%.4f, sim=%.4f" % (freq, res['target_prob'], res['final_similarity']))
    
    # Zeno strength: does frequent observation freeze the state?
    freq1_sim = zeno_results[1]['final_similarity']
    free_final_sim = free_drifts[-1][1] if free_drifts else 0
    zeno_strength = freq1_sim - free_final_sim

    print("    Zeno strength: %.4f (positive = freezing confirmed)" % zeno_strength)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Free evolution drift
    ax = axes[0]
    if free_drifts:
        layers_f, sims_f = zip(*free_drifts)
        ax.plot(layers_f, sims_f, 'o-', color='#2196F3', linewidth=2,
                markersize=4, label='Free evolution')
    ax.axhline(1.0, color='green', ls='--', alpha=0.5, label='Perfect preservation')
    ax.axvline(INJECT_LAYER, color='red', ls=':', alpha=0.5, label='Injection layer')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Cosine similarity to S-Qubit')
    ax.set_title('(a) S-Qubit State Drift\nFree evolution through layers',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Zeno effect: similarity vs re-injection frequency
    ax = axes[1]
    freqs = sorted(zeno_results.keys())
    sims = [zeno_results[f]['final_similarity'] for f in freqs]
    probs_z = [zeno_results[f]['target_prob'] for f in freqs]
    ax.plot(freqs, sims, 'o-', color='#FF5722', linewidth=2, markersize=8, label='State similarity')
    if free_drifts:
        ax.axhline(free_final_sim, color='#2196F3', ls='--', alpha=0.7, label='Free evolution')
    ax.set_xlabel('Re-injection frequency (every N layers)')
    ax.set_ylabel('Final state similarity to S-Qubit')
    ax.set_title('(b) Quantum Zeno Effect\nFrequent observation freezes state',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Task performance vs observation frequency
    ax = axes[2]
    ax.plot(freqs, probs_z, 's-', color='#4CAF50', linewidth=2, markersize=8,
            label='Zeno injection')
    ax.axhline(single_prob, color='#FF5722', ls='--', alpha=0.7,
               label='Single injection')
    ax.axhline(base_prob, color='gray', ls=':', alpha=0.5,
               label='No injection')
    ax.set_xlabel('Re-injection frequency (every N layers)')
    ax.set_ylabel('P(correct answer)')
    ax.set_title('(c) Task Performance\nOptimal observation frequency',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle('Phase Q61: Quantum Zeno Effect in S-Qubit Systems\n'
                 '"A watched qubit never decoheres"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q61_zeno.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q61', 'name': 'quantum_zeno_effect',
        'base_prob': round(base_prob, 4),
        'single_inject_prob': round(single_prob, 4),
        'zeno_results': {str(k): v for k, v in zeno_results.items()},
        'free_final_similarity': round(float(free_final_sim), 4) if free_drifts else None,
        'zeno_strength': round(float(zeno_strength), 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q61_zeno.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q61 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
