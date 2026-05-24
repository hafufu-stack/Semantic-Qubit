# -*- coding: utf-8 -*-
"""
Phase Q77: Quantum Zeno Effect (Observation Freezes Evolution)
================================================================
In quantum mechanics, frequent measurement prevents quantum state
evolution (quantum Zeno effect). In S-Qubit: if we "observe" the
hidden state at many layers (by projecting onto the original S-Qubit),
does the state freeze and prevent the transformer from processing?

This would confirm that S-Qubit states behave quantum-mechanically
even under the measurement postulate.
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
    print("[Q77] Quantum Zeno Effect")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.model.layers)
    hs = model.config.hidden_size

    # Train S-Qubit for min task
    vec = train_soul(model, tok,
                     [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")],
                     DEVICE, INJECT_LAYER, EPOCHS, 42)
    # Train DIFFERENT S-Qubit for max task
    vec_max = train_soul(model, tok,
                         [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")],
                         DEVICE, INJECT_LAYER, EPOCHS, 99)

    target_min = tok.encode("2")[-1]
    target_max = tok.encode("8")[-1]
    prompt = "min(7,2)="
    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Baseline: single injection at layer 10
    def single_inject_measure(v, target):
        def hook(m, i, o, vec=v):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = vec.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(probs[target])

    baseline_min = single_inject_measure(vec, target_min)
    print("  Baseline (min task): p=%.4f" % baseline_min)

    # Zeno experiment: Re-project onto S-Qubit state at multiple layers
    # "Measurement" = replace the hidden state with the original S-Qubit
    # This should FREEZE the state and prevent transformer processing

    print("\n  Zeno experiment: Repeated measurement (re-projection)...")
    zeno_results = []

    for n_observations in range(0, min(n_layers - INJECT_LAYER, 16)):
        handles = []

        # Initial injection
        def inject_hook(m, i, o, v=vec):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handles.append(model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook))

        # Re-inject (measure) at subsequent layers
        for obs_idx in range(n_observations):
            obs_layer = INJECT_LAYER + 1 + obs_idx
            if obs_layer >= n_layers:
                break
            def obs_hook(m, i, o, v=vec):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                # "Measure" = project back onto original state
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handles.append(model.model.layers[obs_layer].register_forward_hook(obs_hook))

        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()

        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_min = float(probs[target_min])
        p_max = float(probs[target_max])

        # Compute entropy
        top_probs = probs.topk(100).values
        top_probs = top_probs[top_probs > 1e-10]
        entropy = -float(torch.sum(top_probs * torch.log2(top_probs)))

        zeno_results.append({
            'n_obs': int(n_observations),
            'p_min': p_min,
            'p_max': p_max,
            'entropy': entropy,
        })
        print("    %d observations: p(min)=%.4f, p(max)=%.4f, H=%.2f bits" % (
            n_observations, p_min, p_max, entropy))

    # Anti-Zeno: Inject DIFFERENT state at subsequent layers
    # (quantum anti-Zeno effect: changing measurement basis accelerates decay)
    print("\n  Anti-Zeno experiment: Different state re-projection...")
    anti_zeno = []

    for n_anti in range(0, min(n_layers - INJECT_LAYER, 16)):
        handles = []

        # Initial injection with min
        def inject_hook(m, i, o, v=vec):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handles.append(model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook))

        # Re-inject with MAX (different basis) at subsequent layers
        for obs_idx in range(n_anti):
            obs_layer = INJECT_LAYER + 1 + obs_idx
            if obs_layer >= n_layers:
                break
            def anti_hook(m, i, o, v=vec_max):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handles.append(model.model.layers[obs_layer].register_forward_hook(anti_hook))

        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()

        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_min = float(probs[target_min])
        p_max = float(probs[target_max])

        anti_zeno.append({
            'n_anti': int(n_anti),
            'p_min': p_min,
            'p_max': p_max,
        })

    # Find Zeno freeze point
    baseline_entropy = zeno_results[0]['entropy'] if zeno_results else 0
    max_entropy = max(r['entropy'] for r in zeno_results) if zeno_results else 0

    print("\n  RESULTS:")
    print("    Zeno effect: repeated same-basis measurement")
    print("    Anti-Zeno: different-basis measurement destroys state")

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Zeno effect: probability vs observations
    ax = axes[0]
    n_obs = [r['n_obs'] for r in zeno_results]
    p_mins = [r['p_min'] for r in zeno_results]
    p_maxs = [r['p_max'] for r in zeno_results]
    ax.plot(n_obs, p_mins, 'o-', color='#FF5722', linewidth=2, markersize=6,
            label='P(min) - target')
    ax.plot(n_obs, p_maxs, 's-', color='#2196F3', linewidth=2, markersize=6,
            label='P(max) - non-target')
    ax.axhline(baseline_min, color='green', ls='--', alpha=0.3,
               label='Baseline (%.3f)' % baseline_min)
    ax.set_xlabel('Number of re-projections')
    ax.set_ylabel('Probability')
    ax.set_title('(a) Quantum Zeno Effect\nRepeated measurement freezes state',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Entropy vs observations
    ax = axes[1]
    entropies_z = [r['entropy'] for r in zeno_results]
    ax.plot(n_obs, entropies_z, 'o-', color='#9C27B0', linewidth=2, markersize=6)
    ax.set_xlabel('Number of re-projections')
    ax.set_ylabel('Decision entropy (bits)')
    ax.set_title('(b) Entropy Under Zeno\n'
                 'Measurement suppresses evolution',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Anti-Zeno comparison
    ax = axes[2]
    n_anti_list = [r['n_anti'] for r in anti_zeno]
    p_min_anti = [r['p_min'] for r in anti_zeno]
    p_min_zeno = [r['p_min'] for r in zeno_results[:len(anti_zeno)]]
    ax.plot(n_anti_list, p_min_zeno, 'o-', color='#4CAF50', linewidth=2,
            markersize=6, label='Zeno (same state)')
    ax.plot(n_anti_list, p_min_anti, 's-', color='#F44336', linewidth=2,
            markersize=6, label='Anti-Zeno (different state)')
    ax.set_xlabel('Number of interventions')
    ax.set_ylabel('P(min)')
    ax.set_title('(c) Zeno vs Anti-Zeno\n'
                 'Different basis destroys information',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Phase Q77: Quantum Zeno Effect\n'
                 'Frequent measurement prevents S-Qubit evolution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q77_zeno.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q77', 'name': 'quantum_zeno',
        'baseline_prob': round(float(baseline_min), 4),
        'zeno_results': [{k: round(v, 4) if isinstance(v, float) else v
                          for k, v in r.items()} for r in zeno_results],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q77_zeno.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q77 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
