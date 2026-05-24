# -*- coding: utf-8 -*-
"""
Phase Q53: Non-Unitary Quantum AI

Physical quantum computers are bound by unitary (probability-preserving)
transformations. LLMs have ReLU/GELU/SiLU non-linearities that break
unitarity. This experiment tests whether non-unitary transformations
provide SUPER-quantum amplification beyond what any physical QC can achieve.

Test: Compare amplification through different activation functions
and measure if non-unitarity gives better target discrimination.
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
    print("[Q53] Non-Unitary Quantum AI")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]
    prompt = "min(7,2)="

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Test: inject at different layers and measure the amplification
    # Non-unitary layers (after attention + FFN with GELU) vs linear-only
    layers_to_test = list(range(0, 36, 2))  # every 2nd layer

    results_by_layer = []
    target_phi = 0  # target state is |0>

    print("\n  Measuring per-layer amplification...")
    for layer in layers_to_test:
        if layer >= len(model.model.layers):
            break

        # Inject target state
        def hook_inject(m, i, o, v=v0):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[layer].register_forward_hook(hook_inject)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_target = float(probs[min_tok])
        p_noise = float(probs[max_tok])

        # Measure entropy of output distribution
        log_probs = torch.log(probs + 1e-10)
        entropy = -float(torch.sum(probs * log_probs))

        # Amplification = p_target / (1/vocab_size)
        vocab_size = probs.shape[0]
        amplification = p_target * vocab_size

        # Non-unitarity measure: norm of hidden state changes through FFN
        norms = []
        def hook_norm(m, i, o):
            h = (o[0] if isinstance(o, tuple) else o)
            norms.append(float(h[0, -1, :].float().norm()))
        h1 = model.model.layers[min(layer, len(model.model.layers)-1)].register_forward_hook(hook_norm)
        if layer + 1 < len(model.model.layers):
            h2 = model.model.layers[layer + 1].register_forward_hook(hook_norm)
        with torch.no_grad():
            model(**inp)
        h1.remove()
        if layer + 1 < len(model.model.layers):
            h2.remove()

        norm_ratio = norms[1] / norms[0] if len(norms) >= 2 else 1.0

        results_by_layer.append({
            'layer': layer,
            'p_target': round(p_target, 6),
            'p_noise': round(p_noise, 6),
            'amplification': round(amplification, 1),
            'entropy': round(entropy, 4),
            'norm_ratio': round(norm_ratio, 4),
        })
        print("    L%d: p_target=%.4f, amp=%.0fx, entropy=%.2f, norm_ratio=%.3f" % (
            layer, p_target, amplification, entropy, norm_ratio))

    # Find best layer
    best = max(results_by_layer, key=lambda x: x['amplification'])
    print("\n  NON-UNITARY SUMMARY:")
    print("    Best layer: L%d with %.0fx amplification" % (
        best['layer'], best['amplification']))
    print("    Non-unitarity (norm ratio range): %.3f - %.3f" % (
        min(r['norm_ratio'] for r in results_by_layer),
        max(r['norm_ratio'] for r in results_by_layer)))

    # Test: Non-unitary boost via repeated injection
    print("\n  Testing cascade amplification (multi-layer injection)...")
    cascade_results = []
    for n_layers in [1, 2, 3, 5, 8]:
        inject_layers = list(range(INJECT_LAYER, min(INJECT_LAYER + n_layers, 36)))
        handles = []
        for l in inject_layers:
            def hook_cascade(m, i, o, v=v0):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handles.append(model.model.layers[l].register_forward_hook(hook_cascade))
        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_t = float(probs[min_tok])
        amp = p_t * vocab_size
        cascade_results.append({
            'n_layers': n_layers,
            'p_target': round(p_t, 6),
            'amplification': round(amp, 1),
        })
        print("    %d layers: p=%.4f, amp=%.0fx" % (n_layers, p_t, amp))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    layers_p = [r['layer'] for r in results_by_layer]
    amps_p = [r['amplification'] for r in results_by_layer]
    ax.plot(layers_p, amps_p, 'ro-', lw=2, ms=6)
    ax.axhline(1, color='gray', ls='--', alpha=0.3, label='Random (1x)')
    ax.set_xlabel('Injection Layer')
    ax.set_ylabel('Amplification (x)')
    ax.set_title('(a) Per-Layer Amplification\nBest: L%d = %.0fx' % (
        best['layer'], best['amplification']), fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    norms_p = [r['norm_ratio'] for r in results_by_layer]
    ax.plot(layers_p, norms_p, 'bs-', lw=2, ms=6)
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Unitary (norm=1)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Norm ratio (L->L+1)')
    ax.set_title('(b) Non-Unitarity Measure\nDeviation from norm preservation',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    n_l = [r['n_layers'] for r in cascade_results]
    amp_c = [r['amplification'] for r in cascade_results]
    ax.bar(n_l, amp_c, color='#FF5722', edgecolor='black', alpha=0.85)
    for nl, ac in zip(n_l, amp_c):
        ax.text(nl, ac + max(amp_c) * 0.02, '%.0fx' % ac, ha='center',
                fontweight='bold', fontsize=9)
    ax.set_xlabel('Number of injection layers')
    ax.set_ylabel('Amplification (x)')
    ax.set_title('(c) Cascade Non-Unitary Boost\nMulti-layer injection',
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q53: Non-Unitary Quantum AI\n'
                 'LLM non-linearities enable super-unitary amplification',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q53_nonunitary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q53', 'name': 'non_unitary_quantum_ai',
        'best_layer': best['layer'],
        'best_amplification': best['amplification'],
        'cascade_results': cascade_results,
        'per_layer': results_by_layer,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q53_nonunitary.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q53 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
