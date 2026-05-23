# -*- coding: utf-8 -*-
"""
Phase Q8: Multi-Layer Bell Test (Layer-Sweep Interference Map)

Q2 found interference amplitude=0.4984 at injection layer=8.
Q8 sweeps ALL 28 layers to find the "quantum coherence peak":
  - For each layer L in 0..27:
      sweep phi in [0, 4*pi] with 37 points
      inject vec(phi) = cos(phi/2)*min_vec + sin(phi/2)*max_vec at layer L
      record P(min_token)
      compute interference amplitude = (max - min) / 2
  - Result: interference amplitude profile across layers
  - Find the "quantum layer" = argmax of interference amplitude

This answers: WHERE in the LLM does quantum-like coherence peak?
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
N_PHI = 37  # phi sweep points (0 to 4*pi)


def train_soul(model, tok, data, device, layer=8, epochs=150, seed=42):
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
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def get_p_min(model, tok, prompt, device, inject_vec, inject_layer, min_tok_id):
    """Single forward pass with injection, return P(min_token)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[inject_layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[min_tok_id])


def sweep_phi_at_layer(model, tok, prompt, device, min_vec, max_vec,
                        min_tok_id, inject_layer, n_phi=N_PHI):
    """Sweep phi at a given injection layer, return (phis, P_min_vals)."""
    phis = np.linspace(0, 4 * np.pi, n_phi)
    p_vals = []
    for phi in phis:
        vec = (np.cos(phi / 2) * min_vec + np.sin(phi / 2) * max_vec)
        # Normalize to keep scale consistent
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm * min_vec.norm()
        p = get_p_min(model, tok, prompt, device, vec, inject_layer, min_tok_id)
        p_vals.append(p)
    return phis, np.array(p_vals)


def interference_amplitude(p_vals):
    """(max - min) / 2 of the probability curve."""
    return (np.max(p_vals) - np.min(p_vals)) / 2.0


def main():
    print("[Q8] Multi-Layer Bell Test (Layer-Sweep Interference Map)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    num_layers = len(model.model.layers)
    print("  Model: %d layers, hidden=%d" % (num_layers, model.config.hidden_size))

    # Train soul vectors (same as Q1/Q2)
    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"),
    ]
    print("  Training basis vectors (layer 8, 150 epochs)...")
    train_layer = 8
    min_vec = train_soul(model, tok, min_data, DEVICE, layer=train_layer, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, layer=train_layer, seed=99)
    print("  Basis vectors trained.")

    # Token IDs for evaluation
    prompt = "min(7,2)="
    min_tok_id = tok.encode("2")[-1]   # correct min answer for min(7,2)

    # Sweep all layers
    print("  Running Bell Test sweep across all %d layers..." % num_layers)
    layer_amplitudes = []
    layer_phi_curves = {}
    for li in range(num_layers):
        phis, p_vals = sweep_phi_at_layer(
            model, tok, prompt, DEVICE, min_vec, max_vec,
            min_tok_id, inject_layer=li
        )
        amp = interference_amplitude(p_vals)
        layer_amplitudes.append(amp)
        layer_phi_curves[li] = p_vals.tolist()
        if li % 4 == 0 or amp > 0.3:
            print("    L%02d: amp=%.4f  P_min: min=%.4f max=%.4f" % (
                li, amp, p_vals.min(), p_vals.max()))

    peak_layer = int(np.argmax(layer_amplitudes))
    peak_amp = layer_amplitudes[peak_layer]
    print("\n  PEAK: Layer %d with interference amplitude = %.4f" % (peak_layer, peak_amp))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Interference amplitude profile across layers
    ax = axes[0]
    layers = list(range(num_layers))
    colors = ['#E91E63' if a == peak_amp else '#2196F3' for a in layer_amplitudes]
    ax.bar(layers, layer_amplitudes, color=colors, edgecolor='none', alpha=0.85)
    ax.axvline(peak_layer, color='red', linestyle='--', lw=2, alpha=0.8,
               label='Peak L%d (amp=%.3f)' % (peak_layer, peak_amp))
    ax.axhline(0.4984, color='gray', linestyle=':', lw=1.5, label='Q2 baseline (L8=0.498)')
    ax.set_xlabel('Injection Layer', fontsize=11)
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax.set_title('Bell Test: Interference Amplitude\nper Injection Layer', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: phi curves for top-3 layers
    ax = axes[1]
    top3 = np.argsort(layer_amplitudes)[-3:][::-1]
    phis_plot = np.linspace(0, 4 * np.pi, N_PHI)
    palette = ['#E91E63', '#9C27B0', '#2196F3']
    for i, li in enumerate(top3):
        ax.plot(phis_plot / np.pi, layer_phi_curves[li],
                color=palette[i], lw=2,
                label='L%d (amp=%.3f)' % (li, layer_amplitudes[li]))
    ax.set_xlabel('Phase phi (x pi)', fontsize=11)
    ax.set_ylabel('P(min token)', fontsize=11)
    ax.set_title('Interference Fringes\n(Top-3 Layers)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Heatmap of phi curves across ALL layers
    ax = axes[2]
    mat = np.array([layer_phi_curves[li] for li in range(num_layers)])
    im = ax.imshow(mat, aspect='auto', cmap='plasma',
                   extent=[0, 4, 0, num_layers])
    plt.colorbar(im, ax=ax, label='P(min token)')
    ax.axhline(num_layers - peak_layer - 0.5, color='white', lw=1.5, linestyle='--',
               label='Peak L%d' % peak_layer)
    ax.set_xlabel('Phase phi (x pi)', fontsize=11)
    ax.set_ylabel('Injection Layer', fontsize=11)
    ax.set_title('Interference Map\n(All Layers x All Phases)', fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle(
        'Phase Q8: Multi-Layer Bell Test\n'
        '"Where does quantum coherence peak in the LLM?"',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q8_multilayer_bell_test.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    output = {
        'phase': 'Q8', 'name': 'multilayer_bell_test',
        'num_layers': num_layers,
        'n_phi': N_PHI,
        'prompt': prompt,
        'layer_amplitudes': [round(a, 6) for a in layer_amplitudes],
        'peak_layer': peak_layer,
        'peak_amplitude': round(peak_amp, 6),
        'q2_baseline_amp': 0.4984,
        'top3_layers': top3.tolist(),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q8_multilayer_bell_test.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Q8 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
