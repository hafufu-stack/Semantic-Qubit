# -*- coding: utf-8 -*-
"""
Phase Q14: Two-Qubit Coupling vs Layer Distance

Q12v3 found: SQ2@L16,pos=-2 -> SQ1@L8,pos=-1 cross-coupling amp = 0.4020
              Separability residual = 0.27 (163x larger than additive case)

Q14 asks: how does cross-coupling DEPEND ON LAYER DISTANCE between SQ1 and SQ2?
  - Fix SQ1@L8, pos=-1 (same as Q12v3)
  - Sweep SQ2 at layers [9, 10, 12, 14, 16, 18, 20, 22, 24, 26], pos=-2
  - Measure: SQ2->SQ1 cross-coupling amplitude (fix SQ1=|0>, vary phi2)

Physical interpretation:
  - Coupling decreases with distance -> local attention mechanism
  - Coupling increases with distance -> global information mixing
  - Coupling is flat -> attention is position-independent beyond a threshold
  - Peak at some middle layer -> resonance (entanglement sweet spot)

This identifies the optimal SQ2 placement for maximum 2-qubit coupling.
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

SQ1_LAYER = 8
SQ1_POS   = -1
SQ2_POS   = -2
N_PHI_CROSS = 25  # phi points for cross-coupling measurement
EPOCHS_SQ2  = 75   # faster training for SQ2 sweeps


def train_soul(model, tok, data, device, layer, pos=-1, epochs=100, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            seq_len = inp['input_ids'].shape[1]
            actual_pos = pos if pos >= 0 else seq_len + pos
            def hook(m, i, o, v=vec, p=actual_pos):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, p, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def two_qubit_pos_forward(model, tok, prompt, device, v1, v2, l1, p1, l2, p2):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    a1 = p1 if p1 >= 0 else seq_len + p1
    a2 = p2 if p2 >= 0 else seq_len + p2

    def hook1(m, i, o, v=v1, p=a1):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    def hook2(m, i, o, v=v2, p=a2):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    h1 = model.model.layers[l1].register_forward_hook(hook1)
    h2 = model.model.layers[l2].register_forward_hook(hook2)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def measure_cross_coupling(model, tok, prompt, device, sq1_vec, sq2_0, sq2_1,
                             sq2_layer, sq1_tok, sq1_tok_1, n_phi=N_PHI_CROSS):
    """Fix SQ1=|0>, sweep SQ2 phi -> measure cross-coupling amplitude at pos-1."""
    phis = np.linspace(0, 2 * np.pi, n_phi)
    E_vals = []
    sq1_norm = sq1_vec.norm()
    for phi in phis:
        v2 = np.cos(phi / 2) * sq2_0 + np.sin(phi / 2) * sq2_1
        n2 = v2.norm()
        if n2 > 0:
            v2 = v2 / n2 * sq2_0.norm()
        probs = two_qubit_pos_forward(
            model, tok, prompt, DEVICE,
            sq1_vec, v2, SQ1_LAYER, SQ1_POS, sq2_layer, SQ2_POS)
        E_vals.append(float(probs[sq1_tok]) - float(probs[sq1_tok_1]))
    E_arr = np.array(E_vals)
    amp = (E_arr.max() - E_arr.min()) / 2.0
    return amp, E_arr


def main():
    print("[Q14] Two-Qubit Coupling vs SQ2 Layer Distance")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    num_layers = len(model.model.layers)
    print("  SQ1 fixed at L%d, pos=%d. Sweeping SQ2 layer..." % (SQ1_LAYER, SQ1_POS))

    prompt = "min(7,2)="
    sq1_tok = tok.encode("2")[-1]
    sq1_tok_1 = tok.encode("7")[-1]

    sq1_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                  ("min(4,6)=","4"),("min(9,3)=","3")]
    sq1_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                  ("min(4,6)=","6"),("min(9,3)=","9")]
    # SQ2: color domain (same as Q12v3)
    sq2_0_data = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                  ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_1_data = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                  ("The forest is","green"),("Grass color is","green")]

    print("  Training SQ1@L%d pos=%d (100 epochs)..." % (SQ1_LAYER, SQ1_POS))
    sq1_min = train_soul(model, tok, sq1_0_data, DEVICE, SQ1_LAYER, SQ1_POS, 100, 42)

    # SQ2 layer sweep
    sq2_layers = [9, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    # Also include SQ1_LAYER itself (same layer, different position)
    sq2_layers_full = [SQ1_LAYER] + sq2_layers

    results = {}
    print("\n  Sweeping SQ2 layers (train %d epochs each)..." % EPOCHS_SQ2)
    for sq2_l in sq2_layers_full:
        dist = sq2_l - SQ1_LAYER
        # Train SQ2 at this layer, pos=-2
        sq2_0 = train_soul(model, tok, sq2_0_data, DEVICE, sq2_l, SQ2_POS, EPOCHS_SQ2, 42)
        sq2_1 = train_soul(model, tok, sq2_1_data, DEVICE, sq2_l, SQ2_POS, EPOCHS_SQ2, 99)

        amp, E_arr = measure_cross_coupling(
            model, tok, prompt, DEVICE, sq1_min, sq2_0, sq2_1,
            sq2_l, sq1_tok, sq1_tok_1)

        results[sq2_l] = {
            'distance': dist,
            'cross_coupling_amp': round(float(amp), 6),
            'E_min': round(float(E_arr.min()), 4),
            'E_max': round(float(E_arr.max()), 4),
        }
        print("    SQ2@L%02d (dist=%+2d): cross_amp=%.4f  E:[%.4f, %.4f]" % (
            sq2_l, dist, amp, E_arr.min(), E_arr.max()))

    peak_layer = max(results, key=lambda l: results[l]['cross_coupling_amp'])
    peak_amp = results[peak_layer]['cross_coupling_amp']
    print("\n  PEAK coupling: SQ2@L%d (dist=%d, amp=%.4f)" % (
        peak_layer, results[peak_layer]['distance'], peak_amp))

    # === PLOT ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sq2_ls = sorted(results.keys())
    dists = [results[l]['distance'] for l in sq2_ls]
    amps  = [results[l]['cross_coupling_amp'] for l in sq2_ls]

    # Panel 1: Coupling amplitude vs SQ2 layer
    ax = axes[0]
    bar_colors = ['#E91E63' if l == peak_layer else '#2196F3' for l in sq2_ls]
    ax.bar(sq2_ls, amps, color=bar_colors, edgecolor='black', alpha=0.85, width=1.2)
    ax.axvline(SQ1_LAYER, color='gray', linestyle='--', lw=2, label='SQ1@L%d' % SQ1_LAYER)
    ax.axvline(peak_layer, color='red', linestyle=':', lw=2,
               label='Peak coupling@L%d (%.3f)' % (peak_layer, peak_amp))
    ax.set_xlabel('SQ2 Layer', fontsize=11)
    ax.set_ylabel('Cross-Coupling Amplitude', fontsize=11)
    ax.set_title('SQ2->SQ1 Coupling vs SQ2 Layer\n(SQ1 fixed at L%d, pos=-1)' % SQ1_LAYER,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Coupling vs distance (offset from SQ1)
    ax = axes[1]
    ax.plot(dists, amps, '#9C27B0', lw=2, marker='o', ms=8)
    ax.axhline(0.4020, color='red', linestyle='--', lw=1.5,
               label='Q12v3 L16 coupling=0.402')
    # Annotate peak
    peak_dist = results[peak_layer]['distance']
    ax.annotate('Peak\n@dist=%d\namp=%.3f' % (peak_dist, peak_amp),
                xy=(peak_dist, peak_amp), xytext=(peak_dist+1, peak_amp-0.05),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel('Layer Distance from SQ1 (SQ2 - SQ1)', fontsize=11)
    ax.set_ylabel('Cross-Coupling Amplitude', fontsize=11)
    ax.set_title('Coupling vs Layer Distance\n"Does attention couple near or far layers??"',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle(
        'Phase Q14: Two-Qubit Coupling Strength vs Layer Distance\n'
        'SQ1@L%d,pos=-1  |  SQ2@pos=-2 swept across layers' % SQ1_LAYER,
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q14_coupling_distance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q14', 'name': 'coupling_distance',
        'sq1_layer': SQ1_LAYER, 'sq1_pos': SQ1_POS, 'sq2_pos': SQ2_POS,
        'sq2_epochs': EPOCHS_SQ2,
        'results': {str(l): results[l] for l in sq2_ls},
        'peak_layer': peak_layer,
        'peak_amplitude': peak_amp,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q14_coupling_distance.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q14 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
