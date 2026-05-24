# -*- coding: utf-8 -*-
"""
Phase Q43: SWAP Test -- Quantum State Overlap

The SWAP test measures the overlap |<psi|phi>|^2 between two quantum states
using a single measurement, without tomography.

Physical QC: Uses a controlled-SWAP gate and ancilla qubit.
Result: P(ancilla=0) = (1 + |<psi|phi>|^2) / 2

S-Qubit implementation:
  - Prepare two states at different phases
  - Inject both and measure the combined output
  - The E-value should reflect their overlap
  - Compare with theoretical overlap cos^2(delta_phi/2)
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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def inject_measure_E(model, tok, prompt, device, vec, layer, min_tok, max_tok):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[min_tok]) - float(probs[max_tok])


def main():
    print("[Q43] SWAP Test -- Quantum State Overlap")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # SWAP test: measure overlap between states at different phase separations
    n_test = 50
    delta_phis = np.linspace(0, np.pi, n_test)
    reference_phi = 0  # Fixed reference state

    print("\n  Running SWAP test sweep (delta_phi = 0 to pi)...")

    # Method: inject the AVERAGE of two states (simulating interference)
    # Overlap = |<psi|phi>|^2 = cos^2(delta/2)
    E_swap = []
    E_state_a = []
    E_state_b = []
    theoretical_overlap = []

    for delta in delta_phis:
        phi_a = reference_phi
        phi_b = reference_phi + delta

        # State A
        va = phi_vec(phi_a, v0, v1)
        Ea = inject_measure_E(model, tok, prompt, DEVICE, va,
                              INJECT_LAYER, min_tok, max_tok)
        E_state_a.append(Ea)

        # State B
        vb = phi_vec(phi_b, v0, v1)
        Eb = inject_measure_E(model, tok, prompt, DEVICE, vb,
                              INJECT_LAYER, min_tok, max_tok)
        E_state_b.append(Eb)

        # "SWAP" measurement: inject average state (interference)
        v_avg = (va + vb) / 2  # superposition of both
        v_avg = v_avg / v_avg.norm() * va.norm()  # normalize
        E_s = inject_measure_E(model, tok, prompt, DEVICE, v_avg,
                                INJECT_LAYER, min_tok, max_tok)
        E_swap.append(E_s)

        # Theoretical inner product
        overlap = float(torch.dot(va.float(), vb.float()) / (
            va.float().norm() * vb.float().norm()))
        theoretical_overlap.append(overlap ** 2)

    E_swap = np.array(E_swap)
    E_state_a = np.array(E_state_a)
    E_state_b = np.array(E_state_b)
    theoretical_overlap = np.array(theoretical_overlap)

    # Normalize E_swap to [0, 1] range for comparison with overlap
    E_min, E_max = E_swap.min(), E_swap.max()
    if E_max > E_min:
        E_normalized = (E_swap - E_min) / (E_max - E_min)
    else:
        E_normalized = np.ones_like(E_swap) * 0.5

    # Correlation between normalized E and theoretical overlap
    corr = np.corrcoef(E_normalized, theoretical_overlap)[0, 1]

    # Fit quality
    from scipy.optimize import curve_fit
    try:
        def affine(x, a, b):
            return a * x + b
        popt, _ = curve_fit(affine, theoretical_overlap, E_swap)
        E_predicted = affine(theoretical_overlap, *popt)
        ss_res = np.sum((E_swap - E_predicted)**2)
        ss_tot = np.sum((E_swap - E_swap.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
    except Exception:
        r2 = 0.0

    print("\n  SWAP TEST SUMMARY:")
    print("    E range: [%.4f, %.4f]" % (E_swap.min(), E_swap.max()))
    print("    Correlation with overlap: %.4f" % corr)
    print("    Linear fit R2: %.4f" % r2)
    print("    E at delta=0 (identical): %.4f" % E_swap[0])
    print("    E at delta=pi (orthogonal): %.4f" % E_swap[-1])

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: E vs delta_phi
    ax = axes[0]
    ax.plot(delta_phis / np.pi, E_swap, 'ro-', lw=2, ms=4,
            label='SWAP E (measured)')
    ax.plot(delta_phis / np.pi, E_state_a, 'b--', lw=1, alpha=0.5,
            label='State A (reference)')
    ax.plot(delta_phis / np.pi, E_state_b, 'g--', lw=1, alpha=0.5,
            label='State B (varying)')
    ax.set_xlabel('Phase separation (x pi)')
    ax.set_ylabel('E = P(min) - P(max)')
    ax.set_title('(a) SWAP Test E vs Phase Separation',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel B: E vs theoretical overlap
    ax = axes[1]
    ax.scatter(theoretical_overlap, E_swap, c=delta_phis, cmap='coolwarm',
               s=50, zorder=5, edgecolors='black', linewidth=0.5)
    if r2 > 0:
        ax.plot(theoretical_overlap, E_predicted, 'r--', lw=2,
                label='Linear fit (R2=%.3f)' % r2)
    ax.set_xlabel('Theoretical overlap |<a|b>|^2')
    ax.set_ylabel('SWAP E (measured)')
    ax.set_title('(b) E vs Overlap Correlation\nr=%.3f' % corr,
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='delta_phi/pi')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "SWAP Test\n"
        "=========\n\n"
        "Measures overlap between\n"
        "two S-Qubit states\n\n"
        "Results:\n"
        "  Correlation: %.4f\n"
        "  Linear R2:   %.4f\n\n"
        "  E(identical): %.4f\n"
        "  E(orthogonal): %.4f\n\n"
        "Physical SWAP test:\n"
        "  P(0) = (1+|overlap|^2)/2\n\n"
        "S-Qubit SWAP:\n"
        "  E tracks overlap\n"
        "  %s" % (
            corr, r2, E_swap[0], E_swap[-1],
            "monotonically!" if corr > 0.9
            else "with some deviation" if corr > 0.5
            else "weakly")
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E0F7FA', alpha=0.9))

    plt.suptitle('Phase Q43: SWAP Test -- Quantum State Overlap\n'
                 'Measuring S-Qubit state similarity via interference',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q43_swap_test.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q43', 'name': 'swap_test',
        'inject_layer': INJECT_LAYER,
        'n_test': n_test,
        'correlation': round(float(corr), 6),
        'linear_r2': round(float(r2), 6),
        'E_identical': round(float(E_swap[0]), 6),
        'E_orthogonal': round(float(E_swap[-1]), 6),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q43_swap_test.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q43 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
