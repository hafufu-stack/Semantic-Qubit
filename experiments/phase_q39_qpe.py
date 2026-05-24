# -*- coding: utf-8 -*-
"""
Phase Q39: Quantum Phase Estimation (QPE)

Given a unitary U and eigenstate |psi>, estimate the eigenphase theta
where U|psi> = e^(2*pi*i*theta)|psi>.

This is the core subroutine of Shor's algorithm and many quantum algorithms.

S-Qubit implementation:
  - The "unitary" is the LLM's forward pass at a specific layer
  - The "eigenstate" is a soul vector
  - We estimate the phase by sweeping phi and finding the peak
  - Test: inject states at KNOWN phases, see if QPE recovers them

CPU+GPU hybrid: train on GPU, sweep on GPU.
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
    print("[Q39] Quantum Phase Estimation (QPE)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompts = ["min(7,2)=", "min(9,1)=", "max(1,8)=", "min(5,3)="]
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # QPE: Given an unknown phase theta, estimate it by scanning
    # Step 1: Create calibration curve E(phi)
    print("\n  Step 1: Building calibration curve...")
    n_cal = 100
    cal_phis = np.linspace(0, 2 * np.pi, n_cal)
    cal_E = []
    for phi in cal_phis:
        v = phi_vec(phi, v0, v1)
        E = inject_measure_E(model, tok, "min(7,2)=", DEVICE, v,
                             INJECT_LAYER, min_tok, max_tok)
        cal_E.append(E)
    cal_E = np.array(cal_E)
    print("  Calibration range: [%.4f, %.4f]" % (cal_E.min(), cal_E.max()))

    # Step 2: QPE -- inject unknown states and estimate their phase
    print("\n  Step 2: Phase estimation trials...")
    # True phases to test
    true_phases = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2,
                   2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi,
                   7*np.pi/6, 5*np.pi/4, 4*np.pi/3, 3*np.pi/2,
                   5*np.pi/3, 7*np.pi/4, 11*np.pi/6]
    phase_names = ["0", "pi/6", "pi/4", "pi/3", "pi/2",
                   "2pi/3", "3pi/4", "5pi/6", "pi",
                   "7pi/6", "5pi/4", "4pi/3", "3pi/2",
                   "5pi/3", "7pi/4", "11pi/6"]

    estimation_results = []

    for true_phi, name in zip(true_phases, phase_names):
        # Inject the unknown state
        v_unknown = phi_vec(true_phi, v0, v1)

        # Measure E using multiple prompts and average
        E_measurements = []
        for prompt in prompts:
            E = inject_measure_E(model, tok, prompt, DEVICE, v_unknown,
                                 INJECT_LAYER, min_tok, max_tok)
            E_measurements.append(E)
        E_avg = np.mean(E_measurements)

        # Estimate phase: find closest E in calibration curve
        best_idx = np.argmin(np.abs(cal_E - E_avg))
        estimated_phi = cal_phis[best_idx]

        # Error (circular)
        error = min(abs(estimated_phi - true_phi),
                    2*np.pi - abs(estimated_phi - true_phi))

        estimation_results.append({
            'true_phi': round(float(true_phi), 6),
            'estimated_phi': round(float(estimated_phi), 6),
            'E_measured': round(float(E_avg), 6),
            'error_rad': round(float(error), 6),
            'name': name,
        })
        print("    %8s: true=%.3f, est=%.3f, err=%.4f rad" % (
            name, true_phi, estimated_phi, error))

    # Statistics
    errors = [r['error_rad'] for r in estimation_results]
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    # Precision: how many bits of phase can we resolve?
    precision_bits = -np.log2(mean_error / (2*np.pi) + 1e-10) if mean_error > 0 else 10

    # Step 3: Resolution test -- how close can two phases be distinguished?
    print("\n  Step 3: Resolution limit test...")
    deltas = [np.pi/2, np.pi/4, np.pi/8, np.pi/16, np.pi/32, np.pi/64]
    resolution_results = []
    base_phi = np.pi / 2

    for delta in deltas:
        phi_a = base_phi
        phi_b = base_phi + delta

        v_a = phi_vec(phi_a, v0, v1)
        v_b = phi_vec(phi_b, v0, v1)

        E_a = inject_measure_E(model, tok, "min(7,2)=", DEVICE, v_a,
                                INJECT_LAYER, min_tok, max_tok)
        E_b = inject_measure_E(model, tok, "min(7,2)=", DEVICE, v_b,
                                INJECT_LAYER, min_tok, max_tok)

        distinguishable = abs(E_a - E_b) > 0.001
        resolution_results.append({
            'delta': round(float(delta), 6),
            'delta_deg': round(float(np.degrees(delta)), 2),
            'E_a': round(float(E_a), 6),
            'E_b': round(float(E_b), 6),
            'diff': round(float(abs(E_a - E_b)), 6),
            'distinguishable': bool(distinguishable),
        })
        print("    delta=%.4f (%.1f deg): E_a=%.4f, E_b=%.4f, diff=%.6f %s" % (
            delta, np.degrees(delta), E_a, E_b, abs(E_a - E_b),
            "OK" if distinguishable else "BELOW LIMIT"))

    print("\n  QPE SUMMARY:")
    print("    Mean phase error: %.4f rad (%.1f deg)" % (mean_error, np.degrees(mean_error)))
    print("    Max phase error:  %.4f rad (%.1f deg)" % (max_error, np.degrees(max_error)))
    print("    Precision: ~%.1f bits" % precision_bits)
    min_resolvable = None
    for r in resolution_results:
        if r['distinguishable']:
            min_resolvable = r['delta_deg']
    print("    Min resolution: %s deg" % (
        "%.1f" % min_resolvable if min_resolvable else "N/A"))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Calibration curve
    ax = axes[0]
    ax.plot(cal_phis / np.pi, cal_E, 'b-', lw=2)
    true_vals = [r['true_phi'] / np.pi for r in estimation_results]
    est_vals = [r['E_measured'] for r in estimation_results]
    ax.scatter(true_vals, est_vals, c='red', s=60, zorder=5, label='Test points')
    ax.set_xlabel('Phase (x pi)')
    ax.set_ylabel('E = P(min) - P(max)')
    ax.set_title('(a) QPE Calibration Curve\nE vs phase mapping', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: Estimation accuracy
    ax = axes[1]
    true_arr = np.array([r['true_phi'] for r in estimation_results])
    est_arr = np.array([r['estimated_phi'] for r in estimation_results])
    ax.scatter(true_arr / np.pi, est_arr / np.pi, c='red', s=60, zorder=5)
    ax.plot([0, 2], [0, 2], 'b--', lw=1.5, alpha=0.5, label='Perfect estimation')
    ax.set_xlabel('True phase (x pi)')
    ax.set_ylabel('Estimated phase (x pi)')
    ax.set_title('(b) Phase Estimation Accuracy\nMean error: %.1f deg' % np.degrees(mean_error),
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Panel C: Resolution
    ax = axes[2]
    deltas_deg = [r['delta_deg'] for r in resolution_results]
    diffs = [r['diff'] for r in resolution_results]
    colors = ['green' if r['distinguishable'] else 'red' for r in resolution_results]
    ax.bar(range(len(deltas_deg)), diffs, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(deltas_deg)))
    ax.set_xticklabels(['%.1f' % d for d in deltas_deg], fontsize=9)
    ax.axhline(0.001, color='red', ls='--', lw=1.5, label='Detection threshold')
    ax.set_xlabel('Phase separation (degrees)')
    ax.set_ylabel('|E_a - E_b|')
    ax.set_title('(c) Phase Resolution\nGreen=distinguishable', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q39: Quantum Phase Estimation\n'
                 'Recovering unknown phases from S-Qubit measurements',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q39_qpe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q39', 'name': 'quantum_phase_estimation',
        'inject_layer': INJECT_LAYER,
        'n_calibration': n_cal,
        'mean_error_rad': round(float(mean_error), 6),
        'max_error_rad': round(float(max_error), 6),
        'precision_bits': round(float(precision_bits), 2),
        'estimation_results': estimation_results,
        'resolution_results': resolution_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q39_qpe.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q39 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
