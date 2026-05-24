# -*- coding: utf-8 -*-
"""
Phase Q49: Quantum State Tomography

Reconstruct the full "quantum state" of an S-Qubit from measurements
in multiple bases. In physical QC, state tomography requires O(4^n)
measurements for n qubits.

S-Qubit tomography:
  1. Prepare a state at unknown phase theta
  2. Measure in 3 "bases" (different prompts)
  3. Reconstruct theta from the measurements
  4. Compare with true state
  5. Test: how many bases are needed for accurate reconstruction?
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
    print("[Q49] Quantum State Tomography")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    # Multiple "measurement bases" = different prompts
    prompts = [
        "min(7,2)=",    # Z-basis
        "min(9,1)=",    # X-basis
        "max(1,8)=",    # Y-basis
        "min(5,3)=",    # ZX-basis
        "max(2,9)=",    # ZY-basis
    ]

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # Step 1: Build calibration curves for each basis
    n_cal = 50
    cal_phis = np.linspace(0, 2 * np.pi, n_cal)
    calibrations = {}

    print("\n  Building calibration curves for %d bases..." % len(prompts))
    for pi, prompt in enumerate(prompts):
        E_cal = []
        for phi in cal_phis:
            v = phi_vec(phi, v0, v1)
            E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                 INJECT_LAYER, min_tok, max_tok)
            E_cal.append(E)
        calibrations[pi] = np.array(E_cal)
        print("    Basis %d: E range [%.4f, %.4f]" % (
            pi, calibrations[pi].min(), calibrations[pi].max()))

    # Step 2: Tomography -- reconstruct unknown states
    n_test = 20
    np.random.seed(42)
    test_phis = np.random.uniform(0, 2 * np.pi, n_test)

    print("\n  Reconstructing %d unknown states..." % n_test)

    # Test with increasing number of bases
    results_by_nbases = {}

    for n_bases in [1, 2, 3, 5]:
        errors = []

        for true_phi in test_phis:
            v_unknown = phi_vec(true_phi, v0, v1)

            # Measure in n_bases bases
            measurements = []
            for bi in range(n_bases):
                E = inject_measure_E(model, tok, prompts[bi], DEVICE, v_unknown,
                                     INJECT_LAYER, min_tok, max_tok)
                measurements.append(E)

            # Reconstruct: find phi that best matches all measurements
            best_phi = 0
            best_score = float('inf')
            for ci, phi_candidate in enumerate(cal_phis):
                score = 0
                for bi in range(n_bases):
                    score += (measurements[bi] - calibrations[bi][ci]) ** 2
                if score < best_score:
                    best_score = score
                    best_phi = phi_candidate

            # Circular error
            error = min(abs(best_phi - true_phi),
                        2 * np.pi - abs(best_phi - true_phi))
            errors.append(error)

        mean_error = np.mean(errors)
        max_error = np.max(errors)
        median_error = np.median(errors)
        precision_bits = -np.log2(mean_error / (2*np.pi) + 1e-10)

        results_by_nbases[n_bases] = {
            'mean_error_rad': round(float(mean_error), 4),
            'max_error_rad': round(float(max_error), 4),
            'median_error_rad': round(float(median_error), 4),
            'mean_error_deg': round(float(np.degrees(mean_error)), 2),
            'precision_bits': round(float(precision_bits), 2),
            'errors': [round(float(e), 4) for e in errors],
        }
        print("    %d bases: mean=%.1f deg, median=%.1f deg, max=%.1f deg, ~%.1f bits" % (
            n_bases, np.degrees(mean_error), np.degrees(median_error),
            np.degrees(max_error), precision_bits))

    # Step 3: Fidelity of reconstructed states
    print("\n  Computing reconstruction fidelity...")
    # Use 3 bases for reconstruction
    fidelities = []
    n_bases_fid = 3
    for true_phi in test_phis:
        v_true = phi_vec(true_phi, v0, v1)

        measurements = []
        for bi in range(n_bases_fid):
            E = inject_measure_E(model, tok, prompts[bi], DEVICE, v_true,
                                 INJECT_LAYER, min_tok, max_tok)
            measurements.append(E)

        best_phi = 0
        best_score = float('inf')
        for ci, phi_candidate in enumerate(cal_phis):
            score = sum((measurements[bi] - calibrations[bi][ci]) ** 2
                        for bi in range(n_bases_fid))
            if score < best_score:
                best_score = score
                best_phi = phi_candidate

        v_reconstructed = phi_vec(best_phi, v0, v1)
        fid = float(torch.dot(v_true.float(), v_reconstructed.float()) / (
            v_true.float().norm() * v_reconstructed.float().norm()))
        fidelities.append(max(fid, 0))

    mean_fidelity = np.mean(fidelities)
    print("    Mean fidelity (3 bases): %.4f" % mean_fidelity)

    print("\n  TOMOGRAPHY SUMMARY:")
    for nb, res in results_by_nbases.items():
        print("    %d bases: %.1f deg mean error, ~%.1f bits" % (
            nb, res['mean_error_deg'], res['precision_bits']))
    print("    Reconstruction fidelity: %.4f" % mean_fidelity)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Calibration curves
    ax = axes[0]
    for pi in range(len(prompts)):
        ax.plot(cal_phis / np.pi, calibrations[pi], lw=1.5, alpha=0.7,
                label='Basis %d' % pi)
    ax.set_xlabel('Phase (x pi)')
    ax.set_ylabel('E = P(min) - P(max)')
    ax.set_title('(a) Multi-Basis Calibration\n%d measurement bases' % len(prompts),
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B: Error vs number of bases
    ax = axes[1]
    nbs = list(results_by_nbases.keys())
    mean_errs = [results_by_nbases[nb]['mean_error_deg'] for nb in nbs]
    median_errs = [np.degrees(results_by_nbases[nb]['median_error_rad']) for nb in nbs]
    ax.plot(nbs, mean_errs, 'ro-', lw=2, ms=8, label='Mean error')
    ax.plot(nbs, median_errs, 'bs-', lw=2, ms=6, label='Median error')
    ax.set_xlabel('Number of measurement bases')
    ax.set_ylabel('Reconstruction error (degrees)')
    ax.set_title('(b) Tomography Precision\nvs Number of Bases', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(nbs)

    # Panel C: Reconstruction scatter
    ax = axes[2]
    r3 = results_by_nbases[3]
    true_arr = test_phis / np.pi
    errors_3b = np.array(r3['errors'])
    # Reconstruct estimated phis from errors (approximate)
    ax.bar(range(len(fidelities)), fidelities, color='#4CAF50',
           edgecolor='black', alpha=0.85)
    ax.axhline(np.mean(fidelities), color='red', ls='--', lw=2,
               label='Mean: %.3f' % mean_fidelity)
    ax.set_xlabel('Test state index')
    ax.set_ylabel('Reconstruction fidelity')
    ax.set_title('(c) State Fidelity\n3-basis tomography', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.suptitle('Phase Q49: Quantum State Tomography\n'
                 'Reconstructing S-Qubit states from multi-basis measurements',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q49_tomography.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q49', 'name': 'quantum_tomography',
        'inject_layer': INJECT_LAYER,
        'n_bases': len(prompts),
        'n_test': n_test,
        'results_by_nbases': {str(k): v for k, v in results_by_nbases.items()},
        'mean_fidelity': round(float(mean_fidelity), 6),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q49_tomography.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q49 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
