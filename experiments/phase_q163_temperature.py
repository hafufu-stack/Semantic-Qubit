# -*- coding: utf-8 -*-
"""
Phase Q163: Temperature-Decoherence Mapping
=============================================
In quantum physics: temperature T causes decoherence (loss of coherence).
In LLM: temperature T controls sampling randomness.

Are these the SAME phenomenon?
Map LLM temperature to quantum decoherence strength
and check if the functional form matches.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def purity(rho):
    """Tr(rho^2) - measure of coherence. Pure state = 1, maximally mixed = 1/d"""
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def main():
    print("=" * 60)
    print("Phase Q163: Temperature-Decoherence Mapping")
    print("  (Is LLM Temperature = Quantum Decoherence?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompt = "The quantum state of the hydrogen atom is"

    # Get base logits
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    base_logits = out.logits[0, -1, :].float().cpu()

    temperatures = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    all_results = []

    for T in temperatures:
        # LLM "temperature": softmax(logits / T)
        probs = torch.softmax(base_logits / T, dim=0).numpy()

        # Quantum interpretation: probability distribution = diagonal of density matrix
        # Use top-k components as "qubit" space
        k = 64  # Effective Hilbert space dimension
        top_k_idx = np.argsort(probs)[-k:]
        p_top = probs[top_k_idx]
        p_top /= p_top.sum()

        # Construct density matrix rho = diag(p)
        rho = np.diag(p_top)

        # Coherence metrics
        pur = purity(rho)
        ent = von_neumann_entropy(rho)
        max_prob = float(np.max(p_top))

        # Participation ratio (effective number of states)
        PR = 1.0 / float(np.sum(p_top ** 2))

        # Quantum coherence: off-diagonal elements
        # In this model rho is diagonal, but we can measure "distance from pure state"
        # Pure state: one eigenvalue = 1, rest = 0
        # Maximally mixed: all eigenvalues = 1/k
        decoherence = 1.0 - pur  # 0 for pure, 1-1/k for maximally mixed

        result = {
            'temperature': float(T),
            'purity': round(pur, 6),
            'entropy': round(ent, 4),
            'max_prob': round(max_prob, 6),
            'participation_ratio': round(PR, 2),
            'decoherence': round(decoherence, 6),
        }
        all_results.append(result)
        print("  T=%.2f: purity=%.4f, entropy=%.2f, PR=%.1f, decoherence=%.4f" %
              (T, pur, ent, PR, decoherence))

    # Fit: decoherence(T) = 1 - 1/(1 + aT^b) (quantum thermal model)
    T_arr = np.array([r['temperature'] for r in all_results])
    D_arr = np.array([r['decoherence'] for r in all_results])

    # Simple exponential fit: D = 1 - exp(-alpha * T)
    from scipy.optimize import curve_fit
    try:
        def thermal_model(T, alpha, beta):
            return 1 - np.exp(-alpha * T ** beta)

        popt, _ = curve_fit(thermal_model, T_arr, D_arr, p0=[1, 0.5],
                            maxfev=5000)
        alpha_fit, beta_fit = popt
        D_fit = thermal_model(T_arr, alpha_fit, beta_fit)
        fit_r2 = 1 - np.sum((D_arr - D_fit) ** 2) / np.sum((D_arr - np.mean(D_arr)) ** 2)
        print("\n--- Thermal Model Fit ---")
        print("  D(T) = 1 - exp(-%.3f * T^%.3f)" % (alpha_fit, beta_fit))
        print("  R2 = %.4f" % fit_r2)
    except Exception as e:
        print("\n  Fit failed: %s" % str(e))
        alpha_fit, beta_fit, fit_r2 = 0, 0, 0
        D_fit = np.zeros_like(D_arr)

    # Quantum reference: thermal state decoherence
    # D_quantum(T) = 1 - 1/Z * sum_n exp(-2*E_n/kT) for harmonic oscillator
    # Simplified: D = 1 - exp(-T/T0) with T0 characteristic temperature
    print("\n--- Comparison with Quantum Thermal Decoherence ---")
    print("  LLM: D(T) ~ T^%.2f (exponent)" % beta_fit)
    print("  Quantum oscillator: D(T) ~ T^1.0")
    print("  Match: %s" % ("GOOD" if 0.5 < beta_fit < 2.0 else "POOR"))

    # Save
    results = {
        'phase': 'Q163',
        'name': 'Temperature-Decoherence Mapping',
        'data': all_results,
        'fit': {
            'alpha': round(float(alpha_fit), 4),
            'beta': round(float(beta_fit), 4),
            'r2': round(float(fit_r2), 4),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q163_temperature.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.semilogx(T_arr, D_arr, 'o-', color='#E91E63', linewidth=2,
                label='LLM decoherence', markersize=8)
    if fit_r2 > 0:
        T_smooth = np.logspace(np.log10(0.01), np.log10(50), 100)
        D_smooth = thermal_model(T_smooth, alpha_fit, beta_fit)
        ax.semilogx(T_smooth, D_smooth, '--', color='blue',
                    label='Fit: 1-exp(-%.2fT^%.2f) R2=%.3f' %
                    (alpha_fit, beta_fit, fit_r2))
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Decoherence (1 - Purity)')
    ax.set_title('(a) Temperature -> Decoherence')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    ax = axes[1]
    ents = [r['entropy'] for r in all_results]
    ax.semilogx(T_arr, ents, 's-', color='#4CAF50', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Von Neumann Entropy (bits)')
    ax.set_title('(b) Entropy vs Temperature')
    ax.grid(alpha=0.3)

    ax = axes[2]
    PRs = [r['participation_ratio'] for r in all_results]
    ax.semilogx(T_arr, PRs, 'D-', color='#2196F3', linewidth=2, markersize=8)
    ax.axhline(64, color='red', ls='--', alpha=0.5, label='Max (k=64)')
    ax.axhline(1, color='green', ls='--', alpha=0.5, label='Pure state')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('(c) Effective # of States')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q163: Temperature = Decoherence? (LLM vs Quantum Thermal)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q163_temperature.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ163 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
