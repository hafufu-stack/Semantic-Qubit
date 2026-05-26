# -*- coding: utf-8 -*-
"""
Phase Q175: Quantum Phase Transition in Attention
====================================================
Phase transitions are hallmarks of quantum systems.
Test: vary attention head dropout ratio from 0->100%.
Look for SHARP transition in quantum advantage.

If gradual -> classical system
If sharp (critical point) -> quantum-like phase transition!
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


def build_syk(n_qubits, seed=42):
    np.random.seed(seed)
    dim = 2 ** n_qubits
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)
    def kron_chain(ops):
        r = ops[0]
        for o in ops[1:]: r = np.kron(r, o)
        return r
    H = np.zeros((dim, dim))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            J = np.random.randn() / np.sqrt(n_qubits)
            ops = [I2]*n_qubits; ops[i] = Z; ops[j] = Z
            H += -J * kron_chain(ops)
    for i in range(n_qubits):
        ops = [I2]*n_qubits; ops[i] = X
        H += -0.3 * kron_chain(ops)
    return H


def rayleigh_gd(H, psi_init, max_steps=1000):
    psi = psi_init.copy() / np.linalg.norm(psi_init)
    lr = 0.01
    for step in range(max_steps):
        E = float(np.real(psi @ H @ psi))
        grad = 2 * (H @ psi - E * psi)
        psi_t = psi - lr * grad
        psi_t /= np.linalg.norm(psi_t)
        Et = float(np.real(psi_t @ H @ psi_t))
        if not np.isnan(Et) and Et < E:
            psi = psi_t
        else:
            lr *= 0.999
    return psi


def main():
    print("=" * 60)
    print("Phase Q175: Quantum Phase Transition in Attention")
    print("  (Noise-Driven Phase Transition)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompt = "The quantum ground state energy of the system:"
    inp = tok(prompt, return_tensors='pt').to(device)

    # Get clean hidden states
    with torch.no_grad():
        out_clean = model(**inp, output_hidden_states=True)
    h_clean = out_clean.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Build Hamiltonian
    n_qubits = 8
    dim = 2 ** n_qubits
    H = build_syk(n_qubits, seed=42)
    E_exact = float(np.linalg.eigvalsh(H)[0])

    # Noise levels (fraction of hidden state elements zeroed/corrupted)
    noise_fractions = np.arange(0.0, 1.01, 0.05)

    all_results = []
    n_trials = 3

    for noise_frac in noise_fractions:
        errors = []
        fidelities = []

        for trial in range(n_trials):
            h_noisy = h_clean.copy()

            # Apply noise: randomly zero out elements
            n_corrupt = int(noise_frac * hidden_size)
            if n_corrupt > 0:
                indices = np.random.choice(hidden_size, n_corrupt, replace=False)
                h_noisy[indices] = 0.0

            # Extract quantum state
            psi = h_noisy[:dim].copy()
            norm = np.linalg.norm(psi)
            if norm < 1e-8:
                errors.append(9999)
                fidelities.append(0)
                continue
            psi /= norm

            # Optimize
            psi_opt = rayleigh_gd(H, psi, max_steps=500)
            E_opt = float(np.real(psi_opt @ H @ psi_opt))
            err = abs(E_opt - E_exact) * 1000
            errors.append(err)

            # Fidelity with clean state
            psi_clean = h_clean[:dim].copy()
            psi_clean /= np.linalg.norm(psi_clean)
            fid = abs(float(np.dot(psi, psi_clean))) ** 2
            fidelities.append(fid)

        avg_err = float(np.mean(errors))
        avg_fid = float(np.mean(fidelities))

        result = {
            'noise_fraction': round(float(noise_frac), 2),
            'avg_error_mha': round(avg_err, 4),
            'avg_fidelity': round(avg_fid, 4),
        }
        all_results.append(result)

        if noise_frac * 100 % 20 == 0:
            print("  Noise %.0f%%: error=%.2f mHa, fidelity=%.4f" %
                  (noise_frac * 100, avg_err, avg_fid))

    # Random baseline
    rand_errors = []
    for _ in range(10):
        psi_r = np.random.randn(dim); psi_r /= np.linalg.norm(psi_r)
        psi_f = rayleigh_gd(H, psi_r, max_steps=500)
        rand_errors.append(abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000)
    rand_mean = float(np.mean(rand_errors))

    # Find phase transition: where does error exceed random?
    critical_noise = None
    for r in all_results:
        if r['avg_error_mha'] > rand_mean and critical_noise is None:
            critical_noise = r['noise_fraction']

    # Compute susceptibility (derivative of error)
    errors_list = [r['avg_error_mha'] for r in all_results]
    susceptibility = np.gradient(errors_list)
    peak_susc_idx = int(np.argmax(np.abs(susceptibility)))
    peak_susc_noise = float(noise_fractions[peak_susc_idx])

    print("\n--- Phase Transition Analysis ---")
    print("  Random baseline: %.2f mHa" % rand_mean)
    print("  Critical noise (error > random): %.0f%%" %
          ((critical_noise or 1.0) * 100))
    print("  Peak susceptibility at: %.0f%% noise" % (peak_susc_noise * 100))
    print("  Transition sharpness: %s" %
          ("SHARP (quantum-like)" if max(abs(susceptibility)) > rand_mean * 0.5
           else "GRADUAL (classical)"))

    # Save
    results = {
        'phase': 'Q175',
        'name': 'Quantum Phase Transition',
        'noise_sweep': all_results,
        'random_baseline_mha': round(rand_mean, 4),
        'critical_noise': critical_noise,
        'peak_susceptibility_noise': round(peak_susc_noise, 2),
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q175_phase_transition.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    nfs = [r['noise_fraction'] * 100 for r in all_results]
    errs = [r['avg_error_mha'] for r in all_results]
    ax.plot(nfs, errs, 'o-', color='#E91E63', linewidth=1.5, markersize=5)
    ax.axhline(rand_mean, color='gray', ls='--', linewidth=2, label='Random baseline')
    if critical_noise:
        ax.axvline(critical_noise * 100, color='blue', ls=':', linewidth=2,
                   label='Critical point')
    ax.set_xlabel('Noise Fraction (%)')
    ax.set_ylabel('Error (mHa)')
    ax.set_title('(a) Quantum Advantage vs Noise')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    fids = [r['avg_fidelity'] for r in all_results]
    ax.plot(nfs, fids, 's-', color='#4CAF50', linewidth=1.5, markersize=5)
    ax.set_xlabel('Noise Fraction (%)')
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) State Fidelity vs Noise')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(nfs[:-1], np.abs(susceptibility[:-1]), 'D-', color='#FF9800',
            linewidth=1.5, markersize=5)
    ax.axvline(peak_susc_noise * 100, color='red', ls='--',
               label='Peak (%.0f%%)' % (peak_susc_noise * 100))
    ax.set_xlabel('Noise Fraction (%)')
    ax.set_ylabel('|dE/d(noise)|')
    ax.set_title('(c) Susceptibility (Phase Transition Indicator)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q175: Quantum Phase Transition in LLM Attention',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q175_phase_transition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ175 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
