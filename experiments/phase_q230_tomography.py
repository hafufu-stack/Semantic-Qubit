# -*- coding: utf-8 -*-
"""
Phase Q230: Quantum Tomography
================================
Reconstruct the full density matrix of the LLM's quantum state
from measurement outcomes. How many measurements needed for
accurate reconstruction? Compare with theoretical minimum.
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


def generate_measurement_bases(dim, n_bases, rng):
    """Generate random measurement bases (Pauli-like)."""
    bases = []
    for _ in range(n_bases):
        # Random unitary -> columns are measurement basis
        A = rng.randn(dim, dim) + 1j * rng.randn(dim, dim)
        Q, _ = np.linalg.qr(A)
        bases.append(Q)
    return bases


def measure_in_basis(rho, basis):
    """Measure density matrix in given basis, return probabilities."""
    probs = np.real(np.diag(basis.conj().T @ rho @ basis))
    probs = np.maximum(probs, 0)
    probs /= probs.sum() + 1e-10
    return probs


def reconstruct_rho(measurements, bases, dim):
    """Maximum likelihood reconstruction (simplified linear inversion)."""
    rho_est = np.zeros((dim, dim), dtype=complex)
    for basis, probs in zip(bases, measurements):
        for k in range(dim):
            v = basis[:, k]
            rho_est += probs[k] * np.outer(v, v.conj())
    rho_est /= len(measurements)
    # Project to valid density matrix
    rho_est = (rho_est + rho_est.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(rho_est)
    eigvals = np.maximum(eigvals, 0)
    eigvals /= eigvals.sum()
    rho_est = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_est


def fidelity(rho1, rho2):
    """Quantum state fidelity."""
    sqrt1 = np.linalg.cholesky(rho1 + 1e-10 * np.eye(len(rho1)))
    M = sqrt1 @ rho2 @ sqrt1.conj().T
    eigvals = np.maximum(np.real(np.linalg.eigvalsh(M)), 0)
    return float(np.sum(np.sqrt(eigvals)) ** 2)


def main():
    print("=" * 60)
    print("Phase Q230: Quantum State Tomography")
    print("  (Reconstruct the LLM's density matrix)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dims = [2, 4, 8]
    n_bases_list = [3, 6, 12, 24, 48]

    prompts = [
        "quantum state tomography measurement",
        "Bell state preparation and detection",
        "density matrix reconstruction",
    ]

    all_results = []

    for dim in dims:
        print("\n--- dim=%d ---" % dim)

        for prompt in prompts[:1]:  # Use first prompt for each dim
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            # Build "true" density matrix from multiple layers
            rho_true = np.zeros((dim, dim), dtype=complex)
            for li in [8, 12, 16, 20, 24]:
                if li < len(out.hidden_states):
                    h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
                    h /= np.linalg.norm(h) + 1e-10
                    rho_true += np.outer(h, h.conj())
            rho_true /= np.trace(rho_true)
            rho_true = 0.7 * rho_true + 0.3 * np.eye(dim) / dim
            rho_true /= np.trace(rho_true)

            for n_bases in n_bases_list:
                rng = np.random.RandomState(42)
                bases = generate_measurement_bases(dim, n_bases, rng)
                measurements = [measure_in_basis(rho_true, b) for b in bases]
                rho_est = reconstruct_rho(measurements, bases, dim)

                try:
                    fid = fidelity(rho_true, rho_est)
                except:
                    fid = float(np.abs(np.trace(rho_true @ rho_est)))

                trace_dist = float(np.sum(np.abs(np.linalg.eigvalsh(rho_true - rho_est)))) / 2

                # Theoretical minimum: dim^2 - 1 real parameters
                n_params = dim ** 2 - 1
                efficiency = n_params / n_bases if n_bases > 0 else 0

                print("  n_bases=%d: F=%.4f, trace_dist=%.4f (need %d params)" %
                      (n_bases, fid, trace_dist, n_params))

                all_results.append({
                    'dim': dim,
                    'n_bases': n_bases,
                    'fidelity': round(fid, 4),
                    'trace_distance': round(trace_dist, 4),
                    'n_params_needed': n_params,
                    'efficiency': round(efficiency, 2),
                })

    # Find minimum bases for F>0.99
    min_bases = {}
    for dim in dims:
        subset = [r for r in all_results if r['dim'] == dim]
        for r in sorted(subset, key=lambda x: x['n_bases']):
            if r['fidelity'] > 0.99:
                min_bases[dim] = r['n_bases']
                break

    verdict = "Tomography: min bases for F>0.99: %s" % (
        ', '.join('dim=%d:%d' % (d, b) for d, b in sorted(min_bases.items()))
        if min_bases else 'not achieved')

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q230',
        'name': 'Quantum State Tomography',
        'data': all_results,
        'summary': {
            'min_bases_for_99': min_bases,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q230_tomography.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for di, dim in enumerate(dims):
        ax = axes[di]
        subset = [r for r in all_results if r['dim'] == dim]
        n_b = [r['n_bases'] for r in subset]
        fids = [r['fidelity'] for r in subset]
        ax.plot(n_b, fids, 'o-', color='#E91E63', lw=2, ms=6)
        ax.axhline(0.99, color='green', ls='--', label='F=0.99')
        ax.axvline(dim**2-1, color='gray', ls=':', label='d^2-1=%d' % (dim**2-1))
        ax.set_xlabel('Number of Bases'); ax.set_ylabel('Fidelity')
        ax.set_title('dim=%d' % dim); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.05)
    plt.suptitle('Q230: Quantum State Tomography\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q230_tomography.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ230 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
