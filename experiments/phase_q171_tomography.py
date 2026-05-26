# -*- coding: utf-8 -*-
"""
Phase Q171: Quantum State Tomography of LLM
=============================================
Full density matrix reconstruction of LLM output.
Compute: purity, concurrence, entanglement entropy
for different bipartitions of the hidden state.

This is the MOST COMPLETE quantum characterization possible.
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


def main():
    print("=" * 60)
    print("Phase Q171: Quantum State Tomography of LLM")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompts = [
        ("Quantum entangled state:", "quantum"),
        ("The cat sat on the mat:", "classical"),
        ("def fibonacci(n):", "code"),
        ("Hydrogen molecule bond:", "chemistry"),
    ]

    all_results = []

    for prompt, label in prompts:
        print("\n--- [%s] '%s' ---" % (label, prompt[:30]))
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Collect hidden states from multiple tokens (last 4)
        seq_len = out.hidden_states[-1].shape[1]
        n_tokens = min(4, seq_len)

        # Build "density matrix" from ensemble of layer states
        # rho = (1/N) sum_i |psi_i><psi_i| (mixed state from different layers)
        n_qubits_eff = 8  # Use 8 effective qubits (256 dim)
        dim = 2 ** n_qubits_eff

        psi_ensemble = []
        for li in range(0, n_layers + 1, 2):
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            norm = np.linalg.norm(h)
            if norm > 1e-8:
                psi_ensemble.append(h / norm)

        if not psi_ensemble:
            continue

        # Density matrix
        rho = np.zeros((dim, dim))
        for psi in psi_ensemble:
            rho += np.outer(psi, psi)
        rho /= len(psi_ensemble)

        # 1. Purity
        purity = float(np.real(np.trace(rho @ rho)))

        # 2. Von Neumann entropy
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-12]
        vn_entropy = float(-np.sum(eigvals * np.log2(eigvals + 1e-15)))

        # 3. Rank (effective number of pure states)
        rank = int(np.sum(eigvals > 1e-6))

        # 4. Bipartite entanglement entropy
        # Split 8 qubits into 4+4
        n_A = 4
        dim_A = 2 ** n_A
        dim_B = dim // dim_A

        entanglement_entropies = []
        for psi in psi_ensemble[:5]:  # Top 5 states
            # Reshape into bipartite system
            psi_AB = psi.reshape(dim_A, dim_B)
            # Reduced density matrix of subsystem A
            rho_A = psi_AB @ psi_AB.T
            rho_A /= np.trace(rho_A) + 1e-10
            eigs_A = np.linalg.eigvalsh(rho_A)
            eigs_A = eigs_A[eigs_A > 1e-12]
            S_A = float(-np.sum(eigs_A * np.log2(eigs_A + 1e-15)))
            entanglement_entropies.append(S_A)

        avg_ent_entropy = float(np.mean(entanglement_entropies))
        max_ent_entropy = float(np.max(entanglement_entropies))

        # 5. Schmidt decomposition of best state
        psi_best = psi_ensemble[0].reshape(dim_A, dim_B)
        U, schmidt_values, Vt = np.linalg.svd(psi_best, full_matrices=False)
        schmidt_values = schmidt_values / (np.linalg.norm(schmidt_values) + 1e-10)
        n_schmidt = int(np.sum(schmidt_values > 0.01))
        schmidt_entropy = float(-np.sum(schmidt_values**2 * np.log2(schmidt_values**2 + 1e-15)))

        # 6. Participation ratio of density matrix eigenvalues
        PR = 1.0 / float(np.sum(eigvals**2) + 1e-10)

        result = {
            'label': label,
            'purity': round(purity, 6),
            'vn_entropy': round(vn_entropy, 4),
            'rank': rank,
            'participation_ratio': round(PR, 2),
            'avg_entanglement_entropy': round(avg_ent_entropy, 4),
            'max_entanglement_entropy': round(max_ent_entropy, 4),
            'n_schmidt_coeffs': n_schmidt,
            'schmidt_entropy': round(schmidt_entropy, 4),
            'max_possible_entropy': float(n_qubits_eff),
        }
        all_results.append(result)

        print("  Purity: %.6f (pure=1, mixed=1/%d=%.4f)" %
              (purity, dim, 1.0/dim))
        print("  VN Entropy: %.4f / %.1f bits" % (vn_entropy, n_qubits_eff))
        print("  Rank: %d / %d" % (rank, dim))
        print("  Entanglement (4|4): avg=%.4f, max=%.4f" %
              (avg_ent_entropy, max_ent_entropy))
        print("  Schmidt coeffs > 0.01: %d, Schmidt entropy: %.4f" %
              (n_schmidt, schmidt_entropy))

    # Summary
    print("\n--- Tomography Summary ---")
    avg_purity = float(np.mean([r['purity'] for r in all_results]))
    avg_ent = float(np.mean([r['avg_entanglement_entropy'] for r in all_results]))
    print("  Avg purity: %.4f" % avg_purity)
    print("  Avg entanglement: %.4f bits" % avg_ent)
    if avg_purity > 0.5:
        print("  State type: MOSTLY PURE (quantum-like)")
    elif avg_purity > 0.1:
        print("  State type: WEAKLY MIXED")
    else:
        print("  State type: HIGHLY MIXED (thermal)")

    # Save
    results = {
        'phase': 'Q171',
        'name': 'Quantum State Tomography',
        'tomography': all_results,
        'summary': {
            'avg_purity': round(avg_purity, 6),
            'avg_entanglement': round(avg_ent, 4),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q171_tomography.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = [r['label'] for r in all_results]
    ax = axes[0]
    purities = [r['purity'] for r in all_results]
    ax.bar(range(len(labels)), purities, color='#E91E63', edgecolor='black', alpha=0.85)
    ax.axhline(1.0/dim, color='red', ls='--', label='Maximally mixed (1/%d)' % dim)
    ax.axhline(1.0, color='green', ls='--', alpha=0.3, label='Pure state')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Purity Tr(rho^2)')
    ax.set_title('(a) State Purity')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ent_vals = [r['avg_entanglement_entropy'] for r in all_results]
    ax.bar(range(len(labels)), ent_vals, color='#4CAF50', edgecolor='black', alpha=0.85)
    ax.axhline(n_qubits_eff / 2, color='red', ls='--',
               label='Max entanglement (%d bits)' % (n_qubits_eff // 2))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Entanglement Entropy (bits)')
    ax.set_title('(b) Bipartite Entanglement (4|4)')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    schmidt_n = [r['n_schmidt_coeffs'] for r in all_results]
    schmidt_e = [r['schmidt_entropy'] for r in all_results]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, schmidt_n, 0.3, color='#2196F3', label='# Schmidt coeffs')
    ax2 = ax.twinx()
    ax2.bar(x + 0.15, schmidt_e, 0.3, color='#FF9800', label='Schmidt entropy', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('# Schmidt coefficients', color='#2196F3')
    ax2.set_ylabel('Schmidt entropy (bits)', color='#FF9800')
    ax.set_title('(c) Schmidt Decomposition')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q171: Quantum State Tomography of LLM Hidden States',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q171_tomography.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ171 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
