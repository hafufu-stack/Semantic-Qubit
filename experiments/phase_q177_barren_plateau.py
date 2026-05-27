# -*- coding: utf-8 -*-
"""
Phase Q177: Barren Plateau Immunity
====================================
Physical quantum VQE suffers from Barren Plateaus: as qubit count grows,
gradients vanish exponentially, making optimization impossible.

Embedding VQE in LLM space operates in continuous R^d, not unitary Hilbert space.
Test: scale the Hamiltonian dimension from 4 to 256 qubits-equivalent and measure
whether gradient norms vanish or remain healthy.

If gradients stay O(1) -> Barren Plateau IMMUNE (classical advantage!)
If gradients decay exp(-n) -> same problem as physical quantum
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


def build_random_hamiltonian(dim, seed=42):
    """Build a random Hermitian Hamiltonian of given dimension."""
    rng = np.random.RandomState(seed)
    A = rng.randn(dim, dim)
    H = (A + A.T) / 2  # Symmetric = Hermitian for real
    # Normalize so eigenvalues are O(1) regardless of dim
    H = H / np.sqrt(dim)
    return H


def main():
    print("=" * 60)
    print("Phase Q177: Barren Plateau Immunity")
    print("  (Does Embedding VQE Scale Without Gradient Death?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    hidden_size = model.config.hidden_size

    embed_layer = model.model.embed_tokens

    # Test dimensions (equivalent qubit counts: 2,3,4,5,6,7,8 qubits = 4,8,16,32,64,128,256)
    dims = [4, 8, 16, 32, 64, 128, 256]
    qubit_labels = [2, 3, 4, 5, 6, 7, 8]
    n_seeds = 5  # multiple random initializations per dimension
    n_steps = 30  # optimization steps per trial

    results_by_dim = []

    for dim, qb in zip(dims, qubit_labels):
        print("\n--- %d qubits (dim=%d) ---" % (qb, dim))

        if dim > hidden_size:
            print("  SKIP: dim > hidden_size (%d)" % hidden_size)
            results_by_dim.append({
                'qubits': qb, 'dim': dim, 'status': 'skipped',
                'avg_grad_norm': 0.0, 'avg_final_error': 0.0,
            })
            continue

        grad_norms_all = []
        final_errors = []
        convergence_curves = []

        for seed in range(n_seeds):
            H_np = build_random_hamiltonian(dim, seed=seed * 100 + dim)
            H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
            E_exact = float(np.linalg.eigvalsh(H_np)[0])

            # Random initial embedding
            rng = np.random.RandomState(seed + dim * 1000)
            init_embed = torch.randn(1, 3, hidden_size, device=device,
                                     dtype=torch.float32) * 0.01

            opt_embeds = init_embed.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([opt_embeds], lr=0.005)

            grad_norms = []
            energies = []

            for step in range(n_steps):
                optimizer.zero_grad()

                outputs = model(inputs_embeds=opt_embeds.float(),
                               output_hidden_states=True)
                h = outputs.hidden_states[-1][0, -1, :]

                psi = h[:dim]
                psi_norm = psi / (torch.norm(psi) + 1e-10)
                E = psi_norm @ H_torch @ psi_norm

                E.backward()

                gn = float(opt_embeds.grad.norm())
                grad_norms.append(gn)
                energies.append(float(E.detach()))

                optimizer.step()

            grad_norms_all.extend(grad_norms)
            final_error = abs(energies[-1] - E_exact) * 1000  # mHa
            final_errors.append(final_error)
            convergence_curves.append(energies)

        avg_grad = float(np.mean(grad_norms_all))
        std_grad = float(np.std(grad_norms_all))
        avg_error = float(np.mean(final_errors))

        print("  Avg grad norm: %.6f +/- %.6f" % (avg_grad, std_grad))
        print("  Avg final error: %.2f mHa" % avg_error)

        results_by_dim.append({
            'qubits': qb,
            'dim': dim,
            'status': 'completed',
            'avg_grad_norm': round(avg_grad, 8),
            'std_grad_norm': round(std_grad, 8),
            'avg_final_error_mHa': round(avg_error, 2),
            'n_seeds': n_seeds,
            'n_steps': n_steps,
            'convergence_example': [round(e, 6) for e in convergence_curves[0]],
        })

    # Analysis: fit gradient norm vs dimension
    completed = [r for r in results_by_dim if r['status'] == 'completed']
    log_dims = np.log([r['dim'] for r in completed])
    log_grads = np.log([r['avg_grad_norm'] for r in completed])

    if len(log_dims) > 2:
        slope, intercept = np.polyfit(log_dims, log_grads, 1)
    else:
        slope, intercept = 0.0, 0.0

    print("\n--- Barren Plateau Analysis ---")
    print("  Gradient scaling: ||grad|| ~ dim^%.3f" % slope)
    if slope > -0.5:
        verdict = "BARREN PLATEAU IMMUNE (gradients stay healthy)"
        immune = True
    else:
        verdict = "BARREN PLATEAU PRESENT (gradients vanish)"
        immune = False
    print("  Physical QC: gradients vanish as exp(-n_qubits)")
    print("  Embedding VQE: gradients scale as dim^%.3f" % slope)
    print("  Verdict: %s" % verdict)

    # Save results
    results = {
        'phase': 'Q177',
        'name': 'Barren Plateau Immunity',
        'by_dimension': results_by_dim,
        'analysis': {
            'gradient_scaling_exponent': round(slope, 4),
            'immune': immune,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q177_barren_plateau.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot: 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Gradient norm vs dimension (log-log)
    ax = axes[0]
    dims_c = [r['dim'] for r in completed]
    grads_c = [r['avg_grad_norm'] for r in completed]
    ax.loglog(dims_c, grads_c, 's-', color='#2196F3', markersize=8,
              linewidth=2, label='Embedding VQE')
    # Add physical QC reference line (exponential decay)
    x_ref = np.array(dims_c)
    qc_ref = grads_c[0] * np.exp(-0.3 * (np.log2(x_ref) - np.log2(dims_c[0])))
    ax.loglog(x_ref, qc_ref, '--', color='red', linewidth=2,
              label='Physical QC (exp decay)')
    ax.set_xlabel('Hamiltonian Dimension')
    ax.set_ylabel('Avg Gradient Norm')
    ax.set_title('(a) Gradient Scaling\n(Barren Plateau Test)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # (b) Final error vs dimension
    ax = axes[1]
    errors_c = [r['avg_final_error_mHa'] for r in completed]
    ax.bar(range(len(dims_c)), errors_c, color='#4CAF50', edgecolor='black',
           alpha=0.85)
    ax.axhline(1.6, color='red', ls='--', label='Chemical accuracy (1.6 mHa)')
    ax.set_xticks(range(len(dims_c)))
    ax.set_xticklabels(['%dq\n(d=%d)' % (q, d) for q, d in
                        zip([r['qubits'] for r in completed], dims_c)],
                       fontsize=8)
    ax.set_ylabel('Final Error (mHa)')
    ax.set_title('(b) Accuracy vs Scale\n(%d steps, %d seeds)' % (n_steps, n_seeds))
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Convergence curves for smallest and largest
    ax = axes[2]
    for r in [completed[0], completed[-1]]:
        curve = r['convergence_example']
        ax.plot(range(len(curve)), curve, '-', linewidth=2,
                label='%d qubits (d=%d)' % (r['qubits'], r['dim']))
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title('(c) Convergence: Small vs Large')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Q177: Barren Plateau Immunity\n'
                 '(Embedding VQE: gradients ~ dim^%.2f, Physical QC: exp(-n))' % slope,
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q177_barren_plateau.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ177 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
