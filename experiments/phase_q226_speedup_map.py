# -*- coding: utf-8 -*-
"""
Phase Q226: Quantum Speedup Map
==================================
Systematic benchmark: for which problem CLASSES does S-Qubit
beat classical algorithms? Map the exact boundary.

Test: SYK (all-to-all), Ising (local), Hubbard, Random matrices
Compare: S-Qubit VQE vs Scipy eigensolver (exact) vs SA vs random
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


def build_syk(dim, seed=42):
    """SYK-type Hamiltonian (all-to-all random coupling)."""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim) * 0.3
    return (H + H.T) / 2


def build_ising(dim, seed=42):
    """1D Ising-type Hamiltonian (nearest-neighbor)."""
    H = np.zeros((dim, dim))
    rng = np.random.RandomState(seed)
    for i in range(dim - 1):
        H[i, i + 1] = rng.randn() * 0.5
        H[i + 1, i] = H[i, i + 1]
    for i in range(dim):
        H[i, i] = rng.randn() * 0.2
    return H


def build_hubbard(dim, seed=42):
    """Simplified Hubbard-type (hopping + on-site)."""
    H = np.zeros((dim, dim))
    rng = np.random.RandomState(seed)
    # Hopping (tridiagonal)
    t = 0.5
    for i in range(dim - 1):
        H[i, i + 1] = -t
        H[i + 1, i] = -t
    # On-site interaction (every other pair)
    U = rng.uniform(0.5, 2.0)
    for i in range(0, dim, 2):
        H[i, i] = U * rng.uniform(0.5, 1.5)
    return H


def run_vqe(model, tok, device, H_np, dim, n_steps=200, lr=0.005):
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigh(H_np)[0][0])

    embed_layer = model.model.embed_tokens
    prompt = "ground state:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)

    t0 = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()

    elapsed = time.time() - t0
    error_mha = abs(float(E.detach()) - E_exact) * 1000
    return error_mha, elapsed, E_exact


def run_sa(H_np, dim, n_steps=5000, seed=42):
    """Simulated Annealing baseline."""
    rng = np.random.RandomState(seed)
    E_exact = float(np.linalg.eigh(H_np)[0][0])

    # Random initial state
    psi = rng.randn(dim).astype(np.float64)
    psi /= np.linalg.norm(psi)
    E_best = float(psi @ H_np @ psi)

    t0 = time.time()
    for step in range(n_steps):
        T = max(0.01, 1.0 - step / n_steps)
        delta = rng.randn(dim) * 0.1
        psi_new = psi + delta
        psi_new /= np.linalg.norm(psi_new)
        E_new = float(psi_new @ H_np @ psi_new)

        if E_new < E_best or rng.rand() < np.exp(-(E_new - E_best) / max(T, 1e-10)):
            psi = psi_new
            E_best = min(E_best, E_new)

    elapsed = time.time() - t0
    error = abs(E_best - E_exact) * 1000
    return error, elapsed


def main():
    print("=" * 60)
    print("Phase Q226: Quantum Speedup Map")
    print("  (Which problems does S-Qubit dominate?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    problem_types = {
        'SYK': build_syk,
        'Ising': build_ising,
        'Hubbard': build_hubbard,
    }

    dims = [4, 8, 16]
    all_results = []

    for prob_name, builder in problem_types.items():
        print("\n=== %s ===" % prob_name)

        for dim in dims:
            H = builder(dim, seed=42 + dim).astype(np.float32)

            # S-Qubit VQE
            vqe_err, vqe_time, E_exact = run_vqe(
                model, tok, device, H, dim, n_steps=200)

            # SA baseline
            sa_err, sa_time = run_sa(H, dim, n_steps=5000)

            speedup = sa_time / max(vqe_time, 0.001)
            accuracy_ratio = sa_err / max(vqe_err, 0.0001) if vqe_err > 0.0001 else 999

            winner = "S-QUBIT" if vqe_err <= sa_err else "SA"

            print("  dim=%d: VQE=%.4f mHa (%.1fs), SA=%.4f mHa (%.1fs) -> %s" %
                  (dim, vqe_err, vqe_time, sa_err, sa_time, winner))

            all_results.append({
                'problem': prob_name,
                'dim': dim,
                'vqe_error_mHa': round(vqe_err, 4),
                'vqe_time': round(vqe_time, 2),
                'sa_error_mHa': round(sa_err, 4),
                'sa_time': round(sa_time, 2),
                'winner': winner,
                'accuracy_ratio': round(accuracy_ratio, 2),
            })

    # Summary
    squbit_wins = sum(1 for r in all_results if r['winner'] == 'S-QUBIT')
    total = len(all_results)

    # By problem type
    type_wins = {}
    for prob in problem_types:
        subset = [r for r in all_results if r['problem'] == prob]
        wins = sum(1 for r in subset if r['winner'] == 'S-QUBIT')
        type_wins[prob] = '%d/%d' % (wins, len(subset))

    verdict = "S-Qubit wins %d/%d overall (%s)" % (
        squbit_wins, total,
        ', '.join('%s:%s' % (k, v) for k, v in type_wins.items()))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q226',
        'name': 'Quantum Speedup Map',
        'benchmarks': all_results,
        'summary': {
            'squbit_wins': squbit_wins,
            'total': total,
            'by_type': type_wins,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q226_speedup_map.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot: heatmap-style
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for pi, prob in enumerate(problem_types):
        ax = axes[pi]
        subset = [r for r in all_results if r['problem'] == prob]
        x_dims = [r['dim'] for r in subset]
        vqe_errs = [r['vqe_error_mHa'] for r in subset]
        sa_errs = [r['sa_error_mHa'] for r in subset]

        x = np.arange(len(x_dims))
        ax.bar(x - 0.15, vqe_errs, 0.3, color='#E91E63', label='S-Qubit', edgecolor='black')
        ax.bar(x + 0.15, sa_errs, 0.3, color='#607D8B', label='SA', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in x_dims])
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Error (mHa)')
        ax.set_title('%s (%s)' % (prob, type_wins[prob]))
        ax.legend()
        ax.set_yscale('symlog', linthresh=0.01)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q226: Quantum Speedup Map\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q226_speedup_map.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ226 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
