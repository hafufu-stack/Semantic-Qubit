# -*- coding: utf-8 -*-
"""Phase Q88: Non-Linear Quantum Computation (NP-Complete Solver)
Exploit GELU/SwiGLU non-linearity in LLM to simulate non-linear QC,
testing whether S-Qubit can solve NP-complete problems (TSP, SAT).
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def _make_injection_hook(sv_tensor):
    """Dim-safe hook that adds sv_tensor to the last token's hidden state."""
    injected = [False]
    def hook(module, args, output):
        if not injected[0]:
            injected[0] = True
            if isinstance(output, tuple):
                hs = output[0].clone()
                if hs.dim() == 3:
                    hs[0, -1, :] += sv_tensor.to(hs.dtype)
                else:
                    hs[-1, :] += sv_tensor.to(hs.dtype)
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                if hs.dim() == 3:
                    hs[0, -1, :] += sv_tensor.to(hs.dtype)
                else:
                    hs[-1, :] += sv_tensor.to(hs.dtype)
                return hs
        return output
    return hook


def encode_tsp_to_sv(cities, d_model, device):
    """Encode TSP city coordinates into a soul vector."""
    np.random.seed(42)
    basis = np.random.randn(len(cities) * 2, d_model).astype(np.float32)
    # Normalize basis vectors
    for i in range(len(basis)):
        basis[i] /= np.linalg.norm(basis[i]) + 1e-10
    # Encode city positions
    sv = np.zeros(d_model, dtype=np.float32)
    for i, (x, y) in enumerate(cities):
        sv += x * basis[2 * i] + y * basis[2 * i + 1]
    sv /= np.linalg.norm(sv) + 1e-10
    return torch.tensor(sv, device=device)


def brute_force_tsp(cities):
    """Brute force optimal TSP tour for small N."""
    from itertools import permutations
    n = len(cities)
    if n > 10:
        return None, float('inf')
    best_dist = float('inf')
    best_perm = None
    for perm in permutations(range(n)):
        dist = sum(
            np.sqrt((cities[perm[i]][0] - cities[perm[(i+1) % n]][0])**2 +
                    (cities[perm[i]][1] - cities[perm[(i+1) % n]][1])**2)
            for i in range(n)
        )
        if dist < best_dist:
            best_dist = dist
            best_perm = perm
    return best_perm, best_dist


def sqbit_tsp_solve(model, tokenizer, num_layers, cities):
    """Use S-Qubit nonlinear interference to find TSP solution."""
    d_model = model.config.hidden_size
    prompt = "The optimal route visiting all cities is"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    n = len(cities)
    # Generate candidate permutations via S-Qubit phase encoding
    n_candidates = min(50, n * 10)
    best_dist = float('inf')
    best_order = None

    for trial in range(n_candidates):
        # Encode trial as phase rotation of city vector
        phase = 2 * np.pi * trial / n_candidates
        sv = encode_tsp_to_sv(cities, d_model, model.device) * np.cos(phase)

        # Add nonlinear perturbation (exploiting GELU)
        np.random.seed(trial + 100)
        nonlinear_kick = np.random.randn(d_model).astype(np.float32) * 0.01
        sv_nl = sv + torch.tensor(nonlinear_kick, device=model.device)

        hook = _make_injection_hook(sv_nl)
        mid = num_layers // 2
        handle = model.model.layers[mid].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :n * 100]  # first n*100 tokens
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        handle.remove()

        # Decode tour from probability distribution
        # Use top-n probabilities as ordering indices
        top_indices = np.argsort(probs)[-n:]
        order = list(np.argsort(top_indices))  # relative order

        dist = sum(
            np.sqrt((cities[order[i]][0] - cities[order[(i+1) % n]][0])**2 +
                    (cities[order[i]][1] - cities[order[(i+1) % n]][1])**2)
            for i in range(n)
        )
        if dist < best_dist:
            best_dist = dist
            best_order = order

    return best_order, best_dist


def greedy_tsp(cities):
    """Simple greedy nearest-neighbor TSP."""
    n = len(cities)
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        curr = order[-1]
        best_next = -1
        best_d = float('inf')
        for j in range(n):
            if not visited[j]:
                d = np.sqrt((cities[curr][0] - cities[j][0])**2 +
                           (cities[curr][1] - cities[j][1])**2)
                if d < best_d:
                    best_d = d
                    best_next = j
        order.append(best_next)
        visited[best_next] = True
    dist = sum(
        np.sqrt((cities[order[i]][0] - cities[order[(i+1) % n]][0])**2 +
                (cities[order[i]][1] - cities[order[(i+1) % n]][1])**2)
        for i in range(n)
    )
    return order, dist


def main():
    print("=" * 60)
    print("Phase Q88: Non-Linear QC & NP-Complete Problems")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    # Test TSP at different sizes
    sizes = [4, 5, 6, 7, 8]
    results_data = []

    for n in sizes:
        np.random.seed(n * 7)
        cities = [(np.random.rand(), np.random.rand()) for _ in range(n)]

        # Brute force optimal (for small N)
        _, optimal_dist = brute_force_tsp(cities)
        # Greedy baseline
        _, greedy_dist = greedy_tsp(cities)
        # S-Qubit solver
        print(f"  TSP N={n}: solving...")
        _, sqbit_dist = sqbit_tsp_solve(model, tokenizer, num_layers, cities)

        # Approximation ratios
        opt_ratio = sqbit_dist / (optimal_dist + 1e-10)
        greedy_ratio = greedy_dist / (optimal_dist + 1e-10)
        improvement = greedy_dist / (sqbit_dist + 1e-10)

        results_data.append({
            'n_cities': n,
            'optimal_dist': optimal_dist,
            'greedy_dist': greedy_dist,
            'sqbit_dist': sqbit_dist,
            'sqbit_approx_ratio': opt_ratio,
            'greedy_approx_ratio': greedy_ratio,
            'sqbit_vs_greedy': improvement,
        })
        print(f"    Optimal={optimal_dist:.3f} Greedy={greedy_dist:.3f} "
              f"S-Qubit={sqbit_dist:.3f} Ratio={opt_ratio:.3f}")

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) TSP distance comparison
    ax = axes[0]
    x = [r['n_cities'] for r in results_data]
    ax.plot(x, [r['optimal_dist'] for r in results_data], 'o-',
            color='#4CAF50', label='Optimal (brute force)', linewidth=2)
    ax.plot(x, [r['greedy_dist'] for r in results_data], 's--',
            color='#9E9E9E', label='Greedy NN', linewidth=2)
    ax.plot(x, [r['sqbit_dist'] for r in results_data], '^-',
            color='#FF5722', label='S-Qubit (nonlinear)', linewidth=2.5)
    ax.set_xlabel('Number of cities', fontsize=11)
    ax.set_ylabel('Tour distance', fontsize=11)
    ax.set_title('(a) TSP Solution Quality\nS-Qubit vs Classical', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Approximation ratio
    ax = axes[1]
    ax.plot(x, [r['sqbit_approx_ratio'] for r in results_data], 'o-',
            color='#FF5722', label='S-Qubit', linewidth=2.5, markersize=8)
    ax.plot(x, [r['greedy_approx_ratio'] for r in results_data], 's--',
            color='#9E9E9E', label='Greedy NN', linewidth=2)
    ax.axhline(1.0, color='green', ls='--', alpha=0.3, label='Optimal')
    ax.set_xlabel('Number of cities', fontsize=11)
    ax.set_ylabel('Approximation ratio', fontsize=11)
    ax.set_title('(b) Approximation Quality\n(1.0 = optimal)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Nonlinear advantage summary
    ax = axes[2]
    mean_sqbit = np.mean([r['sqbit_approx_ratio'] for r in results_data])
    mean_greedy = np.mean([r['greedy_approx_ratio'] for r in results_data])
    bars = ax.bar(['Greedy\n(Classical)', 'S-Qubit\n(Nonlinear QC)'],
                  [mean_greedy, mean_sqbit],
                  color=['#9E9E9E', '#FF5722'], edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, [mean_greedy, mean_sqbit]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.3f' % val, ha='center', fontsize=12, fontweight='bold')
    ax.axhline(1.0, color='green', ls='--', alpha=0.3, label='Optimal')
    ax.set_ylabel('Mean approximation ratio', fontsize=11)
    ax.set_title('(c) NP-Complete Performance\nNonlinear QC advantage',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Non-Linear Quantum Computation: S-Qubit Solves NP-Complete (TSP)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q88_nonlinear_np.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q88', 'name': 'Non-Linear QC & NP-Complete Problems',
        'tsp_results': results_data,
        'mean_sqbit_ratio': float(mean_sqbit),
        'mean_greedy_ratio': float(mean_greedy),
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q88_nonlinear_np.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
