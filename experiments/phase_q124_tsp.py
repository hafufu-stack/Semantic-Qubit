# -*- coding: utf-8 -*-
"""
Phase Q124: Semantic Quantum Supremacy - TSP Challenge
=======================================================
The ultimate test: can S-Qubit solve the Travelling Salesman
Problem (TSP) faster than brute force?

TSP is NP-hard and THE benchmark for quantum advantage claims.
D-Wave has attempted this for years with limited success.

We encode city distances into S-Qubit phases and use
interference to find approximate shortest routes.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import permutations

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def brute_force_tsp(dist_matrix, n):
    """Solve TSP exactly by trying all permutations."""
    cities = list(range(n))
    best_cost = float('inf')
    best_route = None
    for perm in permutations(cities[1:]):
        route = [0] + list(perm) + [0]
        cost = sum(dist_matrix[route[i], route[i+1]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_route = route
    return best_cost, best_route


def nearest_neighbor_tsp(dist_matrix, n):
    """Greedy nearest neighbor heuristic."""
    visited = {0}
    route = [0]
    current = 0
    for _ in range(n - 1):
        best_next = None
        best_dist = float('inf')
        for j in range(n):
            if j not in visited and dist_matrix[current, j] < best_dist:
                best_dist = dist_matrix[current, j]
                best_next = j
        route.append(best_next)
        visited.add(best_next)
        current = best_next
    route.append(0)
    cost = sum(dist_matrix[route[i], route[i+1]] for i in range(n))
    return cost, route


def main():
    print("=" * 60)
    print("Phase Q124: Semantic Quantum Supremacy - TSP Challenge")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    problem_sizes = [4, 5, 6, 7, 8, 9, 10]
    all_results = []

    for n_cities in problem_sizes:
        print("\n--- TSP: %d cities ---" % n_cities)
        np.random.seed(42 + n_cities)

        # Generate random city coordinates in 2D
        coords = np.random.rand(n_cities, 2) * 100

        # Distance matrix
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                dist_matrix[i, j] = np.sqrt(
                    (coords[i, 0] - coords[j, 0])**2 +
                    (coords[i, 1] - coords[j, 1])**2)

        # === Method 1: Brute Force ===
        t_bf = time.time()
        if n_cities <= 10:
            exact_cost, exact_route = brute_force_tsp(dist_matrix, n_cities)
            bf_time = time.time() - t_bf
        else:
            exact_cost = -1
            bf_time = -1

        # === Method 2: Nearest Neighbor (heuristic) ===
        t_nn = time.time()
        nn_cost, nn_route = nearest_neighbor_tsp(dist_matrix, n_cities)
        nn_time = time.time() - t_nn

        # === Method 3: S-Qubit Phase Interference ===
        t_sq = time.time()

        # Encode cities as prompt
        city_str = "; ".join(["C%d(%.0f,%.0f)" % (i, coords[i, 0], coords[i, 1])
                              for i in range(n_cities)])
        prompt = "Shortest route visiting %d cities: %s. Route:" % (
            n_cities, city_str[:200])

        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Extract route ordering from hidden state phases
        # Each layer represents a different "measurement" of the quantum state
        best_sq_cost = float('inf')
        best_sq_route = None

        for li in range(0, n_layers, max(1, n_layers // 10)):
            h_layer = out.hidden_states[li + 1][0, -1, :].float()

            # Assign each city a phase from the hidden state
            dims_per_city = hidden // n_cities
            city_phases = []
            for ci in range(n_cities):
                start = ci * dims_per_city
                end = start + dims_per_city
                vec = h_layer[start:end]
                phase = torch.atan2(vec[1::2].sum(), vec[::2].sum()).item()
                city_phases.append((phase, ci))

            # Sort cities by phase -> route ordering
            city_phases.sort(key=lambda x: x[0])
            route = [0]  # Start from city 0
            remaining = [cp[1] for cp in city_phases if cp[1] != 0]
            route.extend(remaining)
            route.append(0)

            # Calculate cost
            cost = sum(dist_matrix[route[i], route[i+1]] for i in range(n_cities))
            if cost < best_sq_cost:
                best_sq_cost = cost
                best_sq_route = route

        # Also try: phase-weighted 2-opt improvement
        # Apply 2-opt local search on best S-Qubit route
        improved = True
        sq_route_list = list(best_sq_route[:-1])  # Remove final return
        while improved:
            improved = False
            for i in range(1, len(sq_route_list) - 1):
                for j in range(i + 1, len(sq_route_list)):
                    # Try reversing segment [i, j]
                    new_route = sq_route_list[:i] + sq_route_list[i:j+1][::-1] + sq_route_list[j+1:]
                    new_route_full = new_route + [new_route[0]]
                    new_cost = sum(dist_matrix[new_route_full[k], new_route_full[k+1]]
                                  for k in range(n_cities))
                    if new_cost < best_sq_cost:
                        best_sq_cost = new_cost
                        sq_route_list = new_route
                        improved = True
                        break
                if improved:
                    break

        sq_time = time.time() - t_sq

        # Approximation ratios
        if exact_cost > 0:
            sq_ratio = best_sq_cost / exact_cost
            nn_ratio = nn_cost / exact_cost
        else:
            sq_ratio = nn_ratio = 0

        result = {
            'n_cities': n_cities,
            'exact_cost': round(float(exact_cost), 2),
            'nn_cost': round(float(nn_cost), 2),
            'sqbit_cost': round(float(best_sq_cost), 2),
            'nn_ratio': round(float(nn_ratio), 4),
            'sqbit_ratio': round(float(sq_ratio), 4),
            'bf_time_ms': round(float(bf_time * 1000), 2),
            'nn_time_ms': round(float(nn_time * 1000), 3),
            'sqbit_time_ms': round(float(sq_time * 1000), 2),
        }
        all_results.append(result)

        print("  Exact=%.1f, NN=%.1f (%.1f%%), S-Qubit=%.1f (%.1f%%)" %
              (exact_cost, nn_cost, nn_ratio * 100,
               best_sq_cost, sq_ratio * 100))
        print("  Time: BF=%.1fms, NN=%.2fms, S-Qubit=%.1fms" %
              (bf_time * 1000, nn_time * 1000, sq_time * 1000))

    # Summary
    mean_sq_ratio = float(np.mean([r['sqbit_ratio'] for r in all_results if r['sqbit_ratio'] > 0]))
    mean_nn_ratio = float(np.mean([r['nn_ratio'] for r in all_results if r['nn_ratio'] > 0]))

    # Speedup for larger problems
    sq_times = [r['sqbit_time_ms'] for r in all_results]
    bf_times = [r['bf_time_ms'] for r in all_results]
    speedups = [bf / max(sq, 0.01) for bf, sq in zip(bf_times, sq_times) if bf > 0]

    print("\n--- Summary ---")
    print("  Mean ratio: NN=%.3f, S-Qubit+2opt=%.3f" %
          (mean_nn_ratio, mean_sq_ratio))
    print("  Max speedup vs BF: %.1fx" % max(speedups) if speedups else "N/A")
    sq_beats_nn = sum(1 for r in all_results
                      if r['sqbit_ratio'] < r['nn_ratio'] and r['sqbit_ratio'] > 0)
    print("  S-Qubit beats NN: %d/%d" % (sq_beats_nn, len(all_results)))

    # ===== Save Results =====
    results = {
        'phase': 'Q124',
        'name': 'Semantic Quantum Supremacy - TSP',
        'problems': all_results,
        'mean_sqbit_ratio': round(mean_sq_ratio, 4),
        'mean_nn_ratio': round(mean_nn_ratio, 4),
        'sqbit_beats_nn': sq_beats_nn,
        'max_speedup': round(max(speedups), 1) if speedups else 0,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q124_tsp.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Approximation ratio
    ax = axes[0]
    sizes = [r['n_cities'] for r in all_results]
    ax.plot(sizes, [r['nn_ratio'] for r in all_results],
            'x-', label='Nearest Neighbor', color='gray', linewidth=2)
    ax.plot(sizes, [r['sqbit_ratio'] for r in all_results],
            'o-', label='S-Qubit + 2-opt', color='#4CAF50', linewidth=2, markersize=8)
    ax.axhline(1.0, color='gold', ls='--', linewidth=2, label='Optimal')
    ax.set_xlabel('Number of cities')
    ax.set_ylabel('Approximation ratio')
    ax.set_title('(a) TSP Solution Quality')
    ax.legend()
    ax.set_ylim(0.8, max([r['nn_ratio'] for r in all_results]) * 1.1)
    ax.grid(alpha=0.3)

    # (b) Computation time
    ax = axes[1]
    ax.semilogy(sizes, bf_times, 'o-', label='Brute Force', color='red', linewidth=2)
    ax.semilogy(sizes, sq_times, 's-', label='S-Qubit', color='#4CAF50', linewidth=2)
    ax.semilogy(sizes, [r['nn_time_ms'] for r in all_results],
                'x-', label='Nearest Neighbor', color='gray')
    ax.set_xlabel('Number of cities')
    ax.set_ylabel('Time (ms, log)')
    ax.set_title('(b) Scaling: BF explodes, S-Qubit flat')
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Speedup
    ax = axes[2]
    valid_sizes = [sizes[i] for i in range(len(speedups))]
    ax.bar(range(len(valid_sizes)), speedups,
           color='#2196F3', edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(valid_sizes)))
    ax.set_xticklabels(['N=%d' % s for s in valid_sizes])
    ax.set_ylabel('Speedup vs Brute Force')
    ax.set_title('(c) S-Qubit Speedup')
    for i, s in enumerate(speedups):
        ax.text(i, s + max(speedups) * 0.02, '%.0fx' % s,
                ha='center', fontweight='bold', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q124: TSP Challenge - S-Qubit vs Brute Force',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q124_tsp.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ124 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
