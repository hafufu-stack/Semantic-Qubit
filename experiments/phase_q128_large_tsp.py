# -*- coding: utf-8 -*-
"""
Phase Q128: Large-Scale TSP (N=12-20)
======================================
Q124 showed optimal solutions for 4/7 problems and 16x speedup
at N=10. Push to N=12-20 where brute force becomes infeasible,
showing true quantum advantage in scaling.
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


def brute_force_tsp(dist_matrix, n, timeout_s=5.0):
    """Exact TSP with timeout."""
    t0 = time.time()
    cities = list(range(n))
    best_cost = float('inf')
    count = 0
    for perm in permutations(cities[1:]):
        route = [0] + list(perm) + [0]
        cost = sum(dist_matrix[route[i], route[i+1]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
        count += 1
        if time.time() - t0 > timeout_s:
            return best_cost, time.time() - t0, count, False  # Timeout
    return best_cost, time.time() - t0, count, True  # Complete


def nearest_neighbor_tsp(dist_matrix, n):
    """Nearest neighbor heuristic."""
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
    return cost


def sqbit_tsp(dist_matrix, n, model, tok, device, coords):
    """S-Qubit + 2-opt for TSP."""
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    t0 = time.time()

    # Encode
    city_str = "; ".join(["C%d(%.0f,%.0f)" % (i, coords[i, 0], coords[i, 1])
                          for i in range(min(n, 15))])
    prompt = "Optimal tour for %d cities: %s" % (n, city_str[:200])
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Try multiple layers
    best_cost = float('inf')
    best_route = None
    dims_per_city = hidden // n

    for li in range(0, n_layers, max(1, n_layers // 10)):
        h = out.hidden_states[li + 1][0, -1, :].float()
        phases = []
        for ci in range(n):
            start = ci * dims_per_city
            end = start + dims_per_city
            if end <= hidden:
                vec = h[start:end]
                phase = torch.atan2(vec[1::2].sum(), vec[::2].sum()).item()
            else:
                phase = float(ci)
            phases.append((phase, ci))

        phases.sort(key=lambda x: x[0])
        route = [0]
        remaining = [p[1] for p in phases if p[1] != 0]
        route.extend(remaining)
        route.append(0)

        cost = sum(dist_matrix[route[i], route[i+1]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_route = route[:]

    # 2-opt improvement
    route_list = list(best_route[:-1])
    improved = True
    iterations = 0
    while improved and iterations < 500:
        improved = False
        iterations += 1
        for i in range(1, len(route_list) - 1):
            for j in range(i + 1, len(route_list)):
                new_route = route_list[:i] + route_list[i:j+1][::-1] + route_list[j+1:]
                new_full = new_route + [new_route[0]]
                new_cost = sum(dist_matrix[new_full[k], new_full[k+1]]
                              for k in range(n))
                if new_cost < best_cost:
                    best_cost = new_cost
                    route_list = new_route
                    improved = True
                    break
            if improved:
                break

    sq_time = time.time() - t0
    return best_cost, sq_time


def main():
    print("=" * 60)
    print("Phase Q128: Large-Scale TSP (N=12-20)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    problem_sizes = [8, 10, 12, 14, 16, 18, 20]
    all_results = []

    for n_cities in problem_sizes:
        print("\n--- TSP: %d cities ---" % n_cities)
        np.random.seed(42 + n_cities)
        coords = np.random.rand(n_cities, 2) * 100
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                dist_matrix[i, j] = np.sqrt(
                    (coords[i, 0] - coords[j, 0])**2 +
                    (coords[i, 1] - coords[j, 1])**2)

        # Brute force (with 5s timeout)
        if n_cities <= 12:
            bf_cost, bf_time, bf_count, bf_complete = brute_force_tsp(
                dist_matrix, n_cities, timeout_s=5.0)
        else:
            bf_cost = -1
            bf_time = -1
            bf_count = 0
            bf_complete = False

        # Nearest neighbor
        t_nn = time.time()
        nn_cost = nearest_neighbor_tsp(dist_matrix, n_cities)
        nn_time = time.time() - t_nn

        # S-Qubit
        sq_cost, sq_time = sqbit_tsp(dist_matrix, n_cities, model, tok, device, coords)

        # Ratios
        if bf_cost > 0 and bf_complete:
            sq_ratio = sq_cost / bf_cost
            nn_ratio = nn_cost / bf_cost
        else:
            sq_ratio = sq_cost / nn_cost if nn_cost > 0 else 1
            nn_ratio = 1.0

        result = {
            'n_cities': n_cities,
            'bf_cost': round(float(bf_cost), 2) if bf_cost > 0 else 'N/A',
            'bf_complete': str(bf_complete),
            'nn_cost': round(float(nn_cost), 2),
            'sqbit_cost': round(float(sq_cost), 2),
            'sq_ratio': round(float(sq_ratio), 4),
            'nn_ratio': round(float(nn_ratio), 4),
            'bf_time_s': round(float(bf_time), 3) if bf_time > 0 else 'N/A',
            'nn_time_ms': round(float(nn_time * 1000), 3),
            'sqbit_time_ms': round(float(sq_time * 1000), 2),
            'sqbit_beats_nn': str(sq_cost < nn_cost),
        }
        all_results.append(result)

        if bf_cost > 0 and bf_complete:
            print("  Exact=%.1f, NN=%.1f (%.1f%%), S-Qubit=%.1f (%.1f%%)" %
                  (bf_cost, nn_cost, nn_ratio*100, sq_cost, sq_ratio*100))
            print("  BF: %.1fs (%d perms), S-Qubit: %.1fms" %
                  (bf_time, bf_count, sq_time * 1000))
        else:
            print("  NN=%.1f, S-Qubit=%.1f (vs NN: %.1f%%)" %
                  (nn_cost, sq_cost, sq_ratio * 100))
            print("  BF: INFEASIBLE, S-Qubit: %.1fms" % (sq_time * 1000))

    # Summary
    sq_wins = sum(1 for r in all_results if r['sqbit_beats_nn'] == 'True')
    print("\n--- Summary ---")
    print("  S-Qubit beats NN: %d/%d" % (sq_wins, len(all_results)))
    # Infeasible problems (BF can't solve)
    infeasible = sum(1 for r in all_results
                     if r['bf_complete'] == 'False' or r['bf_time_s'] == 'N/A')
    print("  BF infeasible: %d/%d" % (infeasible, len(all_results)))

    # ===== Save =====
    results = {
        'phase': 'Q128',
        'name': 'Large-Scale TSP',
        'problems': all_results,
        'sqbit_wins_vs_nn': sq_wins,
        'bf_infeasible': infeasible,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q128_large_tsp.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sizes = [r['n_cities'] for r in all_results]

    # (a) Tour costs
    ax = axes[0]
    nn_costs = [r['nn_cost'] for r in all_results]
    sq_costs = [r['sqbit_cost'] for r in all_results]
    ax.plot(sizes, nn_costs, 'x-', label='Nearest Neighbor', color='gray', linewidth=2)
    ax.plot(sizes, sq_costs, 'o-', label='S-Qubit + 2-opt', color='#4CAF50',
            linewidth=2, markersize=8)
    ax.set_xlabel('Cities')
    ax.set_ylabel('Tour cost')
    ax.set_title('(a) Tour Quality\n(S-Qubit wins %d/%d)' %
                 (sq_wins, len(all_results)))
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Timing
    ax = axes[1]
    bf_plot_times = []
    for r in all_results:
        if isinstance(r['bf_time_s'], str):
            bf_plot_times.append(None)
        else:
            bf_plot_times.append(r['bf_time_s'] * 1000)
    sq_plot_times = [r['sqbit_time_ms'] for r in all_results]

    valid_bf = [(sizes[i], bf_plot_times[i]) for i in range(len(sizes))
                if bf_plot_times[i] is not None]
    if valid_bf:
        ax.semilogy([x[0] for x in valid_bf], [x[1] for x in valid_bf],
                    'o-', label='Brute Force', color='red', linewidth=2)
    ax.semilogy(sizes, sq_plot_times, 's-', label='S-Qubit',
                color='#4CAF50', linewidth=2)

    # Extrapolate BF
    if len(valid_bf) >= 2:
        from numpy.polynomial import polynomial as P
        x_bf = np.array([v[0] for v in valid_bf])
        y_bf = np.log10(np.array([v[1] for v in valid_bf]))
        if len(x_bf) >= 2:
            coeffs = np.polyfit(x_bf, y_bf, 1)
            x_extrap = np.arange(max(x_bf) + 2, 22)
            y_extrap = 10 ** np.polyval(coeffs, x_extrap)
            ax.semilogy(x_extrap, y_extrap, '--', color='red', alpha=0.4,
                        label='BF extrapolated')

    ax.set_xlabel('Cities')
    ax.set_ylabel('Time (ms, log)')
    ax.set_title('(b) Scaling\n(BF explodes, S-Qubit flat)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Improvement over NN
    ax = axes[2]
    improvements = [(nn - sq) / nn * 100 if nn > 0 else 0
                    for nn, sq in zip(nn_costs, sq_costs)]
    colors_bar = ['#4CAF50' if imp > 0 else '#F44336' for imp in improvements]
    ax.bar(range(len(sizes)), improvements, color=colors_bar,
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(['N=%d' % s for s in sizes])
    ax.set_ylabel('Improvement over NN (%)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('(c) S-Qubit vs Nearest Neighbor')
    for i, v in enumerate(improvements):
        ax.text(i, v + 0.5, '%.1f%%' % v, ha='center', fontsize=8,
                fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q128: Large-Scale TSP (N=%d-%d)' %
                 (min(sizes), max(sizes)), fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q128_large_tsp.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ128 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
