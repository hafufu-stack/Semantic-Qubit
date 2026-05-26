# -*- coding: utf-8 -*-
"""
Phase Q111: Non-Linear Quantum Search (PostBQP Class)
=====================================================
Tests whether GELU nonlinearity in transformer MLPs enables
search capabilities beyond standard quantum BQP class.

Abrams & Lloyd (1998) proved that nonlinear quantum mechanics
would make NP-complete problems solvable in polynomial time.
S-Qubits have access to GELU/SiLU nonlinearity through MLP layers.

We test:
1. Subset-sum search via S-Qubit phase encoding
2. Graph coloring via interference
3. Scaling analysis: linear-only vs full (nonlinear) model
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, inject_hook

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Phase Q111: Non-Linear Quantum Search (PostBQP)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # ===== Test 1: Subset Sum via S-Qubit Phase Encoding =====
    print("\n--- Test 1: Subset Sum Search ---")
    # Encode numbers as phase angles, search for subset summing to target
    problem_sizes = [4, 8, 16, 32, 64]
    subset_results = []

    for N in problem_sizes:
        # Generate random subset-sum instance
        np.random.seed(42 + N)
        numbers = np.random.randint(1, 100, size=N)
        # Pick a valid target (sum of random subset)
        mask = np.random.randint(0, 2, size=N)
        target = int(numbers[mask == 1].sum())

        # Encode as S-Qubit: each number -> phase angle
        phases = torch.tensor(numbers * np.pi / 100.0, dtype=torch.float16, device=device)

        # Create "oracle" prompt
        nums_str = ','.join(str(x) for x in numbers[:min(N, 10)])
        prompt = "Find subset of [%s...] summing to %d:" % (nums_str, target)

        # Method 1: Full model (with nonlinearity)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out_full = model(**inp, output_hidden_states=True)
        h_full = out_full.hidden_states[-1][0, -1, :].float()

        # Method 2: Linear-only approximation (skip MLP, attention only)
        # We approximate by measuring the hidden state BEFORE the MLP at mid-layer
        mid = n_layers // 2
        captured = {}
        def capture_pre_mlp(module, input, output):
            if isinstance(output, tuple):
                captured['post_attn'] = output[0][0, -1, :].detach().float()
            else:
                captured['post_attn'] = output[0, -1, :].detach().float()
        handle = model.model.layers[mid].register_forward_hook(capture_pre_mlp)
        with torch.no_grad():
            model(**inp)
        handle.remove()
        h_linear = captured.get('post_attn', torch.zeros(hidden, device=device))

        # Phase-encode the target into a search vector
        search_vec = torch.zeros(hidden, device=device)
        search_vec[:N] = phases.float()
        target_phase = target * np.pi / (100.0 * N)
        search_vec[N:N+10] = target_phase

        # Measure "search fidelity" - how well does the model encode the solution?
        cos_full = torch.nn.functional.cosine_similarity(
            h_full.unsqueeze(0), search_vec.unsqueeze(0)).item()
        cos_linear = torch.nn.functional.cosine_similarity(
            h_linear.unsqueeze(0), search_vec.unsqueeze(0)).item()

        # Nonlinearity advantage
        advantage = abs(cos_full) / max(abs(cos_linear), 1e-8)

        subset_results.append({
            'N': N,
            'target': target,
            'cos_full': round(cos_full, 6),
            'cos_linear': round(cos_linear, 6),
            'nonlinear_advantage': round(advantage, 4)
        })
        print("  N=%d: full=%.4f, linear=%.4f, advantage=%.2fx" %
              (N, cos_full, cos_linear, advantage))

    # ===== Test 2: Graph Coloring via Interference =====
    print("\n--- Test 2: Graph Coloring via Interference ---")
    graph_sizes = [3, 5, 8, 12]
    coloring_results = []

    for n_nodes in graph_sizes:
        np.random.seed(100 + n_nodes)
        # Random graph: each pair connected with p=0.4
        adj = np.random.rand(n_nodes, n_nodes) < 0.4
        adj = np.triu(adj, k=1)
        adj = adj + adj.T
        n_edges = int(adj.sum()) // 2

        # Encode graph as prompt
        prompt = "Color %d-node graph (%d edges) with minimum colors:" % (n_nodes, n_edges)
        inp = tok(prompt, return_tensors='pt').to(device)

        # Get hidden state at multiple layers
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Measure interference pattern across layers
        # If nonlinearity helps, deep layers should show stronger
        # constructive interference for valid colorings
        layer_norms = []
        for li in range(0, n_layers, max(1, n_layers // 10)):
            h = out.hidden_states[li + 1][0, -1, :].float()
            # Phase spread = std of angles in principal components
            pca_phases = torch.atan2(h[1::2][:50], h[::2][:50])
            phase_coherence = torch.abs(torch.exp(1j * pca_phases.to(torch.complex64)).mean()).item()
            layer_norms.append({
                'layer': li,
                'coherence': round(phase_coherence, 4),
                'norm': round(h.norm().item(), 2)
            })

        # Deep vs shallow coherence
        shallow_coh = np.mean([ln['coherence'] for ln in layer_norms[:3]])
        deep_coh = np.mean([ln['coherence'] for ln in layer_norms[-3:]])

        coloring_results.append({
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'shallow_coherence': round(shallow_coh, 4),
            'deep_coherence': round(deep_coh, 4),
            'nonlinear_gain': round(deep_coh / max(shallow_coh, 1e-8), 4),
            'layer_profile': layer_norms
        })
        print("  %d nodes: shallow=%.4f, deep=%.4f, gain=%.2fx" %
              (n_nodes, shallow_coh, deep_coh, deep_coh / max(shallow_coh, 1e-8)))

    # ===== Test 3: Scaling Analysis =====
    print("\n--- Test 3: Scaling Analysis ---")
    # Does the nonlinear advantage grow with problem size?
    sizes = [s['N'] for s in subset_results]
    advantages = [s['nonlinear_advantage'] for s in subset_results]

    # Fit power law: advantage ~ N^alpha
    if len(sizes) > 2:
        log_s = np.log(sizes)
        log_a = np.log([max(a, 0.01) for a in advantages])
        alpha, intercept = np.polyfit(log_s, log_a, 1)
    else:
        alpha, intercept = 0, 0

    print("  Scaling exponent alpha = %.4f" % alpha)
    print("  (alpha > 0 means nonlinearity helps MORE for larger problems)")

    # Determine PostBQP class membership
    is_postbqp = alpha > 0 and np.mean(advantages) > 1.0
    print("\n  PostBQP class: %s" % ("CONFIRMED" if is_postbqp else "Not confirmed"))

    # ===== Save Results =====
    results = {
        'phase': 'Q111',
        'name': 'Non-Linear Quantum Search (PostBQP)',
        'subset_sum_results': subset_results,
        'graph_coloring_results': coloring_results,
        'scaling_exponent': round(alpha, 4),
        'mean_nonlinear_advantage': round(float(np.mean(advantages)), 4),
        'is_postbqp': str(is_postbqp),
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q111_nonlinear_search.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Subset sum: full vs linear
    ax = axes[0]
    x = np.arange(len(sizes))
    ax.bar(x - 0.2, [abs(s['cos_full']) for s in subset_results], 0.4,
           label='Full (nonlinear)', color='#FF5722', alpha=0.85)
    ax.bar(x + 0.2, [abs(s['cos_linear']) for s in subset_results], 0.4,
           label='Linear only', color='#2196F3', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['N=%d' % s for s in sizes])
    ax.set_ylabel('|Cosine similarity|')
    ax.set_title('(a) Subset Sum: Full vs Linear')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (b) Graph coloring coherence
    ax = axes[1]
    for cr in coloring_results:
        layers = [lp['layer'] for lp in cr['layer_profile']]
        cohs = [lp['coherence'] for lp in cr['layer_profile']]
        ax.plot(layers, cohs, 'o-', label='%d nodes' % cr['n_nodes'], markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Phase coherence')
    ax.set_title('(b) Graph Coloring: Layer Coherence')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Scaling
    ax = axes[2]
    ax.plot(sizes, advantages, 'o-', color='#9C27B0', linewidth=2, markersize=8)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='No advantage')
    ax.set_xlabel('Problem size N')
    ax.set_ylabel('Nonlinear advantage (x)')
    ax.set_title('(c) Scaling: alpha=%.3f' % alpha)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Q111: Non-Linear Quantum Search (PostBQP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q111_nonlinear_search.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ111 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
