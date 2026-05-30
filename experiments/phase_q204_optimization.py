# -*- coding: utf-8 -*-
"""
Phase Q204: Quantum-Like Combinatorial Optimization
=====================================================
Can S-Qubit's high-dimensional "phase tunneling" beat classical
optimization heuristics (Simulated Annealing, Greedy)?

We test on MAX-CUT and a simplified TSP, comparing:
1. S-Qubit (Embedding VQE on QUBO Hamiltonian)
2. Simulated Annealing (classical baseline)
3. Random search (null baseline)
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


def generate_maxcut_graph(n_nodes, n_edges, seed=42):
    """Generate a random graph for MAX-CUT."""
    rng = np.random.RandomState(seed)
    edges = set()
    while len(edges) < n_edges:
        i, j = rng.randint(0, n_nodes, size=2)
        if i != j:
            edges.add((min(i, j), max(i, j)))
    weights = {e: rng.uniform(0.5, 2.0) for e in edges}
    return list(edges), weights


def maxcut_to_hamiltonian(n_nodes, edges, weights):
    """Convert MAX-CUT to QUBO Hamiltonian (dim = n_nodes)."""
    dim = n_nodes
    H = np.zeros((dim, dim))
    for (i, j) in edges:
        w = weights[(i, j)]
        H[i, i] -= w / 2
        H[j, j] -= w / 2
        H[i, j] += w / 2
        H[j, i] += w / 2
    return H


def maxcut_value(assignment, edges, weights):
    """Compute cut value for a binary assignment."""
    cut = 0
    for (i, j) in edges:
        if assignment[i] != assignment[j]:
            cut += weights[(i, j)]
    return cut


def simulated_annealing(n_nodes, edges, weights, n_iter=1000, seed=0):
    """Classical simulated annealing for MAX-CUT."""
    rng = np.random.RandomState(seed)
    x = rng.randint(0, 2, size=n_nodes)
    best_x = x.copy()
    best_val = maxcut_value(x, edges, weights)
    history = [best_val]
    T = 2.0

    for step in range(n_iter):
        T = max(0.01, T * 0.995)
        i = rng.randint(0, n_nodes)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]
        val_new = maxcut_value(x_new, edges, weights)
        delta = val_new - maxcut_value(x, edges, weights)
        if delta > 0 or rng.random() < np.exp(delta / T):
            x = x_new
            if val_new > best_val:
                best_val = val_new
                best_x = x_new.copy()
        history.append(best_val)

    return best_val, best_x, history


def random_search(n_nodes, edges, weights, n_iter=1000, seed=0):
    """Random search baseline."""
    rng = np.random.RandomState(seed)
    best_val = 0
    history = []
    for _ in range(n_iter):
        x = rng.randint(0, 2, size=n_nodes)
        val = maxcut_value(x, edges, weights)
        if val > best_val:
            best_val = val
        history.append(best_val)
    return best_val, history


def llm_vqe_maxcut(model, tok, device, H_np, n_nodes, edges, weights,
                    n_steps=300, lr=0.005):
    """Solve MAX-CUT using Embedding VQE."""
    dim = n_nodes
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    embed_layer = model.model.embed_tokens

    prompt = "optimize graph cut:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)

    history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        # Use sigmoid to get soft binary assignment
        psi = torch.sigmoid(h * 5)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()

        # Evaluate cut value with hard assignment
        with torch.no_grad():
            assignment = (psi > 0.5).long().cpu().numpy()
            cut_val = maxcut_value(assignment, edges, weights)
            history.append(cut_val)

    # Final assignment
    with torch.no_grad():
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h_final = out.hidden_states[-1][0, -1, :dim]
        final_assign = (torch.sigmoid(h_final * 5) > 0.5).long().cpu().numpy()
        final_cut = maxcut_value(final_assign, edges, weights)

    return final_cut, final_assign, history


def main():
    print("=" * 60)
    print("Phase Q204: Quantum-Like Combinatorial Optimization")
    print("  (S-Qubit vs SA vs Random on MAX-CUT)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Test problems with increasing size
    problems = [
        {'n_nodes': 8, 'n_edges': 12, 'seed': 42},
        {'n_nodes': 12, 'n_edges': 20, 'seed': 123},
        {'n_nodes': 16, 'n_edges': 30, 'seed': 456},
        {'n_nodes': 20, 'n_edges': 40, 'seed': 789},
    ]

    n_sa_trials = 5
    n_steps = 300
    all_results = []

    for prob in problems:
        n = prob['n_nodes']
        print("\n--- MAX-CUT N=%d, E=%d ---" % (n, prob['n_edges']))
        edges, weights = generate_maxcut_graph(n, prob['n_edges'], prob['seed'])

        # Brute force optimal (only for small graphs)
        if n <= 16:
            best_brute = 0
            for bits in range(2 ** n):
                x = np.array([(bits >> i) & 1 for i in range(n)])
                val = maxcut_value(x, edges, weights)
                if val > best_brute:
                    best_brute = val
        else:
            best_brute = None

        H = maxcut_to_hamiltonian(n, edges, weights)

        # S-Qubit VQE
        llm_cut, llm_assign, llm_hist = llm_vqe_maxcut(
            model, tok, device, H, n, edges, weights, n_steps=n_steps)
        print("  S-Qubit: cut=%.2f" % llm_cut)

        # Simulated Annealing
        sa_cuts = []
        sa_hists = []
        for trial in range(n_sa_trials):
            sa_cut, _, sa_hist = simulated_annealing(
                n, edges, weights, n_iter=n_steps, seed=trial)
            sa_cuts.append(sa_cut)
            sa_hists.append(sa_hist)
        avg_sa = np.mean(sa_cuts)
        best_sa = max(sa_cuts)
        print("  SA (avg): cut=%.2f (best=%.2f)" % (avg_sa, best_sa))

        # Random search
        rand_cut, rand_hist = random_search(n, edges, weights,
                                             n_iter=n_steps, seed=42)
        print("  Random: cut=%.2f" % rand_cut)

        if best_brute:
            print("  Optimal: cut=%.2f" % best_brute)

        result = {
            'n_nodes': n,
            'n_edges': prob['n_edges'],
            'optimal': float(best_brute) if best_brute else None,
            'sqbit': {'cut': float(llm_cut), 'history': llm_hist},
            'sa': {
                'avg_cut': round(float(avg_sa), 2),
                'best_cut': round(float(best_sa), 2),
                'cuts': [round(float(c), 2) for c in sa_cuts],
            },
            'random': {'cut': round(float(rand_cut), 2)},
            'sqbit_vs_sa': round(float(llm_cut / max(avg_sa, 0.01)), 3),
        }
        all_results.append(result)

    # Summary
    avg_ratio = np.mean([r['sqbit_vs_sa'] for r in all_results])
    n_wins = sum(1 for r in all_results
                 if r['sqbit']['cut'] >= r['sa']['avg_cut'])

    if avg_ratio > 1.0:
        verdict = "S-QUBIT WINS: %.1f%% better than SA on average" % ((avg_ratio - 1) * 100)
    elif avg_ratio > 0.9:
        verdict = "COMPETITIVE: S-Qubit within 10%% of SA"
    else:
        verdict = "SA WINS: Classical heuristic superior"

    print("\n--- Summary ---")
    print("  S-Qubit/SA ratio: %.3f" % avg_ratio)
    print("  S-Qubit wins: %d/%d problems" % (n_wins, len(problems)))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q204',
        'name': 'Quantum-Like Combinatorial Optimization',
        'problems': all_results,
        'summary': {
            'avg_sqbit_sa_ratio': round(avg_ratio, 3),
            'sqbit_wins': n_wins,
            'total_problems': len(problems),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q204_optimization.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, r in enumerate(all_results):
        ax = axes[idx // 2][idx % 2]
        n = r['n_nodes']

        # Convergence curves
        llm_h = r['sqbit']['history']
        ax.plot(range(len(llm_h)), llm_h,
                color='#E91E63', lw=2, label='S-Qubit VQE')

        ax.axhline(r['sa']['avg_cut'], color='#2196F3', ls='--',
                   lw=1.5, label='SA (avg=%.1f)' % r['sa']['avg_cut'])
        ax.axhline(r['random']['cut'], color='#9E9E9E', ls=':',
                   label='Random (%.1f)' % r['random']['cut'])

        if r['optimal']:
            ax.axhline(r['optimal'], color='green', ls='-',
                       alpha=0.5, label='Optimal (%.1f)' % r['optimal'])

        ax.set_xlabel('Step')
        ax.set_ylabel('Cut Value')
        ax.set_title('MAX-CUT N=%d (ratio=%.2f)' % (n, r['sqbit_vs_sa']))
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    plt.suptitle('Q204: Quantum-Like Combinatorial Optimization\n'
                 'S-Qubit vs SA on MAX-CUT (%s)' % verdict[:50],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q204_optimization.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ204 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
