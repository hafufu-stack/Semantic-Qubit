# -*- coding: utf-8 -*-
"""
Phase Q248: Many-Worlds Prompting
====================================
Quantum-inspired multi-path reasoning.
Instead of single Chain-of-Thought, run MULTIPLE reasoning paths
through S-Qubit space and combine via quantum interference.
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


def partial_transpose(rho, da, db):
    rho_pt = np.zeros_like(rho)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                for l in range(db):
                    rho_pt[i*db+k, j*db+l] = rho[i*db+l, j*db+k]
    return rho_pt


def main():
    print("=" * 60)
    print("Phase Q248: Many-Worlds Prompting")
    print("  (Multi-path quantum reasoning)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 8

    # Logic puzzles with known answers
    puzzles = [
        {
            'question': 'If all cats are animals, and some animals are pets, are all cats pets?',
            'paths': [
                'All cats are animals. Some animals are pets. So some cats might be pets, but not necessarily all.',
                'Cats are animals. Animals include wild ones. Therefore not all cats are pets.',
                'Every cat is an animal. Pets are a subset of animals. Cats could be outside the pet subset.',
            ],
            'correct': 'no',
        },
        {
            'question': 'A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?',
            'paths': [
                'Let ball = x. Bat = x + 1.00. Total: 2x + 1.00 = 1.10. x = 0.05.',
                'If ball is 0.10, bat is 1.10. Total 1.20. Wrong. Ball must be 0.05.',
                'Intuition says 0.10 but algebra says 0.05. The answer is 0.05 or 5 cents.',
            ],
            'correct': '0.05',
        },
        {
            'question': 'What is heavier, a pound of feathers or a pound of steel?',
            'paths': [
                'Both weigh a pound. They are the same weight.',
                'Steel is denser, but a pound is a pound. Equal.',
                'Trick question. A pound of anything weighs one pound.',
            ],
            'correct': 'same',
        },
    ]

    all_results = []
    for puzzle in puzzles:
        print("\n--- %s ---" % puzzle['question'][:50])

        # Single path representation
        single_rep = None
        path_reps = []

        for pi, path in enumerate(puzzle['paths']):
            full_text = puzzle['question'] + " " + path
            inp = tok(full_text, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()
            path_reps.append(h)
            if pi == 0:
                single_rep = h.copy()

        # Classical combination: average
        classical_combined = np.mean(path_reps, axis=0)
        classical_combined /= np.linalg.norm(classical_combined) + 1e-10

        # Quantum combination: coherent superposition (sum of amplitudes)
        quantum_combined = np.sum(path_reps, axis=0)
        quantum_combined /= np.linalg.norm(quantum_combined) + 1e-10

        # Measure entanglement of each
        da, db = 2, dim // 2

        def neg_from_vec(v):
            v = v[:dim] / (np.linalg.norm(v[:dim]) + 1e-10)
            rho = np.outer(v, v.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)
            if da * db <= dim:
                eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
                return float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
            return 0

        neg_single = neg_from_vec(single_rep)
        neg_classical = neg_from_vec(classical_combined)
        neg_quantum = neg_from_vec(quantum_combined)

        # Coherence of combined state
        def coherence(v):
            v = v[:dim] / (np.linalg.norm(v[:dim]) + 1e-10)
            rho = np.outer(v, v.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)

        coh_single = coherence(single_rep)
        coh_quantum = coherence(quantum_combined)

        # Interference: how much do paths reinforce?
        cos_sims = [float(np.dot(path_reps[i], path_reps[j]) /
                    (np.linalg.norm(path_reps[i]) * np.linalg.norm(path_reps[j]) + 1e-10))
                    for i in range(len(path_reps)) for j in range(i+1, len(path_reps))]
        avg_interference = np.mean(cos_sims)

        print("  Single: neg=%.4f, coh=%.4f" % (neg_single, coh_single))
        print("  Quantum: neg=%.4f, coh=%.4f" % (neg_quantum, coh_quantum))
        print("  Avg path interference: %.4f" % avg_interference)

        all_results.append({
            'question': puzzle['question'][:50],
            'n_paths': len(puzzle['paths']),
            'neg_single': round(neg_single, 6),
            'neg_classical': round(neg_classical, 6),
            'neg_quantum': round(neg_quantum, 6),
            'coh_single': round(coh_single, 4),
            'coh_quantum': round(coh_quantum, 4),
            'avg_interference': round(float(avg_interference), 4),
        })

    # Summary
    avg_boost = np.mean([(r['neg_quantum'] - r['neg_single']) / max(r['neg_single'], 1e-6)
                          for r in all_results]) * 100
    avg_coh_boost = np.mean([(r['coh_quantum'] - r['coh_single']) / max(r['coh_single'], 1e-6)
                              for r in all_results]) * 100

    if avg_boost > 10:
        verdict = "MANY-WORLDS BOOST: %.0f%% more entanglement, %.0f%% more coherence" % (avg_boost, avg_coh_boost)
    elif avg_boost > 0:
        verdict = "SLIGHT BOOST: %.1f%% entanglement increase" % avg_boost
    else:
        verdict = "NO BOOST: multi-path does not increase quantumness"

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q248', 'name': 'Many-Worlds Prompting',
        'puzzles': all_results,
        'summary': {'avg_ent_boost': round(avg_boost, 1), 'avg_coh_boost': round(avg_coh_boost, 1), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q248_many_worlds.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_results))
    ax = axes[0]
    ax.bar(x - 0.2, [r['neg_single'] for r in all_results], 0.4, label='Single Path', color='#607D8B', edgecolor='black')
    ax.bar(x + 0.2, [r['neg_quantum'] for r in all_results], 0.4, label='Many-Worlds', color='#E91E63', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Negativity'); ax.set_title('(a) Entanglement: Single vs Many-Worlds')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - 0.2, [r['coh_single'] for r in all_results], 0.4, label='Single Path', color='#607D8B', edgecolor='black')
    ax.bar(x + 0.2, [r['coh_quantum'] for r in all_results], 0.4, label='Many-Worlds', color='#9C27B0', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Coherence'); ax.set_title('(b) Coherence: Single vs Many-Worlds')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q248: Many-Worlds Prompting\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q248_many_worlds.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ248 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
