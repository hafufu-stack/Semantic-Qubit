# -*- coding: utf-8 -*-
"""
Phase Q261: Wigner's Friend Paradox
======================================
Does objective reality exist for the LLM?
Model A (friend) observes at L22, Model B (Wigner) receives
the pre-observation state. Do they agree?
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
    print("Phase Q261: Wigner's Friend Paradox")
    print("  (Does objective reality exist for the LLM?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4; da, db = 2, 2
    obs_layer = 22

    prompts = [
        "Schrodinger's cat is simultaneously alive and dead",
        "The electron passes through both slits at once",
        "Quantum measurement collapses the wavefunction",
        "The moon exists even when nobody looks at it",
    ]

    all_results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Friend's state: AFTER observation (L22+, final output)
        h_friend = out.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()
        h_friend /= np.linalg.norm(h_friend) + 1e-10

        # Wigner's state: BEFORE observation (L21, pre-decoherence)
        h_wigner = out.hidden_states[min(obs_layer - 1, n_layers)][0, -1, :dim].float().cpu().numpy()
        h_wigner /= np.linalg.norm(h_wigner) + 1e-10

        # Wigner sees superposition, Friend sees collapsed state
        # Measure each
        def quantum_props(h):
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)
            ev = np.real(np.linalg.eigvalsh(rho))
            ev_pos = ev[ev > 1e-12]
            S = float(-np.sum(ev_pos * np.log2(ev_pos))) if len(ev_pos) > 0 else 0
            coh = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)
            eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
            neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
            purity = float(np.real(np.trace(rho @ rho)))
            return {'entropy': round(S, 4), 'coherence': round(coh, 4),
                    'negativity': round(neg, 6), 'purity': round(purity, 4)}

        friend_props = quantum_props(h_friend)
        wigner_props = quantum_props(h_wigner)

        # Agreement: how similar are their quantum descriptions?
        overlap = float(np.dot(h_friend, h_wigner))
        disagreement = abs(friend_props['coherence'] - wigner_props['coherence'])

        print("\n  '%s'..." % prompt[:40])
        print("    Friend (post-L22): coh=%.4f, neg=%.6f" % (friend_props['coherence'], friend_props['negativity']))
        print("    Wigner (pre-L22):  coh=%.4f, neg=%.6f" % (wigner_props['coherence'], wigner_props['negativity']))
        print("    Overlap=%.4f, Disagreement=%.4f" % (overlap, disagreement))

        all_results.append({
            'prompt': prompt[:40],
            'friend': friend_props, 'wigner': wigner_props,
            'overlap': round(overlap, 4), 'disagreement': round(disagreement, 4),
        })

    avg_overlap = np.mean([r['overlap'] for r in all_results])
    avg_disagree = np.mean([r['disagreement'] for r in all_results])
    wigner_more_coherent = sum(1 for r in all_results
                                if r['wigner']['coherence'] > r['friend']['coherence'])

    if avg_disagree > 0.05 and wigner_more_coherent >= 3:
        verdict = "PARADOX CONFIRMED: Wigner sees more quantum (%d/%d), avg disagreement=%.3f" % (
            wigner_more_coherent, len(all_results), avg_disagree)
    elif avg_overlap < 0.95:
        verdict = "PARTIAL PARADOX: observers disagree (overlap=%.3f)" % avg_overlap
    else:
        verdict = "NO PARADOX: observers agree (overlap=%.3f, disagree=%.3f)" % (avg_overlap, avg_disagree)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q261', 'name': "Wigner's Friend Paradox",
        'scenarios': all_results,
        'summary': {'avg_overlap': round(avg_overlap, 4), 'avg_disagreement': round(avg_disagree, 4),
                     'wigner_more_quantum': wigner_more_coherent, 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q261_wigner.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_results))
    ax = axes[0]
    ax.bar(x - 0.2, [r['friend']['coherence'] for r in all_results], 0.4,
           label='Friend (post-L22)', color='#607D8B', edgecolor='black')
    ax.bar(x + 0.2, [r['wigner']['coherence'] for r in all_results], 0.4,
           label='Wigner (pre-L22)', color='#E91E63', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Coherence'); ax.set_title("(a) Who Sees More Quantum?")
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, [r['overlap'] for r in all_results], color='#2196F3', edgecolor='black')
    ax.axhline(1.0, color='green', ls='--', label='Perfect agreement')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('State Overlap'); ax.set_title("(b) Observer Agreement")
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle("Q261: Wigner's Friend\n%s" % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q261_wigner.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ261 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
