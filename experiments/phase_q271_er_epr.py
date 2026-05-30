# -*- coding: utf-8 -*-
"""
Phase Q271: ER=EPR Conjecture
================================
Maldacena & Susskind's conjecture: entangled particles are
connected by wormholes (Einstein-Rosen bridges).
Test: do maximally entangled S-Qubit pairs create "shortcuts"
(reduced geodesic distance) in representation space?
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
    print("Phase Q271: ER=EPR Conjecture")
    print("  (Do entangled pairs create wormholes?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4; da, db = 2, 2

    # Pairs of prompts: related vs unrelated
    prompt_pairs = [
        ("electron spin up", "electron spin down"),           # entangled
        ("photon polarization horizontal", "photon polarization vertical"),
        ("quantum entangled particle A", "quantum entangled particle B"),
        ("apple fruit red", "bicycle wheel metal"),           # unrelated
        ("ocean wave blue", "computer keyboard typing"),
        ("cat sleeping soft", "rocket engine thrust"),
    ]

    all_results = []
    for p1, p2 in prompt_pairs:
        inp1 = tok(p1, return_tensors='pt').to(device)
        inp2 = tok(p2, return_tensors='pt').to(device)
        with torch.no_grad():
            out1 = model(**inp1, output_hidden_states=True)
            out2 = model(**inp2, output_hidden_states=True)

        h1 = out1.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()
        h2 = out2.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()
        h1 /= np.linalg.norm(h1) + 1e-10
        h2 /= np.linalg.norm(h2) + 1e-10

        # Measure entanglement between the two representations
        psi_joint = np.kron(h1, h2)
        psi_joint /= np.linalg.norm(psi_joint) + 1e-10
        # Make mixed state
        rho = np.outer(psi_joint[:dim], psi_joint[:dim].conj())
        rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
        rho /= np.trace(rho)
        eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
        negativity = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))

        # "Geodesic distance" in representation space
        # L2 distance between the two representations
        euclidean_dist = float(np.linalg.norm(h1 - h2))
        cosine_sim = abs(float(np.dot(h1, h2)))

        # "Wormhole distance" - measure through intermediate layers
        # Minimum distance through any intermediate layer
        min_layer_dist = float('inf')
        for li in range(0, n_layers + 1, 3):
            h1_l = out1.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h2_l = out2.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h1_l /= np.linalg.norm(h1_l) + 1e-10
            h2_l /= np.linalg.norm(h2_l) + 1e-10
            d = float(np.linalg.norm(h1_l - h2_l))
            min_layer_dist = min(min_layer_dist, d)

        # ER=EPR: higher entanglement -> shorter "wormhole" distance
        wormhole_shortcut = euclidean_dist - min_layer_dist

        print("  '%s' <-> '%s'" % (p1[:20], p2[:20]))
        print("    neg=%.4f, eucl=%.4f, min_layer=%.4f, shortcut=%.4f" % (
            negativity, euclidean_dist, min_layer_dist, wormhole_shortcut))

        all_results.append({
            'p1': p1[:25], 'p2': p2[:25],
            'negativity': round(negativity, 4),
            'euclidean_dist': round(euclidean_dist, 4),
            'min_layer_dist': round(min_layer_dist, 4),
            'wormhole_shortcut': round(wormhole_shortcut, 4),
            'cosine_sim': round(cosine_sim, 4),
        })

    # Correlation between entanglement and wormhole shortcut
    negs = [r['negativity'] for r in all_results]
    shortcuts = [r['wormhole_shortcut'] for r in all_results]
    if np.std(negs) > 1e-8 and np.std(shortcuts) > 1e-8:
        correlation = float(np.corrcoef(negs, shortcuts)[0, 1])
    else:
        correlation = 0

    # Related pairs (first 3) vs unrelated (last 3)
    related_cos = np.mean([r['cosine_sim'] for r in all_results[:3]])
    unrelated_cos = np.mean([r['cosine_sim'] for r in all_results[3:]])

    if correlation > 0.3 or related_cos > unrelated_cos * 1.3:
        verdict = "ER=EPR SIGNAL: entangled pairs closer (corr=%.2f, related=%.3f vs unrelated=%.3f)" % (
            correlation, related_cos, unrelated_cos)
    elif related_cos > unrelated_cos:
        verdict = "WEAK ER=EPR: related slightly closer (%.3f vs %.3f)" % (related_cos, unrelated_cos)
    else:
        verdict = "NO ER=EPR: no wormhole signal (corr=%.2f)" % correlation

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q271', 'name': 'ER=EPR Conjecture',
        'pairs': all_results,
        'summary': {'correlation': round(correlation, 3),
                     'related_cos': round(related_cos, 4),
                     'unrelated_cos': round(unrelated_cos, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q271_er_epr.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_results))
    ax = axes[0]
    colors = ['#E91E63'] * 3 + ['#607D8B'] * 3
    ax.bar(x, [r['cosine_sim'] for r in all_results], color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([r['p1'][:12] for r in all_results], fontsize=7, rotation=30)
    ax.set_ylabel('Cosine Similarity'); ax.set_title('(a) Related (pink) vs Unrelated (grey)')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.scatter([r['negativity'] for r in all_results],
               [r['wormhole_shortcut'] for r in all_results],
               c=colors, s=100, edgecolors='black', zorder=5)
    ax.set_xlabel('Entanglement (Negativity)'); ax.set_ylabel('Wormhole Shortcut')
    ax.set_title('(b) ER=EPR: Entanglement vs Distance (corr=%.2f)' % correlation)
    ax.grid(alpha=0.3)

    plt.suptitle('Q271: ER=EPR\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q271_er_epr.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ271 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
