# -*- coding: utf-8 -*-
"""
Phase Q274: Landauer's Principle
===================================
Is there a minimum energy cost for erasing information in LLM?
Landauer: erasing 1 bit costs kT*ln(2) energy.
Test by measuring hidden state entropy change during "forgetting".
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

def main():
    print("=" * 60)
    print("Phase Q274: Landauer's Principle")
    print("  (Minimum cost of information erasure)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 16

    # Pairs: information-rich -> information-erased
    erasure_pairs = [
        ("The capital of France is Paris", "I don't know anything"),
        ("Einstein discovered E equals mc squared", "Nothing happened ever"),
        ("Water boils at 100 degrees Celsius", "No information available"),
        ("The speed of light is 299792458 meters", "Everything is unknown"),
        ("DNA has a double helix structure", "I have forgotten everything"),
    ]

    all_results = []
    for info_text, erase_text in erasure_pairs:
        # Information state
        inp_i = tok(info_text, return_tensors='pt').to(device)
        with torch.no_grad():
            out_i = model(**inp_i, output_hidden_states=True)

        # Erased state
        inp_e = tok(erase_text, return_tensors='pt').to(device)
        with torch.no_grad():
            out_e = model(**inp_e, output_hidden_states=True)

        # Track entropy through layers for both
        S_info = []; S_erase = []; norms_info = []; norms_erase = []
        for li in range(n_layers + 1):
            for out, S_list, n_list in [(out_i, S_info, norms_info), (out_e, S_erase, norms_erase)]:
                h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
                n_list.append(float(np.linalg.norm(h)))
                h_norm = h / (np.linalg.norm(h) + 1e-10)
                rho = np.outer(h_norm, h_norm.conj())
                rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
                rho /= np.trace(rho)
                ev = np.real(np.linalg.eigvalsh(rho))
                ev_pos = ev[ev > 1e-12]
                S = float(-np.sum(ev_pos * np.log2(ev_pos))) if len(ev_pos) > 0 else 0
                S_list.append(S)

        # Entropy change = "heat dissipated"
        delta_S = S_erase[-1] - S_info[-1]
        # Norm change = "energy cost"
        delta_norm = norms_erase[-1] - norms_info[-1]

        # Landauer bound: delta_S >= 0 when erasing (entropy must increase or stay)
        landauer_satisfied = delta_S >= -0.01  # Small tolerance

        print("  '%s' -> '%s'" % (info_text[:25], erase_text[:20]))
        print("    S_info=%.4f, S_erase=%.4f, delta_S=%.4f (Landauer: %s)" % (
            S_info[-1], S_erase[-1], delta_S, "YES" if landauer_satisfied else "NO"))

        all_results.append({
            'info': info_text[:30], 'erase': erase_text[:25],
            'S_info_final': round(S_info[-1], 4),
            'S_erase_final': round(S_erase[-1], 4),
            'delta_S': round(delta_S, 4),
            'delta_norm': round(delta_norm, 4),
            'landauer': bool(landauer_satisfied),
            'S_info_profile': [round(s, 3) for s in S_info[::5]],
            'S_erase_profile': [round(s, 3) for s in S_erase[::5]],
        })

    n_landauer = sum(1 for r in all_results if r['landauer'])
    avg_delta_S = np.mean([r['delta_S'] for r in all_results])

    if n_landauer == len(all_results):
        verdict = "LANDAUER HOLDS: %d/%d, avg delta_S=%.3f bits" % (n_landauer, len(all_results), avg_delta_S)
    elif n_landauer > len(all_results) // 2:
        verdict = "PARTIAL LANDAUER: %d/%d satisfy bound" % (n_landauer, len(all_results))
    else:
        verdict = "LANDAUER VIOLATED: only %d/%d (info erasure too cheap?)" % (n_landauer, len(all_results))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q274', 'name': "Landauer's Principle",
        'pairs': all_results,
        'summary': {'n_landauer': n_landauer, 'total': len(all_results),
                     'avg_delta_S': round(avg_delta_S, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q274_landauer.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_results))
    ax = axes[0]
    ax.bar(x - 0.2, [r['S_info_final'] for r in all_results], 0.4,
           label='Info state', color='#2196F3', edgecolor='black')
    ax.bar(x + 0.2, [r['S_erase_final'] for r in all_results], 0.4,
           label='Erased state', color='#F44336', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Entropy (bits)'); ax.set_title("(a) Information vs Erased")
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    colors = ['#4CAF50' if r['landauer'] else '#F44336' for r in all_results]
    ax.bar(x, [r['delta_S'] for r in all_results], color=colors, edgecolor='black')
    ax.axhline(0, color='red', ls='--', lw=2, label='Landauer bound')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Delta S (bits)'); ax.set_title("(b) Landauer's Bound")
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle("Q274: Landauer's Principle\n%s" % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q274_landauer.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ274 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
