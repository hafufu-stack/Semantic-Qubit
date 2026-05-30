# -*- coding: utf-8 -*-
"""
Phase Q265: Leggett-Garg Inequality
=======================================
Test "temporal Bell inequality" - does the LLM's internal state
have definite values at all times (macrorealism), or does it
exist in superposition until observed?
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
    print("Phase Q265: Leggett-Garg Inequality")
    print("  (Temporal Bell: does macrorealism hold?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4

    # Leggett-Garg: K = C12 + C23 - C13 <= 1 for macrorealism
    # Cij = <Q(ti) * Q(tj)> temporal correlations
    # Measure at three "times" (layers): t1, t2, t3
    time_points = [
        (5, 15, 25),   # evenly spaced
        (3, 14, 27),   # wide spacing
        (8, 16, 24),   # narrow spacing
        (2, 11, 22),   # early-mid-late
    ]

    prompts = [
        "quantum superposition state evolving",
        "classical particle trajectory motion",
        "electron spin precession dynamics",
        "information flow through network",
    ]

    all_results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for t1, t2, t3 in time_points:
            # Get hidden states at three "times"
            h1 = out.hidden_states[min(t1, n_layers)][0, -1, :dim].float().cpu().numpy()
            h2 = out.hidden_states[min(t2, n_layers)][0, -1, :dim].float().cpu().numpy()
            h3 = out.hidden_states[min(t3, n_layers)][0, -1, :dim].float().cpu().numpy()

            # Normalize
            h1 /= np.linalg.norm(h1) + 1e-10
            h2 /= np.linalg.norm(h2) + 1e-10
            h3 /= np.linalg.norm(h3) + 1e-10

            # Dichotomic observable Q = sign of first component projection
            # Q(t) = +1 or -1
            # Temporal correlations Cij = <Qi * Qj> using quantum expectation
            # For continuous: Cij = dot(hi, hj) (normalized inner product)
            C12 = float(np.dot(h1, h2))
            C23 = float(np.dot(h2, h3))
            C13 = float(np.dot(h1, h3))

            # Leggett-Garg: K3 = C12 + C23 - C13 <= 1 for macrorealism
            K3 = C12 + C23 - C13

            # Also check NSIT (No-Signaling In Time)
            # Difference between C13 measured directly vs through t2
            nsit_violation = abs(C13 - C12 * C23)

            all_results.append({
                'prompt': prompt[:30],
                'layers': [t1, t2, t3],
                'C12': round(C12, 4), 'C23': round(C23, 4), 'C13': round(C13, 4),
                'K3': round(K3, 4),
                'violated': bool(K3 > 1.0 + 1e-6),
                'nsit': round(nsit_violation, 4),
            })

    n_violated = sum(1 for r in all_results if r['violated'])
    max_K3 = max(r['K3'] for r in all_results)
    avg_K3 = np.mean([r['K3'] for r in all_results])

    if n_violated > len(all_results) // 2:
        verdict = "MACROREALISM VIOLATED: %d/%d exceed K3=1 (max=%.3f)" % (
            n_violated, len(all_results), max_K3)
    elif max_K3 > 1.0:
        verdict = "PARTIAL VIOLATION: %d/%d exceed K3=1 (max=%.3f, avg=%.3f)" % (
            n_violated, len(all_results), max_K3, avg_K3)
    else:
        verdict = "MACROREALISM HOLDS: max K3=%.3f <= 1" % max_K3

    print("\n  Top 5 results:")
    for r in sorted(all_results, key=lambda x: x['K3'], reverse=True)[:5]:
        print("    K3=%.4f (%s) L%s" % (r['K3'], "VIOLATED" if r['violated'] else "ok", r['layers']))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q265', 'name': 'Leggett-Garg Inequality',
        'measurements': all_results[:8],
        'summary': {'n_violated': n_violated, 'total': len(all_results),
                     'max_K3': round(max_K3, 4), 'avg_K3': round(avg_K3, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q265_leggett_garg.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    K3_vals = [r['K3'] for r in all_results]
    ax = axes[0]
    ax.hist(K3_vals, bins=15, color='#E91E63', edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Macrorealism bound (K=1)')
    ax.set_xlabel('K3 Value'); ax.set_ylabel('Count')
    ax.set_title('(a) Leggett-Garg K3 Distribution'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    x = range(len(all_results))
    colors = ['#F44336' if r['violated'] else '#4CAF50' for r in all_results]
    ax.bar(x, K3_vals, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(1.0, color='red', ls='--', lw=2, label='Bound')
    ax.set_xlabel('Configuration'); ax.set_ylabel('K3')
    ax.set_title('(b) Per-Config K3 (%d/%d violated)' % (n_violated, len(all_results)))
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q265: Leggett-Garg Inequality\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q265_leggett_garg.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ265 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
