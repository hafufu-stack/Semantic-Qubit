# -*- coding: utf-8 -*-
"""
Phase Q258: Entanglement Entropy Production Rate
===================================================
MY IDEA: Connect entanglement dynamics to chaos theory.
Measure entropy production rate per layer and compare to
the scrambling rate from Q232 (OTOC Lyapunov exponent).
If they match, Transformer information dynamics = quantum chaos.
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
    print("Phase Q258: Entanglement Entropy Production Rate")
    print("  (Chaos meets quantum: Lyapunov vs entropy production)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4; da, db = 2, 2

    prompts = [
        "quantum chaos scrambling dynamics",
        "classical harmonic oscillator motion",
        "black hole information paradox entropy",
        "protein folding energy landscape",
        "stock market volatility prediction",
        "neural network gradient flow",
    ]

    scrambling_rate_q232 = 0.012  # From Q232

    all_production_rates = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        entropies = []
        negativities = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)

            ev = np.real(np.linalg.eigvalsh(rho))
            ev = ev[ev > 1e-12]
            S = float(-np.sum(ev * np.log2(ev))) if len(ev) > 0 else 0
            entropies.append(S)

            eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
            neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
            negativities.append(neg)

        # Entropy production rate = dS/dL
        dS = np.diff(entropies)
        avg_production = float(np.mean(np.abs(dS)))

        # Peak production rate
        peak_rate = float(np.max(np.abs(dS)))
        peak_layer = int(np.argmax(np.abs(dS))) + 1

        # Kolmogorov-Sinai entropy proxy: positive entropy production
        ks_entropy = float(np.mean(dS[dS > 0])) if np.any(dS > 0) else 0

        all_production_rates.append({
            'prompt': prompt[:30],
            'avg_production': round(avg_production, 6),
            'peak_rate': round(peak_rate, 6),
            'peak_layer': peak_layer,
            'ks_entropy': round(ks_entropy, 6),
        })

    avg_prod = np.mean([r['avg_production'] for r in all_production_rates])
    avg_ks = np.mean([r['ks_entropy'] for r in all_production_rates])

    # Compare to scrambling rate
    ratio = avg_prod / max(scrambling_rate_q232, 1e-10)

    if 0.5 < ratio < 2.0:
        verdict = "CHAOS=QUANTUM: entropy production (%.4f) matches scrambling rate (%.4f), ratio=%.2f" % (
            avg_prod, scrambling_rate_q232, ratio)
    elif ratio > 2.0:
        verdict = "FASTER ENTROPY: production=%.4f >> scrambling=%.4f (ratio=%.1f)" % (
            avg_prod, scrambling_rate_q232, ratio)
    else:
        verdict = "SLOWER ENTROPY: production=%.4f < scrambling=%.4f (ratio=%.2f)" % (
            avg_prod, scrambling_rate_q232, ratio)

    print("\n--- Summary ---")
    print("  Avg entropy production: %.6f bits/layer" % avg_prod)
    print("  Avg KS entropy: %.6f" % avg_ks)
    print("  Scrambling rate (Q232): %.4f" % scrambling_rate_q232)
    print("  %s" % verdict)

    results = {
        'phase': 'Q258', 'name': 'Entanglement Entropy Production Rate',
        'prompts': all_production_rates,
        'scrambling_rate_q232': scrambling_rate_q232,
        'summary': {'avg_production': round(avg_prod, 6), 'avg_ks': round(avg_ks, 6),
                     'ratio_to_scrambling': round(ratio, 2), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q258_entropy_production.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(all_production_rates))
    ax.bar(x, [r['avg_production'] for r in all_production_rates], color='#E91E63', edgecolor='black', alpha=0.7)
    ax.axhline(scrambling_rate_q232, color='#2196F3', ls='--', lw=2, label='Scrambling rate (Q232)')
    ax.set_xticks(x); ax.set_xticklabels([r['prompt'][:12] for r in all_production_rates], fontsize=7, rotation=30)
    ax.set_ylabel('Entropy Production (bits/layer)')
    ax.set_title('(a) Per-Prompt Entropy Production'); ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    bars = ax.bar(['Entropy\nProduction', 'KS Entropy', 'Scrambling\nRate (Q232)'],
                  [avg_prod, avg_ks, scrambling_rate_q232],
                  color=['#E91E63', '#FF9800', '#2196F3'], edgecolor='black')
    ax.set_ylabel('Rate'); ax.set_title('(b) Comparison')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q258: Entanglement Entropy Production\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q258_entropy_production.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ258 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
