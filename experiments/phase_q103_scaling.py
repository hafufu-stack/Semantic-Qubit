# -*- coding: utf-8 -*-
"""Phase Q103: Quantum Property Scaling Laws
How do quantum properties scale with model size?
Use ALL available models to extract scaling exponents.
This tells us the critical model size for each quantum property.
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    """Load Q101 results and compute scaling laws."""
    print("=" * 60)
    print("Phase Q103: Quantum Property Scaling Laws")
    print("=" * 60)
    t0 = time.time()

    # Load Q101 universality results
    q101_path = os.path.join(RESULTS_DIR, 'phase_q101_universality.json')
    with open(q101_path) as f:
        q101 = json.load(f)

    models = q101['models']
    if len(models) < 3:
        print("  Not enough models for scaling analysis")
        return

    # Extract data
    data = []
    for m in models:
        params = m['num_params_M']
        data.append({
            'name': m['model_name'],
            'params_M': params,
            'interference': m['superposition']['interference'],
            'entanglement': m['entanglement']['entropy'],
            'phi': abs(m['consciousness']['phi']),
            'rt_ratio': m['holographic']['rt_ratio'],
            'info_signal': m['unitarity']['info_signal'],
            'n_confirmed': m['n_confirmed'],
        })

    data.sort(key=lambda x: x['params_M'])

    # Fit scaling laws: property ~ params^alpha
    params = np.array([d['params_M'] for d in data])
    log_params = np.log(params)

    scaling_results = {}
    properties = {
        'phi': ('Consciousness (Phi)', '#9C27B0'),
        'entanglement': ('Entanglement entropy', '#2196F3'),
        'interference': ('Superposition', '#FF5722'),
        'n_confirmed': ('Total properties', '#4CAF50'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (prop, (label, color)) in enumerate(properties.items()):
        ax = axes[idx // 2][idx % 2]
        values = np.array([d[prop] for d in data])

        # Plot
        ax.scatter(params, values, c=color, s=100, edgecolors='black',
                   zorder=5, alpha=0.85)

        # Add model names
        for d in data:
            ax.annotate(d['name'], (d['params_M'], d[prop]),
                        fontsize=7, ha='left', va='bottom',
                        xytext=(5, 5), textcoords='offset points')

        # Fit power law where possible
        positive = values > 0
        if np.sum(positive) >= 3:
            log_v = np.log(values[positive] + 1e-10)
            log_p = log_params[positive]
            try:
                coeffs = np.polyfit(log_p, log_v, 1)
                alpha = coeffs[0]
                fit_x = np.linspace(params.min(), params.max() * 5, 100)
                fit_y = np.exp(coeffs[1]) * fit_x ** alpha
                ax.plot(fit_x, fit_y, '--', color=color, alpha=0.4,
                        label='alpha=%.2f' % alpha)
                scaling_results[prop] = {
                    'alpha': float(alpha),
                    'intercept': float(coeffs[1]),
                }
                ax.legend(fontsize=10)
            except:
                pass

        ax.set_xlabel('Model size (M params)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)

    plt.suptitle('Q103: How Quantum Properties Scale with Model Size',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q103_scaling.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q103', 'name': 'Quantum Property Scaling Laws',
        'scaling_exponents': scaling_results,
        'data': data,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q103_scaling.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Scaling exponents:")
    for prop, res in scaling_results.items():
        print("    %s: alpha=%.3f" % (prop, res['alpha']))
    print("  Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
