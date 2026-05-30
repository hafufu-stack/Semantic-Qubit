# -*- coding: utf-8 -*-
"""
Phase Q257: Quantum Critical Exponents at Layer 22
=====================================================
MY IDEA: Is the Layer 22 transition a TRUE quantum phase transition?
Measure critical exponents (correlation length, susceptibility)
near the transition point. If power-law scaling exists, it's genuine.
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
    print("Phase Q257: Quantum Critical Exponents")
    print("  (Is Layer 22 a genuine quantum phase transition?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 8
    tc = 22  # Critical layer from Q251

    prompts = [
        "quantum entanglement superposition", "classical physics mechanics",
        "information entropy processing", "mathematical proof theorem",
        "neural network optimization", "molecular orbital chemistry",
        "electromagnetic field theory", "statistical mechanics equilibrium",
    ]

    # Measure order parameter, susceptibility, correlation length per layer
    all_profiles = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        profile = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10

            # Order parameter: off-diagonal coherence
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)
            order_param = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)

            # Susceptibility: variance of order parameter fluctuations
            # (approximate via eigenvalue spread)
            ev = np.real(np.linalg.eigvalsh(rho))
            susceptibility = float(np.var(ev)) * dim * 100  # Scale up

            # Correlation: how similar to adjacent layers
            if li > 0:
                h_prev = out.hidden_states[li-1][0, -1, :dim].float().cpu().numpy()
                h_prev /= np.linalg.norm(h_prev) + 1e-10
                correlation = float(np.dot(h, h_prev))
            else:
                correlation = 1.0

            profile.append({
                'layer': li,
                'order_param': round(order_param, 4),
                'susceptibility': round(susceptibility, 4),
                'correlation': round(correlation, 4),
            })
        all_profiles.append(profile)

    # Average across prompts
    avg_order = np.zeros(n_layers + 1)
    avg_susc = np.zeros(n_layers + 1)
    avg_corr = np.zeros(n_layers + 1)
    for p in all_profiles:
        for d in p:
            avg_order[d['layer']] += d['order_param']
            avg_susc[d['layer']] += d['susceptibility']
            avg_corr[d['layer']] += d['correlation']
    avg_order /= len(prompts)
    avg_susc /= len(prompts)
    avg_corr /= len(prompts)

    # Check for critical behavior near L22
    # Susceptibility peak?
    peak_susc_layer = int(np.argmax(avg_susc))
    # Correlation drop?
    min_corr_layer = int(np.argmin(avg_corr[1:])) + 1
    # Order parameter drop?
    order_drop = avg_order[:tc].mean() - avg_order[tc:].mean()

    # Power-law fit near critical point
    # |m(L)| ~ |L - Lc|^beta
    near_tc = [(li, avg_order[li]) for li in range(max(0, tc-5), min(n_layers+1, tc+6)) if li != tc]
    if near_tc:
        x_fit = np.array([abs(li - tc) + 0.1 for li, _ in near_tc])
        y_fit = np.array([m for _, m in near_tc])
        # Log-log fit
        valid = (x_fit > 0) & (y_fit > 0)
        if np.sum(valid) > 2:
            log_x = np.log(x_fit[valid])
            log_y = np.log(y_fit[valid])
            beta, intercept = np.polyfit(log_x, log_y, 1)
        else:
            beta = 0
    else:
        beta = 0

    is_critical = abs(peak_susc_layer - tc) <= 3 and order_drop > 0.01
    if is_critical:
        verdict = "CRITICAL POINT at L%d: beta=%.2f, peak susc at L%d, order drop=%.3f" % (
            tc, beta, peak_susc_layer, order_drop)
    else:
        verdict = "NOT CRITICAL: susc peak at L%d (expected L%d), order drop=%.3f" % (
            peak_susc_layer, tc, order_drop)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q257', 'name': 'Quantum Critical Exponents',
        'critical_layer': tc,
        'peak_susceptibility_layer': peak_susc_layer,
        'min_correlation_layer': min_corr_layer,
        'order_drop': round(order_drop, 4),
        'critical_exponent_beta': round(beta, 3),
        'summary': {'is_critical': bool(is_critical), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q257_critical.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layers = list(range(n_layers + 1))

    ax = axes[0]
    ax.plot(layers, avg_order, 'o-', color='#E91E63', ms=3, lw=1.5)
    ax.axvline(tc, color='red', ls='--', lw=2, label='L%d' % tc)
    ax.set_xlabel('Layer'); ax.set_ylabel('Order Parameter')
    ax.set_title('(a) Order Parameter'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(layers, avg_susc, 'o-', color='#2196F3', ms=3, lw=1.5)
    ax.axvline(tc, color='red', ls='--', lw=2, label='L%d' % tc)
    ax.set_xlabel('Layer'); ax.set_ylabel('Susceptibility')
    ax.set_title('(b) Susceptibility (peak at L%d)' % peak_susc_layer); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(layers[1:], avg_corr[1:], 'o-', color='#4CAF50', ms=3, lw=1.5)
    ax.axvline(tc, color='red', ls='--', lw=2, label='L%d' % tc)
    ax.set_xlabel('Layer'); ax.set_ylabel('Correlation')
    ax.set_title('(c) Inter-layer Correlation'); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q257: Quantum Critical Exponents\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q257_critical.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ257 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
