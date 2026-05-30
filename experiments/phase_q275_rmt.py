# -*- coding: utf-8 -*-
"""
Phase Q275: Random Matrix Theory - Level Spacing Statistics
=============================================================
Quantum chaos signature: eigenvalue spacing follows
Wigner-Dyson (GOE) distribution, not Poisson.
This is THE diagnostic for quantum vs classical chaos.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kstest

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def wigner_dyson_goe(s):
    """GOE level spacing distribution (Wigner surmise)."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def poisson_spacing(s):
    """Poisson level spacing distribution."""
    return np.exp(-s)

def main():
    print("=" * 60)
    print("Phase Q275: Random Matrix Theory")
    print("  (Level spacing: Wigner-Dyson vs Poisson)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 64

    prompts = [
        "quantum chaos scrambling dynamics",
        "integrable system conservation laws",
        "random matrix eigenvalue statistics",
        "neural network weight distribution",
        "classical harmonic oscillator energy",
        "turbulent fluid flow dynamics",
        "crystalline lattice vibration modes",
        "prime number distribution pattern",
    ]

    all_spacings = []
    per_prompt = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Construct "Hamiltonian" from weight matrix interaction
        h = out.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()
        h_mid = out.hidden_states[n_layers // 2][0, -1, :dim].float().cpu().numpy()

        H = np.outer(h, h_mid)
        H = (H + H.T) / 2  # Hermitianize
        eigvals = np.sort(np.linalg.eigvalsh(H))

        # Unfolding: normalize spacings to mean=1
        spacings = np.diff(eigvals)
        if np.mean(np.abs(spacings)) > 1e-10:
            spacings = spacings / np.mean(np.abs(spacings))
            spacings = np.abs(spacings)
            all_spacings.extend(spacings.tolist())

            # KS test against Poisson and Wigner-Dyson
            # Poisson CDF: F(s) = 1 - exp(-s)
            ks_poisson, p_poisson = kstest(spacings, 'expon')
            # For Wigner-Dyson, compare to half-normal-like
            ks_wd = float(np.mean(np.abs(np.sort(spacings) -
                          np.sort(np.random.RandomState(42).rayleigh(np.sqrt(2/np.pi), len(spacings))))))

            per_prompt.append({
                'prompt': prompt[:30],
                'ks_poisson': round(ks_poisson, 4),
                'p_poisson': round(p_poisson, 4),
                'ks_wd': round(ks_wd, 4),
                'mean_spacing': round(float(np.mean(spacings)), 4),
                'var_spacing': round(float(np.var(spacings)), 4),
            })
            print("  '%s': KS_Poisson=%.3f (p=%.3f), KS_WD=%.3f" % (
                prompt[:25], ks_poisson, p_poisson, ks_wd))

    all_spacings = np.array(all_spacings)
    # Overall statistics
    if len(all_spacings) > 10:
        ks_poisson_all, p_poisson_all = kstest(all_spacings, 'expon')
    else:
        ks_poisson_all, p_poisson_all = 1, 0

    # Brody parameter: interpolates between Poisson (beta=0) and GOE (beta=1)
    # P(s) ~ s^beta * exp(-c * s^(beta+1))
    # Simple estimate: beta ~ <s^2> / <s>^2 - 1 mapped to [0,1]
    if len(all_spacings) > 0:
        s2_s = np.mean(all_spacings**2) / (np.mean(all_spacings)**2 + 1e-10)
        # Poisson: <s^2>/<s>^2 = 2, GOE: ~1.27
        brody = max(0, min(1, (2 - s2_s) / (2 - 1.27)))
    else:
        brody = 0

    if brody > 0.6:
        verdict = "QUANTUM CHAOS (GOE): Brody=%.2f, repulsion present" % brody
    elif brody > 0.3:
        verdict = "INTERMEDIATE: Brody=%.2f (between Poisson and GOE)" % brody
    else:
        verdict = "INTEGRABLE (Poisson): Brody=%.2f, no level repulsion" % brody

    print("\n--- Summary ---")
    print("  Brody parameter: %.3f" % brody)
    print("  %s" % verdict)

    results = {
        'phase': 'Q275', 'name': 'Random Matrix Theory',
        'per_prompt': per_prompt,
        'summary': {'brody': round(brody, 3),
                     'ks_poisson': round(ks_poisson_all, 4),
                     'n_spacings': len(all_spacings), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q275_rmt.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(all_spacings[all_spacings < 4], bins=40, density=True,
            color='#2196F3', edgecolor='black', alpha=0.7, label='LLM data')
    s_range = np.linspace(0, 4, 200)
    ax.plot(s_range, poisson_spacing(s_range), 'r--', lw=2, label='Poisson')
    ax.plot(s_range, wigner_dyson_goe(s_range), 'g-', lw=2, label='Wigner-Dyson (GOE)')
    ax.set_xlabel('Normalized Spacing s'); ax.set_ylabel('P(s)')
    ax.set_title('(a) Level Spacing Distribution'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(per_prompt))
    ax.bar(x, [r['ks_poisson'] for r in per_prompt], 0.4, label='KS vs Poisson',
           color='#F44336', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(per_prompt))], fontsize=7)
    ax.set_ylabel('KS Statistic'); ax.set_title('(b) Deviation from Poisson')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q275: Random Matrix Theory (Brody=%.2f)\n%s' % (brody, verdict[:60]),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q275_rmt.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ275 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
