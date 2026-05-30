# -*- coding: utf-8 -*-
"""
Phase Q270: Entanglement Entropy Area Law
============================================
MY IDEA: In quantum field theory, entanglement entropy of a
region scales with the BOUNDARY AREA, not the volume.
Does LLM hidden state entanglement follow this fundamental law?
Test by varying the size of the "region" (number of dimensions).
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
    print("Phase Q270: Entanglement Entropy Area Law")
    print("  (Does S ~ boundary area or volume?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    prompts = [
        "quantum field theory vacuum state",
        "classical thermal equilibrium state",
        "information processing neural computation",
        "electromagnetic radiation spectrum",
        "gravitational field spacetime curvature",
        "biological neural network activity",
    ]

    region_sizes = [2, 4, 8, 16, 32, 64, 128]

    all_entropies = {sz: [] for sz in region_sizes}

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h_full = out.hidden_states[n_layers][0, -1, :].float().cpu().numpy()

        for sz in region_sizes:
            if sz > len(h_full):
                continue
            # Region A = first sz dimensions, Region B = rest
            h_region = h_full[:sz]
            h_region /= np.linalg.norm(h_region) + 1e-10

            # Entanglement entropy of region A
            # Construct reduced density matrix
            rho = np.outer(h_region, h_region.conj())
            rho = 0.7 * rho + 0.3 * np.eye(sz) / sz
            rho /= np.trace(rho)

            ev = np.real(np.linalg.eigvalsh(rho))
            ev_pos = ev[ev > 1e-12]
            S = float(-np.sum(ev_pos * np.log2(ev_pos))) if len(ev_pos) > 0 else 0

            all_entropies[sz].append(S)

    avg_S = {sz: round(np.mean(vals), 4) for sz, vals in all_entropies.items() if vals}
    sizes = sorted(avg_S.keys())
    entropies = [avg_S[s] for s in sizes]

    # Fit: S ~ L^alpha where L = region size
    # Area law: alpha ~ 0 (or log correction)
    # Volume law: alpha ~ 1
    if len(sizes) >= 3:
        log_sizes = np.log2(sizes)
        log_S = np.log2([max(s, 1e-10) for s in entropies])
        valid = np.isfinite(log_S)
        if np.sum(valid) >= 2:
            alpha, _ = np.polyfit(log_sizes[valid], log_S[valid], 1)
        else:
            alpha = 0
    else:
        alpha = 0

    # Boundary in 1D = constant (2 points), so area law means S = const
    # In higher D: S ~ L^(d-1) where d = dimension
    if abs(alpha) < 0.3:
        verdict = "AREA LAW: S ~ L^%.2f (alpha~0, boundary scaling)" % alpha
    elif alpha < 0.7:
        verdict = "SUB-VOLUME: S ~ L^%.2f (between area and volume)" % alpha
    else:
        verdict = "VOLUME LAW: S ~ L^%.2f (thermal/random state)" % alpha

    print("\n  Entropy vs region size:")
    for s, e in zip(sizes, entropies):
        print("    dim=%d: S=%.4f bits" % (s, e))
    print("  Scaling exponent alpha=%.3f" % alpha)
    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q270', 'name': 'Entanglement Entropy Area Law',
        'region_sizes': sizes,
        'avg_entropy': entropies,
        'scaling_exponent': round(alpha, 3),
        'summary': {'alpha': round(alpha, 3), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q270_area_law.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(sizes, entropies, 'o-', color='#E91E63', lw=2, ms=8)
    ax.set_xlabel('Region Size (dimensions)'); ax.set_ylabel('Entanglement Entropy (bits)')
    ax.set_title('(a) S vs Region Size'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.log2(sizes), [np.log2(max(e, 1e-10)) for e in entropies], 'o-', color='#2196F3', lw=2, ms=8)
    # Fit line
    x_fit = np.linspace(min(np.log2(sizes)), max(np.log2(sizes)), 50)
    if len(sizes) >= 3 and np.sum(valid) >= 2:
        a, b = np.polyfit(log_sizes[valid], log_S[valid], 1)
        ax.plot(x_fit, a * x_fit + b, '--', color='red', lw=2,
                label='Fit: alpha=%.2f' % a)
    ax.set_xlabel('log2(Region Size)'); ax.set_ylabel('log2(Entropy)')
    ax.set_title('(b) Log-Log (slope=scaling exponent)'); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q270: Area Law\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q270_area_law.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ270 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
