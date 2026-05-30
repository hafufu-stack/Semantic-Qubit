# -*- coding: utf-8 -*-
"""
Phase Q237: Coherence-Entanglement Complementarity
=====================================================
Quantum mechanics has a fundamental trade-off:
coherence + entanglement <= maximum (complementarity).
Does the LLM respect this quantum law?
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
    print("Phase Q237: Coherence-Entanglement Complementarity")
    print("  (Does the LLM obey quantum complementarity?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 4
    da, db = 2, 2

    prompts = [
        "quantum coherent superposition", "decoherence thermal bath",
        "entangled Bell pair", "separable product state",
        "GHZ state multipartite", "vacuum ground state",
        "Schrodinger cat state", "laser coherent light",
    ]

    all_data = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(0, n_layers + 1, 2):
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)

            # Coherence: l1-norm normalized
            l1 = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)

            # Entanglement: negativity normalized
            eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
            neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
            neg_norm = min(neg / 0.5, 1.0)  # normalize by max negativity for 2x2

            # Complementarity check: C^2 + E^2 <= 1 (wave-particle-entanglement)
            sum_sq = l1**2 + neg_norm**2
            respects = sum_sq <= 1.05  # allow 5% tolerance

            all_data.append({
                'prompt': prompt[:25], 'layer': li,
                'coherence': round(l1, 4), 'entanglement': round(neg_norm, 4),
                'sum_sq': round(sum_sq, 4), 'respects': bool(respects),
            })

    # Summary
    n_respects = sum(1 for d in all_data if d['respects'])
    total = len(all_data)
    avg_sum = np.mean([d['sum_sq'] for d in all_data])

    if n_respects / total > 0.9:
        verdict = "COMPLEMENTARITY OBEYED: %d/%d (avg C^2+E^2=%.3f)" % (n_respects, total, avg_sum)
    elif n_respects / total > 0.5:
        verdict = "PARTIAL COMPLEMENTARITY: %d/%d" % (n_respects, total)
    else:
        verdict = "COMPLEMENTARITY VIOLATED: %d/%d" % (n_respects, total)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q237', 'name': 'Coherence-Entanglement Complementarity',
        'data': all_data,
        'summary': {'n_respects': n_respects, 'total': total, 'avg_sum_sq': round(avg_sum, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q237_complementarity.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    cohs = [d['coherence'] for d in all_data]
    ents = [d['entanglement'] for d in all_data]
    colors = ['#4CAF50' if d['respects'] else '#F44336' for d in all_data]
    ax.scatter(cohs, ents, c=colors, alpha=0.5, s=20, edgecolors='none')
    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=2, label='C^2+E^2=1')
    ax.set_xlabel('Coherence (normalized)'); ax.set_ylabel('Entanglement (normalized)')
    ax.set_title('(a) Complementarity Plane'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)

    ax = axes[1]
    layers_unique = sorted(set(d['layer'] for d in all_data))
    avg_coh = [np.mean([d['coherence'] for d in all_data if d['layer'] == l]) for l in layers_unique]
    avg_ent = [np.mean([d['entanglement'] for d in all_data if d['layer'] == l]) for l in layers_unique]
    ax.plot(layers_unique, avg_coh, 'o-', color='#9C27B0', label='Coherence', ms=3)
    ax.plot(layers_unique, avg_ent, 's-', color='#E91E63', label='Entanglement', ms=3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Value')
    ax.set_title('(b) Layer Evolution'); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q237: Coherence-Entanglement Complementarity\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q237_complementarity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ237 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
