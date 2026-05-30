# -*- coding: utf-8 -*-
"""
Phase Q251: Quantum Decoherence Map
======================================
MY IDEA: At which layer does "quantum" become "classical"?
Track the quantum-classical transition by measuring decoherence
rate (loss of coherence and entanglement) per layer.
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
    print("Phase Q251: Quantum Decoherence Map")
    print("  (Where does quantum become classical?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4; da, db = 2, 2

    prompts = [
        "quantum superposition of electron states",
        "classical mechanics planetary motion",
        "thermodynamic entropy heat transfer",
        "quantum computing error correction",
        "neural network deep learning",
    ]

    all_profiles = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        prev_neg, prev_coh = None, None
        layers = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)

            eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
            neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
            coh = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)
            purity = float(np.real(np.trace(rho @ rho)))

            d_neg = neg - prev_neg if prev_neg is not None else 0
            d_coh = coh - prev_coh if prev_coh is not None else 0
            prev_neg, prev_coh = neg, coh

            layers.append({
                'layer': li, 'neg': round(neg, 6), 'coh': round(coh, 4),
                'purity': round(purity, 4),
                'decoherence_neg': round(d_neg, 6),
                'decoherence_coh': round(d_coh, 6),
            })

        all_profiles.append({'prompt': prompt[:30], 'layers': layers})

    # Find transition layer (where decoherence is strongest)
    avg_dcoh = np.zeros(n_layers + 1)
    avg_dneg = np.zeros(n_layers + 1)
    for p in all_profiles:
        for d in p['layers']:
            avg_dcoh[d['layer']] += d['decoherence_coh']
            avg_dneg[d['layer']] += d['decoherence_neg']
    avg_dcoh /= len(all_profiles)
    avg_dneg /= len(all_profiles)

    # Transition = where decoherence rate is most negative
    transition_layer = int(np.argmin(avg_dcoh[1:])) + 1
    max_decoherence = float(np.min(avg_dcoh[1:]))

    verdict = "Quantum-Classical transition at layer %d (max decoherence=%.4f)" % (
        transition_layer, max_decoherence)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q251', 'name': 'Quantum Decoherence Map',
        'profiles': all_profiles,
        'summary': {'transition_layer': transition_layer,
                     'max_decoherence': round(max_decoherence, 6), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q251_decoherence.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for p in all_profiles:
        ls = [d['layer'] for d in p['layers']]
        ax.plot(ls, [d['coh'] for d in p['layers']], 'o-', ms=2, alpha=0.6, label=p['prompt'][:15])
    ax.axvline(transition_layer, color='red', ls='--', lw=2, label='Transition')
    ax.set_xlabel('Layer'); ax.set_ylabel('Coherence')
    ax.set_title('(a) Coherence per Layer'); ax.legend(fontsize=6); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(range(1, n_layers + 1), avg_dcoh[1:],
           color=['#F44336' if d < 0 else '#4CAF50' for d in avg_dcoh[1:]], edgecolor='none')
    ax.axhline(0, color='black', ls='-'); ax.axvline(transition_layer, color='red', ls='--', lw=2)
    ax.set_xlabel('Layer'); ax.set_ylabel('dCoherence/dLayer')
    ax.set_title('(b) Decoherence Rate'); ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    for p in all_profiles:
        ls = [d['layer'] for d in p['layers']]
        ax.plot(ls, [d['neg'] for d in p['layers']], 'o-', ms=2, alpha=0.6)
    ax.axvline(transition_layer, color='red', ls='--', lw=2)
    ax.set_xlabel('Layer'); ax.set_ylabel('Negativity')
    ax.set_title('(c) Entanglement per Layer'); ax.grid(alpha=0.3)

    plt.suptitle('Q251: Quantum Decoherence Map\n%s' % verdict, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q251_decoherence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ251 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
