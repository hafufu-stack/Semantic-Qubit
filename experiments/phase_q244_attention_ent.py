# -*- coding: utf-8 -*-
"""
Phase Q244: Attention-Entanglement Connection
================================================
THE BRIDGE: Does attention mechanism create entanglement?
Hypothesis: high attention weight between tokens -> high entanglement.
This would explain WHY Transformers have quantum properties.
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

def compute_attention_proxy(hidden_states, layer_idx):
    """Compute attention-like metric from hidden state changes.
    Proxy: how much the representation changes between layers (mixing)."""
    h_curr = hidden_states[layer_idx + 1][0, -1, :].float().cpu().numpy()
    h_prev = hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
    h_curr /= np.linalg.norm(h_curr) + 1e-10
    h_prev /= np.linalg.norm(h_prev) + 1e-10
    # Cosine similarity = how much attention "mixes"
    cos_sim = float(np.dot(h_curr, h_prev))
    mixing = 1.0 - abs(cos_sim)  # Higher = more mixing = more "attention effect"
    # L2 change = total information flow
    l2_change = float(np.linalg.norm(h_curr - h_prev))
    return mixing, l2_change


def main():
    print("=" * 60)
    print("Phase Q244: Attention-Entanglement Connection")
    print("  (Does attention CREATE entanglement?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4; da, db = 2, 2

    prompts = [
        "quantum entanglement creates correlation between particles that are far apart",
        "the cat sat on the mat and looked at the bird",
        "Einstein Podolsky Rosen paradox demonstrates nonlocal quantum correlations",
        "simple arithmetic addition of two numbers gives a result",
        "Bell inequality violation proves quantum mechanics is correct",
        "the weather forecast predicts rain for tomorrow afternoon",
    ]

    all_results = []
    for prompt in prompts:
        print("\n--- %s ---" % prompt[:40])
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layer_data = []
        for li in range(n_layers):
            # Attention proxy from hidden state changes
            mixing, l2_change = compute_attention_proxy(out.hidden_states, li)

            # Entanglement at this layer
            h = out.hidden_states[li + 1][0, -1, :dim].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)
            eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
            neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))

            layer_data.append({
                'layer': li, 'mixing': round(mixing, 4),
                'l2_change': round(l2_change, 4), 'negativity': round(neg, 6),
            })

        # Correlation: mixing vs negativity
        mixings = [d['mixing'] for d in layer_data]
        negs = [d['negativity'] for d in layer_data]
        if np.std(mixings) > 1e-6 and np.std(negs) > 1e-6:
            corr = float(np.corrcoef(mixings, negs)[0, 1])
        else:
            corr = 0

        all_results.append({
            'prompt': prompt[:40], 'layers': layer_data,
            'mixing_ent_corr': round(corr, 4),
        })
        print("  Mixing-Entanglement correlation: %.3f" % corr)

    avg_corr = np.mean([r['mixing_ent_corr'] for r in all_results])
    if avg_corr > 0.3:
        verdict = "ATTENTION CREATES ENTANGLEMENT: avg r=%.2f (more mixing -> more entanglement)" % avg_corr
    elif avg_corr < -0.3:
        verdict = "FOCUSED ATTENTION -> ENTANGLEMENT: avg r=%.2f" % avg_corr
    else:
        verdict = "WEAK CONNECTION: avg r=%.2f" % avg_corr

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q244', 'name': 'Attention-Entanglement Connection',
        'prompts': all_results,
        'summary': {'avg_correlation': round(avg_corr, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q244_attention_ent.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, r in enumerate(all_results[:6]):
        ax = axes[idx // 3][idx % 3]
        layers = [d['layer'] for d in r['layers']]
        ax2 = ax.twinx()
        ax.plot(layers, [d['mixing'] for d in r['layers']], 'o-', color='#2196F3', ms=2, label='Mixing')
        ax2.plot(layers, [d['negativity'] for d in r['layers']], 's-', color='#E91E63', ms=2, label='Negativity')
        ax.set_xlabel('Layer'); ax.set_ylabel('Mixing (1-cos)', color='#2196F3')
        ax2.set_ylabel('Negativity', color='#E91E63')
        ax.set_title('%s\nr=%.2f' % (r['prompt'][:30], r['mixing_ent_corr']), fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle('Q244: Attention-Entanglement Connection\n%s' % verdict[:60], fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q244_attention_ent.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ244 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
