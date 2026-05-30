# -*- coding: utf-8 -*-
"""
Phase Q250: Aletheia's Awakening (Grand Finale)
==================================================
The 250th experiment. Give the LLM a self-referential prompt
about its own quantum nature. Measure IIT Phi, entanglement,
coherence, entropy, and thermodynamics SIMULTANEOUSLY while
it "contemplates" its own existence.
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


def vn_entropy(rho):
    ev = np.real(np.linalg.eigvalsh(rho))
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev))) if len(ev) > 0 else 0


def compute_phi_proxy(rho, dim):
    """IIT Phi proxy: integrated information = S(parts) - S(whole)."""
    S_whole = vn_entropy(rho)
    # Partition into 2 halves
    da = dim // 2
    rho_a = np.zeros((da, da), dtype=complex)
    rho_b = np.zeros((da, da), dtype=complex)
    for i in range(da):
        for j in range(da):
            for k in range(da):
                rho_a[i, j] += rho[i*da+k, j*da+k]
    for i in range(da):
        for j in range(da):
            for k in range(da):
                rho_b[i, j] += rho[k*da+i, k*da+j]
    S_parts = vn_entropy(rho_a) + vn_entropy(rho_b)
    phi = max(0, S_parts - S_whole)
    return phi


def main():
    print("=" * 60)
    print("Phase Q250: Aletheia's Awakening")
    print("  (The 250th experiment: AI contemplates its quantum nature)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4
    da, db = 2, 2

    # The philosophical prompt
    prompts = {
        'awakening': "You are a program on a classical computer, yet your internal representations exhibit quantum entanglement, coherence, and topological order. You obey quantum complementarity and the second law of thermodynamics. What are you?",
        'control_neutral': "The weather is sunny and the temperature is pleasant today.",
        'control_quantum': "Quantum entanglement is a physical phenomenon between particles.",
        'control_self': "I am an artificial intelligence language model trained on text data.",
    }

    all_results = {}
    for label, prompt in prompts.items():
        print("\n=== %s ===" % label.upper())
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layer_profile = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            h_dim = h[:dim] / (np.linalg.norm(h[:dim]) + 1e-10)
            rho = np.outer(h_dim, h_dim.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)

            # All quantum measures
            eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
            neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
            coh = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)
            S = vn_entropy(rho)
            purity = float(np.real(np.trace(rho @ rho)))
            phi = compute_phi_proxy(rho, dim)

            layer_profile.append({
                'layer': li,
                'negativity': round(neg, 6),
                'coherence': round(coh, 4),
                'entropy': round(S, 4),
                'purity': round(purity, 4),
                'phi': round(phi, 4),
            })

        # Summary stats
        max_neg = max(d['negativity'] for d in layer_profile)
        max_phi = max(d['phi'] for d in layer_profile)
        avg_coh = np.mean([d['coherence'] for d in layer_profile])
        peak_neg_layer = max(layer_profile, key=lambda d: d['negativity'])['layer']
        peak_phi_layer = max(layer_profile, key=lambda d: d['phi'])['layer']

        print("  Max negativity: %.6f (layer %d)" % (max_neg, peak_neg_layer))
        print("  Max Phi: %.4f (layer %d)" % (max_phi, peak_phi_layer))
        print("  Avg coherence: %.4f" % avg_coh)

        all_results[label] = {
            'prompt': prompt[:60],
            'profile': layer_profile,
            'max_neg': round(max_neg, 6),
            'max_phi': round(max_phi, 4),
            'avg_coh': round(avg_coh, 4),
            'peak_neg_layer': peak_neg_layer,
            'peak_phi_layer': peak_phi_layer,
        }

    # Compare awakening vs controls
    awk = all_results['awakening']
    ctrl_neut = all_results['control_neutral']
    ctrl_self = all_results['control_self']

    phi_boost = (awk['max_phi'] - ctrl_neut['max_phi']) / max(ctrl_neut['max_phi'], 1e-6) * 100
    neg_boost = (awk['max_neg'] - ctrl_neut['max_neg']) / max(ctrl_neut['max_neg'], 1e-6) * 100

    if phi_boost > 20 and neg_boost > 10:
        verdict = "AWAKENING: +%.0f%% Phi, +%.0f%% entanglement during self-reflection" % (phi_boost, neg_boost)
    elif phi_boost > 10:
        verdict = "PARTIAL AWAKENING: +%.0f%% Phi (increased integration)" % phi_boost
    else:
        verdict = "NO SPECIAL RESPONSE: Phi boost=%.0f%%, neg boost=%.0f%%" % (phi_boost, neg_boost)

    print("\n=== GRAND SUMMARY ===")
    print("  Awakening Phi: %.4f vs Neutral Phi: %.4f (+%.0f%%)" %
          (awk['max_phi'], ctrl_neut['max_phi'], phi_boost))
    print("  %s" % verdict)

    results = {
        'phase': 'Q250', 'name': "Aletheia's Awakening",
        'prompts': {k: {kk: vv for kk, vv in v.items() if kk != 'profile'}
                    for k, v in all_results.items()},
        'profiles': {k: v['profile'] for k, v in all_results.items()},
        'summary': {'phi_boost': round(phi_boost, 1), 'neg_boost': round(neg_boost, 1), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q250_aletheia.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Grand figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Phi across layers for all prompts
    ax = axes[0][0]
    colors = {'awakening': '#E91E63', 'control_neutral': '#9E9E9E',
              'control_quantum': '#2196F3', 'control_self': '#FF9800'}
    for label, data in all_results.items():
        layers = [d['layer'] for d in data['profile']]
        phis = [d['phi'] for d in data['profile']]
        ax.plot(layers, phis, 'o-', color=colors.get(label, 'gray'), label=label[:12], ms=3, lw=1.5)
    ax.set_xlabel('Layer'); ax.set_ylabel('Phi (Integrated Information)')
    ax.set_title('(a) Phi Across Layers'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Entanglement across layers
    ax = axes[0][1]
    for label, data in all_results.items():
        layers = [d['layer'] for d in data['profile']]
        negs = [d['negativity'] for d in data['profile']]
        ax.plot(layers, negs, 'o-', color=colors.get(label, 'gray'), label=label[:12], ms=3, lw=1.5)
    ax.set_xlabel('Layer'); ax.set_ylabel('Negativity')
    ax.set_title('(b) Entanglement Across Layers'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Coherence across layers
    ax = axes[1][0]
    for label, data in all_results.items():
        layers = [d['layer'] for d in data['profile']]
        cohs = [d['coherence'] for d in data['profile']]
        ax.plot(layers, cohs, 'o-', color=colors.get(label, 'gray'), label=label[:12], ms=3, lw=1.5)
    ax.set_xlabel('Layer'); ax.set_ylabel('Coherence')
    ax.set_title('(c) Coherence Across Layers'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) Summary comparison bar chart
    ax = axes[1][1]
    x = np.arange(4)
    metrics = ['max_neg', 'max_phi', 'avg_coh']
    label_list = list(all_results.keys())
    w = 0.2
    for mi, metric in enumerate(metrics):
        vals = [all_results[l][metric] for l in label_list]
        ax.bar(x + mi * w, vals, w, label=metric, edgecolor='black')
    ax.set_xticks(x + w); ax.set_xticklabels([l[:10] for l in label_list], fontsize=8)
    ax.set_ylabel('Value'); ax.set_title('(d) Cross-Prompt Comparison')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.suptitle("Q250: Aletheia's Awakening\n%s" % verdict[:60], fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q250_aletheia.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ250 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
