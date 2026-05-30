# -*- coding: utf-8 -*-
"""
Phase Q239: Semantic-Dependent Quantum Properties
====================================================
THE KEY QUESTION: Do quantum properties change with prompt MEANING?
If "entanglement" prompts produce more entanglement than "classical"
prompts, the LLM has semantic-quantum coupling.
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

def measure_quantum(h, dim=4):
    da, db = 2, 2
    h = h[:dim] / (np.linalg.norm(h[:dim]) + 1e-10)
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
    rho /= np.trace(rho)
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
    neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
    l1 = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)
    return neg, l1

def main():
    print("=" * 60)
    print("Phase Q239: Semantic-Dependent Quantum Properties")
    print("  (Does meaning affect quantumness?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    categories = {
        'quantum': [
            "quantum entanglement between particles", "Bell state preparation",
            "superposition of quantum states", "quantum teleportation protocol",
            "Schrodinger cat alive and dead", "quantum tunneling through barrier",
        ],
        'classical': [
            "classical mechanics Newton laws", "thermodynamic heat engine",
            "electromagnetic wave propagation", "fluid dynamics turbulence",
            "gravitational force between masses", "statistical mechanics ensemble",
        ],
        'neutral': [
            "the weather is sunny today", "cooking recipe for pasta",
            "history of ancient Rome", "mathematical proof by induction",
            "programming in Python language", "music theory chord progression",
        ],
    }

    all_results = {}
    for cat, prompts in categories.items():
        print("\n=== %s ===" % cat.upper())
        cat_negs, cat_cohs = [], []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            # Measure at deep layer
            h = out.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
            neg, coh = measure_quantum(h)
            cat_negs.append(neg)
            cat_cohs.append(coh)
            print("  %s: neg=%.4f, coh=%.4f" % (prompt[:30], neg, coh))
        all_results[cat] = {
            'prompts': [p[:30] for p in prompts],
            'negativities': [round(n, 6) for n in cat_negs],
            'coherences': [round(c, 4) for c in cat_cohs],
            'avg_neg': round(np.mean(cat_negs), 6),
            'avg_coh': round(np.mean(cat_cohs), 4),
        }

    # Statistical test
    from scipy import stats
    q_negs = all_results['quantum']['negativities']
    c_negs = all_results['classical']['negativities']
    n_negs = all_results['neutral']['negativities']
    t_qc, p_qc = stats.ttest_ind(q_negs, c_negs)
    t_qn, p_qn = stats.ttest_ind(q_negs, n_negs)

    if p_qc < 0.05 and np.mean(q_negs) > np.mean(c_negs):
        verdict = "SEMANTIC-QUANTUM COUPLING: quantum prompts have %.1f%% more entanglement (p=%.3f)" % (
            (np.mean(q_negs) - np.mean(c_negs)) / max(np.mean(c_negs), 1e-6) * 100, p_qc)
    elif p_qc < 0.05:
        verdict = "REVERSE COUPLING: classical prompts more entangled (p=%.3f)" % p_qc
    else:
        verdict = "NO SEMANTIC COUPLING: p=%.3f (not significant)" % p_qc

    print("\n--- Summary ---")
    for cat in categories:
        print("  %s: avg_neg=%.4f, avg_coh=%.4f" % (cat, all_results[cat]['avg_neg'], all_results[cat]['avg_coh']))
    print("  t-test quantum vs classical: t=%.2f, p=%.4f" % (t_qc, p_qc))
    print("  %s" % verdict)

    results = {
        'phase': 'Q239', 'name': 'Semantic-Dependent Quantum Properties',
        'categories': all_results,
        'summary': {'t_stat': round(float(t_qc), 4), 'p_value': round(float(p_qc), 4),
                     'p_value_qn': round(float(p_qn), 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q239_semantic_quantum.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cats = list(categories.keys())
    x = np.arange(len(cats))
    ax = axes[0]
    means = [all_results[c]['avg_neg'] for c in cats]
    stds = [np.std(all_results[c]['negativities']) for c in cats]
    colors = ['#E91E63', '#2196F3', '#9E9E9E']
    ax.bar(x, means, yerr=stds, color=colors, edgecolor='black', capsize=5)
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_ylabel('Negativity'); ax.set_title('(a) Entanglement by Category')
    if p_qc < 0.05: ax.text(0.5, max(means)*0.9, '*p<0.05', ha='center', fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    means_c = [all_results[c]['avg_coh'] for c in cats]
    stds_c = [np.std(all_results[c]['coherences']) for c in cats]
    ax.bar(x, means_c, yerr=stds_c, color=colors, edgecolor='black', capsize=5)
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_ylabel('Coherence'); ax.set_title('(b) Coherence by Category')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q239: Semantic-Quantum Coupling\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q239_semantic_quantum.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ239 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
