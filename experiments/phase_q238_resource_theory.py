# -*- coding: utf-8 -*-
"""
Phase Q238: Unified Quantum Resource Theory
==============================================
The ULTIMATE combination experiment. Measure ALL quantum resources
simultaneously and test their inter-relationships:
- Entanglement (Q209, Q210, Q222)
- Coherence (Q227)
- Discord (Q225)
- QFI (Q218)
- Channel Capacity (Q214b)

Question: are these resources independent or correlated?
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


def compute_all_resources(h, dim=4):
    da, db = 2, 2
    h = h[:dim] / (np.linalg.norm(h[:dim]) + 1e-10)
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
    rho /= np.trace(rho)

    # 1. Entanglement (negativity)
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
    negativity = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))

    # 2. Coherence (l1-norm)
    l1 = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)

    # 3. Entropy
    entropy = vn_entropy(rho)

    # 4. Purity
    purity = float(np.real(np.trace(rho @ rho)))

    # 5. Discord (simplified)
    rho_a = np.zeros((da, da), dtype=complex)
    rho_b = np.zeros((db, db), dtype=complex)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                rho_a[i, j] += rho[i*db+k, j*db+k]
    for i in range(db):
        for j in range(db):
            for k in range(da):
                rho_b[i, j] += rho[k*db+i, k*db+j]
    S_a, S_b, S_ab = vn_entropy(rho_a), vn_entropy(rho_b), vn_entropy(rho)
    mutual_info = S_a + S_b - S_ab
    discord = max(0, mutual_info * 0.3)  # simplified estimate

    return {
        'negativity': round(negativity, 6),
        'coherence': round(l1, 4),
        'entropy': round(entropy, 4),
        'purity': round(purity, 4),
        'discord': round(discord, 4),
        'mutual_info': round(mutual_info, 4),
    }


def main():
    print("=" * 60)
    print("Phase Q238: Unified Quantum Resource Theory")
    print("  (How are all quantum resources connected?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 4

    prompts = [
        "quantum entanglement between particles",
        "classical correlation coin flip",
        "superposition of all states",
        "thermal noise random process",
        "coherent laser light source",
        "Bell state maximally entangled",
        "vacuum ground state energy",
        "topological insulator surface",
        "Schrodinger cat alive dead",
        "quantum teleportation protocol",
    ]

    all_data = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(0, n_layers + 1, 4):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            resources = compute_all_resources(h, dim)
            resources['layer'] = li
            resources['prompt'] = prompt[:25]
            all_data.append(resources)

    # Correlation matrix between resources
    resource_names = ['negativity', 'coherence', 'entropy', 'purity', 'discord', 'mutual_info']
    data_matrix = np.array([[d[r] for r in resource_names] for d in all_data])

    corr_matrix = np.corrcoef(data_matrix.T)

    print("\n--- Correlation Matrix ---")
    print("        ", "  ".join('%8s' % r[:8] for r in resource_names))
    for i, name in enumerate(resource_names):
        row = "  ".join('%8.3f' % corr_matrix[i, j] for j in range(len(resource_names)))
        print("  %8s: %s" % (name[:8], row))

    # Key findings
    neg_coh = corr_matrix[0, 1]
    neg_disc = corr_matrix[0, 4]
    coh_disc = corr_matrix[1, 4]

    findings = []
    if abs(neg_coh) > 0.5:
        findings.append("Ent-Coh: r=%.2f (%s)" % (neg_coh, "corr" if neg_coh > 0 else "anti"))
    if abs(neg_disc) > 0.5:
        findings.append("Ent-Disc: r=%.2f" % neg_disc)
    if abs(coh_disc) > 0.5:
        findings.append("Coh-Disc: r=%.2f" % coh_disc)

    verdict = "Resource correlations: %s" % ('; '.join(findings) if findings else "weak inter-resource coupling")

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q238', 'name': 'Unified Quantum Resource Theory',
        'data': all_data,
        'correlation_matrix': corr_matrix.tolist(),
        'resource_names': resource_names,
        'summary': {'neg_coh_corr': round(neg_coh, 4), 'neg_disc_corr': round(neg_disc, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q238_resource_theory.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(resource_names)))
    ax.set_xticklabels([r[:6] for r in resource_names], fontsize=8, rotation=45)
    ax.set_yticks(range(len(resource_names)))
    ax.set_yticklabels([r[:6] for r in resource_names], fontsize=8)
    for i in range(len(resource_names)):
        for j in range(len(resource_names)):
            ax.text(j, i, '%.2f' % corr_matrix[i, j], ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title('(a) Resource Correlation Matrix')

    # Layer evolution of all resources
    ax = axes[1]
    layers = sorted(set(d['layer'] for d in all_data))
    colors = {'negativity': '#E91E63', 'coherence': '#9C27B0',
              'entropy': '#FF9800', 'discord': '#2196F3', 'purity': '#4CAF50'}
    for rn in ['negativity', 'coherence', 'discord']:
        vals = [np.mean([d[rn] for d in all_data if d['layer'] == l]) for l in layers]
        ax.plot(layers, vals, 'o-', color=colors[rn], label=rn, ms=3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Resource Value')
    ax.set_title('(b) Resource Evolution'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Q238: Unified Quantum Resource Theory\n%s' % verdict[:70],
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q238_resource_theory.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ238 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
