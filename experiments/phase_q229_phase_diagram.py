# -*- coding: utf-8 -*-
"""
Phase Q229: Layer-wise Quantum Phase Diagram
===============================================
Combine Q210 (entanglement), Q218 (QFI), Q224 (entropy), Q227 (coherence)
into a single phase diagram. Which quantum phase is each layer in?

Phases: Topological | Critical | Thermal | Classical
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


def compute_all_metrics(h, dim=8):
    """Compute entanglement, coherence, entropy for a hidden state."""
    h = h[:dim]
    h /= np.linalg.norm(h) + 1e-10
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
    rho /= np.trace(rho)

    # Entanglement (4x2 bipartition)
    da, db = 4, 2
    if dim == 8:
        eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
        negativity = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
    else:
        negativity = 0

    # Coherence
    l1_coherence = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))
    norm_coherence = l1_coherence / (dim - 1)

    # Entropy
    eigvals_rho = np.real(np.linalg.eigvalsh(rho))
    eigvals_rho = eigvals_rho[eigvals_rho > 1e-12]
    entropy = float(-np.sum(eigvals_rho * np.log2(eigvals_rho)))

    # Purity
    purity = float(np.real(np.trace(rho @ rho)))

    return {
        'negativity': round(negativity, 6),
        'coherence': round(norm_coherence, 4),
        'entropy': round(entropy, 4),
        'purity': round(purity, 4),
    }


def classify_phase(metrics):
    """Classify quantum phase based on metrics."""
    neg = metrics['negativity']
    coh = metrics['coherence']
    ent = metrics['entropy']
    pur = metrics['purity']

    if neg > 0.1 and coh > 0.3 and pur > 0.3:
        return "Topological"
    elif neg > 0.05 and coh > 0.2:
        return "Critical"
    elif ent > 2.5 and pur < 0.2:
        return "Thermal"
    elif coh < 0.1:
        return "Classical"
    else:
        return "Quantum"


def main():
    print("=" * 60)
    print("Phase Q229: Layer-wise Quantum Phase Diagram")
    print("  (Map the quantum phases of the Transformer)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 8

    prompts = [
        "quantum entanglement between particles",
        "ground state of hydrogen molecule",
        "thermal noise at room temperature",
        "topological insulator surface states",
        "many-body quantum phase transition",
        "superconducting qubit coherence",
    ]

    layer_phases = []

    for li in range(n_layers + 1):
        metrics_list = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            m = compute_all_metrics(h, dim)
            metrics_list.append(m)

        avg_metrics = {
            'negativity': round(np.mean([m['negativity'] for m in metrics_list]), 6),
            'coherence': round(np.mean([m['coherence'] for m in metrics_list]), 4),
            'entropy': round(np.mean([m['entropy'] for m in metrics_list]), 4),
            'purity': round(np.mean([m['purity'] for m in metrics_list]), 4),
        }
        phase = classify_phase(avg_metrics)
        avg_metrics['layer'] = li
        avg_metrics['phase'] = phase
        layer_phases.append(avg_metrics)

        if li % 4 == 0:
            print("  L%02d: neg=%.4f, coh=%.3f, S=%.3f, pur=%.3f -> %s" %
                  (li, avg_metrics['negativity'], avg_metrics['coherence'],
                   avg_metrics['entropy'], avg_metrics['purity'], phase))

    # Count phases
    phase_counts = {}
    for lp in layer_phases:
        p = lp['phase']
        phase_counts[p] = phase_counts.get(p, 0) + 1

    dominant = max(phase_counts, key=phase_counts.get)
    verdict = "Dominant phase: %s (%d/%d layers). Phases: %s" % (
        dominant, phase_counts[dominant], len(layer_phases),
        ', '.join('%s:%d' % (k, v) for k, v in sorted(phase_counts.items(), key=lambda x: -x[1])))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q229',
        'name': 'Quantum Phase Diagram',
        'layers': layer_phases,
        'summary': {
            'phase_counts': phase_counts,
            'dominant_phase': dominant,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q229_phase_diagram.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot: multi-panel phase diagram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = [lp['layer'] for lp in layer_phases]
    phase_colors = {'Topological': '#9C27B0', 'Critical': '#FF9800',
                    'Thermal': '#F44336', 'Classical': '#607D8B', 'Quantum': '#2196F3'}

    # (a) Entanglement + Coherence
    ax = axes[0][0]
    negs = [lp['negativity'] for lp in layer_phases]
    cohs = [lp['coherence'] for lp in layer_phases]
    ax.plot(layers, negs, 'o-', color='#E91E63', label='Negativity', ms=3)
    ax.plot(layers, cohs, 's-', color='#9C27B0', label='Coherence', ms=3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Value')
    ax.set_title('(a) Entanglement & Coherence'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Entropy + Purity
    ax = axes[0][1]
    ents = [lp['entropy'] for lp in layer_phases]
    purs = [lp['purity'] for lp in layer_phases]
    ax.plot(layers, ents, 'D-', color='#FF9800', label='Entropy', ms=3)
    ax2 = ax.twinx()
    ax2.plot(layers, purs, '^-', color='#4CAF50', label='Purity', ms=3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Entropy', color='#FF9800')
    ax2.set_ylabel('Purity', color='#4CAF50')
    ax.set_title('(b) Entropy & Purity'); ax.grid(alpha=0.3)

    # (c) Phase classification
    ax = axes[1][0]
    phase_list = [lp['phase'] for lp in layer_phases]
    unique_phases = list(phase_colors.keys())
    phase_nums = [unique_phases.index(p) if p in unique_phases else -1 for p in phase_list]
    colors = [phase_colors.get(p, '#000000') for p in phase_list]
    ax.bar(layers, [1]*len(layers), color=colors, edgecolor='none')
    ax.set_xlabel('Layer'); ax.set_yticks([])
    ax.set_title('(c) Phase Map')
    for p, c in phase_colors.items():
        if p in phase_list:
            ax.bar([], [], color=c, label=p)
    ax.legend(fontsize=8, ncol=3)

    # (d) Phase counts pie
    ax = axes[1][1]
    pie_colors = [phase_colors.get(p, '#000') for p in phase_counts.keys()]
    ax.pie(phase_counts.values(), labels=phase_counts.keys(),
           colors=pie_colors, autopct='%1.0f%%', startangle=90)
    ax.set_title('(d) Phase Distribution')

    plt.suptitle('Q229: Quantum Phase Diagram\n%s' % verdict[:70],
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q229_phase_diagram.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ229 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
