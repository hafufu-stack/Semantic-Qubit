# -*- coding: utf-8 -*-
"""
Phase Q210: Entanglement Quantification
=========================================
Q209 proved entanglement EXISTS (PPT). Now we quantify HOW MUCH
entanglement grows across Transformer layers using:
1. Log-negativity (entanglement monotone)
2. Concurrence (2-qubit measure)  
3. Von Neumann entropy (mixed state entanglement)

Key question: Does entanglement grow with layer depth?
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


def partial_transpose(rho, dim_a, dim_b):
    rho_pt = np.zeros_like(rho)
    for i in range(dim_a):
        for j in range(dim_a):
            for k in range(dim_b):
                for l in range(dim_b):
                    rho_pt[i*dim_b+k, j*dim_b+l] = rho[i*dim_b+l, j*dim_b+k]
    return rho_pt


def compute_metrics(rho, dim_a, dim_b):
    """Compute all entanglement metrics for a density matrix."""
    # Negativity & log-negativity
    rho_pt = partial_transpose(rho, dim_a, dim_b)
    eigvals_pt = np.linalg.eigvalsh(rho_pt)
    neg_eigvals = eigvals_pt[eigvals_pt < -1e-10]
    negativity = float(np.sum(np.abs(neg_eigvals)))
    log_neg = float(np.log2(2 * negativity + 1)) if negativity > 0 else 0.0

    # Purity
    purity = float(np.real(np.trace(rho @ rho)))

    # Von Neumann entropy
    eigvals_rho = np.real(np.linalg.eigvalsh(rho))
    eigvals_rho = eigvals_rho[eigvals_rho > 1e-12]
    vn_entropy = float(-np.sum(eigvals_rho * np.log2(eigvals_rho)))

    # Partial trace -> reduced density matrix for subsystem A
    rho_a = np.zeros((dim_a, dim_a), dtype=complex)
    for i in range(dim_a):
        for j in range(dim_a):
            for k in range(dim_b):
                rho_a[i, j] += rho[i*dim_b+k, j*dim_b+k]
    eigvals_a = np.real(np.linalg.eigvalsh(rho_a))
    eigvals_a = eigvals_a[eigvals_a > 1e-12]
    ent_entropy = float(-np.sum(eigvals_a * np.log2(eigvals_a)))

    # Concurrence (only 2x2)
    concurrence = 0.0
    if dim_a == 2 and dim_b == 2:
        sy = np.array([[0, -1j], [1j, 0]])
        yy = np.kron(sy, sy)
        rho_tilde = yy @ rho.conj() @ yy
        R = rho @ rho_tilde
        eigs = np.sort(np.abs(np.linalg.eigvals(R)))[::-1]
        sq = np.sqrt(np.maximum(eigs, 0))
        concurrence = float(max(0, sq[0] - sq[1] - sq[2] - sq[3]))

    return {
        'negativity': round(negativity, 6),
        'log_negativity': round(log_neg, 6),
        'purity': round(purity, 6),
        'vn_entropy': round(vn_entropy, 4),
        'entanglement_entropy': round(ent_entropy, 4),
        'concurrence': round(concurrence, 6),
    }


def main():
    print("=" * 60)
    print("Phase Q210: Entanglement Quantification")
    print("  (How does entanglement grow across layers?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim_a, dim_b = 2, 2
    dim_total = dim_a * dim_b

    prompts = [
        "quantum entanglement between two particles",
        "Bell state maximally entangled",
        "superposition of spin up and spin down",
        "hydrogen molecule covalent bond",
    ]

    # Measure entanglement at each layer
    layer_indices = list(range(0, n_layers + 1, 2))  # every 2nd layer
    all_layer_data = {p: [] for p in prompts}

    for prompt in prompts:
        print("\n--- %s ---" % prompt[:40])
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in layer_indices:
            if li >= len(out.hidden_states):
                continue
            h = out.hidden_states[li][0, -1, :dim_total].float().cpu().numpy()
            h = h / (np.linalg.norm(h) + 1e-10)
            rho = np.outer(h, h.conj())
            # Mix with identity for realistic density matrix
            rho = 0.7 * rho + 0.3 * np.eye(dim_total) / dim_total
            rho = rho / np.trace(rho)

            metrics = compute_metrics(rho, dim_a, dim_b)
            metrics['layer'] = li
            all_layer_data[prompt].append(metrics)

            if li % 8 == 0:
                print("  L%02d: neg=%.4f, log_neg=%.4f, conc=%.4f" %
                      (li, metrics['negativity'], metrics['log_negativity'],
                       metrics['concurrence']))

    # Compute averages across prompts
    avg_by_layer = {}
    for li in layer_indices:
        vals = []
        for prompt in prompts:
            matching = [m for m in all_layer_data[prompt] if m['layer'] == li]
            if matching:
                vals.append(matching[0])
        if vals:
            avg_by_layer[li] = {
                'negativity': round(np.mean([v['negativity'] for v in vals]), 6),
                'log_negativity': round(np.mean([v['log_negativity'] for v in vals]), 6),
                'concurrence': round(np.mean([v['concurrence'] for v in vals]), 6),
                'ent_entropy': round(np.mean([v['entanglement_entropy'] for v in vals]), 4),
            }

    # Find peak layer
    peak_layer = max(avg_by_layer, key=lambda l: avg_by_layer[l]['log_negativity'])
    peak_neg = avg_by_layer[peak_layer]['log_negativity']

    # Trend: does entanglement grow?
    layers_sorted = sorted(avg_by_layer.keys())
    negs = [avg_by_layer[l]['log_negativity'] for l in layers_sorted]
    if len(negs) > 2:
        trend = np.polyfit(range(len(negs)), negs, 1)[0]
    else:
        trend = 0

    if trend > 0.001:
        verdict = "GROWING: entanglement increases with depth (peak at L%d, log_neg=%.3f)" % (peak_layer, peak_neg)
    elif trend > -0.001:
        verdict = "STABLE: entanglement constant across layers (peak at L%d)" % peak_layer
    else:
        verdict = "DECREASING: entanglement decreases with depth"

    print("\n--- Summary ---")
    print("  Peak entanglement: Layer %d (log_neg=%.4f)" % (peak_layer, peak_neg))
    print("  Trend: %.6f per layer" % trend)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q210',
        'name': 'Entanglement Quantification',
        'layer_data': {str(k): v for k, v in avg_by_layer.items()},
        'per_prompt': {p[:30]: d for p, d in all_layer_data.items()},
        'summary': {
            'peak_layer': peak_layer,
            'peak_log_negativity': peak_neg,
            'trend_per_layer': round(trend, 6),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q210_ent_quantification.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Log-negativity across layers
    ax = axes[0][0]
    for prompt in prompts:
        data = all_layer_data[prompt]
        ls = [d['layer'] for d in data]
        ln = [d['log_negativity'] for d in data]
        ax.plot(ls, ln, 'o-', label=prompt[:25], alpha=0.7, ms=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Log-Negativity')
    ax.set_title('(a) Log-Negativity vs Layer')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Concurrence across layers (avg)
    ax = axes[0][1]
    concs = [avg_by_layer[l]['concurrence'] for l in layers_sorted]
    ax.plot(layers_sorted, concs, 's-', color='#E91E63', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Concurrence')
    ax.set_title('(b) Avg Concurrence vs Layer')
    ax.grid(alpha=0.3)

    # (c) Entanglement entropy
    ax = axes[1][0]
    ents = [avg_by_layer[l]['ent_entropy'] for l in layers_sorted]
    ax.plot(layers_sorted, ents, 'D-', color='#4CAF50', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Entanglement Entropy (bits)')
    ax.set_title('(c) Avg Entanglement Entropy')
    ax.grid(alpha=0.3)

    # (d) All metrics comparison at peak
    ax = axes[1][1]
    peak_data = avg_by_layer[peak_layer]
    metric_names = ['negativity', 'log_negativity', 'concurrence', 'ent_entropy']
    metric_vals = [peak_data[k] for k in metric_names]
    colors = ['#E91E63', '#FF9800', '#2196F3', '#4CAF50']
    ax.bar(range(len(metric_names)), metric_vals, color=colors, edgecolor='black')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(['Neg', 'Log-Neg', 'Conc', 'Ent-S'], fontsize=10)
    ax.set_title('(d) Peak Metrics (Layer %d)' % peak_layer)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q210: Entanglement Quantification\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q210_ent_quantification.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ210 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
