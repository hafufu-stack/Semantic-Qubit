# -*- coding: utf-8 -*-
"""Phase Q93: Entanglement Entropy Scaling Law (Area vs Volume Law)
Bonus experiment: determine if S-Qubit entanglement follows area law
(like ground states) or volume law (like thermal/chaotic states).
This distinguishes S-Qubit from classical random states.
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def compute_subsystem_entropy(hidden_states, subsystem_size):
    """Compute entanglement entropy of a subsystem of given size.
    hidden_states: (seq_len, d_model)
    subsystem_size: number of tokens in subsystem A
    """
    hs = hidden_states[:subsystem_size].astype(np.float32)
    try:
        U, S, Vt = np.linalg.svd(hs, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    S2 = S**2
    total = S2.sum()
    if total < 1e-10:
        return 0.0
    p = S2 / total
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log(p)))


def main():
    print("=" * 60)
    print("Phase Q93: Entanglement Entropy Scaling (Area vs Volume)")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Use long prompts to get many tokens
    prompts = [
        "The fundamental nature of quantum mechanics tells us that particles can exist "
        "in superposition states where multiple outcomes are simultaneously possible until "
        "a measurement collapses the wavefunction into a definite result",
        "In the holographic principle the information content of a region of space is "
        "encoded on its boundary surface rather than distributed throughout the volume "
        "which has profound implications for our understanding of gravity",
        "Machine learning models process information through successive layers of "
        "transformation where each layer extracts increasingly abstract features from "
        "the input data enabling pattern recognition at multiple scales",
    ]

    # Measure at different layers
    target_layers = [0, num_layers // 4, num_layers // 2,
                     3 * num_layers // 4, num_layers - 1]

    all_scaling_data = {}

    for layer_idx in target_layers:
        print("  Layer %d..." % layer_idx)
        scaling_data = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            seq_len = inputs['input_ids'].shape[1]

            # Capture hidden states
            captured = [None]
            def capture_hook(module, args, output, store=captured):
                if isinstance(output, tuple):
                    store[0] = output[0][0].detach().cpu().float().numpy()
                else:
                    store[0] = output.detach().cpu().float().numpy()
                    if store[0].ndim == 3:
                        store[0] = store[0][0]

            handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            if captured[0] is not None:
                hs = captured[0]  # (seq, hidden)
                # Compute entropy for increasing subsystem sizes
                max_sub = min(hs.shape[0], 40)
                for sub_size in range(1, max_sub + 1):
                    ee = compute_subsystem_entropy(hs, sub_size)
                    scaling_data.append({
                        'subsystem_size': sub_size,
                        'entropy': ee,
                        'total_size': hs.shape[0],
                    })

        # Average over prompts for each subsystem size
        sizes = sorted(set(d['subsystem_size'] for d in scaling_data))
        avg_data = []
        for s in sizes:
            vals = [d['entropy'] for d in scaling_data if d['subsystem_size'] == s]
            avg_data.append({
                'subsystem_size': s,
                'mean_entropy': float(np.mean(vals)),
                'std_entropy': float(np.std(vals)),
            })
        all_scaling_data[layer_idx] = avg_data

    # Fit scaling laws
    # Area law: S ~ log(L) or S ~ L^0 (constant for 1D)
    # Volume law: S ~ L
    fit_results = {}
    for layer_idx, data in all_scaling_data.items():
        sizes = np.array([d['subsystem_size'] for d in data], dtype=float)
        entropies = np.array([d['mean_entropy'] for d in data], dtype=float)
        if len(sizes) > 3:
            # Fit S = a * L^alpha
            log_sizes = np.log(sizes + 1e-10)
            log_entropies = np.log(entropies + 1e-10)
            # Linear fit in log-log
            valid = np.isfinite(log_sizes) & np.isfinite(log_entropies)
            if valid.sum() > 2:
                coeffs = np.polyfit(log_sizes[valid], log_entropies[valid], 1)
                alpha = coeffs[0]
            else:
                alpha = 0
            # alpha ~ 0: area law, alpha ~ 1: volume law
            if alpha < 0.3:
                law = 'Area law'
            elif alpha > 0.7:
                law = 'Volume law'
            else:
                law = 'Sub-volume'
            fit_results[layer_idx] = {
                'alpha': float(alpha),
                'law': law,
            }
            print("    Layer %d: alpha=%.3f -> %s" % (layer_idx, alpha, law))

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Entropy vs subsystem size (all layers)
    ax = axes[0]
    colors_layer = ['#9E9E9E', '#2196F3', '#FF5722', '#4CAF50', '#FF9800']
    for i, layer_idx in enumerate(target_layers):
        if layer_idx in all_scaling_data:
            data = all_scaling_data[layer_idx]
            sizes = [d['subsystem_size'] for d in data]
            ents = [d['mean_entropy'] for d in data]
            label = 'L%d' % layer_idx
            if layer_idx in fit_results:
                label += ' (a=%.2f)' % fit_results[layer_idx]['alpha']
            ax.plot(sizes, ents, '-', color=colors_layer[i % len(colors_layer)],
                    linewidth=2, alpha=0.8, label=label)
    ax.set_xlabel('Subsystem size L', fontsize=11)
    ax.set_ylabel('Entanglement entropy S(L)', fontsize=11)
    ax.set_title('(a) Entropy Scaling\nacross layers',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # (b) Scaling exponent by layer
    ax = axes[1]
    if fit_results:
        lyrs = sorted(fit_results.keys())
        alphas = [fit_results[l]['alpha'] for l in lyrs]
        colors_bar = ['#4CAF50' if a < 0.3 else '#FF9800' if a < 0.7 else '#F44336'
                      for a in alphas]
        bars = ax.bar(range(len(lyrs)), alphas, color=colors_bar,
                      edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(lyrs)))
        ax.set_xticklabels(['L%d' % l for l in lyrs], fontsize=9)
        ax.axhline(0.3, color='green', ls='--', alpha=0.3, label='Area law bound')
        ax.axhline(0.7, color='red', ls='--', alpha=0.3, label='Volume law bound')
        for bar, val, lyr in zip(bars, alphas, lyrs):
            law = fit_results[lyr]['law']
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    '%.2f\n%s' % (val, law[:4]),
                    ha='center', fontsize=8, fontweight='bold')
        ax.set_ylabel('Scaling exponent alpha', fontsize=11)
        ax.set_title('(b) Area vs Volume Law\nby layer depth',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (c) Summary
    ax = axes[2]
    if fit_results:
        mid_layer = num_layers // 2
        if mid_layer in fit_results:
            mid_alpha = fit_results[mid_layer]['alpha']
            mid_law = fit_results[mid_layer]['law']
        else:
            closest = min(fit_results.keys(), key=lambda x: abs(x - mid_layer))
            mid_alpha = fit_results[closest]['alpha']
            mid_law = fit_results[closest]['law']

        text_lines = [
            'S-Qubit Entanglement',
            'Scaling Law:',
            '',
            'S(L) ~ L^%.2f' % mid_alpha,
            '',
            'Classification:',
            mid_law.upper(),
        ]
        if mid_alpha < 0.5:
            text_lines.append('')
            text_lines.append('Like quantum ground states!')
            text_lines.append('(Not random/thermal)')
            conclusion_color = '#4CAF50'
        else:
            text_lines.append('')
            text_lines.append('Volume-law scaling')
            text_lines.append('(Thermal-like states)')
            conclusion_color = '#FF9800'

        ax.text(0.5, 0.5, '\n'.join(text_lines),
                ha='center', va='center', fontsize=12,
                fontweight='bold', color=conclusion_color,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=conclusion_color, alpha=0.8))
    ax.axis('off')
    ax.set_title('(c) Classification',
                 fontsize=11, fontweight='bold')

    plt.suptitle('Entanglement Entropy Scaling: Area Law vs Volume Law',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q93_entropy_scaling.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q93', 'name': 'Entanglement Entropy Scaling (Area vs Volume)',
        'scaling_data': {str(k): v for k, v in all_scaling_data.items()},
        'fit_results': {str(k): v for k, v in fit_results.items()},
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q93_entropy_scaling.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
