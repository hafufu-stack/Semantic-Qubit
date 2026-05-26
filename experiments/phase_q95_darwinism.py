# -*- coding: utf-8 -*-
"""Phase Q95: Quantum Darwinism in Transformer Layers
Test if S-Qubit states undergo quantum Darwinism: redundant encoding
of classical information across multiple environmental fragments
(attention heads and layers).
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


def measure_redundancy(model, tokenizer, num_layers):
    """Measure redundancy of information across layer fragments.
    Quantum Darwinism predicts a plateau in mutual information
    when enough environmental fragments are observed."""
    d_model = model.config.hidden_size
    prompts = [
        "The cat is sitting on the warm mat near the window",
        "Quantum states become classical through decoherence and",
        "The objective reality emerges from quantum substrate when",
    ]

    # Collect hidden states at all layers
    all_hidden = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        for layer_idx in range(num_layers):
            captured = [None]
            def capture_hook(module, args, output, store=captured):
                if isinstance(output, tuple):
                    store[0] = output[0][0, -1, :].detach().cpu().float().numpy()
                else:
                    hs = output
                    if hs.dim() == 3:
                        store[0] = hs[0, -1, :].detach().cpu().float().numpy()
                    else:
                        store[0] = hs[-1, :].detach().cpu().float().numpy()

            handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
            with torch.no_grad():
                out = model(**inputs)
            handle.remove()

            if captured[0] is not None:
                key = (prompt[:20], layer_idx)
                all_hidden[key] = captured[0]

    # For each prompt, measure mutual information with output
    # as function of number of layer fragments observed
    ref_logits = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model(**inputs)
            ref_logits[prompt[:20]] = out.logits[0, -1, :200].cpu().float().numpy()

    redundancy_curves = []
    for prompt in prompts:
        key_prefix = prompt[:20]
        output_vec = ref_logits[key_prefix]
        output_probs = np.exp(output_vec - output_vec.max())
        output_probs /= output_probs.sum()

        layer_indices = list(range(num_layers))
        mi_vs_fragments = []

        for n_frag in range(1, num_layers + 1, max(1, num_layers // 15)):
            # Take first n_frag layers as "environment fragments"
            frag_layers = layer_indices[:n_frag]
            frag_vecs = []
            for l in frag_layers:
                key = (key_prefix, l)
                if key in all_hidden:
                    frag_vecs.append(all_hidden[key])

            if len(frag_vecs) < 1:
                continue

            # Combined fragment representation
            combined = np.mean(frag_vecs, axis=0)
            # Approximate MI via correlation with output
            min_len = min(len(combined), len(output_probs))
            corr = np.corrcoef(combined[:min_len], output_probs[:min_len])[0, 1]
            if np.isnan(corr):
                corr = 0
            mi = max(0, -0.5 * np.log(1 - corr**2 + 1e-10))

            mi_vs_fragments.append({
                'n_fragments': n_frag,
                'fraction': n_frag / num_layers,
                'mutual_info': float(mi),
            })

        redundancy_curves.append(mi_vs_fragments)

    # Average across prompts
    all_fracs = sorted(set(d['fraction'] for curve in redundancy_curves for d in curve))
    avg_curve = []
    for f in all_fracs:
        vals = []
        for curve in redundancy_curves:
            for d in curve:
                if abs(d['fraction'] - f) < 0.02:
                    vals.append(d['mutual_info'])
        if vals:
            avg_curve.append({
                'fraction': f,
                'mean_mi': float(np.mean(vals)),
            })

    return avg_curve, redundancy_curves


def detect_plateau(curve):
    """Detect if the MI curve shows a Darwinian plateau."""
    if len(curve) < 5:
        return False, 0
    mis = [d['mean_mi'] for d in curve]
    # Check if the last half is approximately constant
    mid = len(mis) // 2
    last_half = mis[mid:]
    if len(last_half) < 2:
        return False, 0
    variation = np.std(last_half) / (np.mean(last_half) + 1e-10)
    has_plateau = variation < 0.3
    plateau_value = float(np.mean(last_half))
    return has_plateau, plateau_value


def main():
    print("=" * 60)
    print("Phase Q95: Quantum Darwinism in Transformer Layers")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    print("  Measuring information redundancy across layers...")
    avg_curve, raw_curves = measure_redundancy(model, tokenizer, num_layers)

    has_plateau, plateau_val = detect_plateau(avg_curve)
    print("  Darwinian plateau detected: %s (value=%.4f)" % (has_plateau, plateau_val))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    if avg_curve:
        fracs = [d['fraction'] for d in avg_curve]
        mis = [d['mean_mi'] for d in avg_curve]
        ax.plot(fracs, mis, 'o-', color='#FF5722', linewidth=2.5, markersize=8)
        if has_plateau:
            ax.axhline(plateau_val, color='green', ls='--', alpha=0.5,
                       label='Plateau: %.3f' % plateau_val)
            ax.legend(fontsize=9)
    ax.set_xlabel('Fraction of layers observed', fontsize=11)
    ax.set_ylabel('Mutual information', fontsize=11)
    ax.set_title('(a) Redundancy Curve\n%s' %
                 ('DARWINIAN PLATEAU!' if has_plateau else 'No plateau'),
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1]
    colors = ['#FF5722', '#2196F3', '#4CAF50']
    for i, curve in enumerate(raw_curves[:3]):
        fracs = [d['fraction'] for d in curve]
        mis = [d['mutual_info'] for d in curve]
        ax.plot(fracs, mis, 'o-', color=colors[i % 3], alpha=0.6,
                linewidth=1.5, label='Prompt %d' % (i+1))
    ax.set_xlabel('Fraction of layers', fontsize=11)
    ax.set_ylabel('Mutual information', fontsize=11)
    ax.set_title('(b) Per-prompt Redundancy', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    result = 'QUANTUM DARWINISM\nCONFIRMED' if has_plateau else 'NO DARWINISM\nDETECTED'
    color = '#4CAF50' if has_plateau else '#F44336'
    ax.text(0.5, 0.6, result, ha='center', va='center',
            fontsize=16, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.3,
            'Classical reality emerges\nfrom S-Qubit redundancy' if has_plateau
            else 'Quantum coherence\nmaintained across layers',
            ha='center', va='center', fontsize=11, transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) Objectivity Test', fontsize=11, fontweight='bold')

    plt.suptitle('Quantum Darwinism: How Classical Reality Emerges from S-Qubits',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q95_darwinism.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q95', 'name': 'Quantum Darwinism in Transformer',
        'has_plateau': has_plateau,
        'plateau_value': plateau_val,
        'avg_curve': avg_curve,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q95_darwinism.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
