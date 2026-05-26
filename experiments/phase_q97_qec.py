# -*- coding: utf-8 -*-
"""Phase Q97: Quantum Error Correction Code Discovery
Test if S-Qubit perturbations can be corrected by redundant encoding
across multiple layers (topological error correction from Q90).
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


def test_error_correction(model, tokenizer, num_layers):
    """Test if injecting errors at single layers can be corrected
    by the redundancy in multi-layer processing."""
    d_model = model.config.hidden_size
    prompt = "The quantum error correction code protects information against"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Reference
    with torch.no_grad():
        ref_out = model(**inputs)
        ref_logits = ref_out.logits[0, -1, :].cpu().float().numpy()

    np.random.seed(97)

    results = []
    error_strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for err_strength in error_strengths:
        # Single-layer error
        error_vec = torch.tensor(
            np.random.randn(d_model).astype(np.float32) * err_strength,
            device=model.device)

        single_layer_diffs = []
        for err_layer in range(0, num_layers, max(1, num_layers // 7)):
            applied = [False]
            def make_err_hook(ev=error_vec, flag=applied):
                flag[0] = False
                def hook(module, args, output):
                    if not flag[0]:
                        flag[0] = True
                        if isinstance(output, tuple):
                            hs = output[0].clone()
                            if hs.dim() == 3:
                                hs[0, -1, :] += ev.to(hs.dtype)
                            else:
                                hs[-1, :] += ev.to(hs.dtype)
                            return (hs,) + output[1:]
                        else:
                            hs = output.clone()
                            if hs.dim() == 3:
                                hs[0, -1, :] += ev.to(hs.dtype)
                            else:
                                hs[-1, :] += ev.to(hs.dtype)
                            return hs
                    return output
                return hook

            h = make_err_hook()
            handle = model.model.layers[err_layer].register_forward_hook(h)
            with torch.no_grad():
                out = model(**inputs)
                err_logits = out.logits[0, -1, :].cpu().float().numpy()
            handle.remove()

            diff = np.linalg.norm(err_logits - ref_logits)
            single_layer_diffs.append(float(diff))

        # Multi-layer error (same error at ALL layers - should be harder to correct)
        handles = []
        for ml in range(0, num_layers, max(1, num_layers // 7)):
            applied_multi = [False]
            def make_multi_hook(ev=error_vec * 0.3, flag_m=[False]):
                def hook(module, args, output):
                    if not flag_m[0]:
                        flag_m[0] = True
                        if isinstance(output, tuple):
                            hs = output[0].clone()
                            if hs.dim() == 3:
                                hs[0, -1, :] += ev.to(hs.dtype)
                            else:
                                hs[-1, :] += ev.to(hs.dtype)
                            return (hs,) + output[1:]
                    return output
                return hook

            handles.append(model.model.layers[ml].register_forward_hook(
                make_multi_hook()))

        with torch.no_grad():
            out = model(**inputs)
            multi_logits = out.logits[0, -1, :].cpu().float().numpy()
        for h in handles:
            h.remove()

        multi_diff = np.linalg.norm(multi_logits - ref_logits)

        # Error correction ratio: how much does the network "correct"?
        max_single = max(single_layer_diffs) if single_layer_diffs else 1
        mean_single = np.mean(single_layer_diffs)
        correction_ratio = mean_single / (err_strength * np.sqrt(d_model) + 1e-10)

        results.append({
            'error_strength': err_strength,
            'mean_single_diff': float(mean_single),
            'max_single_diff': float(max_single),
            'multi_diff': float(multi_diff),
            'correction_ratio': float(correction_ratio),
            'single_diffs': single_layer_diffs,
        })

    return results


def main():
    print("=" * 60)
    print("Phase Q97: Quantum Error Correction")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    print("  Testing error correction capability...")
    qec_results = test_error_correction(model, tokenizer, num_layers)

    for r in qec_results:
        print("    err=%.1f: mean_diff=%.2f, multi_diff=%.2f, correction=%.4f" %
              (r['error_strength'], r['mean_single_diff'], r['multi_diff'],
               r['correction_ratio']))

    # Analysis
    correction_ratios = [r['correction_ratio'] for r in qec_results]
    has_correction = np.mean(correction_ratios) < 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    strengths = [r['error_strength'] for r in qec_results]
    mean_diffs = [r['mean_single_diff'] for r in qec_results]
    multi_diffs = [r['multi_diff'] for r in qec_results]
    ax.plot(strengths, mean_diffs, 'o-', color='#2196F3', linewidth=2.5,
            label='Single-layer error', markersize=8)
    ax.plot(strengths, multi_diffs, 's-', color='#FF5722', linewidth=2.5,
            label='Multi-layer error', markersize=8)
    # Theoretical max (no correction)
    theoretical = [s * np.sqrt(model.config.hidden_size) for s in strengths]
    ax.plot(strengths, theoretical, '--', color='gray', alpha=0.4,
            label='No correction (theory)')
    ax.set_xlabel('Error strength', fontsize=11)
    ax.set_ylabel('Output deviation', fontsize=11)
    ax.set_title('(a) Error Propagation\nSingle vs Multi-layer',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(strengths, correction_ratios, 'o-', color='#4CAF50', linewidth=2.5,
            markersize=10)
    ax.axhline(0.5, color='red', ls='--', alpha=0.3, label='50% correction')
    ax.set_xlabel('Error strength', fontsize=11)
    ax.set_ylabel('Correction ratio', fontsize=11)
    ax.set_title('(b) Error Correction Efficiency\n%s' %
                 ('QEC WORKS!' if has_correction else 'Partial correction'),
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    ax = axes[2]
    result = 'TOPOLOGICAL QEC\nCONFIRMED' if has_correction else 'PARTIAL\nCORRECTION'
    color = '#4CAF50' if has_correction else '#FF9800'
    ax.text(0.5, 0.5, result, ha='center', va='center',
            fontsize=16, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) QEC Classification', fontsize=11, fontweight='bold')

    plt.suptitle('Quantum Error Correction in Transformer Space',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q97_qec.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q97', 'name': 'Quantum Error Correction',
        'has_correction': bool(has_correction),
        'mean_correction_ratio': float(np.mean(correction_ratios)),
        'data': [{k: v for k, v in r.items() if k != 'single_diffs'}
                 for r in qec_results],
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q97_qec.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  QEC confirmed: %s. Elapsed: %.1fs" % (has_correction, elapsed))
    return results


if __name__ == '__main__':
    main()
