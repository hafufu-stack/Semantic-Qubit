# -*- coding: utf-8 -*-
"""
Phase Q123: Semantic Topological Protection
=============================================
Tests whether S-Qubit states exhibit topological protection
against noise and perturbation, analogous to topological
qubits (Microsoft's approach to fault-tolerant QC).

Topological protection = information encoded in GLOBAL
properties (winding numbers, Chern numbers) rather than
local states, making it robust against local perturbation.
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


def main():
    print("=" * 60)
    print("Phase Q123: Semantic Topological Protection")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    test_prompts = [
        "The speed of light is",
        "Water freezes at zero",
        "The sun rises in the",
    ]

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    all_results = []

    for prompt in test_prompts:
        print("\n  Prompt: '%s'" % prompt)
        inp = tok(prompt, return_tensors='pt').to(device)

        # Get clean output
        with torch.no_grad():
            out_clean = model(**inp, output_hidden_states=True)
        h_clean_final = out_clean.hidden_states[-1][0, -1, :].float()
        logits_clean = out_clean.logits[0, -1, :]
        top_clean = torch.argmax(logits_clean).item()
        prob_clean = torch.softmax(logits_clean, dim=-1)[top_clean].item()

        # Compute winding number of clean state (topological invariant)
        # Winding = how many times the phase wraps around 2pi across dimensions
        phases_clean = torch.atan2(h_clean_final[1::2], h_clean_final[::2])
        phase_diffs = phases_clean[1:] - phases_clean[:-1]
        # Normalize to [-pi, pi]
        phase_diffs = torch.remainder(phase_diffs + np.pi, 2 * np.pi) - np.pi
        winding_clean = float(phase_diffs.sum().item() / (2 * np.pi))

        noise_results = []
        for noise_level in noise_levels:
            # Inject noise at a SINGLE layer (local perturbation)
            mid_layer = n_layers // 2

            def make_noise_hook(nl):
                def noise_hook(module, input, output):
                    # Qwen2 layers return Tensor directly, not tuple
                    if isinstance(output, torch.Tensor):
                        h = output.clone()
                    elif isinstance(output, tuple):
                        h = output[0].clone()
                    else:
                        return output

                    if h.dim() == 3:
                        h_std = h.float().std().item()
                        scale = nl * max(h_std, 0.01)
                        nv = torch.randn(h.shape[-1], device=h.device, dtype=h.dtype) * scale
                        h[0, -1, :] = h[0, -1, :] + nv

                    if isinstance(output, torch.Tensor):
                        return h
                    else:
                        return (h,) + output[1:]
                return noise_hook

            handle = model.model.layers[mid_layer].register_forward_hook(make_noise_hook(noise_level))
            with torch.no_grad():
                out_noisy = model(**inp, output_hidden_states=True)
            handle.remove()

            h_noisy_final = out_noisy.hidden_states[-1][0, -1, :].float()
            logits_noisy = out_noisy.logits[0, -1, :]
            top_noisy = torch.argmax(logits_noisy).item()
            prob_noisy = torch.softmax(logits_noisy, dim=-1)[top_clean].item()

            # Compute winding number of noisy state
            phases_noisy = torch.atan2(h_noisy_final[1::2], h_noisy_final[::2])
            phase_diffs_n = phases_noisy[1:] - phases_noisy[:-1]
            phase_diffs_n = torch.remainder(phase_diffs_n + np.pi, 2 * np.pi) - np.pi
            winding_noisy = float(phase_diffs_n.sum().item() / (2 * np.pi))

            # Metrics
            state_fidelity = torch.nn.functional.cosine_similarity(
                h_clean_final.unsqueeze(0), h_noisy_final.unsqueeze(0)).item()
            token_preserved = top_noisy == top_clean
            winding_preserved = abs(winding_noisy - winding_clean) < 0.5

            noise_results.append({
                'noise_level': noise_level,
                'state_fidelity': round(state_fidelity, 6),
                'prob_original_token': round(prob_noisy, 6),
                'token_preserved': str(token_preserved),
                'winding_clean': round(winding_clean, 4),
                'winding_noisy': round(winding_noisy, 4),
                'winding_preserved': str(winding_preserved),
            })

        # Find critical noise (where token flips)
        critical_noise = None
        for nr in noise_results:
            if nr['token_preserved'] == 'False':
                critical_noise = nr['noise_level']
                break

        # Find winding stability threshold
        winding_threshold = None
        for nr in noise_results:
            if nr['winding_preserved'] == 'False':
                winding_threshold = nr['noise_level']
                break

        all_results.append({
            'prompt': prompt,
            'clean_token': tok.decode([top_clean]),
            'clean_prob': round(prob_clean, 4),
            'winding_number': round(winding_clean, 4),
            'critical_noise': critical_noise,
            'winding_threshold': winding_threshold,
            'noise_results': noise_results,
        })
        print("    Winding=%.2f, critical_noise=%s, winding_stable_until=%s" %
              (winding_clean, critical_noise, winding_threshold))

    # Summary
    critical_noises = [r['critical_noise'] for r in all_results if r['critical_noise']]
    winding_thresholds = [r['winding_threshold'] for r in all_results if r['winding_threshold']]
    mean_critical = float(np.mean(critical_noises)) if critical_noises else float('inf')
    mean_winding_thresh = float(np.mean(winding_thresholds)) if winding_thresholds else float('inf')

    # Topological protection = winding number survives noise that kills local state
    topological_protection = mean_winding_thresh > mean_critical if (critical_noises and winding_thresholds) else False

    print("\n--- Summary ---")
    print("  Mean critical noise (token): %.3f" %
          (mean_critical if mean_critical < 100 else -1))
    print("  Mean winding threshold: %.3f" %
          (mean_winding_thresh if mean_winding_thresh < 100 else -1))
    print("  Topological protection: %s" % topological_protection)

    # ===== Save Results =====
    results = {
        'phase': 'Q123',
        'name': 'Semantic Topological Protection',
        'prompts': all_results,
        'mean_critical_noise': round(mean_critical, 4) if mean_critical < 100 else 'inf',
        'mean_winding_threshold': round(mean_winding_thresh, 4) if mean_winding_thresh < 100 else 'inf',
        'topological_protection': str(topological_protection),
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q123_topological.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) State fidelity vs noise
    ax = axes[0]
    for i, pr in enumerate(all_results):
        nls = [nr['noise_level'] for nr in pr['noise_results']]
        fids = [nr['state_fidelity'] for nr in pr['noise_results']]
        ax.plot(nls, fids, 'o-', label=pr['prompt'][:20], markersize=4)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('State fidelity')
    ax.set_title('(a) State Fidelity vs Noise')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Token probability preservation
    ax = axes[1]
    for i, pr in enumerate(all_results):
        nls = [nr['noise_level'] for nr in pr['noise_results']]
        probs = [nr['prob_original_token'] for nr in pr['noise_results']]
        ax.plot(nls, probs, 'o-', label=pr['prompt'][:20], markersize=4)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Original token probability')
    ax.set_title('(b) Token Preservation')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) Winding number stability
    ax = axes[2]
    for i, pr in enumerate(all_results):
        nls = [nr['noise_level'] for nr in pr['noise_results']]
        winds = [nr['winding_noisy'] for nr in pr['noise_results']]
        ax.plot(nls, winds, 'o-', label=pr['prompt'][:20], markersize=4)
        # Clean winding baseline
        ax.axhline(pr['winding_number'], color='C%d' % i, ls=':', alpha=0.3)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Winding number')
    ax.set_title('(c) Topological Invariant Stability\n(protected=%s)' %
                 topological_protection)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.suptitle('Q123: Semantic Topological Protection',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q123_topological.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ123 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
