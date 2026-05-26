# -*- coding: utf-8 -*-
"""
Phase Q126: Noise Resilience Deep-Dive
=======================================
Q123 showed COMPLETE topological protection - tokens survived
even noise=2.0! This is suspicious. Let's find the ACTUAL
breaking point by pushing noise to extreme levels, and test
with DISTRIBUTED noise (all layers, not just one).
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
    print("Phase Q126: Noise Resilience Deep-Dive")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    prompts = [
        ("The capital of France is", " Paris"),
        ("Water boils at one hundred", " degrees"),
        ("The color of the sky is", " blue"),
        ("Two plus two equals", " four"),
    ]

    # Fine-grained noise levels to find real breaking points
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    # === Experiment 1: Single-layer noise (as Q123 did) ===
    print("\n--- Exp 1: Single-layer noise (mid-layer) ---")
    single_results = []
    for prompt, expected in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Clean baseline
        with torch.no_grad():
            out_clean = model(**inp)
        top_clean = torch.argmax(out_clean.logits[0, -1, :]).item()
        expected_id = tok.encode(expected)[0] if expected else top_clean

        prompt_results = []
        for noise_level in noise_levels:
            mid = n_layers // 2

            def make_hook(nl):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        h = output.clone()
                    elif isinstance(output, tuple):
                        h = output[0].clone()
                    else:
                        return output
                    h_std = h.float().std().item()
                    scale = nl * max(h_std, 0.01)
                    nv = torch.randn(h.shape[-1], device=h.device, dtype=h.dtype) * scale
                    h[0, -1, :] = h[0, -1, :] + nv
                    if isinstance(output, torch.Tensor):
                        return h
                    return (h,) + output[1:]
                return hook

            handle = model.model.layers[mid].register_forward_hook(
                make_hook(noise_level))
            with torch.no_grad():
                out_noisy = model(**inp)
            handle.remove()

            logits_n = out_noisy.logits[0, -1, :].float()
            if torch.isnan(logits_n).any() or torch.isinf(logits_n).any():
                top_noisy = -1
                prob_correct = 0.0
            else:
                top_noisy = torch.argmax(logits_n).item()
                prob_correct = torch.softmax(logits_n, dim=-1)[expected_id].item()

            prompt_results.append({
                'noise': noise_level,
                'correct': str(top_noisy == top_clean),
                'prob_correct': round(float(prob_correct), 6),
            })

        # Find breaking point
        breaking = None
        for pr in prompt_results:
            if pr['correct'] == 'False':
                breaking = pr['noise']
                break

        single_results.append({
            'prompt': prompt[:30],
            'breaking_point': breaking,
            'noise_results': prompt_results,
        })
        print("  '%s' -> breaking at noise=%.1f" %
              (prompt[:30], breaking if breaking else float('inf')))

    # === Experiment 2: ALL-layer distributed noise ===
    print("\n--- Exp 2: Distributed noise (ALL layers) ---")
    distributed_results = []
    for prompt, expected in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            out_clean = model(**inp)
        top_clean = torch.argmax(out_clean.logits[0, -1, :]).item()
        expected_id = tok.encode(expected)[0] if expected else top_clean

        prompt_results = []
        for noise_level in noise_levels:
            # Add noise to EVERY layer
            handles = []
            for li in range(n_layers):
                def make_hook_dist(nl):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            h = output.clone()
                        elif isinstance(output, tuple):
                            h = output[0].clone()
                        else:
                            return output
                        h_std = h.float().std().item()
                        scale = nl * max(h_std, 0.01)
                        nv = torch.randn(h.shape[-1], device=h.device, dtype=h.dtype) * scale
                        h[0, -1, :] = h[0, -1, :] + nv
                        if isinstance(output, torch.Tensor):
                            return h
                        return (h,) + output[1:]
                    return hook

                handles.append(
                    model.model.layers[li].register_forward_hook(make_hook_dist(noise_level)))

            with torch.no_grad():
                out_noisy = model(**inp)

            for h in handles:
                h.remove()

            logits_n = out_noisy.logits[0, -1, :].float()
            if torch.isnan(logits_n).any() or torch.isinf(logits_n).any():
                top_noisy = -1
                prob_correct = 0.0
            else:
                top_noisy = torch.argmax(logits_n).item()
                prob_correct = torch.softmax(logits_n, dim=-1)[expected_id].item()

            prompt_results.append({
                'noise': noise_level,
                'correct': str(top_noisy == top_clean),
                'prob_correct': round(float(prob_correct), 6),
            })

        breaking = None
        for pr in prompt_results:
            if pr['correct'] == 'False':
                breaking = pr['noise']
                break

        distributed_results.append({
            'prompt': prompt[:30],
            'breaking_point': breaking,
            'noise_results': prompt_results,
        })
        print("  '%s' -> breaking at noise=%.1f" %
              (prompt[:30], breaking if breaking else float('inf')))

    # === Experiment 3: Adversarial noise (targeted) ===
    print("\n--- Exp 3: Adversarial noise (anti-gradient) ---")
    adversarial_results = []
    for prompt, expected in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Get gradient direction (which direction flips the token fastest)
        with torch.no_grad():
            out_clean = model(**inp, output_hidden_states=True)
        top_clean = torch.argmax(out_clean.logits[0, -1, :]).item()
        expected_id = tok.encode(expected)[0] if expected else top_clean

        # Use the hidden state direction as adversarial
        h_clean = out_clean.hidden_states[n_layers // 2 + 1][0, -1, :].float()
        adv_direction = -h_clean / h_clean.norm()  # Anti-parallel

        prompt_results = []
        for noise_level in noise_levels:
            nv_dir = (adv_direction * noise_level)
            mid = n_layers // 2

            def make_hook_adv(nv_d):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        h = output.clone()
                    elif isinstance(output, tuple):
                        h = output[0].clone()
                    else:
                        return output
                    h_std = h.float().std().item()
                    nv = (nv_d * max(h_std, 0.01)).to(h.dtype)
                    h[0, -1, :] = h[0, -1, :] + nv
                    if isinstance(output, torch.Tensor):
                        return h
                    return (h,) + output[1:]
                return hook

            handle = model.model.layers[mid].register_forward_hook(
                make_hook_adv(nv_dir))
            with torch.no_grad():
                out_noisy = model(**inp)
            handle.remove()

            logits_n = out_noisy.logits[0, -1, :].float()
            if torch.isnan(logits_n).any() or torch.isinf(logits_n).any():
                top_noisy = -1
                prob_correct = 0.0
            else:
                top_noisy = torch.argmax(logits_n).item()
                prob_correct = torch.softmax(logits_n, dim=-1)[expected_id].item()

            prompt_results.append({
                'noise': noise_level,
                'correct': str(top_noisy == top_clean),
                'prob_correct': round(float(prob_correct), 6),
            })

        breaking = None
        for pr in prompt_results:
            if pr['correct'] == 'False':
                breaking = pr['noise']
                break

        adversarial_results.append({
            'prompt': prompt[:30],
            'breaking_point': breaking,
            'noise_results': prompt_results,
        })
        print("  '%s' -> breaking at noise=%.1f" %
              (prompt[:30], breaking if breaking else float('inf')))

    # Summary
    print("\n--- Summary ---")
    for name, results in [('Single', single_results),
                           ('Distributed', distributed_results),
                           ('Adversarial', adversarial_results)]:
        breaks = [r['breaking_point'] for r in results]
        mean_break = float(np.mean([b for b in breaks if b])) if any(breaks) else float('inf')
        print("  %s: mean breaking=%.1f, unbroken=%d/%d" %
              (name, mean_break if mean_break < 1000 else -1,
               sum(1 for b in breaks if not b), len(breaks)))

    # ===== Save =====
    results = {
        'phase': 'Q126',
        'name': 'Noise Resilience Deep-Dive',
        'single_layer': single_results,
        'distributed': distributed_results,
        'adversarial': adversarial_results,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q126_noise.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, res_list) in zip(axes,
            [('(a) Single-layer', single_results),
             ('(b) Distributed (all layers)', distributed_results),
             ('(c) Adversarial', adversarial_results)]):
        for i, pr in enumerate(res_list):
            nls = [nr['noise'] for nr in pr['noise_results']]
            probs = [nr['prob_correct'] for nr in pr['noise_results']]
            ax.semilogx(nls[1:], probs[1:], 'o-',
                        label=pr['prompt'][:20], markersize=4)
        ax.axhline(0.5, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('Noise level (log)')
        ax.set_ylabel('P(correct token)')
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('Q126: Noise Resilience Deep-Dive (Breaking Points)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q126_noise.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ126 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
