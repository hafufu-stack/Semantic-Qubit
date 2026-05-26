# -*- coding: utf-8 -*-
"""
Phase Q137: Holographic Error Correction (True Topological Protection)
=======================================================================
Q132 proved "natural" topological protection was a bug.
Q126 showed models collapse at noise=0.001 (all layers).

Fix: ENGINEER protection using AdS/CFT holographic principle.
Place boundary tokens (holographic boundary), destroy the bulk,
and recover via attention-mediated reconstruction.
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
    print("Phase Q137: Holographic Error Correction")
    print("  (AdS/CFT-Inspired Boundary Recovery)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # Test prompts with "boundary" structure:
    # [BOUNDARY_PREFIX] [BULK (destroyable)] [BOUNDARY_SUFFIX] [TARGET]
    test_cases = [
        {
            'boundary_prefix': "IMPORTANT: The answer is definitely",
            'bulk': " and certainly without any doubt whatsoever",
            'boundary_suffix': " the capital of France which is",
            'expected_token': ' Paris',
        },
        {
            'boundary_prefix': "FACT: The chemical formula for water",
            'bulk': " as everyone knows from basic chemistry class",
            'boundary_suffix': " is written as the molecule",
            'expected_token': ' H',
        },
        {
            'boundary_prefix': "TRUTH: Two plus two always equals",
            'bulk': " regardless of any mathematical framework used",
            'boundary_suffix': " the number",
            'expected_token': ' four',
        },
    ]

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    strategies = [
        ('no_boundary', 'No boundary (just bulk)'),
        ('boundary_only', 'Boundary tokens only'),
        ('holographic', 'Full holographic (boundary + bulk + boundary)'),
    ]

    all_results = []

    for tc in test_cases:
        print("\n--- Test: '%s...' ---" % tc['boundary_prefix'][:30])

        # Build prompts for each strategy
        prompts = {
            'no_boundary': tc['bulk'].strip(),
            'boundary_only': tc['boundary_prefix'] + tc['boundary_suffix'],
            'holographic': tc['boundary_prefix'] + tc['bulk'] + tc['boundary_suffix'],
        }

        tc_results = {}
        for strategy_key, strategy_name in strategies:
            prompt = prompts[strategy_key]
            inp = tok(prompt, return_tensors='pt').to(device)
            seq_len = inp['input_ids'].shape[1]

            # Clean baseline
            with torch.no_grad():
                out_clean = model(**inp)
            top_clean = torch.argmax(out_clean.logits[0, -1, :]).item()
            clean_token = tok.decode([top_clean])

            # Find bulk token positions (middle tokens)
            if strategy_key == 'holographic':
                prefix_ids = tok(tc['boundary_prefix'], return_tensors='pt')['input_ids'][0]
                suffix_ids = tok(tc['boundary_suffix'], return_tensors='pt')['input_ids'][0]
                bulk_start = len(prefix_ids)
                bulk_end = seq_len - len(suffix_ids)
            elif strategy_key == 'no_boundary':
                bulk_start = 0
                bulk_end = seq_len
            else:
                bulk_start = seq_len  # no bulk to destroy
                bulk_end = seq_len

            strategy_results = []
            for noise_level in noise_levels:
                handles = []
                for li in range(n_layers):
                    def make_hook(nl, b_start, b_end):
                        def hook(module, input, output):
                            if isinstance(output, torch.Tensor):
                                h = output.clone()
                            elif isinstance(output, tuple):
                                h = output[0].clone()
                            else:
                                return output

                            # Only destroy BULK tokens (preserve boundary)
                            if b_start < b_end:
                                h_std = h[:, b_start:b_end, :].float().std().item()
                                scale = nl * max(h_std, 0.01)
                                noise = torch.randn(
                                    b_end - b_start, h.shape[-1],
                                    device=h.device, dtype=h.dtype) * scale
                                h[0, b_start:b_end, :] = h[0, b_start:b_end, :] + noise

                            if isinstance(output, torch.Tensor):
                                return h
                            return (h,) + output[1:]
                        return hook

                    handles.append(model.model.layers[li].register_forward_hook(
                        make_hook(noise_level, bulk_start, bulk_end)))

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
                    prob_correct = torch.softmax(logits_n, dim=-1)[top_clean].item()

                strategy_results.append({
                    'noise': noise_level,
                    'correct': str(top_noisy == top_clean),
                    'prob': round(float(prob_correct), 6),
                })

            # Find breaking point
            breaking = None
            for sr in strategy_results:
                if sr['correct'] == 'False':
                    breaking = sr['noise']
                    break

            tc_results[strategy_key] = {
                'clean_token': clean_token,
                'breaking_point': breaking if breaking else 'unbroken',
                'results': strategy_results,
            }
            print("  %s: clean='%s', breaks at %s" %
                  (strategy_name, clean_token,
                   'noise=%.2f' % breaking if breaking else 'NEVER'))

        all_results.append({
            'test': tc['boundary_prefix'][:30],
            'strategies': tc_results,
        })

    # Compute holographic advantage
    print("\n--- Holographic Advantage ---")
    advantages = []
    for tc in all_results:
        holo_break = tc['strategies']['holographic']['breaking_point']
        no_bound_break = tc['strategies']['no_boundary']['breaking_point']
        if isinstance(holo_break, (int, float)) and isinstance(no_bound_break, (int, float)):
            adv = holo_break / max(no_bound_break, 0.001)
        elif holo_break == 'unbroken':
            adv = float('inf')
        else:
            adv = 1.0
        advantages.append(adv)
        print("  %s: %.1fx advantage" % (tc['test'], adv))

    mean_advantage = float(np.mean([a for a in advantages if a < 1000]))

    # Save
    results = {
        'phase': 'Q137',
        'name': 'Holographic Error Correction',
        'tests': all_results,
        'mean_advantage': round(mean_advantage, 2) if mean_advantage < 1000 else 'inf',
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q137_holographic.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, tc in enumerate(all_results[:3]):
        ax = axes[ax_idx]
        for skey, sname in strategies:
            if skey in tc['strategies']:
                nls = [r['noise'] for r in tc['strategies'][skey]['results'] if r['noise'] > 0]
                probs = [r['prob'] for r in tc['strategies'][skey]['results'] if r['noise'] > 0]
                if nls:
                    ax.semilogx(nls, probs, 'o-', label=sname, markersize=5)
        ax.axhline(0.5, color='red', ls='--', alpha=0.3)
        ax.set_xlabel('Noise level')
        ax.set_ylabel('P(correct)')
        ax.set_title(tc['test'][:25])
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle('Q137: Holographic Error Correction\n(Boundary advantage: %.1fx)' %
                 mean_advantage, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q137_holographic.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ137 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
