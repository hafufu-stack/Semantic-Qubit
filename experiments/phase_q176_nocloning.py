# -*- coding: utf-8 -*-
"""
Phase Q176: The No-Cloning Test
================================
Quantum no-cloning theorem: you can't perfectly copy an unknown state.
Test: perturb prompt by adding/changing one character.
Measure sensitivity of hidden state.

If infinitesimally sensitive -> "quantum-like" (no cloning)
If robust to small perturbations -> classical (easily cloneable)
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


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 60)
    print("Phase Q176: The No-Cloning Test")
    print("  (Sensitivity to Single-Character Perturbation)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    def get_all_hidden(prompt):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        return [out.hidden_states[li][0, -1, :].float().cpu().numpy()
                for li in range(n_layers + 1)]

    # Test prompts
    base_prompts = [
        "The quantum state of the hydrogen atom",
        "Neural networks learn from data",
        "The quick brown fox jumps over the lazy dog",
    ]
    prompt_labels = ['Quantum', 'Neural', 'Fox']

    all_results = []

    for base, label in zip(base_prompts, prompt_labels):
        print("\n--- [%s] '%s' ---" % (label, base[:35]))

        h_base = get_all_hidden(base)

        perturbations = []

        # Type 1: Add single character at end
        for char in ['!', '.', ':', ' ', 'x']:
            perturbed = base + char
            h_pert = get_all_hidden(perturbed)
            # Measure similarity at each layer
            layer_sims = [cosine_sim(h_base[li], h_pert[li])
                          for li in range(n_layers + 1)]
            perturbations.append({
                'type': 'append_%s' % char.replace(' ', 'sp'),
                'layer_sims': [round(s, 6) for s in layer_sims],
                'output_sim': round(layer_sims[-1], 6),
                'input_sim': round(layer_sims[0], 6),
            })

        # Type 2: Change last character
        for replacement in ['a', 'z', '1', '#']:
            perturbed = base[:-1] + replacement
            h_pert = get_all_hidden(perturbed)
            layer_sims = [cosine_sim(h_base[li], h_pert[li])
                          for li in range(n_layers + 1)]
            perturbations.append({
                'type': 'replace_last_%s' % replacement,
                'layer_sims': [round(s, 6) for s in layer_sims],
                'output_sim': round(layer_sims[-1], 6),
                'input_sim': round(layer_sims[0], 6),
            })

        # Type 3: Capitalization change
        perturbed = base.upper()
        h_pert = get_all_hidden(perturbed)
        layer_sims = [cosine_sim(h_base[li], h_pert[li])
                      for li in range(n_layers + 1)]
        perturbations.append({
            'type': 'uppercase',
            'layer_sims': [round(s, 6) for s in layer_sims],
            'output_sim': round(layer_sims[-1], 6),
            'input_sim': round(layer_sims[0], 6),
        })

        # Statistics
        output_sims = [p['output_sim'] for p in perturbations]
        input_sims = [p['input_sim'] for p in perturbations]
        avg_output = float(np.mean(output_sims))
        avg_input = float(np.mean(input_sims))

        # Amplification factor: how much does a small input change get amplified?
        amplification = (1 - avg_output) / max(1 - avg_input, 1e-10)

        result = {
            'label': label,
            'base_prompt': base[:40],
            'perturbations': perturbations,
            'avg_input_sim': round(avg_input, 4),
            'avg_output_sim': round(avg_output, 4),
            'amplification_factor': round(amplification, 2),
        }
        all_results.append(result)

        print("  Input similarity:  %.4f (avg over perturbations)" % avg_input)
        print("  Output similarity: %.4f" % avg_output)
        print("  Amplification: %.1fx (change magnified through layers)" % amplification)

        # Layer profile for one perturbation
        p = perturbations[0]
        for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
            print("    Layer %2d: cos=%.4f" % (li, p['layer_sims'][li]))

    # No-Cloning assessment
    print("\n--- No-Cloning Assessment ---")
    avg_amp = float(np.mean([r['amplification_factor'] for r in all_results]))
    avg_out = float(np.mean([r['avg_output_sim'] for r in all_results]))
    print("  Avg amplification: %.1fx" % avg_amp)
    print("  Avg output similarity: %.4f" % avg_out)

    if avg_out > 0.9:
        verdict = "CLASSICAL (robust to perturbation, easily cloneable)"
    elif avg_out > 0.5:
        verdict = "SEMI-QUANTUM (sensitive but not maximally)"
    else:
        verdict = "QUANTUM-LIKE (highly sensitive, hard to clone)"
    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q176',
        'name': 'No-Cloning Test',
        'prompts': all_results,
        'summary': {
            'avg_amplification': round(avg_amp, 2),
            'avg_output_similarity': round(avg_out, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q176_nocloning.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Layer-by-layer sensitivity (first perturbation of each prompt)
    ax = axes[0]
    for i, r in enumerate(all_results):
        sims = r['perturbations'][0]['layer_sims']
        ax.plot(range(len(sims)), sims, 'o-', linewidth=1.5, markersize=3,
                label=r['label'])
    ax.axhline(1.0, color='green', ls='--', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity (original vs perturbed)')
    ax.set_title('(a) Layer-by-Layer Sensitivity\n(1 char perturbation)')
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Amplification factors
    ax = axes[1]
    labs = [r['label'] for r in all_results]
    amps = [r['amplification_factor'] for r in all_results]
    ax.bar(range(len(labs)), amps, color='#E91E63', edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='gray', ls='--', label='No amplification')
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_ylabel('Amplification Factor')
    ax.set_title('(b) Perturbation Amplification')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (c) Input vs Output similarity
    ax = axes[2]
    in_sims = [r['avg_input_sim'] for r in all_results]
    out_sims = [r['avg_output_sim'] for r in all_results]
    x = np.arange(len(labs))
    ax.bar(x - 0.15, in_sims, 0.3, color='#4CAF50', label='Input (embedding)', alpha=0.85)
    ax.bar(x + 0.15, out_sims, 0.3, color='#2196F3', label='Output (last layer)', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel('Avg Cosine Similarity')
    ax.set_title('(c) Input vs Output Robustness')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.suptitle('Q176: No-Cloning Test (Can You Copy the Quantum State?)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q176_nocloning.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ176 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
