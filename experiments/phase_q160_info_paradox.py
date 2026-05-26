# -*- coding: utf-8 -*-
"""
Phase Q160: The Information Paradox
=====================================
Black hole info paradox: info goes in, gets scrambled, can it come out?
LLM version: encode info in early layers, does it survive to output?

Measure: mutual information between input token embeddings
and deep hidden states. If preserved -> "unitary" (quantum-like).
If lost -> "dissipative" (classical thermodynamic).
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
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def linear_probe_accuracy(source, targets):
    """Can we linearly decode source from targets?
    Simple: correlation between source and each target.
    """
    correlations = []
    for t in targets:
        correlations.append(abs(cosine_sim(source, t)))
    return float(max(correlations)) if correlations else 0.0


def main():
    print("=" * 60)
    print("Phase Q160: The Information Paradox")
    print("  (Does LLM Preserve or Destroy Information?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Test: different inputs, track through layers
    test_prompts = [
        ("The hydrogen atom has one electron", "science"),
        ("The cat sat on the warm mat", "common"),
        ("Tokyo tower stands tall in Minato", "geography"),
        ("def fibonacci(n): return n if n < 2", "code"),
        ("The quick brown fox jumps over", "pangram"),
    ]

    all_results = []

    for prompt, category in test_prompts:
        print("\n--- [%s] '%s' ---" % (category, prompt[:35]))
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Track last token hidden state across ALL layers
        trajectory = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            trajectory.append(h)

        # Input embedding (layer 0)
        input_emb = trajectory[0]

        # Measure: how recoverable is input from each layer?
        recovery = []
        for li in range(n_layers + 1):
            # Cosine similarity: direct recovery
            cos = cosine_sim(input_emb, trajectory[li])

            # Effective dimension (how many components are "active")
            vals = np.abs(trajectory[li])
            sorted_vals = np.sort(vals)[::-1]
            cumsum = np.cumsum(sorted_vals ** 2)
            total = cumsum[-1] if len(cumsum) > 0 else 1
            eff_dim = int(np.searchsorted(cumsum, 0.9 * total) + 1)

            # Norm evolution
            norm = float(np.linalg.norm(trajectory[li]))

            recovery.append({
                'layer': int(li),
                'cos_with_input': round(cos, 4),
                'effective_dim': int(eff_dim),
                'norm': round(norm, 2),
            })

        # Information preservation index
        # = area under recovery curve (higher = more preservation)
        cos_values = [r['cos_with_input'] for r in recovery]
        preservation_index = float(np.trapz(cos_values)) / n_layers

        # Scrambling point: where does cos drop below 0.5?
        scramble_layer = n_layers
        for r in recovery:
            if abs(r['cos_with_input']) < 0.5:
                scramble_layer = r['layer']
                break

        result = {
            'prompt': prompt[:35],
            'category': category,
            'preservation_index': round(preservation_index, 4),
            'scramble_layer': int(scramble_layer),
            'final_cos': round(cos_values[-1], 4),
            'recovery_curve': recovery,
        }
        all_results.append(result)

        print("  Preservation index: %.4f" % preservation_index)
        print("  Scrambling starts at layer: %d / %d" % (scramble_layer, n_layers))
        print("  Input-Output cosine: %.4f" % cos_values[-1])

        # Print every 4th layer
        for r in recovery[::4]:
            print("    Layer %2d: cos=%.3f, eff_dim=%d, norm=%.1f" %
                  (r['layer'], r['cos_with_input'], r['effective_dim'], r['norm']))

    # Cross-prompt distinguishability: can we tell prompts apart at output?
    print("\n--- Cross-prompt Distinguishability ---")
    final_states = [all_results[i]['recovery_curve'][-1] for i in range(len(all_results))]
    prompts_used = [r['category'] for r in all_results]

    # Get actual final hidden states
    final_hidden = []
    for prompt, _ in test_prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        final_hidden.append(out.hidden_states[-1][0, -1, :].float().cpu().numpy())

    # Pairwise distinguishability
    cross_sims = []
    for i in range(len(final_hidden)):
        for j in range(i+1, len(final_hidden)):
            sim = cosine_sim(final_hidden[i], final_hidden[j])
            cross_sims.append(sim)
            print("  %s vs %s: cos=%.4f" % (prompts_used[i], prompts_used[j], sim))

    avg_cross = float(np.mean(cross_sims))
    print("  Average cross-prompt similarity: %.4f" % avg_cross)
    print("  (Lower = better distinguishability = more info preserved)")

    # Summary
    print("\n--- Information Paradox Summary ---")
    avg_preservation = float(np.mean([r['preservation_index'] for r in all_results]))
    avg_scramble = float(np.mean([r['scramble_layer'] for r in all_results]))
    print("  Avg preservation index: %.4f" % avg_preservation)
    print("  Avg scrambling layer: %.1f / %d" % (avg_scramble, n_layers))
    verdict = "UNITARY (quantum-like)" if avg_preservation > 0.5 else \
              "PARTIALLY UNITARY" if avg_preservation > 0.2 else \
              "DISSIPATIVE (classical)"
    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q160',
        'name': 'Information Paradox',
        'prompts': all_results,
        'cross_similarity': round(avg_cross, 4),
        'avg_preservation': round(avg_preservation, 4),
        'avg_scramble_layer': round(avg_scramble, 1),
        'verdict': verdict,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q160_info_paradox.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Recovery curves
    ax = axes[0]
    colors = ['#E91E63', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    for i, r in enumerate(all_results):
        layers = [x['layer'] for x in r['recovery_curve']]
        cos_vals = [x['cos_with_input'] for x in r['recovery_curve']]
        ax.plot(layers, cos_vals, 'o-', color=colors[i % len(colors)],
                label=r['category'], linewidth=1.5, markersize=2)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Scrambling threshold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine with Input')
    ax.set_title('(a) Information Recovery Curves')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # (b) Effective dimension
    ax = axes[1]
    for i, r in enumerate(all_results):
        layers = [x['layer'] for x in r['recovery_curve']]
        dims = [x['effective_dim'] for x in r['recovery_curve']]
        ax.plot(layers, dims, 'o-', color=colors[i % len(colors)],
                label=r['category'], linewidth=1.5, markersize=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Effective Dimension (90% energy)')
    ax.set_title('(b) Information Spreading')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # (c) Preservation summary
    ax = axes[2]
    cats = [r['category'] for r in all_results]
    pres = [r['preservation_index'] for r in all_results]
    ax.bar(range(len(cats)), pres, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0.5, color='red', ls='--', label='Unitary threshold')
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel('Preservation Index')
    ax.set_title('(c) Is LLM Unitary or Dissipative?')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q160: The Information Paradox (Scrambling vs Preservation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q160_info_paradox.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ160 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
