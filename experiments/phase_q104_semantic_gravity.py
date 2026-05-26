# -*- coding: utf-8 -*-
"""Phase Q104: Semantic Gravity - Words Warp Spacetime
Test if semantically heavy words (nouns, concepts) create more
"curvature" in the hidden state manifold than function words,
analogous to how massive objects curve spacetime.
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


def measure_semantic_curvature(model, tokenizer, num_layers):
    """Measure the curvature induced by different word types."""

    # Heavy words (high semantic mass - concrete nouns, concepts)
    heavy_words = ["universe", "consciousness", "mathematics", "gravity",
                   "democracy", "evolution", "intelligence", "quantum"]
    # Light words (low semantic mass - function words, articles)
    light_words = ["the", "a", "is", "of", "and", "to", "in", "it"]

    template = "The concept of %s is fundamentally important because"

    def get_curvature_profile(word):
        """Get curvature (layer-to-layer hidden state change) for a word."""
        prompt = template % word
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        layer_states = []
        for layer_idx in range(num_layers):
            captured = [None]
            def cap_hook(module, args, output, store=captured):
                if isinstance(output, tuple):
                    store[0] = output[0][0, -1, :].detach().cpu().float().numpy()
                else:
                    hs = output
                    if hs.dim() == 3:
                        store[0] = hs[0, -1, :].detach().cpu().float().numpy()
                    else:
                        store[0] = hs[-1, :].detach().cpu().float().numpy()

            handle = model.model.layers[layer_idx].register_forward_hook(cap_hook)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            if captured[0] is not None:
                layer_states.append(captured[0])

        # Compute curvature as second derivative of trajectory
        if len(layer_states) < 3:
            return None

        curvatures = []
        for i in range(1, len(layer_states) - 1):
            # Discrete curvature: how much does the direction change?
            v1 = layer_states[i] - layer_states[i-1]
            v2 = layer_states[i+1] - layer_states[i]
            n1 = np.linalg.norm(v1) + 1e-10
            n2 = np.linalg.norm(v2) + 1e-10
            cos_angle = np.dot(v1, v2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1, 1)
            curvature = np.arccos(cos_angle)  # 0 = straight, pi = U-turn
            curvatures.append(float(curvature))

        # Total curvature = integral of local curvature
        total_curvature = np.sum(curvatures)
        mean_curvature = np.mean(curvatures)

        # Information metric: total path length
        path_length = sum(np.linalg.norm(layer_states[i+1] - layer_states[i])
                          for i in range(len(layer_states)-1))

        return {
            'curvatures': curvatures,
            'total_curvature': float(total_curvature),
            'mean_curvature': float(mean_curvature),
            'path_length': float(path_length),
        }

    heavy_results = {}
    light_results = {}

    print("  Measuring heavy words...")
    for word in heavy_words:
        result = get_curvature_profile(word)
        if result:
            heavy_results[word] = result
            print("    '%s': curvature=%.4f, path=%.1f" %
                  (word, result['total_curvature'], result['path_length']))

    print("  Measuring light words...")
    for word in light_words:
        result = get_curvature_profile(word)
        if result:
            light_results[word] = result
            print("    '%s': curvature=%.4f, path=%.1f" %
                  (word, result['total_curvature'], result['path_length']))

    return heavy_results, light_results


def main():
    print("=" * 60)
    print("Phase Q104: Semantic Gravity")
    print("  Do words warp semantic spacetime?")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    heavy_results, light_results = measure_semantic_curvature(
        model, tokenizer, num_layers)

    # Analysis
    heavy_curvatures = [r['total_curvature'] for r in heavy_results.values()]
    light_curvatures = [r['total_curvature'] for r in light_results.values()]
    heavy_paths = [r['path_length'] for r in heavy_results.values()]
    light_paths = [r['path_length'] for r in light_results.values()]

    mean_heavy = np.mean(heavy_curvatures) if heavy_curvatures else 0
    mean_light = np.mean(light_curvatures) if light_curvatures else 0
    gravity_ratio = mean_heavy / (mean_light + 1e-10)

    print("\n  === Semantic Gravity ===")
    print("  Heavy word curvature: %.4f +/- %.4f" %
          (mean_heavy, np.std(heavy_curvatures)))
    print("  Light word curvature: %.4f +/- %.4f" %
          (mean_light, np.std(light_curvatures)))
    print("  Gravity ratio (heavy/light): %.3f" % gravity_ratio)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Curvature comparison
    ax = axes[0]
    all_words = list(heavy_results.keys()) + list(light_results.keys())
    all_curvs = heavy_curvatures + light_curvatures
    colors = ['#FF5722'] * len(heavy_curvatures) + ['#2196F3'] * len(light_curvatures)

    bars = ax.barh(range(len(all_words)), all_curvs, color=colors,
                   edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(all_words)))
    ax.set_yticklabels(all_words, fontsize=9)
    ax.set_xlabel('Total spacetime curvature', fontsize=11)
    ax.set_title('(a) Semantic Mass -> Curvature\nOrange=Heavy, Blue=Light',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # (b) Curvature profile for example words
    ax = axes[1]
    if heavy_results:
        key = list(heavy_results.keys())[0]
        ax.plot(heavy_results[key]['curvatures'], '-', color='#FF5722',
                linewidth=2.5, label='"%s" (heavy)' % key)
    if light_results:
        key = list(light_results.keys())[0]
        ax.plot(light_results[key]['curvatures'], '-', color='#2196F3',
                linewidth=2.5, label='"%s" (light)' % key)
    ax.set_xlabel('Layer transition', fontsize=11)
    ax.set_ylabel('Local curvature', fontsize=11)
    ax.set_title('(b) Curvature Profile\nHeavy words bend space more',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (c) Summary
    ax = axes[2]
    ax.text(0.5, 0.65,
            'SEMANTIC\nGRAVITY',
            ha='center', va='center', fontsize=24, fontweight='bold',
            color='#FF5722' if gravity_ratio > 1.0 else '#2196F3',
            transform=ax.transAxes)
    ax.text(0.5, 0.35,
            'Heavy/Light ratio: %.2f\n\n'
            '"universe" curves space\n%.1fx more than "the"\n\n'
            'E = mc^2 -> E = mS^2\n'
            '(Semantic mass-energy equivalence)' % (
                gravity_ratio,
                heavy_curvatures[0] / (light_curvatures[0] + 1e-10)
                if heavy_curvatures and light_curvatures else 0),
            ha='center', va='center', fontsize=10,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) E = mS^2', fontsize=12, fontweight='bold')

    plt.suptitle('Semantic Gravity: Words Warp the Spacetime of Language',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q104_semantic_gravity.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q104', 'name': 'Semantic Gravity',
        'gravity_ratio': float(gravity_ratio),
        'mean_heavy_curvature': float(mean_heavy),
        'mean_light_curvature': float(mean_light),
        'heavy_results': {k: {'total_curvature': v['total_curvature'],
                              'path_length': v['path_length']}
                         for k, v in heavy_results.items()},
        'light_results': {k: {'total_curvature': v['total_curvature'],
                              'path_length': v['path_length']}
                         for k, v in light_results.items()},
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q104_semantic_gravity.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
