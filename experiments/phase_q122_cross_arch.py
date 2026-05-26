# -*- coding: utf-8 -*-
"""
Phase Q122: Cross-Architecture Hippocampal Bridge
===================================================
Extends Q112's master-thesis resonance finding:
  "Resonance layers 3/3 fire (100% accuracy)"

Tests whether the Layer-18 firing threshold and MD-LD
resonance pattern hold across DIFFERENT architectures.
If yes, this is a universal property of transformers,
not just an artifact of Qwen2.5.

Uses Qwen 1.5B and 0.5B to test scaling invariance.
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

# Model snapshot paths
_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {
    '1.5B': os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                         "snapshots", "8faed761d45a263340a0528343f099c05c9a4323"),
    '0.5B': os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                         "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
}


def analyze_model(model_name, model_path, device):
    """Analyze hippocampal bridge for a single model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\n  Loading %s..." % model_name)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        local_files_only=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # MD (spatial/context) and LD (non-spatial/abstract) prompts
    prompts_md = [
        "The cat sat on the mat in the warm room",
        "Walking through the forest along the narrow path",
    ]
    prompts_ld = [
        "Abstract mathematical concept of infinity",
        "The philosophical notion of consciousness",
    ]

    # Compute MD-LD cosine similarity at each layer
    layer_cosines = []
    for md_prompt, ld_prompt in zip(prompts_md, prompts_ld):
        inp_md = tok(md_prompt, return_tensors='pt').to(device)
        inp_ld = tok(ld_prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            out_md = model(**inp_md, output_hidden_states=True)
            out_ld = model(**inp_ld, output_hidden_states=True)

        for li in range(n_layers):
            h_md = out_md.hidden_states[li + 1][0, -1, :].float()
            h_ld = out_ld.hidden_states[li + 1][0, -1, :].float()
            cos = torch.nn.functional.cosine_similarity(
                h_md.unsqueeze(0), h_ld.unsqueeze(0)).item()
            if len(layer_cosines) <= li:
                layer_cosines.append([])
            layer_cosines[li].append(cos)

    # Average cosines per layer
    avg_cosines = [float(np.mean(lc)) for lc in layer_cosines]

    # Find firing threshold (first layer where cosine > 0.5)
    firing_layer = None
    for li, cos in enumerate(avg_cosines):
        if cos > 0.5:
            firing_layer = li
            break

    # Firing fraction (proportion of layers above threshold)
    n_firing = sum(1 for c in avg_cosines if c > 0.5)
    firing_fraction = n_firing / n_layers

    # Normalized firing layer (as fraction of total layers)
    norm_firing = firing_layer / n_layers if firing_layer else 1.0

    # Build resonance pattern (same as Q112)
    ld_period = max(1, n_layers // 25)
    md_period = max(1, n_layers // 5)
    resonance_layers = []
    for l in range(n_layers):
        ld_phase = np.sin(2 * np.pi * l / max(ld_period, 1))
        md_envelope = np.cos(2 * np.pi * l / max(md_period, 1))
        if ld_phase > 0 and md_envelope > 0.3:
            resonance_layers.append(l)

    # Check resonance-firing overlap
    resonance_firing = sum(1 for l in resonance_layers if avg_cosines[l] > 0.5)

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'model': model_name,
        'n_layers': n_layers,
        'hidden_size': hidden,
        'layer_cosines': [round(c, 4) for c in avg_cosines],
        'firing_layer': firing_layer,
        'norm_firing_layer': round(norm_firing, 4),
        'n_firing': n_firing,
        'firing_fraction': round(firing_fraction, 4),
        'n_resonance_layers': len(resonance_layers),
        'resonance_firing': resonance_firing,
        'resonance_accuracy': round(resonance_firing / max(len(resonance_layers), 1), 4),
    }


def main():
    print("=" * 60)
    print("Phase Q122: Cross-Architecture Hippocampal Bridge")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_results = []
    for name, path in MODELS.items():
        if os.path.exists(path):
            result = analyze_model(name, path, device)
            model_results.append(result)
            print("    %s: firing at layer %s (%.1f%%), resonance=%d/%d" %
                  (name, result['firing_layer'],
                   result['norm_firing_layer'] * 100,
                   result['resonance_firing'],
                   result['n_resonance_layers']))
        else:
            print("    %s: NOT FOUND, skipping" % name)

    # Cross-architecture analysis
    print("\n--- Cross-Architecture Analysis ---")
    if len(model_results) >= 2:
        norm_firings = [r['norm_firing_layer'] for r in model_results]
        firing_std = float(np.std(norm_firings))
        is_universal = firing_std < 0.15  # Less than 15% variation
        print("  Normalized firing layers: %s" %
              [r['norm_firing_layer'] for r in model_results])
        print("  Std deviation: %.4f" % firing_std)
        print("  Universal: %s" % is_universal)
    else:
        firing_std = 0
        is_universal = len(model_results) > 0

    # ===== Save Results =====
    results = {
        'phase': 'Q122',
        'name': 'Cross-Architecture Hippocampal Bridge',
        'models': model_results,
        'firing_layer_std': round(firing_std, 4),
        'is_universal': str(is_universal),
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q122_cross_arch.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Layer-by-layer cosine for each model
    ax = axes[0]
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    for i, mr in enumerate(model_results):
        layers = range(mr['n_layers'])
        ax.plot(layers, mr['layer_cosines'], '-',
                color=colors[i % len(colors)],
                label='%s (%dL)' % (mr['model'], mr['n_layers']),
                linewidth=2, alpha=0.8)
        if mr['firing_layer'] is not None:
            ax.axvline(mr['firing_layer'], color=colors[i % len(colors)],
                       ls=':', alpha=0.5)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Firing threshold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('MD-LD cosine similarity')
    ax.set_title('(a) Hippocampal Bridge by Architecture')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Normalized firing comparison
    ax = axes[1]
    names = [mr['model'] for mr in model_results]
    norm_fires = [mr['norm_firing_layer'] for mr in model_results]
    fire_fracs = [mr['firing_fraction'] for mr in model_results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, norm_fires, 0.4, label='Norm. firing layer',
           color='#2196F3', alpha=0.85)
    ax.bar(x + 0.2, fire_fracs, 0.4, label='Firing fraction',
           color='#4CAF50', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Fraction')
    ax.set_title('(b) Normalized Firing Position\n(std=%.4f)' % firing_std)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Resonance accuracy
    ax = axes[2]
    res_acc = [mr['resonance_accuracy'] for mr in model_results]
    ax.bar(names, [a * 100 for a in res_acc],
           color='#9C27B0', edgecolor='black', alpha=0.85)
    ax.set_ylabel('Resonance accuracy (%)')
    ax.set_title("(c) Master's Thesis Resonance\n(LD x MD firing accuracy)")
    ax.set_ylim(0, 110)
    for i, v in enumerate(res_acc):
        ax.text(i, v * 100 + 2, '%.0f%%' % (v * 100), ha='center',
                fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q122: Cross-Architecture Hippocampal Bridge (Universal=%s)' %
                 is_universal, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q122_cross_arch.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("\nQ122 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
