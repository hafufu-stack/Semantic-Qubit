# -*- coding: utf-8 -*-
"""
Phase Q173: The Holographic Principle
=======================================
AdS/CFT correspondence: boundary encodes bulk.

Test: Does the LLM's output layer (boundary) encode the same
quantum information as internal layers (bulk)?

Measure mutual information between output layer and each
internal layer. If holographic: output should contain ALL info.
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


def rank_correlation(x, y):
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    try:
        r, _ = spearmanr(x, y)
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


def main():
    print("=" * 60)
    print("Phase Q173: The Holographic Principle")
    print("  (Does the Output Layer Encode Everything?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompts = [
        "The ground state energy of hydrogen:",
        "Chemical bond formation:",
        "Quantum entanglement between particles:",
        "The cat sat on the mat:",
        "Hello world program:",
        "Shakespeare wrote many plays:",
        "The laws of thermodynamics state:",
        "Neural networks learn patterns:",
    ]

    # For each prompt, compute all layer hidden states
    # Then test: can we reconstruct any layer from the output (last) layer?

    layer_similarities = np.zeros((n_layers + 1,))  # avg cos sim with output
    layer_reconstructability = np.zeros((n_layers + 1,))
    
    all_layer_data = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Collect all layer states
        states = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            states.append(h)

        output_state = states[-1]  # "Boundary"

        for li in range(n_layers + 1):
            # Similarity between layer and output
            sim = cosine_sim(states[li], output_state)
            layer_similarities[li] += sim

            # Reconstructability: can output predict this layer?
            # Use linear projection: find best alpha such that alpha*output ~ layer
            # R2 = cos^2
            r2 = sim ** 2
            layer_reconstructability[li] += r2

    layer_similarities /= len(prompts)
    layer_reconstructability /= len(prompts)

    # Information flow analysis
    # How much "new" information does each layer add?
    info_added = np.zeros(n_layers)
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for li in range(1, n_layers + 1):
            h_prev = out.hidden_states[li-1][0, -1, :].float().cpu().numpy()
            h_curr = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            residual = h_curr - h_prev
            info_added[li-1] += np.linalg.norm(residual) / (np.linalg.norm(h_curr) + 1e-10)
    info_added /= len(prompts)

    # Quantum holographic test:
    # If holographic, output contains ALL information -> similarity should be high everywhere
    # If NOT holographic, early layers should have low similarity with output
    avg_early = float(np.mean(layer_similarities[:5]))
    avg_mid = float(np.mean(layer_similarities[n_layers//3:2*n_layers//3]))
    avg_late = float(np.mean(layer_similarities[-5:]))

    print("\n--- Holographic Analysis ---")
    for li in range(0, n_layers + 1, 4):
        print("  Layer %2d: cos(layer, output)=%.4f, R2=%.4f" %
              (li, layer_similarities[li], layer_reconstructability[li]))
    print("\n  Early layers (0-4): avg cos = %.4f" % avg_early)
    print("  Mid layers: avg cos = %.4f" % avg_mid)
    print("  Late layers: avg cos = %.4f" % avg_late)

    is_holographic = avg_early > 0.3  # If early layers similar to output
    print("\n  Holographic? %s" %
          ("YES (boundary encodes bulk)" if is_holographic
           else "NO (information is transformed, not preserved)"))

    # Save
    results = {
        'phase': 'Q173',
        'name': 'Holographic Principle',
        'layer_similarities': [round(float(x), 4) for x in layer_similarities],
        'layer_reconstructability': [round(float(x), 4) for x in layer_reconstructability],
        'info_added_per_layer': [round(float(x), 4) for x in info_added],
        'summary': {
            'avg_early_sim': round(avg_early, 4),
            'avg_mid_sim': round(avg_mid, 4),
            'avg_late_sim': round(avg_late, 4),
            'is_holographic': is_holographic,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q173_holographic.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(range(n_layers + 1), layer_similarities, 'o-', color='#E91E63',
            linewidth=1.5, markersize=5)
    ax.axhline(1.0, color='green', ls='--', alpha=0.3, label='Identity')
    ax.axhspan(0, 0.3, alpha=0.1, color='red', label='Non-holographic zone')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity with Output')
    ax.set_title('(a) Layer-Output Similarity\n(Holographic = flat line)')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(range(n_layers + 1), layer_reconstructability, 's-',
            color='#4CAF50', linewidth=1.5, markersize=5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('R2 (Output -> Layer)')
    ax.set_title('(b) Reconstructability from Output')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(range(n_layers), info_added, color='#2196F3', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Relative Info Added')
    ax.set_title('(c) New Information per Layer')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q173: The Holographic Principle (Does Output Encode Everything?)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q173_holographic.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ173 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
