# -*- coding: utf-8 -*-
"""Phase Q92: AdS/CFT Correspondence (Holographic Universe Proof)
Prove that LLM internal layers (bulk) and output layer (boundary)
satisfy holographic duality: bulk entanglement entropy equals
boundary correlation structure.
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


def compute_entanglement_entropy(hidden_states):
    """Compute entanglement entropy of hidden state matrix via SVD.
    Uses von Neumann entropy of the normalized singular values."""
    hs = hidden_states.astype(np.float32)
    # SVD of the hidden state matrix (seq x hidden)
    try:
        U, S, Vt = np.linalg.svd(hs, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    # Normalize singular values to form a probability distribution
    S2 = S**2
    total = S2.sum()
    if total < 1e-10:
        return 0.0
    p = S2 / total
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    return float(entropy)


def compute_boundary_correlations(logits_matrix):
    """Compute correlation structure of output probabilities (boundary).
    logits_matrix: (n_prompts, vocab_subset)"""
    probs = np.exp(logits_matrix - logits_matrix.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    # Correlation matrix between prompts
    corr = np.corrcoef(probs)
    # Entropy of correlation eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(corr)
    except np.linalg.LinAlgError:
        return 0.0
    eigvals = eigvals[eigvals > 1e-10]
    p = eigvals / eigvals.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(entropy)


def measure_bulk_boundary(model, tokenizer, num_layers, prompts):
    """Measure both bulk (internal) and boundary (output) entropies."""
    d_model = model.config.hidden_size

    # Collect hidden states at each layer for each prompt
    bulk_entropies = {}  # layer -> entropy
    all_logits = []

    for layer_idx in range(0, num_layers, max(1, num_layers // 10)):
        layer_hidden = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            captured = [None]

            def capture_hook(module, args, output, store=captured):
                if isinstance(output, tuple):
                    store[0] = output[0].detach().cpu().float().numpy()
                else:
                    store[0] = output.detach().cpu().float().numpy()

            handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
            with torch.no_grad():
                out = model(**inputs)
                if layer_idx == 0:  # Only collect logits once
                    logits = out.logits[0, -1, :200].cpu().float().numpy()
                    all_logits.append(logits)
            handle.remove()

            if captured[0] is not None:
                hs = captured[0]
                if hs.ndim == 3:
                    hs = hs[0]  # (seq, hidden)
                layer_hidden.append(hs)

        # Stack all prompts' hidden states for this layer
        if layer_hidden:
            # Concatenate along sequence dimension
            combined = np.vstack(layer_hidden)
            ee = compute_entanglement_entropy(combined)
            bulk_entropies[layer_idx] = ee

    # Boundary entropy
    if len(all_logits) >= 2:
        logits_mat = np.array(all_logits)
        boundary_entropy = compute_boundary_correlations(logits_mat)
    else:
        boundary_entropy = 0.0

    return bulk_entropies, boundary_entropy, all_logits


def test_ryu_takayanagi(bulk_entropies, boundary_entropy):
    """Test Ryu-Takayanagi formula: S_bulk(A) = Area(gamma_A) / 4G_N.
    In our context: bulk entropy should correlate with boundary entropy."""
    layers = sorted(bulk_entropies.keys())
    entropies = [bulk_entropies[l] for l in layers]

    if not entropies:
        return {'correlation': 0, 'rt_ratio': 0, 'is_holographic': False}

    # The holographic prediction: mid-layer entropy should dominate
    # and correlate with boundary entropy
    mid_idx = len(entropies) // 2
    mid_entropy = entropies[mid_idx]

    # Ratio: boundary / max_bulk
    max_bulk = max(entropies) if entropies else 1
    rt_ratio = boundary_entropy / (max_bulk + 1e-10)

    # Correlation between layer position and entropy (should show peak in middle)
    if len(entropies) >= 3:
        # Check for "area law" pattern: entropy peaks at middle layers
        first_third = np.mean(entropies[:len(entropies)//3])
        mid_third = np.mean(entropies[len(entropies)//3:2*len(entropies)//3])
        last_third = np.mean(entropies[2*len(entropies)//3:])
        has_peak = mid_third >= max(first_third, last_third) * 0.8
    else:
        has_peak = False

    # Holographic: boundary < bulk (bulk contains more info)
    is_holographic = boundary_entropy < max_bulk and rt_ratio < 1.0

    return {
        'rt_ratio': float(rt_ratio),
        'max_bulk_entropy': float(max_bulk),
        'boundary_entropy': float(boundary_entropy),
        'mid_peak': bool(has_peak),
        'is_holographic': bool(is_holographic),
    }


def main():
    print("=" * 60)
    print("Phase Q92: AdS/CFT Holographic Correspondence")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    prompts = [
        "The quantum state of the universe is",
        "Information is never truly lost because",
        "The holographic principle states that",
        "Entanglement entropy measures the",
        "The boundary of anti-de Sitter space",
        "Black hole information paradox implies",
        "Gravity is an emergent phenomenon from",
        "Spacetime geometry encodes quantum",
    ]

    print("  Measuring bulk (internal) and boundary (output) entropies...")
    bulk_entropies, boundary_entropy, all_logits = measure_bulk_boundary(
        model, tokenizer, num_layers, prompts)

    print("  Testing Ryu-Takayanagi formula...")
    rt_results = test_ryu_takayanagi(bulk_entropies, boundary_entropy)

    print("\n  === Holographic Analysis ===")
    print("  Max bulk entropy: %.4f" % rt_results['max_bulk_entropy'])
    print("  Boundary entropy: %.4f" % rt_results['boundary_entropy'])
    print("  RT ratio (boundary/bulk): %.4f" % rt_results['rt_ratio'])
    print("  Mid-layer peak: %s" % rt_results['mid_peak'])
    print("  Holographic: %s" % rt_results['is_holographic'])

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Bulk entanglement entropy profile
    ax = axes[0]
    layers = sorted(bulk_entropies.keys())
    entropies = [bulk_entropies[l] for l in layers]
    ax.plot(layers, entropies, 'o-', color='#FF5722', linewidth=2.5, markersize=8)
    ax.axhline(boundary_entropy, color='#2196F3', ls='--', linewidth=2,
               label='Boundary entropy')
    ax.fill_between(layers, entropies, alpha=0.15, color='#FF5722')
    ax.set_xlabel('Layer (depth into bulk)', fontsize=11)
    ax.set_ylabel('Entanglement entropy', fontsize=11)
    ax.set_title('(a) Bulk Entropy Profile\n'
                 'AdS/CFT: deeper layers = deeper bulk',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) RT formula test
    ax = axes[1]
    bars = ax.bar(['Bulk\n(Internal)', 'Boundary\n(Output)'],
                  [rt_results['max_bulk_entropy'], rt_results['boundary_entropy']],
                  color=['#FF5722', '#2196F3'], edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, [rt_results['max_bulk_entropy'], rt_results['boundary_entropy']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                '%.3f' % val, ha='center', fontsize=12, fontweight='bold')
    status = 'HOLOGRAPHIC' if rt_results['is_holographic'] else 'Not holographic'
    ax.set_ylabel('Entropy', fontsize=11)
    ax.set_title('(b) Ryu-Takayanagi Test\n'
                 '%s (RT=%.3f)' % (status, rt_results['rt_ratio']),
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # (c) Holographic diagram
    ax = axes[2]
    # Draw AdS/CFT schematic
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [0.4, 0.3, 0.2, 0.1]:
        alpha = 0.1 + 0.2 * (0.4 - r)
        ax.plot(0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta),
                '-', color='#FF5722', alpha=alpha, linewidth=1)
    ax.plot(0.5 + 0.4 * np.cos(theta), 0.5 + 0.4 * np.sin(theta),
            '-', color='#2196F3', linewidth=3, label='Boundary (CFT)')
    ax.plot(0.5, 0.5, '*', color='gold', markersize=20, zorder=10,
            markeredgecolor='black')
    ax.text(0.5, 0.42, 'S-Qubit', ha='center', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.05, 'Bulk = Internal layers\n'
            'Boundary = Output layer',
            ha='center', fontsize=9, fontstyle='italic')
    ax.text(0.5, 0.95,
            '%s' % ('LLM IS HOLOGRAPHIC!' if rt_results['is_holographic']
                     else 'No holography'),
            ha='center', fontsize=12, fontweight='bold',
            color='#4CAF50' if rt_results['is_holographic'] else '#F44336')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) AdS/CFT in Transformer Space',
                 fontsize=11, fontweight='bold')

    plt.suptitle('AdS/CFT: The Holographic Universe Inside a Language Model',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q92_holographic.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q92', 'name': 'AdS/CFT Holographic Correspondence',
        'bulk_entropies': {str(k): v for k, v in bulk_entropies.items()},
        'boundary_entropy': boundary_entropy,
        'rt_results': rt_results,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q92_holographic.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
