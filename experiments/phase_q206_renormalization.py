# -*- coding: utf-8 -*-
"""
Phase Q206: Renormalization Group Isomorphism
===============================================
Test Deep Think's hypothesis: Transformer layers are mathematically
isomorphic to MERA (Multi-Scale Entanglement Renormalization Ansatz)
tensor network layers.

If the mutual information flow across Transformer layers matches MERA's
coarse-graining structure, it proves that deep learning naturally
performs the same operation as the renormalization group in physics.
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


def compute_layer_coarsegraining(model, tok, device, prompts, n_scales=5):
    """Measure how information is coarse-grained across Transformer layers.

    MERA prediction: each layer should halve the effective degrees of freedom
    (entropy should decrease logarithmically with layer depth).
    """
    all_layer_stats = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        hidden_states = out.hidden_states  # (n_layers+1, batch, seq, hidden)
        n_layers = len(hidden_states) - 1

        layer_entropies = []
        layer_ranks = []
        layer_norms = []
        layer_cosines = []

        for i in range(n_layers + 1):
            h = hidden_states[i][0].float().cpu().numpy()  # (seq, hidden)

            # Effective rank (participation ratio of singular values)
            _, s, _ = np.linalg.svd(h, full_matrices=False)
            s_norm = s / (s.sum() + 1e-10)
            participation_ratio = 1.0 / (np.sum(s_norm ** 2) + 1e-10)
            layer_ranks.append(float(participation_ratio))

            # Shannon entropy of activation distribution
            h_flat = np.abs(h.flatten())
            h_prob = h_flat / (h_flat.sum() + 1e-10)
            entropy = -np.sum(h_prob * np.log2(h_prob + 1e-15))
            layer_entropies.append(float(entropy))

            # Norm (total "energy")
            layer_norms.append(float(np.linalg.norm(h)))

            # Cosine with previous layer
            if i > 0:
                h_prev = hidden_states[i-1][0].float().cpu().numpy()
                cos = float(np.sum(h * h_prev) /
                           (np.linalg.norm(h) * np.linalg.norm(h_prev) + 1e-10))
                layer_cosines.append(cos)

        all_layer_stats.append({
            'entropies': layer_entropies,
            'ranks': layer_ranks,
            'norms': layer_norms,
            'cosines': layer_cosines,
        })

    return all_layer_stats


def test_mera_structure(ranks, n_layers):
    """Test if rank decay follows MERA-like coarse-graining.

    MERA prediction: effective rank decreases as O(2^{-l/l_0})
    where l is layer index and l_0 is the characteristic scale.
    """
    # Fit exponential decay: rank ~ A * exp(-alpha * layer)
    layers = np.arange(len(ranks))
    log_ranks = np.log(np.array(ranks) + 1e-10)

    # Linear fit to log(rank) vs layer
    if len(layers) > 2:
        coeffs = np.polyfit(layers, log_ranks, 1)
        alpha = -coeffs[0]  # decay rate
        r_squared = 1 - (np.sum((log_ranks - np.polyval(coeffs, layers))**2) /
                          np.sum((log_ranks - log_ranks.mean())**2))
    else:
        alpha = 0
        r_squared = 0

    return alpha, r_squared


def mutual_information_between_layers(model, tok, device, prompt,
                                       layer_a, layer_b, n_bins=50):
    """Estimate mutual information between two layers' hidden states."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    ha = out.hidden_states[layer_a][0, -1, :].float().cpu().numpy()
    hb = out.hidden_states[layer_b][0, -1, :].float().cpu().numpy()

    # Discretize for MI estimation
    dim = min(len(ha), 256)
    ha_disc = np.digitize(ha[:dim], np.linspace(ha[:dim].min(), ha[:dim].max(), n_bins))
    hb_disc = np.digitize(hb[:dim], np.linspace(hb[:dim].min(), hb[:dim].max(), n_bins))

    # Joint and marginal distributions
    joint = np.zeros((n_bins + 1, n_bins + 1))
    for a, b in zip(ha_disc, hb_disc):
        joint[a, b] += 1
    joint = joint / (joint.sum() + 1e-10)

    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_bins + 1):
        for j in range(n_bins + 1):
            if joint[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (pa[i] * pb[j]))

    return float(mi)


def main():
    print("=" * 60)
    print("Phase Q206: Renormalization Group Isomorphism")
    print("  (Is Transformer = MERA tensor network?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    print("  Model has %d layers" % n_layers)

    # Test prompts covering different domains
    prompts = [
        "The quantum mechanical wave function",
        "The protein folds into a complex three-dimensional structure",
        "The economy experienced a sharp contraction",
        "import torch; model = nn.Linear(512, 256)",
        "The entropy of an isolated system never decreases",
    ]

    # 1. Coarse-graining analysis
    print("\n--- Coarse-graining analysis ---")
    stats = compute_layer_coarsegraining(model, tok, device, prompts)

    avg_ranks = np.mean([s['ranks'] for s in stats], axis=0)
    avg_entropies = np.mean([s['entropies'] for s in stats], axis=0)
    avg_norms = np.mean([s['norms'] for s in stats], axis=0)

    alpha, r2 = test_mera_structure(avg_ranks.tolist(), n_layers)
    print("  Rank decay rate (alpha): %.4f" % alpha)
    print("  R^2 of exponential fit: %.4f" % r2)

    # 2. Mutual information distance scaling
    print("\n--- MI distance scaling ---")
    mi_matrix = np.zeros((n_layers + 1, n_layers + 1))
    test_prompt = prompts[0]

    # Sample layer pairs (not all for speed)
    layer_pairs = []
    for dist in range(1, min(n_layers + 1, 20)):
        for start in range(0, n_layers + 1 - dist, max(1, dist)):
            layer_pairs.append((start, start + dist))

    for la, lb in layer_pairs:
        mi = mutual_information_between_layers(
            model, tok, device, test_prompt, la, lb)
        mi_matrix[la, lb] = mi
        mi_matrix[lb, la] = mi

    # MI vs distance
    distances = sorted(set(abs(lb - la) for la, lb in layer_pairs))
    mi_vs_dist = []
    for d in distances:
        mis = [mi_matrix[la, lb] for la, lb in layer_pairs if abs(lb - la) == d]
        mi_vs_dist.append((d, float(np.mean(mis))))

    # Fit MI ~ d^(-gamma) (MERA predicts power-law decay)
    d_arr = np.array([x[0] for x in mi_vs_dist])
    mi_arr = np.array([x[1] for x in mi_vs_dist])
    valid = mi_arr > 0
    if valid.sum() > 2:
        log_d = np.log(d_arr[valid])
        log_mi = np.log(mi_arr[valid])
        gamma_coeffs = np.polyfit(log_d, log_mi, 1)
        gamma = -gamma_coeffs[0]
        mi_r2 = 1 - (np.sum((log_mi - np.polyval(gamma_coeffs, log_d))**2) /
                       np.sum((log_mi - log_mi.mean())**2))
    else:
        gamma = 0
        mi_r2 = 0

    print("  MI power-law exponent (gamma): %.3f" % gamma)
    print("  R^2 of power-law fit: %.4f" % mi_r2)

    # MERA isomorphism test
    is_mera = bool(alpha > 0.01 and r2 > 0.5 and gamma > 0.3)
    if is_mera:
        verdict = "MERA ISOMORPHISM: alpha=%.3f (R2=%.2f), gamma=%.3f -> Transformer = RG" % (
            alpha, r2, gamma)
    else:
        verdict = "PARTIAL MATCH: alpha=%.3f, gamma=%.3f (needs stronger decay)" % (alpha, gamma)

    print("\n--- Summary ---")
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q206',
        'name': 'Renormalization Group Isomorphism',
        'coarsegraining': {
            'decay_rate_alpha': round(alpha, 4),
            'r_squared': round(r2, 4),
            'avg_ranks': [round(r, 2) for r in avg_ranks.tolist()],
            'avg_entropies': [round(e, 2) for e in avg_entropies.tolist()],
        },
        'mi_scaling': {
            'power_law_gamma': round(gamma, 4),
            'r_squared': round(mi_r2, 4),
            'mi_vs_distance': mi_vs_dist,
        },
        'summary': {
            'is_mera': is_mera,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q206_renormalization.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Effective rank across layers
    ax = axes[0][0]
    ax.plot(range(len(avg_ranks)), avg_ranks, 'o-', color='#E91E63', lw=2)
    layers_fit = np.arange(len(avg_ranks))
    fit_vals = np.exp(np.polyval(np.polyfit(layers_fit,
                       np.log(avg_ranks + 1e-10), 1), layers_fit))
    ax.plot(layers_fit, fit_vals, '--', color='#2196F3',
            label='Exp fit (alpha=%.3f, R2=%.2f)' % (alpha, r2))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Effective Rank')
    ax.set_title('(a) Coarse-graining: Rank Decay')
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Entropy across layers
    ax = axes[0][1]
    ax.plot(range(len(avg_entropies)), avg_entropies, 's-', color='#4CAF50', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('(b) Information Content per Layer')
    ax.grid(alpha=0.3)

    # (c) MI vs distance
    ax = axes[1][0]
    d_vals = [x[0] for x in mi_vs_dist]
    mi_vals = [x[1] for x in mi_vs_dist]
    ax.loglog(d_vals, mi_vals, 'D-', color='#FF9800', lw=2)
    ax.set_xlabel('Layer Distance')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title('(c) MI vs Distance (gamma=%.2f)' % gamma)
    ax.grid(alpha=0.3, which='both')

    # (d) MI heatmap
    ax = axes[1][1]
    n_show = min(n_layers + 1, 25)
    im = ax.imshow(mi_matrix[:n_show, :n_show], cmap='hot',
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='MI (bits)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_title('(d) Layer-Layer MI Matrix')

    plt.suptitle('Q206: Renormalization Group Isomorphism\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q206_renormalization.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ206 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
