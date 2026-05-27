# -*- coding: utf-8 -*-
"""
Phase Q187: The MERA Isomorphism (LLM = Holographic Universe)
================================================================
Q184: Entanglement grows 5.85x through layers (monotonic).
MERA (Multiscale Entanglement Renormalization Ansatz) predicts
entanglement entropy grows logarithmically with system size.

AdS/CFT (Ryu-Takayanagi): S(A) = Area(gamma_A) / 4G_N
MERA implements this holographic principle in tensor networks.

Test: Does LLM's entanglement growth match MERA predictions?
If yes -> LLM IS a holographic universe simulator.
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


def entanglement_entropy(h_matrix, partition_size):
    """
    Compute entanglement entropy of a set of token representations.
    Uses SVD of the (tokens x hidden) matrix as a proxy for quantum entropy.
    """
    if h_matrix.shape[0] < 2:
        return 0.0
    # SVD
    U, S, Vt = np.linalg.svd(h_matrix[:partition_size], full_matrices=False)
    S = S[S > 1e-10]
    # Normalize
    S = S / np.sum(S)
    # Von Neumann entropy
    entropy = -np.sum(S * np.log(S + 1e-15))
    return float(entropy)


def main():
    print("=" * 60)
    print("Phase Q187: The MERA Isomorphism")
    print("  (Is LLM a Holographic Universe?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # === Test 1: Entanglement Entropy Scaling ===
    print("\n--- Test 1: Entanglement Entropy vs Layer Depth ---")

    prompts = [
        "Paris and Berlin are both capitals of major European countries",
        "The sun and the moon illuminate the sky in different ways",
        "Cats and dogs are beloved pets that bring joy to families",
        "Fire and ice represent opposite forces in nature and mythology",
    ]

    entropy_profiles = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        n_tokens = inp['input_ids'].shape[1]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Compute entanglement entropy at each layer
        layer_entropies = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0].float().cpu().numpy()  # (seq, hidden)
            # Partition: first half vs second half of tokens
            half = max(1, n_tokens // 2)
            ent = entanglement_entropy(h, half)
            layer_entropies.append(ent)

        entropy_profiles.append(layer_entropies)
        print("  '%s...' -> S_max=%.4f at L%d" %
              (prompt[:30], max(layer_entropies),
               np.argmax(layer_entropies)))

    avg_entropy = np.mean(entropy_profiles, axis=0)

    # === Test 2: MERA Fit ===
    print("\n--- Test 2: Fitting to MERA/CFT Predictions ---")

    layers = np.arange(n_layers + 1)

    # MERA prediction: S(l) = (c/3) * log(l + 1) + const
    # where c is the central charge of the CFT
    # Fit: S = a * log(l + 1) + b
    from numpy.polynomial import polynomial as P
    log_layers = np.log(layers + 1)

    # Linear fit in log space
    valid = avg_entropy > 0
    if np.sum(valid) > 2:
        coeffs = np.polyfit(log_layers[valid], avg_entropy[valid], 1)
        a_fit, b_fit = coeffs
        c_central = 3 * a_fit  # central charge
        mera_fit = a_fit * log_layers + b_fit
        residuals = avg_entropy[valid] - (a_fit * log_layers[valid] + b_fit)
        r_squared = 1 - np.var(residuals) / np.var(avg_entropy[valid])
    else:
        a_fit, b_fit, c_central = 0, 0, 0
        mera_fit = np.zeros_like(avg_entropy)
        r_squared = 0

    print("  MERA fit: S = %.4f * log(l+1) + %.4f" % (a_fit, b_fit))
    print("  Central charge c = %.4f" % c_central)
    print("  R^2 = %.4f" % r_squared)

    # Alternative fit: linear S = a*l + b (trivial)
    coeffs_lin = np.polyfit(layers[valid], avg_entropy[valid], 1)
    linear_fit = coeffs_lin[0] * layers + coeffs_lin[1]
    res_lin = avg_entropy[valid] - (coeffs_lin[0] * layers[valid] + coeffs_lin[1])
    r2_linear = 1 - np.var(res_lin) / np.var(avg_entropy[valid])

    # Power law: S = a * l^b
    log_ent = np.log(avg_entropy[valid] + 1e-10)
    log_l = np.log(layers[valid] + 1)
    if len(log_l) > 2:
        coeffs_pow = np.polyfit(log_l, log_ent, 1)
        power_exp = coeffs_pow[0]
        r2_power = 1 - np.var(log_ent - (coeffs_pow[0] * log_l + coeffs_pow[1])) / np.var(log_ent)
    else:
        power_exp = 0
        r2_power = 0

    print("\n  Model comparison:")
    print("    Logarithmic (MERA/CFT): R^2=%.4f" % r_squared)
    print("    Linear:                 R^2=%.4f" % r2_linear)
    print("    Power law (exp=%.2f):   R^2=%.4f" % (power_exp, r2_power))

    # === Test 3: Ryu-Takayanagi Area Law ===
    print("\n--- Test 3: Area Law (Ryu-Takayanagi) ---")

    # Test how entanglement scales with partition size
    prompt = "The quick brown fox jumps over the lazy dog in the park"
    inp = tok(prompt, return_tensors='pt').to(device)
    n_tokens = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # At the deepest layer, measure S(A) for different partition sizes
    h_deep = out.hidden_states[-1][0].float().cpu().numpy()

    partition_sizes = list(range(1, n_tokens))
    partition_entropies = []

    for ps in partition_sizes:
        ent = entanglement_entropy(h_deep, ps)
        partition_entropies.append(ent)

    # Area law: S(A) peaks at half-system size (for 1D CFT)
    peak_idx = np.argmax(partition_entropies)
    peak_size = partition_sizes[peak_idx]
    print("  Peak entropy at partition=%d/%d (%.1f%% of system)" %
          (peak_size, n_tokens, 100 * peak_size / n_tokens))
    print("  -> Expected peak at 50%% (Area law), got %.1f%%" %
          (100 * peak_size / n_tokens))

    # Fit to CFT: S = (c/3) * log(L/pi * sin(pi*l/L))
    L = n_tokens
    x = np.array(partition_sizes)
    cft_prediction = np.log(L / np.pi * np.sin(np.pi * x / L) + 1e-10)
    if np.var(cft_prediction) > 0:
        c_area = np.polyfit(cft_prediction, partition_entropies, 1)[0] * 3
    else:
        c_area = 0
    print("  Central charge from area law: c = %.4f" % c_area)

    # Determine verdict
    if r_squared > 0.9:
        verdict = "STRONG MERA ISOMORPHISM: R^2=%.4f, c=%.2f" % (r_squared, c_central)
    elif r_squared > 0.7:
        verdict = "MODERATE MERA MATCH: R^2=%.4f" % r_squared
    else:
        best_model = 'logarithmic' if r_squared > max(r2_linear, r2_power) else \
                     'linear' if r2_linear > r2_power else 'power-law'
        verdict = "BEST FIT: %s (R^2=%.4f)" % (best_model,
                  max(r_squared, r2_linear, r2_power))

    print("\n  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q187',
        'name': 'MERA Isomorphism',
        'entropy_growth': [round(e, 4) for e in avg_entropy.tolist()],
        'fits': {
            'logarithmic_MERA': {'r_squared': round(r_squared, 4),
                                  'central_charge': round(c_central, 4)},
            'linear': {'r_squared': round(r2_linear, 4)},
            'power_law': {'r_squared': round(r2_power, 4),
                         'exponent': round(power_exp, 4)},
        },
        'area_law': {
            'peak_partition_pct': round(100 * peak_size / n_tokens, 1),
            'central_charge': round(c_area, 4),
        },
        'verdict': verdict,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q187_mera.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Entanglement entropy growth + fits
    ax = axes[0]
    ax.plot(layers, avg_entropy, 'ko-', linewidth=2, markersize=4, label='LLM data')
    ax.plot(layers, mera_fit, 'r--', linewidth=2,
            label='MERA/CFT (R^2=%.3f)' % r_squared)
    ax.plot(layers, linear_fit, 'b:', linewidth=2,
            label='Linear (R^2=%.3f)' % r2_linear)
    ax.set_xlabel('Layer (= holographic depth)')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('(a) Entropy Growth vs Layer\n(MERA: S ~ log(l), c=%.2f)' % c_central)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Per-prompt entropy profiles
    ax = axes[1]
    for i, (profile, prompt) in enumerate(zip(entropy_profiles, prompts)):
        ax.plot(layers, profile, '-', linewidth=1.5, alpha=0.7,
                label=prompt[:15] + '...')
    ax.plot(layers, avg_entropy, 'k-', linewidth=3, label='Average')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('(b) Individual Prompt Profiles\n(Universal shape?)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) Area law (partition scaling)
    ax = axes[2]
    ax.plot(partition_sizes, partition_entropies, 'o-', color='#E91E63',
            linewidth=2, markersize=4)
    ax.axvline(n_tokens / 2, color='green', ls='--',
               label='Half-system (Area law peak)')
    ax.axvline(peak_size, color='red', ls=':', label='Actual peak')
    ax.set_xlabel('Partition Size (tokens)')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('(c) Area Law (Ryu-Takayanagi)\n(Peak at %.0f%% vs expected 50%%)' %
                (100 * peak_size / n_tokens))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Q187: MERA Isomorphism\n'
                 'LLM as Holographic Universe (c=%.2f, R^2=%.3f)' %
                 (c_central, r_squared),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q187_mera.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ187 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
