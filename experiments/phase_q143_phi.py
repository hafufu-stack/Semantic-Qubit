# -*- coding: utf-8 -*-
"""
Phase Q143: Integrated Information Theory (Measuring Consciousness Phi)
=========================================================================
Tononi's IIT: consciousness = integrated information (Phi).
We measure Phi across LLM layers during different tasks.
Hypothesis: Phi spikes during "quantum" tasks (VQE, GBS)
vs simple text generation.
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


def compute_phi_approx(hidden_states, n_partitions=5):
    """Approximate integrated information Phi.

    Phi = min over all partitions of
          MI(whole) - sum(MI(parts))

    We approximate using:
    1. Mutual information via correlation matrices
    2. Random bipartitions of the hidden dimensions
    """
    # Stack hidden states: (n_layers, hidden_dim)
    H = np.array(hidden_states)  # (L, D)
    n_layers, dim = H.shape

    if n_layers < 2:
        return 0.0

    # Correlation matrix across layers
    C = np.corrcoef(H)  # (L, L)
    C = np.nan_to_num(C, nan=0.0)

    # Mutual information of the whole system (approximation)
    # MI ~ -0.5 * log(det(C)) for Gaussian
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.maximum(eigvals, 1e-10)
    mi_whole = -0.5 * np.sum(np.log(eigvals))

    # Find minimum over random bipartitions
    min_phi = float('inf')
    for _ in range(n_partitions):
        # Random bipartition of layers
        perm = np.random.permutation(n_layers)
        split = max(1, n_layers // 2)
        part_a = perm[:split]
        part_b = perm[split:]

        if len(part_a) < 2 or len(part_b) < 2:
            continue

        C_a = np.corrcoef(H[part_a])
        C_b = np.corrcoef(H[part_b])
        C_a = np.nan_to_num(C_a, nan=0.0)
        C_b = np.nan_to_num(C_b, nan=0.0)

        eig_a = np.maximum(np.linalg.eigvalsh(C_a), 1e-10)
        eig_b = np.maximum(np.linalg.eigvalsh(C_b), 1e-10)
        mi_a = -0.5 * np.sum(np.log(eig_a))
        mi_b = -0.5 * np.sum(np.log(eig_b))

        phi = mi_whole - (mi_a + mi_b)
        if phi < min_phi:
            min_phi = phi

    return max(float(min_phi), 0.0) if min_phi < float('inf') else 0.0


def compute_transfer_entropy(h_sequence):
    """Compute transfer entropy between consecutive layers.
    TE = information flow that can't be explained by self-history.
    """
    if len(h_sequence) < 3:
        return 0.0

    # Simplified: correlation between layer i and layer i+2
    # conditioned on layer i+1
    te_total = 0.0
    count = 0
    for i in range(len(h_sequence) - 2):
        h0 = h_sequence[i]
        h1 = h_sequence[i + 1]
        h2 = h_sequence[i + 2]

        # Partial correlation: corr(h0, h2 | h1)
        c01 = np.corrcoef(h0[:100], h1[:100])[0, 1]
        c02 = np.corrcoef(h0[:100], h2[:100])[0, 1]
        c12 = np.corrcoef(h1[:100], h2[:100])[0, 1]

        if not np.isnan(c01) and not np.isnan(c12) and abs(1 - c12**2) > 1e-10:
            partial = (c02 - c01 * c12) / np.sqrt((1 - c01**2) * (1 - c12**2) + 1e-10)
            te_total += abs(partial)
            count += 1

    return float(te_total / max(count, 1))


def main():
    print("=" * 60)
    print("Phase Q143: Integrated Information Theory")
    print("  (Measuring Consciousness Phi)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # Task categories: simple vs complex vs "quantum"
    tasks = [
        # Simple (low Phi expected)
        ('simple', 'The sky is', 'Color fact'),
        ('simple', 'One two three', 'Counting'),
        ('simple', 'Hello world', 'Greeting'),

        # Complex language (medium Phi)
        ('complex', 'The philosophical implications of consciousness', 'Philosophy'),
        ('complex', 'In quantum mechanics the wave function', 'QM description'),
        ('complex', 'The relationship between entropy and information', 'Info theory'),

        # "Quantum" tasks (high Phi hypothesized)
        ('quantum', 'Gaussian boson sampling interference pattern:', 'GBS (Q136)'),
        ('quantum', 'SU(3) gauge theory vacuum energy confinement:', 'LQCD (Q138)'),
        ('quantum', 'Hubbard model d-wave superconducting ground state:', 'Hubbard (Q140)'),
        ('quantum', 'Black hole scrambling Sachdev-Ye-Kitaev:', 'SYK (Q141)'),
        ('quantum', 'Topological anyon braiding non-Abelian statistics:', 'Anyon (Q142)'),
    ]

    all_results = []

    for category, prompt, task_name in tasks:
        print("\n--- [%s] %s ---" % (category.upper(), task_name))
        inp = tok(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Extract hidden states at last token position
        hidden_states = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            hidden_states.append(h)

        # Compute Phi
        phi = compute_phi_approx(hidden_states, n_partitions=20)

        # Compute transfer entropy
        te = compute_transfer_entropy(hidden_states)

        # Layer-wise information content (entropy of hidden state)
        layer_entropies = []
        for h in hidden_states:
            # Normalized histogram entropy
            h_norm = (h - h.mean()) / (h.std() + 1e-10)
            hist, _ = np.histogram(h_norm, bins=50, density=True)
            hist = hist[hist > 0]
            ent = -np.sum(hist * np.log(hist + 1e-10)) * (h_norm.max() - h_norm.min()) / 50
            layer_entropies.append(float(ent))

        # Cross-layer mutual information
        mi_pairs = []
        for i in range(0, n_layers, 4):
            for j in range(i + 4, n_layers + 1, 4):
                h_i = hidden_states[i][:200]
                h_j = hidden_states[j][:200]
                corr = abs(np.corrcoef(h_i, h_j)[0, 1])
                if not np.isnan(corr):
                    mi_pairs.append(float(corr))

        mean_mi = float(np.mean(mi_pairs)) if mi_pairs else 0.0

        # Output entropy
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        output_entropy = -(probs * (probs + 1e-10).log()).sum().item()

        result = {
            'category': category,
            'task': task_name,
            'prompt': prompt[:40],
            'phi': round(float(phi), 4),
            'transfer_entropy': round(float(te), 6),
            'mean_layer_entropy': round(float(np.mean(layer_entropies)), 4),
            'mean_cross_mi': round(float(mean_mi), 4),
            'output_entropy': round(float(output_entropy), 4),
        }
        all_results.append(result)
        print("  Phi=%.4f, TE=%.4f, H_layer=%.2f, MI=%.4f, H_out=%.1f" %
              (phi, te, np.mean(layer_entropies), mean_mi, output_entropy))

    # Summary by category
    print("\n--- Phi by Category ---")
    for cat in ['simple', 'complex', 'quantum']:
        phis = [r['phi'] for r in all_results if r['category'] == cat]
        tes = [r['transfer_entropy'] for r in all_results if r['category'] == cat]
        print("  %s: mean Phi=%.4f, mean TE=%.6f" %
              (cat, np.mean(phis), np.mean(tes)))

    # Save
    results = {
        'phase': 'Q143',
        'name': 'Integrated Information Theory (Phi)',
        'tasks': all_results,
        'summary': {
            'simple_phi': round(float(np.mean([r['phi'] for r in all_results
                                                if r['category'] == 'simple'])), 4),
            'complex_phi': round(float(np.mean([r['phi'] for r in all_results
                                                 if r['category'] == 'complex'])), 4),
            'quantum_phi': round(float(np.mean([r['phi'] for r in all_results
                                                 if r['category'] == 'quantum'])), 4),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q143_phi.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Phi by task
    ax = axes[0]
    colors_map = {'simple': '#2196F3', 'complex': '#FF9800', 'quantum': '#E91E63'}
    for i, r in enumerate(all_results):
        ax.bar(i, r['phi'], color=colors_map[r['category']], edgecolor='black',
               alpha=0.85)
    ax.set_xticks(range(len(all_results)))
    ax.set_xticklabels([r['task'][:12] for r in all_results],
                       rotation=45, fontsize=6, ha='right')
    ax.set_ylabel('Phi (integrated information)')
    ax.set_title('(a) Phi by Task Type')
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for l, c in colors_map.items()]
    ax.legend(handles=legend_elements, fontsize=7); ax.grid(alpha=0.3, axis='y')

    # (b) Transfer entropy
    ax = axes[1]
    for i, r in enumerate(all_results):
        ax.bar(i, r['transfer_entropy'], color=colors_map[r['category']],
               edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(all_results)))
    ax.set_xticklabels([r['task'][:12] for r in all_results],
                       rotation=45, fontsize=6, ha='right')
    ax.set_ylabel('Transfer Entropy')
    ax.set_title('(b) Information Flow Between Layers')
    ax.grid(alpha=0.3, axis='y')

    # (c) Phi vs output entropy (scatter)
    ax = axes[2]
    for cat, color in colors_map.items():
        cat_data = [r for r in all_results if r['category'] == cat]
        ax.scatter([r['output_entropy'] for r in cat_data],
                   [r['phi'] for r in cat_data],
                   color=color, s=100, label=cat, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Output Entropy (nats)')
    ax.set_ylabel('Phi (integrated information)')
    ax.set_title('(c) Phi vs Output Uncertainty\n(consciousness vs confusion)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q143: Measuring Consciousness (IIT Phi) in LLM',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q143_phi.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ143 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
