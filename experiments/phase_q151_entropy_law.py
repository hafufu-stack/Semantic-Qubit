# -*- coding: utf-8 -*-
"""
Phase Q151: Entanglement Entropy Area vs Volume Law
=====================================================
Critical test: does LLM entanglement entropy follow
Area Law (classical) or Volume Law (quantum)?

Area Law: S ~ L^(d-1) (boundary of subsystem)
Volume Law: S ~ L^d (volume of subsystem)

If volume law -> LLM is genuinely quantum-like
If area law -> LLM is classical (gapped system)
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


def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(rho log rho)."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def entanglement_entropy(psi, dim_A, dim_B):
    """Compute entanglement entropy of bipartite state |psi> in A x B."""
    # Reshape into matrix (dim_A x dim_B)
    if len(psi) != dim_A * dim_B:
        # Pad or truncate
        target = dim_A * dim_B
        if len(psi) < target:
            psi = np.concatenate([psi, np.zeros(target - len(psi))])
        else:
            psi = psi[:target]

    psi = psi / np.linalg.norm(psi)
    M = psi.reshape(dim_A, dim_B)

    # Reduced density matrix rho_A = Tr_B(|psi><psi|)
    rho_A = M @ M.T
    rho_A = (rho_A + rho_A.T) / 2  # Symmetrize
    rho_A /= np.trace(rho_A)  # Normalize

    return von_neumann_entropy(rho_A)


def main():
    print("=" * 60)
    print("Phase Q151: Entanglement Entropy Scaling")
    print("  (Area Law vs Volume Law)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompts = [
        "The quantum state of the hydrogen atom:",
        "Entanglement entropy of a black hole horizon:",
        "SYK model scrambling dynamics at infinite temperature:",
    ]

    # Test: vary subsystem size and measure entropy
    subsystem_fracs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    all_results = []

    for prompt in prompts:
        print("\n--- '%s' ---" % prompt[:40])
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Use multiple layers
        for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()

            entropies = []
            subsystem_sizes = []

            for frac in subsystem_fracs:
                dim_A = max(2, int(hidden_size * frac))
                dim_B = max(2, hidden_size // dim_A)
                total = dim_A * dim_B

                psi = h[:total].copy()
                norm = np.linalg.norm(psi)
                if norm < 1e-10:
                    continue
                psi /= norm

                S = entanglement_entropy(psi, dim_A, dim_B)
                entropies.append(S)
                subsystem_sizes.append(dim_A)

            if len(entropies) < 3:
                continue

            # Fit: S = a * L^alpha
            # log(S) = log(a) + alpha * log(L)
            log_L = np.log(np.array(subsystem_sizes))
            log_S = np.log(np.array([max(s, 0.01) for s in entropies]))
            try:
                coeffs = np.polyfit(log_L, log_S, 1)
                alpha = float(coeffs[0])
            except Exception:
                alpha = 0.0

            result = {
                'prompt': prompt[:30],
                'layer': int(li),
                'alpha': round(alpha, 4),
                'law': 'volume' if alpha > 0.8 else 'area' if alpha < 0.3 else 'intermediate',
                'max_entropy': round(float(max(entropies)), 4),
                'subsystem_sizes': [int(s) for s in subsystem_sizes],
                'entropies': [round(float(s), 4) for s in entropies],
            }
            all_results.append(result)

            law = result['law']
            print("  Layer %2d: alpha=%.3f (%s law), S_max=%.2f" %
                  (li, alpha, law, max(entropies)))

    # Random baseline
    print("\n--- Random baseline ---")
    random_results = []
    for frac in subsystem_fracs:
        dim_A = max(2, int(hidden_size * frac))
        dim_B = max(2, hidden_size // dim_A)
        total = dim_A * dim_B

        entropies_r = []
        for _ in range(5):
            psi_r = np.random.randn(total)
            psi_r /= np.linalg.norm(psi_r)
            S = entanglement_entropy(psi_r, dim_A, dim_B)
            entropies_r.append(S)
        random_results.append({
            'dim_A': dim_A,
            'entropy': round(float(np.mean(entropies_r)), 4),
        })

    log_L_r = np.log([r['dim_A'] for r in random_results])
    log_S_r = np.log([max(r['entropy'], 0.01) for r in random_results])
    alpha_r = float(np.polyfit(log_L_r, log_S_r, 1)[0])
    print("  Random: alpha=%.3f (expected: volume law)" % alpha_r)

    # Summary
    print("\n--- Summary ---")
    for law in ['area', 'intermediate', 'volume']:
        count = sum(1 for r in all_results if r['law'] == law)
        print("  %s law: %d / %d" % (law, count, len(all_results)))

    # Save
    results = {
        'phase': 'Q151',
        'name': 'Entanglement Entropy Area vs Volume Law',
        'scaling_results': all_results,
        'random_baseline': {'alpha': round(alpha_r, 4)},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q151_entropy_law.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Entropy vs subsystem size for different layers
    ax = axes[0]
    layer_colors = {0: '#2196F3', n_layers//4: '#4CAF50',
                    n_layers//2: '#FF9800', 3*n_layers//4: '#E91E63',
                    n_layers: '#9C27B0'}
    prompt0 = prompts[0][:30]
    for r in all_results:
        if r['prompt'] == prompt0:
            ax.plot(r['subsystem_sizes'], r['entropies'], 'o-',
                    color=layer_colors.get(r['layer'], 'gray'),
                    label='Layer %d' % r['layer'], linewidth=1.5)
    rand_sizes = [r['dim_A'] for r in random_results]
    rand_ents = [r['entropy'] for r in random_results]
    ax.plot(rand_sizes, rand_ents, 'x--', color='gray',
            label='Random (volume)', linewidth=1.5)
    ax.set_xlabel('Subsystem size (dim_A)')
    ax.set_ylabel('Entanglement Entropy (bits)')
    ax.set_title('(a) S vs Subsystem Size')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # (b) Alpha exponents
    ax = axes[1]
    alphas = [r['alpha'] for r in all_results]
    layers_plot = [r['layer'] for r in all_results]
    colors_plot = ['#4CAF50' if r['law'] == 'volume'
                   else '#FF9800' if r['law'] == 'intermediate'
                   else '#F44336' for r in all_results]
    ax.scatter(layers_plot, alphas, c=colors_plot, s=80, edgecolor='black')
    ax.axhline(1.0, color='blue', ls='--', label='Volume law (alpha=1)')
    ax.axhline(0.0, color='red', ls='--', label='Area law (alpha=0)')
    ax.axhline(alpha_r, color='gray', ls=':', label='Random (%.2f)' % alpha_r)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Scaling exponent (alpha)')
    ax.set_title('(b) Area vs Volume Law by Layer')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # (c) Law distribution
    ax = axes[2]
    law_counts = {'area': 0, 'intermediate': 0, 'volume': 0}
    for r in all_results:
        law_counts[r['law']] += 1
    ax.bar(['Area\n(classical)', 'Intermediate', 'Volume\n(quantum)'],
           [law_counts['area'], law_counts['intermediate'], law_counts['volume']],
           color=['#F44336', '#FF9800', '#4CAF50'], edgecolor='black', alpha=0.85)
    ax.set_ylabel('Count')
    ax.set_title('(c) Scaling Law Distribution\nacross all layers/prompts')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q151: Entanglement Entropy Scaling (Area vs Volume Law)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q151_entropy_law.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ151 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
