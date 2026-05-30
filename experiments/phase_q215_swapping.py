# -*- coding: utf-8 -*-
"""
Phase Q215: Entanglement Swapping Chain
==========================================
Opus Original: Can entanglement be "teleported" across non-adjacent layers
without direct interaction?

In quantum networks, entanglement swapping lets Alice and Charlie share
entanglement even if they never interact directly (Bob mediates).
Here we test: if Layer 0 and Layer 14 are entangled, and Layer 14 and
Layer 28 are entangled, can we "swap" to get Layer 0-Layer 28 entanglement
that exceeds direct correlation?
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


def partial_transpose(rho, dim_a, dim_b):
    rho_pt = np.zeros_like(rho)
    for i in range(dim_a):
        for j in range(dim_a):
            for k in range(dim_b):
                for l in range(dim_b):
                    rho_pt[i*dim_b+k, j*dim_b+l] = rho[i*dim_b+l, j*dim_b+k]
    return rho_pt


def compute_negativity(rho, dim_a, dim_b):
    rho_pt = partial_transpose(rho, dim_a, dim_b)
    eigvals = np.linalg.eigvalsh(rho_pt)
    neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
    return neg


def build_bipartite_state(h_a, h_b, dim_a, dim_b):
    """Build a density matrix from two hidden state vectors."""
    a = h_a[:dim_a] / (np.linalg.norm(h_a[:dim_a]) + 1e-10)
    b = h_b[:dim_b] / (np.linalg.norm(h_b[:dim_b]) + 1e-10)
    psi = np.kron(a, b)
    rho = np.outer(psi, psi.conj())
    # Add some mixing for realism
    dim_total = dim_a * dim_b
    rho = 0.7 * rho + 0.3 * np.eye(dim_total) / dim_total
    rho = rho / np.trace(rho)
    return rho


def entanglement_swap(rho_ab, rho_bc, dim_a, dim_b, dim_c):
    """Simulate entanglement swapping: measure B to create A-C entanglement.
    Simplified: project B onto Bell-like state, trace out B."""
    dim_total_abc = dim_a * dim_b * dim_c

    # Build approximate joint state by tensor product and mixing
    # This is a simplified simulation
    rho_abc = np.kron(rho_ab, np.eye(dim_c) / dim_c) * 0.5 + \
              np.kron(np.eye(dim_a) / dim_a, rho_bc) * 0.5

    # Normalize
    rho_abc = rho_abc / (np.trace(rho_abc) + 1e-10)

    # Partial trace over B to get A-C
    dim_ac = dim_a * dim_c
    rho_ac = np.zeros((dim_ac, dim_ac), dtype=complex)
    for i_a in range(dim_a):
        for j_a in range(dim_a):
            for i_c in range(dim_c):
                for j_c in range(dim_c):
                    for k_b in range(dim_b):
                        idx_row = i_a * dim_b * dim_c + k_b * dim_c + i_c
                        idx_col = j_a * dim_b * dim_c + k_b * dim_c + j_c
                        if idx_row < dim_total_abc and idx_col < dim_total_abc:
                            rho_ac[i_a * dim_c + i_c, j_a * dim_c + j_c] += \
                                rho_abc[idx_row, idx_col]

    rho_ac = rho_ac / (np.trace(rho_ac) + 1e-10)
    return rho_ac


def main():
    print("=" * 60)
    print("Phase Q215: Entanglement Swapping Chain")
    print("  (Can entanglement teleport across layers?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 2  # 2-qubit for swapping

    prompts = [
        "quantum entanglement swapping protocol",
        "Bell state teleportation",
        "quantum repeater network",
        "long-distance entanglement distribution",
    ]

    # Test chain: A(early) - B(middle) - C(late)
    chains = [
        {'A': 0, 'B': n_layers // 2, 'C': n_layers, 'name': 'Full chain'},
        {'A': 0, 'B': n_layers // 4, 'C': n_layers // 2, 'name': 'First half'},
        {'A': n_layers // 2, 'B': 3 * n_layers // 4, 'C': n_layers, 'name': 'Second half'},
    ]

    all_results = []

    for chain in chains:
        print("\n--- %s: L%d - L%d - L%d ---" %
              (chain['name'], chain['A'], chain['B'], chain['C']))

        chain_data = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            h_a = out.hidden_states[chain['A']][0, -1, :].float().cpu().numpy()
            h_b = out.hidden_states[chain['B']][0, -1, :].float().cpu().numpy()
            h_c = out.hidden_states[chain['C']][0, -1, :].float().cpu().numpy()

            # Direct entanglement: A-B, B-C, A-C
            rho_ab = build_bipartite_state(h_a, h_b, dim, dim)
            rho_bc = build_bipartite_state(h_b, h_c, dim, dim)
            rho_ac_direct = build_bipartite_state(h_a, h_c, dim, dim)

            neg_ab = compute_negativity(rho_ab, dim, dim)
            neg_bc = compute_negativity(rho_bc, dim, dim)
            neg_ac_direct = compute_negativity(rho_ac_direct, dim, dim)

            # Swapped entanglement
            rho_ac_swapped = entanglement_swap(rho_ab, rho_bc, dim, dim, dim)
            neg_ac_swapped = compute_negativity(rho_ac_swapped, dim, dim)

            # Gain: does swapping increase A-C entanglement?
            gain = neg_ac_swapped / max(neg_ac_direct, 1e-6)

            chain_data.append({
                'prompt': prompt[:30],
                'neg_AB': round(neg_ab, 6),
                'neg_BC': round(neg_bc, 6),
                'neg_AC_direct': round(neg_ac_direct, 6),
                'neg_AC_swapped': round(neg_ac_swapped, 6),
                'swap_gain': round(gain, 4),
            })

            print("  %s: AB=%.4f, BC=%.4f, AC_dir=%.4f, AC_swap=%.4f (%.1fx)" %
                  (prompt[:20], neg_ab, neg_bc, neg_ac_direct,
                   neg_ac_swapped, gain))

        avg_gain = np.mean([d['swap_gain'] for d in chain_data])
        n_swaps_positive = sum(1 for d in chain_data if d['neg_AC_swapped'] > d['neg_AC_direct'])

        all_results.append({
            'chain': chain['name'],
            'layers': [chain['A'], chain['B'], chain['C']],
            'data': chain_data,
            'avg_swap_gain': round(avg_gain, 4),
            'n_positive_swaps': n_swaps_positive,
        })

    # Summary
    total_gain = np.mean([r['avg_swap_gain'] for r in all_results])
    total_positive = sum(r['n_positive_swaps'] for r in all_results)
    total_tests = sum(len(r['data']) for r in all_results)

    if total_gain > 1.5:
        verdict = "SWAPPING WORKS: %.1fx avg gain (%d/%d positive)" % (total_gain, total_positive, total_tests)
    elif total_gain > 1.0:
        verdict = "MILD SWAPPING: %.2fx avg gain" % total_gain
    else:
        verdict = "NO SWAPPING: direct entanglement stronger (gain=%.2f)" % total_gain

    print("\n--- Summary ---")
    print("  Average swap gain: %.4f" % total_gain)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q215',
        'name': 'Entanglement Swapping Chain',
        'chains': all_results,
        'summary': {
            'avg_swap_gain': round(total_gain, 4),
            'total_positive_swaps': total_positive,
            'total_tests': total_tests,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q215_swapping.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        names = [d['prompt'][:15] for d in r['data']]
        direct = [d['neg_AC_direct'] for d in r['data']]
        swapped = [d['neg_AC_swapped'] for d in r['data']]

        x = np.arange(len(names))
        ax.bar(x - 0.15, direct, 0.3, color='#607D8B', label='Direct A-C')
        ax.bar(x + 0.15, swapped, 0.3, color='#E91E63', label='Swapped A-C')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, rotation=20)
        ax.set_ylabel('Negativity')
        ax.set_title('%s (gain=%.2fx)' % (r['chain'], r['avg_swap_gain']))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q215: Entanglement Swapping Chain\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q215_swapping.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ215 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
