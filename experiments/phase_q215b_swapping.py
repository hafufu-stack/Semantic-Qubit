# -*- coding: utf-8 -*-
"""
Phase Q215b: Entanglement Swapping (Fixed)
============================================
Q215 was zero because pure product states can't be entangled.
Fix: use mixed density matrices (same approach as Q209/Q210)
with proper multi-layer averaging.
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


def partial_transpose(rho, da, db):
    rho_pt = np.zeros_like(rho)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                for l in range(db):
                    rho_pt[i*db+k, j*db+l] = rho[i*db+l, j*db+k]
    return rho_pt


def negativity(rho, da, db):
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
    return float(np.sum(np.abs(eigvals[eigvals < -1e-10])))


def build_bipartite_mixed(hidden_states_list, layer_a, layer_b, dim=4):
    """Build mixed bipartite state from multiple hidden states."""
    da = db = dim
    dt = da * db
    rho = np.zeros((dt, dt), dtype=complex)

    for hs in hidden_states_list:
        ha = hs[layer_a][:dim].astype(np.float64)
        hb = hs[layer_b][:dim].astype(np.float64)
        ha /= (np.linalg.norm(ha) + 1e-10)
        hb /= (np.linalg.norm(hb) + 1e-10)
        psi = np.kron(ha, hb)
        rho += np.outer(psi, psi.conj())

    rho /= len(hidden_states_list)
    # Mix with identity for realism
    rho = 0.6 * rho + 0.4 * np.eye(dt) / dt
    rho /= np.trace(rho)
    return rho


def main():
    print("=" * 60)
    print("Phase Q215b: Entanglement Swapping (Fixed)")
    print("  (Mixed states for proper entanglement)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 4  # larger subsystem

    prompts = [
        "quantum entanglement swapping", "Bell state teleportation",
        "quantum repeater", "long-distance entanglement",
        "EPR pair distribution", "quantum network routing",
        "photon pair generation", "quantum key distribution",
    ]

    # Gather hidden states
    all_hs = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]
        all_hs.append(hs)

    # Chain tests
    chains = [
        (0, n_layers // 2, n_layers, 'Full'),
        (0, n_layers // 4, n_layers // 2, 'Early'),
        (n_layers // 4, n_layers // 2, 3 * n_layers // 4, 'Middle'),
        (n_layers // 2, 3 * n_layers // 4, n_layers, 'Late'),
    ]

    all_results = []
    for la, lb, lc, name in chains:
        print("\n--- %s: L%d-L%d-L%d ---" % (name, la, lb, lc))

        rho_ab = build_bipartite_mixed(all_hs, la, lb, dim)
        rho_bc = build_bipartite_mixed(all_hs, lb, lc, dim)
        rho_ac = build_bipartite_mixed(all_hs, la, lc, dim)

        neg_ab = negativity(rho_ab, dim, dim)
        neg_bc = negativity(rho_bc, dim, dim)
        neg_ac = negativity(rho_ac, dim, dim)

        # Geometric mean as "swapped" prediction
        neg_swap_pred = np.sqrt(neg_ab * neg_bc) if neg_ab > 0 and neg_bc > 0 else 0
        gain = neg_ac / max(neg_swap_pred, 1e-6) if neg_swap_pred > 0 else 0

        print("  AB=%.4f, BC=%.4f, AC=%.4f, swap_pred=%.4f, gain=%.2f" %
              (neg_ab, neg_bc, neg_ac, neg_swap_pred, gain))

        all_results.append({
            'chain': name, 'layers': [la, lb, lc],
            'neg_AB': round(neg_ab, 6), 'neg_BC': round(neg_bc, 6),
            'neg_AC': round(neg_ac, 6), 'neg_swap_pred': round(neg_swap_pred, 6),
            'gain': round(gain, 4),
        })

    # Monogamy test: does A-B + B-C entanglement limit A-C?
    # CKW inequality: N(A|BC) >= N(A|B) + N(A|C) would violate monogamy
    monogamy_violations = sum(1 for r in all_results
                              if r['neg_AC'] > r['neg_AB'] + r['neg_BC'] + 0.001)

    avg_gain = np.mean([r['gain'] for r in all_results if r['gain'] > 0])
    if avg_gain != avg_gain:  # NaN check
        avg_gain = 0

    if avg_gain > 1.5:
        verdict = "SWAPPING ENHANCED: AC > sqrt(AB*BC) by %.1fx" % avg_gain
    elif any(r['neg_AC'] > 0.01 for r in all_results):
        verdict = "TRANSITIVE ENTANGLEMENT: AC entangled through B"
    else:
        verdict = "NO TRANSITIVE ENTANGLEMENT"

    print("\n--- Summary ---")
    print("  Monogamy violations: %d/%d" % (monogamy_violations, len(all_results)))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q215b',
        'name': 'Entanglement Swapping (Fixed)',
        'chains': all_results,
        'summary': {
            'avg_gain': round(avg_gain, 4) if avg_gain == avg_gain else 0,
            'monogamy_violations': monogamy_violations,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q215b_swapping.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    names = [r['chain'] for r in all_results]
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - w, [r['neg_AB'] for r in all_results], w, label='A-B', color='#2196F3')
    ax.bar(x, [r['neg_BC'] for r in all_results], w, label='B-C', color='#FF9800')
    ax.bar(x + w, [r['neg_AC'] for r in all_results], w, label='A-C (direct)', color='#E91E63')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Negativity')
    ax.set_title('Q215b: Entanglement Swapping (Fixed)\n%s' % verdict[:60])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q215b_swapping.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ215b complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
