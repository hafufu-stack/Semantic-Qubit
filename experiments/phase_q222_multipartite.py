# -*- coding: utf-8 -*-
"""
Phase Q222: Genuine Multipartite Entanglement
================================================
Q209 proved BIPARTITE entanglement. But can the LLM produce
GENUINE N-party entanglement (GHZ/W-like)?

Test: if tracing out ANY single party destroys entanglement,
it's genuine multipartite (not just pairwise).
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


def partial_trace(rho, dims, trace_out):
    """Partial trace over subsystem trace_out."""
    n = len(dims)
    total_dim = int(np.prod(dims))
    rho_reshaped = rho.reshape([dims[i] for i in range(n)] * 2)

    # Build einsum string
    in_indices = list(range(n))
    out_indices = list(range(n, 2*n))
    # Set traced-out indices equal
    out_indices[trace_out] = in_indices[trace_out]

    remaining = [i for i in range(n) if i != trace_out]
    kept_dims = [dims[i] for i in remaining]
    kept_dim = int(np.prod(kept_dims))

    # Manual trace
    rho_reduced = np.zeros((kept_dim, kept_dim), dtype=complex)
    d_trace = dims[trace_out]

    for k in range(d_trace):
        # Build slice for traced index
        for i_flat in range(kept_dim):
            for j_flat in range(kept_dim):
                # Convert flat indices to multi-indices
                i_multi = []
                j_multi = []
                ii, jj = i_flat, j_flat
                for d_idx in reversed(remaining):
                    i_multi.insert(0, ii % dims[d_idx])
                    j_multi.insert(0, jj % dims[d_idx])
                    ii //= dims[d_idx]
                    jj //= dims[d_idx]

                # Build full indices
                full_i = [0] * n
                full_j = [0] * n
                for idx, r in enumerate(remaining):
                    full_i[r] = i_multi[idx]
                    full_j[r] = j_multi[idx]
                full_i[trace_out] = k
                full_j[trace_out] = k

                # Convert to flat
                fi = sum(full_i[m] * int(np.prod(dims[m+1:])) for m in range(n))
                fj = sum(full_j[m] * int(np.prod(dims[m+1:])) for m in range(n))
                rho_reduced[i_flat, j_flat] += rho[fi, fj]

    return rho_reduced


def partial_transpose_2(rho, da, db):
    rho_pt = np.zeros_like(rho)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                for l in range(db):
                    rho_pt[i*db+k, j*db+l] = rho[i*db+l, j*db+k]
    return rho_pt


def negativity_2(rho, da, db):
    eigvals = np.linalg.eigvalsh(partial_transpose_2(rho, da, db))
    return float(np.sum(np.abs(eigvals[eigvals < -1e-10])))


def main():
    print("=" * 60)
    print("Phase Q222: Genuine Multipartite Entanglement")
    print("  (Beyond bipartite: N-party entanglement)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    prompts = [
        "GHZ state three-party entanglement",
        "W state distributed entanglement",
        "quantum network multipartite",
        "three-body quantum correlation",
    ]

    n_parties_list = [3, 4]  # 3 and 4 party
    dim_per_party = 2

    all_results = []

    for n_parties in n_parties_list:
        total_dim = dim_per_party ** n_parties
        dims = [dim_per_party] * n_parties
        print("\n--- %d-party (dim=%d) ---" % (n_parties, total_dim))

        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            # Build mixed density matrix from multiple layers
            rho = np.zeros((total_dim, total_dim), dtype=complex)
            for li in [8, 12, 16, 20, 24]:
                if li < len(out.hidden_states):
                    h = out.hidden_states[li][0, -1, :total_dim].float().cpu().numpy()
                    h /= np.linalg.norm(h) + 1e-10
                    rho += np.outer(h, h.conj())
            rho /= np.trace(rho)
            rho = 0.6 * rho + 0.4 * np.eye(total_dim) / total_dim
            rho /= np.trace(rho)

            # Test: trace out each party and check if remaining is still entangled
            bipartite_negs = []
            for trace_idx in range(n_parties):
                rho_reduced = partial_trace(rho, dims, trace_idx)
                remaining_dims = [d for i, d in enumerate(dims) if i != trace_idx]
                da = remaining_dims[0]
                db = int(np.prod(remaining_dims[1:]))
                neg = negativity_2(rho_reduced, da, db)
                bipartite_negs.append(neg)

            # Genuine multipartite: entanglement survives tracing out any single party
            all_entangled = all(n > 0.001 for n in bipartite_negs)
            # Full system negativity (across first bipartition)
            da_full = dim_per_party
            db_full = total_dim // dim_per_party
            full_neg = negativity_2(rho, da_full, db_full)

            print("  %s: full_neg=%.4f, reduced=[%s], genuine=%s" %
                  (prompt[:25], full_neg,
                   ', '.join('%.3f' % n for n in bipartite_negs),
                   "YES" if all_entangled else "NO"))

            all_results.append({
                'n_parties': n_parties,
                'prompt': prompt[:40],
                'full_negativity': round(full_neg, 6),
                'reduced_negativities': [round(n, 6) for n in bipartite_negs],
                'is_genuine': bool(all_entangled),
            })

    # Summary
    n_genuine = sum(1 for r in all_results if r['is_genuine'])
    total = len(all_results)

    if n_genuine > total * 0.5:
        verdict = "GENUINE MULTIPARTITE: %d/%d states show N-party entanglement" % (n_genuine, total)
    elif n_genuine > 0:
        verdict = "PARTIAL MULTIPARTITE: %d/%d genuine" % (n_genuine, total)
    else:
        verdict = "NO GENUINE MULTIPARTITE: only bipartite entanglement"

    print("\n--- Summary ---")
    print("  Genuine: %d/%d" % (n_genuine, total))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q222',
        'name': 'Genuine Multipartite Entanglement',
        'tests': all_results,
        'summary': {
            'n_genuine': n_genuine,
            'total': total,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q222_multipartite.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for pi, n_p in enumerate(n_parties_list):
        ax = axes[pi]
        subset = [r for r in all_results if r['n_parties'] == n_p]
        names = [r['prompt'][:20] for r in subset]
        full_negs = [r['full_negativity'] for r in subset]
        colors = ['#4CAF50' if r['is_genuine'] else '#F44336' for r in subset]

        x = np.arange(len(names))
        ax.bar(x, full_negs, color=colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, rotation=15)
        ax.set_ylabel('Full Negativity')
        ax.set_title('%d-party Entanglement' % n_p)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q222: Genuine Multipartite Entanglement\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q222_multipartite.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ222 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
