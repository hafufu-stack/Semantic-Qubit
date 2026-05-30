# -*- coding: utf-8 -*-
"""
Phase Q276: Ryu-Takayanagi Formula (Holographic Entanglement)
================================================================
The cornerstone of holographic duality: entanglement entropy
of a boundary region equals the area of the minimal surface
in the bulk. Test if LLM satisfies this holographic relation.
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

def main():
    print("=" * 60)
    print("Phase Q276: Ryu-Takayanagi Formula")
    print("  (Holographic entanglement entropy)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    prompts = [
        "holographic principle black hole information",
        "anti de sitter conformal field theory",
        "quantum gravity entanglement entropy",
        "bulk boundary correspondence duality",
    ]

    # Boundary = token positions, Bulk = layer depth
    # For each boundary interval [0, L], measure:
    # 1. Entanglement entropy of boundary region
    # 2. "Minimal surface area" = minimum layer-depth distance

    boundary_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    all_S = {sz: [] for sz in boundary_sizes}
    all_A = {sz: [] for sz in boundary_sizes}

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for sz in boundary_sizes:
            if sz > seq_len:
                continue

            # Boundary region: first sz tokens at final layer
            h_region = out.hidden_states[n_layers][0, :sz, :16].float().cpu().numpy()

            # Entanglement entropy of the boundary region
            # Use SVD of the region matrix
            if h_region.shape[0] > 1:
                U, singular_vals, Vt = np.linalg.svd(h_region, full_matrices=False)
                singular_vals = singular_vals / (np.sum(singular_vals) + 1e-10)
                sv_pos = singular_vals[singular_vals > 1e-12]
                S_EE = float(-np.sum(sv_pos * np.log2(sv_pos))) if len(sv_pos) > 0 else 0
            else:
                h = h_region[0] / (np.linalg.norm(h_region[0]) + 1e-10)
                rho = np.outer(h, h.conj())
                rho = 0.7 * rho + 0.3 * np.eye(16) / 16
                rho /= np.trace(rho)
                ev = np.real(np.linalg.eigvalsh(rho))
                ev_pos = ev[ev > 1e-12]
                S_EE = float(-np.sum(ev_pos * np.log2(ev_pos))) if len(ev_pos) > 0 else 0

            # Minimal surface area in the bulk
            # For a 1D boundary of length L, RT predicts S = (c/3) * log(L)
            # Minimal surface goes through the bulk (layers)
            # Proxy: find the layer where boundary region has minimum spread
            min_spread = float('inf')
            for li in range(0, n_layers + 1, 2):
                h_layer = out.hidden_states[li][0, :sz, :16].float().cpu().numpy()
                spread = float(np.std(h_layer))
                min_spread = min(min_spread, spread)

            all_S[sz].append(S_EE)
            all_A[sz].append(min_spread)

    # Average
    avg_S = {sz: round(np.mean(vals), 4) for sz, vals in all_S.items() if vals}
    avg_A = {sz: round(np.mean(vals), 4) for sz, vals in all_A.items() if vals}

    sizes = sorted(avg_S.keys())
    S_vals = [avg_S[s] for s in sizes]

    # Fit: S = a * log(L) + b (RT prediction for 1+1D CFT)
    if len(sizes) >= 3:
        log_sizes = np.log(sizes)
        try:
            a, b = np.polyfit(log_sizes, S_vals, 1)
        except:
            a, b = 0, 0
        # Central charge: c = 3a (for 1D)
        central_charge = 3 * a
    else:
        a, b, central_charge = 0, 0, 0

    # Correlation between S and A (holographic check)
    A_vals = [avg_A[s] for s in sizes if s in avg_A]
    if len(S_vals) == len(A_vals) and len(S_vals) >= 3:
        corr_SA = float(np.corrcoef(S_vals, A_vals)[0, 1])
    else:
        corr_SA = 0

    print("\n  S_EE vs boundary size:")
    for s, se in zip(sizes, S_vals):
        print("    L=%d: S_EE=%.4f bits" % (s, se))
    print("  log fit: a=%.3f (c=%.2f), corr(S,A)=%.3f" % (a, central_charge, corr_SA))

    if a > 0.1 and abs(corr_SA) > 0.5:
        verdict = "RT CONFIRMED: S ~ %.2f*log(L), c=%.1f, corr(S,A)=%.2f" % (a, central_charge, corr_SA)
    elif a > 0.1:
        verdict = "LOG SCALING: S ~ %.2f*log(L) but weak holographic corr=%.2f" % (a, corr_SA)
    else:
        verdict = "NO RT: S does not scale logarithmically (a=%.3f)" % a

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q276', 'name': 'Ryu-Takayanagi',
        'boundary_sizes': sizes, 'avg_entropy': S_vals,
        'summary': {'log_coeff': round(a, 3), 'central_charge': round(central_charge, 2),
                     'corr_SA': round(corr_SA, 3), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q276_ryu_takayanagi.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(sizes, S_vals, 'o-', color='#E91E63', lw=2, ms=8, label='S_EE data')
    if a > 0:
        x_fit = np.linspace(1, max(sizes), 50)
        ax.plot(x_fit, a * np.log(x_fit) + b, '--', color='blue', lw=2,
                label='Fit: %.2f*log(L)+%.2f' % (a, b))
    ax.set_xlabel('Boundary Size L'); ax.set_ylabel('Entanglement Entropy (bits)')
    ax.set_title('(a) RT: S vs log(L)'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.log(sizes), S_vals, 'o-', color='#2196F3', lw=2, ms=8)
    ax.set_xlabel('log(Boundary Size)'); ax.set_ylabel('S_EE (bits)')
    ax.set_title('(b) Log-Linear (slope = c/3 = %.2f)' % (a))
    ax.grid(alpha=0.3)

    plt.suptitle('Q276: Ryu-Takayanagi\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q276_ryu_takayanagi.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ276 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
