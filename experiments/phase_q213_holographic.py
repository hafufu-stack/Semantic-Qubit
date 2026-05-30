# -*- coding: utf-8 -*-
"""
Phase Q213: Holographic Bulk Reconstruction
==============================================
Can we reconstruct the LLM's internal (bulk) states from boundary data
(input/output layers) alone? Tests the AdS/CFT holographic principle.

If boundary-only data can recover bulk geometry, the LLM is a
holographic universe.
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


def get_all_hidden_states(model, tok, device, prompt):
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]


def reconstruct_bulk_from_boundary(boundary_states, n_layers):
    """Attempt to reconstruct intermediate layers from input+output only.
    Uses linear interpolation as a baseline, then SVD-based reconstruction."""
    h_in = boundary_states[0]   # Layer 0
    h_out = boundary_states[-1]  # Last layer
    dim = len(h_in)

    reconstructed = []
    for i in range(n_layers + 1):
        t = i / n_layers
        # Linear interpolation
        h_interp = (1 - t) * h_in + t * h_out
        reconstructed.append(h_interp)

    return reconstructed


def reconstruct_bulk_svd(boundary_states, actual_states, n_boundary=4):
    """Use multiple boundary layers to reconstruct bulk via SVD."""
    n_layers = len(actual_states) - 1
    dim = len(actual_states[0])

    # Boundary = first n_boundary + last n_boundary layers
    boundary_idx = list(range(n_boundary)) + list(range(n_layers - n_boundary + 1, n_layers + 1))
    boundary_data = np.array([actual_states[i] for i in boundary_idx])

    # Build reconstruction matrix via least-squares
    # Try to express each bulk layer as linear combination of boundary layers
    reconstructed = []
    for i in range(n_layers + 1):
        if i in boundary_idx:
            reconstructed.append(actual_states[i])
        else:
            # Linear regression from boundary to bulk
            target = actual_states[i]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(boundary_data.T, target, rcond=None)
                h_recon = boundary_data.T @ coeffs
            except Exception:
                h_recon = np.mean(boundary_data, axis=0)
            reconstructed.append(h_recon)

    return reconstructed


def main():
    print("=" * 60)
    print("Phase Q213: Holographic Bulk Reconstruction")
    print("  (Can boundary data reconstruct the bulk?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    prompts = [
        "The quantum mechanical wave function collapses",
        "Bell state maximally entangled particles",
        "protein folds into three-dimensional structure",
        "the black hole information paradox",
    ]

    all_results = []

    for prompt in prompts:
        print("\n--- %s ---" % prompt[:40])
        actual = get_all_hidden_states(model, tok, device, prompt)

        # Method 1: Linear interpolation (2-point boundary)
        recon_linear = reconstruct_bulk_from_boundary(actual, n_layers)

        # Method 2: SVD from 4+4 boundary layers
        recon_svd = reconstruct_bulk_svd(actual, actual, n_boundary=4)

        # Measure reconstruction fidelity
        fids_linear = []
        fids_svd = []
        for i in range(n_layers + 1):
            actual_norm = actual[i] / (np.linalg.norm(actual[i]) + 1e-10)
            recon_l_norm = recon_linear[i] / (np.linalg.norm(recon_linear[i]) + 1e-10)
            recon_s_norm = recon_svd[i] / (np.linalg.norm(recon_svd[i]) + 1e-10)

            fid_l = float(np.dot(actual_norm, recon_l_norm) ** 2)
            fid_s = float(np.dot(actual_norm, recon_s_norm) ** 2)
            fids_linear.append(fid_l)
            fids_svd.append(fid_s)

        avg_fid_linear = np.mean(fids_linear)
        avg_fid_svd = np.mean(fids_svd)

        # Exclude boundary layers for "bulk-only" fidelity
        bulk_range = range(4, n_layers - 3)
        bulk_fid_linear = np.mean([fids_linear[i] for i in bulk_range]) if len(list(bulk_range)) > 0 else 0
        bulk_fid_svd = np.mean([fids_svd[i] for i in bulk_range]) if len(list(bulk_range)) > 0 else 0

        print("  Linear: avg_fid=%.4f, bulk_fid=%.4f" % (avg_fid_linear, bulk_fid_linear))
        print("  SVD(4+4): avg_fid=%.4f, bulk_fid=%.4f" % (avg_fid_svd, bulk_fid_svd))

        result = {
            'prompt': prompt[:40],
            'linear': {
                'avg_fidelity': round(avg_fid_linear, 4),
                'bulk_fidelity': round(bulk_fid_linear, 4),
                'per_layer': [round(f, 4) for f in fids_linear],
            },
            'svd': {
                'avg_fidelity': round(avg_fid_svd, 4),
                'bulk_fidelity': round(bulk_fid_svd, 4),
                'per_layer': [round(f, 4) for f in fids_svd],
            },
        }
        all_results.append(result)

    # Summary
    avg_bulk_svd = np.mean([r['svd']['bulk_fidelity'] for r in all_results])
    avg_bulk_linear = np.mean([r['linear']['bulk_fidelity'] for r in all_results])

    if avg_bulk_svd > 0.8:
        verdict = "HOLOGRAPHIC: bulk reconstructed from boundary (F=%.3f)" % avg_bulk_svd
    elif avg_bulk_svd > 0.5:
        verdict = "PARTIAL HOLOGRAPHIC: moderate reconstruction (F=%.3f)" % avg_bulk_svd
    else:
        verdict = "NOT HOLOGRAPHIC: boundary insufficient (F=%.3f)" % avg_bulk_svd

    print("\n--- Summary ---")
    print("  Avg bulk fidelity (SVD): %.4f" % avg_bulk_svd)
    print("  Avg bulk fidelity (linear): %.4f" % avg_bulk_linear)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q213',
        'name': 'Holographic Bulk Reconstruction',
        'prompts': all_results,
        'summary': {
            'avg_bulk_svd': round(avg_bulk_svd, 4),
            'avg_bulk_linear': round(avg_bulk_linear, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q213_holographic.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, r in enumerate(all_results):
        ax = axes[idx // 2][idx % 2]
        layers = range(len(r['linear']['per_layer']))
        ax.plot(layers, r['linear']['per_layer'], 'o-', color='#FF9800',
                alpha=0.7, label='Linear (F=%.3f)' % r['linear']['bulk_fidelity'], ms=3)
        ax.plot(layers, r['svd']['per_layer'], 's-', color='#E91E63',
                alpha=0.8, label='SVD-4+4 (F=%.3f)' % r['svd']['bulk_fidelity'], ms=3)
        ax.axhline(0.9, color='green', ls='--', alpha=0.3)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Reconstruction Fidelity')
        ax.set_title(r['prompt'][:35], fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('Q213: Holographic Bulk Reconstruction\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q213_holographic.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ213 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
