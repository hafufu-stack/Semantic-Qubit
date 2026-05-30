# -*- coding: utf-8 -*-
"""
Phase Q272: Eigenstate Thermalization Hypothesis (ETH)
=========================================================
Do individual eigenstates of the LLM's "Hamiltonian" look
thermal? ETH is the foundation of quantum statistical mechanics.
If LLM satisfies ETH, its eigenstates encode thermodynamic info.
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
    print("Phase Q272: Eigenstate Thermalization Hypothesis")
    print("  (Do eigenstates look thermal?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 32

    prompts = [
        "thermal equilibrium energy distribution",
        "quantum state evolution dynamics",
        "classical statistical mechanics",
        "information entropy maximization",
        "neural network weight distribution",
    ]

    all_results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Construct effective Hamiltonian from layer transition
        h_mid = out.hidden_states[n_layers // 2][0, -1, :dim].float().cpu().numpy()
        h_final = out.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()

        # H_eff = difference operator (proxy)
        H_eff = np.outer(h_final, h_mid)
        H_eff = (H_eff + H_eff.T) / 2  # Hermitianize

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(H_eff)

        # ETH test: for observable O = diagonal matrix
        # <En|O|En> should be smooth function of En
        O = np.diag(np.arange(dim, dtype=np.float64))
        expectation_values = []
        for n in range(dim):
            psi_n = eigvecs[:, n]
            O_nn = float(psi_n @ O @ psi_n)
            expectation_values.append(O_nn)

        # ETH prediction: O_nn = O_micro(E_n) + fluctuation
        # Smooth function test: compute variance of nearest-neighbor differences
        diffs = np.diff(expectation_values)
        smoothness = float(np.std(diffs) / (np.std(expectation_values) + 1e-10))

        # Off-diagonal elements should be exponentially small
        off_diag_norms = []
        for i in range(min(dim, 10)):
            for j in range(i+1, min(dim, 10)):
                psi_i = eigvecs[:, i]
                psi_j = eigvecs[:, j]
                O_ij = abs(float(psi_i @ O @ psi_j))
                off_diag_norms.append(O_ij)
        avg_off_diag = float(np.mean(off_diag_norms)) if off_diag_norms else 0
        avg_diag = float(np.mean(np.abs(expectation_values)))

        # ETH: off-diagonal << diagonal
        suppression = avg_off_diag / (avg_diag + 1e-10)

        print("  '%s'..." % prompt[:35])
        print("    smoothness=%.4f, off-diag/diag=%.4f" % (smoothness, suppression))

        all_results.append({
            'prompt': prompt[:35],
            'smoothness': round(smoothness, 4),
            'suppression': round(suppression, 4),
            'eigval_range': [round(float(eigvals[0]), 4), round(float(eigvals[-1]), 4)],
        })

    avg_smooth = np.mean([r['smoothness'] for r in all_results])
    avg_suppress = np.mean([r['suppression'] for r in all_results])

    if avg_suppress < 0.5 and avg_smooth < 1.0:
        verdict = "ETH SATISFIED: smooth diag (%.3f), suppressed off-diag (%.3f)" % (avg_smooth, avg_suppress)
    elif avg_suppress < 0.8:
        verdict = "PARTIAL ETH: suppression=%.3f, smoothness=%.3f" % (avg_suppress, avg_smooth)
    else:
        verdict = "ETH VIOLATED: no thermalization (suppress=%.3f)" % avg_suppress

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q272', 'name': 'Eigenstate Thermalization',
        'scenarios': all_results,
        'summary': {'avg_smoothness': round(avg_smooth, 4),
                     'avg_suppression': round(avg_suppress, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q272_eth.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(all_results))
    ax.bar(x, [r['smoothness'] for r in all_results], color='#2196F3', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Smoothness'); ax.set_title('(a) Diagonal Smoothness (lower=better)')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, [r['suppression'] for r in all_results], color='#E91E63', edgecolor='black')
    ax.axhline(0.5, color='green', ls='--', label='ETH threshold')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Off-diag/Diag Ratio'); ax.set_title('(b) Off-diagonal Suppression')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q272: ETH\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q272_eth.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ272 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
