# -*- coding: utf-8 -*-
"""
Phase Q228: Quantum Advantage Scaling Law
============================================
Q226 showed 9/9 wins. But HOW does the advantage scale with dimension?
If VQE/SA error ratio grows exponentially -> quantum advantage.
If polynomial -> classical-like improvement.
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


def run_vqe(model, tok, device, H_np, dim, n_steps=200, lr=0.005):
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigh(H_np)[0][0])
    embed_layer = model.model.embed_tokens
    prompt = "ground state:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()
    error = abs(float(E.detach()) - E_exact) * 1000
    return max(error, 1e-6)


def run_sa(H_np, dim, n_steps=5000, seed=42):
    rng = np.random.RandomState(seed)
    E_exact = float(np.linalg.eigh(H_np)[0][0])
    psi = rng.randn(dim).astype(np.float64)
    psi /= np.linalg.norm(psi)
    E_best = float(psi @ H_np @ psi)
    for step in range(n_steps):
        T = max(0.01, 1.0 - step / n_steps)
        psi_new = psi + rng.randn(dim) * 0.1
        psi_new /= np.linalg.norm(psi_new)
        E_new = float(psi_new @ H_np @ psi_new)
        if E_new < E_best or rng.rand() < np.exp(-(E_new - E_best) / max(T, 1e-10)):
            psi = psi_new
            E_best = min(E_best, E_new)
    return max(abs(E_best - E_exact) * 1000, 1e-6)


def main():
    print("=" * 60)
    print("Phase Q228: Quantum Advantage Scaling")
    print("  (Does advantage grow exponentially?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dims = [2, 4, 8, 16, 32, 64]
    n_trials = 3

    all_results = []

    for dim in dims:
        print("\n--- dim=%d ---" % dim)
        rng = np.random.RandomState(42 + dim)
        H = rng.randn(dim, dim).astype(np.float32) * 0.3
        H = (H + H.T) / 2

        vqe_errors = []
        sa_errors = []
        for trial in range(n_trials):
            H_trial = H + rng.randn(dim, dim).astype(np.float32) * 0.01
            H_trial = (H_trial + H_trial.T) / 2

            vqe_err = run_vqe(model, tok, device, H_trial, dim, n_steps=200)
            sa_err = run_sa(H_trial, dim, n_steps=5000, seed=trial)
            vqe_errors.append(vqe_err)
            sa_errors.append(sa_err)

        avg_vqe = np.mean(vqe_errors)
        avg_sa = np.mean(sa_errors)
        ratio = avg_sa / avg_vqe

        print("  VQE=%.4f, SA=%.4f, ratio=%.1f" % (avg_vqe, avg_sa, ratio))

        all_results.append({
            'dim': dim,
            'avg_vqe_mHa': round(avg_vqe, 4),
            'avg_sa_mHa': round(avg_sa, 4),
            'advantage_ratio': round(ratio, 2),
        })

    # Fit scaling
    dims_arr = np.array([r['dim'] for r in all_results], dtype=float)
    ratios = np.array([r['advantage_ratio'] for r in all_results])

    valid = ratios > 0
    if valid.sum() > 2:
        log_dims = np.log(dims_arr[valid])
        log_ratios = np.log(ratios[valid])
        slope, intercept = np.polyfit(log_dims, log_ratios, 1)
        # Exponential fit
        exp_fit = np.polyfit(dims_arr[valid], log_ratios, 1)
        r2_poly = 1 - np.sum((log_ratios - np.polyval([slope, intercept], log_dims))**2) / np.sum((log_ratios - np.mean(log_ratios))**2)
        r2_exp = 1 - np.sum((log_ratios - np.polyval(exp_fit, dims_arr[valid]))**2) / np.sum((log_ratios - np.mean(log_ratios))**2)
    else:
        slope = 0
        r2_poly = 0
        r2_exp = 0

    if r2_exp > r2_poly and r2_exp > 0.5:
        scaling = "EXPONENTIAL"
    elif slope > 1:
        scaling = "SUPER-POLYNOMIAL (dim^%.1f)" % slope
    elif slope > 0:
        scaling = "POLYNOMIAL (dim^%.1f)" % slope
    else:
        scaling = "NO SCALING"

    verdict = "%s advantage scaling (R2_poly=%.3f, R2_exp=%.3f)" % (scaling, r2_poly, r2_exp)

    print("\n--- Summary ---")
    print("  Scaling exponent: %.2f" % slope)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q228',
        'name': 'Quantum Advantage Scaling',
        'dimensions': all_results,
        'summary': {
            'scaling_type': scaling,
            'exponent': round(slope, 4),
            'r2_polynomial': round(r2_poly, 4),
            'r2_exponential': round(r2_exp, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q228_advantage_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.loglog(dims_arr, [r['avg_vqe_mHa'] for r in all_results], 'o-', color='#E91E63', lw=2, label='S-Qubit VQE')
    ax.loglog(dims_arr, [r['avg_sa_mHa'] for r in all_results], 's--', color='#607D8B', lw=2, label='Simulated Annealing')
    ax.set_xlabel('Dimension'); ax.set_ylabel('Error (mHa)')
    ax.set_title('(a) Error Scaling'); ax.legend(); ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.semilogy(dims_arr, ratios, 'D-', color='#4CAF50', lw=2, ms=8)
    ax.set_xlabel('Dimension'); ax.set_ylabel('Advantage Ratio (SA/VQE)')
    ax.set_title('(b) Advantage Scaling (%s)' % scaling); ax.grid(alpha=0.3)

    plt.suptitle('Q228: Quantum Advantage Scaling\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q228_advantage_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ228 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
