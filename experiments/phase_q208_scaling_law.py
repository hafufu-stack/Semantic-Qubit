# -*- coding: utf-8 -*-
"""
Phase Q208: Quantum Advantage Scaling Law
==========================================
How does the LLM's VQE advantage scale with problem dimension?

Sweep Hamiltonian sizes from 2 to 64 and measure:
1. VQE error (mHa) vs dimension
2. Convergence steps vs dimension
3. Compare with random initialization scaling

This reveals whether the advantage grows, shrinks, or remains constant
as problems get larger -- crucial for practical utility.
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


def build_random_hamiltonian(dim, seed=42):
    """Build a random symmetric Hamiltonian of given dimension."""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim) * 0.3
    H = (H + H.T) / 2
    H[np.diag_indices_from(H)] = np.sort(rng.randn(dim)) * 0.5
    return H


def run_vqe_scaling(model, tok, device, H_np, dim, n_steps=200, lr=0.005):
    """Run Embedding VQE and return error and convergence."""
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    eigvals = np.linalg.eigh(H_np)[0]
    E_exact = eigvals[0]

    embed_layer = model.model.embed_tokens
    prompt = "ground state energy dim %d:" % dim
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)

    history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()
        history.append(float(E.detach()))

    error_mha = abs(history[-1] - E_exact) * 1000
    converged = next((i for i, e in enumerate(history)
                      if abs(e - E_exact) < 0.002), n_steps)

    return {
        'E_exact': float(E_exact),
        'E_final': round(history[-1], 6),
        'error_mHa': round(error_mha, 4),
        'converged_step': converged,
    }


def main():
    print("=" * 60)
    print("Phase Q208: Quantum Advantage Scaling Law")
    print("  (How does VQE advantage scale with dimension?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dimensions = [2, 4, 8, 16, 32, 64]
    n_trials = 3
    n_steps = 200

    all_results = []

    for dim in dimensions:
        print("\n--- Dimension %d ---" % dim)
        trial_results = []

        for trial in range(n_trials):
            H = build_random_hamiltonian(dim, seed=42 + trial * 100 + dim)
            r = run_vqe_scaling(model, tok, device, H, dim, n_steps=n_steps)
            trial_results.append(r)
            print("  Trial %d: error=%.4f mHa, conv@%d" %
                  (trial, r['error_mHa'], r['converged_step']))

        avg_error = np.mean([r['error_mHa'] for r in trial_results])
        avg_conv = np.mean([r['converged_step'] for r in trial_results])

        dim_result = {
            'dim': dim,
            'avg_error_mHa': round(avg_error, 4),
            'avg_converged_step': round(avg_conv, 1),
            'trials': trial_results,
        }
        all_results.append(dim_result)
        print("  Average: error=%.4f mHa, conv@%.0f" % (avg_error, avg_conv))

    # Fit scaling law: error ~ dim^alpha
    dims_arr = np.array([r['dim'] for r in all_results], dtype=float)
    errors_arr = np.array([r['avg_error_mHa'] for r in all_results])
    conv_arr = np.array([r['avg_converged_step'] for r in all_results])

    # Log-log fit for error scaling
    valid = errors_arr > 0
    if valid.sum() > 2:
        log_d = np.log(dims_arr[valid])
        log_e = np.log(errors_arr[valid])
        alpha_coeffs = np.polyfit(log_d, log_e, 1)
        alpha = alpha_coeffs[0]
        alpha_r2 = 1 - (np.sum((log_e - np.polyval(alpha_coeffs, log_d))**2) /
                         np.sum((log_e - log_e.mean())**2))
    else:
        alpha = 0
        alpha_r2 = 0

    # Convergence scaling
    if len(dims_arr) > 2:
        conv_coeffs = np.polyfit(np.log(dims_arr), np.log(conv_arr + 1), 1)
        beta = conv_coeffs[0]
    else:
        beta = 0

    print("\n--- Scaling Laws ---")
    print("  Error scaling: error ~ dim^%.2f (R2=%.3f)" % (alpha, alpha_r2))
    print("  Convergence scaling: steps ~ dim^%.2f" % beta)

    if alpha < 1.0:
        verdict = "FAVORABLE: error ~ dim^%.2f (sub-linear, R2=%.2f)" % (alpha, alpha_r2)
    elif alpha < 2.0:
        verdict = "LINEAR: error ~ dim^%.2f" % alpha
    else:
        verdict = "UNFAVORABLE: error ~ dim^%.2f (super-linear)" % alpha

    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q208',
        'name': 'Quantum Advantage Scaling Law',
        'dimensions': all_results,
        'scaling': {
            'error_exponent_alpha': round(alpha, 4),
            'error_r_squared': round(alpha_r2, 4),
            'convergence_exponent_beta': round(beta, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q208_scaling_law.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Error vs dimension (log-log)
    ax = axes[0]
    ax.loglog(dims_arr, errors_arr, 'o-', color='#E91E63', lw=2, ms=8)
    if valid.sum() > 2:
        fit_d = np.logspace(np.log10(dims_arr.min()), np.log10(dims_arr.max()), 50)
        fit_e = np.exp(np.polyval(alpha_coeffs, np.log(fit_d)))
        ax.loglog(fit_d, fit_e, '--', color='#2196F3',
                  label='Fit: dim^%.2f (R2=%.2f)' % (alpha, alpha_r2))
    ax.set_xlabel('Hamiltonian Dimension')
    ax.set_ylabel('VQE Error (mHa)')
    ax.set_title('(a) Error Scaling Law')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # (b) Convergence steps vs dimension
    ax = axes[1]
    ax.semilogx(dims_arr, conv_arr, 's-', color='#4CAF50', lw=2, ms=8)
    ax.set_xlabel('Hamiltonian Dimension')
    ax.set_ylabel('Convergence Steps')
    ax.set_title('(b) Convergence Scaling (beta=%.2f)' % beta)
    ax.grid(alpha=0.3)

    # (c) Error per dimension
    ax = axes[2]
    error_per_dim = errors_arr / dims_arr
    ax.bar(range(len(dims_arr)), error_per_dim,
           color='#FF9800', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(dims_arr)))
    ax.set_xticklabels([str(int(d)) for d in dims_arr])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Error per Dimension (mHa/dim)')
    ax.set_title('(c) Normalized Error')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q208: Quantum Advantage Scaling Law\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q208_scaling_law.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ208 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
