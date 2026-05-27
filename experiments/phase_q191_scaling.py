# -*- coding: utf-8 -*-
"""
Phase Q191: VQE Scaling Analysis
===================================
How does Embedding VQE error scale with Hamiltonian dimension?

Critical question for quantum advantage:
- Physical QC: error grows EXPONENTIALLY with qubits (Barren Plateau)
- LLM VQE: Q177 showed polynomial scaling (dim^-0.44)
- Q191: Formally verify this across dim=2,4,8,16,32,64

If error stays polynomial -> LLM can solve problems physical QC cannot.
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
    print("Phase Q191: VQE Scaling Analysis")
    print("  (Error vs Dimension: Polynomial or Exponential?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    hidden_size = model.config.hidden_size
    embed_layer = model.model.embed_tokens

    dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
    # Filter to dims that fit in hidden_size
    dimensions = [d for d in dimensions if d <= hidden_size]

    n_trials = 10
    n_steps = 200

    all_results = []

    for dim in dimensions:
        print("\n--- Dimension: %d (= %d qubits) ---" % (dim, int(np.log2(dim))))

        errors = []
        convergence_steps = []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 100 + dim)

            # Random symmetric Hamiltonian
            A = rng.randn(dim, dim).astype(np.float64)
            H = (A + A.T) / 2 / np.sqrt(dim)
            E_exact = float(np.linalg.eigvalsh(H)[0])
            H_torch = torch.tensor(H, dtype=torch.float32, device=device)

            # Embedding VQE
            seed_prompt = "Ground state dim %d trial %d:" % (dim, trial)
            seed_ids = tok(seed_prompt, return_tensors='pt')['input_ids'].to(device)
            seed_embeds = embed_layer(seed_ids).detach().clone()

            opt = seed_embeds.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([opt], lr=0.003)

            conv_step = n_steps
            best_error = float('inf')

            for step in range(n_steps):
                optimizer.zero_grad()
                outputs = model(inputs_embeds=opt.float(),
                               output_hidden_states=True)
                h = outputs.hidden_states[-1][0, -1, :]
                psi = h[:dim]
                psi_n = psi / (torch.norm(psi) + 1e-10)
                E = psi_n @ H_torch @ psi_n
                E.backward()
                optimizer.step()

                err = abs(float(E.detach()) - E_exact) * 1000
                if err < best_error:
                    best_error = err
                if err < 1.6 and conv_step == n_steps:
                    conv_step = step

            errors.append(best_error)
            convergence_steps.append(conv_step)

        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        chem_acc = sum(1 for e in errors if e < 1.6)
        mean_conv = float(np.mean(convergence_steps))

        result = {
            'dim': dim,
            'qubits': int(np.log2(dim)),
            'mean_error_mHa': round(mean_err, 2),
            'median_error_mHa': round(median_err, 2),
            'chem_accuracy_pct': round(100 * chem_acc / n_trials, 1),
            'mean_conv_steps': round(mean_conv, 1),
            'all_errors': [round(e, 2) for e in errors],
        }
        all_results.append(result)

        print("  dim=%d: mean=%.2f, median=%.2f mHa, chem_acc=%d/%d, conv@%.0f" %
              (dim, mean_err, median_err, chem_acc, n_trials, mean_conv))

    # === Scaling Analysis ===
    print("\n--- Scaling Analysis ---")

    dims = np.array([r['dim'] for r in all_results])
    mean_errors = np.array([r['mean_error_mHa'] for r in all_results])
    median_errors = np.array([r['median_error_mHa'] for r in all_results])

    # Fit 1: Power law (polynomial) - err ~ dim^alpha
    valid = mean_errors > 0
    if np.sum(valid) > 2:
        log_dims = np.log(dims[valid])
        log_errs = np.log(mean_errors[valid] + 1e-10)
        poly_coeffs = np.polyfit(log_dims, log_errs, 1)
        alpha = poly_coeffs[0]
        poly_r2 = 1 - np.var(log_errs - np.polyval(poly_coeffs, log_dims)) / np.var(log_errs)
    else:
        alpha = 0
        poly_r2 = 0

    # Fit 2: Exponential - err ~ exp(beta * dim)
    if np.sum(valid) > 2:
        log_errs_lin = np.log(mean_errors[valid] + 1e-10)
        exp_coeffs = np.polyfit(dims[valid], log_errs_lin, 1)
        beta = exp_coeffs[0]
        exp_r2 = 1 - np.var(log_errs_lin - np.polyval(exp_coeffs, dims[valid])) / np.var(log_errs_lin)
    else:
        beta = 0
        exp_r2 = 0

    print("  Power law: err ~ dim^%.2f (R^2=%.4f)" % (alpha, poly_r2))
    print("  Exponential: err ~ exp(%.4f * dim) (R^2=%.4f)" % (beta, exp_r2))

    if poly_r2 > exp_r2:
        scaling = "POLYNOMIAL (dim^%.2f)" % alpha
        print("  -> POLYNOMIAL SCALING CONFIRMED!")
        if alpha > 0:
            print("  -> Error GROWS with dimension (exponent=%.2f)" % alpha)
        else:
            print("  -> Error DECREASES with dimension (exponent=%.2f)" % alpha)
    else:
        scaling = "EXPONENTIAL (exp(%.4f*dim))" % beta
        print("  -> Exponential scaling detected")

    # Chemical accuracy rate
    chem_rates = [r['chem_accuracy_pct'] for r in all_results]
    print("\n  Chemical accuracy rates:")
    for r in all_results:
        print("    dim=%3d (%d qubits): %.0f%%" %
              (r['dim'], r['qubits'], r['chem_accuracy_pct']))

    # Save
    results = {
        'phase': 'Q191',
        'name': 'VQE Scaling Analysis',
        'dimensions': all_results,
        'scaling': {
            'polynomial_alpha': round(alpha, 4),
            'polynomial_r2': round(poly_r2, 4),
            'exponential_beta': round(beta, 6),
            'exponential_r2': round(exp_r2, 4),
            'best_fit': scaling,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q191_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Error vs dimension (log-log)
    ax = axes[0]
    ax.loglog(dims, mean_errors, 'bo-', linewidth=2, markersize=8, label='Mean error')
    ax.loglog(dims, median_errors, 'gs--', linewidth=1.5, markersize=6, label='Median error')
    # Fit lines
    if poly_r2 > 0:
        fit_dims = np.logspace(np.log10(dims[0]), np.log10(dims[-1]), 50)
        fit_poly = np.exp(poly_coeffs[1]) * fit_dims ** alpha
        ax.loglog(fit_dims, fit_poly, 'r:', linewidth=2,
                  label='Poly fit: dim^%.2f (R^2=%.3f)' % (alpha, poly_r2))
    ax.axhline(1.6, color='green', ls='--', label='Chemical accuracy')
    ax.set_xlabel('Dimension (= 2^qubits)')
    ax.set_ylabel('VQE Error (mHa)')
    ax.set_title('(a) Error Scaling (Log-Log)\n%s' % scaling[:30])
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')

    # (b) Chemical accuracy rate
    ax = axes[1]
    qubits = [r['qubits'] for r in all_results]
    ax.bar(range(len(qubits)), chem_rates,
           color=['#4CAF50' if c >= 50 else '#FF9800' if c >= 20 else '#F44336'
                  for c in chem_rates],
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(qubits)))
    ax.set_xticklabels(['%dq\n(d=%d)' % (q, d) for q, d in zip(qubits, dims)],
                        fontsize=8)
    ax.set_ylabel('Chemical Accuracy Rate (%%)')
    ax.set_title('(b) Success Rate by Dimension')
    ax.grid(alpha=0.3, axis='y')

    # (c) Convergence speed
    ax = axes[2]
    conv_steps = [r['mean_conv_steps'] for r in all_results]
    ax.plot(dims, conv_steps, 'o-', color='#9C27B0', linewidth=2, markersize=8)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Steps to Chemical Accuracy')
    ax.set_title('(c) Convergence Speed\n(Lower = Faster)')
    ax.grid(alpha=0.3)

    plt.suptitle('Q191: VQE Scaling Analysis\n'
                 'dim 2->%d: %s' % (dims[-1], scaling),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q191_scaling.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ191 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
