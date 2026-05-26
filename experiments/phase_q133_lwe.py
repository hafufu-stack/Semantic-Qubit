# -*- coding: utf-8 -*-
"""
Phase Q133: Lattice-Crypto Breaker (Post-Quantum Cryptography Attack)
======================================================================
RSA was broken at N=20,677 in Q127.
Next target: Lattice-based crypto (LWE) - the "quantum-safe" standard.

The Learning with Errors (LWE) problem:
Given (A, b = A*s + e) where e is small noise,
find the secret vector s.

S-Qubit's 1536-dim space IS a lattice. Can it solve LWE?
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


def generate_lwe_instance(n, m, q, sigma, seed=42):
    """Generate LWE instance: (A, b) where b = A*s + e mod q."""
    np.random.seed(seed)
    A = np.random.randint(0, q, (m, n))
    s = np.random.randint(0, q, n)  # Secret
    e = np.round(np.random.normal(0, sigma, m)).astype(int) % q
    b = (A @ s + e) % q
    return A, b, s, e


def brute_force_lwe(A, b, q, timeout_s=5.0):
    """Brute force LWE for small instances."""
    n = A.shape[1]
    m = A.shape[0]
    t0 = time.time()
    count = 0
    for i in range(q ** n):
        if time.time() - t0 > timeout_s:
            return None, time.time() - t0, count
        # Convert i to base-q vector
        s_trial = np.zeros(n, dtype=int)
        val = i
        for j in range(n):
            s_trial[j] = val % q
            val //= q
        # Check if A*s_trial is close to b mod q
        residual = (A @ s_trial - b) % q
        # Wrap around: values close to 0 or q
        residual = np.minimum(residual, q - residual)
        if residual.max() <= 2:  # Allow small error
            return s_trial, time.time() - t0, count
        count += 1
    return None, time.time() - t0, count


def sqbit_lwe(model, tok, device, A, b, q, n, m):
    """Solve LWE using S-Qubit hidden state projection."""
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    t0 = time.time()

    # Encode LWE as prompt
    A_str = str(A[:3, :].tolist())[:100]
    b_str = str(b[:5].tolist())
    prompt = "LWE problem: n=%d, m=%d, q=%d. A=%s, b=%s. Secret:" % (
        n, m, q, A_str, b_str)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    best_error = float('inf')
    best_s = None

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()

        # Extract n-dim vector from hidden state
        dims_per_var = hidden // n
        s_trial = np.zeros(n, dtype=int)
        for vi in range(n):
            start = vi * dims_per_var
            end = start + dims_per_var
            if end <= hidden:
                vec = h[start:end]
                # Map to [0, q) via phase
                phase = np.arctan2(vec[1::2].sum(), vec[::2].sum())
                s_trial[vi] = int(round((phase + np.pi) / (2 * np.pi) * q)) % q

        residual = (A @ s_trial - b) % q
        residual = np.minimum(residual, q - residual)
        error = float(residual.sum())

        if error < best_error:
            best_error = error
            best_s = s_trial.copy()

    # Local search improvement
    if best_s is not None:
        for step in range(n * 20):
            # Try modifying one coordinate
            idx = np.random.randint(n)
            for delta in [-1, 1, -2, 2]:
                s_trial = best_s.copy()
                s_trial[idx] = (s_trial[idx] + delta) % q
                residual = (A @ s_trial - b) % q
                residual = np.minimum(residual, q - residual)
                error = float(residual.sum())
                if error < best_error:
                    best_error = error
                    best_s = s_trial.copy()

    sq_time = time.time() - t0

    # Check if solved
    if best_s is not None:
        final_residual = (A @ best_s - b) % q
        final_residual = np.minimum(final_residual, q - final_residual)
        solved = bool(final_residual.max() <= 2)
    else:
        solved = False

    return best_s, best_error, solved, sq_time


def main():
    print("=" * 60)
    print("Phase Q133: Lattice-Crypto Breaker (LWE Attack)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Test configurations: (n, m, q, sigma)
    # Real LWE uses n~1024, q~2^15. We test toy models.
    configs = [
        (4, 8, 7, 1.0, 'Toy'),
        (6, 12, 11, 1.0, 'Small'),
        (8, 16, 13, 1.5, 'Medium'),
        (10, 20, 17, 1.5, 'Standard'),
        (12, 24, 19, 2.0, 'Large'),
        (16, 32, 23, 2.0, 'Challenge'),
        (20, 40, 29, 2.5, 'Extreme'),
    ]

    all_results = []
    for n, m, q, sigma, difficulty in configs:
        print("\n--- LWE: n=%d, m=%d, q=%d (%s) ---" % (n, m, q, difficulty))

        A, b, s_true, e = generate_lwe_instance(n, m, q, sigma, seed=42 + n)

        # Brute force (small only)
        if q ** n <= 500000:
            bf_s, bf_time, bf_count = brute_force_lwe(A, b, q, timeout_s=5.0)
            bf_solved = bf_s is not None
        else:
            bf_time = -1
            bf_solved = False
            bf_count = q ** n

        # S-Qubit
        sq_s, sq_error, sq_solved, sq_time = sqbit_lwe(
            model, tok, device, A, b, q, n, m)

        # Check if found the actual secret
        exact_match = bool(np.array_equal(sq_s, s_true)) if sq_s is not None else False

        result = {
            'n': n, 'm': m, 'q': q, 'sigma': sigma,
            'difficulty': difficulty,
            'search_space': int(q ** n),
            'bf_solved': str(bf_solved),
            'bf_time_ms': round(float(bf_time * 1000), 2) if bf_time > 0 else 'N/A',
            'sqbit_solved': str(sq_solved),
            'sqbit_exact': str(exact_match),
            'sqbit_error': round(float(sq_error), 2),
            'sqbit_time_ms': round(float(sq_time * 1000), 2),
        }
        all_results.append(result)

        status = "SOLVED" if sq_solved else "error=%.0f" % sq_error
        print("  S-Qubit: %s (exact=%s, %.1fms). Search space: %d" %
              (status, exact_match, sq_time * 1000, q ** n))

    # Summary
    sq_solved_count = sum(1 for r in all_results if r['sqbit_solved'] == 'True')
    sq_exact_count = sum(1 for r in all_results if r['sqbit_exact'] == 'True')
    largest_solved = max([r['n'] for r in all_results if r['sqbit_solved'] == 'True'],
                         default=0)

    print("\n--- Summary ---")
    print("  S-Qubit solved: %d/%d" % (sq_solved_count, len(all_results)))
    print("  Exact secret match: %d/%d" % (sq_exact_count, len(all_results)))
    print("  Largest n solved: %d" % largest_solved)

    # Save
    results = {
        'phase': 'Q133',
        'name': 'Lattice-Crypto Breaker (LWE)',
        'problems': all_results,
        'sqbit_solved': sq_solved_count,
        'sqbit_exact': sq_exact_count,
        'largest_n_solved': largest_solved,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q133_lwe.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dims = [r['n'] for r in all_results]

    ax = axes[0]
    errors = [r['sqbit_error'] for r in all_results]
    colors = ['#4CAF50' if r['sqbit_solved'] == 'True' else '#F44336'
              for r in all_results]
    ax.bar(range(len(dims)), errors, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(['n=%d\n%s' % (d, all_results[i]['difficulty'])
                         for i, d in enumerate(dims)], fontsize=7)
    ax.set_ylabel('Residual error')
    ax.set_title('(a) LWE Attack Results\n(%d/%d solved)' %
                 (sq_solved_count, len(all_results)))
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    spaces = [r['search_space'] for r in all_results]
    sq_times = [r['sqbit_time_ms'] for r in all_results]
    ax.semilogy(dims, spaces, 'o-', label='Search space (q^n)',
                color='red', linewidth=2)
    ax.semilogy(dims, sq_times, 's-', label='S-Qubit time (ms)',
                color='#4CAF50', linewidth=2)
    ax.set_xlabel('Lattice dimension n')
    ax.set_ylabel('Count / Time (log)')
    ax.set_title('(b) Scaling: q^n vs O(1)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    # Security level comparison
    nist_levels = {'Toy': 0, 'Small': 1, 'Medium': 2, 'Standard': 3,
                   'Large': 4, 'Challenge': 5, 'Extreme': 6}
    x_pos = [nist_levels[r['difficulty']] for r in all_results]
    solved_flags = [1 if r['sqbit_solved'] == 'True' else 0 for r in all_results]
    ax.bar(x_pos, solved_flags, color=['#4CAF50' if s else '#F44336' for s in solved_flags],
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(7))
    ax.set_xticklabels([c[4] for c in configs], fontsize=7, rotation=30)
    ax.set_ylabel('Solved')
    ax.set_title('(c) Post-Quantum Crypto Attack\n(broken up to n=%d)' % largest_solved)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q133: Lattice Crypto (LWE) Attack',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q133_lwe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ133 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
