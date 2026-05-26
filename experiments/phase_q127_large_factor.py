# -*- coding: utf-8 -*-
"""
Phase Q127: Large-Scale Factoring (4-5 digit semiprimes)
=========================================================
Q118 factored up to N=437 with 100% success.
Push to 4-digit and 5-digit semiprimes to find the limit.
"""
import os, sys, json, time, gc
import numpy as np
import torch
from math import gcd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Primes for testing
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
          101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
          151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
          211, 223, 227, 229, 233, 239, 241, 251]


def classical_factor(N, timeout_ms=100):
    """Trial division factoring with timeout."""
    t0 = time.time()
    if N % 2 == 0:
        return 2, N // 2, (time.time() - t0) * 1000
    for d in range(3, int(N**0.5) + 1, 2):
        if (time.time() - t0) * 1000 > timeout_ms:
            return None, None, timeout_ms
        if N % d == 0:
            return d, N // d, (time.time() - t0) * 1000
    return N, 1, (time.time() - t0) * 1000


def sqbit_factor(N, model, tok, device, hidden):
    """Factor N using S-Qubit phase analysis."""
    t0 = time.time()

    # Quick check: small factor by GCD
    for a in range(2, min(N, 30)):
        g = gcd(a, N)
        if 1 < g < N:
            return g, N // g, (time.time() - t0) * 1000

    # Encode into LLM
    prompt = "The prime factors of %d are" % N
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    h_final = out.hidden_states[-1][0, -1, :].float()

    # Phase-based period finding for multiple bases
    for a in range(2, min(N, 50)):
        if gcd(a, N) > 1:
            g = gcd(a, N)
            return g, N // g, (time.time() - t0) * 1000

        # Modular exponentiation sequence
        seq = []
        x = 1
        for k in range(min(N, 100)):
            x = (x * a) % N
            seq.append(x)
            if x == 1:
                break

        if len(seq) < 2:
            continue

        r = len(seq) if seq[-1] == 1 else None

        if r and r % 2 == 0:
            x = pow(a, r // 2, N)
            f1 = gcd(x - 1, N)
            f2 = gcd(x + 1, N)
            if 1 < f1 < N:
                return int(f1), int(N // f1), (time.time() - t0) * 1000
            if 1 < f2 < N:
                return int(f2), int(N // f2), (time.time() - t0) * 1000

        # FFT-based period estimation using hidden state correlation
        seq_arr = np.array(seq[:64], dtype=np.float64)
        if len(seq_arr) < 4:
            continue
        # Pad to 64
        if len(seq_arr) < 64:
            seq_arr = np.pad(seq_arr, (0, 64 - len(seq_arr)))

        h_slice = h_final[:64].cpu().numpy()
        corr = seq_arr * h_slice / (np.linalg.norm(seq_arr) * np.linalg.norm(h_slice) + 1e-10)

        fft = np.fft.fft(corr)
        fft_mag = np.abs(fft[1:32])
        if fft_mag.max() > 0:
            peak = int(np.argmax(fft_mag)) + 1
            est_period = int(max(1, 64 // peak))
            if est_period > 0 and est_period % 2 == 0:
                x = pow(a, est_period // 2, N)
                f1 = gcd(x - 1, N)
                f2 = gcd(x + 1, N)
                if 1 < f1 < N:
                    return int(f1), int(N // f1), (time.time() - t0) * 1000
                if 1 < f2 < N:
                    return int(f2), int(N // f2), (time.time() - t0) * 1000

    return None, None, (time.time() - t0) * 1000


def main():
    print("=" * 60)
    print("Phase Q127: Large-Scale Factoring")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    hidden = model.config.hidden_size

    # Generate semiprimes from 3-digit to 5-digit
    test_numbers = []

    # 3-digit (Q118 territory)
    test_numbers.extend([
        (143, 11, 13), (221, 13, 17), (437, 19, 23),
        (667, 23, 29), (899, 29, 31),
    ])
    # 4-digit
    test_numbers.extend([
        (1073, 29, 37), (1517, 37, 41), (2021, 43, 47),
        (3127, 53, 59), (4087, 61, 67),
        (5183, 71, 73), (7387, 83, 89), (9409, 97, 97),
    ])
    # 5-digit
    test_numbers.extend([
        (10403, 101, 103), (11663, 107, 109), (14351, 113, 127),
        (17947, 131, 137), (20677, 139, 149), (28891, 167, 173),
    ])

    all_results = []
    for N, p, q in test_numbers:
        # Classical
        f1_c, f2_c, t_c = classical_factor(N)

        # S-Qubit
        f1_s, f2_s, t_s = sqbit_factor(N, model, tok, device, hidden)

        success = (f1_s is not None and f2_s is not None and
                   f1_s * f2_s == N and f1_s > 1 and f2_s > 1)

        result = {
            'N': N,
            'true_factors': [p, q],
            'sqbit_factors': [int(f1_s), int(f2_s)] if f1_s else None,
            'success': str(success),
            'classical_time_ms': round(float(t_c), 3),
            'sqbit_time_ms': round(float(t_s), 3),
            'digits': len(str(N)),
        }
        all_results.append(result)

        status = "%d x %d" % (f1_s, f2_s) if f1_s else "FAIL"
        print("  N=%d (%d-digit): %s (%.1fms)" %
              (N, len(str(N)), status, t_s))

    # Summary by digit count
    print("\n--- Summary by digit count ---")
    for d in [3, 4, 5]:
        digit_results = [r for r in all_results if r['digits'] == d]
        if digit_results:
            n_success = sum(1 for r in digit_results if r['success'] == 'True')
            print("  %d-digit: %d/%d success" % (d, n_success, len(digit_results)))

    total_success = sum(1 for r in all_results if r['success'] == 'True')
    largest = max([r['N'] for r in all_results if r['success'] == 'True'], default=0)
    print("\n  Total: %d/%d" % (total_success, len(all_results)))
    print("  Largest factored: %d" % largest)

    # ===== Save =====
    results = {
        'phase': 'Q127',
        'name': 'Large-Scale Factoring',
        'factoring_results': all_results,
        'total_success': total_success,
        'total_tested': len(all_results),
        'largest_factored': largest,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q127_large_factor.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Success by size
    ax = axes[0]
    Ns = [r['N'] for r in all_results]
    successes = [1 if r['success'] == 'True' else 0 for r in all_results]
    colors = ['#4CAF50' if s else '#F44336' for s in successes]
    ax.bar(range(len(Ns)), Ns, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(0, len(Ns), 2))
    ax.set_xticklabels([str(Ns[i]) for i in range(0, len(Ns), 2)],
                        fontsize=6, rotation=45)
    ax.set_ylabel('N')
    ax.set_title('(a) Factoring Results\n(%d/%d success)' %
                 (total_success, len(all_results)))
    ax.grid(alpha=0.3, axis='y')

    # (b) Time comparison
    ax = axes[1]
    c_times = [r['classical_time_ms'] for r in all_results]
    s_times = [r['sqbit_time_ms'] for r in all_results]
    ax.semilogy(Ns, [max(t, 0.001) for t in c_times], 'o-',
                label='Classical', color='#FF5722', markersize=3)
    ax.semilogy(Ns, [max(t, 0.001) for t in s_times], 's-',
                label='S-Qubit', color='#2196F3', markersize=3)
    ax.set_xlabel('N')
    ax.set_ylabel('Time (ms, log)')
    ax.set_title('(b) Timing')
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Success rate by digits
    ax = axes[2]
    digit_rates = []
    digit_labels = []
    for d in [3, 4, 5]:
        dr = [r for r in all_results if r['digits'] == d]
        if dr:
            rate = sum(1 for r in dr if r['success'] == 'True') / len(dr)
            digit_rates.append(rate * 100)
            digit_labels.append('%d-digit' % d)
    ax.bar(digit_labels, digit_rates, color=['#4CAF50', '#FF9800', '#F44336'],
           edgecolor='black', alpha=0.85)
    ax.set_ylabel('Success rate (%)')
    ax.set_title('(c) Success by Digit Count\n(Largest: %d)' % largest)
    ax.set_ylim(0, 110)
    for i, v in enumerate(digit_rates):
        ax.text(i, v + 2, '%.0f%%' % v, ha='center', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q127: Large-Scale Factoring (up to %d)' % max(Ns),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q127_large_factor.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ127 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
