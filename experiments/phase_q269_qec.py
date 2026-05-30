# -*- coding: utf-8 -*-
"""
Phase Q269: Quantum Error Correction in LLM
=============================================
MY IDEA: Can we encode quantum info redundantly in LLM space,
introduce errors, and recover? If yes, LLM supports QEC codes.
Test a simple 3-qubit bit-flip code.
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
    print("Phase Q269: Quantum Error Correction")
    print("  (3-qubit bit-flip code in LLM representations)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    prompts = [
        "quantum information protection",
        "error correction encoding",
        "fault tolerant computation",
        "noise resilient data storage",
    ]

    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    all_results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h_original = out.hidden_states[n_layers][0, -1, :6].float().cpu().numpy()
        h_original /= np.linalg.norm(h_original) + 1e-10

        # Encode: 3-qubit repetition code
        # |psi> = a|0> + b|1> -> a|000> + b|111>
        # In representation space: triplicate the 2-dim logical qubit
        logical = h_original[:2]
        encoded = np.concatenate([logical, logical, logical])  # 6-dim

        noise_results = []
        for noise in noise_levels:
            # Introduce bit-flip error on one "qubit" (2-dim block)
            corrupted = encoded.copy()
            if noise > 0:
                # Flip the first qubit pair with probability proportional to noise
                rng = np.random.RandomState(42)
                error_block = rng.randint(0, 3)
                corrupted[error_block*2:error_block*2+2] += rng.randn(2) * noise

            # Decode: majority vote
            votes = [corrupted[:2], corrupted[2:4], corrupted[4:6]]
            # Similarity-based majority: which 2 agree most?
            sims = []
            for i in range(3):
                for j in range(i+1, 3):
                    vi = votes[i] / (np.linalg.norm(votes[i]) + 1e-10)
                    vj = votes[j] / (np.linalg.norm(votes[j]) + 1e-10)
                    sims.append((i, j, float(np.dot(vi, vj))))
            best_pair = max(sims, key=lambda x: x[2])
            decoded = (votes[best_pair[0]] + votes[best_pair[1]]) / 2
            decoded /= np.linalg.norm(decoded) + 1e-10

            # No correction: just use corrupted directly
            uncorrected = corrupted[:2] / (np.linalg.norm(corrupted[:2]) + 1e-10)

            # Fidelity
            logical_norm = logical / (np.linalg.norm(logical) + 1e-10)
            fid_corrected = float(np.dot(decoded, logical_norm)) ** 2
            fid_uncorrected = float(np.dot(uncorrected, logical_norm)) ** 2

            noise_results.append({
                'noise': noise,
                'fid_corrected': round(max(0, fid_corrected), 4),
                'fid_uncorrected': round(max(0, fid_uncorrected), 4),
            })

        all_results.append({
            'prompt': prompt[:30],
            'noise_sweep': noise_results,
        })

    # Average across prompts
    avg_corrected = []
    avg_uncorrected = []
    for ni in range(len(noise_levels)):
        avg_c = np.mean([r['noise_sweep'][ni]['fid_corrected'] for r in all_results])
        avg_u = np.mean([r['noise_sweep'][ni]['fid_uncorrected'] for r in all_results])
        avg_corrected.append(round(avg_c, 4))
        avg_uncorrected.append(round(avg_u, 4))

    qec_works = sum(1 for c, u in zip(avg_corrected, avg_uncorrected) if c > u + 0.01)
    if qec_works >= len(noise_levels) // 2:
        verdict = "QEC WORKS: correction better in %d/%d noise levels" % (qec_works, len(noise_levels))
    elif qec_works > 0:
        verdict = "PARTIAL QEC: %d/%d improved" % (qec_works, len(noise_levels))
    else:
        verdict = "NO QEC ADVANTAGE: correction does not help"

    print("\n  Avg fidelity comparison:")
    for ni, noise in enumerate(noise_levels):
        print("    noise=%.1f: corrected=%.4f, uncorrected=%.4f" % (
            noise, avg_corrected[ni], avg_uncorrected[ni]))
    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q269', 'name': 'Quantum Error Correction',
        'noise_levels': noise_levels,
        'avg_corrected': avg_corrected, 'avg_uncorrected': avg_uncorrected,
        'summary': {'qec_better': qec_works, 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q269_qec.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(noise_levels, avg_corrected, 'o-', color='#4CAF50', lw=2, ms=8, label='With QEC')
    ax.plot(noise_levels, avg_uncorrected, 's--', color='#F44336', lw=2, ms=8, label='Without QEC')
    ax.set_xlabel('Noise Level'); ax.set_ylabel('Fidelity')
    ax.set_title('Q269: Quantum Error Correction\n%s' % verdict[:60], fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q269_qec.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ269 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
