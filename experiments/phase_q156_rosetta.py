# -*- coding: utf-8 -*-
"""
Phase Q156: Chaos-Attention Rosetta Stone
==========================================
SNN-Comprypto has Lyapunov exponents proving chaos.
Does the LLM's layer-to-layer dynamics also show chaos?

Measure: Lyapunov exponent of hidden state trajectories across layers.
If positive -> LLM is chaotic (like SNN!)
If zero/negative -> LLM is ordered

This would mathematically UNIFY the two systems.
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


def estimate_lyapunov(trajectory, dt=1.0):
    """Estimate largest Lyapunov exponent from trajectory.
    trajectory: (n_steps, dim) array of hidden states across layers.
    """
    n_steps = len(trajectory)
    if n_steps < 3:
        return 0.0

    lyap_sum = 0
    count = 0
    for t in range(n_steps - 1):
        diff = trajectory[t + 1] - trajectory[t]
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            lyap_sum += np.log(norm)
            count += 1

    return float(lyap_sum / max(count, 1) / dt)


def compute_autocorrelation(trajectory, max_lag=10):
    """Autocorrelation of trajectory norms."""
    norms = np.array([np.linalg.norm(t) for t in trajectory])
    norms = norms - np.mean(norms)
    var = np.var(norms)
    if var < 1e-10:
        return [1.0] + [0.0] * (max_lag - 1)

    autocorr = []
    for lag in range(max_lag):
        if lag >= len(norms):
            autocorr.append(0.0)
            continue
        c = np.mean(norms[:len(norms) - lag] * norms[lag:]) / var
        autocorr.append(float(c))
    return autocorr


def shannon_entropy_trajectory(trajectory, n_bins=50):
    """Shannon entropy of trajectory state distribution."""
    all_vals = np.concatenate([t.flatten() for t in trajectory])
    hist, _ = np.histogram(all_vals, bins=n_bins, density=True)
    hist = hist[hist > 0]
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def main():
    print("=" * 60)
    print("Phase Q156: Chaos-Attention Rosetta Stone")
    print("  (Lyapunov Exponents of LLM Dynamics)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompts = [
        "The chaotic dynamics of the system evolve over time",
        "Spiking neural network membrane potentials fluctuate",
        "Quantum scrambling in the SYK model at high temperature",
        "The cat sat quietly on the mat doing nothing at all",
        "Hello world this is a simple test sentence",
    ]
    prompt_types = ['Chaos-related', 'SNN-related', 'SYK-related',
                    'Static/boring', 'Generic']

    all_results = []

    for prompt, ptype in zip(prompts, prompt_types):
        print("\n--- %s: '%s' ---" % (ptype, prompt[:40]))
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Trajectory = hidden states across layers (last token)
        trajectory = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            trajectory.append(h)

        trajectory = np.array(trajectory)

        # Lyapunov exponent
        lyap = estimate_lyapunov(trajectory)

        # Autocorrelation
        autocorr = compute_autocorrelation(trajectory, max_lag=min(10, n_layers))

        # Shannon entropy
        entropy = shannon_entropy_trajectory(trajectory)

        # Layer-to-layer distance profile
        distances = []
        for t in range(len(trajectory) - 1):
            d = float(np.linalg.norm(trajectory[t + 1] - trajectory[t]))
            distances.append(d)

        # Cosine similarity between adjacent layers
        cos_sims = []
        for t in range(len(trajectory) - 1):
            na = np.linalg.norm(trajectory[t])
            nb = np.linalg.norm(trajectory[t + 1])
            if na > 1e-10 and nb > 1e-10:
                cos_sims.append(float(np.dot(trajectory[t], trajectory[t+1]) / (na * nb)))

        result = {
            'prompt_type': ptype,
            'prompt': prompt[:40],
            'lyapunov': round(lyap, 4),
            'chaos': 'chaotic' if lyap > 0 else 'ordered',
            'entropy': round(entropy, 4),
            'autocorr_lag1': round(autocorr[1] if len(autocorr) > 1 else 0, 4),
            'avg_layer_distance': round(float(np.mean(distances)), 4),
            'avg_cos_sim': round(float(np.mean(cos_sims)), 4),
        }
        all_results.append(result)

        print("  Lyapunov: %.4f (%s)" % (lyap, result['chaos']))
        print("  Entropy: %.4f bits" % entropy)
        print("  Autocorr(lag=1): %.4f" % result['autocorr_lag1'])
        print("  Avg cos(layer, layer+1): %.4f" % result['avg_cos_sim'])

    # Comparison with SNN-Comprypto values
    print("\n--- Rosetta Stone: LLM vs SNN-Comprypto ---")
    snn_lyap = 25.98  # From paper
    snn_entropy = 7.998
    snn_autocorr = 0.008

    avg_llm_lyap = float(np.mean([r['lyapunov'] for r in all_results]))
    avg_llm_ent = float(np.mean([r['entropy'] for r in all_results]))
    avg_llm_autocorr = float(np.mean([r['autocorr_lag1'] for r in all_results]))

    print("  %-20s: Lyap=%.2f, Entropy=%.2f, Autocorr=%.3f" %
          ("SNN-Comprypto", snn_lyap, snn_entropy, snn_autocorr))
    print("  %-20s: Lyap=%.2f, Entropy=%.2f, Autocorr=%.3f" %
          ("LLM (avg)", avg_llm_lyap, avg_llm_ent, avg_llm_autocorr))
    print("  Lyapunov ratio: %.2f" % (avg_llm_lyap / snn_lyap))
    print("  Both chaotic?" if avg_llm_lyap > 0 else "  LLM is ordered!")

    # Save
    results = {
        'phase': 'Q156',
        'name': 'Chaos-Attention Rosetta Stone',
        'llm_dynamics': all_results,
        'snn_comparison': {
            'snn_lyapunov': snn_lyap,
            'snn_entropy': snn_entropy,
            'snn_autocorr': snn_autocorr,
            'llm_avg_lyapunov': round(avg_llm_lyap, 4),
            'llm_avg_entropy': round(avg_llm_ent, 4),
            'llm_avg_autocorr': round(avg_llm_autocorr, 4),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q156_rosetta.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Lyapunov comparison
    ax = axes[0]
    names = [r['prompt_type'] for r in all_results] + ['SNN-100', 'SNN-300']
    lyaps = [r['lyapunov'] for r in all_results] + [25.98, 26.69]
    colors = ['#4CAF50' if l > 0 else '#F44336' for l in lyaps[:-2]]
    colors += ['#2196F3', '#2196F3']
    ax.barh(range(len(names)), lyaps, color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='red', ls='--', linewidth=2, label='Chaos boundary')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Lyapunov Exponent')
    ax.set_title('(a) Chaos: LLM vs SNN-Comprypto')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='x')

    # (b) Layer-to-layer dynamics for one prompt
    ax = axes[1]
    prompt_idx = 0  # Chaos-related
    inp = tok(prompts[prompt_idx], return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    traj = [out.hidden_states[li][0, -1, :].float().cpu().numpy()
            for li in range(n_layers + 1)]
    dists = [float(np.linalg.norm(traj[i+1] - traj[i]))
             for i in range(len(traj) - 1)]
    ax.plot(range(len(dists)), dists, 'o-', color='#E91E63', linewidth=1.5)
    ax.set_xlabel('Layer transition')
    ax.set_ylabel('L2 distance')
    ax.set_title('(b) Layer-to-Layer Dynamics\n(distance between consecutive layers)')
    ax.grid(alpha=0.3)

    # (c) Chaos metrics radar-like comparison
    ax = axes[2]
    metrics = ['Lyapunov\n(normalized)', 'Entropy\n(bits)', 'Autocorr\n(inverted)']
    llm_vals = [avg_llm_lyap / max(snn_lyap, 0.01),
                avg_llm_ent / max(snn_entropy, 0.01),
                (1 - avg_llm_autocorr)]
    snn_vals = [1.0, 1.0, 1 - snn_autocorr]

    x_r = np.arange(len(metrics))
    w = 0.35
    ax.bar(x_r - w/2, llm_vals, w, color='#E91E63', label='LLM', alpha=0.85)
    ax.bar(x_r + w/2, snn_vals, w, color='#2196F3', label='SNN', alpha=0.85)
    ax.set_xticks(x_r)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel('Normalized value')
    ax.set_title('(c) Rosetta Stone:\nLLM vs SNN Chaos Metrics')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q156: Chaos-Attention Rosetta Stone (LLM vs SNN Dynamics)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q156_rosetta.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ156 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
