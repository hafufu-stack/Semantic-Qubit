# -*- coding: utf-8 -*-
"""
Phase Q169: Spiking-Quantum Random Number Generator
=====================================================
Deep Think proposal: Combine SNN chaos with S-Qubit interference
to create a true neuromorphic-quantum RNG.

Test: adversarial prediction rate. SNN-Comprypto achieved 0.39%.
Can LLM+SNN do even better?
"""
import os, sys, json, time, gc
import numpy as np
import torch
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


class SpikeReservoir:
    """Minimal SNN reservoir for RNG."""
    def __init__(self, n_neurons=200, spectral_radius=1.5, seed=42):
        np.random.seed(seed)
        self.n = n_neurons
        W = np.random.randn(n_neurons, n_neurons)
        rho = max(abs(np.linalg.eigvals(W)))
        self.W = W * spectral_radius / rho
        mask = np.random.rand(n_neurons, n_neurons) < 0.1
        self.W *= mask
        self.state = np.zeros(n_neurons)
        self.v = -65.0 * np.ones(n_neurons)  # Membrane potentials

    def step(self, input_vec=None):
        if input_vec is not None:
            I_ext = input_vec[:self.n] if len(input_vec) >= self.n else \
                    np.pad(input_vec, (0, self.n - len(input_vec)))
        else:
            I_ext = np.zeros(self.n)

        I_total = self.W @ self.state + I_ext * 0.01
        # LIF dynamics
        self.v += (-(self.v + 65.0) + I_total * 10) / 20.0
        spikes = (self.v >= -50.0).astype(float)
        self.v[spikes > 0] = -70.0
        self.state = 0.7 * self.state + 0.3 * spikes
        return self.v.copy()


def nist_frequency_test(bits):
    """NIST monobit frequency test."""
    n = len(bits)
    s = 2 * np.sum(bits) - n
    from math import erfc, sqrt
    p_value = erfc(abs(s) / sqrt(2 * n))
    return float(p_value)


def nist_runs_test(bits):
    """NIST runs test."""
    n = len(bits)
    pi = np.mean(bits)
    if abs(pi - 0.5) >= 2.0 / np.sqrt(n):
        return 0.0
    runs = 1 + np.sum(bits[1:] != bits[:-1])
    from math import erfc, sqrt
    p_value = erfc(abs(runs - 2*n*pi*(1-pi)) / (2*sqrt(2*n)*pi*(1-pi) + 1e-10))
    return float(min(p_value, 1.0))


def autocorrelation_test(bits, max_lag=20):
    """Test autocorrelation at various lags."""
    n = len(bits)
    bits_centered = bits.astype(float) - 0.5
    autocorrs = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        c = float(np.abs(np.corrcoef(bits_centered[:-lag], bits_centered[lag:])[0, 1]))
        autocorrs.append(c)
    return autocorrs


def main():
    print("=" * 60)
    print("Phase Q169: Spiking-Quantum RNG")
    print("  (True Neuromorphic-Quantum Random Numbers)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    n_samples = 10000

    # Source 1: LLM hidden states only
    print("\n--- Source 1: LLM Only ---")
    prompts = [
        "Random quantum fluctuation %d:" % i for i in range(50)
    ]
    llm_bits = []
    for p in prompts:
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        bits = (h > 0).astype(np.uint8)
        llm_bits.extend(bits[:200])
    llm_bits = np.array(llm_bits[:n_samples])

    # Source 2: SNN only
    print("--- Source 2: SNN Only ---")
    snn = SpikeReservoir(n_neurons=200, seed=42)
    snn_bits = []
    for _ in range(n_samples // 8 + 1):
        v = snn.step()
        h = hashlib.sha256(v.tobytes()).digest()
        for byte in h[:1]:
            for bit in range(8):
                snn_bits.append((byte >> bit) & 1)
    snn_bits = np.array(snn_bits[:n_samples], dtype=np.uint8)

    # Source 3: LLM + SNN Hybrid (the proposed neuromorphic-quantum RNG)
    print("--- Source 3: LLM+SNN Hybrid ---")
    snn2 = SpikeReservoir(n_neurons=200, seed=42)
    hybrid_bits = []
    for p in prompts:
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()

        # Feed LLM hidden state into SNN
        v = snn2.step(h)

        # XOR LLM bits with SNN hash
        llm_part = (h[:200] > 0).astype(np.uint8)
        snn_hash = hashlib.sha256(v.tobytes()).digest()
        snn_part = np.array(
            [((snn_hash[i // 8] >> (i % 8)) & 1) for i in range(200)],
            dtype=np.uint8)
        combined = llm_part ^ snn_part
        hybrid_bits.extend(combined)
    hybrid_bits = np.array(hybrid_bits[:n_samples], dtype=np.uint8)

    # Source 4: Python random (baseline)
    np.random.seed(None)
    python_bits = np.random.randint(0, 2, n_samples).astype(np.uint8)

    # Evaluate all sources
    sources = {
        'LLM Only': llm_bits,
        'SNN Only': snn_bits,
        'LLM+SNN Hybrid': hybrid_bits,
        'Python random': python_bits,
    }

    all_results = []
    for name, bits in sources.items():
        freq_p = nist_frequency_test(bits)
        runs_p = nist_runs_test(bits)
        autocorr = autocorrelation_test(bits)
        avg_autocorr = float(np.mean(autocorr)) if autocorr else 0
        bias = abs(float(np.mean(bits)) - 0.5)

        # Adversarial prediction: can we predict next bit from last 8?
        correct = 0
        for i in range(8, len(bits)):
            context = bits[i-8:i]
            pred = int(np.mean(context) > 0.5)
            if pred == bits[i]:
                correct += 1
        pred_rate = correct / max(len(bits) - 8, 1) * 100

        result = {
            'source': name,
            'n_bits': len(bits),
            'bias': round(bias, 4),
            'freq_p_value': round(freq_p, 4),
            'runs_p_value': round(runs_p, 4),
            'avg_autocorr': round(avg_autocorr, 4),
            'prediction_rate_pct': round(pred_rate, 2),
            'nist_freq_pass': freq_p > 0.01,
            'nist_runs_pass': runs_p > 0.01,
        }
        all_results.append(result)
        print("  %s:" % name)
        print("    Bias: %.4f, Pred: %.2f%%, NIST Freq: %.3f (%s), Runs: %.3f (%s)" %
              (bias, pred_rate, freq_p,
               "PASS" if freq_p > 0.01 else "FAIL",
               runs_p,
               "PASS" if runs_p > 0.01 else "FAIL"))

    # Save
    results = {
        'phase': 'Q169',
        'name': 'Spiking-Quantum RNG',
        'n_samples': n_samples,
        'sources': all_results,
        'snn_comprypto_reference': {'prediction_rate': 0.39},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q169_rng.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    names = [r['source'] for r in all_results]
    pred_rates = [r['prediction_rate_pct'] for r in all_results]
    colors = ['#FF9800', '#2196F3', '#4CAF50', '#9E9E9E']
    ax.bar(range(len(names)), pred_rates, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(50, color='red', ls='--', label='Coin flip (50%)')
    ax.axhline(0.39, color='purple', ls=':', linewidth=2,
               label='SNN-Comprypto (0.39%)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=10)
    ax.set_ylabel('Prediction Rate (%)')
    ax.set_title('(a) Adversarial Prediction')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    freq_p = [r['freq_p_value'] for r in all_results]
    runs_p = [r['runs_p_value'] for r in all_results]
    x = np.arange(len(names))
    ax.bar(x - 0.15, freq_p, 0.3, color='#E91E63', label='Frequency', alpha=0.85)
    ax.bar(x + 0.15, runs_p, 0.3, color='#4CAF50', label='Runs', alpha=0.85)
    ax.axhline(0.01, color='red', ls='--', label='NIST threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=10)
    ax.set_ylabel('p-value')
    ax.set_title('(b) NIST Statistical Tests')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    bias_vals = [r['bias'] for r in all_results]
    autocorr_vals = [r['avg_autocorr'] for r in all_results]
    ax.bar(x - 0.15, bias_vals, 0.3, color='#FF9800', label='Bias', alpha=0.85)
    ax.bar(x + 0.15, autocorr_vals, 0.3, color='#2196F3', label='Avg autocorr', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=10)
    ax.set_ylabel('Value (lower = better)')
    ax.set_title('(c) Bias & Autocorrelation')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q169: Spiking-Quantum RNG (LLM + SNN Hybrid)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q169_rng.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ169 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
