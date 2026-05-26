# -*- coding: utf-8 -*-
"""
Phase Q153: SNN-LLM Hybrid Crypto
====================================
Combining SNN-Comprypto's chaotic avalanche with LLM semantic keys.

Protocol:
1. User provides natural language passphrase
2. LLM extracts semantic key (hidden state)
3. SNN reservoir amplifies avalanche effect
4. Result: semantic passphrase + chaotic security

Tests whether LLM+SNN hybrid achieves better avalanche than LLM alone.
"""
import os, sys, json, time, gc, hashlib
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


class SNNReservoir:
    """Simplified SNN reservoir (from SNN-Comprypto)."""

    def __init__(self, n_neurons=100, spectral_radius=1.4, seed=42):
        np.random.seed(seed)
        self.n = n_neurons
        self.tau = 20.0
        self.v_rest = -65.0
        self.v_thresh = -55.0
        self.v_reset = -70.0
        self.dt = 1.0

        # Sparse recurrent weights
        W = np.random.randn(n_neurons, n_neurons) * 0.1
        mask = np.random.rand(n_neurons, n_neurons) < 0.1
        W *= mask
        # Scale to spectral radius
        eigvals = np.abs(np.linalg.eigvals(W))
        max_eig = max(eigvals) if len(eigvals) > 0 else 1.0
        if max_eig > 0:
            W *= spectral_radius / max_eig
        self.W = W

        # State
        self.v = np.full(n_neurons, self.v_rest)

    def reset(self, seed_vec):
        """Initialize state from a seed vector."""
        np.random.seed(int(abs(seed_vec.sum() * 1000)) % (2**31))
        self.v = self.v_rest + seed_vec[:self.n] * 5 if len(seed_vec) >= self.n \
            else self.v_rest + np.resize(seed_vec, self.n) * 5

    def step(self, input_current, temperature=1.0):
        """One timestep of LIF dynamics."""
        I_syn = self.W @ (self.v > self.v_thresh).astype(float) * 10
        noise = np.random.randn(self.n) * 0.5 * temperature
        self.v += self.dt / self.tau * (-(self.v - self.v_rest) + I_syn + input_current + noise)

        spikes = self.v > self.v_thresh
        self.v[spikes] = self.v_reset
        return self.v.copy(), spikes

    def run(self, input_seq, n_steps=50, temperature=1.0):
        """Run reservoir and collect membrane potentials."""
        states = []
        for t in range(n_steps):
            I = input_seq[t % len(input_seq)] if len(input_seq) > 0 else 0
            v, _ = self.step(I * np.ones(self.n), temperature)
            states.append(v.copy())
        return np.array(states)


def hidden_to_bits(h, n_bits=256):
    bits = (h[:n_bits] > 0).astype(np.uint8)
    return bits


def avalanche_test(bits1, bits2):
    return float(np.mean(bits1 == bits2)) * 100


def main():
    print("=" * 60)
    print("Phase Q153: SNN-LLM Hybrid Crypto")
    print("  (SNN-Comprypto + Semantic Keys)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    def get_hidden(prompt):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float().cpu().numpy()

    snn = SNNReservoir(n_neurons=100, spectral_radius=1.4)

    # Test pairs: original + tiny modification
    test_pairs = [
        ("The secret meeting is at midnight under the bridge",
         "The secret meeting is at midnight under the Bridge"),
        ("My password is quantum2024secure",
         "My password is quantum2024Secure"),
        ("Transfer funds to account 12345678",
         "Transfer funds to account 12345679"),
        ("The launch code is alpha-bravo-charlie",
         "The launch code is alpha-bravo-Charlie"),
        ("Encrypt this message with key XYZ",
         "Encrypt this message with key xyz"),
    ]

    all_results = []

    for p_orig, p_mod in test_pairs:
        print("\n--- '%s' ---" % p_orig[:40])

        # Method 1: LLM only
        h_orig = get_hidden(p_orig)
        h_mod = get_hidden(p_mod)
        bits_llm_orig = hidden_to_bits(h_orig)
        bits_llm_mod = hidden_to_bits(h_mod)
        llm_match = avalanche_test(bits_llm_orig, bits_llm_mod)

        # Method 2: LLM + SNN hybrid
        snn.reset(h_orig)
        states_orig = snn.run(h_orig[:50], n_steps=100, temperature=1.0)
        final_orig = states_orig[-1]

        snn.reset(h_mod)
        states_mod = snn.run(h_mod[:50], n_steps=100, temperature=1.0)
        final_mod = states_mod[-1]

        bits_hybrid_orig = hidden_to_bits(final_orig, n_bits=100)
        bits_hybrid_mod = hidden_to_bits(final_mod, n_bits=100)
        hybrid_match = avalanche_test(bits_hybrid_orig, bits_hybrid_mod)

        # Method 3: LLM + SHA-256 (standard approach)
        hash_orig = hashlib.sha256(h_orig.tobytes()).digest()
        hash_mod = hashlib.sha256(h_mod.tobytes()).digest()
        bits_hash_orig = np.unpackbits(np.frombuffer(hash_orig, dtype=np.uint8))[:256]
        bits_hash_mod = np.unpackbits(np.frombuffer(hash_mod, dtype=np.uint8))[:256]
        hash_match = avalanche_test(bits_hash_orig, bits_hash_mod)

        # Method 4: SNN only (random seed, like SNN-Comprypto)
        seed_orig = int(hashlib.md5(p_orig.encode()).hexdigest(), 16) % (2**31)
        seed_mod = int(hashlib.md5(p_mod.encode()).hexdigest(), 16) % (2**31)
        np.random.seed(seed_orig)
        snn_orig_state = np.random.randn(100)
        snn.reset(snn_orig_state)
        snn_states_orig = snn.run(snn_orig_state[:50], n_steps=100)

        np.random.seed(seed_mod)
        snn_mod_state = np.random.randn(100)
        snn.reset(snn_mod_state)
        snn_states_mod = snn.run(snn_mod_state[:50], n_steps=100)

        bits_snn_orig = hidden_to_bits(snn_states_orig[-1], 100)
        bits_snn_mod = hidden_to_bits(snn_states_mod[-1], 100)
        snn_match = avalanche_test(bits_snn_orig, bits_snn_mod)

        result = {
            'prompt_orig': p_orig[:40],
            'prompt_mod': p_mod[:40],
            'llm_only_match': round(llm_match, 2),
            'llm_snn_hybrid_match': round(hybrid_match, 2),
            'llm_sha256_match': round(hash_match, 2),
            'snn_only_match': round(snn_match, 2),
            'ideal': 50.0,
        }
        all_results.append(result)

        print("  LLM only:      %.1f%% match" % llm_match)
        print("  LLM+SNN hybrid: %.1f%% match" % hybrid_match)
        print("  LLM+SHA-256:   %.1f%% match" % hash_match)
        print("  SNN only:      %.1f%% match" % snn_match)
        print("  (ideal=50%%)")

    # Summary
    print("\n--- Summary (avg match rate, lower=better avalanche) ---")
    avg_llm = float(np.mean([r['llm_only_match'] for r in all_results]))
    avg_hybrid = float(np.mean([r['llm_snn_hybrid_match'] for r in all_results]))
    avg_sha = float(np.mean([r['llm_sha256_match'] for r in all_results]))
    avg_snn = float(np.mean([r['snn_only_match'] for r in all_results]))

    print("  LLM only:       %.1f%%" % avg_llm)
    print("  LLM+SNN hybrid: %.1f%%" % avg_hybrid)
    print("  LLM+SHA-256:    %.1f%%" % avg_sha)
    print("  SNN only:       %.1f%%" % avg_snn)
    print("  Ideal:          50.0%%")

    # Save
    results = {
        'phase': 'Q153',
        'name': 'SNN-LLM Hybrid Crypto',
        'tests': all_results,
        'summary': {
            'llm_only': round(avg_llm, 2),
            'llm_snn_hybrid': round(avg_hybrid, 2),
            'llm_sha256': round(avg_sha, 2),
            'snn_only': round(avg_snn, 2),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q153_hybrid.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    methods = ['LLM only', 'LLM+SNN\nhybrid', 'LLM+SHA256', 'SNN only']
    avgs = [avg_llm, avg_hybrid, avg_sha, avg_snn]
    colors = ['#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
    bars = ax.bar(range(len(methods)), avgs, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(50, color='red', ls='--', label='Ideal (50%)', linewidth=2)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Avg Match Rate (%, lower=better)')
    ax.set_title('(a) Avalanche Effect Comparison')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                '%.1f%%' % val, ha='center', fontsize=9, fontweight='bold')

    ax = axes[1]
    x = np.arange(len(all_results))
    w = 0.2
    for i, (method, key, color) in enumerate([
        ('LLM', 'llm_only_match', '#FF9800'),
        ('Hybrid', 'llm_snn_hybrid_match', '#4CAF50'),
        ('SHA256', 'llm_sha256_match', '#2196F3'),
        ('SNN', 'snn_only_match', '#9C27B0'),
    ]):
        vals = [r[key] for r in all_results]
        ax.bar(x + i * w, vals, w, color=color, label=method, alpha=0.85)
    ax.axhline(50, color='red', ls='--')
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(['Test %d' % (i+1) for i in range(len(all_results))], fontsize=8)
    ax.set_ylabel('Match Rate (%)')
    ax.set_title('(b) Per-Test Avalanche')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q153: SNN-LLM Hybrid Crypto (Avalanche Amplification)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q153_hybrid.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ153 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
