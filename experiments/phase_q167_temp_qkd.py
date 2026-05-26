# -*- coding: utf-8 -*-
"""
Phase Q167: Chaotic Temperature QKD
=====================================
Deep Think proposal: SNN-Comprypto temperature sensitivity + LLM S-Qubit.

Temperature 0.0001 difference should collapse quantum state decryption.
Tests: inject chaotic thermal noise into LLM attention, then decrypt.
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


class ChaoticReservoir:
    """Minimal SNN-Comprypto reservoir for keystream generation."""
    def __init__(self, seed, n_neurons=100, temperature=1.0):
        np.random.seed(seed)
        self.n = n_neurons
        self.temperature = temperature
        W = np.random.randn(n_neurons, n_neurons)
        rho = max(abs(np.linalg.eigvals(W)))
        self.W = W * 1.4 / rho
        mask = np.random.rand(n_neurons, n_neurons) < 0.1
        self.W *= mask
        self.state = np.zeros(n_neurons)
        self.rng = np.random.RandomState(seed)

    def step(self, input_val=0.0):
        I_ext = input_val * np.random.randn(self.n)
        noise = self.rng.normal(0, 0.5 * self.temperature, self.n)
        I_total = self.W @ self.state + I_ext + noise
        self.state = 0.7 * self.state + 0.3 * np.tanh(I_total)
        return self.state.copy()

    def get_keystream(self, n_bytes):
        keys = []
        for _ in range(n_bytes):
            self.step()
            h = hashlib.sha256(self.state.tobytes()).digest()
            keys.append(h[0])
        return np.array(keys, dtype=np.uint8)


def main():
    print("=" * 60)
    print("Phase Q167: Chaotic Temperature QKD")
    print("  (SNN-Comprypto Temperature Sensitivity)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Extract S-Qubit (hidden state as "quantum state")
    prompt = "Quantum encrypted message via chaotic dynamics:"
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    squbit = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
    squbit_bytes = (((squbit - squbit.min()) / (squbit.max() - squbit.min() + 1e-10)) * 255).astype(np.uint8)

    # Encrypt with correct temperature
    T_correct = 1.0
    seed = 42
    reservoir_enc = ChaoticReservoir(seed, temperature=T_correct)
    keystream = reservoir_enc.get_keystream(len(squbit_bytes))
    ciphertext = squbit_bytes ^ keystream

    print("  S-Qubit dimension: %d" % len(squbit_bytes))
    print("  Correct temperature: %.4f" % T_correct)

    # Decrypt with various temperature offsets
    offsets = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    all_results = []

    for dT in offsets:
        T_test = T_correct + dT
        reservoir_dec = ChaoticReservoir(seed, temperature=T_test)
        keystream_dec = reservoir_dec.get_keystream(len(squbit_bytes))
        plaintext = ciphertext ^ keystream_dec

        # Compare with original
        match_rate = float(np.mean(plaintext == squbit_bytes)) * 100
        bit_errors = float(np.mean(np.unpackbits(plaintext ^ squbit_bytes)))

        # Fidelity: treat as quantum state
        psi_orig = squbit / (np.linalg.norm(squbit) + 1e-10)
        psi_dec = plaintext.astype(float) / (np.linalg.norm(plaintext.astype(float)) + 1e-10)
        fidelity = float(abs(np.dot(psi_orig, psi_dec)) ** 2)

        result = {
            'delta_T': float(dT),
            'T_test': round(float(T_test), 6),
            'match_rate_pct': round(match_rate, 2),
            'bit_error_rate': round(bit_errors * 100, 2),
            'fidelity': round(fidelity, 6),
        }
        all_results.append(result)

        status = "PERFECT" if match_rate > 99.9 else "COLLAPSED" if match_rate < 1 else "PARTIAL"
        print("  dT=%.6f: match=%.1f%%, fidelity=%.4f -> %s" %
              (dT, match_rate, fidelity, status))

    # Find collapse threshold
    collapse_dT = None
    for r in all_results:
        if r['match_rate_pct'] < 1.0 and collapse_dT is None:
            collapse_dT = r['delta_T']

    print("\n--- Temperature Sensitivity ---")
    print("  Collapse threshold: dT = %s" %
          (str(collapse_dT) if collapse_dT else "not found"))
    print("  Key space: 1 / %.1e = %.1e possible temperatures" %
          (collapse_dT if collapse_dT and collapse_dT > 0 else 1e-6,
           1.0 / (collapse_dT if collapse_dT and collapse_dT > 0 else 1e-6)))

    # Save
    results = {
        'phase': 'Q167',
        'name': 'Chaotic Temperature QKD',
        'squbit_dim': len(squbit_bytes),
        'correct_temperature': T_correct,
        'results': all_results,
        'collapse_threshold': collapse_dT,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q167_temp_qkd.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    dTs = [r['delta_T'] for r in all_results if r['delta_T'] > 0]
    matches = [r['match_rate_pct'] for r in all_results if r['delta_T'] > 0]
    ax.semilogx(dTs, matches, 'o-', color='#E91E63', linewidth=2, markersize=8)
    ax.axhline(100, color='green', ls='--', alpha=0.3, label='Perfect')
    ax.axhline(0.39, color='red', ls='--', alpha=0.5, label='Random (0.39%)')
    ax.set_xlabel('Temperature Offset (dT)')
    ax.set_ylabel('Match Rate (%)')
    ax.set_title('(a) Decryption vs Temperature Error')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    fids = [r['fidelity'] for r in all_results if r['delta_T'] > 0]
    ax.semilogx(dTs, fids, 's-', color='#4CAF50', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature Offset (dT)')
    ax.set_ylabel('Quantum Fidelity')
    ax.set_title('(b) S-Qubit Fidelity vs Temperature Error')
    ax.grid(alpha=0.3)

    ax = axes[2]
    # Show ciphertext vs plaintext
    ax.plot(squbit_bytes[:100], color='blue', alpha=0.7, label='Original S-Qubit')
    ax.plot(ciphertext[:100], color='red', alpha=0.5, label='Ciphertext')
    if len(all_results) > 3:
        # Wrong temperature decryption
        T_wrong = T_correct + 0.01
        res_wrong = ChaoticReservoir(seed, temperature=T_wrong)
        ks_wrong = res_wrong.get_keystream(100)
        wrong_dec = ciphertext[:100] ^ ks_wrong
        ax.plot(wrong_dec, color='gray', alpha=0.5, label='Wrong T decryption')
    ax.set_xlabel('Byte index')
    ax.set_ylabel('Value')
    ax.set_title('(c) Encryption Visual')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    plt.suptitle('Q167: Chaotic Temperature QKD (SNN-Comprypto x S-Qubit)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q167_temp_qkd.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ167 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
