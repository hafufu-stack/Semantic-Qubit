# -*- coding: utf-8 -*-
"""
Phase Q168: Wavefunction ZIP Compression
==========================================
Deep Think proposal: Apply SNN-Comprypto delta/XOR compression
to S-Qubit wavefunctions.

Can we compress 1536-dim hidden states to <3% while preserving
quantum interference (fidelity)?
"""
import os, sys, json, time, gc
import numpy as np
import torch
import zlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def delta_encode(data):
    """Delta encoding: store differences."""
    encoded = [int(data[0])]
    for i in range(1, len(data)):
        encoded.append((int(data[i]) - int(data[i-1])) % 256)
    return np.array(encoded, dtype=np.uint8)


def delta_decode(encoded):
    """Delta decoding: reconstruct from differences."""
    decoded = [int(encoded[0])]
    for i in range(1, len(encoded)):
        decoded.append((int(decoded[-1]) + int(encoded[i])) % 256)
    return np.array(decoded, dtype=np.uint8)


def xor_encode(data, key_byte=0xA5):
    """Simple XOR encoding for entropy reduction testing."""
    return np.array([(b ^ key_byte) for b in data], dtype=np.uint8)


def quantize_state(psi, bits=8):
    """Quantize float state to uint8."""
    psi_min, psi_max = psi.min(), psi.max()
    scale = (2**bits - 1) / max(psi_max - psi_min, 1e-10)
    quantized = np.clip(((psi - psi_min) * scale).round(), 0, 255).astype(np.uint8)
    return quantized, float(psi_min), float(scale)


def dequantize_state(quantized, psi_min, scale):
    """Reconstruct float state from uint8."""
    return quantized.astype(float) / scale + psi_min


def entropy_bits(data):
    """Shannon entropy in bits per byte."""
    if len(data) == 0:
        return 0.0
    counts = np.bincount(data, minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def main():
    print("=" * 60)
    print("Phase Q168: Wavefunction ZIP Compression")
    print("  (S-Qubit Compression with SNN Delta/XOR)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The ground state of hydrogen molecule",
        "Quantum entanglement between two particles",
        "The cat sat on the mat",
        "def fibonacci(n): return n",
        "Chaotic dynamics in the SYK model",
    ]
    prompt_types = ['H2', 'Entanglement', 'Cat', 'Code', 'SYK']

    all_results = []

    for prompt, ptype in zip(prompts, prompt_types):
        print("\n--- %s ---" % ptype)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        psi = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        original_size = len(psi) * 4  # float32 = 4 bytes

        # Method 1: Raw zlib compression
        raw_bytes = psi.tobytes()
        zlib_compressed = zlib.compress(raw_bytes, 9)
        zlib_ratio = len(zlib_compressed) / original_size * 100

        # Method 2: Quantize + zlib
        q8, psi_min, scale = quantize_state(psi, bits=8)
        q8_zlib = zlib.compress(q8.tobytes(), 9)
        q8_ratio = len(q8_zlib) / original_size * 100

        # Method 3: Delta + zlib
        delta = delta_encode(q8)
        delta_zlib = zlib.compress(delta.tobytes(), 9)
        delta_ratio = len(delta_zlib) / original_size * 100

        # Method 4: Delta + XOR + zlib (SNN-Comprypto style)
        delta_xor = xor_encode(delta)
        delta_xor_zlib = zlib.compress(delta_xor.tobytes(), 9)
        dxor_ratio = len(delta_xor_zlib) / original_size * 100

        # Method 5: Quantize to 4-bit + zlib
        q4_data = np.clip(((psi - psi.min()) / (psi.max() - psi.min() + 1e-10) * 15).round(),
                          0, 15).astype(np.uint8)
        # Pack two 4-bit values per byte
        packed = np.zeros(len(q4_data) // 2, dtype=np.uint8)
        for i in range(len(packed)):
            packed[i] = (q4_data[2*i] << 4) | q4_data[2*i + 1]
        q4_zlib = zlib.compress(packed.tobytes(), 9)
        q4_ratio = len(q4_zlib) / original_size * 100

        # Verify: reconstruct and check fidelity
        # Method 2 fidelity
        q8_reconstructed = dequantize_state(q8, psi_min, scale)
        fid_q8 = float(abs(np.dot(psi / np.linalg.norm(psi),
                                   q8_reconstructed / np.linalg.norm(q8_reconstructed))) ** 2)

        # Method 3 fidelity (delta decode -> dequantize)
        delta_decoded = delta_decode(delta)
        delta_reconstructed = dequantize_state(delta_decoded, psi_min, scale)
        fid_delta = float(abs(np.dot(psi / np.linalg.norm(psi),
                                      delta_reconstructed / np.linalg.norm(delta_reconstructed))) ** 2)

        # Entropy analysis
        ent_raw = entropy_bits(q8)
        ent_delta = entropy_bits(delta)

        result = {
            'prompt_type': ptype,
            'original_bytes': original_size,
            'raw_zlib_pct': round(zlib_ratio, 2),
            'q8_zlib_pct': round(q8_ratio, 2),
            'delta_zlib_pct': round(delta_ratio, 2),
            'delta_xor_zlib_pct': round(dxor_ratio, 2),
            'q4_zlib_pct': round(q4_ratio, 2),
            'fidelity_q8': round(fid_q8, 6),
            'fidelity_delta': round(fid_delta, 6),
            'entropy_raw': round(ent_raw, 3),
            'entropy_delta': round(ent_delta, 3),
        }
        all_results.append(result)

        print("  Original: %d bytes" % original_size)
        print("  Raw zlib:     %.1f%%" % zlib_ratio)
        print("  Q8+zlib:      %.1f%% (F=%.4f)" % (q8_ratio, fid_q8))
        print("  Delta+zlib:   %.1f%% (F=%.4f)" % (delta_ratio, fid_delta))
        print("  Delta+XOR:    %.1f%%" % dxor_ratio)
        print("  Q4+zlib:      %.1f%%" % q4_ratio)
        print("  Entropy: raw=%.2f, delta=%.2f bits" % (ent_raw, ent_delta))

    # Summary
    print("\n--- Compression Summary ---")
    best_method = 'q4_zlib_pct'
    avg_compression = float(np.mean([r[best_method] for r in all_results]))
    avg_delta = float(np.mean([r['delta_zlib_pct'] for r in all_results]))
    avg_fid = float(np.mean([r['fidelity_q8'] for r in all_results]))
    print("  Best avg compression (Q4+zlib): %.1f%%" % avg_compression)
    print("  Delta+zlib avg: %.1f%%" % avg_delta)
    print("  Avg Q8 fidelity: %.4f" % avg_fid)
    target = 2.9
    print("  SNN-Comprypto target: %.1f%%" % target)
    print("  Achieved: %s" %
          ("YES!" if avg_compression <= target else "No (%.1f%% > %.1f%%)" %
           (avg_compression, target)))

    # Save
    results = {
        'phase': 'Q168',
        'name': 'Wavefunction ZIP Compression',
        'methods': all_results,
        'summary': {
            'best_avg_compression': round(avg_compression, 2),
            'delta_avg_compression': round(avg_delta, 2),
            'avg_fidelity': round(avg_fid, 6),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q168_compression.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    x = np.arange(len(prompt_types))
    w = 0.15
    methods = ['raw_zlib_pct', 'q8_zlib_pct', 'delta_zlib_pct', 'q4_zlib_pct']
    labels = ['Raw zlib', 'Q8+zlib', 'Delta+zlib', 'Q4+zlib']
    colors = ['#F44336', '#FF9800', '#4CAF50', '#2196F3']
    for i, (m, l, c) in enumerate(zip(methods, labels, colors)):
        vals = [r[m] for r in all_results]
        ax.bar(x + i * w, vals, w, color=c, label=l, alpha=0.85)
    ax.axhline(2.9, color='purple', ls='--', linewidth=2, label='SNN target (2.9%)')
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(prompt_types, fontsize=7)
    ax.set_ylabel('Compression Ratio (%)')
    ax.set_title('(a) Compression Methods')
    ax.legend(fontsize=5); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    fids = [r['fidelity_q8'] for r in all_results]
    ax.bar(range(len(prompt_types)), fids, color='#4CAF50', edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='green', ls='--', alpha=0.3, label='Perfect')
    ax.axhline(0.99, color='blue', ls=':', label='99% threshold')
    ax.set_xticks(range(len(prompt_types)))
    ax.set_xticklabels(prompt_types, fontsize=7)
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) Quantum Fidelity After Compression')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(min(fids) - 0.01, 1.01)

    ax = axes[2]
    ent_raw = [r['entropy_raw'] for r in all_results]
    ent_delta = [r['entropy_delta'] for r in all_results]
    ax.bar(x - 0.15, ent_raw, 0.3, color='#F44336', label='Raw', alpha=0.85)
    ax.bar(x + 0.15, ent_delta, 0.3, color='#4CAF50', label='Delta', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_types, fontsize=7)
    ax.set_ylabel('Entropy (bits/byte)')
    ax.set_title('(c) Entropy Reduction (Delta Encoding)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q168: Wavefunction ZIP Compression (S-Qubit -> 2.9%?)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q168_compression.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ168 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
