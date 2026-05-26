# -*- coding: utf-8 -*-
"""
Phase Q149: Semantic Avalanche Effect
======================================
SNN-Comprypto showed: temperature diff of 0.0001 -> complete decryption failure.
Can LLM hidden states show the same avalanche effect?

Test: tiny prompt changes -> massive hidden state changes
This bridges SNN-Comprypto (chaotic encryption) with S-Qubit.

If LLM shows avalanche -> "semantic encryption" is possible
(natural language sentences as cryptographic keys!)
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


def hamming_distance(a, b):
    """Bit-level Hamming distance between two float vectors."""
    # Convert to bytes and compare bit by bit
    a_bytes = a.tobytes()
    b_bytes = b.tobytes()
    diff_bits = 0
    total_bits = 0
    for ab, bb in zip(a_bytes, b_bytes):
        xor = ab ^ bb
        diff_bits += bin(xor).count('1')
        total_bits += 8
    return diff_bits / total_bits if total_bits > 0 else 0


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 60)
    print("Phase Q149: Semantic Avalanche Effect")
    print("  (SNN-Comprypto meets S-Qubit)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # ===== TEST 1: Single character change =====
    print("\n--- Test 1: Single Character Avalanche ---")
    base_prompt = "The secret key is hidden in the quantum state"
    char_variants = [
        ("original", base_prompt),
        ("+period", base_prompt + "."),
        ("+space", base_prompt + " "),
        ("cap T->t", "the secret key is hidden in the quantum state"),
        ("key->Key", base_prompt.replace("key", "Key")),
        ("state->State", base_prompt.replace("state", "State")),
        ("quantum->Quantum", base_prompt.replace("quantum", "Quantum")),
        ("hidden->Hidden", base_prompt.replace("hidden", "Hidden")),
    ]

    def get_hidden(prompt):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        # Last layer, last token
        return out.hidden_states[-1][0, -1, :].float().cpu().numpy()

    base_h = get_hidden(base_prompt)
    char_results = []
    for name, variant in char_variants:
        h = get_hidden(variant)
        cos = cosine_sim(base_h, h)
        hamming = hamming_distance(
            base_h.astype(np.float32), h.astype(np.float32))
        l2 = float(np.linalg.norm(base_h - h))
        # Byte-level match rate (like SNN-Comprypto's metric)
        b_base = np.sign(base_h).astype(np.int8)
        b_var = np.sign(h).astype(np.int8)
        match_rate = float(np.mean(b_base == b_var)) * 100

        result = {
            'variant': name,
            'cosine_sim': round(cos, 6),
            'hamming_diff': round(hamming, 4),
            'l2_distance': round(l2, 2),
            'sign_match': round(match_rate, 2),
        }
        char_results.append(result)
        if name == "original":
            print("  %20s: cos=1.000 (self)" % name)
        else:
            print("  %20s: cos=%.4f, hamming=%.2f%%, sign_match=%.1f%%" %
                  (name, cos, hamming * 100, match_rate))

    # ===== TEST 2: Semantic equivalence =====
    print("\n--- Test 2: Semantic Equivalence ---")
    semantic_pairs = [
        ("The cat sat on the mat", "The feline sat on the mat"),
        ("Water freezes at zero degrees", "H2O solidifies at 0 celsius"),
        ("The dog chased the ball", "The puppy ran after the sphere"),
        ("I love programming", "I enjoy coding"),
        ("The sun rises in the east", "Solar ascent occurs eastward"),
    ]

    semantic_results = []
    for p1, p2 in semantic_pairs:
        h1 = get_hidden(p1)
        h2 = get_hidden(p2)
        cos = cosine_sim(h1, h2)
        hamming = hamming_distance(
            h1.astype(np.float32), h2.astype(np.float32))
        b1 = np.sign(h1).astype(np.int8)
        b2 = np.sign(h2).astype(np.int8)
        match_rate = float(np.mean(b1 == b2)) * 100

        result = {
            'prompt1': p1[:30],
            'prompt2': p2[:30],
            'cosine_sim': round(cos, 4),
            'hamming_diff': round(hamming, 4),
            'sign_match': round(match_rate, 2),
        }
        semantic_results.append(result)
        print("  '%s...' vs '%s...': cos=%.4f, sign=%.1f%%" %
              (p1[:20], p2[:20], cos, match_rate))

    # ===== TEST 3: Layer-wise avalanche =====
    print("\n--- Test 3: Layer-wise Avalanche Propagation ---")
    p_base = "Ground state energy of hydrogen molecule"
    p_perturbed = "Ground state energy of hydrogen atom"

    inp_b = tok(p_base, return_tensors='pt').to(device)
    inp_p = tok(p_perturbed, return_tensors='pt').to(device)
    with torch.no_grad():
        out_b = model(**inp_b, output_hidden_states=True)
        out_p = model(**inp_p, output_hidden_states=True)

    layer_avalanche = []
    for li in range(n_layers + 1):
        hb = out_b.hidden_states[li][0, -1, :].float().cpu().numpy()
        hp = out_p.hidden_states[li][0, -1, :].float().cpu().numpy()
        cos = cosine_sim(hb, hp)
        hamming = hamming_distance(
            hb.astype(np.float32), hp.astype(np.float32))
        layer_avalanche.append({
            'layer': int(li),
            'cosine_sim': round(cos, 4),
            'hamming_diff': round(hamming, 4),
        })
        if li % 4 == 0 or li == n_layers:
            print("  Layer %2d: cos=%.4f, hamming=%.2f%%" %
                  (li, cos, hamming * 100))

    # ===== TEST 4: Comparison with SNN-Comprypto =====
    print("\n--- Test 4: vs SNN-Comprypto Temperature Avalanche ---")
    # SNN-Comprypto: T=1.0000 vs T=1.0001 -> match rate 0.40%
    # Our LLM: single char change -> what match rate?
    snn_comparison = {
        'snn_temp_0001': {'match_rate': 0.40, 'description': 'T diff = 0.0001'},
        'llm_char_change': {
            'match_rate': round(float(np.mean(
                [r['sign_match'] for r in char_results
                 if r['variant'] != 'original']
            )), 2),
            'description': 'Single char change',
        },
        'llm_semantic_equiv': {
            'match_rate': round(float(np.mean(
                [r['sign_match'] for r in semantic_results]
            )), 2),
            'description': 'Semantic equivalent',
        },
    }
    for k, v in snn_comparison.items():
        print("  %25s: sign match = %.2f%%" % (v['description'], v['match_rate']))

    # Save
    results = {
        'phase': 'Q149',
        'name': 'Semantic Avalanche Effect',
        'char_avalanche': char_results,
        'semantic_equivalence': semantic_results,
        'layer_propagation': layer_avalanche,
        'snn_comparison': snn_comparison,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q149_avalanche.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Character avalanche
    ax = axes[0]
    names = [r['variant'] for r in char_results if r['variant'] != 'original']
    cos_vals = [r['cosine_sim'] for r in char_results if r['variant'] != 'original']
    colors = ['#4CAF50' if c > 0.9 else '#FF9800' if c > 0.7 else '#F44336'
              for c in cos_vals]
    ax.barh(range(len(names)), cos_vals, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Cosine Similarity to Original')
    ax.set_title('(a) Single-Char Avalanche Effect')
    ax.axvline(0.5, color='red', ls='--', alpha=0.5, label='Random baseline')
    ax.set_xlim(0, 1.05); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='x')

    # (b) Layer-wise propagation
    ax = axes[1]
    layers = [r['layer'] for r in layer_avalanche]
    cos_layer = [r['cosine_sim'] for r in layer_avalanche]
    ax.plot(layers, cos_layer, 'o-', color='#E91E63', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(b) Avalanche Propagation Through Layers\n(molecule->atom)')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.grid(alpha=0.3)

    # (c) Comparison with SNN-Comprypto
    ax = axes[2]
    comp_names = ['SNN-Comprypto\n(T+0.0001)',
                  'LLM\n(1 char)', 'LLM\n(semantic eq.)']
    comp_vals = [
        snn_comparison['snn_temp_0001']['match_rate'],
        snn_comparison['llm_char_change']['match_rate'],
        snn_comparison['llm_semantic_equiv']['match_rate'],
    ]
    comp_colors = ['#2196F3', '#4CAF50', '#FF9800']
    ax.bar(range(len(comp_names)), comp_vals, color=comp_colors,
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(comp_names)))
    ax.set_xticklabels(comp_names, fontsize=8)
    ax.set_ylabel('Sign Match Rate (%)')
    ax.set_title('(c) SNN-Comprypto vs LLM Avalanche\n(lower = stronger avalanche)')
    ax.axhline(50, color='red', ls=':', label='Random baseline (50%)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q149: Semantic Avalanche (SNN-Comprypto x S-Qubit)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q149_avalanche.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ149 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
