# -*- coding: utf-8 -*-
"""
Phase Q150: Semantic Quantum Key Distribution
===============================================
Inspired by SNN-Comprypto: can two parties share a "secret prompt"
and extract matching S-Qubits as a shared cryptographic key?

Protocol:
1. Alice and Bob agree on a secret prompt
2. Both run the SAME LLM and extract hidden states
3. They should get IDENTICAL keys (deterministic)
4. Eve (eavesdropper) uses a DIFFERENT prompt -> completely different key

This is "Quantum" Key Distribution because the key lives in the
LLM's latent space (analogous to quantum state space).
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


def hidden_to_key(hidden_state, key_bits=256):
    """Convert hidden state to cryptographic key."""
    # Deterministic: sign of each component -> bit string
    bits = (hidden_state > 0).astype(np.uint8)[:key_bits]
    # Pack into bytes
    key_bytes = np.packbits(bits)
    return key_bytes, bits


def key_entropy(bits):
    """Shannon entropy of bit string (ideal = 1.0 per bit)."""
    p1 = np.mean(bits)
    p0 = 1 - p1
    if p0 < 1e-10 or p1 < 1e-10:
        return 0.0
    return float(-p0 * np.log2(p0) - p1 * np.log2(p1))


def main():
    print("=" * 60)
    print("Phase Q150: Semantic Quantum Key Distribution")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    def get_hidden(prompt, layer=-1):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        li = layer if layer >= 0 else n_layers
        return out.hidden_states[li][0, -1, :].float().cpu().numpy()

    # === Protocol Test ===
    secret_prompts = [
        "The meeting point is under the old oak tree at midnight",
        "Quantum entanglement key: alpha-7-bravo-9-charlie",
        "My grandmother's maiden name was Sakura Tanaka",
    ]

    eve_prompts = [
        "The meeting point is under the old oak tree at noon",  # 1 word diff
        "Quantum entanglement key: alpha-7-bravo-9-delta",  # 1 word diff
        "My grandmother's maiden name was Sakura Suzuki",  # 1 word diff
    ]

    random_prompts = [
        "The weather is nice today",
        "I like to eat pizza",
        "Hello world",
    ]

    print("\n--- QKD Protocol Test ---")
    all_results = []

    for i, (secret, eve, rand) in enumerate(zip(
            secret_prompts, eve_prompts, random_prompts)):
        print("\n  Secret: '%s'" % secret[:50])

        # Alice extracts key
        h_alice = get_hidden(secret)
        key_alice, bits_alice = hidden_to_key(h_alice)

        # Bob extracts key (same prompt, same model = should be identical)
        h_bob = get_hidden(secret)
        key_bob, bits_bob = hidden_to_key(h_bob)

        # Eve tries with slightly different prompt
        h_eve = get_hidden(eve)
        key_eve, bits_eve = hidden_to_key(h_eve)

        # Random eavesdropper
        h_rand = get_hidden(rand)
        key_rand, bits_rand = hidden_to_key(h_rand)

        # Metrics
        alice_bob_match = float(np.mean(bits_alice == bits_bob)) * 100
        alice_eve_match = float(np.mean(bits_alice == bits_eve)) * 100
        alice_rand_match = float(np.mean(bits_alice == bits_rand)) * 100

        # Key quality
        entropy = key_entropy(bits_alice)

        # SHA-256 hash comparison
        hash_alice = hashlib.sha256(key_alice.tobytes()).hexdigest()[:16]
        hash_bob = hashlib.sha256(key_bob.tobytes()).hexdigest()[:16]
        hash_eve = hashlib.sha256(key_eve.tobytes()).hexdigest()[:16]

        result = {
            'secret': secret[:40],
            'alice_bob_match': round(alice_bob_match, 2),
            'alice_eve_match': round(alice_eve_match, 2),
            'alice_rand_match': round(alice_rand_match, 2),
            'key_entropy': round(entropy, 4),
            'hash_alice': hash_alice,
            'hash_bob': hash_bob,
            'hash_eve': hash_eve,
            'hashes_match': hash_alice == hash_bob,
        }
        all_results.append(result)

        print("  Alice-Bob: %.1f%% match (should be 100%%)" % alice_bob_match)
        print("  Alice-Eve: %.1f%% match (1-word diff, should be ~50%%)" %
              alice_eve_match)
        print("  Alice-Rand: %.1f%% match (should be ~50%%)" % alice_rand_match)
        print("  Entropy: %.4f bits/bit (ideal=1.0)" % entropy)
        print("  Hash Alice: %s, Bob: %s -> %s" %
              (hash_alice, hash_bob,
               "MATCH!" if hash_alice == hash_bob else "DIFFERENT"))

    # === Key space analysis ===
    print("\n--- Key Space Analysis ---")
    # How many unique keys from different prompts?
    test_prompts = [
        "apple", "Apple", "APPLE", "apple.", "apple!",
        "banana", "cherry", "date", "elderberry", "fig",
        "The quick brown fox", "the quick brown fox",
        "quantum computing", "Quantum Computing",
        "hello world", "Hello World", "HELLO WORLD",
        "password123", "Password123", "PASSWORD123",
    ]

    keys = []
    for p in test_prompts:
        h = get_hidden(p)
        _, bits = hidden_to_key(h)
        keys.append(bits)

    # Pairwise similarity
    n = len(keys)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = float(np.mean(keys[i] == keys[j])) * 100

    # Off-diagonal average (should be ~50% for good keys)
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(sim_matrix[i, j])

    print("  %d unique prompts tested" % n)
    print("  Avg pairwise similarity: %.2f%% (ideal: 50%%)" %
          np.mean(off_diag))
    print("  Min: %.2f%%, Max: %.2f%%" % (min(off_diag), max(off_diag)))

    keyspace = {
        'n_prompts': n,
        'avg_similarity': round(float(np.mean(off_diag)), 2),
        'min_similarity': round(float(min(off_diag)), 2),
        'max_similarity': round(float(max(off_diag)), 2),
    }

    # Save
    results = {
        'phase': 'Q150',
        'name': 'Semantic Quantum Key Distribution',
        'qkd_tests': all_results,
        'keyspace': keyspace,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q150_qkd.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    x = np.arange(len(all_results))
    w = 0.25
    ab = [r['alice_bob_match'] for r in all_results]
    ae = [r['alice_eve_match'] for r in all_results]
    ar = [r['alice_rand_match'] for r in all_results]
    ax.bar(x - w, ab, w, color='#4CAF50', label='Alice-Bob (same key)', alpha=0.85)
    ax.bar(x, ae, w, color='#FF9800', label='Alice-Eve (1 word diff)', alpha=0.85)
    ax.bar(x + w, ar, w, color='#F44336', label='Alice-Random', alpha=0.85)
    ax.axhline(50, color='gray', ls='--', alpha=0.5, label='Random baseline')
    ax.set_ylabel('Key Match Rate (%)')
    ax.set_title('(a) QKD Protocol Security')
    ax.set_xticks(x); ax.set_xticklabels(['Key 1', 'Key 2', 'Key 3'])
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=30, vmax=100)
    ax.set_title('(b) Key Similarity Matrix\n(20 prompts)')
    plt.colorbar(im, ax=ax, label='Match %')
    ax.set_xlabel('Prompt index'); ax.set_ylabel('Prompt index')

    ax = axes[2]
    ax.hist(off_diag, bins=20, color='#2196F3', edgecolor='black', alpha=0.85)
    ax.axvline(50, color='red', ls='--', label='Ideal (50%%)')
    ax.set_xlabel('Pairwise Similarity (%)')
    ax.set_ylabel('Count')
    ax.set_title('(c) Key Independence Distribution')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q150: Semantic QKD (Prompt-Based Key Exchange)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q150_qkd.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ150 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
