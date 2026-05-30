# -*- coding: utf-8 -*-
"""
Phase Q207: Quantum Error Correction Code Discovery
=====================================================
Can the LLM's embedding space discover new quantum error correcting codes?

We test if LLM can find stabilizer-like codes by optimizing embeddings
to maximize error recovery under various noise channels.
This is a CPU-friendly experiment (no heavy GPU VQE).
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


def encode_state(model, tok, device, message_bits, n_physical=8):
    """Use LLM to encode logical bits into a higher-dimensional code."""
    embed_layer = model.model.embed_tokens
    n_logical = len(message_bits)

    prompt = "encode quantum error correcting code:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    # Encode message into embedding
    with torch.no_grad():
        for i, bit in enumerate(message_bits):
            embeds[0, -1, i] = float(bit) * 2 - 1  # map 0,1 -> -1,+1

    with torch.no_grad():
        out = model(inputs_embeds=embeds.float(), output_hidden_states=True)
        codeword = out.hidden_states[-1][0, -1, :n_physical].cpu().numpy()
        codeword = codeword / (np.linalg.norm(codeword) + 1e-10)

    return codeword


def apply_noise(codeword, noise_type, error_rate, rng):
    """Apply noise to codeword."""
    noisy = codeword.copy()
    n = len(noisy)

    if noise_type == 'bitflip':
        mask = rng.random(n) < error_rate
        noisy[mask] *= -1
    elif noise_type == 'erasure':
        mask = rng.random(n) < error_rate
        noisy[mask] = 0
    elif noise_type == 'gaussian':
        noisy += rng.randn(n) * error_rate
    elif noise_type == 'depolarizing':
        mask = rng.random(n) < error_rate
        noisy[mask] = rng.randn(mask.sum()) * 0.5

    return noisy


def decode_state(model, tok, device, noisy_codeword, n_logical):
    """Use LLM to decode noisy codeword back to logical bits."""
    embed_layer = model.model.embed_tokens
    n_physical = len(noisy_codeword)

    prompt = "decode quantum error correcting code:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    # Inject noisy codeword
    with torch.no_grad():
        noisy_torch = torch.tensor(noisy_codeword.astype(np.float32), device=device)
        embeds[0, -1, :n_physical] = noisy_torch

    with torch.no_grad():
        out = model(inputs_embeds=embeds.float(), output_hidden_states=True)
        decoded = out.hidden_states[-1][0, -1, :n_logical].cpu().numpy()
        # Threshold to get bits
        decoded_bits = (decoded > 0).astype(int)

    return decoded_bits


def test_code(model, tok, device, n_logical, n_physical,
              noise_type, error_rates, n_trials=50):
    """Test error correction at various error rates."""
    rng = np.random.RandomState(42)
    results = []

    for error_rate in error_rates:
        n_correct = 0
        for trial in range(n_trials):
            # Random message
            message = rng.randint(0, 2, size=n_logical)

            # Encode
            codeword = encode_state(model, tok, device, message, n_physical)

            # Apply noise
            noisy = apply_noise(codeword, noise_type, error_rate, rng)

            # Decode
            decoded = decode_state(model, tok, device, noisy, n_logical)

            # Check
            if np.array_equal(message, decoded):
                n_correct += 1

        accuracy = n_correct / n_trials
        results.append({
            'error_rate': error_rate,
            'accuracy': round(accuracy, 4),
            'n_correct': n_correct,
            'n_trials': n_trials,
        })

    return results


def main():
    print("=" * 60)
    print("Phase Q207: Quantum Error Correction Code Discovery")
    print("  (Can LLM learn error correcting codes?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Test configurations
    configs = [
        {'n_logical': 2, 'n_physical': 8, 'name': '[[8,2]] code'},
        {'n_logical': 3, 'n_physical': 12, 'name': '[[12,3]] code'},
        {'n_logical': 4, 'n_physical': 16, 'name': '[[16,4]] code'},
    ]

    noise_types = ['bitflip', 'erasure', 'gaussian', 'depolarizing']
    error_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    n_trials = 30

    all_results = []

    for config in configs:
        n_l = config['n_logical']
        n_p = config['n_physical']
        name = config['name']
        print("\n--- %s (k=%d, n=%d) ---" % (name, n_l, n_p))

        code_results = {'name': name, 'n_logical': n_l, 'n_physical': n_p,
                         'noise_tests': {}}

        for noise in noise_types:
            print("  Noise: %s" % noise)
            res = test_code(model, tok, device, n_l, n_p,
                            noise, error_rates, n_trials=n_trials)
            code_results['noise_tests'][noise] = res

            # Find threshold (50% accuracy point)
            threshold = 0
            for r in res:
                if r['accuracy'] >= 0.5:
                    threshold = r['error_rate']
            print("    Threshold (50%% accuracy): %.2f" % threshold)

        all_results.append(code_results)

    # Summary: average threshold across all codes and noise types
    all_thresholds = []
    for code_r in all_results:
        for noise, res_list in code_r['noise_tests'].items():
            threshold = max(r['error_rate'] for r in res_list if r['accuracy'] >= 0.5) \
                if any(r['accuracy'] >= 0.5 for r in res_list) else 0
            all_thresholds.append(threshold)

    avg_threshold = np.mean(all_thresholds) if all_thresholds else 0

    if avg_threshold > 0.2:
        verdict = "STRONG CODE: avg threshold=%.2f (>20%% error tolerance)" % avg_threshold
    elif avg_threshold > 0.1:
        verdict = "MODERATE CODE: avg threshold=%.2f" % avg_threshold
    else:
        verdict = "WEAK CODE: avg threshold=%.2f" % avg_threshold

    print("\n--- Summary ---")
    print("  Average error threshold: %.2f" % avg_threshold)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q207',
        'name': 'Quantum Error Correction Code Discovery',
        'codes': all_results,
        'summary': {
            'avg_threshold': round(avg_threshold, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q207_qec_discovery.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(len(configs), len(noise_types),
                              figsize=(16, 4 * len(configs)))
    if len(configs) == 1:
        axes = axes.reshape(1, -1)

    for i, code_r in enumerate(all_results):
        for j, noise in enumerate(noise_types):
            ax = axes[i][j]
            res_list = code_r['noise_tests'][noise]
            er = [r['error_rate'] for r in res_list]
            acc = [r['accuracy'] for r in res_list]
            ax.plot(er, acc, 'o-', color='#E91E63', lw=2)
            ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
            ax.set_xlabel('Error Rate')
            ax.set_ylabel('Recovery Accuracy')
            ax.set_title('%s / %s' % (code_r['name'], noise), fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)

    plt.suptitle('Q207: Quantum Error Correction Code Discovery\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q207_qec_discovery.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ207 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
