# -*- coding: utf-8 -*-
"""
Phase Q157: Semantic Bell Inequality
======================================
Bell test for LLM: can two prompts be "entangled"?

Classical limit (CHSH): S <= 2
Quantum limit: S <= 2*sqrt(2) ~ 2.83
If LLM exceeds 2 -> non-classical correlations exist in hidden states!
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


def measure_in_basis(psi, theta):
    """Measure qubit-like observable at angle theta.
    Returns expectation value in [-1, 1].
    """
    # Project onto measurement basis rotated by theta
    n = len(psi)
    indices = np.arange(n)
    # Measurement operator: cos(theta)*Z + sin(theta)*X equivalent
    # Using components as pseudo-qubit
    even = psi[::2]
    odd = psi[1::2] if len(psi) > 1 else np.zeros(1)
    min_len = min(len(even), len(odd))
    even, odd = even[:min_len], odd[:min_len]

    p_up = np.sum(even ** 2)
    p_down = np.sum(odd ** 2)
    total = p_up + p_down
    if total < 1e-10:
        return 0.0

    # Rotated measurement
    result = np.cos(theta) * (p_up - p_down) / total
    result += np.sin(theta) * 2 * np.sum(even * odd) / total
    return float(np.clip(result, -1, 1))


def main():
    print("=" * 60)
    print("Phase Q157: Semantic Bell Inequality")
    print("  (CHSH Test for LLM Correlations)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Bell test setup: pairs of prompts that should be "entangled"
    # (semantically related but independently processable)
    entangled_pairs = [
        ("The electron orbits the nucleus at", "The proton attracts the electron with"),
        ("Quantum state A is spin up along", "Quantum state B is spin down along"),
        ("Alice measures her particle in the", "Bob measures his particle in the"),
        ("The left half of the wormhole contains", "The right half of the wormhole contains"),
        ("Energy is conserved when the system", "Momentum is conserved when the system"),
    ]

    # Separable pairs (should NOT be entangled)
    separable_pairs = [
        ("The weather is sunny today", "I like chocolate ice cream"),
        ("Python programming language", "Mount Everest is tall"),
        ("Tokyo is the capital of Japan", "Bananas are yellow fruit"),
    ]

    # CHSH measurement angles
    # Alice: a1=0, a2=pi/4; Bob: b1=pi/8, b2=3pi/8
    a1, a2 = 0, np.pi / 4
    b1, b2 = np.pi / 8, 3 * np.pi / 8

    def get_hidden(prompt, layer_idx=-1):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        li = layer_idx if layer_idx >= 0 else n_layers
        return out.hidden_states[li][0, -1, :].float().cpu().numpy()

    def compute_chsh(psi_A, psi_B):
        """Compute CHSH value S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)"""
        # Correlation E(a,b) = <A(a) * B(b)>
        def E(theta_a, theta_b):
            mA = measure_in_basis(psi_A, theta_a)
            mB = measure_in_basis(psi_B, theta_b)
            return mA * mB

        S = E(a1, b1) - E(a1, b2) + E(a2, b1) + E(a2, b2)
        return float(S)

    # Test across layers
    print("\n--- Entangled pairs ---")
    all_results = []

    for p_A, p_B in entangled_pairs:
        layer_results = []
        for li in range(0, n_layers + 1, 4):
            psi_A = get_hidden(p_A, li)
            psi_B = get_hidden(p_B, li)
            # Normalize
            psi_A = psi_A / max(np.linalg.norm(psi_A), 1e-10)
            psi_B = psi_B / max(np.linalg.norm(psi_B), 1e-10)
            S = compute_chsh(psi_A, psi_B)
            layer_results.append({'layer': int(li), 'S': round(S, 4)})

        # Best S across layers
        best_S = max(abs(r['S']) for r in layer_results)
        best_layer = max(layer_results, key=lambda r: abs(r['S']))['layer']

        result = {
            'pair': '%s | %s' % (p_A[:30], p_B[:30]),
            'type': 'entangled',
            'best_S': round(best_S, 4),
            'best_layer': int(best_layer),
            'violates_classical': best_S > 2.0,
            'layer_results': layer_results,
        }
        all_results.append(result)
        status = "VIOLATES S>2!" if best_S > 2.0 else "classical"
        print("  S=%.4f (layer %d) %s" % (best_S, best_layer, status))
        print("    '%s...' x '%s...'" % (p_A[:25], p_B[:25]))

    print("\n--- Separable pairs ---")
    for p_A, p_B in separable_pairs:
        layer_results = []
        for li in range(0, n_layers + 1, 4):
            psi_A = get_hidden(p_A, li)
            psi_B = get_hidden(p_B, li)
            psi_A = psi_A / max(np.linalg.norm(psi_A), 1e-10)
            psi_B = psi_B / max(np.linalg.norm(psi_B), 1e-10)
            S = compute_chsh(psi_A, psi_B)
            layer_results.append({'layer': int(li), 'S': round(S, 4)})

        best_S = max(abs(r['S']) for r in layer_results)
        result = {
            'pair': '%s | %s' % (p_A[:30], p_B[:30]),
            'type': 'separable',
            'best_S': round(best_S, 4),
            'violates_classical': best_S > 2.0,
        }
        all_results.append(result)
        print("  S=%.4f %s" % (best_S, "VIOLATES!" if best_S > 2.0 else "classical"))

    # Random baseline
    print("\n--- Random baseline ---")
    rand_S = []
    for _ in range(100):
        pA = np.random.randn(hidden_size); pA /= np.linalg.norm(pA)
        pB = np.random.randn(hidden_size); pB /= np.linalg.norm(pB)
        rand_S.append(abs(compute_chsh(pA, pB)))
    print("  Random mean |S|: %.4f, max: %.4f" % (np.mean(rand_S), max(rand_S)))

    # Save
    results = {
        'phase': 'Q157',
        'name': 'Semantic Bell Inequality (CHSH)',
        'classical_limit': 2.0,
        'quantum_limit': 2.83,
        'results': all_results,
        'random_baseline': {
            'mean_S': round(float(np.mean(rand_S)), 4),
            'max_S': round(float(max(rand_S)), 4),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q157_bell.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ent_S = [r['best_S'] for r in all_results if r['type'] == 'entangled']
    sep_S = [r['best_S'] for r in all_results if r['type'] == 'separable']
    x_ent = range(len(ent_S))
    x_sep = range(len(ent_S), len(ent_S) + len(sep_S))
    ax.bar(x_ent, ent_S, color='#E91E63', label='Entangled pairs', alpha=0.85)
    ax.bar(x_sep, sep_S, color='#2196F3', label='Separable pairs', alpha=0.85)
    ax.axhline(2.0, color='red', ls='--', linewidth=2, label='Classical limit (S=2)')
    ax.axhline(2.83, color='purple', ls=':', linewidth=2, label='Quantum limit (2sqrt2)')
    ax.axhline(float(np.mean(rand_S)), color='gray', ls=':', label='Random avg')
    ax.set_ylabel('|S| (CHSH value)')
    ax.set_title('(a) Bell Test: Entangled vs Separable')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    # Layer-wise S for first entangled pair
    if all_results and 'layer_results' in all_results[0]:
        lr = all_results[0]['layer_results']
        ax.plot([r['layer'] for r in lr], [abs(r['S']) for r in lr],
                'o-', color='#E91E63', linewidth=2, label='Entangled pair 1')
    if len(all_results) > 1 and 'layer_results' in all_results[1]:
        lr = all_results[1]['layer_results']
        ax.plot([r['layer'] for r in lr], [abs(r['S']) for r in lr],
                's-', color='#4CAF50', linewidth=2, label='Entangled pair 2')
    ax.axhline(2.0, color='red', ls='--', label='Classical limit')
    ax.set_xlabel('Layer')
    ax.set_ylabel('|S|')
    ax.set_title('(b) CHSH Value Across Layers')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('Q157: Semantic Bell Inequality (CHSH Test)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q157_bell.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ157 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
