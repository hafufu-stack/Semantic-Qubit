# -*- coding: utf-8 -*-
"""
Phase Q219: Quantum Contextuality Test
=========================================
Contextuality is the STRONGEST form of non-classicality.
Bell inequality violations prove non-locality, but contextuality is
even more fundamental (doesn't require spatial separation).

Peres-Mermin square test: if measurement outcomes depend on which
OTHER measurements are performed simultaneously, the system is contextual.
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


# Pauli matrices (2x2)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def peres_mermin_square():
    """Build the Peres-Mermin magic square operators (3x3 grid).
    Each row and column should multiply to +I or -I.
    A non-contextual hidden variable model cannot reproduce this."""
    square = [
        [np.kron(X, I2), np.kron(I2, X), np.kron(X, X)],   # Row 1
        [np.kron(I2, Y), np.kron(Y, I2), np.kron(Y, Y)],   # Row 2
        [np.kron(X, Y), np.kron(Y, X), np.kron(Z, Z)],     # Row 3
    ]
    return square


def measure_operator(state, operator):
    """Measure expectation value of operator in given state."""
    return float(np.real(state.conj() @ operator @ state))


def test_contextuality(model, tok, device, prompt, dim=4):
    """Test Peres-Mermin contextuality using LLM hidden states."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Use multiple layers to get different "measurement contexts"
    square = peres_mermin_square()
    n_layers = len(out.hidden_states) - 1

    # Test at several layers
    layer_results = []
    for layer_idx in range(0, n_layers + 1, max(1, n_layers // 8)):
        h = out.hidden_states[layer_idx][0, -1, :dim].float().cpu().numpy()
        h = h / (np.linalg.norm(h) + 1e-10)

        # Measure all 9 operators
        measurements = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                measurements[i, j] = measure_operator(h, square[i][j])

        # Row products (should be +1 for rows 1,2 and -1 for row 3)
        row_products = [np.prod(np.sign(measurements[i, :])) for i in range(3)]

        # Column products (should all be +1)
        col_products = [np.prod(np.sign(measurements[:, j])) for j in range(3)]

        # Non-contextual model: product of all row_products = product of all col_products
        # Quantum: row1*row2*row3 = -1, but col1*col2*col3 = +1 (CONTRADICTION!)
        row_total = np.prod(row_products)
        col_total = np.prod(col_products)
        is_contextual = (row_total != col_total)

        # Contextuality witness value
        witness = abs(row_total - col_total)

        layer_results.append({
            'layer': layer_idx,
            'measurements': measurements.tolist(),
            'row_products': [float(r) for r in row_products],
            'col_products': [float(c) for c in col_products],
            'row_total': float(row_total),
            'col_total': float(col_total),
            'is_contextual': bool(is_contextual),
            'witness': float(witness),
        })

    return layer_results


def main():
    print("=" * 60)
    print("Phase Q219: Quantum Contextuality (Peres-Mermin)")
    print("  (The strongest non-classicality test)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    prompts = [
        "quantum entanglement Bell state",
        "classical correlation without entanglement",
        "superposition of all basis states",
        "Peres Mermin magic square contextuality",
        "hidden variable theory deterministic",
        "quantum computing gate operation",
    ]

    all_results = []
    for prompt in prompts:
        print("\n--- %s ---" % prompt[:40])
        layer_data = test_contextuality(model, tok, device, prompt, dim=4)

        n_contextual = sum(1 for d in layer_data if d['is_contextual'])
        avg_witness = np.mean([d['witness'] for d in layer_data])

        print("  Contextual layers: %d/%d (avg witness=%.4f)" %
              (n_contextual, len(layer_data), avg_witness))

        all_results.append({
            'prompt': prompt[:40],
            'layers': layer_data,
            'n_contextual': n_contextual,
            'total_layers': len(layer_data),
            'avg_witness': round(avg_witness, 4),
        })

    # Summary
    total_contextual = sum(r['n_contextual'] for r in all_results)
    total_tests = sum(r['total_layers'] for r in all_results)
    ctx_rate = total_contextual / max(total_tests, 1)
    avg_witness_all = np.mean([r['avg_witness'] for r in all_results])

    if ctx_rate > 0.5:
        verdict = "CONTEXTUAL: %.0f%% of states show contextuality (witness=%.3f)" % (
            ctx_rate * 100, avg_witness_all)
    elif ctx_rate > 0.1:
        verdict = "PARTIALLY CONTEXTUAL: %.0f%% contextual" % (ctx_rate * 100)
    else:
        verdict = "NON-CONTEXTUAL: %.0f%% contextual" % (ctx_rate * 100)

    print("\n--- Summary ---")
    print("  Contextual: %d/%d (%.1f%%)" % (total_contextual, total_tests, ctx_rate * 100))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q219',
        'name': 'Quantum Contextuality (Peres-Mermin)',
        'prompts': all_results,
        'summary': {
            'total_contextual': total_contextual,
            'total_tests': total_tests,
            'contextuality_rate': round(ctx_rate, 4),
            'avg_witness': round(avg_witness_all, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q219_contextuality.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, r in enumerate(all_results[:6]):
        ax = axes_flat[idx]
        layers = [d['layer'] for d in r['layers']]
        witnesses = [d['witness'] for d in r['layers']]
        colors = ['#E91E63' if d['is_contextual'] else '#9E9E9E' for d in r['layers']]

        ax.bar(range(len(layers)), witnesses, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers], fontsize=7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Witness')
        ax.set_title('%s (%d/%d ctx)' % (r['prompt'][:25],
                     r['n_contextual'], r['total_layers']), fontsize=9)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q219: Quantum Contextuality (Peres-Mermin)\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q219_contextuality.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ219 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
