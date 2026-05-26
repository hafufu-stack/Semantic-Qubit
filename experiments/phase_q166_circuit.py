# -*- coding: utf-8 -*-
"""
Phase Q166: LLM as Quantum Circuit
=====================================
Decompose the LLM forward pass into quantum circuit elements.

Each layer = quantum gate. Measure:
1. Unitarity (how close to unitary is each layer's transformation?)
2. Gate count (how many quantum gates would be needed?)
3. Circuit depth (how deep is the equivalent circuit?)

Compare with real quantum circuits for VQE.
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


def unitarity_measure(M):
    """How close is matrix M to unitary? Returns 0 (unitary) to 1 (far)."""
    # For unitary: M @ M^H = I
    n = min(M.shape)
    M_sq = M[:n, :n]
    product = M_sq @ M_sq.T
    I = np.eye(n)
    # Normalize
    product /= (np.trace(product) / n + 1e-10)
    return float(np.linalg.norm(product - I, 'fro') / np.sqrt(n))


def effective_gate_count(M, threshold=0.01):
    """Estimate equivalent quantum gate count from SVD."""
    s = np.linalg.svd(M, compute_uv=False)
    s /= (s[0] + 1e-10)
    n_significant = int(np.sum(s > threshold))
    # Each significant singular value ~ 1 rotation gate
    # Plus CNOT gates for entanglement ~ n_significant^2
    return n_significant, n_significant ** 2


def main():
    print("=" * 60)
    print("Phase Q166: LLM as Quantum Circuit")
    print("  (Layer Decomposition)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Probe: pass different inputs and measure layer Jacobians
    prompt = "The quantum state of the system evolves:"
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    layer_data = []

    for li in range(n_layers):
        h_in = out.hidden_states[li][0, -1, :].float().cpu().numpy()
        h_out = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()

        # Approximate Jacobian from the transformation
        # J ~ (h_out - h_in) projected onto input directions
        delta = h_out - h_in  # Residual connection effect

        # Unitarity of the transformation
        # Approximate: compute h_out/||h_out|| @ h_in/||h_in||
        n_in = np.linalg.norm(h_in)
        n_out = np.linalg.norm(h_out)
        norm_ratio = n_out / max(n_in, 1e-10)

        # Cosine between input and output (how much does the layer change?)
        cos_change = float(np.dot(h_in, h_out) / (n_in * n_out + 1e-10))

        # Residual magnitude (how much does the layer add?)
        residual_ratio = float(np.linalg.norm(delta) / max(n_in, 1e-10))

        # Effective dimension of the transformation
        # Using the weight matrices
        layer_obj = model.model.layers[li]
        attn = layer_obj.self_attn

        # Attention: Q, K, V projections
        Wq = attn.q_proj.weight.detach().float().cpu().numpy()
        Wk = attn.k_proj.weight.detach().float().cpu().numpy()

        # QK^T matrix effective rank
        QK_sample = Wq[:64, :] @ Wk[:64, :].T
        unitarity = unitarity_measure(QK_sample)
        n_rot, n_cnot = effective_gate_count(QK_sample)

        # MLP weights
        mlp = layer_obj.mlp
        W_gate = mlp.gate_proj.weight.detach().float().cpu().numpy()
        n_rot_mlp, n_cnot_mlp = effective_gate_count(W_gate[:64, :64])

        total_gates = n_rot + n_cnot + n_rot_mlp + n_cnot_mlp

        layer_data.append({
            'layer': int(li),
            'cos_change': round(float(cos_change), 4),
            'norm_ratio': round(float(norm_ratio), 4),
            'residual_ratio': round(float(residual_ratio), 4),
            'unitarity_deviation': round(float(unitarity), 4),
            'attn_rotation_gates': int(n_rot),
            'attn_cnot_gates': int(n_cnot),
            'mlp_rotation_gates': int(n_rot_mlp),
            'mlp_cnot_gates': int(n_cnot_mlp),
            'total_equiv_gates': int(total_gates),
        })

        if li % 7 == 0:
            print("  Layer %2d: cos=%.3f, unitarity_dev=%.3f, gates=%d" %
                  (li, cos_change, unitarity, total_gates))

    # Total circuit statistics
    total_circuit_gates = sum(d['total_equiv_gates'] for d in layer_data)
    avg_unitarity = float(np.mean([d['unitarity_deviation'] for d in layer_data]))
    circuit_depth = n_layers

    print("\n--- Quantum Circuit Summary ---")
    print("  Total equivalent gates: %d" % total_circuit_gates)
    print("  Circuit depth: %d layers" % circuit_depth)
    print("  Average unitarity deviation: %.4f" % avg_unitarity)
    print("  (0 = perfectly unitary, 1 = far from unitary)")

    # Compare with real quantum circuits
    print("\n--- Comparison with Real VQE Circuits ---")
    # Typical VQE for H2: ~50-100 gates, depth 10-20
    # Typical VQE for LiH: ~200-500 gates, depth 30-50
    real_h2_gates = 80
    real_lih_gates = 350
    print("  LLM equivalent: %d gates, depth %d" % (total_circuit_gates, circuit_depth))
    print("  Real H2 VQE:    ~%d gates, depth ~15" % real_h2_gates)
    print("  Real LiH VQE:   ~%d gates, depth ~40" % real_lih_gates)
    print("  LLM/Real ratio: %.0fx gates" % (total_circuit_gates / real_h2_gates))

    # Save
    results = {
        'phase': 'Q166',
        'name': 'LLM as Quantum Circuit',
        'layer_decomposition': layer_data,
        'summary': {
            'total_gates': total_circuit_gates,
            'circuit_depth': circuit_depth,
            'avg_unitarity_deviation': round(avg_unitarity, 4),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q166_circuit.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    layers = [d['layer'] for d in layer_data]
    unitarities = [d['unitarity_deviation'] for d in layer_data]
    ax.plot(layers, unitarities, 'o-', color='#E91E63', linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Unitarity Deviation')
    ax.set_title('(a) How Unitary is Each Layer?\n(0=perfect, 1=far)')
    ax.grid(alpha=0.3)

    ax = axes[1]
    gate_counts = [d['total_equiv_gates'] for d in layer_data]
    ax.bar(layers, gate_counts, color='#4CAF50', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Equivalent Gates')
    ax.set_title('(b) Gate Count per Layer')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    cos_changes = [d['cos_change'] for d in layer_data]
    residuals = [d['residual_ratio'] for d in layer_data]
    ax.plot(layers, cos_changes, 'o-', color='#2196F3', label='cos(in, out)', linewidth=1.5)
    ax.plot(layers, residuals, 's-', color='#FF9800', label='residual ratio', linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Value')
    ax.set_title('(c) Layer Dynamics')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q166: LLM as Quantum Circuit (Layer Decomposition)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q166_circuit.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ166 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
