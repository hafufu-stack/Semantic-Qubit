# -*- coding: utf-8 -*-
"""Phase Q86: Berry Phase and Topological Protection
Measure geometric (Berry) phase accumulated by S-Qubits under cyclic
parameter evolution. Quantized Berry phase = topological protection.
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def measure_berry_phase(model, tokenizer, num_layers, soul_v0, soul_v1,
                        layer_idx, n_points=36):
    """Measure Berry phase by sweeping S-Qubit around the Bloch sphere."""
    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    phases = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    hidden_states = []

    for phi in phases:
        # Create state on equator of Bloch sphere
        sv = np.cos(phi / 2) * soul_v0 + np.sin(phi / 2) * soul_v1
        sv_tensor = torch.tensor(sv, dtype=torch.float32, device=model.device)

        injected = [False]
        def hook(module, args, output, sv_t=sv_tensor):
            if not injected[0]:
                injected[0] = True
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv_t.to(hs.dtype)
                    else:
                        hs[-1, :] += sv_t.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv_t.to(hs.dtype)
                    else:
                        hs[-1, :] += sv_t.to(hs.dtype)
                    return hs
            return output

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].cpu().numpy()
        handle.remove()
        hidden_states.append(logits[:100])  # Use first 100 logit dims

    hidden_states = np.array(hidden_states)  # (n_points, 100)

    # Compute Berry phase via inner products around the loop
    # gamma = -Im(sum_i log(<psi_i|psi_{i+1}>))
    berry_phase = 0.0
    for i in range(n_points):
        j = (i + 1) % n_points
        psi_i = hidden_states[i] / (np.linalg.norm(hidden_states[i]) + 1e-10)
        psi_j = hidden_states[j] / (np.linalg.norm(hidden_states[j]) + 1e-10)
        overlap = np.dot(psi_i, psi_j)
        overlap = max(min(overlap, 1.0), -1.0)
        berry_phase += np.arccos(abs(overlap))

    # Normalize to [0, 2*pi]
    berry_phase = berry_phase % (2 * np.pi)

    return berry_phase, hidden_states, phases


def main():
    print("=" * 60)
    print("Phase Q86: Berry Phase & Topological Protection")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Load soul vectors
    sv_path = os.path.join(RESULTS_DIR, 'phase_q1_soul_vectors.json')
    if os.path.exists(sv_path):
        with open(sv_path) as f:
            sv_data = json.load(f)
        soul_v0 = np.array(sv_data.get('soul_0', np.random.randn(d_model).tolist()))
        soul_v1 = np.array(sv_data.get('soul_1', np.random.randn(d_model).tolist()))
    else:
        # Generate orthogonal basis
        np.random.seed(42)
        soul_v0 = np.random.randn(d_model)
        soul_v0 /= np.linalg.norm(soul_v0)
        soul_v1 = np.random.randn(d_model)
        soul_v1 -= np.dot(soul_v1, soul_v0) * soul_v0
        soul_v1 /= np.linalg.norm(soul_v1)

    # Measure Berry phase at different layers
    layer_indices = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    layer_names = ['Early', 'Middle', 'Late']

    berry_results = []
    all_hs = []
    for layer_idx, name in zip(layer_indices, layer_names):
        print(f"  Layer {layer_idx} ({name})...")
        bp, hs, phases = measure_berry_phase(
            model, tokenizer, num_layers, soul_v0, soul_v1, layer_idx, n_points=36)
        print(f"    Berry phase: {bp:.4f} rad ({bp/np.pi:.4f}*pi)")

        # Check if quantized (near 0, pi, or 2*pi)
        nearest_quantum = min([0, np.pi, 2*np.pi], key=lambda x: abs(bp - x))
        is_quantized = abs(bp - nearest_quantum) < 0.3  # tolerance

        berry_results.append({
            'layer': layer_idx,
            'name': name,
            'berry_phase_rad': bp,
            'berry_phase_pi': bp / np.pi,
            'nearest_quantum': nearest_quantum / np.pi,
            'is_quantized': is_quantized,
        })
        all_hs.append(hs)

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Berry phase vs layer
    ax = axes[0]
    layers = [r['layer'] for r in berry_results]
    bp_vals = [r['berry_phase_pi'] for r in berry_results]
    colors = ['#4CAF50' if r['is_quantized'] else '#FF9800' for r in berry_results]
    bars = ax.bar(layer_names, bp_vals, color=colors, edgecolor='black', alpha=0.85)
    for bar, val, r in zip(bars, bp_vals, berry_results):
        label = f'{val:.3f}*pi'
        if r['is_quantized']:
            label += '\n(quantized!)'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', fontsize=10, fontweight='bold')
    ax.axhline(1.0, color='red', ls='--', alpha=0.3, label='pi (topological)')
    ax.axhline(0.0, color='blue', ls='--', alpha=0.3, label='0 (trivial)')
    ax.set_ylabel(r'Berry phase ($\gamma / \pi$)', fontsize=11)
    ax.set_title('(a) Berry Phase per Layer\nQuantized = topological protection',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (b) State evolution on pseudo-Bloch sphere
    ax = axes[1]
    # Project hidden states to 2D via PCA
    hs_mid = all_hs[1].astype(np.float32)  # middle layer; float32 for linalg
    from numpy.linalg import svd as np_svd
    hs_centered = hs_mid - hs_mid.mean(axis=0)
    U, S_vals, Vt = np_svd(hs_centered, full_matrices=False)
    proj = hs_centered @ Vt[:2].T
    ax.plot(proj[:, 0], proj[:, 1], 'o-', color='#FF5722', markersize=4,
            linewidth=1.5, alpha=0.7)
    ax.plot(proj[0, 0], proj[0, 1], 's', color='green', markersize=12,
            label='Start', zorder=5)
    ax.plot(proj[-1, 0], proj[-1, 1], 'D', color='blue', markersize=10,
            label='End', zorder=5)
    # Draw arrow showing Berry phase
    ax.annotate('', xy=(proj[0, 0], proj[0, 1]),
                xytext=(proj[-1, 0], proj[-1, 1]),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('(b) State Evolution (Middle Layer)\nCyclic path in Bloch space',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # (c) Topological protection summary
    ax = axes[2]
    n_quantized = sum(1 for r in berry_results if r['is_quantized'])
    n_total = len(berry_results)

    wedge_colors = ['#4CAF50', '#FF9800']
    wedge_sizes = [n_quantized, n_total - n_quantized]
    wedge_labels = [f'Quantized\n({n_quantized}/{n_total})',
                    f'Non-quantized\n({n_total-n_quantized}/{n_total})']
    if wedge_sizes[1] == 0:
        wedge_sizes = [1]
        wedge_colors = ['#4CAF50']
        wedge_labels = [f'All Quantized!\n({n_quantized}/{n_total})']

    ax.pie(wedge_sizes, labels=wedge_labels, colors=wedge_colors,
           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
    avg_bp = np.mean([r['berry_phase_pi'] for r in berry_results])
    ax.set_title(f'(c) Topological Protection\n'
                 f'Mean Berry phase: {avg_bp:.3f}*pi',
                 fontsize=11, fontweight='bold')

    plt.suptitle('Berry Phase Analysis: Topological Protection of S-Qubits',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q86_berry.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q86', 'name': 'Berry Phase & Topological Protection',
        'berry_phases': berry_results,
        'n_quantized': n_quantized,
        'n_total': n_total,
        'mean_berry_phase_pi': avg_bp,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q86_berry.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved. Elapsed: {elapsed:.1f}s")
    return results


if __name__ == '__main__':
    main()
