# -*- coding: utf-8 -*-
"""
Phase Q214b: Quantum Channel Capacity (Fixed)
===============================================
Q214 measured zero because pure states have zero entropy.
Fix: use mixed states (average over multiple prompts/tokens) to
build realistic quantum channels between layers.
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


def von_neumann_entropy(rho):
    eigvals = np.real(np.linalg.eigvalsh(rho))
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log2(eigvals)))


def build_mixed_channel(model, tok, device, prompts, layer_in, layer_out, dim=8):
    """Build mixed-state density matrices by averaging over multiple prompts."""
    rho_in = np.zeros((dim, dim), dtype=complex)
    rho_out = np.zeros((dim, dim), dtype=complex)

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        h_in = out.hidden_states[layer_in][0, -1, :dim].float().cpu().numpy()
        h_out = out.hidden_states[layer_out][0, -1, :dim].float().cpu().numpy()

        h_in = h_in / (np.linalg.norm(h_in) + 1e-10)
        h_out = h_out / (np.linalg.norm(h_out) + 1e-10)

        rho_in += np.outer(h_in, h_in.conj())
        rho_out += np.outer(h_out, h_out.conj())

    rho_in /= len(prompts)
    rho_out /= len(prompts)
    return rho_in, rho_out


def main():
    print("=" * 60)
    print("Phase Q214b: Quantum Channel Capacity (Fixed)")
    print("  (Mixed states for proper entropy measurement)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 8

    prompt_sets = [
        "quantum entanglement", "classical mechanics", "protein folding",
        "stock market crash", "neural network training", "black hole entropy",
        "photosynthesis efficiency", "superconducting circuits",
        "DNA replication", "gravitational waves", "machine learning",
        "chemical reaction", "thermodynamic equilibrium", "dark matter",
        "quantum computing error", "biological evolution",
    ]

    layer_results = []
    for li in range(n_layers):
        rho_in, rho_out = build_mixed_channel(
            model, tok, device, prompt_sets, li, li + 1, dim=dim)

        S_in = von_neumann_entropy(rho_in)
        S_out = von_neumann_entropy(rho_out)

        # Purity (how mixed)
        purity_in = float(np.real(np.trace(rho_in @ rho_in)))
        purity_out = float(np.real(np.trace(rho_out @ rho_out)))

        # Overlap (channel fidelity)
        overlap = float(np.abs(np.trace(rho_in @ rho_out)))

        # Holevo capacity estimate
        holevo = S_out

        # Information gain/loss
        delta_S = S_out - S_in

        layer_results.append({
            'layer': li,
            'S_in': round(S_in, 4),
            'S_out': round(S_out, 4),
            'delta_S': round(delta_S, 4),
            'holevo': round(holevo, 4),
            'purity_in': round(purity_in, 4),
            'purity_out': round(purity_out, 4),
            'overlap': round(overlap, 6),
        })

        if li % 4 == 0:
            print("  L%d->%d: S_in=%.3f, S_out=%.3f, dS=%.3f, purity=%.3f" %
                  (li, li+1, S_in, S_out, delta_S, purity_out))

    # Find highway and bottleneck
    holevos = [r['holevo'] for r in layer_results]
    highway = int(np.argmax(holevos))
    bottleneck = int(np.argmin(holevos))
    avg_holevo = np.mean(holevos)

    # Information flow direction
    delta_Ss = [r['delta_S'] for r in layer_results]
    n_compressing = sum(1 for d in delta_Ss if d < -0.01)
    n_expanding = sum(1 for d in delta_Ss if d > 0.01)

    verdict = "Highway=L%d (%.3f bits), Bottleneck=L%d (%.3f bits), %d compressing/%d expanding layers" % (
        highway, holevos[highway], bottleneck, holevos[bottleneck],
        n_compressing, n_expanding)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q214b',
        'name': 'Quantum Channel Capacity (Fixed)',
        'layers': layer_results,
        'summary': {
            'highway_layer': highway,
            'bottleneck_layer': bottleneck,
            'avg_holevo': round(avg_holevo, 4),
            'n_compressing': n_compressing,
            'n_expanding': n_expanding,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q214b_channel_capacity.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers_x = [r['layer'] for r in layer_results]

    ax = axes[0][0]
    ax.bar(layers_x, holevos, color='#E91E63', edgecolor='black', alpha=0.8)
    ax.axhline(avg_holevo, color='gray', ls='--', label='Avg=%.3f' % avg_holevo)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Holevo Bound (bits)')
    ax.set_title('(a) Channel Capacity per Layer')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[0][1]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in delta_Ss]
    ax.bar(layers_x, delta_Ss, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Delta S (bits)')
    ax.set_title('(b) Information Gain/Loss per Layer')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1][0]
    purities = [r['purity_out'] for r in layer_results]
    ax.plot(layers_x, purities, 's-', color='#FF9800', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Purity')
    ax.set_title('(c) State Purity across Layers')
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    overlaps = [r['overlap'] for r in layer_results]
    ax.plot(layers_x, overlaps, 'D-', color='#2196F3', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Overlap')
    ax.set_title('(d) Input-Output State Overlap')
    ax.grid(alpha=0.3)

    plt.suptitle('Q214b: Quantum Channel Capacity (Fixed)\n%s' % verdict[:70],
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q214b_channel_capacity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ214b complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
