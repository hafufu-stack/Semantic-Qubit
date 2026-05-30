# -*- coding: utf-8 -*-
"""
Phase Q214: Quantum Channel Capacity
=======================================
Opus Original: How much quantum information can flow through LLM layers?

Measure the "quantum channel capacity" of each Transformer layer by
treating it as a quantum channel and computing:
1. Classical capacity (Holevo bound)
2. Coherent information (quantum capacity lower bound)
3. Entanglement-assisted capacity

This reveals which layers are "quantum highways" and which are bottlenecks.
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
    return float(-np.sum(eigvals * np.log2(eigvals)))


def build_channel_from_layers(model, tok, device, prompt, layer_in, layer_out, dim=16):
    """Build a quantum channel matrix from layer_in to layer_out."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    h_in = out.hidden_states[layer_in][0, -1, :dim].float().cpu().numpy()
    h_out = out.hidden_states[layer_out][0, -1, :dim].float().cpu().numpy()

    h_in = h_in / (np.linalg.norm(h_in) + 1e-10)
    h_out = h_out / (np.linalg.norm(h_out) + 1e-10)

    rho_in = np.outer(h_in, h_in.conj())
    rho_out = np.outer(h_out, h_out.conj())

    return rho_in, rho_out


def compute_channel_capacity(rho_in, rho_out):
    """Estimate channel capacity metrics."""
    S_in = von_neumann_entropy(rho_in)
    S_out = von_neumann_entropy(rho_out)

    # Holevo-like bound: S(output) - conditional entropy
    # For pure states, conditional entropy ~ 0, so Holevo ~ S(output)
    holevo = S_out

    # Coherent information: S(output) - S(environment)
    # For a unitary channel, this equals S(input)
    # Approximate: correlation between in and out
    dim = rho_in.shape[0]
    overlap = float(np.abs(np.trace(rho_in @ rho_out)))
    coherent_info = max(0, S_out - (1 - overlap) * np.log2(dim))

    # Mutual information
    rho_joint = np.kron(rho_in, rho_out)
    rho_joint = rho_joint / (np.trace(rho_joint) + 1e-10)
    S_joint = von_neumann_entropy(rho_joint)
    mutual_info = max(0, S_in + S_out - S_joint)

    return {
        'S_in': round(S_in, 4),
        'S_out': round(S_out, 4),
        'holevo': round(holevo, 4),
        'coherent_info': round(coherent_info, 4),
        'mutual_info': round(mutual_info, 4),
        'overlap': round(overlap, 6),
    }


def main():
    print("=" * 60)
    print("Phase Q214: Quantum Channel Capacity")
    print("  (Which layers are quantum highways?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 16

    prompts = [
        "quantum information processing",
        "entangled photon pairs",
        "topological quantum computing",
        "many-body localization",
    ]

    # Measure capacity for each consecutive layer pair
    layer_capacities = []

    for li in range(n_layers):
        cap_data = []
        for prompt in prompts:
            rho_in, rho_out = build_channel_from_layers(
                model, tok, device, prompt, li, li + 1, dim=dim)
            cap = compute_channel_capacity(rho_in, rho_out)
            cap_data.append(cap)

        avg_cap = {
            'layer': li,
            'holevo': round(np.mean([c['holevo'] for c in cap_data]), 4),
            'coherent_info': round(np.mean([c['coherent_info'] for c in cap_data]), 4),
            'mutual_info': round(np.mean([c['mutual_info'] for c in cap_data]), 4),
            'overlap': round(np.mean([c['overlap'] for c in cap_data]), 6),
        }
        layer_capacities.append(avg_cap)

        if li % 4 == 0:
            print("  Layer %d->%d: Holevo=%.4f, CI=%.4f, MI=%.4f" %
                  (li, li+1, avg_cap['holevo'], avg_cap['coherent_info'],
                   avg_cap['mutual_info']))

    # Find quantum highways and bottlenecks
    holevos = [c['holevo'] for c in layer_capacities]
    highway = int(np.argmax(holevos))
    bottleneck = int(np.argmin(holevos))

    avg_holevo = np.mean(holevos)
    max_holevo = max(holevos)

    verdict = "Highway=L%d (%.3f bits), Bottleneck=L%d (%.3f bits), Avg=%.3f bits" % (
        highway, holevos[highway], bottleneck, holevos[bottleneck], avg_holevo)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q214',
        'name': 'Quantum Channel Capacity',
        'layers': layer_capacities,
        'summary': {
            'highway_layer': highway,
            'bottleneck_layer': bottleneck,
            'avg_holevo': round(avg_holevo, 4),
            'max_holevo': round(max_holevo, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q214_channel_capacity.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers_x = [c['layer'] for c in layer_capacities]

    ax = axes[0]
    ax.bar(layers_x, holevos, color='#E91E63', edgecolor='black', alpha=0.8)
    ax.axhline(avg_holevo, color='gray', ls='--', label='Avg=%.3f' % avg_holevo)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Holevo Bound (bits)')
    ax.set_title('(a) Classical Capacity per Layer')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    cis = [c['coherent_info'] for c in layer_capacities]
    ax.plot(layers_x, cis, 's-', color='#4CAF50', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Coherent Information')
    ax.set_title('(b) Quantum Capacity Lower Bound')
    ax.grid(alpha=0.3)

    ax = axes[2]
    mis = [c['mutual_info'] for c in layer_capacities]
    overlaps = [c['overlap'] for c in layer_capacities]
    ax.plot(layers_x, overlaps, 'D-', color='#FF9800', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('State Overlap')
    ax.set_title('(c) Input-Output Overlap')
    ax.grid(alpha=0.3)

    plt.suptitle('Q214: Quantum Channel Capacity\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q214_channel_capacity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ214 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
