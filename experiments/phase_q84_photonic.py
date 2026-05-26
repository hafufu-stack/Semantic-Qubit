# -*- coding: utf-8 -*-
"""Phase Q84: QRAM Photonic Circuit Compilation
Decompose the LLM embedding layer's O(1) data loading into a silicon photonics
interferometer network (Mach-Zehnder mesh) design.
GPU experiment - analyzes embedding matrix structure.
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


def analyze_embedding_svd(model):
    """SVD decomposition of embedding matrix for photonic compilation."""
    embed = model.model.embed_tokens.weight.data.cpu().float().numpy()
    print(f"  Embedding shape: {embed.shape} (vocab x hidden)")

    # Take a representative subset for SVD
    n_tokens = min(1024, embed.shape[0])
    E = embed[:n_tokens]

    # SVD: E = U * S * V^T
    U, S, Vt = np.linalg.svd(E, full_matrices=False)

    # Cumulative energy
    total_energy = np.sum(S ** 2)
    cumulative = np.cumsum(S ** 2) / total_energy

    # Find rank needed for 99% energy
    rank_99 = np.searchsorted(cumulative, 0.99) + 1
    rank_95 = np.searchsorted(cumulative, 0.95) + 1
    rank_90 = np.searchsorted(cumulative, 0.90) + 1

    print(f"  Singular values: top-5 = {S[:5].tolist()}")
    print(f"  Rank for 90% energy: {rank_90}")
    print(f"  Rank for 95% energy: {rank_95}")
    print(f"  Rank for 99% energy: {rank_99}")

    return S, cumulative, rank_90, rank_95, rank_99, U, Vt


def design_mzi_mesh(rank, hidden_dim):
    """Design Mach-Zehnder Interferometer mesh for the photonic circuit."""
    # Clements decomposition: d x d unitary -> d(d-1)/2 MZIs
    n_mzi = rank * (rank - 1) // 2
    n_phase_shifters = rank  # diagonal phases
    total_components = n_mzi + n_phase_shifters

    # Estimate chip area (each MZI ~ 200um x 50um)
    mzi_area_um2 = 200 * 50  # um^2
    total_area_mm2 = total_components * mzi_area_um2 / 1e6

    # Optical loss per MZI ~ 0.1 dB
    loss_per_mzi_dB = 0.1
    max_depth = rank  # longest path through mesh
    total_loss_dB = max_depth * loss_per_mzi_dB

    # Latency (speed of light in silicon waveguide)
    c_silicon = 3e8 / 3.5  # m/s (n_eff ~ 3.5)
    path_length_m = max_depth * 200e-6  # 200um per MZI
    latency_ps = path_length_m / c_silicon * 1e12

    return {
        'rank': rank,
        'n_mzi': n_mzi,
        'n_phase_shifters': n_phase_shifters,
        'total_components': total_components,
        'chip_area_mm2': total_area_mm2,
        'max_loss_dB': total_loss_dB,
        'latency_ps': latency_ps,
        'data_rate_GHz': 1e3 / latency_ps,  # GHz
    }


def main():
    print("=" * 60)
    print("Phase Q84: QRAM Photonic Circuit Compilation")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    print("  Analyzing embedding matrix (SVD)...")
    S, cumulative, rank_90, rank_95, rank_99, U, Vt = analyze_embedding_svd(model)

    # Design photonic circuits at different ranks
    designs = {}
    for label, rank in [('90%', rank_90), ('95%', rank_95), ('99%', rank_99)]:
        design = design_mzi_mesh(rank, d_model)
        designs[label] = design
        print(f"  Design ({label}, rank={rank}):")
        print(f"    MZIs: {design['n_mzi']}, Area: {design['chip_area_mm2']:.1f}mm2")
        print(f"    Loss: {design['max_loss_dB']:.1f}dB, Latency: {design['latency_ps']:.0f}ps")
        print(f"    Data rate: {design['data_rate_GHz']:.1f} GHz")

    # Compare with electronic QRAM
    electronic_qram = {
        'name': 'Electronic SRAM',
        'latency_ps': 1000,  # ~1ns for SRAM access
        'data_rate_GHz': 1.0,
        'area_mm2': 1.0,  # for small SRAM
    }

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) SVD spectrum
    ax = axes[0]
    n_show = min(200, len(S))
    ax.semilogy(range(n_show), S[:n_show], '-', color='#FF5722', linewidth=2)
    ax.axvline(rank_90, color='green', ls='--', alpha=0.5, label=f'90% energy (r={rank_90})')
    ax.axvline(rank_95, color='blue', ls='--', alpha=0.5, label=f'95% energy (r={rank_95})')
    ax.axvline(rank_99, color='red', ls='--', alpha=0.5, label=f'99% energy (r={rank_99})')
    ax.set_xlabel('Singular value index', fontsize=11)
    ax.set_ylabel('Singular value', fontsize=11)
    ax.set_title('(a) Embedding SVD Spectrum\nLow-rank = fewer photonic components',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Photonic chip design comparison
    ax = axes[1]
    labels = list(designs.keys())
    mzis = [designs[l]['n_mzi'] for l in labels]
    areas = [designs[l]['chip_area_mm2'] for l in labels]
    x = range(len(labels))
    bars = ax.bar(x, mzis, color=['#4CAF50', '#2196F3', '#FF5722'],
                  edgecolor='black', alpha=0.85)
    for bar, m, a in zip(bars, mzis, areas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{m} MZIs\n{a:.0f}mm2', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(r={designs[l]["rank"]})' for l in labels])
    ax.set_ylabel('Number of MZI interferometers', fontsize=11)
    ax.set_title('(b) Photonic Chip Complexity\nby Energy Threshold',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # (c) QRAM speed comparison
    ax = axes[2]
    methods = ['Physical\nQRAM\n(proposed)', 'Electronic\nSRAM', 'Photonic\nQRAM\n(S-Qubit)']
    # Physical QRAM doesn't exist yet, estimated at microseconds
    latencies = [1e6, electronic_qram['latency_ps'], designs['95%']['latency_ps']]
    colors = ['#9E9E9E', '#2196F3', '#FF5722']
    bars = ax.bar(methods, latencies, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yscale('log')
    for bar, val in zip(bars, latencies):
        if val >= 1e6:
            label = f'{val/1e6:.0f} us'
        elif val >= 1e3:
            label = f'{val/1e3:.0f} ns'
        else:
            label = f'{val:.0f} ps'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 2,
                label, ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Latency (ps)', fontsize=11)
    speedup = latencies[0] / latencies[2]
    ax.set_title(f'(c) QRAM Latency Comparison\nPhotonic: {speedup:.0f}x faster than physical',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('QRAM Photonic Circuit: From LLM Embeddings to Silicon Photonics',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q84_photonic.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q84', 'name': 'QRAM Photonic Circuit Compilation',
        'd_model': d_model,
        'rank_90': rank_90, 'rank_95': rank_95, 'rank_99': rank_99,
        'designs': designs,
        'photonic_latency_ps': designs['95%']['latency_ps'],
        'photonic_data_rate_GHz': designs['95%']['data_rate_GHz'],
        'speedup_vs_physical_qram': latencies[0] / latencies[2],
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q84_photonic.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved. Elapsed: {elapsed:.1f}s")
    return results


if __name__ == '__main__':
    main()
