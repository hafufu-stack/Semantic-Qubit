# -*- coding: utf-8 -*-
"""
Phase Q152: Attention Head Topology Analysis
==============================================
WHY does LLM win on SYK (all-to-all) but lose on Ising (local)?

Instead of extracting runtime attention patterns (which can fail),
analyze the WEIGHT matrices of each attention head directly.

Measure: weight matrix entropy, effective rank, and connectivity pattern.
High entropy / high rank = all-to-all (SYK-compatible)
Low entropy / low rank = local/sparse (Ising-compatible)
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


def effective_rank(W):
    """Effective rank via singular value entropy.
    High = all-to-all, Low = low-rank/local.
    """
    s = np.linalg.svd(W, compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    ent = -float(np.sum(p * np.log(p)))
    return float(np.exp(ent))


def weight_entropy(W):
    """Shannon entropy of absolute weight distribution (binned)."""
    w_abs = np.abs(W).flatten()
    hist, _ = np.histogram(w_abs, bins=50, density=True)
    hist = hist[hist > 0]
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def sparsity(W, threshold=0.01):
    """Fraction of near-zero weights."""
    return float(np.mean(np.abs(W) < threshold))


def main():
    print("=" * 60)
    print("Phase Q152: Attention Head Topology Analysis")
    print("  (Weight Matrix Structure)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = hidden_size // n_heads

    all_head_data = []

    for li in range(n_layers):
        layer = model.model.layers[li]
        attn = layer.self_attn

        # Get Q, K, V projection weight matrices
        W_q = attn.q_proj.weight.detach().float().cpu().numpy()
        W_k = attn.k_proj.weight.detach().float().cpu().numpy()
        W_v = attn.v_proj.weight.detach().float().cpu().numpy()

        # W_q might be (n_heads * head_dim, hidden)
        # Split into per-head blocks
        n_q_heads = W_q.shape[0] // head_dim
        n_kv_heads = W_k.shape[0] // head_dim  # GQA: may differ

        for h in range(n_q_heads):
            start = h * head_dim
            end = start + head_dim
            Wq_h = W_q[start:end, :]

            # For GQA, KV heads may be fewer
            kv_h = h % n_kv_heads
            kv_start = kv_h * head_dim
            kv_end = kv_start + head_dim
            Wk_h = W_k[kv_start:kv_end, :]
            Wv_h = W_v[kv_start:kv_end, :]

            # QK^T product determines attention pattern
            # Effective attention template: W_q @ W_k^T
            QK = Wq_h @ Wk_h.T  # (head_dim, head_dim)

            eff_rank = effective_rank(QK)
            w_ent = weight_entropy(QK)
            sparse = sparsity(QK)

            # V matrix rank determines output expressiveness
            v_rank = effective_rank(Wv_h)

            all_head_data.append({
                'layer': int(li),
                'head': int(h),
                'qk_effective_rank': round(eff_rank, 2),
                'qk_entropy': round(w_ent, 4),
                'qk_sparsity': round(sparse, 4),
                'v_effective_rank': round(v_rank, 2),
            })

        if li % 7 == 0:
            layer_data = [d for d in all_head_data if d['layer'] == li]
            avg_rank = np.mean([d['qk_effective_rank'] for d in layer_data])
            avg_ent = np.mean([d['qk_entropy'] for d in layer_data])
            avg_sparse = np.mean([d['qk_sparsity'] for d in layer_data])
            print("  Layer %2d: avg_rank=%.1f, entropy=%.2f, sparsity=%.3f" %
                  (li, avg_rank, avg_ent, avg_sparse))

    # Classify heads
    all_ranks = [d['qk_effective_rank'] for d in all_head_data]
    all_ents = [d['qk_entropy'] for d in all_head_data]
    median_rank = float(np.median(all_ranks))
    median_ent = float(np.median(all_ents))

    n_global = sum(1 for d in all_head_data
                   if d['qk_effective_rank'] > median_rank and d['qk_entropy'] > median_ent)
    n_local = sum(1 for d in all_head_data
                  if d['qk_effective_rank'] < median_rank and d['qk_entropy'] < median_ent)
    n_total = len(all_head_data)

    print("\n--- Head Topology Summary ---")
    print("  Total heads: %d" % n_total)
    print("  Global (SYK-compatible, high rank+entropy): %d (%.1f%%)" %
          (n_global, n_global / n_total * 100))
    print("  Local (Ising-compatible, low rank+entropy): %d (%.1f%%)" %
          (n_local, n_local / n_total * 100))
    print("  Mixed: %d (%.1f%%)" %
          (n_total - n_global - n_local, (n_total - n_global - n_local) / n_total * 100))
    print("  Median effective rank: %.1f / %d (max)" % (median_rank, head_dim))
    print("  Median entropy: %.2f bits" % median_ent)

    # Layer-wise trend
    layer_summaries = []
    for li in range(n_layers):
        ld = [d for d in all_head_data if d['layer'] == li]
        if not ld:
            continue
        layer_summaries.append({
            'layer': int(li),
            'avg_rank': round(float(np.mean([d['qk_effective_rank'] for d in ld])), 2),
            'avg_entropy': round(float(np.mean([d['qk_entropy'] for d in ld])), 4),
            'avg_sparsity': round(float(np.mean([d['qk_sparsity'] for d in ld])), 4),
            'avg_v_rank': round(float(np.mean([d['v_effective_rank'] for d in ld])), 2),
            'n_global': int(sum(1 for d in ld
                                if d['qk_effective_rank'] > median_rank
                                and d['qk_entropy'] > median_ent)),
            'n_heads': len(ld),
        })

    # Save
    results = {
        'phase': 'Q152',
        'name': 'Attention Head Topology (Weight Analysis)',
        'summary': {
            'total_heads': n_total,
            'global_pct': round(n_global / n_total * 100, 2),
            'local_pct': round(n_local / n_total * 100, 2),
            'median_rank': round(median_rank, 2),
            'median_entropy': round(median_ent, 4),
            'head_dim': int(head_dim),
        },
        'layer_summaries': layer_summaries,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q152_topology.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Rank vs Entropy scatter
    ax = axes[0]
    colors_scatter = []
    for d in all_head_data:
        frac = d['layer'] / n_layers
        colors_scatter.append(plt.cm.viridis(frac))
    ax.scatter([d['qk_effective_rank'] for d in all_head_data],
               [d['qk_entropy'] for d in all_head_data],
               c=colors_scatter, alpha=0.4, s=10)
    ax.axvline(median_rank, color='red', ls='--', alpha=0.5, label='Median rank')
    ax.axhline(median_ent, color='blue', ls='--', alpha=0.5, label='Median entropy')
    ax.set_xlabel('QK Effective Rank')
    ax.set_ylabel('QK Weight Entropy (bits)')
    ax.set_title('(a) Head Topology Map\n(color=layer depth)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.text(0.95, 0.95, 'GLOBAL\n(SYK)', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='red', fontweight='bold')
    ax.text(0.05, 0.05, 'LOCAL\n(Ising)', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=9, color='blue', fontweight='bold')

    # (b) Layer-wise rank profile
    ax = axes[1]
    layers_x = [d['layer'] for d in layer_summaries]
    ranks_y = [d['avg_rank'] for d in layer_summaries]
    ents_y = [d['avg_entropy'] for d in layer_summaries]
    ax.plot(layers_x, ranks_y, 'o-', color='#E91E63', label='Avg QK rank', linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(layers_x, ents_y, 's-', color='#2196F3', label='Avg entropy', linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Effective Rank', color='#E91E63')
    ax2.set_ylabel('Entropy (bits)', color='#2196F3')
    ax.set_title('(b) Topology Across Layers')
    ax.grid(alpha=0.3)

    # (c) Global vs local distribution
    ax = axes[2]
    global_counts = [d['n_global'] for d in layer_summaries]
    local_counts = [d['n_heads'] - d['n_global'] for d in layer_summaries]
    ax.bar(layers_x, global_counts, color='#E91E63', label='Global (SYK)', alpha=0.7)
    ax.bar(layers_x, local_counts, bottom=global_counts,
           color='#4CAF50', label='Local (Ising)', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of heads')
    ax.set_title('(c) Global vs Local Heads')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q152: Attention Head Topology (Why SYK Wins)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q152_topology.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ152 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
