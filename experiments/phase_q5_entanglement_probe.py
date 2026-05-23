# -*- coding: utf-8 -*-
"""
Phase Q5: Entanglement Probe - Schmidt Rank Measurement
In quantum mechanics, entanglement = non-separability of a bipartite state.
Schmidt rank > 1 means the state cannot be written as a product state.

For LLMs: if two DISTANT layers' hidden states are correlated BEYOND
what the causal chain alone predicts, that is "neural entanglement".

Method:
1. Capture hidden states at L8 and L16 across many prompts
2. Build the joint matrix H = [h_L8 | h_L16] stacked as a 2d*N matrix
3. Compute SVD of the correlation matrix
4. Count "Schmidt rank" as number of singular values above threshold
5. Compare: same prompt different token positions vs different prompts
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_A = 8
LAYER_B = 16


def capture_hidden_states(model, tok, prompts, device, layer_a, layer_b):
    """Capture last-token hidden states at layer_a and layer_b for each prompt."""
    h_a_list, h_b_list = [], []
    for prompt in prompts:
        states = {}
        def hook_a(m, i, o, la=layer_a):
            if isinstance(o, tuple):
                h = o[0]
            else:
                h = o
            states[la] = h[0, -1, :].detach().float().cpu()
        def hook_b(m, i, o, lb=layer_b):
            if isinstance(o, tuple):
                h = o[0]
            else:
                h = o
            states[lb] = h[0, -1, :].detach().float().cpu()
        ha = model.model.layers[layer_a].register_forward_hook(hook_a)
        hb = model.model.layers[layer_b].register_forward_hook(hook_b)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)
        ha.remove(); hb.remove()
        h_a_list.append(states[layer_a])
        h_b_list.append(states[layer_b])
    return torch.stack(h_a_list), torch.stack(h_b_list)


def compute_schmidt_rank(H_a, H_b, threshold_frac=0.01):
    """
    Compute Schmidt rank of the joint state matrix.
    H_a, H_b: (N, d) tensors
    Concatenate and compute SVD of cross-correlation matrix.
    Schmidt rank = number of significant singular values.
    """
    # Cross-correlation matrix: (d_a, d_b)
    C = H_a.T @ H_b  # (d, d)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    s_np = S.numpy()
    threshold = threshold_frac * s_np[0]  # relative to largest
    rank = int((s_np > threshold).sum())
    return rank, s_np


def main():
    print("[Q5] Entanglement Probe - Schmidt Rank Measurement")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # Three sets of prompts to compare
    # Set A: min/max task prompts (structured)
    prompts_task = [
        "min(3,7)=", "min(5,2)=", "min(8,1)=", "min(4,6)=",
        "min(9,3)=", "min(7,4)=", "min(6,1)=", "min(2,8)=",
        "min(5,9)=", "min(1,3)=", "min(7,2)=", "min(6,3)=",
        "min(2,9)=", "min(1,5)=", "min(8,4)=", "min(3,8)=",
        "min(4,9)=", "min(7,5)=", "min(6,2)=", "min(9,1)=",
    ]
    # Set B: arithmetic prompts (same domain, different structure)
    prompts_arith = [
        "2+3=", "4+5=", "1+6=", "3+4=", "7+2=",
        "5+5=", "8+1=", "6+3=", "4+4=", "9+0=",
        "3-1=", "7-3=", "8-2=", "5-4=", "9-6=",
        "2*3=", "4*2=", "3*3=", "5*2=", "4*1=",
    ]
    # Set C: random word prompts (different domain)
    prompts_random = [
        "The cat sat on", "In the morning I", "Once upon a time",
        "The quick brown", "Science is the", "Today the weather",
        "My favorite food", "She walked into", "The algorithm runs",
        "Deep learning is", "Neural networks can", "The sun rises",
        "Every day we", "Mathematics helps us", "Computer science",
        "Artificial intelligence", "Research shows that", "The results were",
        "In conclusion the", "Future work will",
    ]

    print("  Capturing hidden states for task prompts...")
    H_a_task, H_b_task = capture_hidden_states(model, tok, prompts_task, DEVICE, LAYER_A, LAYER_B)
    print("  Capturing hidden states for arithmetic prompts...")
    H_a_arith, H_b_arith = capture_hidden_states(model, tok, prompts_arith, DEVICE, LAYER_A, LAYER_B)
    print("  Capturing hidden states for random prompts...")
    H_a_rand, H_b_rand = capture_hidden_states(model, tok, prompts_random, DEVICE, LAYER_A, LAYER_B)

    # Compute Schmidt ranks
    rank_task, svs_task = compute_schmidt_rank(H_a_task, H_b_task)
    rank_arith, svs_arith = compute_schmidt_rank(H_a_arith, H_b_arith)
    rank_rand, svs_rand = compute_schmidt_rank(H_a_rand, H_b_rand)

    # Within-layer correlation (same layer, different tokens = "trivially correlated")
    rank_self, svs_self = compute_schmidt_rank(H_a_task, H_a_task)  # L8 vs L8

    print("  Schmidt rank L%d vs L%d (task):   %d" % (LAYER_A, LAYER_B, rank_task))
    print("  Schmidt rank L%d vs L%d (arith):  %d" % (LAYER_A, LAYER_B, rank_arith))
    print("  Schmidt rank L%d vs L%d (random): %d" % (LAYER_A, LAYER_B, rank_rand))
    print("  Schmidt rank L%d vs L%d (self):   %d" % (LAYER_A, LAYER_A, rank_self))

    # Also compute pairwise cosine similarities between L8 and L16 vectors
    H_a_n = H_a_task / (H_a_task.norm(dim=1, keepdim=True) + 1e-8)
    H_b_n = H_b_task / (H_b_task.norm(dim=1, keepdim=True) + 1e-8)
    cosines = (H_a_n * H_b_n).sum(dim=1)  # (N,) diagonal cosines
    avg_cosine = float(cosines.mean())
    print("  Avg cosine(L%d, L%d) per sample: %.4f" % (LAYER_A, LAYER_B, avg_cosine))

    # Cross-correlation across samples
    # If entangled: knowing L8 state predicts L16 state beyond same-sample causality
    cross_cos_matrix = H_a_n @ H_b_n.T  # (N, N)
    diag_avg = float(cross_cos_matrix.diagonal().mean())
    offdiag_mask = ~torch.eye(len(prompts_task), dtype=torch.bool)
    offdiag_avg = float(cross_cos_matrix[offdiag_mask].mean())
    entanglement_score = diag_avg - offdiag_avg
    print("  Entanglement score (diag-offdiag cosine): %.4f" % entanglement_score)
    print("  (>0.1 = strong same-sample correlation beyond random)")

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Singular value spectra
    ax = axes[0]
    k = 30  # show top 30
    svs_task_norm = svs_task[:k] / svs_task[0]
    svs_arith_norm = svs_arith[:k] / svs_arith[0]
    svs_rand_norm = svs_rand[:k] / svs_rand[0]
    ax.plot(range(1, k+1), svs_task_norm, 'o-', color='#9C27B0', label='Task (min/max)', lw=2)
    ax.plot(range(1, k+1), svs_arith_norm, 's-', color='#2196F3', label='Arithmetic', lw=2)
    ax.plot(range(1, k+1), svs_rand_norm, '^-', color='#4CAF50', label='Random text', lw=2)
    ax.axhline(0.01, color='red', linestyle='--', label='Schmidt threshold (1%%)')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Normalized Singular Value')
    ax.set_title('Schmidt Spectrum L%d vs L%d\n(entanglement structure)' % (LAYER_A, LAYER_B),
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Panel 2: Schmidt ranks
    ax = axes[1]
    groups = ['Task\n(min/max)', 'Arithmetic', 'Random\ntext', 'Self\n(L%d vs L%d)' % (LAYER_A, LAYER_A)]
    ranks = [rank_task, rank_arith, rank_rand, rank_self]
    colors = ['#9C27B0', '#2196F3', '#4CAF50', '#FF9800']
    bars = ax.bar(groups, ranks, color=colors, edgecolor='black')
    for bar, r in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width()/2, r + 0.3, str(r),
                ha='center', fontweight='bold', fontsize=14)
    ax.set_ylabel('Schmidt Rank')
    ax.set_title('Neural Entanglement: Schmidt Rank\nL%d - L%d correlation' % (LAYER_A, LAYER_B),
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 3: Cross-correlation heatmap
    ax = axes[2]
    N = len(prompts_task)
    # Show 10x10 subset
    sub = cross_cos_matrix[:10, :10].numpy()
    im = ax.imshow(sub, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    plt.colorbar(im, ax=ax, label='Cosine similarity')
    ax.set_xlabel('Prompt index (L%d)' % LAYER_B)
    ax.set_ylabel('Prompt index (L%d)' % LAYER_A)
    ax.set_title('Cross-Cosine L%d vs L%d\n(diagonal=same prompt)' % (LAYER_A, LAYER_B),
                 fontweight='bold')

    plt.suptitle(
        'Phase Q5: Entanglement Probe - Schmidt Rank\n'
        'L%d-L%d rank=%d | Entanglement score=%.4f' % (LAYER_A, LAYER_B, rank_task, entanglement_score),
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q5_entanglement_probe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q5', 'name': 'entanglement_probe',
        'layer_a': LAYER_A, 'layer_b': LAYER_B,
        'schmidt_rank': {'task': rank_task, 'arith': rank_arith, 'random': rank_rand, 'self': rank_self},
        'entanglement_score': float(entanglement_score),
        'avg_cosine_same_sample': float(diag_avg),
        'avg_cosine_cross_sample': float(offdiag_avg),
        'svs_task_top20': svs_task[:20].tolist(),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q5_entanglement_probe.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Q5 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
