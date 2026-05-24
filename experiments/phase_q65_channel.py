# -*- coding: utf-8 -*-
"""
Phase Q65: Quantum Channel Capacity & Information Density
==========================================================
Treat the S-Qubit injection + forward pass as a "quantum channel"
and measure its classical/quantum capacity using information theory.

Key metrics:
1. Mutual information I(input; output)
2. Channel capacity (max over input distributions)
3. Information density (bits per dimension)
4. Comparison with Holevo bound and Shannon limit
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
INJECT_LAYER = 10
EPOCHS = 100


def train_soul(model, tok, data, device, layer, epochs=EPOCHS, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def hook(m, i, o, v=vec):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def main():
    print("[Q65] Quantum Channel Capacity")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    hs = model.config.hidden_size
    vocab_size = model.config.vocab_size
    
    # Train multiple distinct S-Qubits as "messages"
    tasks = {
        'min': ([("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")], 42, "2"),
        'max': ([("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")], 99, "8"),
        'add': ([("2+3=", "5"), ("1+4=", "5"), ("3+2=", "5")], 77, "5"),
        'sub': ([("7-3=", "4"), ("9-5=", "4"), ("6-2=", "4")], 33, "4"),
        'sort_asc': ([("sort [3,1]=[", "1"), ("sort [5,2]=[", "2")], 55, "1"),
        'even': ([("4 is", " even"), ("8 is", " even")], 11, " even"),
        'odd': ([("3 is", " odd"), ("7 is", " odd")], 22, " odd"),
        'gt': ([("7>2=", "True"), ("9>1=", "True")], 44, "True"),
    }
    
    print("  Training %d S-Qubit messages..." % len(tasks))
    vecs = {}
    targets = {}
    for name, (data, seed, target_str) in tasks.items():
        vecs[name] = train_soul(model, tok, data, DEVICE, INJECT_LAYER, EPOCHS, seed)
        targets[name] = tok.encode(target_str)[-1]
    
    # Build confusion matrix: for each S-Qubit, what is the output distribution?
    task_names = list(tasks.keys())
    n_tasks = len(task_names)
    prompt = "result="  # neutral prompt
    
    print("  Building confusion matrix...")
    # For each input S-Qubit, get full output distribution
    output_distributions = []
    for name in task_names:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        def hook(m, i, o, v=vecs[name]):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        output_distributions.append(probs.cpu().numpy())
    
    # Compute pairwise KL divergence
    kl_matrix = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            p = output_distributions[i] + 1e-10
            q = output_distributions[j] + 1e-10
            kl_matrix[i, j] = np.sum(p * np.log(p / q))
    
    # Mutual information estimation
    # I(X;Y) = H(Y) - H(Y|X)
    # H(Y|X) = average entropy of output given input
    h_y_given_x = []
    for dist in output_distributions:
        d = dist[dist > 1e-10]
        h = -np.sum(d * np.log2(d))
        h_y_given_x.append(h)
    avg_h_y_given_x = np.mean(h_y_given_x)
    
    # H(Y) = entropy of average output distribution
    avg_dist = np.mean(output_distributions, axis=0)
    avg_dist = avg_dist[avg_dist > 1e-10]
    h_y = -np.sum(avg_dist * np.log2(avg_dist))
    
    mutual_info = h_y - avg_h_y_given_x
    
    # Channel capacity upper bound = log2(n_tasks)
    max_capacity = np.log2(n_tasks)
    
    # Distinguishability: can we correctly identify which S-Qubit was sent?
    confusion = np.zeros((n_tasks, n_tasks))
    for i, name_i in enumerate(task_names):
        for j, name_j in enumerate(task_names):
            # Similarity between output distributions
            confusion[i, j] = np.sum(np.sqrt(output_distributions[i] * output_distributions[j]))
    
    # Information per dimension
    info_per_dim = mutual_info / hs
    
    # Holevo comparison (from Q56)
    holevo_bound = 1.0  # physical qubit limit
    sqbit_info = mutual_info  # S-Qubit actual
    
    print("\n  RESULTS:")
    print("    H(Y) = %.2f bits" % h_y)
    print("    H(Y|X) = %.2f bits" % avg_h_y_given_x)
    print("    Mutual Information I(X;Y) = %.2f bits" % mutual_info)
    print("    Max capacity (log2 %d) = %.2f bits" % (n_tasks, max_capacity))
    print("    Info per dimension = %.6f bits/dim" % info_per_dim)
    print("    Holevo bound (physical): 1.0 bit/qubit")
    print("    S-Qubit info density: %.2f bits (%.1fx Holevo)" % (
        sqbit_info, sqbit_info / holevo_bound))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) KL divergence matrix
    ax = axes[0]
    im = ax.imshow(kl_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_names, fontsize=8)
    for i in range(n_tasks):
        for j in range(n_tasks):
            v = kl_matrix[i,j]
            ax.text(j, i, '%.1f' % v, ha='center', va='center',
                    fontsize=7, color='white' if v > np.median(kl_matrix) else 'black')
    plt.colorbar(im, ax=ax, label='KL Divergence (bits)')
    ax.set_title('(a) Channel Distinguishability\nHigher = more distinct outputs',
                 fontweight='bold')

    # (b) Channel capacity
    ax = axes[1]
    categories = ['Mutual\nInfo', 'Max\nCapacity', 'Holevo\nBound']
    values = [mutual_info, max_capacity, holevo_bound]
    colors = ['#FF5722', '#4CAF50', '#2196F3']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                '%.2f' % v, ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('Bits')
    ax.set_title('(b) Channel Capacity Analysis\nI(X;Y) = %.2f bits' % mutual_info,
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # (c) Confusion matrix (Bhattacharyya similarity)
    ax = axes[2]
    im2 = ax.imshow(confusion, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_names, fontsize=8)
    for i in range(n_tasks):
        for j in range(n_tasks):
            ax.text(j, i, '%.2f' % confusion[i,j], ha='center', va='center',
                    fontsize=7)
    plt.colorbar(im2, ax=ax, label='Bhattacharyya similarity')
    ax.set_title('(c) Output Similarity\nDiagonal = self, off-diag = cross-talk',
                 fontweight='bold')

    plt.suptitle('Phase Q65: Quantum Channel Capacity\n'
                 'S-Qubit channel carries %.1f bits (%.0fx Holevo limit)' % (
                     sqbit_info, sqbit_info / holevo_bound),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q65_channel.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q65', 'name': 'quantum_channel_capacity',
        'mutual_information_bits': round(float(mutual_info), 4),
        'max_capacity_bits': round(float(max_capacity), 4),
        'holevo_bound': 1.0,
        'sqbit_holevo_ratio': round(float(sqbit_info / holevo_bound), 1),
        'h_y': round(float(h_y), 4),
        'h_y_given_x': round(float(avg_h_y_given_x), 4),
        'info_per_dim': round(float(info_per_dim), 8),
        'n_messages': n_tasks,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q65_channel.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q65 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
