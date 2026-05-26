# -*- coding: utf-8 -*-
"""Phase Q106: Attention as Quantum Channel
Do attention heads function as quantum channels?
Measure channel capacity, decoherence rates, and quantum
mutual information between head pairs.
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


def extract_attention_patterns(model, tokenizer):
    """Extract attention maps and analyze as quantum channels."""
    prompt = "The relationship between quantum physics and consciousness reveals"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = attentions[0].shape[2]

    print("  Layers: %d, Heads: %d, Seq: %d" % (num_layers, num_heads, seq_len))

    results = {}

    # 1. Channel capacity: von Neumann entropy of attention patterns
    print("  Computing channel capacities...")
    channel_capacities = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            attn = attentions[l][0, h, :, :].cpu().float().numpy()
            # Entropy of each row (each position's attention distribution)
            row_entropies = []
            for i in range(attn.shape[0]):
                p = np.clip(attn[i], 1e-10, 1.0)
                p = p / p.sum()
                entropy = -np.sum(p * np.log(p + 1e-10))
                if np.isfinite(entropy):
                    row_entropies.append(entropy)
            if row_entropies:
                channel_capacities[l, h] = np.mean(row_entropies)
            else:
                channel_capacities[l, h] = 0.0

    # 2. Quantum mutual information between adjacent layers
    print("  Computing quantum mutual information...")
    mutual_info = []
    for l in range(num_layers - 1):
        attn_l = attentions[l][0, :, :, :].cpu().float().numpy()  # (heads, seq, seq)
        attn_l1 = attentions[l+1][0, :, :, :].cpu().float().numpy()

        # Average over heads
        avg_l = np.clip(attn_l.mean(axis=0), 1e-10, 1.0)  # (seq, seq)
        avg_l1 = np.clip(attn_l1.mean(axis=0), 1e-10, 1.0)

        # MI = H(A) + H(B) - H(A,B)
        p_l = avg_l.flatten()
        p_l = p_l / (p_l.sum() + 1e-10)
        p_l = np.clip(p_l, 1e-10, 1.0)
        H_l = -np.sum(p_l * np.log(p_l))

        p_l1 = avg_l1.flatten()
        p_l1 = p_l1 / (p_l1.sum() + 1e-10)
        p_l1 = np.clip(p_l1, 1e-10, 1.0)
        H_l1 = -np.sum(p_l1 * np.log(p_l1))

        # Approximate joint via product (independent case gives MI~0)
        n_sample = min(len(p_l), 64)
        joint = np.outer(p_l[:n_sample], p_l1[:n_sample]).flatten()
        if len(joint) > 0:
            joint = joint / (joint.sum() + 1e-10)
            joint = np.clip(joint, 1e-10, 1.0)
            H_joint = -np.sum(joint * np.log(joint))
        else:
            H_joint = H_l + H_l1

        mi = max(0, H_l + H_l1 - H_joint)
        if np.isfinite(mi):
            mutual_info.append(float(mi))
        else:
            mutual_info.append(0.0)

    # 3. Decoherence: how quickly does attention become uniform?
    print("  Computing decoherence rates...")
    decoherence = []
    max_entropy = np.log(seq_len)
    for l in range(num_layers):
        mean_cap = channel_capacities[l].mean()
        dec = mean_cap / max_entropy  # 1.0 = fully decoherent, 0.0 = pure state
        decoherence.append(float(dec))

    # 4. Head specialization: variance across heads
    head_specialization = []
    for l in range(num_layers):
        spec = np.std(channel_capacities[l]) / (np.mean(channel_capacities[l]) + 1e-10)
        head_specialization.append(float(spec))

    results = {
        'channel_capacities': channel_capacities.tolist(),
        'mutual_info': mutual_info,
        'decoherence': decoherence,
        'head_specialization': head_specialization,
        'mean_capacity': float(channel_capacities.mean()),
        'max_capacity': float(channel_capacities.max()),
    }

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Channel capacity heatmap
    ax = axes[0][0]
    im = ax.imshow(channel_capacities.T, aspect='auto', cmap='viridis',
                   interpolation='nearest')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Head index', fontsize=11)
    ax.set_title('(a) Attention Channel Capacity\n(von Neumann entropy)',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Entropy (nats)')

    # (b) Quantum mutual information
    ax = axes[0][1]
    ax.plot(range(len(mutual_info)), mutual_info, 'o-', color='#FF5722',
            linewidth=2, markersize=5)
    ax.set_xlabel('Layer transition (L -> L+1)', fontsize=11)
    ax.set_ylabel('Quantum mutual information', fontsize=11)
    ax.set_title('(b) Inter-layer Quantum Channel\nInformation flow between layers',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.fill_between(range(len(mutual_info)), mutual_info,
                    alpha=0.2, color='#FF5722')

    # (c) Decoherence profile
    ax = axes[1][0]
    ax.plot(range(len(decoherence)), decoherence, 's-', color='#9C27B0',
            linewidth=2, markersize=6)
    ax.axhline(1.0, color='red', ls='--', alpha=0.3, label='Max decoherence')
    ax.axhline(0, color='green', ls='--', alpha=0.3, label='Pure state')
    ax.fill_between(range(len(decoherence)), decoherence,
                    alpha=0.15, color='#9C27B0')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Decoherence rate', fontsize=11)
    ax.set_title('(c) Quantum Decoherence\nDo deeper layers decohere?',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Head specialization
    ax = axes[1][1]
    ax.bar(range(len(head_specialization)), head_specialization,
           color='#2196F3', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Head specialization (CV)', fontsize=11)
    ax.set_title('(d) Attention Head Specialization\nHigher = more diverse heads',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q106: Attention Heads as Quantum Channels',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q106_quantum_channels.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    return results


def main():
    print("=" * 60)
    print("Phase Q106: Attention as Quantum Channel")
    print("=" * 60)
    t0 = time.time()

    from utils import _get_model_id
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = _get_model_id()
    tokenizer = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device, local_files_only=True,
        attn_implementation="eager")
    model.eval()
    results = extract_attention_patterns(model, tokenizer)

    print("\n  === Quantum Channel Results ===")
    print("  Mean channel capacity: %.4f nats" % results['mean_capacity'])
    print("  Max channel capacity: %.4f nats" % results['max_capacity'])
    print("  Decoherence range: %.4f - %.4f" %
          (min(results['decoherence']), max(results['decoherence'])))
    if results['mutual_info']:
        print("  Mean mutual info: %.4f" % np.mean(results['mutual_info']))

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results['phase'] = 'Q106'
    results['name'] = 'Attention as Quantum Channel'
    results['elapsed'] = elapsed
    res_path = os.path.join(RESULTS_DIR, 'phase_q106_quantum_channels.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
