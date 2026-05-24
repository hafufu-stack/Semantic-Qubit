# -*- coding: utf-8 -*-
"""
Phase Q63: Quantum Error Correction via Attention Heads
========================================================
BRIDGE: NeuOS Attention Analysis <-> Semantic-Qubit

Hypothesis: Different attention heads in the transformer act as
independent "syndrome detectors" - each head checks a different
aspect of the S-Qubit state. If one head's contribution is
corrupted, the other heads can compensate.

This is analogous to quantum error correction codes where
redundant encoding across multiple qubits protects information.

Test: Systematically ablate (zero out) attention heads and measure
how S-Qubit task performance degrades. If the system is robust,
it has built-in error correction.
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
    print("[Q63] Quantum Error Correction via Attention Heads")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    print("  Architecture: %d layers, %d heads, head_dim=%d" % (n_layers, n_heads, head_dim))

    # Train S-Qubit
    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    vec = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    
    prompt = "min(7,2)="
    target_id = tok.encode("2")[-1]
    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    def inject_and_measure(ablate_heads=None, ablate_layer=None):
        """Measure with optional head ablation."""
        handles = []
        # Inject S-Qubit
        def inject_hook(m, i, o, v=vec):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handles.append(model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook))
        
        # Ablate specific attention heads
        if ablate_heads is not None and ablate_layer is not None:
            def ablate_hook(m, i, o, heads=ablate_heads, hd=head_dim):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                for head_idx in heads:
                    start_idx = head_idx * hd
                    end_idx = (head_idx + 1) * hd
                    h[:, :, start_idx:end_idx] = 0
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handles.append(model.model.layers[ablate_layer].register_forward_hook(ablate_hook))
        
        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(probs[target_id])

    # Baseline
    clean_prob = inject_and_measure()
    print("  Clean baseline: p(2) = %.4f" % clean_prob)

    # Test 1: Single head ablation at injection layer + surrounding layers
    print("\n  Testing single-head ablation...")
    test_layers = [INJECT_LAYER - 1, INJECT_LAYER, INJECT_LAYER + 1, 
                   INJECT_LAYER + 2, n_layers - 1]
    
    ablation_results = {}
    for test_layer in test_layers:
        if test_layer < 0 or test_layer >= n_layers:
            continue
        head_probs = []
        for head_idx in range(n_heads):
            p = inject_and_measure(ablate_heads=[head_idx], ablate_layer=test_layer)
            head_probs.append(p)
        ablation_results[test_layer] = head_probs
        survived = sum(1 for p in head_probs if p > clean_prob * 0.5)
        print("    Layer %d: %d/%d heads removable (>50%% performance)" % (
            test_layer, survived, n_heads))

    # Test 2: Multi-head ablation (how many heads can we remove?)
    print("\n  Testing multi-head ablation at layer %d..." % (INJECT_LAYER + 1))
    multi_results = []
    for n_ablate in range(0, n_heads + 1):
        trials = []
        for trial in range(min(5, max(1, n_heads - n_ablate))):
            torch.manual_seed(trial * 100 + n_ablate)
            heads_to_remove = torch.randperm(n_heads)[:n_ablate].tolist()
            p = inject_and_measure(ablate_heads=heads_to_remove, 
                                   ablate_layer=INJECT_LAYER + 1)
            trials.append(p)
        avg_p = np.mean(trials)
        multi_results.append(avg_p)
        if n_ablate % 4 == 0 or n_ablate == n_heads:
            print("    Ablate %d/%d heads: p=%.4f" % (n_ablate, n_heads, avg_p))

    # Find error correction capacity
    ec_threshold = clean_prob * 0.5
    ec_capacity = 0
    for i, p in enumerate(multi_results):
        if p >= ec_threshold:
            ec_capacity = i
    
    ec_ratio = ec_capacity / n_heads if n_heads > 0 else 0
    print("\n  Error correction capacity: %d/%d heads (%.0f%%)" % (
        ec_capacity, n_heads, ec_ratio * 100))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Single-head ablation heatmap
    ax = axes[0]
    layer_labels = ['L%d' % l for l in sorted(ablation_results.keys())]
    heatmap_data = np.array([ablation_results[l] for l in sorted(ablation_results.keys())])
    im = ax.imshow(heatmap_data / clean_prob, cmap='RdYlGn', vmin=0, vmax=1.2, aspect='auto')
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels)
    ax.set_xlabel('Head index')
    ax.set_ylabel('Layer')
    ax.set_title('(a) Single-Head Ablation\nBrighter = more robust',
                 fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized performance')

    # (b) Multi-head ablation curve
    ax = axes[1]
    ax.plot(range(len(multi_results)), multi_results, 'o-', color='#FF5722',
            linewidth=2, markersize=5)
    ax.axhline(clean_prob, color='green', ls='--', alpha=0.5, label='Clean baseline')
    ax.axhline(ec_threshold, color='red', ls=':', alpha=0.5, label='50%% threshold')
    ax.axvline(ec_capacity, color='blue', ls=':', alpha=0.5, 
               label='EC capacity (%d heads)' % ec_capacity)
    ax.fill_between(range(ec_capacity + 1), 0, clean_prob * 1.1,
                    alpha=0.1, color='green', label='Error-correctable zone')
    ax.set_xlabel('Number of heads ablated')
    ax.set_ylabel('P(correct)')
    ax.set_title('(b) Error Correction Capacity\n%d/%d heads removable' % (ec_capacity, n_heads),
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Comparison with physical QEC codes
    ax = axes[2]
    codes = ['Steane\n[[7,1,3]]', 'Surface\n[[d^2,1,d]]', 'S-Qubit\nAttention QEC']
    redundancy = [7, 17, n_heads]  # qubits needed per logical qubit
    correctable = [1, 2, ec_capacity]
    ratio = [c/r for c, r in zip(correctable, redundancy)]
    
    colors = ['#2196F3', '#2196F3', '#FF5722']
    bars = ax.bar(codes, ratio, color=colors, edgecolor='black', alpha=0.85)
    for bar, r in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                '%.2f' % r, ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel('Correction efficiency\n(correctable / total)')
    ax.set_title('(c) QEC Efficiency Comparison\nS-Qubit vs Physical Codes',
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q63: Quantum Error Correction via Attention\n'
                 'Attention heads provide built-in redundancy',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q63_attention_qec.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q63', 'name': 'attention_qec',
        'clean_prob': round(clean_prob, 4),
        'n_heads': int(n_heads),
        'ec_capacity': int(ec_capacity),
        'ec_ratio': round(float(ec_ratio), 3),
        'head_dim': int(head_dim),
        'bridge': 'NeuOS Attention -> Semantic-Qubit QEC',
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q63_attention_qec.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q63 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
