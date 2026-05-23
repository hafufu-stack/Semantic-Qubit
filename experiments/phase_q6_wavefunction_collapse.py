# -*- coding: utf-8 -*-
"""
Phase Q6: Wavefunction Collapse Timing
Quantum measurement = wavefunction collapse from superposition to eigenstate.
Neural analog: LLM output goes from HIGH entropy (many possibilities)
to LOW entropy (definite answer) at some CRITICAL LAYER.

Experiments:
1. Track per-layer output entropy (if we decode at each layer using logits)
2. Find the "collapse point": layer where entropy drops below threshold
3. Does superposition injection delay the collapse? (coherence maintenance)
4. Does anti-vector injection accelerate collapse? (forced observation)
5. Compare: different task types have different collapse points
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


def get_per_layer_entropy(model, tok, prompt, device, num_layers, inject_vec=None, inject_layer=None):
    """
    Decode logits at each layer by temporarily using the model's lm_head.
    Returns list of entropy values, one per layer.
    """
    entropies = []
    layer_states = {}

    def make_hook(layer_idx):
        def hook(m, i, o):
            if isinstance(o, tuple):
                h = o[0]
            else:
                h = o
            # Apply injection if needed
            if inject_vec is not None and layer_idx == inject_layer:
                h = h.clone()
                h[0, -1, :] = inject_vec.to(h.dtype)
            layer_states[layer_idx] = h[0, -1, :].detach()
        return hook

    handles = []
    for li in range(num_layers):
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        handles.append(h)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)

    for h in handles:
        h.remove()

    # Compute entropy for each layer
    lm_head = model.lm_head
    norm = model.model.norm
    for li in range(num_layers):
        if li in layer_states:
            # hs: (hidden,) -> (1, 1, hidden) for norm [batch, seq, hidden]
            hs = layer_states[li].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)
            with torch.no_grad():
                normed = norm(hs)               # (1, 1, hidden)
                logits = lm_head(normed[0])     # (1, vocab)
                probs = torch.softmax(logits[0].float(), dim=-1)  # (vocab,)
                entropy = float(-(probs * (probs.clamp(min=1e-12)).log()).sum())
            entropies.append(entropy)
        else:
            entropies.append(float('nan'))

    return entropies


def find_collapse_layer(entropies, threshold=8.0):
    """Layer where entropy first drops below threshold."""
    for i, e in enumerate(entropies):
        if not np.isnan(e) and e < threshold:
            return i
    return len(entropies) - 1


def train_soul(model, tok, data, device, layer=8, epochs=150, seed=42):
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
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def main():
    print("[Q6] Wavefunction Collapse Timing")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    num_layers = len(model.model.layers)
    print("  Model has %d layers" % num_layers)

    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"),
    ]

    print("  Training basis vectors...")
    min_vec = train_soul(model, tok, min_data, DEVICE, layer=8, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, layer=8, seed=99)
    super_vec = (min_vec + max_vec) / np.sqrt(2)

    test_prompts = {
        'task_min': "min(7,2)=",
        'task_max_style': "min(3,7)=",
        'arithmetic': "3+4=",
        'natural': "The sky is",
        'code': "def add(x,y): return",
    }

    COLLAPSE_THRESHOLD = 8.0  # nats; below this = "collapsed" to near-certain answer

    results = {}
    all_entropies = {}

    print("  Measuring per-layer entropy (no injection)...")
    for task_name, prompt in test_prompts.items():
        ents = get_per_layer_entropy(model, tok, prompt, DEVICE, num_layers)
        all_entropies[task_name] = ents
        collapse = find_collapse_layer(ents, COLLAPSE_THRESHOLD)
        valid_ents = [e for e in ents if not np.isnan(e)]
        results[task_name] = {
            'collapse_layer': collapse,
            'min_entropy': round(min(valid_ents), 4) if valid_ents else None,
            'max_entropy': round(max(valid_ents), 4) if valid_ents else None,
            'final_entropy': round(ents[-1], 4) if not np.isnan(ents[-1]) else None,
            'entropies': [round(e, 4) if not np.isnan(e) else None for e in ents],
        }
        print("    %s: collapse@L%d, final_entropy=%.2f" % (task_name, collapse, ents[-1]))

    # Now test: does superposition DELAY collapse?
    print("  Testing superposition injection effect on collapse timing...")
    sup_entropies = {}
    for inject_at in [4, 8, 12, 16]:
        prompt = "min(7,2)="
        ents = get_per_layer_entropy(model, tok, prompt, DEVICE, num_layers,
                                      inject_vec=super_vec, inject_layer=inject_at)
        sup_entropies[inject_at] = ents
        collapse = find_collapse_layer(ents, COLLAPSE_THRESHOLD)
        print("    Superposition@L%d: collapse@L%d" % (inject_at, collapse))

    # Test: does MIN injection ACCELERATE collapse?
    print("  Testing MIN injection: does it accelerate collapse?")
    min_entropies = {}
    for inject_at in [4, 8, 12, 16]:
        prompt = "min(7,2)="
        ents = get_per_layer_entropy(model, tok, prompt, DEVICE, num_layers,
                                      inject_vec=min_vec, inject_layer=inject_at)
        min_entropies[inject_at] = ents
        collapse = find_collapse_layer(ents, COLLAPSE_THRESHOLD)
        print("    MIN@L%d: collapse@L%d" % (inject_at, collapse))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    layer_ids = list(range(num_layers))

    # Panel 1: Entropy curves for different task types
    ax = axes[0]
    colors = {'task_min': '#E91E63', 'task_max_style': '#9C27B0',
              'arithmetic': '#2196F3', 'natural': '#4CAF50', 'code': '#FF9800'}
    for task_name, ents in all_entropies.items():
        ents_clean = [e if not np.isnan(e) else None for e in ents]
        valid_layers = [i for i, e in enumerate(ents_clean) if e is not None]
        valid_ents = [e for e in ents_clean if e is not None]
        ax.plot(valid_layers, valid_ents, '-', color=colors.get(task_name, 'gray'),
                label=task_name, lw=2)
    ax.axhline(COLLAPSE_THRESHOLD, color='red', linestyle='--', alpha=0.7,
               label='Collapse threshold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Output Entropy (nats)')
    ax.set_title('Per-Layer Entropy\n"Wavefunction collapse timing"', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Superposition vs MIN injection collapse timing
    ax = axes[1]
    baseline_ents = all_entropies['task_min']
    baseline_collapse = find_collapse_layer(baseline_ents, COLLAPSE_THRESHOLD)

    inject_layers_list = [4, 8, 12, 16]
    sup_collapses = [find_collapse_layer(sup_entropies[l], COLLAPSE_THRESHOLD)
                     for l in inject_layers_list]
    min_collapses = [find_collapse_layer(min_entropies[l], COLLAPSE_THRESHOLD)
                     for l in inject_layers_list]

    x = np.arange(len(inject_layers_list))
    w = 0.25
    ax.bar(x - w, [baseline_collapse]*len(inject_layers_list), w, color='gray',
           label='Baseline (no injection)', edgecolor='black', alpha=0.6)
    ax.bar(x, sup_collapses, w, color='#9C27B0', label='Superposition injection', edgecolor='black')
    ax.bar(x + w, min_collapses, w, color='#E91E63', label='MIN injection', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['Inject@L%d' % l for l in inject_layers_list])
    ax.set_ylabel('Collapse Layer')
    ax.set_title('Injection Effect on Collapse Timing\n"Delayed/Accelerated?"', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel 3: Final entropy summary
    ax = axes[2]
    ax.axis('off')
    task_collapses = {k: results[k]['collapse_layer'] for k in results}
    summary_lines = ["Wavefunction Collapse Summary\n"]
    summary_lines.append("No injection (baseline):")
    for k, v in task_collapses.items():
        summary_lines.append("  %-20s L%d" % (k[:20], v))
    summary_lines.append("\nSuperposition injection effect:")
    for il, sc in zip(inject_layers_list, sup_collapses):
        delta = sc - baseline_collapse
        summary_lines.append("  @L%d: collapse L%d (%+d)" % (il, sc, delta))
    summary_lines.append("\nMIN injection effect:")
    for il, mc in zip(inject_layers_list, min_collapses):
        delta = mc - baseline_collapse
        summary_lines.append("  @L%d: collapse L%d (%+d)" % (il, mc, delta))
    ax.text(0.02, 0.95, '\n'.join(summary_lines), fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.9))

    plt.suptitle(
        'Phase Q6: Wavefunction Collapse Timing\n'
        'When does the LLM "decide"?',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q6_wavefunction_collapse.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q6', 'name': 'wavefunction_collapse',
        'collapse_threshold': COLLAPSE_THRESHOLD,
        'baseline_results': results,
        'superposition_collapses': {str(l): find_collapse_layer(sup_entropies[l], COLLAPSE_THRESHOLD)
                                     for l in inject_layers_list},
        'min_injection_collapses': {str(l): find_collapse_layer(min_entropies[l], COLLAPSE_THRESHOLD)
                                     for l in inject_layers_list},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q6_wavefunction_collapse.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Q6 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
