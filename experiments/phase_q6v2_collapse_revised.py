# -*- coding: utf-8 -*-
"""
Phase Q6v2: Wavefunction Collapse Timing (Revised)

Q6 with COLLAPSE_THRESHOLD=8.0 found collapse@L0 everywhere.
The model's intermediate entropy is 3-6 nats (well below 8.0).

Q6v2 fixes this:
- Lower threshold: 4.0 nats (based on actual observed entropy range)
- Also sweep multiple thresholds: [1.0, 2.0, 3.0, 4.0, 5.0] to see the full picture
- Full 28-layer entropy trajectory visualization (the "collapse curve")
- Highlight: which task types collapse earliest vs latest
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


def get_per_layer_entropy(model, tok, prompt, device, num_layers,
                           inject_vec=None, inject_layer=None):
    """Decode at each intermediate layer via norm+lm_head. Returns entropy list."""
    layer_states = {}

    def make_hook(layer_idx):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if inject_vec is not None and layer_idx == inject_layer:
                h = h.clone()
                h[0, -1, :] = inject_vec.to(h.dtype)
            layer_states[layer_idx] = h[0, -1, :].detach()
        return hook

    handles = [model.model.layers[li].register_forward_hook(make_hook(li))
               for li in range(num_layers)]
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)
    for h in handles:
        h.remove()

    lm_head = model.lm_head
    norm = model.model.norm
    entropies = []
    for li in range(num_layers):
        if li in layer_states:
            hs = layer_states[li].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)
            with torch.no_grad():
                normed = norm(hs)
                logits = lm_head(normed[0])
                probs = torch.softmax(logits[0].float(), dim=-1)
                entropy = float(-(probs * probs.clamp(min=1e-12).log()).sum())
            entropies.append(entropy)
        else:
            entropies.append(float('nan'))
    return entropies


def find_collapse_layer(entropies, threshold):
    for i, e in enumerate(entropies):
        if not np.isnan(e) and e < threshold:
            return i
    return len(entropies) - 1


def train_soul(model, tok, data, device, layer=8, epochs=100, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target in data:
            tid = tok.encode(target)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def hook(m, i, o, v=vec):
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0), torch.tensor([tid], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def main():
    print("[Q6v2] Wavefunction Collapse Timing (Revised Thresholds)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    num_layers = len(model.model.layers)
    print("  Model: %d layers" % num_layers)

    # Thresholds to sweep (based on observed entropy range 0.3 - 6.1 nats)
    THRESHOLDS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"),
    ]
    print("  Training basis vectors (100 epochs)...")
    min_vec = train_soul(model, tok, min_data, DEVICE, layer=8, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, layer=8, seed=99)
    super_vec = (min_vec + max_vec) / np.sqrt(2)

    test_prompts = {
        'task_min':   "min(7,2)=",
        'task_max':   "min(3,7)=",
        'arithmetic': "3+4=",
        'natural':    "The sky is",
        'code':       "def add(x,y): return",
    }

    print("  Computing per-layer entropy for all tasks...")
    all_entropies = {}
    for task_name, prompt in test_prompts.items():
        ents = get_per_layer_entropy(model, tok, prompt, DEVICE, num_layers)
        all_entropies[task_name] = ents
        valid = [e for e in ents if not np.isnan(e)]
        print("    %-15s entropy: L0=%.2f, min=%.2f, max=%.2f, final=%.2f" % (
            task_name, ents[0], min(valid), max(valid), ents[-1]))

    # Threshold sweep: find collapse layer for each task × threshold
    print("  Threshold sweep...")
    collapse_map = {}  # task -> {threshold -> collapse_layer}
    for task_name, ents in all_entropies.items():
        collapse_map[task_name] = {}
        for th in THRESHOLDS:
            cl = find_collapse_layer(ents, th)
            collapse_map[task_name][th] = cl

    # Injection experiments at threshold=4.0
    TH = 4.0
    print("  Injection effect (threshold=%.1f nats)..." % TH)
    sup_collapses, min_collapses = {}, {}
    for inject_at in [4, 8, 12, 16]:
        prompt = "min(7,2)="
        ents_sup = get_per_layer_entropy(model, tok, prompt, DEVICE, num_layers,
                                          inject_vec=super_vec, inject_layer=inject_at)
        ents_min = get_per_layer_entropy(model, tok, prompt, DEVICE, num_layers,
                                          inject_vec=min_vec, inject_layer=inject_at)
        sup_collapses[inject_at] = find_collapse_layer(ents_sup, TH)
        min_collapses[inject_at] = find_collapse_layer(ents_min, TH)
        print("    @L%d: sup_collapse=L%d  min_collapse=L%d" % (
            inject_at, sup_collapses[inject_at], min_collapses[inject_at]))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors_map = {
        'task_min': '#E91E63', 'task_max': '#9C27B0',
        'arithmetic': '#2196F3', 'natural': '#4CAF50', 'code': '#FF9800'
    }
    layer_ids = list(range(num_layers))

    # Panel 1: Full entropy trajectories
    ax = axes[0]
    for task_name, ents in all_entropies.items():
        clean = [e if not np.isnan(e) else None for e in ents]
        vl = [i for i, e in enumerate(clean) if e is not None]
        ve = [e for e in clean if e is not None]
        ax.plot(vl, ve, '-', color=colors_map.get(task_name, 'gray'),
                label=task_name, lw=2)
    # Draw threshold lines
    for th, ls in zip([4.0, 3.0, 2.0, 1.0], ['--', ':', '-.', (0,(3,5,1,5))]):
        ax.axhline(th, color='red', linestyle=ls, alpha=0.5, lw=1.2,
                   label='threshold=%.1f' % th)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Entropy (nats)', fontsize=11)
    ax.set_title('Per-Layer Entropy Trajectories\n"Wavefunction Collapse Curve"',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 2: Collapse layer vs threshold heatmap
    ax = axes[1]
    task_names = list(collapse_map.keys())
    mat = np.array([[collapse_map[t][th] for th in THRESHOLDS] for t in task_names])
    im = ax.imshow(mat, aspect='auto', cmap='viridis_r',
                   extent=[-0.5, len(THRESHOLDS)-0.5, -0.5, len(task_names)-0.5])
    plt.colorbar(im, ax=ax, label='Collapse Layer')
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels(['%.1f' % th for th in THRESHOLDS])
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_xlabel('Threshold (nats)', fontsize=11)
    ax.set_title('Collapse Layer vs Threshold\n(Lower = Earlier Collapse)',
                 fontweight='bold')
    for i in range(len(task_names)):
        for j in range(len(THRESHOLDS)):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                    fontsize=9, color='white' if mat[i, j] > 15 else 'black')

    # Panel 3: Injection effect at threshold=4.0
    ax = axes[2]
    inject_layers_list = [4, 8, 12, 16]
    baseline_task = 'task_min'
    base_cl = collapse_map[baseline_task][TH]
    sup_cl_list = [sup_collapses[il] for il in inject_layers_list]
    min_cl_list = [min_collapses[il] for il in inject_layers_list]
    x = np.arange(len(inject_layers_list))
    w = 0.25
    ax.bar(x - w, [base_cl]*len(inject_layers_list), w, color='gray',
           label='Baseline (L%d)' % base_cl, alpha=0.7, edgecolor='black')
    ax.bar(x, sup_cl_list, w, color='#9C27B0',
           label='Superposition', edgecolor='black')
    ax.bar(x + w, min_cl_list, w, color='#E91E63',
           label='MIN injection', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['@L%d' % il for il in inject_layers_list])
    ax.set_ylabel('Collapse Layer (threshold=%.1f)' % TH, fontsize=10)
    ax.set_title('Injection Effect on Collapse Timing\n(Threshold=%.1f nats)' % TH,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Phase Q6v2: Wavefunction Collapse Timing (Revised)\n'
        'Proper thresholds reveal layer-by-layer dynamics',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q6v2_collapse_revised.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q6v2', 'name': 'wavefunction_collapse_revised',
        'thresholds': THRESHOLDS,
        'all_entropies': {k: [round(e, 4) if not np.isnan(e) else None for e in v]
                          for k, v in all_entropies.items()},
        'collapse_map': {t: {str(th): collapse_map[t][th] for th in THRESHOLDS}
                         for t in task_names},
        'sup_collapses': {str(k): v for k, v in sup_collapses.items()},
        'min_collapses': {str(k): v for k, v in min_collapses.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q6v2_collapse_revised.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Q6v2 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
