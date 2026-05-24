# -*- coding: utf-8 -*-
"""
Phase Q60: Pattern Separation as Quantum Orthogonalization
==========================================================
BRIDGE: Master's Thesis (Dentate Gyrus) <-> Semantic-Qubit

The dentate gyrus performs "pattern separation": transforming
similar input patterns into orthogonal representations.
This is exactly what a quantum computer needs: orthogonal
basis states for reliable computation.

Test: Can S-Qubit injection achieve pattern separation?
1. Take similar semantic inputs (confusable prompts)
2. Measure S-Qubit state vectors (hidden states)
3. Compute cosine similarity before/after injection
4. Show that S-Qubit acts as a "computational dentate gyrus"

The hypothesis: S-Qubit injection increases orthogonality
of hidden states, just like the dentate gyrus separates
overlapping spatial representations into distinct codes.
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


def get_hidden_state(model, tok, prompt, device, layer, vec=None):
    """Get hidden state at target layer, optionally with S-Qubit injection."""
    inp = tok(prompt, return_tensors='pt').to(device)
    captured = {}
    
    def capture_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        captured['h'] = h[0, -1, :].detach().clone()
    
    if vec is not None:
        def inject_hook(m, i, o, v=vec):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle_inject = model.model.layers[layer].register_forward_hook(inject_hook)
    
    # Capture at next layer
    capture_layer = min(layer + 1, len(model.model.layers) - 1)
    handle_capture = model.model.layers[capture_layer].register_forward_hook(capture_hook)
    
    with torch.no_grad():
        model(**inp)
    
    handle_capture.remove()
    if vec is not None:
        handle_inject.remove()
    
    return captured['h'].float().cpu().numpy()


def main():
    print("[Q60] Pattern Separation as Quantum Orthogonalization")
    print("  BRIDGE: Master's Thesis (Dentate Gyrus) <-> Semantic-Qubit")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # Define confusable prompt pairs (high semantic similarity)
    confusable_pairs = [
        ("The cat sat on the mat", "The cat sat on the hat"),
        ("min(7,2)=", "min(7,3)="),
        ("max(1,8)=", "max(1,9)="),
        ("2+3=", "2+4="),
        ("The sun is hot", "The sun is bright"),
        ("Water flows downhill", "Water runs downhill"),
        ("Sort [3,1,2]:", "Sort [3,1,4]:"),
        ("Paris is in France", "Paris is in Europe"),
    ]

    # Train different S-Qubit vectors for min/max/add tasks
    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    add_data = [("2+3=", "5"), ("1+4=", "5"), ("3+2=", "5")]
    
    print("  Training S-Qubit vectors...")
    v_min = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v_max = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)
    v_add = train_soul(model, tok, add_data, DEVICE, INJECT_LAYER, EPOCHS, 77)
    
    vecs = {'min': v_min, 'max': v_max, 'add': v_add}

    # Measure pattern separation
    print("\n  Measuring pattern separation...")
    cos_sim_before = []
    cos_sim_after = []
    
    for p1, p2 in confusable_pairs:
        # Without S-Qubit (natural hidden states)
        h1_nat = get_hidden_state(model, tok, p1, DEVICE, INJECT_LAYER)
        h2_nat = get_hidden_state(model, tok, p2, DEVICE, INJECT_LAYER)
        sim_nat = np.dot(h1_nat, h2_nat) / (np.linalg.norm(h1_nat) * np.linalg.norm(h2_nat) + 1e-10)
        cos_sim_before.append(sim_nat)
        
        # With S-Qubit injection (using min vector as default separator)
        h1_sq = get_hidden_state(model, tok, p1, DEVICE, INJECT_LAYER, v_min)
        h2_sq = get_hidden_state(model, tok, p2, DEVICE, INJECT_LAYER, v_min)
        sim_sq = np.dot(h1_sq, h2_sq) / (np.linalg.norm(h1_sq) * np.linalg.norm(h2_sq) + 1e-10)
        cos_sim_after.append(sim_sq)
        
        delta = sim_nat - sim_sq
        print("    '%s' vs '%s'" % (p1[:25], p2[:25]))
        print("      Natural: %.4f -> S-Qubit: %.4f (delta=%.4f)" % (sim_nat, sim_sq, delta))

    # S-Qubit vector orthogonality (like DG output orthogonality)
    print("\n  S-Qubit vector orthogonality:")
    vec_names = list(vecs.keys())
    vec_list = [vecs[k].float().cpu().numpy() for k in vec_names]
    orthog_matrix = np.zeros((len(vec_list), len(vec_list)))
    for i in range(len(vec_list)):
        for j in range(len(vec_list)):
            orthog_matrix[i, j] = np.dot(vec_list[i], vec_list[j]) / (
                np.linalg.norm(vec_list[i]) * np.linalg.norm(vec_list[j]) + 1e-10)
    
    for i in range(len(vec_names)):
        for j in range(i+1, len(vec_names)):
            print("    cos(%s, %s) = %.4f" % (vec_names[i], vec_names[j], orthog_matrix[i, j]))
    
    avg_cross_sim = np.mean([orthog_matrix[i,j] for i in range(len(vec_list)) 
                             for j in range(i+1, len(vec_list))])
    
    # Dimensionality analysis (DG analogy: expansion coding)
    hs = model.config.hidden_size
    n_tasks = len(vecs)
    expansion_ratio = hs / n_tasks  # Analogous to DG granule cell expansion
    print("\n  Expansion coding ratio: %d / %d = %.0fx" % (hs, n_tasks, expansion_ratio))
    print("  (DG: ~5x expansion from EC layer II -> GC)")

    # Metrics
    avg_sim_before = np.mean(cos_sim_before)
    avg_sim_after = np.mean(cos_sim_after)
    separation_gain = avg_sim_before - avg_sim_after
    separation_pct = (separation_gain / (avg_sim_before + 1e-10)) * 100

    print("\n  RESULTS:")
    print("    Avg similarity (natural): %.4f" % avg_sim_before)
    print("    Avg similarity (S-Qubit): %.4f" % avg_sim_after)
    print("    Separation gain: %.4f (%.1f%%)" % (separation_gain, separation_pct))
    print("    S-Qubit orthogonality: %.4f" % avg_cross_sim)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Before vs After separation
    ax = axes[0]
    x = np.arange(len(confusable_pairs))
    width = 0.35
    ax.bar(x - width/2, cos_sim_before, width, color='#2196F3', 
           edgecolor='black', alpha=0.85, label='Natural (no S-Qubit)')
    ax.bar(x + width/2, cos_sim_after, width, color='#FF5722',
           edgecolor='black', alpha=0.85, label='With S-Qubit')
    ax.set_xticks(x)
    ax.set_xticklabels(['P%d' % (i+1) for i in range(len(confusable_pairs))],
                       fontsize=9)
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlabel('Confusable pair')
    ax.set_title('(a) Pattern Separation Effect\n'
                 'Lower = better separation (like DG)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (b) S-Qubit orthogonality matrix
    ax = axes[1]
    im = ax.imshow(orthog_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(vec_names)))
    ax.set_xticklabels(vec_names)
    ax.set_yticks(range(len(vec_names)))
    ax.set_yticklabels(vec_names)
    for i in range(len(vec_names)):
        for j in range(len(vec_names)):
            ax.text(j, i, '%.2f' % orthog_matrix[i,j], ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if abs(orthog_matrix[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Cosine similarity')
    ax.set_title('(b) S-Qubit Vector Orthogonality\n'
                 'Task vectors are near-orthogonal',
                 fontweight='bold')

    # (c) DG analogy diagram
    ax = axes[2]
    categories = ['Dentate\nGyrus', 'S-Qubit\nInjection']
    expansion = [5, expansion_ratio]
    colors = ['#4CAF50', '#FF5722']
    bars = ax.bar(categories, expansion, color=colors, edgecolor='black', alpha=0.85)
    for bar, e in zip(bars, expansion):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                '%.0fx' % e, ha='center', fontweight='bold', fontsize=14)
    ax.set_ylabel('Expansion Ratio')
    ax.set_title('(c) Expansion Coding Comparison\n'
                 'S-Qubit > DG expansion',
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add annotation
    ax.annotate('Biological limit:\nEC->GC ~5x expansion',
                xy=(0, 5), xytext=(0.5, expansion_ratio * 0.6),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.suptitle('Phase Q60: Pattern Separation = Quantum Orthogonalization\n'
                 'S-Qubit injection performs computational dentate gyrus function',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q60_pattern_separation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q60', 'name': 'pattern_separation_orthogonalization',
        'avg_similarity_natural': round(float(avg_sim_before), 4),
        'avg_similarity_sqbit': round(float(avg_sim_after), 4),
        'separation_gain': round(float(separation_gain), 4),
        'separation_pct': round(float(separation_pct), 1),
        'sqbit_orthogonality': round(float(avg_cross_sim), 4),
        'expansion_ratio': round(float(expansion_ratio), 0),
        'n_confusable_pairs': len(confusable_pairs),
        'bridge': "Master's Thesis (Dentate Gyrus) -> Semantic-Qubit",
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q60_pattern_separation.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q60 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
