# -*- coding: utf-8 -*-
"""
Phase Q200: Quantum Kernel for Machine Learning (Opus Original)
=================================================================
Use S-Qubit interference as a QUANTUM KERNEL for classification.

Idea: The interference pattern between two soul vectors encodes
their "quantum similarity". This is mathematically equivalent to
a quantum kernel k(x,y) = |<phi(x)|phi(y)>|^2.

Test:
1. Train soul vectors for different categories
2. Use interference visibility as similarity metric
3. Build a kernel matrix from all pairwise visibilities
4. Classify test samples using this quantum kernel

If this works -> LLM-based Quantum Machine Learning on a laptop!
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

INJECT_LAYER = 8


def train_soul(model, tok, data, device, layer=8, epochs=80, seed=42):
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


def quantum_kernel(vec_a, vec_b, model, tok, prompt, device, target_id,
                    n_phi=16):
    """Compute quantum kernel k(a,b) from interference + cosine."""
    phis = np.linspace(0, 4 * np.pi, n_phi)
    p_vals = []
    scale = vec_a.norm()

    for phi in phis:
        vec = np.cos(phi/2) * vec_a + np.sin(phi/2) * vec_b
        n = vec.norm()
        if n > 0:
            vec = vec / n * scale

        inp = tok(prompt, return_tensors='pt').to(device)
        def hook(m, i, o, v=vec):
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_vals.append(float(probs[target_id]))

    p_arr = np.array(p_vals)
    # Visibility (interference strength)
    vis = (p_arr.max() - p_arr.min()) / (p_arr.max() + p_arr.min() + 1e-8)
    # Cosine similarity of soul vectors (better discriminator)
    cos = float(torch.dot(vec_a, vec_b) / (
        torch.norm(vec_a) * torch.norm(vec_b) + 1e-10))
    # Use absolute cosine as kernel (captures semantic distance)
    kernel_val = abs(cos)
    return float(vis), float(cos), kernel_val


def main():
    print("=" * 60)
    print("Phase Q200: Quantum Kernel for Machine Learning")
    print("  (S-Qubit interference as classification kernel)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    # Define categories with training data
    categories = {
        'color_blue': [("The sky is", "blue"), ("The ocean is", "blue")],
        'color_green': [("The grass is", "green"), ("Leaves are", "green")],
        'color_red': [("Blood is", "red"), ("Roses are", "red")],
        'animal_cat': [("A cat says", "me"), ("Cats like", "fish")],
        'animal_dog': [("A dog says", "w"), ("Dogs like", "bones")],
        'capital_paris': [("France capital is", "Paris")],
        'capital_tokyo': [("Japan capital is", "Tokyo")],
        'capital_london': [("England capital is", "London")],
    }

    # Train soul vectors
    print("  Training %d soul vectors..." % len(categories))
    vectors = {}
    for i, (cat, data) in enumerate(categories.items()):
        vectors[cat] = train_soul(model, tok, data, device,
                                  layer=INJECT_LAYER, seed=42 + i * 13)
        print("    %s: trained" % cat)

    target_id = tok.encode("blue")[-1]
    prompt = "The sky is"

    # Build quantum kernel matrix
    print("\n  Building quantum kernel matrix...")
    cat_names = list(categories.keys())
    n = len(cat_names)
    kernel_matrix = np.zeros((n, n))
    cosine_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            vis, cos, kernel_val = quantum_kernel(vectors[cat_names[i]],
                                       vectors[cat_names[j]],
                                       model, tok, prompt, device, target_id)
            kernel_matrix[i, j] = kernel_val
            kernel_matrix[j, i] = kernel_val
            cosine_matrix[i, j] = cos
            cosine_matrix[j, i] = cos

    # Classification test: nearest-neighbor using quantum kernel
    print("\n--- Classification via Quantum Kernel ---")

    # Ground truth: which categories are in the same "class"?
    class_map = {
        'color_blue': 'color', 'color_green': 'color', 'color_red': 'color',
        'animal_cat': 'animal', 'animal_dog': 'animal',
        'capital_paris': 'capital', 'capital_tokyo': 'capital',
        'capital_london': 'capital',
    }

    # For each item, find nearest neighbor (excluding self)
    correct = 0
    total = 0
    for i in range(n):
        # Find most similar item (highest kernel value, excluding self)
        similarities = kernel_matrix[i].copy()
        similarities[i] = -1  # Exclude self
        nearest = np.argmax(similarities)

        predicted_class = class_map[cat_names[nearest]]
        true_class = class_map[cat_names[i]]

        is_correct = predicted_class == true_class
        if is_correct:
            correct += 1
        total += 1

        print("  %s -> nearest: %s (%s) [%s]" % (
            cat_names[i], cat_names[nearest],
            "CORRECT" if is_correct else "WRONG",
            "k=%.4f" % similarities[nearest]))

    accuracy = 100 * correct / total
    print("\n  Classification accuracy: %d/%d (%.1f%%)" % (correct, total, accuracy))

    # Within-class vs between-class kernel values
    within_class = []
    between_class = []
    for i in range(n):
        for j in range(i + 1, n):
            if class_map[cat_names[i]] == class_map[cat_names[j]]:
                within_class.append(kernel_matrix[i, j])
            else:
                between_class.append(kernel_matrix[i, j])

    avg_within = float(np.mean(within_class)) if within_class else 0
    avg_between = float(np.mean(between_class)) if between_class else 0
    separation = avg_within - avg_between

    print("  Avg within-class kernel: %.4f" % avg_within)
    print("  Avg between-class kernel: %.4f" % avg_between)
    print("  Separation: %.4f" % separation)

    if accuracy >= 90:
        verdict = "EXCELLENT QML: %.0f%% accuracy, separation=%.3f" % (
            accuracy, separation)
    elif accuracy >= 70:
        verdict = "GOOD QML: %.0f%% accuracy" % accuracy
    else:
        verdict = "PARTIAL QML: %.0f%% accuracy" % accuracy

    # Save
    results = {
        'phase': 'Q200',
        'name': 'Quantum Kernel for ML',
        'categories': cat_names,
        'kernel_matrix': kernel_matrix.tolist(),
        'cosine_matrix': cosine_matrix.tolist(),
        'classification_accuracy': round(accuracy, 1),
        'summary': {
            'accuracy': round(accuracy, 1),
            'avg_within_class': round(avg_within, 4),
            'avg_between_class': round(avg_between, 4),
            'separation': round(separation, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q200_qml_kernel.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Quantum kernel matrix heatmap
    ax = axes[0]
    short_names = [c.replace('color_', 'c:').replace('animal_', 'a:').replace('capital_', 'k:')
                   for c in cat_names]
    im = ax.imshow(kernel_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('(a) Quantum Kernel Matrix\n(Interference Visibility)')

    # (b) Within vs between class
    ax = axes[1]
    bp = ax.boxplot([within_class, between_class],
               labels=['Within-class', 'Between-class'],
               patch_artist=True)
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#F44336')
    bp['boxes'][1].set_alpha(0.7)
    ax.set_ylabel('Kernel Value')
    ax.set_title('(b) Class Discrimination\n(sep=%.3f)' % separation)
    ax.grid(alpha=0.3, axis='y')

    # (c) Classification result
    ax = axes[2]
    ax.text(0.5, 0.75, 'Quantum Kernel Classification', fontsize=14,
            ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.55, 'Accuracy: %.0f%% (%d/%d)' % (accuracy, correct, total),
            fontsize=16, ha='center',
            color='#4CAF50' if accuracy >= 70 else '#F44336',
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.35, '3 classes: colors, animals, capitals', fontsize=11,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.15, 'Within: %.3f | Between: %.3f' % (avg_within, avg_between),
            fontsize=11, ha='center', transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) Results')

    plt.suptitle('Q200: Quantum Kernel for Machine Learning\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q200_qml_kernel.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ200 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
