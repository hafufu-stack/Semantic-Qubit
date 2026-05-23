# -*- coding: utf-8 -*-
"""
Phase Q1: Superposition Basis
Define the S-Qubit as a complex-valued Soul Vector in C^d.
Experiment: Construct |psi> = alpha*|MIN> + beta*e^{i*theta}*|MAX>
and measure how the LLM's output distribution shifts with amplitude alpha/beta.

Key questions:
- Can we train |MIN> and |MAX> basis Soul Vectors?
- Does linear combination of real soul vectors produce intermediate behavior?
- Does scaling alpha/beta change output probabilities proportionally?
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, inject_hook, get_logits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8


def train_soul(model, tok, data, device, layer, epochs=150, lr=0.01, seed=42):
    """Train a Soul Vector (real) via gradient descent on hook injection."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=lr)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def hook(m, i, o, v=vec):
                if isinstance(o, tuple):
                    h = o[0].clone()
                    h[0, -1, :] = v.to(h.dtype) if h.dim() == 3 else v.to(h.dtype)
                    return (h,) + o[1:]
                h = o.clone()
                h[0, -1, :] = v.to(h.dtype)
                return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp)
            handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device)
            )
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def probe_superposition(model, tok, min_vec, max_vec, prompt, device, layer,
                         alpha_range):
    """
    Inject psi = alpha * |MIN> + (1-alpha) * |MAX> at layer.
    Returns (alpha, prob_min_token, prob_max_token) for each alpha.
    """
    results = []
    for alpha in alpha_range:
        psi = alpha * min_vec + (1.0 - alpha) * max_vec
        def hook(m, i, o, v=psi):
            if isinstance(o, tuple):
                h = o[0].clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
        handle = model.model.layers[layer].register_forward_hook(hook)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :], dim=-1)
        results.append({
            'alpha': round(float(alpha), 3),
            'prob_min': round(float(probs[tok.encode('2')[-1]]), 5),
            'prob_max': round(float(probs[tok.encode('7')[-1]]), 5),
        })
    return results


def main():
    print("[Q1] Superposition Basis - S-Qubit Foundation")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # Training data: min(a,b) and max(a,b) on single-digit pairs
    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"), ("min(7,4)=", "4"),
        ("min(6,1)=", "1"), ("min(2,8)=", "2"), ("min(5,9)=", "5"),
        ("min(1,3)=", "1"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"), ("min(7,4)=", "7"),
        ("min(6,1)=", "6"), ("min(2,8)=", "8"), ("min(5,9)=", "9"),
        ("min(1,3)=", "3"),
    ]

    print("  Training |MIN> soul vector...")
    min_vec = train_soul(model, tok, min_data, DEVICE, LAYER, epochs=150, seed=42)
    print("  Training |MAX> soul vector...")
    max_vec = train_soul(model, tok, max_data, DEVICE, LAYER, epochs=150, seed=99)

    # Measure basis accuracy
    test_prompts = [
        ("min(7,2)=", "2", "7"),
        ("min(6,3)=", "3", "6"),
        ("min(2,9)=", "2", "9"),
        ("min(1,5)=", "1", "5"),
        ("min(8,4)=", "4", "8"),
    ]

    min_acc, max_acc = 0, 0
    for prompt, min_ans, max_ans in test_prompts:
        def hook_min(m, i, o, v=min_vec):
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
        def hook_max(m, i, o, v=max_vec):
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h

        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        h = model.model.layers[LAYER].register_forward_hook(hook_min)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax()).strip()
        if pred == min_ans: min_acc += 1

        h = model.model.layers[LAYER].register_forward_hook(hook_max)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax()).strip()
        if pred == max_ans: max_acc += 1

    min_acc /= len(test_prompts)
    max_acc /= len(test_prompts)
    print("  |MIN> accuracy: %.0f%%" % (min_acc * 100))
    print("  |MAX> accuracy: %.0f%%" % (max_acc * 100))

    # Superposition sweep: alpha from 0 to 1
    print("  Probing superposition alpha sweep (0->1)...")
    alpha_range = np.linspace(0, 1, 21)
    sup_results = probe_superposition(
        model, tok, min_vec, max_vec, "min(7,2)=", DEVICE, LAYER, alpha_range
    )
    print("  Sample: alpha=0.0 prob_min=%.4f prob_max=%.4f" %
          (sup_results[0]['prob_min'], sup_results[0]['prob_max']))
    print("  Sample: alpha=0.5 prob_min=%.4f prob_max=%.4f" %
          (sup_results[10]['prob_min'], sup_results[10]['prob_max']))
    print("  Sample: alpha=1.0 prob_min=%.4f prob_max=%.4f" %
          (sup_results[-1]['prob_min'], sup_results[-1]['prob_max']))

    # Norm and cosine of basis vectors
    cos_sim = float(torch.nn.functional.cosine_similarity(
        min_vec.unsqueeze(0), max_vec.unsqueeze(0)
    ))
    print("  |MIN>-|MAX> cosine similarity: %.4f" % cos_sim)
    print("  |MIN> norm: %.4f" % float(min_vec.norm()))
    print("  |MAX> norm: %.4f" % float(max_vec.norm()))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Alpha sweep
    ax = axes[0]
    alphas = [r['alpha'] for r in sup_results]
    probs_min = [r['prob_min'] for r in sup_results]
    probs_max = [r['prob_max'] for r in sup_results]
    ax.plot(alphas, probs_min, 'o-', color='#E91E63', label='P(min answer)', lw=2)
    ax.plot(alphas, probs_max, 's-', color='#2196F3', label='P(max answer)', lw=2)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal mix')
    ax.set_xlabel('alpha (alpha*|MIN> + (1-alpha)*|MAX>)')
    ax.set_ylabel('Token Probability')
    ax.set_title('Superposition Alpha Sweep\n"Quantum" Probability Mixing', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Basis accuracy
    ax = axes[1]
    bars = ax.bar(['|MIN> Soul', '|MAX> Soul'], [min_acc, max_acc],
                  color=['#E91E63', '#2196F3'], edgecolor='black', width=0.4)
    for bar, acc in zip(bars, [min_acc, max_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                '%.0f%%' % (acc*100), ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('S-Qubit Basis Vectors\n|MIN> and |MAX> Training', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 3: Summary text
    ax = axes[2]
    ax.axis('off')
    summary = (
        "S-Qubit Superposition Results\n\n"
        "|MIN> accuracy: %.0f%%\n"
        "|MAX> accuracy: %.0f%%\n\n"
        "Basis cosine similarity:\n  %.4f\n"
        "(0=orthogonal, ideal for qubit)\n\n"
        "Superposition behavior:\n"
        "  alpha=0.0: MAX dominates\n"
        "  alpha=0.5: mixed state\n"
        "  alpha=1.0: MIN dominates\n\n"
        "-> Is this classical mixing\n"
        "   or quantum interference?\n"
        "   (See Phase Q2 for answer)"
    ) % (min_acc*100, max_acc*100, cos_sim)
    ax.text(0.05, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))

    plt.suptitle(
        'Phase Q1: S-Qubit Superposition Basis\n'
        '|psi> = alpha*|MIN> + (1-alpha)*|MAX>',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q1_superposition_basis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q1', 'name': 'superposition_basis',
        'min_acc': min_acc, 'max_acc': max_acc,
        'cos_sim_min_max': cos_sim,
        'min_norm': float(min_vec.norm()),
        'max_norm': float(max_vec.norm()),
        'superposition_sweep': sup_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q1_superposition_basis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Q1 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
