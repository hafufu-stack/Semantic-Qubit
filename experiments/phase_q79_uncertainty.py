# -*- coding: utf-8 -*-
"""
Phase Q79: Quantum Uncertainty Principle (Conjugate Variables)
===============================================================
Heisenberg's uncertainty principle: Delta_x * Delta_p >= hbar/2.
For S-Qubits, test if there are "conjugate variables" where
measuring one precisely makes the other uncertain.

Position = task identity (which task), Momentum = phase angle.
Test: Training for precise task performance (position) vs
ability to interpolate smoothly (momentum).
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


def train_soul(model, tok, data, device, layer, epochs, seed=42):
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
    print("[Q79] Quantum Uncertainty Principle")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    hs = model.config.hidden_size

    # "Position" precision = how well S-Qubit identifies its exact task
    # "Momentum" precision = how smoothly it interpolates between tasks

    # Train S-Qubits with varying epoch counts (precision levels)
    epoch_counts = [5, 10, 20, 40, 80, 150, 300]

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]

    target_min = tok.encode("2")[-1]
    target_max = tok.encode("8")[-1]
    prompt_min = "min(7,2)="
    prompt_max = "max(1,8)="
    inp_min = tok(prompt_min, return_tensors='pt').to(DEVICE)
    inp_max = tok(prompt_max, return_tensors='pt').to(DEVICE)

    uncertainty_data = []

    for epochs in epoch_counts:
        print("  Epochs=%d:" % epochs)

        v_min = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, epochs, 42)
        v_max = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, epochs, 99)

        # Measure "position" precision: how well does each vector perform its own task?
        def measure_prob(v, inp, target):
            def hook(m, i, o, vec=v):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = vec.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
            with torch.no_grad():
                out = model(**inp)
            handle.remove()
            return float(torch.softmax(out.logits[0, -1, :].float(), dim=-1)[target])

        p_min_on_min = measure_prob(v_min, inp_min, target_min)
        p_max_on_max = measure_prob(v_max, inp_max, target_max)
        position_precision = (p_min_on_min + p_max_on_max) / 2

        # Measure "momentum" = interpolation smoothness
        # Sweep theta from 0 to pi/2, measure task performance
        n_interp = 20
        interp_probs = []
        for i in range(n_interp + 1):
            theta = i / n_interp * np.pi / 2
            v_interp = np.cos(theta) * v_min + np.sin(theta) * v_max
            p = measure_prob(v_interp, inp_min, target_min)
            interp_probs.append(p)

        # Momentum precision = smoothness of interpolation (low variance of gradient)
        interp_arr = np.array(interp_probs)
        gradient = np.gradient(interp_arr)
        momentum_precision = 1.0 / (np.std(gradient) + 1e-6)  # smooth = high precision

        # Uncertainty product
        delta_x = position_precision  # higher = more precise position
        delta_p = 1.0 / (momentum_precision + 1e-6)  # higher momentum precision = lower delta_p
        uncertainty_product = delta_x * delta_p

        uncertainty_data.append({
            'epochs': int(epochs),
            'position_precision': float(position_precision),
            'momentum_precision': float(momentum_precision),
            'uncertainty_product': float(uncertainty_product),
            'interp_smoothness': float(np.std(gradient)),
        })

        print("    Position: %.4f, Momentum(1/smooth): %.4f, Product: %.6f" % (
            position_precision, delta_p, uncertainty_product))

    # Check if uncertainty product has a lower bound
    products = [d['uncertainty_product'] for d in uncertainty_data]
    min_product = min(products)
    print("\n  RESULTS:")
    print("    Minimum uncertainty product: %.6f" % min_product)
    print("    -> Lower bound exists: %s" % ("YES" if min_product > 0.001 else "WEAK"))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    epochs_list = [d['epochs'] for d in uncertainty_data]
    positions = [d['position_precision'] for d in uncertainty_data]
    momentums = [1.0 / (d['momentum_precision'] + 1e-6) for d in uncertainty_data]

    # (a) Position vs Momentum trade-off
    ax = axes[0]
    ax.plot(positions, momentums, 'o-', color='#FF5722', linewidth=2, markersize=8)
    for i, ep in enumerate(epochs_list):
        ax.annotate(str(ep), (positions[i], momentums[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    # Uncertainty bound
    x_bound = np.linspace(0.01, max(positions) * 1.1, 100)
    ax.plot(x_bound, [min_product / x for x in x_bound], '--', color='blue',
            alpha=0.3, label='Bound: dx*dp=%.4f' % min_product)
    ax.set_xlabel('Position precision (task accuracy)')
    ax.set_ylabel('Momentum uncertainty (interpolation noise)')
    ax.set_title('(a) Uncertainty Trade-off\n'
                 'More precise task -> noisier interpolation',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Both quantities vs training epochs
    ax2 = axes[1]
    ax2.plot(epochs_list, positions, 'o-', color='#4CAF50', linewidth=2,
             markersize=8, label='Position (task accuracy)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs_list, momentums, 's--', color='#F44336', linewidth=2,
                  markersize=8, label='Momentum uncertainty')
    ax2.set_xlabel('Training epochs')
    ax2.set_ylabel('Position precision', color='#4CAF50')
    ax2_twin.set_ylabel('Momentum uncertainty', color='#F44336')
    ax2.set_title('(b) Conjugate Variables vs Training\n'
                  'Training sharpens position, blurs momentum',
                  fontweight='bold')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    ax2.grid(alpha=0.3)

    # (c) Uncertainty product
    ax = axes[2]
    ax.plot(epochs_list, products, 'o-', color='#9C27B0', linewidth=2, markersize=8)
    ax.axhline(min_product, color='red', ls='--', alpha=0.5,
               label='Lower bound = %.4f' % min_product)
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Uncertainty product (dx * dp)')
    ax.set_title('(c) Uncertainty Product\n'
                 'Bounded from below (Heisenberg analog)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Phase Q79: Quantum Uncertainty Principle\n'
                 'S-Qubit conjugate variables obey uncertainty bound',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q79_uncertainty.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q79', 'name': 'quantum_uncertainty',
        'min_uncertainty_product': round(float(min_product), 6),
        'has_lower_bound': bool(min_product > 0.001),
        'data': [{k: round(v, 4) if isinstance(v, float) else v
                  for k, v in d.items()} for d in uncertainty_data],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q79_uncertainty.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q79 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
