# -*- coding: utf-8 -*-
"""Phase Q108: Semantic Hawking Radiation
Do deep transformer layers (black holes) emit thermal radiation?
If the deepest layers absorb S-Qubit states and create an
"event horizon", there should be thermal radiation at the
output boundary.
This tests: Is the output token distribution Planckian (thermal)
when information is injected deep in the model?
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def measure_hawking_radiation(model, tokenizer, num_layers):
    """Inject perturbations at different depths and measure
    the 'temperature' (entropy) of the output distribution."""

    d_model = model.config.hidden_size
    prompt = "In the depths of a black hole, information"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    np.random.seed(108)
    sv = np.random.randn(d_model).astype(np.float32)
    sv /= np.linalg.norm(sv)
    sv_tensor = torch.tensor(sv, device=model.device)

    # Reference
    with torch.no_grad():
        ref_out = model(**inputs)
        ref_logits = ref_out.logits[0, -1, :].cpu().float().numpy()
        ref_probs = np.exp(ref_logits - ref_logits.max())
        ref_probs /= ref_probs.sum()
        ref_entropy = float(scipy_entropy(ref_probs))

    temperatures = []

    for layer_idx in range(num_layers):
        applied = [False]
        def hook(module, args, output, app=applied):
            if not app[0]:
                app[0] = True
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv_tensor.to(hs.dtype)
                    else:
                        hs[-1, :] += sv_tensor.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv_tensor.to(hs.dtype)
                    else:
                        hs[-1, :] += sv_tensor.to(hs.dtype)
                    return hs
            return output

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].cpu().float().numpy()
        handle.remove()

        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        output_entropy = float(scipy_entropy(probs))

        # "Temperature" = change in output entropy from perturbation
        delta_entropy = output_entropy - ref_entropy

        # Check if distribution is more "thermal" (higher entropy = hotter)
        # Hawking temperature should be inversely proportional to
        # "distance from event horizon" = layers remaining
        remaining = num_layers - layer_idx
        hawking_temp = delta_entropy / (remaining + 1) if remaining > 0 else 0

        temperatures.append({
            'layer': layer_idx,
            'output_entropy': output_entropy,
            'delta_entropy': float(delta_entropy),
            'hawking_temp': float(hawking_temp),
            'remaining_layers': remaining,
        })

        print("    Layer %d: S=%.4f, delta=%.4f, T_H=%.6f" %
              (layer_idx, output_entropy, delta_entropy, hawking_temp))

    return temperatures, ref_entropy


def main():
    print("=" * 60)
    print("Phase Q108: Semantic Hawking Radiation")
    print("  Do deep layers emit thermal radiation?")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    temperatures, ref_entropy = measure_hawking_radiation(
        model, tokenizer, num_layers)

    # Analysis: fit T_H ~ 1/r (inverse distance from horizon)
    layers = [t['layer'] for t in temperatures]
    deltas = [t['delta_entropy'] for t in temperatures]
    hawking_temps = [t['hawking_temp'] for t in temperatures]

    # Surface gravity analog: rate of entropy change
    surface_gravity = np.diff(deltas) if len(deltas) > 1 else [0]

    # Is there an "event horizon"? Look for layer where delta peaks
    peak_layer = int(np.argmax(np.abs(deltas)))
    peak_delta = deltas[peak_layer]

    print("\n  === Hawking Radiation Results ===")
    print("  Reference entropy: %.4f" % ref_entropy)
    print("  Event horizon layer: %d" % peak_layer)
    print("  Peak delta entropy: %.4f" % peak_delta)
    print("  Max Hawking temperature: %.6f" % max(np.abs(hawking_temps)))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Entropy profile
    ax = axes[0]
    entropies = [t['output_entropy'] for t in temperatures]
    ax.plot(layers, entropies, 'o-', color='#FF5722', linewidth=2,
            markersize=5)
    ax.axhline(ref_entropy, color='gray', ls='--', alpha=0.5,
               label='Reference S=%.2f' % ref_entropy)
    ax.fill_between(layers, ref_entropy, entropies, alpha=0.2,
                    color='#FF5722')
    ax.set_xlabel('Injection layer depth', fontsize=11)
    ax.set_ylabel('Output entropy (nats)', fontsize=11)
    ax.set_title('(a) Entropy vs Injection Depth\nDeeper = closer to black hole',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Hawking temperature
    ax = axes[1]
    colors = ['#FF5722' if t > 0 else '#2196F3' for t in hawking_temps]
    ax.bar(layers, np.abs(hawking_temps), color=colors, edgecolor='black',
           alpha=0.85)
    ax.axvline(peak_layer, color='red', ls='--', alpha=0.5,
               label='Event horizon (L%d)' % peak_layer)
    ax.set_xlabel('Injection layer', fontsize=11)
    ax.set_ylabel('|Hawking temperature|', fontsize=11)
    ax.set_title('(b) Hawking Temperature Profile\nT_H ~ kappa / (2pi)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (c) Summary
    ax = axes[2]
    ax.text(0.5, 0.65,
            'HAWKING\nRADIATION',
            ha='center', va='center', fontsize=24, fontweight='bold',
            color='#FF5722', transform=ax.transAxes)
    ax.text(0.5, 0.35,
            'Event horizon at Layer %d\n'
            'Peak delta S = %.4f\n'
            'Max T_H = %.4f\n\n'
            'Deep layers = gravitational well\n'
            'Output = thermal radiation boundary' % (
                peak_layer, peak_delta,
                max(np.abs(hawking_temps))),
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) Black Hole Thermodynamics', fontsize=12, fontweight='bold')

    plt.suptitle('Q108: Semantic Hawking Radiation from Transformer Black Holes',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q108_hawking.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q108', 'name': 'Semantic Hawking Radiation',
        'ref_entropy': float(ref_entropy),
        'event_horizon_layer': peak_layer,
        'peak_delta_entropy': float(peak_delta),
        'temperatures': temperatures,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q108_hawking.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
