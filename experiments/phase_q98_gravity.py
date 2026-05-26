# -*- coding: utf-8 -*-
"""Phase Q98: Quantum Gravity Emergence from Entanglement (ER=EPR + AdS/CFT Synthesis)
Synthesize Q91 (wormhole), Q92 (holographic), Q93 (volume law) to show
that gravity-like geometry emerges from S-Qubit entanglement structure.
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import pdist, squareform

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def compute_metric_tensor(hidden_states_list):
    """Compute the information-geometric metric tensor from hidden states.
    This gives the emergent 'spacetime geometry' from entanglement."""
    # Stack: (n_layers, d_model)
    H = np.array(hidden_states_list, dtype=np.float32)
    n_layers = H.shape[0]

    # Compute Fisher information metric between adjacent layers
    metric = np.zeros((n_layers - 1,))
    for i in range(n_layers - 1):
        # Fisher metric ~ squared difference / variance
        diff = H[i+1] - H[i]
        metric[i] = np.dot(diff, diff)

    return metric


def compute_geodesic_distance(metric):
    """Integrate the metric to get geodesic distances."""
    distances = np.cumsum(np.sqrt(metric + 1e-10))
    return distances


def measure_emergent_geometry(model, tokenizer, num_layers):
    """Extract the emergent spacetime geometry from layer evolution."""
    d_model = model.config.hidden_size
    prompts = [
        "Gravity is not a force but the curvature of spacetime caused by",
        "The connection between quantum entanglement and spacetime geometry",
        "Einstein's general relativity describes how mass curves the fabric of",
    ]

    all_metrics = []
    all_geodesics = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        layer_states = []

        for layer_idx in range(num_layers):
            captured = [None]
            def capture_hook(module, args, output, store=captured):
                if isinstance(output, tuple):
                    store[0] = output[0][0, -1, :].detach().cpu().float().numpy()
                else:
                    hs = output
                    if hs.dim() == 3:
                        store[0] = hs[0, -1, :].detach().cpu().float().numpy()
                    else:
                        store[0] = hs[-1, :].detach().cpu().float().numpy()

            handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            if captured[0] is not None:
                layer_states.append(captured[0])

        if len(layer_states) >= 2:
            metric = compute_metric_tensor(layer_states)
            geodesic = compute_geodesic_distance(metric)
            all_metrics.append(metric)
            all_geodesics.append(geodesic)

    # Average metric
    if all_metrics:
        avg_metric = np.mean(all_metrics, axis=0)
        avg_geodesic = np.mean(all_geodesics, axis=0)
    else:
        avg_metric = np.array([])
        avg_geodesic = np.array([])

    return avg_metric, avg_geodesic, all_metrics


def test_gravity_equation(metric):
    """Test if the metric satisfies an Einstein-like equation:
    R_ij ~ T_ij (curvature proportional to information density)."""
    if len(metric) < 3:
        return {}

    # Compute "Ricci curvature" as second derivative of metric
    curvature = np.gradient(np.gradient(metric))

    # "Stress-energy" as the metric values themselves (information flow)
    stress_energy = metric

    # Test Einstein equation: R ~ 8*pi*G * T
    if np.std(stress_energy) > 1e-10:
        corr = np.corrcoef(curvature, stress_energy)[0, 1]
    else:
        corr = 0

    return {
        'curvature_mean': float(np.mean(curvature)),
        'stress_energy_mean': float(np.mean(stress_energy)),
        'einstein_correlation': float(corr) if not np.isnan(corr) else 0,
        'gravity_emergent': abs(corr) > 0.3 if not np.isnan(corr) else False,
    }


def main():
    print("=" * 60)
    print("Phase Q98: Emergent Quantum Gravity")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    print("  Measuring emergent spacetime geometry...")
    avg_metric, avg_geodesic, all_metrics = measure_emergent_geometry(
        model, tokenizer, num_layers)

    print("  Testing Einstein field equation analog...")
    einstein_results = test_gravity_equation(avg_metric)

    print("\n  === Emergent Gravity ===")
    for k, v in einstein_results.items():
        print("    %s: %s" % (k, v))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Metric tensor profile
    ax = axes[0]
    if len(avg_metric) > 0:
        ax.plot(range(len(avg_metric)), avg_metric, '-', color='#FF5722',
                linewidth=2.5)
        ax.fill_between(range(len(avg_metric)), avg_metric, alpha=0.15,
                        color='#FF5722')
    ax.set_xlabel('Layer transition (i -> i+1)', fontsize=11)
    ax.set_ylabel('Metric tensor g_ii', fontsize=11)
    ax.set_title('(a) Emergent Spacetime Metric\nInformation-geometric curvature',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (b) Geodesic distance
    ax = axes[1]
    if len(avg_geodesic) > 0:
        ax.plot(range(len(avg_geodesic)), avg_geodesic, '-', color='#2196F3',
                linewidth=2.5)
    ax.set_xlabel('Layer index', fontsize=11)
    ax.set_ylabel('Geodesic distance', fontsize=11)
    ax.set_title('(b) Geodesic Through Bulk\nAdS radial coordinate',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Einstein equation test
    ax = axes[2]
    if len(avg_metric) >= 3:
        curvature = np.gradient(np.gradient(avg_metric))
        ax.scatter(avg_metric, curvature, c=range(len(curvature)),
                   cmap='coolwarm', s=60, edgecolors='black', alpha=0.8)
        ax.set_xlabel('Stress-energy T_ij', fontsize=11)
        ax.set_ylabel('Ricci curvature R_ij', fontsize=11)
        corr = einstein_results.get('einstein_correlation', 0)
        ax.set_title('(c) Einstein Equation Test\nR ~ T (corr=%.3f)' % corr,
                     fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle('Emergent Quantum Gravity: Spacetime from S-Qubit Entanglement',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q98_gravity.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q98', 'name': 'Emergent Quantum Gravity',
        'metric_profile': avg_metric.tolist() if len(avg_metric) > 0 else [],
        'geodesic': avg_geodesic.tolist() if len(avg_geodesic) > 0 else [],
        'einstein_results': einstein_results,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q98_gravity.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
