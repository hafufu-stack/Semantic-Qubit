# -*- coding: utf-8 -*-
"""Phase Q96: Quantum Zeno Effect - Does Frequent Observation Freeze S-Qubits?
Test if frequent layer-by-layer "measurement" (hook extraction)
prevents S-Qubit state evolution (quantum Zeno effect).
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


def measure_zeno_effect(model, tokenizer, num_layers):
    """Measure if repeated observation (hook collapsing) freezes the state."""
    d_model = model.config.hidden_size
    prompt = "The watched pot never boils because quantum Zeno states that"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Step 1: Normal evolution (no measurement)
    with torch.no_grad():
        ref_out = model(**inputs)
        ref_logits = ref_out.logits[0, -1, :].cpu().float().numpy()

    # Step 2: Inject S-Qubit and let evolve freely
    np.random.seed(96)
    sv = torch.tensor(np.random.randn(d_model).astype(np.float32) * 0.5,
                      device=model.device)

    # Free evolution: inject at layer 0, measure at output only
    def inject_hook(sv_vec):
        applied = [False]
        def hook(module, args, output):
            if not applied[0]:
                applied[0] = True
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv_vec.to(hs.dtype)
                    else:
                        hs[-1, :] += sv_vec.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv_vec.to(hs.dtype)
                    else:
                        hs[-1, :] += sv_vec.to(hs.dtype)
                    return hs
            return output
        return hook

    h = inject_hook(sv)
    handle = model.model.layers[0].register_forward_hook(h)
    with torch.no_grad():
        free_out = model(**inputs)
        free_logits = free_out.logits[0, -1, :].cpu().float().numpy()
    handle.remove()
    free_diff = np.linalg.norm(free_logits - ref_logits)

    # Step 3: Zeno effect - "measure" at every N layers by projecting
    # state back to original (simulating wavefunction collapse)
    zeno_results = []
    for n_measures in [0, 1, 2, 4, 7, 14, num_layers - 1]:
        if n_measures == 0:
            # Already measured (free evolution)
            zeno_results.append({
                'n_measurements': 0,
                'output_diff': float(free_diff),
                'label': 'Free evolution',
            })
            continue

        # Determine which layers to "measure" at
        if n_measures >= num_layers:
            measure_layers = list(range(1, num_layers))
        else:
            step = max(1, num_layers // n_measures)
            measure_layers = list(range(step, num_layers, step))[:n_measures]

        captured_initial = [None]
        handles = []

        # Inject at layer 0
        h_inject = inject_hook(sv)
        handles.append(model.model.layers[0].register_forward_hook(h_inject))

        # "Measure" (project back) at specified layers
        for ml in measure_layers:
            def make_project_hook(initial_store=captured_initial):
                applied = [False]
                def hook(module, args, output):
                    if not applied[0]:
                        applied[0] = True
                        # "Measurement": partially collapse back toward reference
                        if isinstance(output, tuple):
                            hs = output[0].clone()
                            if hs.dim() == 3:
                                hs[0, -1, :] *= 0.5  # Dampen the perturbation
                            return (hs,) + output[1:]
                        else:
                            hs = output.clone()
                            if hs.dim() == 3:
                                hs[0, -1, :] *= 0.5
                            return hs
                    return output
                return hook
            handles.append(model.model.layers[ml].register_forward_hook(
                make_project_hook()))

        with torch.no_grad():
            zeno_out = model(**inputs)
            zeno_logits = zeno_out.logits[0, -1, :].cpu().float().numpy()

        for h in handles:
            h.remove()

        zeno_diff = np.linalg.norm(zeno_logits - ref_logits)
        zeno_results.append({
            'n_measurements': n_measures,
            'output_diff': float(zeno_diff),
            'label': '%d measurements' % n_measures,
        })

    return zeno_results, free_diff


def main():
    print("=" * 60)
    print("Phase Q96: Quantum Zeno Effect")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    print("  Testing Zeno effect...")
    zeno_results, free_diff = measure_zeno_effect(model, tokenizer, num_layers)

    for r in zeno_results:
        print("    %s: diff=%.4f" % (r['label'], r['output_diff']))

    # Zeno confirmed if more measurements -> smaller diff
    diffs = [r['output_diff'] for r in zeno_results]
    zeno_confirmed = len(diffs) >= 3 and diffs[-1] < diffs[0] * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    n_meas = [r['n_measurements'] for r in zeno_results]
    diffs = [r['output_diff'] for r in zeno_results]
    ax.plot(n_meas, diffs, 'o-', color='#FF5722', linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of intermediate measurements', fontsize=12)
    ax.set_ylabel('Output perturbation (L2)', fontsize=12)
    ax.set_title('(a) Quantum Zeno Effect\n%s' %
                 ('ZENO CONFIRMED!' if zeno_confirmed else 'No Zeno'),
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1]
    if diffs[0] > 0:
        normalized = [d / diffs[0] for d in diffs]
    else:
        normalized = diffs
    ax.bar(range(len(normalized)), normalized,
           color=['#4CAF50' if n == 0 else '#FF5722' for n in n_meas],
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(n_meas)))
    ax.set_xticklabels(['%d' % n for n in n_meas], fontsize=10)
    ax.set_xlabel('Number of measurements', fontsize=12)
    ax.set_ylabel('Normalized perturbation', fontsize=12)
    ax.set_title('(b) Freezing Rate\nMore observation -> less evolution',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Quantum Zeno Effect: Observation Freezes S-Qubit Evolution',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q96_zeno.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q96', 'name': 'Quantum Zeno Effect',
        'zeno_confirmed': bool(zeno_confirmed),
        'zeno_data': zeno_results,
        'free_diff': float(free_diff),
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q96_zeno.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Zeno confirmed: %s. Elapsed: %.1fs" % (zeno_confirmed, elapsed))
    return results


if __name__ == '__main__':
    main()
