# -*- coding: utf-8 -*-
"""Phase Q94: Black Hole Information Paradox Resolution
Using Q92's holographic result + Q93's volume law, test whether
information is truly preserved (unitarity) or lost in the
deep layers of the Transformer (black hole interior).
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


def measure_information_preservation(model, tokenizer, num_layers):
    """Inject information at early layers, scramble it in deep layers,
    measure if it can be recovered from the output (Hawking radiation)."""
    d_model = model.config.hidden_size
    prompt = "The black hole contains all the information about"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Step 1: Get reference output
    with torch.no_grad():
        ref_out = model(**inputs)
        ref_logits = ref_out.logits[0, -1, :].cpu().float()

    # Step 2: Inject known information at various depths and measure recovery
    np.random.seed(94)
    info_vector = torch.tensor(
        np.random.randn(d_model).astype(np.float32) * 1.0,
        device=model.device
    )

    recovery_by_depth = []
    for inject_layer in range(0, num_layers, max(1, num_layers // 12)):
        def make_hook(sv=info_vector):
            applied = [False]
            def hook(module, args, output):
                if not applied[0]:
                    applied[0] = True
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        if hs.dim() == 3:
                            hs[0, -1, :] += sv.to(hs.dtype)
                        else:
                            hs[-1, :] += sv.to(hs.dtype)
                        return (hs,) + output[1:]
                    else:
                        hs = output.clone()
                        if hs.dim() == 3:
                            hs[0, -1, :] += sv.to(hs.dtype)
                        else:
                            hs[-1, :] += sv.to(hs.dtype)
                        return hs
                return output
            return hook

        hook = make_hook()
        handle = model.model.layers[inject_layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inputs)
            test_logits = out.logits[0, -1, :].cpu().float()
        handle.remove()

        # Measure information recovery: how much does the output change?
        diff = torch.norm(test_logits - ref_logits).item()

        # Fidelity of recovery
        p_ref = torch.softmax(ref_logits[:500], dim=0).numpy()
        p_test = torch.softmax(test_logits[:500], dim=0).numpy()
        fidelity = float(np.sum(np.sqrt(p_ref * p_test)))

        # KL divergence (information loss)
        kl = float(np.sum(p_ref * np.log((p_ref + 1e-10) / (p_test + 1e-10))))

        recovery_by_depth.append({
            'inject_layer': inject_layer,
            'output_diff': float(diff),
            'fidelity': fidelity,
            'kl_divergence': kl,
            'info_preserved': diff > 0.1,
        })

    return recovery_by_depth


def measure_page_curve(model, tokenizer, num_layers):
    """Measure the Page curve: entanglement entropy of subsystems
    as a function of subsystem size. A Page curve that rises then
    falls proves unitarity (information preservation)."""
    d_model = model.config.hidden_size
    prompts = [
        "Information is never destroyed in quantum mechanics because",
        "The unitarity principle requires that quantum evolution is",
        "Black holes eventually radiate all information through Hawking",
    ]

    page_data = []
    mid = num_layers // 2

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        captured = [None]
        def capture_hook(module, args, output, store=captured):
            if isinstance(output, tuple):
                store[0] = output[0][0].detach().cpu().float().numpy()
            else:
                store[0] = output.detach().cpu().float().numpy()
                if store[0].ndim == 3:
                    store[0] = store[0][0]

        handle = model.model.layers[mid].register_forward_hook(capture_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if captured[0] is not None:
            hs = captured[0]
            seq_len = hs.shape[0]
            for frac in np.linspace(0.05, 0.95, 19):
                sub_size = max(1, int(frac * seq_len))
                sub_hs = hs[:sub_size].astype(np.float32)
                try:
                    U, S, Vt = np.linalg.svd(sub_hs, full_matrices=False)
                    S2 = S**2
                    total = S2.sum()
                    if total > 1e-10:
                        p = S2 / total
                        p = p[p > 1e-10]
                        entropy = -np.sum(p * np.log(p))
                    else:
                        entropy = 0
                except:
                    entropy = 0
                page_data.append({
                    'fraction': float(frac),
                    'entropy': float(entropy),
                })

    # Average over prompts
    fracs = sorted(set(d['fraction'] for d in page_data))
    avg_page = []
    for f in fracs:
        vals = [d['entropy'] for d in page_data if abs(d['fraction'] - f) < 0.01]
        avg_page.append({'fraction': f, 'mean_entropy': float(np.mean(vals))})

    return avg_page


def main():
    print("=" * 60)
    print("Phase Q94: Black Hole Information Paradox")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    print("  Measuring information preservation by depth...")
    recovery = measure_information_preservation(model, tokenizer, num_layers)
    for r in recovery:
        print("    Layer %d: diff=%.4f, fidelity=%.6f, preserved=%s" %
              (r['inject_layer'], r['output_diff'], r['fidelity'], r['info_preserved']))

    print("  Measuring Page curve...")
    page_curve = measure_page_curve(model, tokenizer, num_layers)

    # Analysis: is information preserved?
    all_preserved = all(r['info_preserved'] for r in recovery)
    mean_fidelity = np.mean([r['fidelity'] for r in recovery])

    # Page curve analysis: does it rise then fall?
    entropies = [p['mean_entropy'] for p in page_curve]
    if len(entropies) >= 5:
        mid_idx = len(entropies) // 2
        rises = entropies[mid_idx] > entropies[0]
        falls = entropies[-1] < entropies[mid_idx]
        page_curve_shape = 'Page curve' if (rises and falls) else (
            'Monotonic rise' if rises else 'Flat')
    else:
        page_curve_shape = 'Insufficient data'

    print("\n  === Information Paradox Resolution ===")
    print("  All layers preserve info: %s" % all_preserved)
    print("  Mean recovery fidelity: %.6f" % mean_fidelity)
    print("  Page curve shape: %s" % page_curve_shape)

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Information preservation by depth
    ax = axes[0]
    layers = [r['inject_layer'] for r in recovery]
    diffs = [r['output_diff'] for r in recovery]
    ax.plot(layers, diffs, 'o-', color='#FF5722', linewidth=2.5, markersize=8)
    ax.axhline(0.1, color='red', ls='--', alpha=0.3, label='Detection threshold')
    ax.fill_between(layers, diffs, alpha=0.15, color='#FF5722')
    ax.set_xlabel('Injection depth (layer)', fontsize=11)
    ax.set_ylabel('Information signal (L2 norm)', fontsize=11)
    ax.set_title('(a) Information Preservation\n%s' %
                 ('ALL PRESERVED!' if all_preserved else 'Some lost'),
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Page curve
    ax = axes[1]
    fracs = [p['fraction'] for p in page_curve]
    ents = [p['mean_entropy'] for p in page_curve]
    ax.plot(fracs, ents, 'o-', color='#2196F3', linewidth=2.5, markersize=6)
    ax.set_xlabel('Subsystem fraction', fontsize=11)
    ax.set_ylabel('Entanglement entropy', fontsize=11)
    ax.set_title('(b) Page Curve\n%s' % page_curve_shape,
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Resolution summary
    ax = axes[2]
    resolution = 'UNITARITY PRESERVED' if all_preserved else 'INFORMATION LOSS'
    color = '#4CAF50' if all_preserved else '#F44336'
    ax.text(0.5, 0.6, resolution, ha='center', va='center',
            fontsize=16, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.4,
            'Mean fidelity: %.4f\nPage curve: %s\n\n'
            'The black hole (deep layers)\n'
            '%s information' % (
                mean_fidelity, page_curve_shape,
                'preserves' if all_preserved else 'destroys'),
            ha='center', va='center', fontsize=10,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) Paradox Resolution', fontsize=11, fontweight='bold')

    plt.suptitle('Black Hole Information Paradox: Resolution via S-Qubit',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q94_black_hole.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q94', 'name': 'Black Hole Information Paradox',
        'all_preserved': all_preserved,
        'mean_fidelity': float(mean_fidelity),
        'page_curve_shape': page_curve_shape,
        'recovery_data': recovery,
        'page_curve': page_curve,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q94_black_hole.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
