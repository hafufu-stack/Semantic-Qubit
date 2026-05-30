# -*- coding: utf-8 -*-
"""
Phase Q227: Quantum Coherence (l1-norm & relative entropy)
============================================================
Coherence measures superposition in a given basis.
Distinct from entanglement (which needs bipartition) and discord.
If coherence is high, LLM maintains genuine quantum superpositions.
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


def l1_coherence(rho):
    """l1-norm of coherence: sum of absolute off-diagonal elements."""
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def relative_entropy_coherence(rho):
    """Relative entropy of coherence: S(diag(rho)) - S(rho)."""
    eigvals = np.real(np.linalg.eigvalsh(rho))
    eigvals = eigvals[eigvals > 1e-12]
    S_rho = -np.sum(eigvals * np.log2(eigvals))

    diag = np.diag(np.diag(rho)).real
    diag_vals = np.diag(diag)
    diag_vals = diag_vals[diag_vals > 1e-12]
    S_diag = -np.sum(diag_vals * np.log2(diag_vals))

    return float(max(0, S_diag - S_rho))


def max_coherence(dim):
    """Maximum possible coherence for a dim x dim density matrix."""
    return dim - 1  # l1-norm max


def main():
    print("=" * 60)
    print("Phase Q227: Quantum Coherence")
    print("  (Does the LLM maintain superpositions?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 8

    prompts = [
        "quantum superposition of states",
        "classical deterministic outcome",
        "Schrodinger cat alive and dead",
        "thermal equilibrium at room temperature",
        "coherent laser light source",
        "decoherence destroys quantum state",
    ]

    key_layers = list(range(0, n_layers + 1, 2))
    all_results = []

    for prompt in prompts:
        print("\n--- %s ---" % prompt[:35])
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layer_data = []
        for li in key_layers:
            if li >= len(out.hidden_states):
                continue
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)

            l1 = l1_coherence(rho)
            re = relative_entropy_coherence(rho)
            max_c = max_coherence(dim)
            norm_coherence = l1 / max_c

            layer_data.append({
                'layer': li,
                'l1_coherence': round(l1, 4),
                'relative_entropy': round(re, 4),
                'normalized': round(norm_coherence, 4),
            })

            if li % 8 == 0:
                print("  L%d: l1=%.4f (%.1f%% of max), RE=%.4f" %
                      (li, l1, norm_coherence * 100, re))

        all_results.append({
            'prompt': prompt[:35],
            'layers': layer_data,
        })

    # Summary
    all_l1 = [d['l1_coherence'] for r in all_results for d in r['layers']]
    all_norm = [d['normalized'] for r in all_results for d in r['layers']]
    avg_l1 = np.mean(all_l1)
    avg_norm = np.mean(all_norm)

    if avg_norm > 0.5:
        verdict = "HIGH COHERENCE: avg %.1f%% of maximum (l1=%.3f)" % (avg_norm * 100, avg_l1)
    elif avg_norm > 0.1:
        verdict = "MODERATE COHERENCE: avg %.1f%% of maximum" % (avg_norm * 100)
    else:
        verdict = "LOW COHERENCE: avg %.1f%%" % (avg_norm * 100)

    print("\n--- Summary ---")
    print("  Avg l1 coherence: %.4f (%.1f%% of max)" % (avg_l1, avg_norm * 100))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q227',
        'name': 'Quantum Coherence',
        'prompts': all_results,
        'summary': {
            'avg_l1': round(avg_l1, 4),
            'avg_normalized': round(avg_norm, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q227_coherence.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, r in enumerate(all_results[:6]):
        ax = axes[idx // 3][idx % 3]
        layers = [d['layer'] for d in r['layers']]
        norms = [d['normalized'] for d in r['layers']]
        ax.plot(layers, norms, 'o-', color='#9C27B0', lw=2, ms=3)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Normalized Coherence')
        ax.set_title(r['prompt'][:30], fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    plt.suptitle('Q227: Quantum Coherence\n%s' % verdict, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q227_coherence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ227 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
