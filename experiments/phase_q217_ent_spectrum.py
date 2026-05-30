# -*- coding: utf-8 -*-
"""
Phase Q217: Entanglement Spectrum Analysis
=============================================
The full entanglement spectrum (eigenvalues of reduced density matrix)
reveals the "entanglement structure" more deeply than single numbers.

Key question: Does the LLM's entanglement spectrum match known
quantum states (GHZ, W, thermal) or something entirely new?
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


def get_entanglement_spectrum(model, tok, device, prompt, layer, dim_a=4, dim_b=4):
    """Get full entanglement spectrum at given layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    dt = dim_a * dim_b
    h = out.hidden_states[layer][0, -1, :dt].float().cpu().numpy()
    h /= (np.linalg.norm(h) + 1e-10)

    # Build density matrix
    rho = np.outer(h, h.conj())
    # Mix for realism
    rho = 0.7 * rho + 0.3 * np.eye(dt) / dt
    rho /= np.trace(rho)

    # Partial trace to get reduced density matrix of subsystem A
    rho_a = np.zeros((dim_a, dim_a), dtype=complex)
    for i in range(dim_a):
        for j in range(dim_a):
            for k in range(dim_b):
                rho_a[i, j] += rho[i * dim_b + k, j * dim_b + k]

    spectrum = np.sort(np.real(np.linalg.eigvalsh(rho_a)))[::-1]
    return spectrum


def classify_spectrum(spectrum):
    """Classify entanglement spectrum type."""
    s = spectrum / (spectrum.sum() + 1e-10)
    n = len(s)

    # GHZ-like: one dominant eigenvalue
    if s[0] > 0.8:
        return "GHZ-like"

    # W-like: two near-equal eigenvalues
    if n >= 2 and abs(s[0] - s[1]) < 0.1 and (n < 3 or s[2] < s[1] * 0.3):
        return "W-like"

    # Thermal: exponential decay
    if n >= 3:
        ratios = [s[i+1] / max(s[i], 1e-10) for i in range(min(3, n-1))]
        if all(0.1 < r < 0.9 for r in ratios):
            return "Thermal"

    # Maximally entangled: flat spectrum
    if np.std(s) < 0.05:
        return "Maximally entangled"

    return "Novel"


def main():
    print("=" * 60)
    print("Phase Q217: Entanglement Spectrum Analysis")
    print("  (What type of entanglement does the LLM produce?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim_a, dim_b = 4, 4

    prompts = [
        "quantum entanglement between particles",
        "the cat is alive and dead simultaneously",
        "hydrogen bond energy landscape",
        "Bell inequality violation experiment",
        "topological quantum error correction",
        "many-body quantum phase transition",
    ]

    # Analyze at key layers
    key_layers = [0, 4, 8, 12, 16, 20, 24, n_layers]
    all_results = []

    for prompt in prompts:
        print("\n--- %s ---" % prompt[:40])
        prompt_data = []

        for layer in key_layers:
            if layer > n_layers:
                continue
            spectrum = get_entanglement_spectrum(
                model, tok, device, prompt, layer, dim_a, dim_b)
            spec_type = classify_spectrum(spectrum)

            # Entanglement entropy
            s_norm = spectrum / (spectrum.sum() + 1e-10)
            s_norm = s_norm[s_norm > 1e-12]
            ent_entropy = float(-np.sum(s_norm * np.log2(s_norm)))

            # Participation ratio
            pr = 1.0 / (np.sum(s_norm**2) + 1e-10)

            prompt_data.append({
                'layer': layer,
                'spectrum': [round(float(s), 6) for s in spectrum],
                'type': spec_type,
                'entropy': round(ent_entropy, 4),
                'participation_ratio': round(float(pr), 4),
            })

            if layer % 8 == 0:
                print("  L%d: [%s] S=%.3f PR=%.2f type=%s" %
                      (layer, ', '.join('%.3f' % s for s in spectrum[:4]),
                       ent_entropy, pr, spec_type))

        all_results.append({
            'prompt': prompt[:40],
            'layers': prompt_data,
        })

    # Count spectrum types
    type_counts = {}
    for pr in all_results:
        for ld in pr['layers']:
            t = ld['type']
            type_counts[t] = type_counts.get(t, 0) + 1

    dominant_type = max(type_counts, key=type_counts.get)
    total = sum(type_counts.values())

    verdict = "Dominant: %s (%d/%d = %.0f%%)" % (
        dominant_type, type_counts[dominant_type], total,
        type_counts[dominant_type] / total * 100)

    print("\n--- Summary ---")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print("  %s: %d (%.0f%%)" % (t, c, c / total * 100))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q217',
        'name': 'Entanglement Spectrum Analysis',
        'prompts': all_results,
        'summary': {
            'type_counts': type_counts,
            'dominant_type': dominant_type,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q217_ent_spectrum.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot: spectrum at different layers for first 3 prompts
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx in range(min(6, len(all_results))):
        ax = axes[idx // 3][idx % 3]
        pr = all_results[idx]
        for ld in pr['layers']:
            ax.plot(range(len(ld['spectrum'])), ld['spectrum'],
                    'o-', label='L%d (%s)' % (ld['layer'], ld['type']),
                    alpha=0.7, ms=4)
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(pr['prompt'][:30], fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    plt.suptitle('Q217: Entanglement Spectrum Analysis\n%s' % verdict,
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q217_ent_spectrum.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ217 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
