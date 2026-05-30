# -*- coding: utf-8 -*-
"""
Phase Q224: Area Law vs Volume Law
=====================================
In quantum many-body physics, entanglement entropy scales as:
- Area law: S ~ L^(d-1) (gapped systems, most ground states)
- Volume law: S ~ L^d (thermal/chaotic systems)

Which law does the LLM follow? This reveals the "phase of matter"
of the Transformer's internal quantum state.
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


def entanglement_entropy(h, subsystem_size, total_dim):
    """Compute entanglement entropy for a bipartition of size subsystem_size."""
    h = h[:total_dim]
    h = h / (np.linalg.norm(h) + 1e-10)

    dim_a = subsystem_size
    dim_b = total_dim // subsystem_size
    if dim_a * dim_b != total_dim:
        return 0.0

    # Reshape as matrix and SVD
    psi_mat = h.reshape(dim_a, dim_b)
    s = np.linalg.svd(psi_mat, compute_uv=False)
    s2 = s ** 2
    s2 = s2[s2 > 1e-12]
    s2 /= s2.sum()

    entropy = float(-np.sum(s2 * np.log2(s2)))
    return entropy


def main():
    print("=" * 60)
    print("Phase Q224: Area Law vs Volume Law")
    print("  (What phase of matter is the Transformer?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    prompts = [
        "quantum ground state of matter",
        "thermal equilibrium at finite temperature",
        "many-body localization disorder",
        "topological insulator surface states",
        "quantum phase transition critical point",
        "superconducting Cooper pairs",
    ]

    # Test system sizes (total_dim = subsystem^2)
    system_configs = [
        (4, [2]),           # 2x2
        (8, [2, 4]),        # 2x4, 4x2
        (16, [2, 4, 8]),    # 2x8, 4x4, 8x2
        (32, [2, 4, 8, 16]),
        (64, [2, 4, 8, 16, 32]),
    ]

    # Use last hidden state
    n_layers = len(model.model.layers)
    key_layers = [0, n_layers // 2, n_layers]

    all_results = []

    for total_dim, subsys_sizes in system_configs:
        print("\n--- Total dim=%d ---" % total_dim)

        for layer_idx in key_layers:
            entropies = []
            for sub_size in subsys_sizes:
                prompt_entropies = []
                for prompt in prompts:
                    inp = tok(prompt, return_tensors='pt').to(device)
                    with torch.no_grad():
                        out = model(**inp, output_hidden_states=True)
                    h = out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
                    S = entanglement_entropy(h, sub_size, total_dim)
                    prompt_entropies.append(S)

                avg_S = np.mean(prompt_entropies)
                entropies.append({
                    'subsystem_size': sub_size,
                    'boundary_size': sub_size,  # 1D: boundary = constant
                    'volume': sub_size,
                    'entropy': round(avg_S, 4),
                })

            all_results.append({
                'total_dim': total_dim,
                'layer': layer_idx,
                'entropies': entropies,
            })

            if layer_idx == n_layers:
                for e in entropies:
                    print("  sub=%d: S=%.4f" % (e['subsystem_size'], e['entropy']))

    # Fit area law vs volume law for largest system
    largest = [r for r in all_results if r['total_dim'] == 64 and r['layer'] == n_layers]
    if largest:
        entropies = largest[0]['entropies']
        sizes = [e['subsystem_size'] for e in entropies]
        Ss = [e['entropy'] for e in entropies]

        if len(sizes) > 2:
            # Volume law: S ~ log(L) or S ~ L
            log_sizes = np.log2(sizes)
            # Linear fit: S = a * log(L) + b (logarithmic correction)
            coeffs_log = np.polyfit(log_sizes, Ss, 1)
            # Linear fit: S = a * L + b (volume law)
            coeffs_vol = np.polyfit(sizes, Ss, 1)
            # Constant fit (area law in 1D)
            coeffs_area = [np.mean(Ss)]

            # R^2 for each
            ss_tot = np.sum((np.array(Ss) - np.mean(Ss)) ** 2)
            if ss_tot > 1e-10:
                r2_log = 1 - np.sum((np.array(Ss) - np.polyval(coeffs_log, log_sizes)) ** 2) / ss_tot
                r2_vol = 1 - np.sum((np.array(Ss) - np.polyval(coeffs_vol, sizes)) ** 2) / ss_tot
            else:
                r2_log = 0
                r2_vol = 0
        else:
            r2_log = 0
            r2_vol = 0
            coeffs_log = [0, 0]
            coeffs_vol = [0, 0]
    else:
        r2_log = 0
        r2_vol = 0
        coeffs_log = [0, 0]
        coeffs_vol = [0, 0]

    if r2_vol > r2_log and r2_vol > 0.5:
        law = "VOLUME LAW"
    elif r2_log > 0.5:
        law = "LOG LAW (critical)"
    else:
        law = "AREA LAW (or flat)"

    verdict = "%s: S ~ %s (R2_vol=%.3f, R2_log=%.3f)" % (
        law,
        "L" if law == "VOLUME LAW" else ("log(L)" if "LOG" in law else "const"),
        r2_vol, r2_log)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q224',
        'name': 'Area Law vs Volume Law',
        'data': all_results,
        'summary': {
            'law': law,
            'r2_volume': round(r2_vol, 4),
            'r2_log': round(r2_log, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q224_area_volume_law.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for li_idx, layer_idx in enumerate(key_layers):
        ax = axes[li_idx]
        for r in all_results:
            if r['layer'] == layer_idx:
                sizes = [e['subsystem_size'] for e in r['entropies']]
                Ss = [e['entropy'] for e in r['entropies']]
                ax.plot(sizes, Ss, 'o-', label='dim=%d' % r['total_dim'], alpha=0.7, ms=4)

        ax.set_xlabel('Subsystem Size')
        ax.set_ylabel('Entanglement Entropy (bits)')
        ax.set_title('Layer %d' % layer_idx)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Q224: Area Law vs Volume Law\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q224_area_volume_law.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ224 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
