# -*- coding: utf-8 -*-
"""
Phase Q209: Entanglement Witness via Partial Transpose
========================================================
Formal test: does the LLM produce GENUINELY entangled states?

We use the Peres-Horodecki criterion (PPT criterion):
if the partial transpose of a density matrix has a negative eigenvalue,
the state is PROVABLY entangled (not just correlated).

This addresses Grok's criticism: "It's just classical correlation."
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


def build_density_matrix(model, tok, device, prompt, dim_a=4, dim_b=4):
    """Build a density matrix from the LLM's hidden state,
    treating the first dim_a dimensions as subsystem A
    and next dim_b as subsystem B."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Use multiple layers to build a mixed state (density matrix)
    dim_total = dim_a * dim_b
    rho = np.zeros((dim_total, dim_total), dtype=complex)

    # Sample from multiple layers for a mixed state
    for layer_idx in [8, 12, 16, 20, 24]:
        if layer_idx < len(out.hidden_states):
            h = out.hidden_states[layer_idx][0, -1, :dim_total].float().cpu().numpy()
            h = h / (np.linalg.norm(h) + 1e-10)
            rho += np.outer(h, h.conj())

    rho = rho / np.trace(rho)  # normalize
    return rho


def partial_transpose(rho, dim_a, dim_b):
    """Compute partial transpose over subsystem B."""
    rho_pt = np.zeros_like(rho)
    for i in range(dim_a):
        for j in range(dim_a):
            for k in range(dim_b):
                for l in range(dim_b):
                    # Transpose indices for subsystem B: k<->l
                    rho_pt[i * dim_b + k, j * dim_b + l] = \
                        rho[i * dim_b + l, j * dim_b + k]
    return rho_pt


def compute_negativity(rho, dim_a, dim_b):
    """Compute negativity (entanglement measure) from partial transpose."""
    rho_pt = partial_transpose(rho, dim_a, dim_b)
    eigvals = np.linalg.eigvalsh(rho_pt)
    negative_eigvals = eigvals[eigvals < -1e-10]
    negativity = float(np.sum(np.abs(negative_eigvals)))
    min_eigval = float(np.min(eigvals))
    return negativity, min_eigval, eigvals


def compute_concurrence_2qubit(rho):
    """Compute concurrence for a 2-qubit (4x4) density matrix."""
    sigma_y = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho.conj() @ yy
    R = rho @ rho_tilde
    eigvals = np.sort(np.abs(np.linalg.eigvals(R)))[::-1]
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    concurrence = max(0, sqrt_eigvals[0] - sqrt_eigvals[1] -
                       sqrt_eigvals[2] - sqrt_eigvals[3])
    return float(concurrence)


def main():
    print("=" * 60)
    print("Phase Q209: Entanglement Witness (PPT Criterion)")
    print("  (Formal proof of genuine entanglement)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Test prompts
    prompts = [
        "quantum entanglement between two particles",
        "the cat is both alive and dead",
        "hydrogen molecule bond",
        "Einstein Podolsky Rosen paradox",
        "superposition of spin up and spin down",
        "Bell state maximally entangled",
        "classical correlation without entanglement",
        "random noise signal",
    ]

    # Test different subsystem sizes
    configs = [
        {'dim_a': 2, 'dim_b': 2, 'name': '2x2 (2-qubit)'},
        {'dim_a': 2, 'dim_b': 4, 'name': '2x4'},
        {'dim_a': 4, 'dim_b': 4, 'name': '4x4'},
    ]

    all_results = []

    for config in configs:
        dim_a, dim_b = config['dim_a'], config['dim_b']
        config_name = config['name']
        print("\n--- %s ---" % config_name)

        config_results = []
        for prompt in prompts:
            rho = build_density_matrix(model, tok, device, prompt, dim_a, dim_b)

            negativity, min_eigval, eigvals = compute_negativity(rho, dim_a, dim_b)

            # Purity
            purity = float(np.real(np.trace(rho @ rho)))

            # Von Neumann entropy
            eigvals_rho = np.real(np.linalg.eigvalsh(rho))
            eigvals_rho = eigvals_rho[eigvals_rho > 1e-12]
            entropy = float(-np.sum(eigvals_rho * np.log2(eigvals_rho)))

            # Concurrence (only for 2x2)
            concurrence = None
            if dim_a == 2 and dim_b == 2:
                concurrence = compute_concurrence_2qubit(rho)

            is_entangled = negativity > 1e-6
            result = {
                'prompt': prompt[:40],
                'negativity': round(negativity, 6),
                'min_eigval_pt': round(min_eigval, 6),
                'is_entangled': is_entangled,
                'purity': round(purity, 4),
                'entropy': round(entropy, 4),
            }
            if concurrence is not None:
                result['concurrence'] = round(concurrence, 6)

            config_results.append(result)

            status = "ENTANGLED" if is_entangled else "separable"
            print("  %s: neg=%.6f (%s)" %
                  (prompt[:30], negativity, status))

        n_entangled = sum(1 for r in config_results if r['is_entangled'])
        all_results.append({
            'config': config_name,
            'dim_a': dim_a, 'dim_b': dim_b,
            'n_entangled': n_entangled,
            'total': len(prompts),
            'results': config_results,
        })
        print("  Entangled: %d/%d prompts" % (n_entangled, len(prompts)))

    # Summary
    total_entangled = sum(r['n_entangled'] for r in all_results)
    total_tests = sum(r['total'] for r in all_results)
    ent_rate = total_entangled / max(total_tests, 1)

    if ent_rate > 0.8:
        verdict = "GENUINE ENTANGLEMENT: %.0f%% of states PPT-certified entangled" % (ent_rate * 100)
    elif ent_rate > 0.3:
        verdict = "PARTIAL ENTANGLEMENT: %.0f%% entangled (context-dependent)" % (ent_rate * 100)
    else:
        verdict = "MOSTLY SEPARABLE: %.0f%% entangled" % (ent_rate * 100)

    print("\n--- Summary ---")
    print("  Total entangled: %d/%d (%.1f%%)" %
          (total_entangled, total_tests, ent_rate * 100))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q209',
        'name': 'Entanglement Witness (PPT Criterion)',
        'configs': all_results,
        'summary': {
            'total_entangled': total_entangled,
            'total_tests': total_tests,
            'entanglement_rate': round(ent_rate, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q209_entanglement_witness.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, cfg_r in enumerate(all_results):
        ax = axes[idx]
        names = [r['prompt'][:20] for r in cfg_r['results']]
        negs = [r['negativity'] for r in cfg_r['results']]
        colors = ['#E91E63' if r['is_entangled'] else '#9E9E9E'
                  for r in cfg_r['results']]

        ax.barh(range(len(names)), negs, color=colors,
                edgecolor='black', alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel('Negativity')
        ax.set_title('%s (%d/%d entangled)' %
                     (cfg_r['config'], cfg_r['n_entangled'], cfg_r['total']),
                     fontsize=10)
        ax.grid(alpha=0.3, axis='x')

    plt.suptitle('Q209: Entanglement Witness (PPT Criterion)\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q209_entanglement_witness.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ209 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
