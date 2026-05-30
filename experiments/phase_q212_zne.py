# -*- coding: utf-8 -*-
"""
Phase Q212: Zero-Noise Extrapolation (ZNE)
============================================
Q207 showed LLM can't discover QEC codes. But modern quantum computers
use QEM (Quantum Error Mitigation) instead. ZNE injects controlled noise
at levels sigma, 2*sigma, 3*sigma, then extrapolates to sigma=0.

If this works, LLM becomes a noise-resilient quantum processor
via engineering, not physics.
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


def run_vqe_with_noise(model, tok, device, H_np, dim, noise_level,
                        n_steps=200, lr=0.005):
    """Run VQE with specified noise injection."""
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = np.linalg.eigh(H_np)[0][0]
    embed_layer = model.model.embed_tokens

    prompt = "quantum chemistry ground state:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]

        # Inject noise at specified level
        if noise_level > 0:
            noise = torch.randn_like(h) * noise_level
            h = h + noise

        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()

    with torch.no_grad():
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h_final = out.hidden_states[-1][0, -1, :dim]
        if noise_level > 0:
            h_final = h_final + torch.randn_like(h_final) * noise_level
        psi_final = h_final / (torch.norm(h_final) + 1e-10)
        E_final = float(torch.dot(psi_final, H_torch @ psi_final))

    return E_final, E_exact


def richardson_extrapolation(noise_levels, energies):
    """Richardson extrapolation to zero noise."""
    n = len(noise_levels)
    if n == 1:
        return energies[0]

    # Linear extrapolation
    coeffs = np.polyfit(noise_levels, energies, min(n - 1, 2))
    E_zne = np.polyval(coeffs, 0)
    return float(E_zne)


def main():
    print("=" * 60)
    print("Phase Q212: Zero-Noise Extrapolation (ZNE)")
    print("  (Quantum Error Mitigation for LLM)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Test molecules
    molecules = {
        'H2': {'dim': 4, 'seed': 42},
        'LiH': {'dim': 8, 'seed': 123},
        'BeH2': {'dim': 16, 'seed': 456},
    }

    # Noise scale factors
    base_noise = 0.05
    scale_factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    n_steps = 200

    all_results = []

    for mol_name, config in molecules.items():
        dim = config['dim']
        print("\n--- %s (dim=%d) ---" % (mol_name, dim))

        rng = np.random.RandomState(config['seed'])
        H = rng.randn(dim, dim) * 0.3
        H = (H + H.T) / 2
        H[np.diag_indices_from(H)] = np.sort(rng.randn(dim)) * 0.5
        E_exact = np.linalg.eigh(H)[0][0]

        # Clean run (no noise)
        E_clean, _ = run_vqe_with_noise(model, tok, device, H, dim, 0.0, n_steps)
        clean_error = abs(E_clean - E_exact) * 1000

        # Noisy runs at multiple levels
        noise_levels = [base_noise * s for s in scale_factors]
        noisy_energies = []
        noisy_errors = []

        for nl in noise_levels:
            E_noisy, _ = run_vqe_with_noise(model, tok, device, H, dim, nl, n_steps)
            noisy_energies.append(E_noisy)
            noisy_errors.append(abs(E_noisy - E_exact) * 1000)
            print("  noise=%.3f: E=%.6f, error=%.4f mHa" %
                  (nl, E_noisy, noisy_errors[-1]))

        # ZNE extrapolation
        E_zne = richardson_extrapolation(noise_levels, noisy_energies)
        zne_error = abs(E_zne - E_exact) * 1000

        # Also try with clean + noisy
        all_levels = [0.0] + noise_levels
        all_energies = [E_clean] + noisy_energies
        E_zne_full = richardson_extrapolation(all_levels, all_energies)
        zne_full_error = abs(E_zne_full - E_exact) * 1000

        print("  Clean: error=%.4f mHa" % clean_error)
        print("  ZNE (noisy only): error=%.4f mHa" % zne_error)
        print("  ZNE (with clean): error=%.4f mHa" % zne_full_error)

        # Best noisy error for comparison
        best_noisy = min(noisy_errors)
        improvement = best_noisy / max(zne_error, 0.0001)

        mol_result = {
            'molecule': mol_name,
            'dim': dim,
            'E_exact': round(E_exact, 6),
            'clean': {'E': round(E_clean, 6), 'error_mHa': round(clean_error, 4)},
            'noisy': [{'noise': round(nl, 3), 'E': round(e, 6),
                       'error_mHa': round(err, 4)}
                      for nl, e, err in zip(noise_levels, noisy_energies, noisy_errors)],
            'zne': {'E': round(E_zne, 6), 'error_mHa': round(zne_error, 4)},
            'zne_full': {'E': round(E_zne_full, 6), 'error_mHa': round(zne_full_error, 4)},
            'improvement_over_noisy': round(improvement, 2),
        }
        all_results.append(mol_result)

    # Summary
    avg_zne_error = np.mean([r['zne']['error_mHa'] for r in all_results])
    avg_noisy_error = np.mean([min(r2['error_mHa'] for r2 in r['noisy']) for r in all_results])
    avg_improvement = np.mean([r['improvement_over_noisy'] for r in all_results])

    if avg_improvement > 5:
        verdict = "ZNE SUCCESS: %.1fx improvement over best noisy" % avg_improvement
    elif avg_improvement > 1:
        verdict = "ZNE HELPS: %.1fx improvement" % avg_improvement
    else:
        verdict = "ZNE INEFFECTIVE: no improvement"

    print("\n--- Summary ---")
    print("  Avg ZNE error: %.4f mHa" % avg_zne_error)
    print("  Avg noisy error: %.4f mHa" % avg_noisy_error)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q212',
        'name': 'Zero-Noise Extrapolation',
        'molecules': all_results,
        'summary': {
            'avg_zne_error_mHa': round(avg_zne_error, 4),
            'avg_noisy_error_mHa': round(avg_noisy_error, 4),
            'avg_improvement': round(avg_improvement, 2),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q212_zne.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        noise_lvls = [n['noise'] for n in r['noisy']]
        errors = [n['error_mHa'] for n in r['noisy']]

        ax.plot(noise_lvls, errors, 'o-', color='#F44336', lw=2, label='Noisy VQE')
        ax.axhline(r['clean']['error_mHa'], color='#4CAF50', ls='-',
                   label='Clean (%.2f mHa)' % r['clean']['error_mHa'])
        ax.axhline(r['zne']['error_mHa'], color='#2196F3', ls='--', lw=2,
                   label='ZNE (%.2f mHa)' % r['zne']['error_mHa'])
        ax.scatter([0], [r['zne']['error_mHa']], color='#2196F3',
                   s=100, zorder=5, marker='*')

        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Error (mHa)')
        ax.set_title('%s (dim=%d)' % (r['molecule'], r['dim']))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Q212: Zero-Noise Extrapolation\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q212_zne.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ212 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
