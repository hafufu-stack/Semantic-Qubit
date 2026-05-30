# -*- coding: utf-8 -*-
"""
Phase Q216: Barren Plateau Visualization
==========================================
WHY does Q211's LLM-seeding break the curse of dimensionality?
Visualize the VQE energy landscape: LLM starts near the global minimum
while random seeds land in barren plateaus.
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


def build_hamiltonian(dim, seed=42):
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim) * 0.3
    H = (H + H.T) / 2
    H[np.diag_indices_from(H)] = np.sort(rng.randn(dim)) * 0.5
    return H


def compute_energy(model, tok, device, H_torch, dim, embedding):
    """Compute energy for given embedding without optimization."""
    with torch.no_grad():
        out = model(inputs_embeds=embedding.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = float(torch.dot(psi, H_torch @ psi))
    return E


def main():
    print("=" * 60)
    print("Phase Q216: Barren Plateau Visualization")
    print("  (Why does LLM-seeding work?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dims = [4, 16, 64]
    n_random_samples = 50
    n_perturbation_steps = 20

    all_results = []

    for dim in dims:
        print("\n--- Dimension %d ---" % dim)
        H = build_hamiltonian(dim, seed=42 + dim)
        H_torch = torch.tensor(H, dtype=torch.float32, device=device)
        E_exact = float(np.linalg.eigh(H)[0][0])
        embed_layer = model.model.embed_tokens

        # LLM starting point
        prompt = "ground state energy dimension %d:" % dim
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        llm_embed = embed_layer(inp).detach().clone()
        E_llm = compute_energy(model, tok, device, H_torch, dim, llm_embed)

        # Random starting points
        random_energies = []
        rng = np.random.RandomState(42)
        for i in range(n_random_samples):
            rand_embed = llm_embed.clone()
            with torch.no_grad():
                rand_embed[0, -1, :dim] = torch.tensor(
                    rng.randn(dim).astype(np.float32), device=device) * 0.5
            E_rand = compute_energy(model, tok, device, H_torch, dim, rand_embed)
            random_energies.append(E_rand)

        # Landscape around LLM point (perturbations)
        llm_landscape = []
        for eps in np.linspace(-1, 1, n_perturbation_steps):
            perturbed = llm_embed.clone()
            with torch.no_grad():
                perturbed[0, -1, 0] += eps
            E_pert = compute_energy(model, tok, device, H_torch, dim, perturbed)
            llm_landscape.append((float(eps), E_pert))

        # Gradient magnitude at LLM vs random
        def compute_gradient_norm(embedding):
            emb = embedding.clone().detach().requires_grad_(True)
            out = model(inputs_embeds=emb.float(), output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :dim]
            psi = h / (torch.norm(h) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward()
            return float(torch.norm(emb.grad).item())

        grad_llm = compute_gradient_norm(llm_embed)

        grad_randoms = []
        for i in range(min(10, n_random_samples)):
            rand_embed = llm_embed.clone()
            with torch.no_grad():
                rand_embed[0, -1, :dim] = torch.tensor(
                    rng.randn(dim).astype(np.float32), device=device) * 0.5
            grad_randoms.append(compute_gradient_norm(rand_embed))

        avg_grad_random = np.mean(grad_randoms)
        grad_ratio = grad_llm / max(avg_grad_random, 1e-10)

        print("  E_exact=%.4f, E_llm=%.4f, E_random_avg=%.4f" %
              (E_exact, E_llm, np.mean(random_energies)))
        print("  LLM distance to exact: %.4f" % abs(E_llm - E_exact))
        print("  Random distance to exact: %.4f" % abs(np.mean(random_energies) - E_exact))
        print("  Gradient: LLM=%.6f, Random=%.6f, ratio=%.2f" %
              (grad_llm, avg_grad_random, grad_ratio))

        result = {
            'dim': dim,
            'E_exact': round(E_exact, 6),
            'E_llm': round(E_llm, 6),
            'llm_distance': round(abs(E_llm - E_exact), 6),
            'random_energies': [round(e, 6) for e in random_energies],
            'random_avg_distance': round(abs(np.mean(random_energies) - E_exact), 6),
            'gradient_llm': round(grad_llm, 6),
            'gradient_random_avg': round(avg_grad_random, 6),
            'gradient_ratio': round(grad_ratio, 4),
            'landscape': llm_landscape,
        }
        all_results.append(result)

    # Summary
    avg_dist_ratio = np.mean([r['llm_distance'] / max(r['random_avg_distance'], 1e-6)
                              for r in all_results])
    avg_grad_ratio = np.mean([r['gradient_ratio'] for r in all_results])

    if avg_dist_ratio < 0.5:
        verdict = "LLM AVOIDS BARREN PLATEAU: %.1fx closer to minimum, %.1fx steeper gradient" % (
            1/avg_dist_ratio, avg_grad_ratio)
    else:
        verdict = "SIMILAR LANDSCAPE: distance ratio=%.2f" % avg_dist_ratio

    print("\n--- Summary ---")
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q216',
        'name': 'Barren Plateau Visualization',
        'dimensions': all_results,
        'summary': {
            'avg_distance_ratio': round(avg_dist_ratio, 4),
            'avg_gradient_ratio': round(avg_grad_ratio, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q216_barren_plateau.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        dim = r['dim']

        # Random energies histogram
        ax.hist(r['random_energies'], bins=15, alpha=0.5, color='#607D8B',
                label='Random starts', density=True)
        ax.axvline(r['E_llm'], color='#E91E63', lw=3,
                   label='LLM start (%.4f)' % r['E_llm'])
        ax.axvline(r['E_exact'], color='green', lw=2, ls='--',
                   label='Exact E0 (%.4f)' % r['E_exact'])

        ax.set_xlabel('Energy')
        ax.set_ylabel('Density')
        ax.set_title('dim=%d (LLM %.0fx closer)' % (
            dim, 1/max(r['llm_distance']/max(r['random_avg_distance'], 1e-6), 0.01)))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Q216: Barren Plateau Visualization\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q216_barren_plateau.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ216 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
