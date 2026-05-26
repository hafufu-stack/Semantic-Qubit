# -*- coding: utf-8 -*-
"""
Phase Q148: Zero-Shot Semantic VQE (No Gradient Descent!)
==========================================================
Q144 showed Rayleigh GD does all the work. So we remove it entirely.

THE ULTIMATE TEST: Can the LLM produce a good quantum state
from JUST a text description, with ZERO optimization?

If this works -> LLM truly "understands" quantum systems
If this fails -> LLM is just a random vector generator (Grok wins)
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


def build_h2_hamiltonian(bond_length):
    """H2 molecule Hamiltonian in STO-3G basis (4 qubits, 16 dim)."""
    dim = 16

    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    I2 = np.eye(2)

    def kron4(a, b, c, d):
        return np.kron(np.kron(np.kron(a, b), c), d)

    # Parameterized H2 Hamiltonian coefficients (approximate)
    # These vary with bond length
    r = bond_length
    g0 = -0.5 - 0.2 * np.exp(-r)
    g1 = 0.2 * np.exp(-0.5 * r)
    g2 = 0.15 * np.exp(-0.3 * r)
    g3 = -0.1 * np.exp(-0.8 * r)

    H = np.real(
        g0 * kron4(I2, I2, I2, I2) +
        g1 * kron4(Z, I2, I2, I2) +
        g1 * kron4(I2, Z, I2, I2) +
        g2 * kron4(Z, Z, I2, I2) +
        g2 * kron4(I2, I2, Z, Z) +
        g3 * kron4(X, X, I2, I2) +
        g3 * kron4(I2, I2, X, X) +
        g3 * kron4(Y, Y, I2, I2) +
        g3 * kron4(I2, I2, Y, Y) +
        g2 * 0.5 * kron4(Z, I2, Z, I2) +
        g2 * 0.5 * kron4(I2, Z, I2, Z)
    )
    return H


def main():
    print("=" * 60)
    print("Phase Q148: Zero-Shot Semantic VQE")
    print("  (NO Gradient Descent!)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    dim = 16  # H2 in STO-3G

    # Test across multiple bond lengths
    bond_lengths = [0.5, 0.7, 0.74, 1.0, 1.5, 2.0, 3.0]

    all_results = []

    for r in bond_lengths:
        print("\n--- Bond length = %.2f Angstrom ---" % r)
        H = build_h2_hamiltonian(r)
        E_exact = float(np.linalg.eigvalsh(H)[0])
        psi_exact = np.linalg.eigh(H)[1][:, 0]

        # === METHOD 1: Zero-shot semantic (molecule-specific prompt) ===
        semantic_prompts = [
            "Molecule: H2. Bond length: %.2f Angstrom. Ground state energy:" % r,
            "Hydrogen molecule at distance %.2f A, wavefunction coefficients:" % r,
            "H2 potential energy surface at r=%.2f, ground state:" % r,
        ]

        zeroshot_psis = []
        for prompt in semantic_prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            # Extract S-Qubit from each layer
            for li in range(0, n_layers, 2):
                h = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
                for offset in range(0, min(hidden_size, dim * 10), dim):
                    if offset + dim <= hidden_size:
                        psi = h[offset:offset + dim].copy()
                        norm = np.linalg.norm(psi)
                        if norm > 1e-8:
                            zeroshot_psis.append(psi / norm)

        # Pick best WITHOUT any gradient descent
        scored = [(float(np.real(p @ H @ p)), p) for p in zeroshot_psis]
        scored.sort(key=lambda x: x[0])

        best_zs = scored[0][1]
        E_zeroshot = scored[0][0]
        zs_error = abs(E_zeroshot - E_exact) * 1000
        zs_fid = float(abs(np.dot(best_zs, psi_exact)) ** 2)

        # Also try pairwise mixing (still no GD!)
        best_E_pair = E_zeroshot
        best_pair = best_zs.copy()
        top_k = min(20, len(scored))
        for i in range(top_k):
            for j in range(i+1, top_k):
                for alpha in [0.3, 0.5, 0.7]:
                    mix = alpha * scored[i][1] + (1-alpha) * scored[j][1]
                    n = np.linalg.norm(mix)
                    if n > 1e-8:
                        mix /= n
                        Em = float(np.real(mix @ H @ mix))
                        if Em < best_E_pair:
                            best_E_pair = Em
                            best_pair = mix.copy()

        pair_error = abs(best_E_pair - E_exact) * 1000
        pair_fid = float(abs(np.dot(best_pair, psi_exact)) ** 2)

        # === METHOD 2: Generic prompt (no molecule info) ===
        generic_prompt = "The answer is:"
        inp_g = tok(generic_prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out_g = model(**inp_g, output_hidden_states=True)

        generic_psis = []
        for li in range(0, n_layers, 2):
            h = out_g.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
            for offset in range(0, min(hidden_size, dim * 10), dim):
                if offset + dim <= hidden_size:
                    psi = h[offset:offset + dim].copy()
                    norm = np.linalg.norm(psi)
                    if norm > 1e-8:
                        generic_psis.append(psi / norm)

        g_scored = [(float(np.real(p @ H @ p)), p) for p in generic_psis]
        g_scored.sort(key=lambda x: x[0])
        E_generic = g_scored[0][0]
        generic_error = abs(E_generic - E_exact) * 1000

        # === METHOD 3: Random baseline (no GD) ===
        random_errors = []
        for _ in range(100):
            psi_r = np.random.randn(dim)
            psi_r /= np.linalg.norm(psi_r)
            E_r = float(np.real(psi_r @ H @ psi_r))
            random_errors.append(abs(E_r - E_exact) * 1000)
        rand_best = min(random_errors)
        rand_mean = float(np.mean(random_errors))

        # Semantic advantage
        semantic_advantage = rand_best / max(zs_error, 0.001)

        result = {
            'bond_length': round(r, 2),
            'exact_energy': round(E_exact, 6),
            'zeroshot': {
                'energy': round(E_zeroshot, 6),
                'error_mha': round(zs_error, 4),
                'fidelity': round(zs_fid, 4),
            },
            'pairwise_zs': {
                'energy': round(best_E_pair, 6),
                'error_mha': round(pair_error, 4),
                'fidelity': round(pair_fid, 4),
            },
            'generic_prompt': {
                'error_mha': round(generic_error, 4),
            },
            'random_no_gd': {
                'best_error_mha': round(rand_best, 4),
                'mean_error_mha': round(rand_mean, 4),
            },
            'semantic_advantage': round(semantic_advantage, 2),
        }
        all_results.append(result)
        print("  Zero-shot:  %.4f mHa (F=%.4f)" % (zs_error, zs_fid))
        print("  + Pairwise: %.4f mHa (F=%.4f)" % (pair_error, pair_fid))
        print("  Generic:    %.4f mHa" % generic_error)
        print("  Random:     %.4f mHa (best of 100)" % rand_best)
        print("  -> Semantic advantage: %.1fx" % semantic_advantage)

    # Save
    results = {
        'phase': 'Q148',
        'name': 'Zero-Shot Semantic VQE (No GD)',
        'configs': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q148_zeroshot.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bls = [r['bond_length'] for r in all_results]

    ax = axes[0]
    zs_e = [r['zeroshot']['error_mha'] for r in all_results]
    pair_e = [r['pairwise_zs']['error_mha'] for r in all_results]
    gen_e = [r['generic_prompt']['error_mha'] for r in all_results]
    rand_e = [r['random_no_gd']['best_error_mha'] for r in all_results]
    ax.semilogy(bls, [max(e, 0.01) for e in zs_e], 'o-', color='#4CAF50',
                label='Semantic zero-shot', linewidth=2)
    ax.semilogy(bls, [max(e, 0.01) for e in pair_e], 's-', color='#2196F3',
                label='+ Pairwise', linewidth=2)
    ax.semilogy(bls, [max(e, 0.01) for e in gen_e], '^--', color='#FF9800',
                label='Generic prompt', linewidth=2)
    ax.semilogy(bls, [max(e, 0.01) for e in rand_e], 'x--', color='#F44336',
                label='Random (best/100)', linewidth=2)
    ax.axhline(1.6, color='blue', ls=':', alpha=0.5, label='Chem accuracy')
    ax.set_xlabel('Bond length (A)')
    ax.set_ylabel('Error (mHa, log)')
    ax.set_title('(a) Zero-Shot VQE (NO gradient descent)')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    ax = axes[1]
    zs_f = [r['zeroshot']['fidelity'] for r in all_results]
    pair_f = [r['pairwise_zs']['fidelity'] for r in all_results]
    ax.plot(bls, zs_f, 'o-', color='#4CAF50', label='Zero-shot', linewidth=2)
    ax.plot(bls, pair_f, 's-', color='#2196F3', label='+ Pairwise', linewidth=2)
    ax.set_xlabel('Bond length (A)')
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) Wavefunction Fidelity (no GD)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    adv = [r['semantic_advantage'] for r in all_results]
    colors = ['#4CAF50' if a > 1 else '#F44336' for a in adv]
    ax.bar(range(len(bls)), adv, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='red', ls='--', label='Random = LLM')
    ax.set_xticks(range(len(bls)))
    ax.set_xticklabels(['%.2f' % b for b in bls])
    ax.set_xlabel('Bond length (A)')
    ax.set_ylabel('Semantic advantage (vs random)')
    ax.set_title('(c) Does molecule-specific prompt help?\n(>1 = LLM wins)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q148: Zero-Shot Semantic VQE (NO Gradient Descent)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q148_zeroshot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ148 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
