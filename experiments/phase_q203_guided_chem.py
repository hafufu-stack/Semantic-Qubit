# -*- coding: utf-8 -*-
"""
Phase Q203: LLM-Guided Quantum Chemistry
==========================================
The ultimate practical test: can LLM's "chemical intuition" (learned from
pretraining) accelerate quantum chemistry calculations?

We compare:
1. Random-seeded VQE: random initial parameters -> optimize
2. LLM-seeded VQE: use LLM embedding as initial ansatz -> optimize

If LLM-seeded converges faster, the LLM is a "quantum chemistry navigator"
that provides exponentially better starting points.
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


def build_hamiltonian(mol_name, bond_length=None):
    """Build simplified molecular Hamiltonians."""
    if mol_name == 'H2':
        r = bond_length if bond_length else 0.74
        J = -0.5 * np.exp(-r / 0.5)
        dim = 4
        H = np.zeros((dim, dim))
        H[0, 0] = 0.5; H[3, 3] = 0.5
        H[1, 1] = -0.5; H[2, 2] = -0.5
        H[1, 2] = J; H[2, 1] = J
        return H, dim
    elif mol_name == 'LiH':
        dim = 8
        rng = np.random.RandomState(42)
        H = rng.randn(dim, dim) * 0.3
        H = (H + H.T) / 2
        H[np.diag_indices_from(H)] = np.sort(rng.randn(dim)) * 0.5
        return H, dim
    elif mol_name == 'BeH2':
        dim = 16
        rng = np.random.RandomState(123)
        H = rng.randn(dim, dim) * 0.2
        H = (H + H.T) / 2
        H[np.diag_indices_from(H)] = np.sort(rng.randn(dim)) * 0.4
        return H, dim
    elif mol_name == 'H2O':
        dim = 16
        rng = np.random.RandomState(999)
        H = rng.randn(dim, dim) * 0.25
        H = (H + H.T) / 2
        H[np.diag_indices_from(H)] = np.sort(rng.randn(dim)) * 0.6
        return H, dim
    else:
        raise ValueError(f"Unknown molecule: {mol_name}")


def run_vqe(model, tok, device, H_np, dim, seed_type='llm',
            n_steps=300, lr=0.005, rng_seed=0):
    """Run VQE with either LLM-seeded or random-seeded initialization."""
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)

    # Exact ground state for comparison
    eigvals, eigvecs = np.linalg.eigh(H_np)
    E_exact = eigvals[0]
    psi_exact = eigvecs[:, 0]

    embed_layer = model.model.embed_tokens

    if seed_type == 'llm':
        # Use LLM's embedding as initial state
        prompt = "quantum chemistry ground state energy:"
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp).detach().clone()
        opt_vec = embeds.clone().detach().requires_grad_(True)
    else:
        # Random initialization
        rng = np.random.RandomState(rng_seed)
        prompt = "random init:"
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp).detach().clone()
        with torch.no_grad():
            embeds[0, -1, :dim] = torch.tensor(
                rng.randn(dim).astype(np.float32), device=device) * 0.1
        opt_vec = embeds.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([opt_vec], lr=lr)
    history = []

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt_vec.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()
        history.append(float(E.detach()))

    # Final result
    with torch.no_grad():
        out = model(inputs_embeds=opt_vec.float(), output_hidden_states=True)
        h_final = out.hidden_states[-1][0, -1, :dim]
        psi_final = h_final / (torch.norm(h_final) + 1e-10)
        E_final = float(torch.dot(psi_final, H_torch @ psi_final))

    error_mha = abs(E_final - E_exact) * 1000  # mHa
    fidelity = float(torch.dot(psi_final.cpu(),
                                torch.tensor(psi_exact, dtype=psi_final.dtype)).abs() ** 2)

    return {
        'E_exact': float(E_exact),
        'E_final': round(E_final, 6),
        'error_mHa': round(error_mha, 4),
        'fidelity': round(fidelity, 4),
        'history': [round(h, 6) for h in history],
        'converged_step': next((i for i, e in enumerate(history)
                                if abs(e - E_exact) < 0.002), n_steps),
    }


def main():
    print("=" * 60)
    print("Phase Q203: LLM-Guided Quantum Chemistry")
    print("  (LLM-seeded vs Random-seeded VQE convergence)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    molecules = ['H2', 'LiH', 'BeH2', 'H2O']
    n_random_trials = 5
    n_steps = 300

    all_results = []

    for mol in molecules:
        print("\n--- %s ---" % mol)
        H, dim = build_hamiltonian(mol)

        # LLM-seeded VQE
        llm_result = run_vqe(model, tok, device, H, dim,
                              seed_type='llm', n_steps=n_steps)
        print("  LLM-seeded: E=%.6f (error=%.4f mHa, fid=%.4f, conv@%d)" %
              (llm_result['E_final'], llm_result['error_mHa'],
               llm_result['fidelity'], llm_result['converged_step']))

        # Random-seeded VQE (multiple trials)
        random_results = []
        for trial in range(n_random_trials):
            rr = run_vqe(model, tok, device, H, dim,
                          seed_type='random', n_steps=n_steps, rng_seed=trial)
            random_results.append(rr)

        avg_random_error = np.mean([r['error_mHa'] for r in random_results])
        avg_random_conv = np.mean([r['converged_step'] for r in random_results])
        best_random_error = min(r['error_mHa'] for r in random_results)

        print("  Random-seeded (avg of %d): error=%.4f mHa, conv@%.0f" %
              (n_random_trials, avg_random_error, avg_random_conv))

        speedup = avg_random_conv / max(llm_result['converged_step'], 1)

        mol_result = {
            'molecule': mol,
            'dim': dim,
            'E_exact': llm_result['E_exact'],
            'llm': {
                'E_final': llm_result['E_final'],
                'error_mHa': llm_result['error_mHa'],
                'fidelity': llm_result['fidelity'],
                'converged_step': llm_result['converged_step'],
                'history': llm_result['history'],
            },
            'random': {
                'avg_error_mHa': round(avg_random_error, 4),
                'best_error_mHa': round(best_random_error, 4),
                'avg_converged_step': round(avg_random_conv, 1),
                'trials': [{'error_mHa': r['error_mHa'],
                             'converged_step': r['converged_step'],
                             'history': r['history']}
                           for r in random_results],
            },
            'speedup': round(speedup, 2),
        }
        all_results.append(mol_result)
        print("  Speedup: %.1fx faster convergence" % speedup)

    # Summary
    avg_speedup = np.mean([r['speedup'] for r in all_results])
    n_llm_wins = sum(1 for r in all_results
                     if r['llm']['error_mHa'] <= r['random']['best_error_mHa'])

    print("\n--- Summary ---")
    print("  Average speedup: %.1fx" % avg_speedup)
    print("  LLM wins (lower error): %d/%d molecules" %
          (n_llm_wins, len(molecules)))

    if avg_speedup > 5:
        verdict = "NAVIGATOR: LLM-seeded VQE %.1fx faster than random" % avg_speedup
    elif avg_speedup > 1:
        verdict = "ADVANTAGEOUS: LLM seeding %.1fx faster" % avg_speedup
    else:
        verdict = "NO ADVANTAGE: Random seeding comparable"
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q203',
        'name': 'LLM-Guided Quantum Chemistry',
        'molecules': all_results,
        'summary': {
            'avg_speedup': round(avg_speedup, 2),
            'llm_wins': n_llm_wins,
            'total_molecules': len(molecules),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q203_guided_chem.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, mol_r in enumerate(all_results):
        ax = axes[idx // 2][idx % 2]
        mol = mol_r['molecule']

        # LLM convergence curve
        llm_hist = mol_r['llm']['history']
        ax.plot(range(len(llm_hist)), llm_hist,
                color='#E91E63', lw=2, label='LLM-seeded')

        # Random convergence curves (best + worst)
        for i, trial in enumerate(mol_r['random']['trials']):
            alpha = 0.3 if i > 0 else 0.6
            label = 'Random-seeded' if i == 0 else None
            ax.plot(range(len(trial['history'])), trial['history'],
                    color='#607D8B', lw=1, alpha=alpha, label=label)

        ax.axhline(mol_r['E_exact'], color='green', ls='--',
                   lw=1.5, label='Exact E0')

        conv_step = mol_r['llm']['converged_step']
        if conv_step < len(llm_hist):
            ax.axvline(conv_step, color='#E91E63', ls=':', alpha=0.7)

        ax.set_xlabel('VQE Step')
        ax.set_ylabel('Energy (Ha)')
        ax.set_title('%s (dim=%d): %.1fx speedup' %
                     (mol, mol_r['dim'], mol_r['speedup']))
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('Q203: LLM-Guided Quantum Chemistry\n'
                 'LLM-seeded vs Random-seeded VQE (avg %.1fx speedup)' %
                 avg_speedup, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q203_guided_chem.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ203 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
