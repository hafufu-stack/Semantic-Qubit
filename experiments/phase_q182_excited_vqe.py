# -*- coding: utf-8 -*-
"""
Phase Q182: Excited State VQE
================================
Q161/Q180 found the GROUND state of H2 with 0.00 mHa error.
Q182: Can Embedding VQE also find EXCITED states?

Method: After finding ground state |psi_0>, add penalty:
  E_k = <psi|H|psi> + mu * sum_{i<k} |<psi|psi_i>|^2

This is the "deflation" approach - push new states orthogonal to found states.
If LLM can find excited states -> full quantum simulation capability.
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


def build_h2_hamiltonian(bond_length=0.74):
    """Same H2 Hamiltonian as previous experiments."""
    dim = 16
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)
    def kron4(a, b, c, d):
        return np.kron(np.kron(np.kron(a, b), c), d)
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
        g3 * kron4(Z, I2, Z, I2) * 0.5 * g2 / g3 +
        g3 * kron4(I2, Z, I2, Z) * 0.5 * g2 / g3)
    return H


def main():
    print("=" * 60)
    print("Phase Q182: Excited State VQE")
    print("  (Can LLM Find Higher Energy Eigenstates?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    hidden_size = model.config.hidden_size

    dim = 16
    H_np = build_h2_hamiltonian(0.74)
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)

    # Exact spectrum
    exact_evals, exact_evecs = np.linalg.eigh(H_np)
    n_states = min(6, dim)  # find first 6 eigenstates
    print("  Exact spectrum (first %d):" % n_states)
    for i in range(n_states):
        print("    E_%d = %.6f Ha" % (i, exact_evals[i]))

    # Get embedding layer
    embed_layer = model.model.embed_tokens
    seed_prompt = "Chemical bond quantum state energy level:"
    seed_ids = tok(seed_prompt, return_tensors='pt')['input_ids'].to(device)
    seed_embeds = embed_layer(seed_ids).detach().clone()

    found_states = []  # list of (psi_tensor, energy)
    all_convergence = []
    all_results = []

    for k in range(n_states):
        print("\n--- Finding E_%d ---" % k)

        # Fresh initialization for each state
        opt_embeds = seed_embeds.clone().detach()
        # Add random perturbation to break symmetry
        opt_embeds = opt_embeds + torch.randn_like(opt_embeds) * 0.01 * (k + 1)
        opt_embeds = opt_embeds.requires_grad_(True)
        optimizer = torch.optim.Adam([opt_embeds], lr=0.001)

        mu = 50.0  # penalty strength for orthogonality
        n_steps = 300

        energies = []
        for step in range(n_steps):
            optimizer.zero_grad()

            outputs = model(inputs_embeds=opt_embeds.float(),
                           output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]

            psi = h[:dim]
            psi_norm = psi / (torch.norm(psi) + 1e-10)

            # Energy
            E = psi_norm @ H_torch @ psi_norm

            # Orthogonality penalty: push away from all previously found states
            penalty = torch.tensor(0.0, device=device)
            for psi_prev, _ in found_states:
                overlap = torch.dot(psi_norm, psi_prev)
                penalty = penalty + overlap ** 2

            loss = E + mu * penalty
            loss.backward()
            optimizer.step()

            energies.append(float(E.detach()))

        # Extract final state
        with torch.no_grad():
            outputs = model(inputs_embeds=opt_embeds.float(),
                           output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            psi_final = h[:dim].float()
            psi_final = psi_final / (torch.norm(psi_final) + 1e-10)

        E_final = float(psi_final @ H_torch @ psi_final)
        error_mHa = abs(E_final - exact_evals[k]) * 1000

        # Fidelity with exact eigenstate
        fidelity = abs(float(psi_final.cpu().numpy() @ exact_evecs[:, k])) ** 2

        # Orthogonality check
        overlaps = []
        for psi_prev, _ in found_states:
            ov = abs(float(torch.dot(psi_final, psi_prev)))
            overlaps.append(ov)

        found_states.append((psi_final.clone(), E_final))
        all_convergence.append(energies)

        result = {
            'state': k,
            'E_vqe': round(E_final, 6),
            'E_exact': round(float(exact_evals[k]), 6),
            'error_mHa': round(error_mHa, 2),
            'fidelity': round(fidelity, 6),
            'overlaps_with_prev': [round(o, 6) for o in overlaps],
        }
        all_results.append(result)

        print("  E_%d: VQE=%.6f, Exact=%.6f, Error=%.2f mHa" %
              (k, E_final, exact_evals[k], error_mHa))
        print("  Fidelity=%.6f" % fidelity)
        if overlaps:
            print("  Max overlap with previous: %.6f" % max(overlaps))

    # Summary
    print("\n--- Excited State Summary ---")
    chem_acc = sum(1 for r in all_results if r['error_mHa'] < 1.6)
    sub_10 = sum(1 for r in all_results if r['error_mHa'] < 10.0)
    avg_fid = float(np.mean([r['fidelity'] for r in all_results]))
    print("  Chemical accuracy (<1.6 mHa): %d/%d" % (chem_acc, n_states))
    print("  <10 mHa: %d/%d" % (sub_10, n_states))
    print("  Avg fidelity: %.4f" % avg_fid)

    # Save
    results = {
        'phase': 'Q182',
        'name': 'Excited State VQE',
        'molecule': 'H2',
        'n_states': n_states,
        'states': all_results,
        'exact_spectrum': [round(float(e), 6) for e in exact_evals[:n_states]],
        'summary': {
            'chem_accuracy_count': chem_acc,
            'sub_10mHa_count': sub_10,
            'avg_fidelity': round(avg_fid, 4),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q182_excited_vqe.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Energy spectrum comparison
    ax = axes[0]
    x = np.arange(n_states)
    ax.bar(x - 0.15, [r['E_exact'] for r in all_results], 0.3,
           color='#4CAF50', label='Exact', alpha=0.85, edgecolor='black')
    ax.bar(x + 0.15, [r['E_vqe'] for r in all_results], 0.3,
           color='#2196F3', label='LLM VQE', alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['E_%d' % i for i in range(n_states)])
    ax.set_ylabel('Energy (Ha)')
    ax.set_title('(a) Energy Spectrum\n(Exact vs LLM VQE)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (b) Error per state
    ax = axes[1]
    errors = [r['error_mHa'] for r in all_results]
    colors = ['#4CAF50' if e < 1.6 else '#FF9800' if e < 10 else '#F44336'
              for e in errors]
    ax.bar(x, errors, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.6, color='red', ls='--', label='Chemical accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(['E_%d' % i for i in range(n_states)])
    ax.set_ylabel('Error (mHa)')
    ax.set_title('(b) Error per Eigenstate\n(%d/%d chemical accuracy)' %
                (chem_acc, n_states))
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Convergence curves
    ax = axes[2]
    for k in range(min(4, n_states)):
        ax.plot(range(len(all_convergence[k])), all_convergence[k],
                '-', linewidth=1.5, label='E_%d' % k)
        ax.axhline(exact_evals[k], ls=':', alpha=0.3)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title('(c) Convergence\n(Ground + Excited States)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Q182: Excited State VQE\n'
                 '(LLM finds %d eigenstates, avg fidelity=%.4f)' %
                 (n_states, avg_fid),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q182_excited_vqe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ182 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
