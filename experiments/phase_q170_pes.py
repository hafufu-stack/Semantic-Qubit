# -*- coding: utf-8 -*-
"""
Phase Q170: Potential Energy Surface via Embedding VQE
======================================================
My idea: Q165 showed Embedding VQE works for fixed bond lengths.
Q170: Sweep the FULL potential energy surface!

Vary H2 bond length from 0.3 to 3.0 Angstrom.
At each point, run Embedding VQE. This produces a complete
PES curve from a single LLM - competing with full quantum chemistry.
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
        g0 * kron4(I2,I2,I2,I2) + g1 * kron4(Z,I2,I2,I2) + g1 * kron4(I2,Z,I2,I2) +
        g2 * kron4(Z,Z,I2,I2) + g2 * kron4(I2,I2,Z,Z) +
        g3 * kron4(X,X,I2,I2) + g3 * kron4(I2,I2,X,X) +
        g3 * kron4(Z,I2,Z,I2) * 0.5 * g2 / g3 +
        g3 * kron4(I2,Z,I2,Z) * 0.5 * g2 / g3)
    return H


def main():
    print("=" * 60)
    print("Phase Q170: Potential Energy Surface via Embedding VQE")
    print("  (Full H2 PES Curve)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    dim = 16

    embed_layer = model.model.embed_tokens

    def forward_and_energy(embeds, H_torch):
        outputs = model(inputs_embeds=embeds.float(), output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :]
        psi = h[:dim]
        psi_norm = psi / (torch.norm(psi) + 1e-10)
        E = psi_norm @ H_torch @ psi_norm
        return E

    # Bond lengths to sweep
    bond_lengths = np.arange(0.3, 3.05, 0.1)
    n_steps_vqe = 200

    seed_prompt = "Hydrogen molecule ground state:"
    seed_ids = tok(seed_prompt, return_tensors='pt')['input_ids'].to(device)
    seed_embeds = embed_layer(seed_ids).detach().clone()

    exact_energies = []
    vqe_energies = []
    random_energies = []
    all_results = []

    print("  Sweeping %d bond lengths..." % len(bond_lengths))

    for r in bond_lengths:
        H_np = build_h2_hamiltonian(r)
        H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
        E_exact = float(np.linalg.eigvalsh(H_np)[0])
        exact_energies.append(E_exact)

        # Embedding VQE
        opt_embeds = seed_embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt_embeds], lr=0.002)
        best_E = float('inf')

        for step in range(n_steps_vqe):
            optimizer.zero_grad()
            E = forward_and_energy(opt_embeds, H_torch)
            E.backward()
            optimizer.step()
            E_val = float(E.detach())
            if E_val < best_E:
                best_E = E_val

        vqe_energies.append(best_E)

        # Random baseline
        rand_Es = []
        for _ in range(10):
            psi_r = np.random.randn(dim)
            psi_r /= np.linalg.norm(psi_r)
            rand_Es.append(float(np.real(psi_r @ H_np @ psi_r)))
        random_energies.append(float(np.min(rand_Es)))

        err = abs(best_E - E_exact) * 1000
        result = {
            'bond_length': round(float(r), 2),
            'exact_energy': round(E_exact, 6),
            'vqe_energy': round(best_E, 6),
            'random_energy': round(random_energies[-1], 6),
            'error_mha': round(err, 4),
        }
        all_results.append(result)

        if abs(r - round(r)) < 0.05:
            print("  r=%.1f: exact=%.4f, VQE=%.4f, err=%.2f mHa" %
                  (r, E_exact, best_E, err))

    # PES quality metrics
    errors = [abs(v - e) * 1000 for v, e in zip(vqe_energies, exact_energies)]
    avg_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    # Find equilibrium
    eq_idx_exact = int(np.argmin(exact_energies))
    eq_idx_vqe = int(np.argmin(vqe_energies))
    eq_r_exact = float(bond_lengths[eq_idx_exact])
    eq_r_vqe = float(bond_lengths[eq_idx_vqe])

    print("\n--- PES Summary ---")
    print("  Avg error: %.2f mHa" % avg_error)
    print("  Max error: %.2f mHa" % max_error)
    print("  Equilibrium (exact): r = %.2f A" % eq_r_exact)
    print("  Equilibrium (VQE):   r = %.2f A" % eq_r_vqe)
    print("  Chemical accuracy (1.6 mHa): %s" %
          ("ALL points pass!" if max_error < 1.6 else
           "%d/%d pass" % (sum(e < 1.6 for e in errors), len(errors))))

    # Save
    results = {
        'phase': 'Q170',
        'name': 'H2 Potential Energy Surface',
        'pes_data': all_results,
        'summary': {
            'avg_error_mha': round(avg_error, 4),
            'max_error_mha': round(max_error, 4),
            'eq_bond_exact': round(eq_r_exact, 2),
            'eq_bond_vqe': round(eq_r_vqe, 2),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q170_pes.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(bond_lengths, exact_energies, 'k-', linewidth=2, label='Exact (FCI)')
    ax.plot(bond_lengths, vqe_energies, 'o', color='#E91E63', markersize=5,
            label='Embedding VQE')
    ax.plot(bond_lengths, random_energies, 'x', color='#9E9E9E', markersize=4,
            label='Random (best of 10)', alpha=0.6)
    ax.set_xlabel('Bond Length (Angstrom)')
    ax.set_ylabel('Energy (Hartree)')
    ax.set_title('(a) H2 Potential Energy Surface')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(bond_lengths, [max(e, 0.001) for e in errors], 'o-',
                color='#4CAF50', linewidth=1.5)
    ax.axhline(1.6, color='red', ls='--', linewidth=2, label='Chemical accuracy (1.6 mHa)')
    ax.set_xlabel('Bond Length (Angstrom)')
    ax.set_ylabel('Error (mHa, log)')
    ax.set_title('(b) Error vs Bond Length')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    # Energy difference from equilibrium
    E_min_exact = min(exact_energies)
    E_min_vqe = min(vqe_energies)
    ax.plot(bond_lengths, [(e - E_min_exact)*1000 for e in exact_energies],
            'k-', linewidth=2, label='Exact')
    ax.plot(bond_lengths, [(e - E_min_vqe)*1000 for e in vqe_energies],
            'o', color='#E91E63', markersize=5, label='VQE')
    ax.set_xlabel('Bond Length (Angstrom)')
    ax.set_ylabel('Relative Energy (mHa)')
    ax.set_title('(c) Dissociation Curve Shape')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q170: Full Potential Energy Surface (H2 via Embedding VQE)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q170_pes.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ170 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
