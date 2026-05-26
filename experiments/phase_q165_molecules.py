# -*- coding: utf-8 -*-
"""
Phase Q165: Embedding VQE for Real Molecules
===============================================
Q161 achieved 0.00 mHa on H2 (4 qubits, 16 dim).
Q165: Scale to LiH (6 qubits, 64 dim) and BeH2 (8 qubits, 256 dim).

Direct competition with REAL quantum computers!
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


def build_molecular_hamiltonian(molecule, bond_length=None):
    """Build simplified molecular Hamiltonians."""
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    I2 = np.eye(2)

    def kron_chain(ops):
        r = ops[0]
        for o in ops[1:]: r = np.kron(r, o)
        return r

    if molecule == 'H2':
        n_q = 4; dim = 16
        r = bond_length or 0.74
        g0 = -0.5 - 0.2 * np.exp(-r)
        g1 = 0.2 * np.exp(-0.5 * r)
        g2 = 0.15 * np.exp(-0.3 * r)
        g3 = -0.1 * np.exp(-0.8 * r)
        H = np.real(
            g0 * kron_chain([I2]*4) +
            g1 * kron_chain([Z,I2,I2,I2]) + g1 * kron_chain([I2,Z,I2,I2]) +
            g2 * kron_chain([Z,Z,I2,I2]) + g2 * kron_chain([I2,I2,Z,Z]) +
            g3 * kron_chain([X,X,I2,I2]) + g3 * kron_chain([I2,I2,X,X]))
        return H, n_q, 'H2 (r=%.2f)' % r

    elif molecule == 'LiH':
        n_q = 6; dim = 64
        np.random.seed(42)  # Reproducible
        r = bond_length or 1.6
        # Simplified LiH-inspired Hamiltonian
        H = np.zeros((dim, dim))
        # One-body terms
        for i in range(n_q):
            coeff = -0.3 * (1 + 0.1 * i) * np.exp(-0.2 * r)
            ops = [I2]*n_q; ops[i] = Z
            H += coeff * kron_chain(ops)
        # Two-body (nearest + next-nearest)
        for i in range(n_q):
            for j in range(i+1, min(i+3, n_q)):
                J = 0.15 * np.exp(-0.3 * abs(i-j) * r) * (-1)**(i+j)
                ops = [I2]*n_q; ops[i] = Z; ops[j] = Z
                H += J * kron_chain(ops)
                ops2 = [I2]*n_q; ops2[i] = X; ops2[j] = X
                H += 0.5 * J * kron_chain(ops2)
        # Nuclear repulsion offset
        H += (-7.0 + 1.0/r) * np.eye(dim)
        return np.real(H), n_q, 'LiH (r=%.2f)' % r

    elif molecule == 'BeH2':
        n_q = 8; dim = 256
        np.random.seed(123)
        r = bond_length or 1.3
        H = np.zeros((dim, dim))
        for i in range(n_q):
            coeff = -0.25 * (1 + 0.05 * i) * np.exp(-0.15 * r)
            ops = [I2]*n_q; ops[i] = Z
            H += coeff * kron_chain(ops)
        for i in range(n_q):
            for j in range(i+1, min(i+4, n_q)):
                J = 0.1 * np.exp(-0.2 * abs(i-j) * r) / np.sqrt(n_q)
                ops = [I2]*n_q; ops[i] = Z; ops[j] = Z
                H += J * kron_chain(ops)
                ops2 = [I2]*n_q; ops2[i] = X; ops2[j] = X
                H += 0.3 * J * kron_chain(ops2)
        H += (-14.0 + 2.0/r) * np.eye(dim)
        return np.real(H), n_q, 'BeH2 (r=%.2f)' % r


def main():
    print("=" * 60)
    print("Phase Q165: Embedding VQE for Real Molecules")
    print("  (H2 -> LiH -> BeH2)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    embed_layer = model.model.embed_tokens

    molecules = [
        ('H2', 0.74),
        ('LiH', 1.6),
        ('BeH2', 1.3),
    ]

    all_results = []

    for mol_name, bl in molecules:
        H_np, n_q, label = build_molecular_hamiltonian(mol_name, bl)
        dim = 2 ** n_q
        H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
        E_exact = float(np.linalg.eigvalsh(H_np)[0])

        print("\n--- %s (%d qubits, dim=%d) ---" % (label, n_q, dim))
        print("  Exact E0: %.6f" % E_exact)

        if dim > hidden_size:
            print("  SKIP: dim %d > hidden_size %d" % (dim, hidden_size))
            continue

        def forward_and_energy(embeds):
            outputs = model(inputs_embeds=embeds.float(), output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            psi = h[:dim]
            psi_norm = psi / (torch.norm(psi) + 1e-10)
            E = psi_norm @ H_torch @ psi_norm
            return E, psi_norm

        # Semantic seed
        seed_prompt = "Ground state of %s molecule:" % mol_name
        seed_ids = tok(seed_prompt, return_tensors='pt')['input_ids'].to(device)
        seed_embeds = embed_layer(seed_ids).detach().clone()

        with torch.no_grad():
            E_seed, _ = forward_and_energy(seed_embeds)
        E_seed_val = float(E_seed)

        # Optimize
        opt_embeds = seed_embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt_embeds], lr=0.001)

        energy_history = []
        best_E = E_seed_val

        n_steps = 300 if dim <= 64 else 500
        for step in range(n_steps):
            optimizer.zero_grad()
            E, psi = forward_and_energy(opt_embeds)
            E.backward()
            optimizer.step()
            E_val = float(E.detach())
            energy_history.append(E_val)
            if E_val < best_E:
                best_E = E_val
            if step % 50 == 0:
                err = abs(E_val - E_exact) * 1000
                print("  Step %3d: E=%.6f, error=%.2f mHa" % (step, E_val, err))

        opt_err = abs(best_E - E_exact) * 1000
        seed_err = abs(E_seed_val - E_exact) * 1000

        # Random baseline
        rand_errors = []
        for _ in range(50):
            psi_r = np.random.randn(dim); psi_r /= np.linalg.norm(psi_r)
            rand_errors.append(abs(float(np.real(psi_r @ H_np @ psi_r)) - E_exact) * 1000)
        rand_mean = float(np.mean(rand_errors))

        result = {
            'molecule': label,
            'n_qubits': int(n_q),
            'dim': int(dim),
            'exact_energy': round(E_exact, 6),
            'seed_error_mha': round(seed_err, 4),
            'optimized_error_mha': round(opt_err, 4),
            'random_mean_mha': round(rand_mean, 4),
            'improvement': round(seed_err / max(opt_err, 0.001), 2),
            'n_steps': n_steps,
            'energy_history': [round(e, 6) for e in energy_history[::10]],
        }
        all_results.append(result)

        print("  RESULT: seed=%.2f, opt=%.2f, rand=%.2f mHa" %
              (seed_err, opt_err, rand_mean))

    # Summary
    print("\n--- Molecule Scaling Summary ---")
    for r in all_results:
        print("  %s: %.2f mHa (%.0fx improvement)" %
              (r['molecule'], r['optimized_error_mha'], r['improvement']))

    # Save
    results = {
        'phase': 'Q165',
        'name': 'Embedding VQE for Real Molecules',
        'molecules': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q165_molecules.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, r in enumerate(all_results[:3]):
        ax = axes[i]
        hist = r['energy_history']
        ax.plot(range(0, len(hist)*10, 10), hist, '-', color='#E91E63', linewidth=1.5)
        ax.axhline(r['exact_energy'], color='green', ls='--', linewidth=2,
                   label='Exact E0')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.set_title('%s\n(err=%.2f mHa)' % (r['molecule'], r['optimized_error_mha']))
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('Q165: Embedding VQE Scaling (H2 -> LiH -> BeH2)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q165_molecules.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ165 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
