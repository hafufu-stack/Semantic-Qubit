# -*- coding: utf-8 -*-
"""
Phase Q125: Multi-Molecule VQE (LiH, BeH2, H2O)
=================================================
Extends Q120's triumph (0.6 mHa for H2) to larger molecules.

Q120 showed that variational optimization over hidden state
basis vectors achieves chemical accuracy. Can it scale?

- LiH: 4 qubits (16-dim Hilbert space)
- BeH2: 6 qubits (64-dim Hilbert space)
- H2O: 8 qubits (256-dim Hilbert space) - the "holy grail"
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


def build_molecular_hamiltonian(molecule, n_qubits):
    """Build approximate molecular Hamiltonian as sum of Pauli terms."""
    dim = 2 ** n_qubits
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)

    def kron_chain(ops):
        """Build tensor product of a list of 2x2 operators."""
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def Z_i(i, n):
        """Z operator on qubit i in n-qubit system."""
        ops = [I2] * n
        ops[i] = Z
        return kron_chain(ops)

    def ZZ_ij(i, j, n):
        ops = [I2] * n
        ops[i] = Z
        ops[j] = Z
        return kron_chain(ops)

    def XX_ij(i, j, n):
        ops = [I2] * n
        ops[i] = X
        ops[j] = X
        return kron_chain(ops)

    H = np.zeros((dim, dim))

    if molecule == 'LiH':
        # LiH: 4-qubit Hamiltonian (simplified Bravyi-Kitaev)
        n = 4
        H += -0.22 * np.eye(dim)
        H += 0.17 * Z_i(0, n) + 0.12 * Z_i(1, n)
        H += -0.17 * Z_i(2, n) + 0.17 * Z_i(3, n)
        H += 0.12 * ZZ_ij(0, 1, n) + 0.04 * ZZ_ij(0, 2, n)
        H += 0.17 * ZZ_ij(1, 2, n) + 0.04 * ZZ_ij(2, 3, n)
        H += 0.04 * XX_ij(0, 1, n) + 0.04 * XX_ij(2, 3, n)

    elif molecule == 'BeH2':
        # BeH2: 6-qubit simplified Hamiltonian
        n = 6
        H += -0.15 * np.eye(dim)
        for i in range(n):
            H += (0.10 - 0.03 * i) * Z_i(i, n)
        for i in range(n - 1):
            H += (0.08 - 0.02 * i) * ZZ_ij(i, i+1, n)
            H += 0.03 * XX_ij(i, i+1, n)

    elif molecule == 'H2O':
        # H2O: 8-qubit (very simplified)
        n = 8
        H += -0.32 * np.eye(dim)
        for i in range(n):
            H += (0.15 - 0.04 * i) * Z_i(i, n)
        for i in range(n - 1):
            H += (0.10 - 0.02 * i) * ZZ_ij(i, i+1, n)
            H += 0.02 * XX_ij(i, i+1, n)
        # Long-range interactions
        for i in range(n - 2):
            H += 0.01 * ZZ_ij(i, i+2, n)

    return H


def sqbit_vqe(model, tok, device, H, molecule_name, n_qubits):
    """Run S-Qubit VQE for a given Hamiltonian."""
    dim = 2 ** n_qubits
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # Exact ground state
    E_exact = float(np.linalg.eigvalsh(H)[0])

    # Get hidden states
    prompt = "Ground state energy of %s molecule. Qubits=%d:" % (molecule_name, n_qubits)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Collect basis vectors from all layers
    basis = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
        # Extract dim-sized slices
        for offset in range(0, min(hidden, dim * 8), dim):
            if offset + dim <= hidden:
                psi = h[offset:offset+dim].copy()
                norm = np.linalg.norm(psi)
                if norm > 1e-8:
                    psi /= norm
                    basis.append(psi)

    # Score all basis vectors
    scored = []
    for psi in basis:
        E = float(np.real(psi @ H @ psi))
        if not np.isnan(E):
            scored.append((E, psi))
    scored.sort(key=lambda x: x[0])

    if not scored:
        return E_exact, float('inf'), float('inf')

    best_E = scored[0][0]
    best_psi = scored[0][1].copy()

    # Linear combination search (top candidates)
    top_k = min(15, len(scored))
    for i in range(top_k):
        for j in range(i+1, top_k):
            for alpha in np.linspace(0, 1, 11):
                psi_mix = alpha * scored[i][1] + (1 - alpha) * scored[j][1]
                norm = np.linalg.norm(psi_mix)
                if norm > 1e-8:
                    psi_mix /= norm
                    E_mix = float(np.real(psi_mix @ H @ psi_mix))
                    if not np.isnan(E_mix) and E_mix < best_E:
                        best_E = E_mix
                        best_psi = psi_mix.copy()

    # Gradient-free refinement
    for _ in range(100):
        perturbation = np.random.randn(dim) * 0.03
        psi_trial = best_psi + perturbation
        psi_trial /= np.linalg.norm(psi_trial)
        E_trial = float(np.real(psi_trial @ H @ psi_trial))
        if not np.isnan(E_trial) and E_trial < best_E:
            best_E = E_trial
            best_psi = psi_trial.copy()

    error_ha = abs(best_E - E_exact)
    error_mha = error_ha * 1000

    return E_exact, best_E, error_mha


def main():
    print("=" * 60)
    print("Phase Q125: Multi-Molecule VQE (LiH, BeH2, H2O)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    molecules = [
        ('H2', 2),
        ('LiH', 4),
        ('BeH2', 6),
        ('H2O', 8),
    ]

    mol_results = []
    for mol_name, n_qubits in molecules:
        print("\n--- %s (%d qubits, %d-dim) ---" %
              (mol_name, n_qubits, 2**n_qubits))
        t_mol = time.time()

        if mol_name == 'H2':
            # Use Q120's hamiltonian
            from phase_q120_improved_vqe import h2_hamiltonian
            H = h2_hamiltonian(0.74)
        else:
            H = build_molecular_hamiltonian(mol_name, n_qubits)

        E_exact, E_sqbit, error_mha = sqbit_vqe(
            model, tok, device, H, mol_name, n_qubits)
        mol_time = time.time() - t_mol

        chem_acc = error_mha < 1.6
        result = {
            'molecule': mol_name,
            'n_qubits': n_qubits,
            'hilbert_dim': 2 ** n_qubits,
            'exact_energy': round(float(E_exact), 6),
            'sqbit_energy': round(float(E_sqbit), 6),
            'error_mha': round(float(error_mha), 2),
            'chemical_accuracy': str(chem_acc),
            'time_s': round(mol_time, 2),
        }
        mol_results.append(result)
        print("  Exact=%.4f, S-Qubit=%.4f, error=%.2f mHa %s (%.1fs)" %
              (E_exact, E_sqbit, error_mha,
               "PASS" if chem_acc else "FAIL", mol_time))

    # Summary
    n_pass = sum(1 for r in mol_results if r['chemical_accuracy'] == 'True')
    mean_error = float(np.mean([r['error_mha'] for r in mol_results]))

    print("\n--- Summary ---")
    print("  Chemical accuracy: %d/%d" % (n_pass, len(mol_results)))
    print("  Mean error: %.2f mHa" % mean_error)
    print("  Largest molecule with chem accuracy: %s" %
          max([r['molecule'] for r in mol_results
               if r['chemical_accuracy'] == 'True'],
              key=lambda x: {'H2': 2, 'LiH': 4, 'BeH2': 6, 'H2O': 8}.get(x, 0),
              default='None'))

    # ===== Save =====
    results = {
        'phase': 'Q125',
        'name': 'Multi-Molecule VQE',
        'molecules': mol_results,
        'n_chemical_accuracy': n_pass,
        'mean_error_mha': round(mean_error, 2),
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q125_multi_vqe.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Energy comparison
    ax = axes[0]
    names = [r['molecule'] for r in mol_results]
    exact = [r['exact_energy'] for r in mol_results]
    sqbit = [r['sqbit_energy'] for r in mol_results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, exact, 0.4, label='Exact', color='#333333', alpha=0.85)
    ax.bar(x + 0.2, sqbit, 0.4, label='S-Qubit', color='#4CAF50', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Energy (Ha)')
    ax.set_title('(a) Ground State Energies')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (b) Error scaling with qubits
    ax = axes[1]
    qubits = [r['n_qubits'] for r in mol_results]
    errors = [max(r['error_mha'], 0.01) for r in mol_results]
    colors_bar = ['#4CAF50' if r['chemical_accuracy'] == 'True' else '#F44336'
                  for r in mol_results]
    ax.bar(names, errors, color=colors_bar, edgecolor='black', alpha=0.85)
    ax.axhline(1.6, color='blue', ls='--', linewidth=2, label='Chem. accuracy')
    ax.axhline(20, color='orange', ls=':', label='IBM Quantum')
    ax.set_ylabel('Error (mHa)')
    ax.set_yscale('log')
    ax.set_title('(b) Error Scaling\n(green=pass, red=fail)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    for i, v in enumerate(errors):
        ax.text(i, v * 1.5, '%.1f' % v, ha='center', fontweight='bold', fontsize=10)

    # (c) Hilbert space dimension vs time
    ax = axes[2]
    dims = [r['hilbert_dim'] for r in mol_results]
    times = [r['time_s'] for r in mol_results]
    ax.plot(dims, times, 'o-', color='#2196F3', markersize=10, linewidth=2)
    ax.set_xlabel('Hilbert space dimension')
    ax.set_ylabel('Compute time (s)')
    ax.set_title('(c) Scaling: Time vs Hilbert dim')
    for i, (d, t) in enumerate(zip(dims, times)):
        ax.annotate(names[i], (d, t), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle('Q125: Multi-Molecule VQE (mean=%.1f mHa)' % mean_error,
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q125_multi_vqe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ125 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
