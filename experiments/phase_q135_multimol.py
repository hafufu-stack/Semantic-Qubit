# -*- coding: utf-8 -*-
"""
Phase Q135: Multi-Molecule Chemical Accuracy (Gradient Descent)
================================================================
Q134 proved LiH (4 qubits) can reach EXACT solution (0.00 mHa).
Key: Rayleigh quotient gradient descent from LLM initial point.

Critical question: Does LLM provide good enough starting points
for BeH2 (6q, 64-dim) and H2O (8q, 256-dim)?
For Rayleigh quotient, ALL critical points are eigenstates.
The LLM starting point determines WHICH eigenstate we converge to.
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


def build_hamiltonian(molecule, n_qubits):
    """Build molecular Hamiltonian."""
    dim = 2 ** n_qubits
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def Z_i(i, n):
        ops = [I2] * n; ops[i] = Z; return kron_chain(ops)

    def ZZ_ij(i, j, n):
        ops = [I2] * n; ops[i] = Z; ops[j] = Z; return kron_chain(ops)

    def XX_ij(i, j, n):
        ops = [I2] * n; ops[i] = X; ops[j] = X; return kron_chain(ops)

    H = np.zeros((dim, dim))
    if molecule == 'H2':
        n = 2
        H += -0.4804 * np.eye(4)
        H += 0.3435 * Z_i(0, n) - 0.4347 * Z_i(1, n)
        H += 0.5716 * ZZ_ij(0, 1, n) + 0.0910 * XX_ij(0, 1, n)
    elif molecule == 'LiH':
        n = 4
        H += -0.22 * np.eye(dim)
        H += 0.17 * Z_i(0, n) + 0.12 * Z_i(1, n)
        H += -0.17 * Z_i(2, n) + 0.17 * Z_i(3, n)
        H += 0.12 * ZZ_ij(0, 1, n) + 0.04 * ZZ_ij(0, 2, n)
        H += 0.17 * ZZ_ij(1, 2, n) + 0.04 * ZZ_ij(2, 3, n)
        H += 0.04 * XX_ij(0, 1, n) + 0.04 * XX_ij(2, 3, n)
    elif molecule == 'BeH2':
        n = 6
        H += -0.15 * np.eye(dim)
        for i in range(n):
            H += (0.10 - 0.03 * i) * Z_i(i, n)
        for i in range(n - 1):
            H += (0.08 - 0.02 * i) * ZZ_ij(i, i + 1, n)
            H += 0.03 * XX_ij(i, i + 1, n)
    elif molecule == 'H2O':
        n = 8
        H += -0.32 * np.eye(dim)
        for i in range(n):
            H += (0.15 - 0.04 * i) * Z_i(i, n)
        for i in range(n - 1):
            H += (0.10 - 0.02 * i) * ZZ_ij(i, i + 1, n)
            H += 0.02 * XX_ij(i, i + 1, n)
        for i in range(n - 2):
            H += 0.01 * ZZ_ij(i, i + 2, n)
    return H


def solve_molecule(model, tok, device, molecule, n_qubits):
    """Full pipeline: LLM basis + gradient descent."""
    dim = 2 ** n_qubits
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    H = build_hamiltonian(molecule, n_qubits)
    E_exact = float(np.linalg.eigvalsh(H)[0])
    psi_exact = np.linalg.eigh(H)[1][:, 0]

    prompts = [
        "Ground state of %s molecule:" % molecule,
        "Quantum chemistry %s energy:" % molecule,
        "%s molecular orbital:" % molecule,
        "Variational ansatz for %s:" % molecule,
    ]

    # Collect basis vectors
    all_basis = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(n_layers):
            layer = model.model.layers[li]
            with torch.no_grad():
                q_w = layer.self_attn.q_proj.weight.float()
                k_w = layer.self_attn.k_proj.weight.float()
                v_w = layer.self_attn.v_proj.weight.float()

            h_all = out.hidden_states[li + 1][0].float()
            for ti in range(h_all.shape[0]):
                h_t = h_all[ti]
                h_np = h_t.cpu().numpy()
                for offset in range(0, min(hidden, dim * 6), dim):
                    if offset + dim <= hidden:
                        psi = h_np[offset:offset + dim].copy()
                        norm = np.linalg.norm(psi)
                        if norm > 1e-8:
                            all_basis.append(psi / norm)

                for proj in [q_w, k_w, v_w]:
                    projected = (proj @ h_t).cpu().numpy()
                    for offset in range(0, min(len(projected), dim * 4), dim):
                        if offset + dim <= len(projected):
                            psi = projected[offset:offset + dim].copy()
                            norm = np.linalg.norm(psi)
                            if norm > 1e-8:
                                all_basis.append(psi / norm)

    # Score basis vectors
    scored = []
    for psi in all_basis:
        E = float(np.real(psi @ H @ psi))
        if not np.isnan(E) and not np.isinf(E):
            scored.append((E, psi))
    scored.sort(key=lambda x: x[0])

    best_E = scored[0][0]
    best_psi = scored[0][1].copy()
    single_error = abs(best_E - E_exact) * 1000
    convergence = [('LLM_single', single_error)]

    # Pairwise combinations
    top_k = min(30, len(scored))
    for i in range(top_k):
        for j in range(i + 1, top_k):
            for alpha in np.linspace(0, 1, 21):
                psi_mix = alpha * scored[i][1] + (1 - alpha) * scored[j][1]
                norm = np.linalg.norm(psi_mix)
                if norm > 1e-8:
                    psi_mix /= norm
                    E_mix = float(np.real(psi_mix @ H @ psi_mix))
                    if not np.isnan(E_mix) and E_mix < best_E:
                        best_E = E_mix
                        best_psi = psi_mix.copy()
    pair_error = abs(best_E - E_exact) * 1000
    convergence.append(('LLM_pairwise', pair_error))

    # Triple combinations
    top_k2 = min(10, len(scored))
    for i in range(top_k2):
        for j in range(i + 1, top_k2):
            for k in range(j + 1, top_k2):
                for a1 in np.linspace(0, 1, 7):
                    for a2 in np.linspace(0, 1 - a1, 7):
                        a3 = 1 - a1 - a2
                        psi_mix = a1 * scored[i][1] + a2 * scored[j][1] + a3 * scored[k][1]
                        norm = np.linalg.norm(psi_mix)
                        if norm > 1e-8:
                            psi_mix /= norm
                            E_mix = float(np.real(psi_mix @ H @ psi_mix))
                            if not np.isnan(E_mix) and E_mix < best_E:
                                best_E = E_mix
                                best_psi = psi_mix.copy()
    triple_error = abs(best_E - E_exact) * 1000
    convergence.append(('LLM_triple', triple_error))

    # Rayleigh quotient gradient descent (the key step!)
    lr = 0.01
    for step in range(5000):
        E_cur = float(np.real(best_psi @ H @ best_psi))
        grad = 2 * (H @ best_psi - E_cur * best_psi)
        psi_trial = best_psi - lr * grad
        psi_trial /= np.linalg.norm(psi_trial)
        E_trial = float(np.real(psi_trial @ H @ psi_trial))
        if not np.isnan(E_trial) and E_trial < best_E:
            best_E = E_trial
            best_psi = psi_trial.copy()
        else:
            lr *= 0.999
    grad_error = abs(best_E - E_exact) * 1000
    convergence.append(('gradient', grad_error))

    fidelity = abs(np.dot(best_psi, psi_exact)) ** 2
    chem_acc = grad_error < 1.6

    return {
        'molecule': molecule,
        'n_qubits': n_qubits,
        'dim': dim,
        'exact_energy': round(float(E_exact), 6),
        'sqbit_energy': round(float(best_E), 6),
        'error_mha': round(float(grad_error), 4),
        'chemical_accuracy': str(chem_acc),
        'fidelity': round(float(fidelity), 6),
        'convergence': [(n, round(e, 4)) for n, e in convergence],
        'n_basis': len(all_basis),
    }


def main():
    print("=" * 60)
    print("Phase Q135: Multi-Molecule Chemical Accuracy")
    print("  (LLM initial point + Rayleigh gradient descent)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    molecules = [('H2', 2), ('LiH', 4), ('BeH2', 6), ('H2O', 8)]
    all_results = []

    for mol, nq in molecules:
        print("\n--- %s (%d qubits, %d-dim) ---" % (mol, nq, 2**nq))
        t_mol = time.time()
        result = solve_molecule(model, tok, device, mol, nq)
        result['time_s'] = round(time.time() - t_mol, 2)
        all_results.append(result)

        marker = "PASS" if result['chemical_accuracy'] == 'True' else "FAIL"
        print("  LLM single: %.2f mHa" % result['convergence'][0][1])
        print("  LLM pair:   %.2f mHa" % result['convergence'][1][1])
        print("  LLM triple: %.2f mHa" % result['convergence'][2][1])
        print("  + Gradient: %.4f mHa -> %s (fidelity=%.4f)" %
              (result['error_mha'], marker, result['fidelity']))

    n_pass = sum(1 for r in all_results if r['chemical_accuracy'] == 'True')
    print("\n=== SUMMARY ===")
    print("Chemical accuracy: %d/%d molecules" % (n_pass, len(all_results)))

    # Q125 vs Q129 vs Q135
    q125_errors = {'H2': 0.41, 'LiH': 168.64, 'BeH2': 144.33, 'H2O': 475.85}
    q129_errors = {'H2': 0.00, 'LiH': 2.92, 'BeH2': 61.11, 'H2O': 294.34}
    print("\nEvolution:")
    for r in all_results:
        m = r['molecule']
        print("  %s: Q125=%.1f -> Q129=%.1f -> Q135=%.4f mHa (%.0fx improvement)" %
              (m, q125_errors[m], q129_errors[m],
               r['error_mha'], q125_errors[m] / max(r['error_mha'], 0.0001)))

    # Save
    results = {
        'phase': 'Q135',
        'name': 'Multi-Molecule Chemical Accuracy',
        'molecules': all_results,
        'n_chemical_accuracy': n_pass,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q135_multimol.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = [r['molecule'] for r in all_results]

    # (a) Evolution across versions
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, [q125_errors[n] for n in names], w, label='Q125 (naive)',
           color='#F44336', alpha=0.85)
    ax.bar(x, [q129_errors[n] for n in names], w, label='Q129 (tensor)',
           color='#FF9800', alpha=0.85)
    ax.bar(x + w, [max(r['error_mha'], 0.001) for r in all_results], w,
           label='Q135 (gradient)', color='#4CAF50', alpha=0.85)
    ax.axhline(1.6, color='blue', ls='--', linewidth=2, label='Chem. accuracy')
    ax.axhline(20, color='purple', ls=':', label='IBM Quantum')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('Error (mHa)'); ax.set_yscale('log')
    ax.set_title('(a) Evolution: Q125 -> Q129 -> Q135')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    # (b) Fidelity
    ax = axes[1]
    fids = [r['fidelity'] for r in all_results]
    colors = ['#4CAF50' if r['chemical_accuracy'] == 'True' else '#F44336'
              for r in all_results]
    ax.bar(names, fids, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0.99, color='gold', ls='--', label='99% fidelity')
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) Wavefunction Fidelity')
    ax.set_ylim(0, 1.1)
    for i, f in enumerate(fids):
        ax.text(i, f + 0.02, '%.4f' % f, ha='center', fontsize=9, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (c) Convergence paths
    ax = axes[2]
    for r in all_results:
        steps = [c[0] for c in r['convergence']]
        errors = [max(c[1], 0.001) for c in r['convergence']]
        ax.semilogy(range(len(steps)), errors, 'o-', label=r['molecule'], markersize=6)
    ax.axhline(1.6, color='blue', ls='--', alpha=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['LLM\nsingle', 'LLM\npairwise', 'LLM\ntriple', 'Rayleigh\ngradient'],
                        fontsize=7)
    ax.set_ylabel('Error (mHa, log)')
    ax.set_title('(c) Convergence Paths')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q135: Multi-Molecule VQE (%d/%d chem. accuracy)' %
                 (n_pass, len(all_results)), fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q135_multimol.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ135 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
