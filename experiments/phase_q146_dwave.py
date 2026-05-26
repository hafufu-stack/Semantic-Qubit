# -*- coding: utf-8 -*-
"""
Phase Q146: The D-Wave Killer (Dense Spin Glass)
==================================================
D-Wave's quantum annealer (~$20M) fails on densely connected
problems because physical wiring is 2D (Pegasus graph).
SK model (Sherrington-Kirkpatrick) is ALL-TO-ALL random couplings.

Q145 showed LLM wins on SYK (all-to-all). Test if this extends
to the optimization version: finding ground states of SK spin glasses.
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


def build_sk_hamiltonian(n_spins, seed=42):
    """Sherrington-Kirkpatrick model: H = -sum_{i<j} J_ij Z_i Z_j
    All-to-all random couplings. D-Wave's nightmare.
    """
    np.random.seed(seed)
    n_qubits = n_spins
    dim = 2 ** n_qubits

    Z = np.array([[1, 0], [0, -1]])
    I2 = np.eye(2)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    H = np.zeros((dim, dim))
    J_matrix = np.random.randn(n_spins, n_spins) / np.sqrt(n_spins)
    J_matrix = (J_matrix + J_matrix.T) / 2
    np.fill_diagonal(J_matrix, 0)

    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            ops = [I2] * n_qubits
            ops[i] = Z; ops[j] = Z
            H += -J_matrix[i, j] * kron_chain(ops)

    # Transverse field (makes it quantum)
    X = np.array([[0, 1], [1, 0]])
    for i in range(n_spins):
        ops = [I2] * n_qubits
        ops[i] = X
        H += -0.3 * kron_chain(ops)

    return H, J_matrix


def build_sparse_ising(n_spins, seed=42):
    """Nearest-neighbor Ising (what D-Wave CAN solve easily)."""
    np.random.seed(seed)
    n_qubits = n_spins
    dim = 2 ** n_qubits

    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    H = np.zeros((dim, dim))
    for i in range(n_spins - 1):
        J = np.random.randn() / np.sqrt(n_spins)
        ops = [I2] * n_qubits; ops[i] = Z; ops[i+1] = Z
        H += -J * kron_chain(ops)

    for i in range(n_spins):
        ops = [I2] * n_qubits; ops[i] = X
        H += -0.3 * kron_chain(ops)

    return H


def rayleigh_gd(H, psi_init, max_steps=3000):
    psi = psi_init.copy() / np.linalg.norm(psi_init)
    E_exact = float(np.linalg.eigvalsh(H)[0])
    lr = 0.01
    for step in range(max_steps):
        E_cur = float(np.real(psi @ H @ psi))
        grad = 2 * (H @ psi - E_cur * psi)
        psi_trial = psi - lr * grad
        psi_trial /= np.linalg.norm(psi_trial)
        E_trial = float(np.real(psi_trial @ H @ psi_trial))
        if not np.isnan(E_trial) and E_trial < E_cur:
            psi = psi_trial
        else:
            lr *= 0.999
    return psi


def main():
    print("=" * 60)
    print("Phase Q146: The D-Wave Killer")
    print("  (Dense SK Spin Glass)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    configs = [
        (6, 'N=6 (toy)'),
        (8, 'N=8 (D-Wave easy)'),
        (10, 'N=10 (D-Wave limit)'),
    ]

    prompt = "Spin glass ground state optimization:"
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_random = 5
    all_results = []

    for n_spins, desc in configs:
        dim = 2 ** n_spins
        if dim > hidden_size:
            continue
        print("\n--- %s (dim=%d) ---" % (desc, dim))

        # Build both dense (SK) and sparse (Ising)
        H_sk, J_mat = build_sk_hamiltonian(n_spins)
        H_ising = build_sparse_ising(n_spins)

        for h_name, H in [('SK (dense)', H_sk), ('Ising (sparse)', H_ising)]:
            E_exact = float(np.linalg.eigvalsh(H)[0])
            psi_exact = np.linalg.eigh(H)[1][:, 0]

            # Random baseline
            rand_errors = []
            for _ in range(n_random):
                psi_r = np.random.randn(dim)
                psi_r /= np.linalg.norm(psi_r)
                psi_f = rayleigh_gd(H, psi_r)
                err = abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000
                rand_errors.append(err)

            # LLM basis
            llm_basis = []
            for li in range(0, n_layers, 4):
                h_np = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
                for offset in range(0, min(hidden_size, dim * 4), dim):
                    if offset + dim <= hidden_size:
                        psi = h_np[offset:offset + dim].copy()
                        norm = np.linalg.norm(psi)
                        if norm > 1e-8:
                            llm_basis.append(psi / norm)

            if not llm_basis:
                llm_basis = [np.random.randn(dim)]
                llm_basis[0] /= np.linalg.norm(llm_basis[0])

            scored = [(float(np.real(p @ H @ p)), p) for p in llm_basis]
            scored.sort(key=lambda x: x[0])
            best = scored[0][1].copy()

            # Pairwise
            top_k = min(10, len(scored))
            best_E = scored[0][0]
            for i in range(top_k):
                for j in range(i+1, top_k):
                    mix = 0.5 * scored[i][1] + 0.5 * scored[j][1]
                    n = np.linalg.norm(mix)
                    if n > 1e-8:
                        mix /= n
                        Em = float(np.real(mix @ H @ mix))
                        if Em < best_E:
                            best_E = Em; best = mix.copy()

            psi_llm = rayleigh_gd(H, best)
            llm_err = abs(float(np.real(psi_llm @ H @ psi_llm)) - E_exact) * 1000
            llm_fid = float(abs(np.dot(psi_llm, psi_exact)) ** 2)

            rand_mean = float(np.mean(rand_errors))
            advantage = rand_mean / max(llm_err, 0.001)

            result = {
                'n_spins': int(n_spins),
                'dim': int(dim),
                'type': h_name,
                'description': desc,
                'exact_energy': round(E_exact, 4),
                'random_error': round(rand_mean, 4),
                'llm_error': round(llm_err, 4),
                'llm_fidelity': round(llm_fid, 4),
                'advantage': round(advantage, 2),
            }
            all_results.append(result)
            print("  %12s: Random=%.4f, LLM=%.4f -> %.1fx" %
                  (h_name, rand_mean, llm_err, advantage))

    # Save
    results = {
        'phase': 'Q146',
        'name': 'D-Wave Killer (SK Spin Glass)',
        'configs': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q146_dwave.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for qi, n_s in enumerate([6, 8, 10]):
        ax = axes[qi]
        subset = [r for r in all_results if r['n_spins'] == n_s]
        if not subset:
            continue
        x = np.arange(len(subset))
        rand_e = [max(r['random_error'], 0.001) for r in subset]
        llm_e = [max(r['llm_error'], 0.001) for r in subset]
        w = 0.35
        ax.bar(x - w/2, rand_e, w, color='#F44336', label='Random', alpha=0.85)
        ax.bar(x + w/2, llm_e, w, color='#4CAF50', label='LLM', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([r['type'] for r in subset], fontsize=7)
        ax.set_ylabel('Error (mHa)')
        ax.set_title('N=%d spins (dim=%d)' % (n_s, 2**n_s))
        ax.set_yscale('log')
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q146: D-Wave Killer (SK dense vs Ising sparse)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q146_dwave.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ146 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
