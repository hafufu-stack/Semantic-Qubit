# -*- coding: utf-8 -*-
"""
Phase Q145: Cross-Problem Universality
========================================
Does the LLM advantage persist across DIFFERENT types of Hamiltonians?
Or is it problem-specific?

Test the SAME LLM features on:
1. Random symmetric (hardest, no structure)
2. Ising model (physics structure)
3. Heisenberg model (SU(2) symmetry)
4. Hubbard-like (fermionic)
5. SYK-like (all-to-all, chaotic)

If LLM works equally well on ALL -> universal quantum structure
If LLM only works on some -> problem-specific bias
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


def build_hamiltonians(n_qubits):
    """Build 5 different types of Hamiltonians with the same number of qubits."""
    dim = 2 ** n_qubits
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    I2 = np.eye(2)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    hamiltonians = {}

    # 1. Random symmetric
    np.random.seed(42)
    A = np.random.randn(dim, dim)
    hamiltonians['Random'] = np.real((A + A.T) / 2)

    # 2. Ising: H = -J sum ZZ - h sum X
    H_ising = np.zeros((dim, dim))
    for i in range(n_qubits - 1):
        ops = [I2] * n_qubits; ops[i] = Z; ops[i+1] = Z
        H_ising += -1.0 * kron_chain(ops)
    for i in range(n_qubits):
        ops = [I2] * n_qubits; ops[i] = X
        H_ising += -0.5 * kron_chain(ops)
    hamiltonians['Ising'] = np.real(H_ising)

    # 3. Heisenberg: H = J sum (XX + YY + ZZ)
    H_heis = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        for pauli in [X, Y, Z]:
            ops = [I2] * n_qubits; ops[i] = pauli; ops[i+1] = pauli
            H_heis += 1.0 * kron_chain(ops)
    hamiltonians['Heisenberg'] = np.real(H_heis)

    # 4. Hubbard-like: hopping + on-site
    H_hub = np.zeros((dim, dim), dtype=complex)
    n_sites = n_qubits // 2
    for spin_off in [0, n_sites]:
        for i in range(n_sites - 1):
            qi, qj = i + spin_off, i + 1 + spin_off
            if qj < n_qubits:
                ops_x = [I2] * n_qubits; ops_x[qi] = X; ops_x[qj] = X
                H_hub += -0.5 * kron_chain(ops_x)
                ops_y = [I2] * n_qubits; ops_y[qi] = Y; ops_y[qj] = Y
                H_hub += -0.5 * kron_chain(ops_y)
    for i in range(n_sites):
        q_up, q_dn = i, i + n_sites
        if q_dn < n_qubits:
            ops = [I2] * n_qubits; ops[q_up] = Z; ops[q_dn] = Z
            H_hub += 4.0 * 0.25 * kron_chain(ops)
    hamiltonians['Hubbard'] = np.real(H_hub)

    # 5. SYK-like: random all-to-all
    np.random.seed(123)
    H_syk = np.zeros((dim, dim), dtype=complex)
    paulis = [X, Y, Z]
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for pi in paulis:
                for pj in paulis:
                    J = np.random.randn() * 0.3
                    ops = [I2] * n_qubits; ops[i] = pi; ops[j] = pj
                    H_syk += J * kron_chain(ops)
    hamiltonians['SYK-like'] = np.real((H_syk + H_syk.conj().T) / 2)

    return hamiltonians


def rayleigh_gd(H, psi_init, max_steps=3000):
    """Rayleigh gradient descent, returns (final_psi, converge_step)."""
    psi = psi_init.copy()
    psi /= np.linalg.norm(psi)
    E_exact = float(np.linalg.eigvalsh(H)[0])
    lr = 0.01
    best_E = float(np.real(psi @ H @ psi))
    converge_step = max_steps

    for step in range(max_steps):
        E_cur = float(np.real(psi @ H @ psi))
        if abs(E_cur - E_exact) * 1000 < 0.01 and converge_step == max_steps:
            converge_step = step

        grad = 2 * (H @ psi - E_cur * psi)
        psi_trial = psi - lr * grad
        psi_trial /= np.linalg.norm(psi_trial)
        E_trial = float(np.real(psi_trial @ H @ psi_trial))
        if not np.isnan(E_trial) and E_trial < E_cur:
            psi = psi_trial
        else:
            lr *= 0.999

    return psi, converge_step


def main():
    print("=" * 60)
    print("Phase Q145: Cross-Problem Universality")
    print("  (Does LLM work on ALL problem types?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    qubit_configs = [6, 8, 10]  # dims: 64, 256, 1024
    n_random_trials = 5
    layer_step = 4

    prompts = [
        "Ground state energy of quantum system:",
    ]

    all_results = []

    for n_q in qubit_configs:
        dim = 2 ** n_q
        if dim > hidden_size:
            continue

        print("\n=== %d qubits (dim=%d) ===" % (n_q, dim))
        hamiltonians = build_hamiltonians(n_q)

        # Collect LLM basis (same for all problem types!)
        llm_basis = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            for li in range(0, n_layers, layer_step):
                h_np = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
                for offset in range(0, min(hidden_size, dim * 4), dim):
                    if offset + dim <= hidden_size:
                        psi = h_np[offset:offset + dim].copy()
                        norm = np.linalg.norm(psi)
                        if norm > 1e-8:
                            llm_basis.append(psi / norm)

        print("  LLM basis vectors: %d" % len(llm_basis))

        for h_name, H in hamiltonians.items():
            E_exact = float(np.linalg.eigvalsh(H)[0])
            psi_exact = np.linalg.eigh(H)[1][:, 0]

            # Random baseline
            rand_errors = []
            rand_fids = []
            for _ in range(n_random_trials):
                psi_r = np.random.randn(dim)
                psi_r /= np.linalg.norm(psi_r)
                psi_f, _ = rayleigh_gd(H, psi_r)
                err = abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000
                fid = float(abs(np.dot(psi_f, psi_exact)) ** 2)
                rand_errors.append(err)
                rand_fids.append(fid)

            # LLM: pick best + pairwise + gradient
            scored = [(float(np.real(p @ H @ p)), p) for p in llm_basis]
            scored.sort(key=lambda x: x[0])

            best_psi = scored[0][1].copy()
            best_E = scored[0][0]

            top_k = min(10, len(scored))
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    psi_mix = 0.5 * scored[i][1] + 0.5 * scored[j][1]
                    norm = np.linalg.norm(psi_mix)
                    if norm > 1e-8:
                        psi_mix /= norm
                        E_mix = float(np.real(psi_mix @ H @ psi_mix))
                        if E_mix < best_E:
                            best_E = E_mix
                            best_psi = psi_mix.copy()

            psi_llm_f, llm_steps = rayleigh_gd(H, best_psi)
            llm_error = abs(float(np.real(psi_llm_f @ H @ psi_llm_f)) - E_exact) * 1000
            llm_fid = float(abs(np.dot(psi_llm_f, psi_exact)) ** 2)

            result = {
                'n_qubits': int(n_q),
                'dim': int(dim),
                'hamiltonian': h_name,
                'exact_energy': round(E_exact, 4),
                'random_error': round(float(np.mean(rand_errors)), 4),
                'random_fidelity': round(float(np.mean(rand_fids)), 4),
                'random_success': round(sum(1 for e in rand_errors if e < 1.6) / n_random_trials, 2),
                'llm_error': round(llm_error, 4),
                'llm_fidelity': round(llm_fid, 4),
                'llm_advantage': round(float(np.mean(rand_errors)) / max(llm_error, 0.001), 2),
            }
            all_results.append(result)

            advantage = float(np.mean(rand_errors)) / max(llm_error, 0.001)
            print("  %12s: Random=%.4f mHa, LLM=%.4f mHa -> %.1fx advantage" %
                  (h_name, np.mean(rand_errors), llm_error, advantage))

    # Save
    results = {
        'phase': 'Q145',
        'name': 'Cross-Problem Universality',
        'configs': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q145_universality.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    h_names = ['Random', 'Ising', 'Heisenberg', 'Hubbard', 'SYK-like']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for qi, n_q in enumerate(qubit_configs):
        dim = 2 ** n_q
        if dim > hidden_size:
            continue
        ax = axes[qi]
        subset = [r for r in all_results if r['n_qubits'] == n_q]
        if not subset:
            continue

        x = np.arange(len(subset))
        rand_e = [max(r['random_error'], 0.001) for r in subset]
        llm_e = [max(r['llm_error'], 0.001) for r in subset]
        names = [r['hamiltonian'] for r in subset]

        w = 0.35
        ax.bar(x - w/2, rand_e, w, color='#F44336', label='Random', alpha=0.85, edgecolor='black')
        ax.bar(x + w/2, llm_e, w, color='#4CAF50', label='LLM', alpha=0.85, edgecolor='black')
        ax.axhline(1.6, color='blue', ls='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('Error (mHa, log)')
        ax.set_title('%d qubits (dim=%d)' % (n_q, dim))
        ax.set_yscale('log')
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q145: Cross-Problem Universality (LLM vs Random)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q145_universality.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ145 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
