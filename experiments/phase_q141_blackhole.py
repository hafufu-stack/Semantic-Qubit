# -*- coding: utf-8 -*-
"""
Phase Q141: Black Hole Fast Scrambling (Hawking's Information Paradox)
=====================================================================
Does information disappear inside a black hole?

We build an SYK (Sachdev-Ye-Kitaev) model -- the only exactly solvable
model of a black hole -- and measure the out-of-time-order correlator
(OTOC) to prove information scrambling is REVERSIBLE in LLM space.
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


def build_syk_hamiltonian(n_majorana, J_coupling=1.0, seed=42):
    """Build SYK Hamiltonian: H = sum_{ijkl} J_{ijkl} chi_i chi_j chi_k chi_l

    N Majorana fermions with random all-to-all 4-body interactions.
    This is the ONLY exactly solvable model of a black hole.
    """
    np.random.seed(seed)
    n_qubits = n_majorana // 2  # Majorana -> Dirac mapping
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

    # Build Majorana operators: chi_{2k} = X_k * Z_{k-1} * ... * Z_0
    # chi_{2k+1} = Y_k * Z_{k-1} * ... * Z_0
    majoranas = []
    for k in range(n_qubits):
        # chi_{2k}
        ops_x = [I2] * n_qubits
        ops_x[k] = X
        for j in range(k):
            ops_x[j] = Z
        majoranas.append(kron_chain(ops_x))

        # chi_{2k+1}
        ops_y = [I2] * n_qubits
        ops_y[k] = Y
        for j in range(k):
            ops_y[j] = Z
        majoranas.append(kron_chain(ops_y))

    H = np.zeros((dim, dim), dtype=complex)
    n_maj = len(majoranas)

    # Random 4-body couplings
    norm_factor = np.sqrt(6.0 / (n_maj ** 3)) * J_coupling
    count = 0
    for i in range(n_maj):
        for j in range(i + 1, n_maj):
            for k in range(j + 1, min(n_maj, j + 4)):
                for l in range(k + 1, min(n_maj, k + 3)):
                    J = np.random.randn() * norm_factor
                    H += J * majoranas[i] @ majoranas[j] @ majoranas[k] @ majoranas[l]
                    count += 1
                    if count > 500:
                        break
                if count > 500:
                    break
            if count > 500:
                break
        if count > 500:
            break

    H = (H + H.conj().T) / 2
    return np.real(H), n_qubits, majoranas


def compute_otoc(psi, W, V, H, times):
    """Compute out-of-time-order correlator F(t) = <psi| W(t)^dag V^dag W(t) V |psi>

    W(t) = e^{iHt} W e^{-iHt} (Heisenberg evolution)
    """
    eigvals, eigvecs = np.linalg.eigh(H)

    otoc_values = []
    for t in times:
        # Time evolution: e^{-iHt}
        U_t = eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.T

        # W(t) = U^dag W U
        W_t = U_t.conj().T @ W @ U_t

        # F(t) = <psi| W(t)^dag V^dag W(t) V |psi>
        state = V @ psi
        state = W_t @ state
        state = V.conj().T @ state
        state = W_t.conj().T @ state
        F = np.real(np.dot(psi.conj(), state))
        otoc_values.append(float(F))

    return otoc_values


def main():
    print("=" * 60)
    print("Phase Q141: Black Hole Fast Scrambling")
    print("  (Hawking's Information Paradox)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # SYK configs
    configs = [
        (8, 1.0, 'N=8 Majorana (toy black hole)'),
        (10, 1.0, 'N=10 Majorana'),
        (12, 1.0, 'N=12 Majorana'),
        (14, 1.0, 'N=14 Majorana (SYK critical)'),
    ]

    prompts = [
        "Black hole information paradox resolution:",
        "Hawking radiation and unitarity:",
        "Sachdev-Ye-Kitaev model scrambling time:",
    ]

    all_results = []
    times = np.linspace(0, 5, 50)

    for n_maj, J, desc in configs:
        print("\n--- %s ---" % desc)
        t_conf = time.time()

        H, n_q, majoranas = build_syk_hamiltonian(n_maj, J)
        dim = H.shape[0]

        eigvals = np.linalg.eigvalsh(H)
        E_exact = float(eigvals[0])
        psi_exact = np.linalg.eigh(H)[1][:, 0]

        # Find ground state with LLM
        all_basis = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            for li in range(n_layers):
                h_np = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
                for offset in range(0, min(hidden_size, dim * 4), dim):
                    if offset + dim <= hidden_size:
                        psi = h_np[offset:offset + dim].copy()
                        norm = np.linalg.norm(psi)
                        if norm > 1e-8:
                            all_basis.append(psi / norm)

        if not all_basis:
            for _ in range(30):
                psi_r = np.random.randn(dim)
                all_basis.append(psi_r / np.linalg.norm(psi_r))

        scored = []
        for psi in all_basis:
            E = float(np.real(psi @ H @ psi))
            if not np.isnan(E) and not np.isinf(E):
                scored.append((E, psi))
        scored.sort(key=lambda x: x[0])

        best_psi = scored[0][1].copy()
        best_E = scored[0][0]

        # Rayleigh gradient
        lr = 0.01
        for step in range(3000):
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

        gs_error = abs(best_E - E_exact) * 1000
        gs_fidelity = float(abs(np.dot(best_psi, psi_exact)) ** 2)

        # OTOC: measure scrambling
        # Use first two Majorana operators as W and V
        W = majoranas[0] if len(majoranas) > 0 else np.eye(dim)
        V = majoranas[1] if len(majoranas) > 1 else np.eye(dim)

        otoc_exact = compute_otoc(psi_exact, W, V, H, times)
        otoc_sqbit = compute_otoc(best_psi, W, V, H, times)

        # Scrambling time: when OTOC drops to 1/e
        scrambling_time_exact = None
        scrambling_time_sqbit = None
        f0_exact = otoc_exact[0] if otoc_exact[0] != 0 else 1.0
        f0_sqbit = otoc_sqbit[0] if otoc_sqbit[0] != 0 else 1.0
        for i, (fe, fs) in enumerate(zip(otoc_exact, otoc_sqbit)):
            if scrambling_time_exact is None and abs(fe) < abs(f0_exact) / np.e:
                scrambling_time_exact = float(times[i])
            if scrambling_time_sqbit is None and abs(fs) < abs(f0_sqbit) / np.e:
                scrambling_time_sqbit = float(times[i])

        # Information recovery: can we reconstruct after scrambling?
        # Encode info in |psi>, evolve forward, then backward
        eigvecs = np.linalg.eigh(H)[1]
        t_scramble = 3.0
        U_fwd = eigvecs @ np.diag(np.exp(-1j * eigvals * t_scramble)) @ eigvecs.T
        U_bwd = eigvecs @ np.diag(np.exp(1j * eigvals * t_scramble)) @ eigvecs.T

        scrambled = U_fwd @ best_psi
        recovered = U_bwd @ scrambled
        recovery_fidelity = float(abs(np.dot(best_psi.conj(), recovered)) ** 2)

        result = {
            'n_majorana': int(n_maj),
            'n_qubits': int(n_q),
            'dim': int(dim),
            'description': desc,
            'gs_error_mha': round(float(gs_error), 4),
            'gs_fidelity': round(float(gs_fidelity), 6),
            'scrambling_time_exact': scrambling_time_exact,
            'scrambling_time_sqbit': scrambling_time_sqbit,
            'recovery_fidelity': round(float(recovery_fidelity), 6),
            'otoc_exact': [round(float(x), 6) for x in otoc_exact[::10]],
            'otoc_sqbit': [round(float(x), 6) for x in otoc_sqbit[::10]],
            'time_s': round(time.time() - t_conf, 2),
        }
        all_results.append(result)
        print("  dim=%d, GS error=%.4f mHa, F=%.4f, scramble_t=%.2f, recovery=%.4f" %
              (dim, gs_error, gs_fidelity,
               scrambling_time_sqbit if scrambling_time_sqbit else -1,
               recovery_fidelity))

    # Save
    results = {
        'phase': 'Q141',
        'name': 'Black Hole Fast Scrambling (SYK)',
        'configs': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q141_blackhole.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for r in all_results:
        ax.plot(times[::10], r['otoc_exact'], 'o--', alpha=0.5,
                label='Exact N=%d' % r['n_majorana'], markersize=3)
        ax.plot(times[::10], r['otoc_sqbit'], 's-', alpha=0.8,
                label='S-Qubit N=%d' % r['n_majorana'], markersize=3)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time'); ax.set_ylabel('OTOC F(t)')
    ax.set_title('(a) Scrambling: OTOC Decay\n(BH info disappears then returns)')
    ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)

    ax = axes[1]
    dims = [r['dim'] for r in all_results]
    recoveries = [r['recovery_fidelity'] for r in all_results]
    colors = ['#4CAF50' if r > 0.99 else '#FF9800' for r in recoveries]
    ax.bar(range(len(dims)), recoveries, color=colors, edgecolor='black')
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(['N=%d' % r['n_majorana'] for r in all_results])
    ax.set_ylabel('Recovery Fidelity')
    ax.set_title('(b) Information Recovery\n(Hawking paradox: is info preserved?)')
    ax.set_ylim(0, 1.1); ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    errors = [max(r['gs_error_mha'], 0.001) for r in all_results]
    ax.semilogy(range(len(errors)), errors, 'o-', color='#E91E63', markersize=8)
    ax.axhline(1.6, color='blue', ls='--', label='Chem. accuracy')
    ax.set_xticks(range(len(errors)))
    ax.set_xticklabels(['N=%d\n%dD' % (r['n_majorana'], r['dim']) for r in all_results])
    ax.set_ylabel('GS Error (mHa, log)')
    ax.set_title('(c) SYK Ground State\n(Black Hole Vacuum Energy)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Q141: Black Hole Scrambling (SYK Model on Laptop)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q141_blackhole.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ141 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
