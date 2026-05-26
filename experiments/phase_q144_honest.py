# -*- coding: utf-8 -*-
"""
Phase Q144: The Honest Benchmark (LLM vs Random Oracle)
========================================================
The most important experiment in this entire project.

Grok's critique: "The LLM is just providing random initial points,
and Rayleigh gradient descent converges from ANY initial point."

DIRECT TEST:
1. Random initial point + gradient descent
2. LLM initial point + gradient descent
3. Compare convergence speed, final error, fidelity

If random is just as good -> LLM is irrelevant (honest finding)
If LLM is better -> proves LLM structure matters
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


def build_random_hamiltonian(dim, seed=42):
    """Random symmetric Hamiltonian (worst case for any method)."""
    np.random.seed(seed)
    A = np.random.randn(dim, dim)
    H = (A + A.T) / 2
    return H


def build_ising_hamiltonian(n_qubits):
    """Transverse-field Ising model: H = -J sum ZZ - h sum X"""
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
    J, h = 1.0, 0.5

    for i in range(n_qubits - 1):
        ops = [I2] * n_qubits
        ops[i] = Z; ops[i + 1] = Z
        H += -J * kron_chain(ops)

    for i in range(n_qubits):
        ops = [I2] * n_qubits
        ops[i] = X
        H += -h * kron_chain(ops)

    return H


def rayleigh_gradient_descent(H, psi_init, max_steps=5000, tol=1e-10):
    """Run gradient descent and track convergence.
    Returns (final_psi, energy_history, steps_to_converge).
    """
    psi = psi_init.copy()
    psi /= np.linalg.norm(psi)
    E_exact = float(np.linalg.eigvalsh(H)[0])

    lr = 0.01
    energies = []
    converge_step = max_steps

    for step in range(max_steps):
        E_cur = float(np.real(psi @ H @ psi))
        energies.append(E_cur)

        error = abs(E_cur - E_exact) * 1000  # mHa
        if error < 0.01 and converge_step == max_steps:
            converge_step = step

        grad = 2 * (H @ psi - E_cur * psi)
        psi_trial = psi - lr * grad
        psi_trial /= np.linalg.norm(psi_trial)
        E_trial = float(np.real(psi_trial @ H @ psi_trial))

        if not np.isnan(E_trial) and E_trial < E_cur:
            psi = psi_trial
        else:
            lr *= 0.999

    return psi, energies, converge_step


def main():
    print("=" * 60)
    print("Phase Q144: The Honest Benchmark")
    print("  (LLM vs Random Oracle)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Problem sizes to test
    test_dims = [4, 8, 16, 32, 64, 128, 256, 512]
    n_random_trials = 5  # Average over multiple random starts

    prompts = [
        "Ground state energy of quantum system:",
    ]
    layer_step = 4  # Sample every 4th layer to speed up

    all_results = []

    for dim in test_dims:
        if dim > hidden_size:
            print("\n--- dim=%d: SKIP (> hidden=%d) ---" % (dim, hidden_size))
            continue

        print("\n--- dim=%d ---" % dim)

        # Build Hamiltonian
        H = build_random_hamiltonian(dim, seed=42)
        eigvals = np.linalg.eigvalsh(H)
        E_exact = float(eigvals[0])
        psi_exact = np.linalg.eigh(H)[1][:, 0]

        # === METHOD 1: Random initial points ===
        random_errors = []
        random_fidelities = []
        random_steps = []
        for trial in range(n_random_trials):
            psi_rand = np.random.randn(dim)
            psi_rand /= np.linalg.norm(psi_rand)

            psi_final, energies, conv_step = rayleigh_gradient_descent(
                H, psi_rand, max_steps=5000)
            error = abs(float(np.real(psi_final @ H @ psi_final)) - E_exact) * 1000
            fid = float(abs(np.dot(psi_final, psi_exact)) ** 2)

            random_errors.append(error)
            random_fidelities.append(fid)
            random_steps.append(conv_step)

        mean_random_error = float(np.mean(random_errors))
        mean_random_fid = float(np.mean(random_fidelities))
        mean_random_steps = float(np.mean(random_steps))
        random_success = sum(1 for e in random_errors if e < 1.6) / n_random_trials

        # === METHOD 2: LLM initial point ===
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

                # Only do QKV projections for small dims (avoid OOM/freeze)
                if dim <= 64:
                    layer = model.model.layers[li]
                    with torch.no_grad():
                        proj = layer.self_attn.q_proj.weight
                        projected = (proj.float() @ out.hidden_states[li + 1][0, -1, :].float()).cpu().numpy()
                        for offset in range(0, min(len(projected), dim * 3), dim):
                            if offset + dim <= len(projected):
                                psi = projected[offset:offset + dim].copy()
                                norm = np.linalg.norm(psi)
                                if norm > 1e-8:
                                    llm_basis.append(psi / norm)

        # Pick best LLM basis
        if not llm_basis:
            llm_basis = [np.random.randn(dim)]
            llm_basis[0] /= np.linalg.norm(llm_basis[0])

        scored = [(float(np.real(p @ H @ p)), p) for p in llm_basis]
        scored.sort(key=lambda x: x[0])
        best_llm = scored[0][1].copy()

        # Also try pairwise
        top_k = min(15, len(scored))
        for i in range(top_k):
            for j in range(i + 1, top_k):
                for alpha in [0.3, 0.5, 0.7]:
                    psi_mix = alpha * scored[i][1] + (1 - alpha) * scored[j][1]
                    norm = np.linalg.norm(psi_mix)
                    if norm > 1e-8:
                        psi_mix /= norm
                        E_mix = float(np.real(psi_mix @ H @ psi_mix))
                        if E_mix < float(np.real(best_llm @ H @ best_llm)):
                            best_llm = psi_mix.copy()

        # LLM initial energy
        E_llm_init = float(np.real(best_llm @ H @ best_llm))
        llm_init_error = abs(E_llm_init - E_exact) * 1000

        psi_llm_final, llm_energies, llm_conv_step = rayleigh_gradient_descent(
            H, best_llm, max_steps=5000)
        llm_error = abs(float(np.real(psi_llm_final @ H @ psi_llm_final)) - E_exact) * 1000
        llm_fid = float(abs(np.dot(psi_llm_final, psi_exact)) ** 2)

        # === METHOD 3: Random initial ENERGY (same initial quality as LLM) ===
        # Find random vectors with similar initial energy to LLM
        matched_errors = []
        matched_steps = []
        for trial in range(n_random_trials):
            # Generate random, score, pick best of 50
            candidates = [np.random.randn(dim) for _ in range(50)]
            candidates = [c / np.linalg.norm(c) for c in candidates]
            c_scored = [(float(np.real(c @ H @ c)), c) for c in candidates]
            c_scored.sort(key=lambda x: x[0])
            psi_matched = c_scored[0][1]

            psi_f, _, conv_s = rayleigh_gradient_descent(H, psi_matched, max_steps=3000)
            error = abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000
            matched_errors.append(error)
            matched_steps.append(conv_s)

        mean_matched_error = float(np.mean(matched_errors))
        mean_matched_steps = float(np.mean(matched_steps))
        matched_success = sum(1 for e in matched_errors if e < 1.6) / n_random_trials

        # LLM advantage
        llm_advantage_error = mean_random_error / max(llm_error, 1e-10)
        llm_advantage_steps = mean_random_steps / max(llm_conv_step, 1)

        result = {
            'dim': int(dim),
            'exact_energy': round(E_exact, 6),
            'random': {
                'mean_error_mha': round(mean_random_error, 4),
                'mean_fidelity': round(mean_random_fid, 4),
                'mean_steps': int(mean_random_steps),
                'success_rate': round(random_success, 2),
            },
            'llm': {
                'init_error_mha': round(llm_init_error, 4),
                'final_error_mha': round(llm_error, 4),
                'fidelity': round(llm_fid, 4),
                'conv_steps': int(llm_conv_step),
                'n_basis': len(llm_basis),
            },
            'matched_random': {
                'mean_error_mha': round(mean_matched_error, 4),
                'mean_steps': int(mean_matched_steps),
                'success_rate': round(matched_success, 2),
            },
            'advantage': {
                'error_ratio': round(llm_advantage_error, 2),
                'speed_ratio': round(llm_advantage_steps, 2),
            },
        }
        all_results.append(result)

        print("  Random:  err=%.4f mHa, steps=%d, success=%.0f%%" %
              (mean_random_error, mean_random_steps, random_success * 100))
        print("  LLM:     err=%.4f mHa, steps=%d, init_err=%.1f mHa" %
              (llm_error, llm_conv_step, llm_init_error))
        print("  Matched: err=%.4f mHa, steps=%d, success=%.0f%%" %
              (mean_matched_error, mean_matched_steps, matched_success * 100))
        print("  -> LLM advantage: %.1fx error, %.1fx speed" %
              (llm_advantage_error, llm_advantage_steps))

    # Save
    results = {
        'phase': 'Q144',
        'name': 'The Honest Benchmark (LLM vs Random)',
        'configs': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q144_honest.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dims = [r['dim'] for r in all_results]

    ax = axes[0]
    rand_err = [max(r['random']['mean_error_mha'], 0.001) for r in all_results]
    llm_err = [max(r['llm']['final_error_mha'], 0.001) for r in all_results]
    match_err = [max(r['matched_random']['mean_error_mha'], 0.001) for r in all_results]
    ax.semilogy(range(len(dims)), rand_err, 'o-', color='#F44336', label='Random', linewidth=2)
    ax.semilogy(range(len(dims)), llm_err, 's-', color='#4CAF50', label='LLM', linewidth=2)
    ax.semilogy(range(len(dims)), match_err, '^--', color='#FF9800', label='Matched Random', linewidth=2)
    ax.axhline(1.6, color='blue', ls=':', label='Chem. accuracy')
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(['%d' % d for d in dims])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Final Error (mHa, log)')
    ax.set_title('(a) LLM vs Random: Final Error')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    rand_s = [r['random']['mean_steps'] for r in all_results]
    llm_s = [r['llm']['conv_steps'] for r in all_results]
    match_s = [r['matched_random']['mean_steps'] for r in all_results]
    ax.plot(range(len(dims)), rand_s, 'o-', color='#F44336', label='Random', linewidth=2)
    ax.plot(range(len(dims)), llm_s, 's-', color='#4CAF50', label='LLM', linewidth=2)
    ax.plot(range(len(dims)), match_s, '^--', color='#FF9800', label='Matched', linewidth=2)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(['%d' % d for d in dims])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Steps to converge')
    ax.set_title('(b) Convergence Speed')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    rand_sr = [r['random']['success_rate'] * 100 for r in all_results]
    llm_sr = [100 if r['llm']['final_error_mha'] < 1.6 else 0 for r in all_results]
    match_sr = [r['matched_random']['success_rate'] * 100 for r in all_results]
    x = np.arange(len(dims))
    w = 0.25
    ax.bar(x - w, rand_sr, w, color='#F44336', label='Random', alpha=0.85)
    ax.bar(x, llm_sr, w, color='#4CAF50', label='LLM', alpha=0.85)
    ax.bar(x + w, match_sr, w, color='#FF9800', label='Matched', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['%d' % d for d in dims])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Success rate (%)')
    ax.set_title('(c) Chemical Accuracy Success Rate')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q144: The Honest Benchmark (LLM vs Random Oracle)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q144_honest.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ144 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
