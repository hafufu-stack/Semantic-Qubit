# -*- coding: utf-8 -*-
"""
Phase Q158: Quantum Advantage Topology Boundary
=================================================
Q145/Q146 showed: all-to-all -> LLM wins, local -> tie.
WHERE is the exact transition?

Systematically vary interaction range from local (k=1) to
all-to-all (k=N) and find the critical point.
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


def build_variable_range_hamiltonian(n_qubits, k_range, seed=42):
    """Build Hamiltonian with variable interaction range k.
    k=1: nearest neighbor (Ising-like)
    k=n_qubits: all-to-all (SYK-like)
    """
    np.random.seed(seed)
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
    n_connections = 0
    for i in range(n_qubits):
        for j in range(i + 1, min(i + k_range + 1, n_qubits)):
            J = np.random.randn() / np.sqrt(n_qubits)
            ops = [I2] * n_qubits; ops[i] = Z; ops[j] = Z
            H += -J * kron_chain(ops)
            n_connections += 1

    # Transverse field
    for i in range(n_qubits):
        ops = [I2] * n_qubits; ops[i] = X
        H += -0.3 * kron_chain(ops)

    return H, n_connections


def rayleigh_gd(H, psi_init, max_steps=2000):
    psi = psi_init.copy() / np.linalg.norm(psi_init)
    lr = 0.01
    for step in range(max_steps):
        E = float(np.real(psi @ H @ psi))
        grad = 2 * (H @ psi - E * psi)
        psi_t = psi - lr * grad
        psi_t /= np.linalg.norm(psi_t)
        Et = float(np.real(psi_t @ H @ psi_t))
        if not np.isnan(Et) and Et < E:
            psi = psi_t
        else:
            lr *= 0.999
    return psi


def main():
    print("=" * 60)
    print("Phase Q158: Quantum Advantage Topology Boundary")
    print("  (Local -> All-to-All Transition)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    n_qubits = 8
    dim = 2 ** n_qubits

    # Vary k from 1 (local) to n_qubits (all-to-all)
    k_values = [1, 2, 3, 4, 5, 6, 7, 8]

    prompt = "Quantum spin system ground state:"
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Collect LLM basis
    llm_basis = []
    for li in range(0, n_layers, 4):
        h = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
        for offset in range(0, min(hidden_size, dim * 4), dim):
            if offset + dim <= hidden_size:
                psi = h[offset:offset + dim].copy()
                norm = np.linalg.norm(psi)
                if norm > 1e-8:
                    llm_basis.append(psi / norm)

    n_random = 5
    all_results = []

    for k in k_values:
        H, n_conn = build_variable_range_hamiltonian(n_qubits, k)
        E_exact = float(np.linalg.eigvalsh(H)[0])
        connectivity = n_conn / (n_qubits * (n_qubits - 1) / 2) * 100

        # Random baseline
        rand_errors = []
        for _ in range(n_random):
            psi_r = np.random.randn(dim); psi_r /= np.linalg.norm(psi_r)
            psi_f = rayleigh_gd(H, psi_r)
            err = abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000
            rand_errors.append(err)

        # LLM
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

        rand_mean = float(np.mean(rand_errors))
        advantage = rand_mean / max(llm_err, 0.001)

        result = {
            'k': int(k),
            'connectivity_pct': round(connectivity, 1),
            'n_connections': int(n_conn),
            'random_error': round(rand_mean, 4),
            'llm_error': round(llm_err, 4),
            'advantage': round(advantage, 2),
        }
        all_results.append(result)
        topo = "local" if k <= 2 else "medium" if k <= 5 else "global"
        print("  k=%d (%s, %.0f%% connected): Random=%.3f, LLM=%.3f -> %.1fx" %
              (k, topo, connectivity, rand_mean, llm_err, advantage))

    # Find transition point
    advantages = [r['advantage'] for r in all_results]
    # Where does advantage first exceed 1.5?
    transition_k = None
    for r in all_results:
        if r['advantage'] > 1.5 and transition_k is None:
            transition_k = r['k']

    print("\n--- Topology Boundary ---")
    print("  Transition to LLM advantage at k=%s" %
          (str(transition_k) if transition_k else "not found"))
    print("  Max advantage: %.1fx at k=%d" %
          (max(advantages), all_results[np.argmax(advantages)]['k']))

    # Save
    results = {
        'phase': 'Q158',
        'name': 'Topology Boundary (Local to All-to-All)',
        'n_qubits': n_qubits,
        'topology_sweep': all_results,
        'transition_k': transition_k,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q158_boundary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ks = [r['k'] for r in all_results]
    rand_e = [r['random_error'] for r in all_results]
    llm_e = [r['llm_error'] for r in all_results]
    ax.semilogy(ks, [max(e, 0.001) for e in rand_e], 'o-', color='#F44336',
                label='Random', linewidth=2)
    ax.semilogy(ks, [max(e, 0.001) for e in llm_e], 's-', color='#4CAF50',
                label='LLM', linewidth=2)
    if transition_k:
        ax.axvline(transition_k, color='blue', ls='--', alpha=0.5,
                   label='Transition k=%d' % transition_k)
    ax.set_xlabel('Interaction range k')
    ax.set_ylabel('Error (mHa, log)')
    ax.set_title('(a) Error vs Interaction Range')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    advs = [r['advantage'] for r in all_results]
    conns = [r['connectivity_pct'] for r in all_results]
    colors = ['#4CAF50' if a > 1.5 else '#FF9800' if a > 1 else '#F44336' for a in advs]
    ax.bar(range(len(ks)), advs, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='red', ls='--', label='No advantage')
    ax.axhline(1.5, color='blue', ls=':', label='Significant advantage')
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(['k=%d\n(%.0f%%)' % (k, c) for k, c in zip(ks, conns)], fontsize=7)
    ax.set_ylabel('LLM Advantage (x)')
    ax.set_title('(b) LLM Advantage vs Connectivity')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q158: Topology Boundary (Where LLM Advantage Begins)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q158_boundary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ158 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
