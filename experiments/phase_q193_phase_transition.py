# -*- coding: utf-8 -*-
"""
Phase Q193: Quantum Phase Transition Detection
=================================================
Use Embedding VQE to autonomously detect quantum phase transitions
in the transverse-field Ising model.

The Ising model has a quantum critical point at h/J = 1.0:
- h/J < 1: Ferromagnetic (ordered) phase
- h/J > 1: Paramagnetic (disordered) phase

Method:
1. Sweep h/J from 0.1 to 3.0
2. At each point, find ground state via VQE
3. Compute order parameter <Z_0 Z_1> and entanglement entropy
4. Detect the phase transition from discontinuities

If LLM correctly identifies the critical point at h/J ~ 1.0
-> autonomous quantum phase discovery!
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


def build_ising(n_sites, J, h_field):
    """Build transverse-field Ising: H = -J sum(ZZ) - h sum(X)"""
    dim = 2 ** n_sites
    Z = np.array([[1,0],[0,-1]]); X = np.array([[0,1],[1,0]]); I2 = np.eye(2)
    H = np.zeros((dim, dim))
    for i in range(n_sites - 1):
        ops = [I2]*n_sites; ops[i] = Z; ops[i+1] = Z
        term = ops[0]
        for k in range(1, n_sites): term = np.kron(term, ops[k])
        H -= J * term
    for i in range(n_sites):
        ops = [I2]*n_sites; ops[i] = X
        term = ops[0]
        for k in range(1, n_sites): term = np.kron(term, ops[k])
        H -= h_field * term
    return H


def compute_zz_correlation(psi, n_sites, i=0, j=1):
    """Compute <Z_i Z_j> for state psi."""
    dim = 2 ** n_sites
    Z = np.array([[1,0],[0,-1]]); I2 = np.eye(2)
    ops = [I2]*n_sites; ops[i] = Z; ops[j] = Z
    ZZ = ops[0]
    for k in range(1, n_sites): ZZ = np.kron(ZZ, ops[k])
    return float(psi @ ZZ @ psi)


def compute_magnetization(psi, n_sites):
    """Compute <sum_i Z_i> / n_sites."""
    dim = 2 ** n_sites
    Z = np.array([[1,0],[0,-1]]); I2 = np.eye(2)
    mag = 0.0
    for i in range(n_sites):
        ops = [I2]*n_sites; ops[i] = Z
        Zi = ops[0]
        for k in range(1, n_sites): Zi = np.kron(Zi, ops[k])
        mag += float(psi @ Zi @ psi)
    return mag / n_sites


def entanglement_entropy_bipartite(psi, n_sites):
    """Von Neumann entropy of half-chain reduced density matrix."""
    dim = 2 ** n_sites
    half = n_sites // 2
    dim_A = 2 ** half
    dim_B = 2 ** (n_sites - half)
    psi_mat = psi.reshape(dim_A, dim_B)
    rho_A = psi_mat @ psi_mat.T
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def main():
    print("=" * 60)
    print("Phase Q193: Quantum Phase Transition Detection")
    print("  (Can LLM Find the Ising Critical Point?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    embed_layer = model.model.embed_tokens

    n_sites = 4
    dim = 2 ** n_sites
    J = 1.0
    h_values = np.linspace(0.1, 3.0, 20)

    print("  Ising chain: %d sites, J=%.1f" % (n_sites, J))
    print("  Sweeping h/J = %.1f to %.1f (%d points)" %
          (h_values[0], h_values[-1], len(h_values)))

    # Arrays for results
    exact_energies = []
    vqe_energies = []
    exact_zz = []
    vqe_zz = []
    exact_mag = []
    vqe_mag = []
    exact_entropy = []
    vqe_entropy = []
    vqe_errors = []

    for h_val in h_values:
        H_np = build_ising(n_sites, J, h_val)
        H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)

        # Exact solution
        evals, evecs = np.linalg.eigh(H_np)
        psi_exact = evecs[:, 0]
        E_exact = evals[0]

        exact_energies.append(E_exact)
        exact_zz.append(compute_zz_correlation(psi_exact, n_sites))
        exact_mag.append(compute_magnetization(psi_exact, n_sites))
        exact_entropy.append(entanglement_entropy_bipartite(psi_exact, n_sites))

        # VQE
        seed = "Ising ground state h=%.2f:" % h_val
        seed_ids = tok(seed, return_tensors='pt')['input_ids'].to(device)
        seed_embeds = embed_layer(seed_ids).detach().clone()
        opt = seed_embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.003)

        for step in range(200):
            optimizer.zero_grad()
            outputs = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h_out = outputs.hidden_states[-1][0, -1, :]
            psi = h_out[:dim]
            psi_n = psi / (torch.norm(psi) + 1e-10)
            E = psi_n @ H_torch @ psi_n
            E.backward()
            optimizer.step()

        # Evaluate VQE state
        with torch.no_grad():
            outputs = model(inputs_embeds=opt.float(), output_hidden_states=True)
            psi_vqe = outputs.hidden_states[-1][0, -1, :][:dim].float()
            psi_vqe = psi_vqe / (torch.norm(psi_vqe) + 1e-10)
            psi_vqe_np = psi_vqe.cpu().numpy()

        E_vqe = float(psi_vqe @ H_torch @ psi_vqe)
        vqe_energies.append(E_vqe)
        vqe_zz.append(compute_zz_correlation(psi_vqe_np, n_sites))
        vqe_mag.append(compute_magnetization(psi_vqe_np, n_sites))
        vqe_entropy.append(entanglement_entropy_bipartite(psi_vqe_np, n_sites))
        vqe_errors.append(abs(E_vqe - E_exact) * 1000)

        if h_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
            print("  h/J=%.1f: E_exact=%.4f, E_vqe=%.4f, err=%.2f mHa" %
                  (h_val, E_exact, E_vqe, vqe_errors[-1]))

    # Phase transition detection
    # Find where d(ZZ)/d(h) is maximum (steepest change)
    exact_zz_arr = np.array(exact_zz)
    vqe_zz_arr = np.array(vqe_zz)
    h_arr = np.array(h_values)

    # Derivative of ZZ correlation
    dZZ_exact = np.gradient(exact_zz_arr, h_arr)
    dZZ_vqe = np.gradient(vqe_zz_arr, h_arr)

    exact_critical = h_arr[np.argmax(np.abs(dZZ_exact))]
    vqe_critical = h_arr[np.argmax(np.abs(dZZ_vqe))]

    # Also: entropy peak (marks quantum critical point)
    exact_ent_peak = h_arr[np.argmax(exact_entropy)]
    vqe_ent_peak = h_arr[np.argmax(vqe_entropy)]

    print("\n--- Phase Transition Detection ---")
    print("  Exact critical point (ZZ): h/J = %.2f" % exact_critical)
    print("  VQE critical point (ZZ):   h/J = %.2f" % vqe_critical)
    print("  Exact entropy peak:        h/J = %.2f" % exact_ent_peak)
    print("  VQE entropy peak:          h/J = %.2f" % vqe_ent_peak)
    print("  True critical point:       h/J = 1.00")

    critical_error = abs(vqe_critical - 1.0)
    entropy_error = abs(vqe_ent_peak - 1.0)

    avg_vqe_error = float(np.mean(vqe_errors))
    n_chem = sum(1 for e in vqe_errors if e < 1.6)

    print("\n  VQE: %d/%d chemical accuracy, avg=%.2f mHa" %
          (n_chem, len(h_values), avg_vqe_error))

    if critical_error < 0.3:
        verdict = "PHASE TRANSITION DETECTED at h/J=%.2f (error=%.2f from true)" % (
            vqe_critical, critical_error)
    else:
        verdict = "PARTIAL: critical point at h/J=%.2f (error=%.2f)" % (
            vqe_critical, critical_error)

    # Save
    results = {
        'phase': 'Q193',
        'name': 'Quantum Phase Transition Detection',
        'model': 'Transverse-field Ising (%d sites)' % n_sites,
        'h_values': [round(h, 3) for h in h_values],
        'exact': {
            'energies': [round(e, 6) for e in exact_energies],
            'zz_correlation': [round(z, 4) for z in exact_zz],
            'magnetization': [round(m, 4) for m in exact_mag],
            'entropy': [round(s, 4) for s in exact_entropy],
            'critical_point': round(exact_critical, 2),
        },
        'vqe': {
            'energies': [round(e, 6) for e in vqe_energies],
            'zz_correlation': [round(z, 4) for z in vqe_zz],
            'entropy': [round(s, 4) for s in vqe_entropy],
            'critical_point': round(vqe_critical, 2),
            'errors_mHa': [round(e, 2) for e in vqe_errors],
        },
        'summary': {
            'critical_error': round(critical_error, 3),
            'entropy_peak_error': round(entropy_error, 3),
            'avg_vqe_error_mHa': round(avg_vqe_error, 2),
            'chem_accuracy_pct': round(100 * n_chem / len(h_values), 1),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q193_phase_transition.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Energy
    ax = axes[0][0]
    ax.plot(h_arr, exact_energies, 'k-', linewidth=2, label='Exact')
    ax.plot(h_arr, vqe_energies, 'ro', markersize=5, label='LLM VQE')
    ax.axvline(1.0, color='green', ls='--', alpha=0.5, label='h/J=1 (critical)')
    ax.set_xlabel('h/J')
    ax.set_ylabel('Ground State Energy')
    ax.set_title('(a) Ground State Energy')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) ZZ correlation (order parameter)
    ax = axes[0][1]
    ax.plot(h_arr, exact_zz, 'k-', linewidth=2, label='Exact')
    ax.plot(h_arr, vqe_zz, 'bo', markersize=5, label='VQE')
    ax.axvline(1.0, color='green', ls='--', alpha=0.5)
    ax.axvline(vqe_critical, color='red', ls=':', label='VQE critical: %.2f' % vqe_critical)
    ax.set_xlabel('h/J')
    ax.set_ylabel('<Z_0 Z_1>')
    ax.set_title('(b) Order Parameter\n(ZZ Correlation)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Entanglement entropy
    ax = axes[1][0]
    ax.plot(h_arr, exact_entropy, 'k-', linewidth=2, label='Exact')
    ax.plot(h_arr, vqe_entropy, 's', color='#E91E63', markersize=5, label='VQE')
    ax.axvline(1.0, color='green', ls='--', alpha=0.5)
    ax.set_xlabel('h/J')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('(c) Entanglement Entropy\n(Peak = Critical Point)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) VQE error
    ax = axes[1][1]
    ax.semilogy(h_arr, vqe_errors, 'o-', color='#FF9800', linewidth=2, markersize=5)
    ax.axhline(1.6, color='green', ls='--', label='Chemical accuracy')
    ax.axvline(1.0, color='red', ls=':', alpha=0.5)
    ax.set_xlabel('h/J')
    ax.set_ylabel('VQE Error (mHa)')
    ax.set_title('(d) VQE Accuracy Across Phase Diagram\n(%d/%d chem acc)' %
                (n_chem, len(h_values)))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    plt.suptitle('Q193: Quantum Phase Transition Detection\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q193_phase_transition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ193 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
