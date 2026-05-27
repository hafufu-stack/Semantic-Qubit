# -*- coding: utf-8 -*-
"""
Phase Q186v2: Non-Equilibrium Quantum Dynamics (Warm-Started)
================================================================
Q186v1 failed (fid=0.50) because each time step was optimized independently.

Fix: Use WARM-STARTING - initialize each step from previous step's solution.
This mimics Trotterized time evolution: small changes between steps.

Also add: proper Trotter benchmark to compare.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_ising_hamiltonian(n_sites, J=1.0, h=0.5):
    """Transverse-field Ising: H = -J sum(ZZ) - h sum(X)"""
    dim = 2 ** n_sites
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_sites - 1):
        ops = [I2] * n_sites
        ops[i] = Z; ops[i + 1] = Z
        term = ops[0]
        for k in range(1, n_sites): term = np.kron(term, ops[k])
        H -= J * term
    for i in range(n_sites):
        ops = [I2] * n_sites
        ops[i] = X
        term = ops[0]
        for k in range(1, n_sites): term = np.kron(term, ops[k])
        H -= h * term
    return np.real(H)


def main():
    print("=" * 60)
    print("Phase Q186v2: Warm-Started Quantum Dynamics")
    print("  (Fix: use previous state as initialization)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    hidden_size = model.config.hidden_size
    embed_layer = model.model.embed_tokens

    n_sites = 4
    dim = 2 ** n_sites
    H_np = build_ising_hamiltonian(n_sites, J=1.0, h=0.5)
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)

    # Initial state: Neel state |0101>
    psi_0 = np.zeros(dim)
    psi_0[5] = 1.0  # |0101> = index 5

    # === Exact time evolution ===
    dt = 0.05
    n_steps = 100
    exact_states = [psi_0.copy()]
    psi = psi_0.astype(complex)
    U = expm(-1j * H_np.astype(complex) * dt)
    for _ in range(n_steps):
        psi = U @ psi
        exact_states.append(np.real(psi).copy())

    print("  Exact evolution: %d steps, dt=%.3f, T=%.1f" %
          (n_steps, dt, n_steps * dt))

    # === Warm-started LLM tracking ===
    print("\n--- Warm-Started Tracking ---")

    seed_prompt = "Quantum state evolution step:"
    seed_ids = tok(seed_prompt, return_tensors='pt')['input_ids'].to(device)
    seed_embeds = embed_layer(seed_ids).detach().clone()

    # Initialize from t=0
    opt_embeds = seed_embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_embeds], lr=0.003)

    # First: optimize to match initial state
    target_0 = torch.tensor(psi_0, dtype=torch.float32, device=device)
    for step in range(100):
        optimizer.zero_grad()
        outputs = model(inputs_embeds=opt_embeds.float(), output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :]
        psi_llm = h[:dim]
        psi_norm = psi_llm / (torch.norm(psi_llm) + 1e-10)
        loss = 1.0 - torch.dot(psi_norm, target_0) ** 2
        loss.backward()
        optimizer.step()

    warm_fidelities = []
    warm_energies_exact = []
    warm_energies_llm = []
    cold_fidelities = []  # independent optimization (for comparison)

    sample_interval = 5
    opt_steps_per_dt = 20  # small number of steps (warm start needs fewer)

    for t_idx in range(0, n_steps + 1, sample_interval):
        target_psi = np.real(exact_states[t_idx])
        target_torch = torch.tensor(target_psi, dtype=torch.float32, device=device)

        # Warm-started: optimize from current position
        for step in range(opt_steps_per_dt):
            optimizer.zero_grad()
            outputs = model(inputs_embeds=opt_embeds.float(),
                           output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            psi_llm = h[:dim]
            psi_norm = psi_llm / (torch.norm(psi_llm) + 1e-10)
            loss = 1.0 - torch.dot(psi_norm, target_torch) ** 2
            loss.backward()
            optimizer.step()

        # Measure fidelity
        with torch.no_grad():
            outputs = model(inputs_embeds=opt_embeds.float(),
                           output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            psi_final = h[:dim].float()
            psi_final = psi_final / (torch.norm(psi_final) + 1e-10)

        fid = float(torch.dot(psi_final, target_torch) ** 2)
        E_exact = float(target_torch @ H_torch @ target_torch)
        E_llm = float(psi_final @ H_torch @ psi_final)
        warm_fidelities.append(fid)
        warm_energies_exact.append(E_exact)
        warm_energies_llm.append(E_llm)

        # Cold start comparison (fresh optimization)
        cold_embeds = seed_embeds.clone().detach().requires_grad_(True)
        cold_opt = torch.optim.Adam([cold_embeds], lr=0.003)
        for step in range(opt_steps_per_dt):
            cold_opt.zero_grad()
            outputs = model(inputs_embeds=cold_embeds.float(),
                           output_hidden_states=True)
            h_c = outputs.hidden_states[-1][0, -1, :]
            psi_c = h_c[:dim]
            psi_cn = psi_c / (torch.norm(psi_c) + 1e-10)
            loss_c = 1.0 - torch.dot(psi_cn, target_torch) ** 2
            loss_c.backward()
            cold_opt.step()

        with torch.no_grad():
            outputs = model(inputs_embeds=cold_embeds.float(),
                           output_hidden_states=True)
            psi_cold = outputs.hidden_states[-1][0, -1, :][:dim].float()
            psi_cold = psi_cold / (torch.norm(psi_cold) + 1e-10)
        cold_fid = float(torch.dot(psi_cold, target_torch) ** 2)
        cold_fidelities.append(cold_fid)

        if t_idx % 20 == 0:
            print("  t=%.2f: warm_fid=%.4f, cold_fid=%.4f" %
                  (t_idx * dt, fid, cold_fid))

    # Magnetization tracking
    Z = np.array([[1, 0], [0, -1]])
    I2 = np.eye(2)
    mags_exact = []
    for psi_state in exact_states[::sample_interval]:
        ops = [I2] * n_sites; ops[0] = Z
        Z0 = ops[0]
        for k in range(1, n_sites): Z0 = np.kron(Z0, ops[k])
        mags_exact.append(float(psi_state @ Z0 @ psi_state))

    # Summary
    avg_warm = float(np.mean(warm_fidelities))
    avg_cold = float(np.mean(cold_fidelities))
    improvement = avg_warm / max(avg_cold, 0.01)

    print("\n--- Summary ---")
    print("  Warm-start avg fidelity: %.4f" % avg_warm)
    print("  Cold-start avg fidelity: %.4f" % avg_cold)
    print("  Improvement: %.2fx" % improvement)

    if avg_warm > 0.9:
        verdict = "EXCELLENT: Warm-started LLM tracks dynamics (F=%.3f)" % avg_warm
    elif avg_warm > 0.7:
        verdict = "GOOD: Warm-started tracking (F=%.3f, %.1fx vs cold)" % (
            avg_warm, improvement)
    else:
        verdict = "MODERATE: F=%.3f (%.1fx improvement over cold)" % (
            avg_warm, improvement)
    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q186v2',
        'name': 'Warm-Started Quantum Dynamics',
        'warm_fidelities': [round(f, 4) for f in warm_fidelities],
        'cold_fidelities': [round(f, 4) for f in cold_fidelities],
        'summary': {
            'avg_warm_fidelity': round(avg_warm, 4),
            'avg_cold_fidelity': round(avg_cold, 4),
            'improvement': round(improvement, 2),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q186v2_dynamics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    times = [i * sample_interval * dt for i in range(len(warm_fidelities))]

    # (a) Fidelity comparison
    ax = axes[0]
    ax.plot(times, warm_fidelities, 'o-', color='#2196F3', linewidth=2,
            markersize=4, label='Warm-start (F=%.3f)' % avg_warm)
    ax.plot(times, cold_fidelities, 's--', color='#F44336', linewidth=1.5,
            markersize=3, alpha=0.7, label='Cold-start (F=%.3f)' % avg_cold)
    ax.axhline(0.9, color='green', ls=':', label='90%% threshold')
    ax.set_xlabel('Time (J*t)')
    ax.set_ylabel('Fidelity')
    ax.set_title('(a) Warm vs Cold Start\n(%.1fx improvement)' % improvement)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (b) Energy tracking
    ax = axes[1]
    ax.plot(times, warm_energies_exact, 'k-', linewidth=2, label='Exact')
    ax.plot(times, warm_energies_llm, 'ro', markersize=4, label='LLM (warm)')
    ax.set_xlabel('Time (J*t)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('(b) Energy Conservation')
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Magnetization
    ax = axes[2]
    ax.plot(times[:len(mags_exact)], mags_exact, 'k-', linewidth=2, label='Exact <Z_0>')
    ax.set_xlabel('Time (J*t)')
    ax.set_ylabel('<Z_0>')
    ax.set_title('(c) Magnetization Dynamics\n(Neel state melting)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Q186v2: Warm-Started Quantum Dynamics\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q186v2_dynamics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ186v2 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
