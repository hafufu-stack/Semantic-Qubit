# -*- coding: utf-8 -*-
"""
Phase Q264: Entanglement Monogamy
====================================
MY IDEA: In quantum mechanics, entanglement is "monogamous" -
if A is maximally entangled with B, it cannot be entangled with C.
Does this fundamental constraint hold in LLM hidden states?
Test the CKW (Coffman-Kundu-Wootters) inequality.
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

def concurrence_2qubit(rho):
    """Compute concurrence for a 2-qubit density matrix."""
    sy = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sy, sy)
    rho_tilde = sy_sy @ rho.conj() @ sy_sy
    R = rho @ rho_tilde
    eigvals = np.sort(np.real(np.sqrt(np.maximum(np.linalg.eigvals(R), 0))))[::-1]
    return max(0.0, float(eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))

def tangle(rho_ab):
    """Tangle = concurrence^2"""
    return concurrence_2qubit(rho_ab) ** 2

def main():
    print("=" * 60)
    print("Phase Q264: Entanglement Monogamy")
    print("  (CKW inequality in LLM representations)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    prompts = [
        "quantum entanglement between three particles",
        "classical information shared among parties",
        "love triangle between three people",
        "energy transfer in molecular chains",
        "neural connections between brain regions",
        "gravitational interaction of three bodies",
    ]

    all_results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Take 3 "qubits" from the hidden state: dims [0:2], [2:4], [4:6]
        h = out.hidden_states[n_layers][0, -1, :6].float().cpu().numpy()
        h /= np.linalg.norm(h) + 1e-10

        # Full 3-qubit state (8-dim, pad with zeros)
        psi_8 = np.zeros(8)
        psi_8[:6] = h
        psi_8 /= np.linalg.norm(psi_8) + 1e-10
        rho_full = np.outer(psi_8, psi_8.conj())
        rho_full = 0.7 * rho_full + 0.3 * np.eye(8) / 8
        rho_full /= np.trace(rho_full)

        # Trace out each qubit to get 2-qubit reduced density matrices
        # rho_AB: trace out C (qubit 3)
        rho_AB = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    rho_AB[i, j] += rho_full[i*2+k, j*2+k]

        # rho_AC: trace out B (qubit 2)
        rho_AC = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        rho_AC[i*2+k, j*2+l] += rho_full[i*4+l*2+k, j*4+l*2+k] if (i*4+l*2+k < 8 and j*4+l*2+k < 8) else 0

        # Simplified: use direct partial trace
        # rho_A: trace out BC
        rho_A = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    rho_A[i, j] += rho_full[i*4+k, j*4+k]

        # Tangle computations
        try:
            tau_AB = tangle(rho_AB / np.trace(rho_AB))
        except:
            tau_AB = 0
        try:
            tau_AC = tangle(rho_AC / np.trace(rho_AC))
        except:
            tau_AC = 0

        # Linear entropy as proxy for tau_A(BC)
        purity_A = float(np.real(np.trace(rho_A @ rho_A)))
        tau_ABC = max(0, 4 * (1 - purity_A) / 3)  # linear entropy proxy

        # CKW inequality: tau_A(BC) >= tau_AB + tau_AC
        ckw_lhs = tau_ABC
        ckw_rhs = tau_AB + tau_AC
        ckw_satisfied = ckw_lhs >= ckw_rhs - 1e-6

        residual = ckw_lhs - ckw_rhs  # positive = monogamy satisfied

        print("  '%s'..." % prompt[:35])
        print("    tau_AB=%.4f, tau_AC=%.4f, tau_ABC=%.4f, CKW: %s (res=%.4f)" % (
            tau_AB, tau_AC, tau_ABC, "YES" if ckw_satisfied else "NO", residual))

        all_results.append({
            'prompt': prompt[:35],
            'tau_AB': round(tau_AB, 4), 'tau_AC': round(tau_AC, 4),
            'tau_ABC': round(tau_ABC, 4),
            'ckw_satisfied': bool(ckw_satisfied), 'residual': round(residual, 4),
        })

    n_satisfied = sum(1 for r in all_results if r['ckw_satisfied'])

    if n_satisfied == len(all_results):
        verdict = "MONOGAMY HOLDS: CKW satisfied %d/%d (quantum law obeyed!)" % (n_satisfied, len(all_results))
    elif n_satisfied > len(all_results) // 2:
        verdict = "MOSTLY MONOGAMOUS: %d/%d satisfy CKW" % (n_satisfied, len(all_results))
    else:
        verdict = "MONOGAMY VIOLATED: only %d/%d satisfy CKW" % (n_satisfied, len(all_results))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q264', 'name': 'Entanglement Monogamy',
        'scenarios': all_results,
        'summary': {'n_satisfied': n_satisfied, 'total': len(all_results), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q264_monogamy.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_results))
    ax = axes[0]
    ax.bar(x - 0.25, [r['tau_AB'] for r in all_results], 0.25, label='tau_AB', color='#E91E63', edgecolor='black')
    ax.bar(x, [r['tau_AC'] for r in all_results], 0.25, label='tau_AC', color='#FF9800', edgecolor='black')
    ax.bar(x + 0.25, [r['tau_ABC'] for r in all_results], 0.25, label='tau_A(BC)', color='#2196F3', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))], fontsize=8)
    ax.set_ylabel('Tangle'); ax.set_title('(a) Tangle Distribution')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    residuals = [r['residual'] for r in all_results]
    colors = ['#4CAF50' if r >= -1e-6 else '#F44336' for r in residuals]
    ax.bar(x, residuals, color=colors, edgecolor='black')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))], fontsize=8)
    ax.set_ylabel('CKW Residual'); ax.set_title('(b) Monogamy (>0 = satisfied)')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q264: Entanglement Monogamy\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q264_monogamy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ264 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
