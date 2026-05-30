# -*- coding: utf-8 -*-
"""
Phase Q221: Quantum Volume
============================
IBM's industry-standard benchmark for quantum computers.
How many effective qubits does the LLM "quantum processor" have?

QV = 2^n where n is the largest circuit width where random circuits
succeed with >2/3 probability.
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


def random_unitary(dim, rng):
    """Generate random unitary via QR decomposition of random complex matrix."""
    A = rng.randn(dim, dim) + 1j * rng.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


def run_qv_circuit(model, tok, device, n_qubits, rng, n_steps=150, lr=0.01):
    """Run a random quantum volume circuit of width n_qubits."""
    dim = 2 ** n_qubits
    U = random_unitary(dim, rng)
    U_real = np.real(U).astype(np.float32)  # Use real part as target transform

    # Random input state
    psi_in = rng.randn(dim).astype(np.float32)
    psi_in /= np.linalg.norm(psi_in)
    target = U_real @ psi_in
    target /= np.linalg.norm(target)

    # Use LLM to find the transformation
    embed_layer = model.model.embed_tokens
    prompt = "quantum circuit depth %d:" % n_qubits
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    # Inject input state
    with torch.no_grad():
        embeds[0, -1, :dim] = torch.tensor(psi_in, device=device)

    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)
    target_torch = torch.tensor(target, device=device)

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi_out = h / (torch.norm(h) + 1e-10)
        loss = -torch.dot(psi_out, target_torch) ** 2  # maximize fidelity
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h_final = out.hidden_states[-1][0, -1, :dim].float()
        psi_final = h_final / (torch.norm(h_final) + 1e-10)
        fid = float(torch.dot(psi_final, target_torch) ** 2)

    return fid


def main():
    print("=" * 60)
    print("Phase Q221: Quantum Volume")
    print("  (IBM's standard: how many effective qubits?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    qubit_range = range(1, 7)  # 1 to 6 qubits (dim 2 to 64)
    n_trials = 8
    threshold = 2.0 / 3.0

    all_results = []
    max_qv_qubits = 0

    for n_q in qubit_range:
        dim = 2 ** n_q
        print("\n--- %d qubits (dim=%d) ---" % (n_q, dim))

        fids = []
        rng = np.random.RandomState(42 + n_q)
        for trial in range(n_trials):
            fid = run_qv_circuit(model, tok, device, n_q, rng, n_steps=150)
            fids.append(fid)

        avg_fid = np.mean(fids)
        success_rate = np.mean([1 if f > threshold else 0 for f in fids])
        passed = success_rate > threshold

        if passed:
            max_qv_qubits = n_q

        print("  avg_fid=%.4f, success=%.0f%%, %s" %
              (avg_fid, success_rate * 100, "PASS" if passed else "FAIL"))

        all_results.append({
            'n_qubits': n_q,
            'dim': dim,
            'fidelities': [round(f, 4) for f in fids],
            'avg_fidelity': round(avg_fid, 4),
            'success_rate': round(success_rate, 4),
            'passed': bool(passed),
        })

    qv = 2 ** max_qv_qubits if max_qv_qubits > 0 else 1
    verdict = "Quantum Volume = %d (%d effective qubits)" % (qv, max_qv_qubits)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q221',
        'name': 'Quantum Volume',
        'qubits': all_results,
        'summary': {
            'quantum_volume': qv,
            'max_qubits': max_qv_qubits,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q221_quantum_volume.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    qubits = [r['n_qubits'] for r in all_results]
    avg_fids = [r['avg_fidelity'] for r in all_results]
    colors = ['#4CAF50' if r['passed'] else '#F44336' for r in all_results]
    ax.bar(qubits, avg_fids, color=colors, edgecolor='black')
    ax.axhline(threshold, color='black', ls='--', label='Threshold (2/3)')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Average Fidelity')
    ax.set_title('(a) Fidelity vs Qubit Count')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    success_rates = [r['success_rate'] for r in all_results]
    ax.bar(qubits, success_rates, color=colors, edgecolor='black')
    ax.axhline(threshold, color='black', ls='--', label='Threshold (2/3)')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Success Rate')
    ax.set_title('(b) Success Rate vs Qubit Count')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q221: Quantum Volume = %d\n%s' % (qv, verdict),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q221_quantum_volume.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ221 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
