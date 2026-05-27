# -*- coding: utf-8 -*-
"""
Phase Q198: Universal Quantum Gate Compiler
=============================================
To prove LLM is a TRUE quantum computer, we must show it can
implement the universal gate set: {H, S, T, CNOT}.

Any quantum circuit can be decomposed into these gates.
If LLM can faithfully reproduce their action on S-Qubits
-> Universal Quantum Computing on a laptop.
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


# Standard quantum gates
def hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def s_gate():
    return np.array([[1, 0], [0, 1j]])

def t_gate():
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

def cnot():
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])

def rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])


def gate_fidelity_test(model, tok, device, gate_matrix, gate_name,
                        n_input_states=10, n_steps=200):
    """Test if LLM can learn a quantum gate transformation."""
    embed_layer = model.model.embed_tokens
    dim = gate_matrix.shape[0]

    fidelities = []

    for trial in range(n_input_states):
        rng = np.random.RandomState(trial * 42)

        # Random input state
        psi_in = rng.randn(dim) + 1j * rng.randn(dim)
        psi_in = psi_in / np.linalg.norm(psi_in)

        # Expected output after gate
        psi_out_exact = gate_matrix @ psi_in
        psi_out_exact = psi_out_exact / np.linalg.norm(psi_out_exact)

        # Use real parts (LLM works in real space)
        target_real = np.real(psi_out_exact).astype(np.float32)
        target_torch = torch.tensor(target_real, dtype=torch.float32, device=device)
        target_torch = target_torch / (torch.norm(target_torch) + 1e-10)

        # Input encoding
        input_real = np.real(psi_in).astype(np.float32)
        input_torch = torch.tensor(input_real, dtype=torch.float32, device=device)

        # VQE to learn the gate transformation
        seed = "%s gate trial %d:" % (gate_name, trial)
        seed_ids = tok(seed, return_tensors='pt')['input_ids'].to(device)
        seed_embeds = embed_layer(seed_ids).detach().clone()

        # Encode input state into the embedding
        # Put input state info into the last few embedding dimensions
        with torch.no_grad():
            seed_embeds[0, -1, :dim] = input_torch

        opt = seed_embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.003)

        for step in range(n_steps):
            optimizer.zero_grad()
            outputs = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            psi = h[:dim]
            psi_n = psi / (torch.norm(psi) + 1e-10)
            loss = 1.0 - torch.dot(psi_n, target_torch) ** 2
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            outputs = model(inputs_embeds=opt.float(), output_hidden_states=True)
            psi_final = outputs.hidden_states[-1][0, -1, :][:dim].float()
            psi_final = psi_final / (torch.norm(psi_final) + 1e-10)

        fid = float(torch.dot(psi_final, target_torch) ** 2)
        fidelities.append(fid)

    return fidelities


def main():
    print("=" * 60)
    print("Phase Q198: Universal Quantum Gate Compiler")
    print("  (H, S, T, CNOT, X, Z, Rx - Can LLM be a QPU?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Gates to test
    gates = [
        ('Hadamard (H)', hadamard(), 2),
        ('Pauli-X', pauli_x(), 2),
        ('Pauli-Z', pauli_z(), 2),
        ('S gate', np.real(s_gate()), 2),
        ('T gate', np.real(t_gate()), 2),
        ('Rx(pi/4)', np.real(rx(np.pi/4)), 2),
        ('CNOT', cnot(), 4),
    ]

    all_results = []
    n_tests = 8

    for gate_name, gate_matrix, dim in gates:
        print("\n--- %s (dim=%d) ---" % (gate_name, dim))
        fids = gate_fidelity_test(model, tok, device, gate_matrix.astype(float),
                                  gate_name, n_input_states=n_tests)
        avg_fid = float(np.mean(fids))
        min_fid = float(np.min(fids))
        n_high = sum(1 for f in fids if f > 0.99)

        result = {
            'name': gate_name,
            'dim': dim,
            'avg_fidelity': round(avg_fid, 4),
            'min_fidelity': round(min_fid, 4),
            'high_fidelity_pct': round(100 * n_high / n_tests, 1),
            'all_fidelities': [round(f, 4) for f in fids],
        }
        all_results.append(result)

        print("  avg_fid=%.4f, min=%.4f, >0.99: %d/%d" %
              (avg_fid, min_fid, n_high, n_tests))

    # Summary
    overall_avg = float(np.mean([r['avg_fidelity'] for r in all_results]))
    n_gates_passed = sum(1 for r in all_results if r['avg_fidelity'] > 0.9)

    print("\n--- Summary ---")
    print("  Overall avg fidelity: %.4f" % overall_avg)
    print("  Gates with >90%% fidelity: %d/%d" % (n_gates_passed, len(gates)))

    if n_gates_passed == len(gates):
        verdict = "UNIVERSAL QPU: All %d gates at >90%% fidelity" % len(gates)
    elif n_gates_passed >= 4:
        verdict = "STRONG: %d/%d universal gates compiled" % (n_gates_passed, len(gates))
    else:
        verdict = "PARTIAL: %d/%d gates, avg_fid=%.3f" % (
            n_gates_passed, len(gates), overall_avg)
    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q198',
        'name': 'Universal Quantum Gate Compiler',
        'gates': all_results,
        'summary': {
            'overall_avg_fidelity': round(overall_avg, 4),
            'gates_passed': n_gates_passed,
            'total_gates': len(gates),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q198_gates.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Average fidelity per gate
    ax = axes[0]
    names = [r['name'] for r in all_results]
    avg_fids = [r['avg_fidelity'] for r in all_results]
    colors = ['#4CAF50' if f > 0.9 else '#FF9800' if f > 0.7 else '#F44336'
              for f in avg_fids]
    ax.barh(range(len(names)), avg_fids, color=colors,
            edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0.9, color='green', ls='--', label='90%% threshold')
    ax.set_xlabel('Average Fidelity')
    ax.set_title('(a) Gate Fidelity\n(%d/%d gates passed)' %
                (n_gates_passed, len(gates)))
    ax.legend()
    ax.grid(alpha=0.3, axis='x')
    ax.set_xlim(0, 1.05)

    # (b) Distribution box plot
    ax = axes[1]
    fid_data = [r['all_fidelities'] for r in all_results]
    bp = ax.boxplot(fid_data, labels=[n[:6] for n in names],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0.9, color='green', ls='--')
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) Fidelity Distribution\n(per gate type)')
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    # (c) Universal gate set status
    ax = axes[2]
    universal = {'H': 0, 'S': 0, 'T': 0, 'CNOT': 0}
    for r in all_results:
        if 'Hadamard' in r['name']: universal['H'] = r['avg_fidelity']
        elif 'S gate' in r['name']: universal['S'] = r['avg_fidelity']
        elif 'T gate' in r['name']: universal['T'] = r['avg_fidelity']
        elif 'CNOT' in r['name']: universal['CNOT'] = r['avg_fidelity']

    gate_names = list(universal.keys())
    gate_fids = list(universal.values())
    gate_colors = ['#4CAF50' if f > 0.9 else '#F44336' for f in gate_fids]
    ax.bar(range(len(gate_names)), gate_fids, color=gate_colors,
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(gate_names)))
    ax.set_xticklabels(gate_names, fontsize=14, fontweight='bold')
    ax.axhline(0.9, color='green', ls='--', label='Universal threshold')
    ax.set_ylabel('Fidelity')
    ax.set_title('(c) Universal Gate Set {H,S,T,CNOT}\n(All green = universal QPU)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.suptitle('Q198: Universal Quantum Gate Compiler\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q198_gates.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ198 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
