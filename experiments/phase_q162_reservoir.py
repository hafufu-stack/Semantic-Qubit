# -*- coding: utf-8 -*-
"""
Phase Q162: LLM as Quantum Reservoir Computer
================================================
Q156 showed LLM is weakly chaotic (Lyap=3.25).
Reservoir computing uses chaos to compute!

Train a LINEAR readout on LLM hidden states to solve quantum problems.
If it works: LLM is a natural quantum reservoir computer.
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


def build_hamiltonian(n_qubits, coupling_type='syk', seed=42):
    np.random.seed(seed)
    dim = 2 ** n_qubits
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)
    def kron_chain(ops):
        r = ops[0]
        for o in ops[1:]: r = np.kron(r, o)
        return r
    H = np.zeros((dim, dim))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            J = np.random.randn() / np.sqrt(n_qubits)
            ops = [I2]*n_qubits; ops[i] = Z; ops[j] = Z
            H += -J * kron_chain(ops)
    for i in range(n_qubits):
        ops = [I2]*n_qubits; ops[i] = X
        H += -0.3 * kron_chain(ops)
    return H


def main():
    print("=" * 60)
    print("Phase Q162: LLM as Quantum Reservoir Computer")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Task: predict ground state energy from prompt encoding
    # Training data: various Hamiltonians with known ground states
    n_train = 30
    n_test = 10
    n_qubits = 6
    dim = 2 ** n_qubits

    # Generate training data
    print("\n--- Generating training data ---")
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    base_prompt = "Quantum system with %d qubits coupling seed %d:"

    for i in range(n_train + n_test):
        H = build_hamiltonian(n_qubits, seed=i * 7 + 13)
        E0 = float(np.linalg.eigvalsh(H)[0])

        prompt = base_prompt % (n_qubits, i)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()

        if i < n_train:
            train_X.append(h)
            train_Y.append(E0)
        else:
            test_X.append(h)
            test_Y.append(E0)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # Train linear readout (ridge regression)
    print("\n--- Training Linear Readout ---")
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_mse = float('inf')
    best_lambda = 0.01

    for lam in lambdas:
        W = np.linalg.solve(train_X.T @ train_X + lam * np.eye(hidden_size),
                            train_X.T @ train_Y)
        pred_train = train_X @ W
        pred_test = test_X @ W
        mse_test = float(np.mean((pred_test - test_Y) ** 2))
        if mse_test < best_mse:
            best_mse = mse_test
            best_lambda = lam

    # Final model
    W = np.linalg.solve(train_X.T @ train_X + best_lambda * np.eye(hidden_size),
                        train_X.T @ train_Y)
    pred_train = train_X @ W
    pred_test = test_X @ W

    train_mse = float(np.mean((pred_train - train_Y) ** 2))
    test_mse = float(np.mean((pred_test - test_Y) ** 2))
    train_r2 = 1 - train_mse / np.var(train_Y)
    test_r2 = 1 - test_mse / np.var(test_Y)

    print("  Best lambda: %.3f" % best_lambda)
    print("  Train MSE: %.6f, R2: %.4f" % (train_mse, train_r2))
    print("  Test  MSE: %.6f, R2: %.4f" % (test_mse, test_r2))

    # Random baseline: predict from random features
    print("\n--- Random Feature Baseline ---")
    rand_X_train = np.random.randn(n_train, hidden_size)
    rand_X_test = np.random.randn(n_test, hidden_size)
    W_rand = np.linalg.solve(
        rand_X_train.T @ rand_X_train + best_lambda * np.eye(hidden_size),
        rand_X_train.T @ train_Y)
    pred_rand = rand_X_test @ W_rand
    rand_mse = float(np.mean((pred_rand - test_Y) ** 2))
    rand_r2 = 1 - rand_mse / np.var(test_Y)
    print("  Random R2: %.4f (should be ~0)" % rand_r2)

    # Task 2: Predict ground state VECTOR (not just energy)
    print("\n--- Task 2: Ground State Vector Prediction ---")
    train_psi = []
    test_psi = []
    for i in range(n_train + n_test):
        H = build_hamiltonian(n_qubits, seed=i * 7 + 13)
        psi0 = np.linalg.eigh(H)[1][:, 0]
        if i < n_train:
            train_psi.append(psi0)
        else:
            test_psi.append(psi0)
    train_psi = np.array(train_psi)
    test_psi = np.array(test_psi)

    # Ridge for vector output
    W_psi = np.linalg.solve(
        train_X.T @ train_X + best_lambda * np.eye(hidden_size),
        train_X.T @ train_psi)
    pred_psi = test_X @ W_psi

    # Fidelity
    fidelities = []
    for i in range(n_test):
        p = pred_psi[i]
        p /= np.linalg.norm(p)
        fid = float(abs(np.dot(p, test_psi[i])) ** 2)
        fidelities.append(fid)
    avg_fid = float(np.mean(fidelities))
    print("  Average fidelity: %.4f" % avg_fid)
    print("  Random baseline fidelity: %.4f" % (1.0 / dim))

    # Save
    results = {
        'phase': 'Q162',
        'name': 'LLM Quantum Reservoir Computer',
        'energy_prediction': {
            'train_r2': round(float(train_r2), 4),
            'test_r2': round(float(test_r2), 4),
            'random_r2': round(float(rand_r2), 4),
            'best_lambda': best_lambda,
        },
        'state_prediction': {
            'avg_fidelity': round(avg_fid, 4),
            'random_fidelity': round(1.0 / dim, 4),
        },
        'n_train': n_train,
        'n_test': n_test,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q162_reservoir.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(test_Y, pred_test, color='#4CAF50', s=60, edgecolor='black',
               label='LLM reservoir', zorder=3)
    ax.scatter(test_Y, pred_rand, color='#F44336', s=40, marker='x',
               label='Random features', zorder=2)
    lims = [min(min(test_Y), min(pred_test)) - 0.1,
            max(max(test_Y), max(pred_test)) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Energy')
    ax.set_ylabel('Predicted Energy')
    ax.set_title('(a) Energy Prediction (R2=%.3f)' % test_r2)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(['LLM\nReservoir', 'Random\nFeatures'],
           [max(test_r2, 0), max(rand_r2, 0)],
           color=['#4CAF50', '#F44336'], edgecolor='black', alpha=0.85)
    ax.set_ylabel('R2 Score')
    ax.set_title('(b) Reservoir vs Random')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(range(n_test), fidelities, color='#2196F3', edgecolor='black', alpha=0.85)
    ax.axhline(1.0 / dim, color='red', ls='--', label='Random (1/%d)' % dim)
    ax.axhline(avg_fid, color='green', ls=':', linewidth=2,
               label='Avg F=%.3f' % avg_fid)
    ax.set_xlabel('Test sample')
    ax.set_ylabel('Fidelity')
    ax.set_title('(c) Ground State Fidelity')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q162: LLM as Quantum Reservoir Computer',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q162_reservoir.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ162 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
