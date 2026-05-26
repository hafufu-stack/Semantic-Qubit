# -*- coding: utf-8 -*-
"""
Phase Q174: Token Sampling as Quantum Measurement
====================================================
The LLM's softmax output = probability distribution over tokens.
This IS a quantum measurement (Born rule: P = |<token|psi>|^2).

Test: sample many tokens, compute "expectation values" of
quantum observables from the token probability distribution.
Compare with exact quantum expectation values.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_syk(n_qubits, seed=42):
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
    print("Phase Q174: Token Sampling = Quantum Measurement")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    hidden_size = model.config.hidden_size

    prompt = "The quantum measurement result is:"
    inp = tok(prompt, return_tensors='pt').to(device)

    # Get logits (pre-softmax) = "wavefunction amplitudes"
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :]  # Next-token logits

    # Temperature sweep: how does measurement precision depend on T?
    temperatures = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0]

    # Build quantum Hamiltonian
    n_qubits = 6
    dim = 2 ** n_qubits
    H = build_syk(n_qubits, seed=42)
    E_exact = float(np.linalg.eigvalsh(H)[0])

    # Get hidden state as quantum state
    with torch.no_grad():
        out2 = model(**inp, output_hidden_states=True)
    h = out2.hidden_states[-1][0, -1, :].float().cpu().numpy()
    psi = h[:dim].copy()
    psi /= np.linalg.norm(psi) + 1e-10
    E_hidden = float(np.real(psi @ H @ psi))

    all_results = []

    for T in temperatures:
        # Apply temperature to logits -> probabilities (Born rule analog)
        probs = F.softmax(logits / T, dim=0).cpu().numpy()

        # "Measurement" statistics
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
        max_prob = float(np.max(probs))
        effective_states = float(np.exp(entropy))  # Participation ratio

        # Sample tokens (quantum measurements)
        n_samples = 100
        sampled_ids = torch.multinomial(
            F.softmax(logits / T, dim=0).unsqueeze(0), n_samples, replacement=True
        )[0].cpu().numpy()

        # Decode sampled tokens
        unique_tokens = set()
        for sid in sampled_ids[:10]:
            unique_tokens.add(tok.decode([sid]).strip())

        # Map token IDs to quantum states and compute expectation
        # Each token ID -> extract bits -> quantum basis state
        token_energies = []
        for sid in sampled_ids:
            # Map token ID to quantum state index (modulo dim)
            state_idx = sid % dim
            # Energy of this basis state
            E_basis = float(H[state_idx, state_idx])
            token_energies.append(E_basis)

        E_sampled = float(np.mean(token_energies))
        E_std = float(np.std(token_energies))

        result = {
            'temperature': float(T),
            'entropy': round(entropy, 4),
            'max_prob': round(max_prob, 6),
            'effective_states': round(effective_states, 1),
            'E_sampled': round(E_sampled, 4),
            'E_std': round(E_std, 4),
            'E_hidden': round(E_hidden, 4),
            'E_exact': round(E_exact, 4),
            'n_unique_tokens': len(set(sampled_ids)),
        }
        all_results.append(result)

        if T in [0.01, 0.5, 1.0, 5.0]:
            sample_text = ', '.join(list(unique_tokens)[:5])
            print("  T=%.2f: entropy=%.2f, states=%.0f, E=%.4f, tokens=[%s]" %
                  (T, entropy, effective_states, E_sampled, sample_text[:40]))

    # Summary
    print("\n--- Measurement Summary ---")
    print("  Exact E0: %.4f" % E_exact)
    print("  Hidden state E: %.4f" % E_hidden)
    best_T = min(all_results, key=lambda r: abs(r['E_sampled'] - E_exact))
    print("  Best sampling T: %.2f (E=%.4f, err=%.2f mHa)" %
          (best_T['temperature'], best_T['E_sampled'],
           abs(best_T['E_sampled'] - E_exact) * 1000))

    # Born rule test: does P(token) follow |<token|psi>|^2?
    print("\n--- Born Rule Test ---")
    top_k = 50
    probs_T1 = F.softmax(logits, dim=0).cpu().numpy()
    top_ids = np.argsort(probs_T1)[-top_k:]
    born_probs = []
    softmax_probs = []
    for tid in top_ids:
        # "Born probability": |<basis|psi>|^2
        state_idx = tid % dim
        born_p = abs(psi[state_idx]) ** 2
        born_probs.append(born_p)
        softmax_probs.append(probs_T1[tid])

    born_corr = float(np.corrcoef(born_probs, softmax_probs)[0, 1])
    if np.isnan(born_corr):
        born_corr = 0.0
    print("  Correlation(Born, Softmax): %.4f" % born_corr)
    print("  Born rule match: %s" %
          ("YES" if abs(born_corr) > 0.3 else "NO (different distributions)"))

    # Save
    results = {
        'phase': 'Q174',
        'name': 'Token Sampling as Quantum Measurement',
        'measurements': all_results,
        'born_rule_correlation': round(born_corr, 4),
        'exact_energy': round(E_exact, 4),
        'hidden_energy': round(E_hidden, 4),
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q174_measurement.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    Ts = [r['temperature'] for r in all_results]
    entropies = [r['entropy'] for r in all_results]
    eff_states = [r['effective_states'] for r in all_results]
    ax.semilogx(Ts, entropies, 'o-', color='#E91E63', linewidth=1.5, label='Entropy')
    ax2 = ax.twinx()
    ax2.semilogx(Ts, eff_states, 's-', color='#4CAF50', linewidth=1.5,
                 label='Effective states')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Entropy (nats)', color='#E91E63')
    ax2.set_ylabel('Effective States', color='#4CAF50')
    ax.set_title('(a) Measurement Entropy vs Temperature')
    ax.grid(alpha=0.3)

    ax = axes[1]
    Es = [r['E_sampled'] for r in all_results]
    ax.semilogx(Ts, Es, 'o-', color='#2196F3', linewidth=1.5, label='Sampled E')
    ax.axhline(E_exact, color='green', ls='--', linewidth=2, label='Exact E0')
    ax.axhline(E_hidden, color='orange', ls=':', linewidth=1.5, label='Hidden state E')
    ax.fill_between(Ts,
                     [r['E_sampled'] - r['E_std'] for r in all_results],
                     [r['E_sampled'] + r['E_std'] for r in all_results],
                     alpha=0.2, color='#2196F3')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Energy')
    ax.set_title('(b) Sampled Energy vs Temperature')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    ax = axes[2]
    if len(born_probs) > 0 and len(softmax_probs) > 0:
        ax.scatter(born_probs, softmax_probs, s=20, alpha=0.6, color='#9C27B0')
        ax.set_xlabel('Born Probability |<basis|psi>|^2')
        ax.set_ylabel('Softmax Probability')
        ax.set_title('(c) Born Rule Test (r=%.3f)' % born_corr)
        ax.grid(alpha=0.3)

    plt.suptitle('Q174: Token Sampling = Quantum Measurement (Born Rule)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q174_measurement.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ174 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
