# -*- coding: utf-8 -*-
"""
Phase Q155: Prompt VQE (Words as Variational Parameters)
==========================================================
Standard VQE: optimize circuit parameters to minimize <psi|H|psi>.
Prompt VQE: optimize PROMPT WORDS to minimize <psi_LLM|H|psi_LLM>.

The variational parameters are WORDS, not angles!
This is the ultimate test of "semantic quantum computing".
"""
import os, sys, json, time, gc, itertools
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


def build_h2_hamiltonian(bond_length=0.74):
    """H2 molecule Hamiltonian (same as Q148)."""
    dim = 16
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)

    def kron4(a, b, c, d):
        return np.kron(np.kron(np.kron(a, b), c), d)

    r = bond_length
    g0 = -0.5 - 0.2 * np.exp(-r)
    g1 = 0.2 * np.exp(-0.5 * r)
    g2 = 0.15 * np.exp(-0.3 * r)
    g3 = -0.1 * np.exp(-0.8 * r)

    H = np.real(
        g0 * kron4(I2, I2, I2, I2) +
        g1 * kron4(Z, I2, I2, I2) +
        g1 * kron4(I2, Z, I2, I2) +
        g2 * kron4(Z, Z, I2, I2) +
        g2 * kron4(I2, I2, Z, Z) +
        g3 * kron4(X, X, I2, I2) +
        g3 * kron4(I2, I2, X, X) +
        g3 * kron4(Z, I2, Z, I2) * 0.5 * g2 / g3 +  # Adjusted
        g3 * kron4(I2, Z, I2, Z) * 0.5 * g2 / g3
    )
    return H


def main():
    print("=" * 60)
    print("Phase Q155: Prompt VQE")
    print("  (Words as Variational Parameters)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    dim = 16  # H2 STO-3G
    H = build_h2_hamiltonian(0.74)
    E_exact = float(np.linalg.eigvalsh(H)[0])
    psi_exact = np.linalg.eigh(H)[1][:, 0]
    print("  Exact ground state energy: %.6f" % E_exact)

    def evaluate_prompt(prompt):
        """Get best S-Qubit energy from a prompt (no GD)."""
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        best_E = float('inf')
        best_psi = None
        for li in range(0, n_layers, 2):
            h = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
            for offset in range(0, min(hidden_size, dim * 10), dim):
                if offset + dim <= hidden_size:
                    psi = h[offset:offset + dim].copy()
                    norm = np.linalg.norm(psi)
                    if norm > 1e-8:
                        psi /= norm
                        E = float(np.real(psi @ H @ psi))
                        if E < best_E:
                            best_E = E
                            best_psi = psi.copy()
        return best_E, best_psi

    # === PROMPT SPACE SEARCH ===
    # Template: "[subject] [verb] [object] [modifier]"

    subjects = [
        "Hydrogen molecule", "H2 system", "Two protons",
        "Quantum state", "Ground state", "Electron pair",
        "Chemical bond", "Molecular orbital", "Wave function",
        "Diatomic hydrogen", "Bonding electrons", "Atom pair",
    ]

    verbs = [
        "at equilibrium", "energy is", "ground state energy",
        "lowest energy", "minimum energy", "wavefunction",
    ]

    modifiers = [
        "in STO-3G basis", "at 0.74 Angstrom",
        "with variational method", "computed exactly",
        "quantum chemistry", "Hartree-Fock",
    ]

    # Also test completely irrelevant prompts
    irrelevant = [
        "The weather is nice today",
        "I like pizza and burgers",
        "The cat sat on the mat",
        "Hello world program in Python",
        "Breaking news from Tokyo",
    ]

    # Phase 1: Individual word impact
    print("\n--- Phase 1: Subject Word Impact ---")
    subject_results = []
    for subj in subjects:
        prompt = "%s ground state energy:" % subj
        E, _ = evaluate_prompt(prompt)
        err = abs(E - E_exact) * 1000
        subject_results.append({
            'subject': subj,
            'prompt': prompt,
            'energy': round(E, 6),
            'error_mha': round(err, 4),
        })
        print("  %-25s: E=%.4f, err=%.2f mHa" % (subj, E, err))

    # Phase 2: Combinatorial search (top subjects x verbs)
    print("\n--- Phase 2: Combinatorial Prompt Search ---")
    # Sort by best subject
    subject_results.sort(key=lambda x: x['error_mha'])
    top_subjects = [r['subject'] for r in subject_results[:4]]

    combo_results = []
    for subj in top_subjects:
        for verb in verbs:
            for mod in modifiers:
                prompt = "%s %s %s" % (subj, verb, mod)
                E, psi = evaluate_prompt(prompt)
                err = abs(E - E_exact) * 1000
                fid = float(abs(np.dot(psi, psi_exact)) ** 2) if psi is not None else 0
                combo_results.append({
                    'prompt': prompt[:50],
                    'energy': round(E, 6),
                    'error_mha': round(err, 4),
                    'fidelity': round(fid, 4),
                })

    combo_results.sort(key=lambda x: x['error_mha'])
    print("  Tested %d combinations" % len(combo_results))
    print("\n  TOP 5 PROMPTS:")
    for i, r in enumerate(combo_results[:5]):
        print("    %d. '%s' -> %.2f mHa (F=%.4f)" %
              (i + 1, r['prompt'], r['error_mha'], r['fidelity']))
    print("\n  WORST 3 PROMPTS:")
    for r in combo_results[-3:]:
        print("    '%s' -> %.2f mHa" % (r['prompt'], r['error_mha']))

    # Phase 3: Irrelevant prompts
    print("\n--- Phase 3: Irrelevant Prompts ---")
    irr_results = []
    for prompt in irrelevant:
        E, _ = evaluate_prompt(prompt)
        err = abs(E - E_exact) * 1000
        irr_results.append({
            'prompt': prompt[:40],
            'error_mha': round(err, 4),
        })
        print("  '%s': %.2f mHa" % (prompt[:30], err))

    # Phase 4: Random baseline
    rand_errors = []
    for _ in range(100):
        psi_r = np.random.randn(dim)
        psi_r /= np.linalg.norm(psi_r)
        E_r = float(np.real(psi_r @ H @ psi_r))
        rand_errors.append(abs(E_r - E_exact) * 1000)

    print("\n--- Summary ---")
    best_semantic = combo_results[0]['error_mha']
    worst_semantic = combo_results[-1]['error_mha']
    best_irr = min(r['error_mha'] for r in irr_results)
    rand_best = min(rand_errors)
    rand_mean = float(np.mean(rand_errors))

    print("  Best semantic prompt:    %.2f mHa" % best_semantic)
    print("  Worst semantic prompt:   %.2f mHa" % worst_semantic)
    print("  Best irrelevant prompt:  %.2f mHa" % best_irr)
    print("  Random (best of 100):    %.2f mHa" % rand_best)
    print("  Random (mean):           %.2f mHa" % rand_mean)
    print("  Semantic advantage:      %.1fx" % (rand_best / max(best_semantic, 0.001)))

    # Save
    results = {
        'phase': 'Q155',
        'name': 'Prompt VQE (Words as Parameters)',
        'exact_energy': round(E_exact, 6),
        'best_prompt': combo_results[0],
        'top_5': combo_results[:5],
        'worst_3': combo_results[-3:],
        'irrelevant': irr_results,
        'random_baseline': {
            'best': round(rand_best, 4),
            'mean': round(rand_mean, 4),
        },
        'n_combos_tested': len(combo_results),
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q155_prompt_vqe.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Error distribution of all combo prompts
    ax = axes[0]
    all_errors = [r['error_mha'] for r in combo_results]
    ax.hist(all_errors, bins=30, color='#4CAF50', edgecolor='black', alpha=0.85)
    ax.axvline(best_irr, color='red', ls='--', linewidth=2,
               label='Best irrelevant')
    ax.axvline(rand_best, color='orange', ls='--', linewidth=2,
               label='Best random')
    ax.set_xlabel('Error (mHa)')
    ax.set_ylabel('Count')
    ax.set_title('(a) Prompt VQE Error Distribution\n(%d semantic combos)' %
                 len(combo_results))
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) Subject word ranking
    ax = axes[1]
    subj_names = [r['subject'][:15] for r in subject_results]
    subj_errors = [r['error_mha'] for r in subject_results]
    colors_b = ['#4CAF50' if e < best_irr else '#FF9800' for e in subj_errors]
    ax.barh(range(len(subj_names)), subj_errors, color=colors_b,
            edgecolor='black', alpha=0.85)
    ax.axvline(best_irr, color='red', ls='--', label='Best irrelevant')
    ax.set_yticks(range(len(subj_names)))
    ax.set_yticklabels(subj_names, fontsize=7)
    ax.set_xlabel('Error (mHa)')
    ax.set_title('(b) Which "subject word" is best?')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='x')

    # (c) Semantic vs Irrelevant vs Random
    ax = axes[2]
    categories = ['Best\nsemantic', 'Median\nsemantic', 'Best\nirrelevant',
                   'Random\n(best/100)', 'Random\n(mean)']
    values = [best_semantic, float(np.median(all_errors)), best_irr,
              rand_best, rand_mean]
    colors_c = ['#4CAF50', '#8BC34A', '#F44336', '#FF9800', '#FF5722']
    ax.bar(range(len(categories)), values, color=colors_c,
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Error (mHa)')
    ax.set_title('(c) Prompt Engineering = Hamiltonian Engineering?')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q155: Prompt VQE (Words as Variational Parameters)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q155_prompt_vqe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ155 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
