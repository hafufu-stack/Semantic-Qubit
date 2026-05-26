# -*- coding: utf-8 -*-
"""
Phase Q172: Quantum Teleportation Protocol
============================================
Full quantum teleportation using LLM:
1. Alice has a "quantum state" (hidden state from prompt A)
2. Alice creates an entangled pair (two prompts sharing meaning)
3. Alice "measures" (dot product) and sends classical bits (2 floats)
4. Bob uses classical bits + his half to reconstruct Alice's state
5. Measure teleportation fidelity

This tests if LLM's semantic entanglement can be used for
state transfer - a key primitive for quantum communication.
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


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 60)
    print("Phase Q172: Quantum Teleportation Protocol")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    hidden_size = model.config.hidden_size

    def get_hidden(prompt):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Teleportation scenarios
    scenarios = [
        {
            'name': 'Physics',
            'alice_state': "Quantum ground state energy of hydrogen",
            'entangled_a': "The wave function describes particle behavior",
            'entangled_b': "Quantum mechanics governs atomic structure",
            'bob_unrelated': "The stock market crashed yesterday",
        },
        {
            'name': 'Chemistry',
            'alice_state': "Chemical bond between carbon atoms",
            'entangled_a': "Molecular orbital theory explains bonding",
            'entangled_b': "Electron sharing creates covalent bonds",
            'bob_unrelated': "My favorite color is blue",
        },
        {
            'name': 'Code',
            'alice_state': "def sort(arr): return sorted(arr)",
            'entangled_a': "Algorithms process data structures efficiently",
            'entangled_b': "Computer science uses mathematical logic",
            'bob_unrelated': "The weather is sunny today",
        },
        {
            'name': 'Literature',
            'alice_state': "To be or not to be that is the question",
            'entangled_a': "Shakespeare wrote plays about human nature",
            'entangled_b': "Classical literature explores existential themes",
            'bob_unrelated': "The pizza was delicious",
        },
    ]

    all_results = []

    for scenario in scenarios:
        name = scenario['name']
        print("\n--- %s ---" % name)

        # Step 1: Alice's state to teleport
        psi_alice = get_hidden(scenario['alice_state'])

        # Step 2: Entangled pair (semantic)
        psi_entA = get_hidden(scenario['entangled_a'])
        psi_entB = get_hidden(scenario['entangled_b'])

        # Step 3: Alice's "Bell measurement" (project onto entangled basis)
        # Classical information: overlap coefficients
        alpha = cosine_sim(psi_alice, psi_entA)
        beta = np.dot(psi_alice, psi_entA) / (np.linalg.norm(psi_entA)**2 + 1e-10)

        # Step 4: Bob's reconstruction
        # Bob applies "correction" using classical bits
        psi_bob_reconstructed = beta * psi_entB + (1 - abs(alpha)) * psi_entB
        psi_bob_reconstructed /= np.linalg.norm(psi_bob_reconstructed) + 1e-10

        # Step 5: Teleportation fidelity
        fidelity = abs(cosine_sim(psi_alice, psi_bob_reconstructed))

        # Baselines
        fid_direct = abs(cosine_sim(psi_alice, psi_entB))
        fid_unrelated = abs(cosine_sim(psi_alice,
                                        get_hidden(scenario['bob_unrelated'])))

        # Random baseline
        rand_fids = []
        for _ in range(100):
            r = np.random.randn(hidden_size)
            r /= np.linalg.norm(r)
            rand_fids.append(abs(cosine_sim(psi_alice, r)))
        fid_random = float(np.mean(rand_fids))

        # Entanglement strength
        ent_strength = abs(cosine_sim(psi_entA, psi_entB))

        result = {
            'scenario': name,
            'teleportation_fidelity': round(fidelity, 4),
            'direct_fidelity': round(fid_direct, 4),
            'unrelated_fidelity': round(fid_unrelated, 4),
            'random_fidelity': round(fid_random, 4),
            'entanglement_strength': round(ent_strength, 4),
            'alpha': round(alpha, 4),
            'advantage_vs_random': round(fidelity / max(fid_random, 0.001), 2),
        }
        all_results.append(result)

        print("  Teleportation F: %.4f" % fidelity)
        print("  Direct (no teleport): %.4f" % fid_direct)
        print("  Unrelated: %.4f" % fid_unrelated)
        print("  Random: %.4f" % fid_random)
        print("  Entanglement: %.4f" % ent_strength)
        print("  Advantage: %.1fx vs random" % (fidelity / max(fid_random, 0.001)))

    # Summary
    print("\n--- Teleportation Summary ---")
    avg_fid = float(np.mean([r['teleportation_fidelity'] for r in all_results]))
    avg_rand = float(np.mean([r['random_fidelity'] for r in all_results]))
    avg_adv = float(np.mean([r['advantage_vs_random'] for r in all_results]))
    print("  Avg teleportation fidelity: %.4f" % avg_fid)
    print("  Avg random fidelity: %.4f" % avg_rand)
    print("  Avg advantage: %.1fx" % avg_adv)

    # Save
    results = {
        'phase': 'Q172',
        'name': 'Quantum Teleportation Protocol',
        'scenarios': all_results,
        'summary': {
            'avg_fidelity': round(avg_fid, 4),
            'avg_random': round(avg_rand, 4),
            'avg_advantage': round(avg_adv, 2),
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q172_teleportation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = [r['scenario'] for r in all_results]
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5*w, [r['teleportation_fidelity'] for r in all_results],
           w, color='#4CAF50', label='Teleportation', alpha=0.85)
    ax.bar(x - 0.5*w, [r['direct_fidelity'] for r in all_results],
           w, color='#2196F3', label='Direct', alpha=0.85)
    ax.bar(x + 0.5*w, [r['unrelated_fidelity'] for r in all_results],
           w, color='#FF9800', label='Unrelated', alpha=0.85)
    ax.bar(x + 1.5*w, [r['random_fidelity'] for r in all_results],
           w, color='#F44336', label='Random', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Fidelity')
    ax.set_title('(a) Teleportation Fidelity')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    advs = [r['advantage_vs_random'] for r in all_results]
    ax.bar(range(len(names)), advs, color='#9C27B0', edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='red', ls='--', label='No advantage')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel('Advantage (x)')
    ax.set_title('(b) Advantage Over Random')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    ent_strengths = [r['entanglement_strength'] for r in all_results]
    tel_fids = [r['teleportation_fidelity'] for r in all_results]
    ax.scatter(ent_strengths, tel_fids, s=150, c='#E91E63', edgecolors='black',
               zorder=5)
    for i, n in enumerate(names):
        ax.annotate(n, (ent_strengths[i], tel_fids[i]),
                    fontsize=8, ha='left', va='bottom')
    ax.set_xlabel('Entanglement Strength')
    ax.set_ylabel('Teleportation Fidelity')
    ax.set_title('(c) Entanglement -> Teleportation')
    ax.grid(alpha=0.3)

    plt.suptitle('Q172: Quantum Teleportation via Semantic Entanglement',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q172_teleportation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ172 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
