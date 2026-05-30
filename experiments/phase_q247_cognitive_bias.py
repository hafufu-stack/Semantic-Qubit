# -*- coding: utf-8 -*-
"""
Phase Q247: Quantum Cognitive Bias
=====================================
Test if the LLM exhibits quantum-like probability interference
in reasoning — the "Linda Problem" and conjunction fallacy.

Quantum Cognition theory: P(A and B) > P(A) happens because
quantum probability amplitudes INTERFERE, unlike classical prob.
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


def get_representation(model, tok, device, text, n_layers):
    inp = tok(text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[n_layers][0, -1, :].float().cpu().numpy()


def quantum_probability(psi_a, psi_b, dim=16):
    """Compute quantum probability with interference term."""
    a = psi_a[:dim] / (np.linalg.norm(psi_a[:dim]) + 1e-10)
    b = psi_b[:dim] / (np.linalg.norm(psi_b[:dim]) + 1e-10)
    # Quantum: |a+b|^2 = |a|^2 + |b|^2 + 2*Re(a.b*) [interference]
    p_a = float(np.linalg.norm(a)**2)
    p_b = float(np.linalg.norm(b)**2)
    interference = 2 * float(np.dot(a, b))
    p_quantum = float(np.linalg.norm(a + b)**2)
    return p_a, p_b, interference, p_quantum


def main():
    print("=" * 60)
    print("Phase Q247: Quantum Cognitive Bias")
    print("  (Does LLM reason with quantum probability?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    # Classic cognitive bias scenarios
    scenarios = [
        {
            'name': 'Linda Problem',
            'context': 'Linda is a single woman who studied philosophy. She is outspoken about social justice.',
            'A': 'Linda is a bank teller.',
            'B': 'Linda is a feminist.',
            'AB': 'Linda is a bank teller and a feminist.',
        },
        {
            'name': 'Bill Problem',
            'context': 'Bill is intelligent but unimaginative. He was good at math in school.',
            'A': 'Bill is an accountant.',
            'B': 'Bill plays jazz music.',
            'AB': 'Bill is an accountant who plays jazz music.',
        },
        {
            'name': 'Terrorism',
            'context': 'Recent news reports about international tensions and security threats.',
            'A': 'There will be a flood somewhere in North America next year.',
            'B': 'An earthquake in California causes a flood.',
            'AB': 'An earthquake in California causes a flood that kills many people.',
        },
        {
            'name': 'Medical Diagnosis',
            'context': 'A 55 year old patient with chest pain and shortness of breath.',
            'A': 'The patient has a heart condition.',
            'B': 'The patient had a heart attack.',
            'AB': 'The patient had a heart attack leading to heart failure.',
        },
    ]

    all_results = []
    conjunction_violations = 0

    for sc in scenarios:
        print("\n--- %s ---" % sc['name'])
        # Get representations
        ctx = sc['context']
        rep_A = get_representation(model, tok, device, ctx + " " + sc['A'], n_layers)
        rep_B = get_representation(model, tok, device, ctx + " " + sc['B'], n_layers)
        rep_AB = get_representation(model, tok, device, ctx + " " + sc['AB'], n_layers)
        rep_ctx = get_representation(model, tok, device, ctx, n_layers)

        # Cosine similarities as "probability proxies"
        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

        sim_A = cos_sim(rep_ctx, rep_A)
        sim_B = cos_sim(rep_ctx, rep_B)
        sim_AB = cos_sim(rep_ctx, rep_AB)

        # Conjunction fallacy: sim_AB > sim_A?
        fallacy = sim_AB > sim_A
        if fallacy:
            conjunction_violations += 1

        # Quantum interference
        p_a, p_b, interference, p_quantum = quantum_probability(
            rep_A - rep_ctx, rep_B - rep_ctx)

        print("  sim(A)=%.4f, sim(B)=%.4f, sim(AB)=%.4f" % (sim_A, sim_B, sim_AB))
        print("  Conjunction fallacy: %s" % ("YES" if fallacy else "NO"))
        print("  Interference term: %.4f" % interference)

        all_results.append({
            'name': sc['name'],
            'sim_A': round(sim_A, 4), 'sim_B': round(sim_B, 4), 'sim_AB': round(sim_AB, 4),
            'conjunction_fallacy': bool(fallacy),
            'interference': round(interference, 4),
        })

    n_scenarios = len(scenarios)
    fallacy_rate = conjunction_violations / n_scenarios * 100

    if fallacy_rate > 50:
        verdict = "QUANTUM COGNITION: %.0f%% conjunction fallacy rate (%d/%d)" % (
            fallacy_rate, conjunction_violations, n_scenarios)
    elif fallacy_rate > 0:
        verdict = "PARTIAL QUANTUM COGNITION: %.0f%% fallacy rate" % fallacy_rate
    else:
        verdict = "CLASSICAL REASONING: 0%% fallacy"

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q247', 'name': 'Quantum Cognitive Bias',
        'scenarios': all_results,
        'summary': {'fallacy_rate': round(fallacy_rate, 1),
                     'n_violations': conjunction_violations,
                     'avg_interference': round(np.mean([r['interference'] for r in all_results]), 4),
                     'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q247_cognitive_bias.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(n_scenarios)
    w = 0.25
    ax.bar(x - w, [r['sim_A'] for r in all_results], w, label='P(A)', color='#2196F3', edgecolor='black')
    ax.bar(x, [r['sim_AB'] for r in all_results], w, label='P(A^B)', color='#E91E63', edgecolor='black')
    ax.bar(x + w, [r['sim_B'] for r in all_results], w, label='P(B)', color='#4CAF50', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([r['name'][:12] for r in all_results], fontsize=8)
    ax.set_ylabel('Similarity (probability proxy)')
    ax.set_title('(a) Conjunction Fallacy Test'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ints = [r['interference'] for r in all_results]
    colors = ['#E91E63' if i > 0 else '#2196F3' for i in ints]
    ax.bar(range(n_scenarios), ints, color=colors, edgecolor='black')
    ax.axhline(0, color='black', ls='--')
    ax.set_xticks(range(n_scenarios)); ax.set_xticklabels([r['name'][:12] for r in all_results], fontsize=8)
    ax.set_ylabel('Interference Term'); ax.set_title('(b) Quantum Interference')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q247: Quantum Cognitive Bias\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q247_cognitive_bias.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ247 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
