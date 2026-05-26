# -*- coding: utf-8 -*-
"""Phase Q105: Quantum Advantage in NLP - Interference-Enhanced Prediction
THE PRACTICAL APPLICATION: Can quantum interference between S-Qubits
improve actual next-token prediction? If so, this proves that
quantum-like processing gives LLMs a computational advantage.
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def measure_interference_advantage(model, tokenizer, num_layers):
    """Test if constructive interference between two semantic concepts
    produces MORE accurate predictions than either concept alone."""

    # Test cases: two related concepts that should constructively interfere
    test_cases = [
        {
            'name': 'Physics+Math',
            'prompt_a': "Physics is the study of",
            'prompt_b': "Mathematics provides the foundation for",
            'combined': "The mathematical physics of quantum mechanics describes",
            'expected_domain': ['particles', 'waves', 'energy', 'fields',
                               'equations', 'theory', 'nature'],
        },
        {
            'name': 'Brain+Computer',
            'prompt_a': "The human brain processes information through",
            'prompt_b': "Computer neural networks learn patterns by",
            'combined': "Artificial intelligence mirrors the brain by learning",
            'expected_domain': ['neural', 'learning', 'patterns', 'connections',
                               'data', 'neurons', 'networks'],
        },
        {
            'name': 'Evolution+DNA',
            'prompt_a': "Biological evolution selects for traits that",
            'prompt_b': "DNA contains the genetic instructions for",
            'combined': "Evolutionary genetics shows how DNA mutations drive",
            'expected_domain': ['adaptation', 'survival', 'species', 'fitness',
                                'diversity', 'selection', 'change'],
        },
        {
            'name': 'Music+Emotion',
            'prompt_a': "Musical harmony creates feelings of",
            'prompt_b': "Human emotions are triggered by",
            'combined': "The emotional power of music comes from its ability to",
            'expected_domain': ['move', 'inspire', 'evoke', 'express',
                                'connect', 'resonate', 'touch'],
        },
    ]

    d_model = model.config.hidden_size

    results = []
    for tc in test_cases:
        print("  Testing: %s" % tc['name'])

        # Get reference output for combined prompt (ground truth)
        inputs_c = tokenizer(tc['combined'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            out_c = model(**inputs_c)
            logits_c = out_c.logits[0, -1, :].cpu().float()
            probs_c = torch.softmax(logits_c, dim=-1).numpy()

        # Get output for prompt A alone
        inputs_a = tokenizer(tc['prompt_a'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            out_a = model(**inputs_a)
            logits_a = out_a.logits[0, -1, :].cpu().float()
            probs_a = torch.softmax(logits_a, dim=-1).numpy()

        # Get output for prompt B alone
        inputs_b = tokenizer(tc['prompt_b'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            out_b = model(**inputs_b)
            logits_b = out_b.logits[0, -1, :].cpu().float()
            probs_b = torch.softmax(logits_b, dim=-1).numpy()

        # Classical mixture: average of individual distributions
        classical_mix = (probs_a + probs_b) / 2

        # Quantum interference: sqrt of product (constructive interference)
        # Geometric mean models interference term
        quantum_mix = np.sqrt(probs_a * probs_b + 1e-20)
        quantum_mix /= quantum_mix.sum()

        # Measure which mix is closer to the combined output (ground truth)
        # KL divergence: lower is better
        eps = 1e-10
        kl_classical = np.sum(probs_c * np.log((probs_c + eps) / (classical_mix + eps)))
        kl_quantum = np.sum(probs_c * np.log((probs_c + eps) / (quantum_mix + eps)))

        # Jensen-Shannon divergence (symmetric)
        m_cl = (probs_c + classical_mix) / 2
        m_qu = (probs_c + quantum_mix) / 2
        js_classical = 0.5 * np.sum(probs_c * np.log((probs_c + eps) / (m_cl + eps))) + \
                        0.5 * np.sum(classical_mix * np.log((classical_mix + eps) / (m_cl + eps)))
        js_quantum = 0.5 * np.sum(probs_c * np.log((probs_c + eps) / (m_qu + eps))) + \
                      0.5 * np.sum(quantum_mix * np.log((quantum_mix + eps) / (m_qu + eps)))

        # Top-k overlap with ground truth
        k = 20
        top_c = set(np.argsort(probs_c)[-k:])
        top_cl = set(np.argsort(classical_mix)[-k:])
        top_qu = set(np.argsort(quantum_mix)[-k:])
        overlap_classical = len(top_c & top_cl) / k
        overlap_quantum = len(top_c & top_qu) / k

        advantage = (js_classical - js_quantum) / (js_classical + 1e-10) * 100

        result = {
            'name': tc['name'],
            'kl_classical': float(kl_classical),
            'kl_quantum': float(kl_quantum),
            'js_classical': float(js_classical),
            'js_quantum': float(js_quantum),
            'top20_classical': float(overlap_classical),
            'top20_quantum': float(overlap_quantum),
            'advantage_pct': float(advantage),
            'quantum_wins': js_quantum < js_classical,
        }
        results.append(result)
        print("    JS(classical)=%.4f, JS(quantum)=%.4f -> %s (%.1f%%)" %
              (js_classical, js_quantum,
               'QUANTUM WINS' if result['quantum_wins'] else 'Classical wins',
               advantage))

    return results


def main():
    print("=" * 60)
    print("Phase Q105: Quantum Advantage in NLP")
    print("  Does interference improve prediction?")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    results = measure_interference_advantage(model, tokenizer, num_layers)

    # Analysis
    n_quantum_wins = sum(1 for r in results if r['quantum_wins'])
    mean_advantage = np.mean([r['advantage_pct'] for r in results])

    print("\n  === Quantum Advantage Results ===")
    print("  Quantum wins: %d/%d tasks" % (n_quantum_wins, len(results)))
    print("  Mean advantage: %.1f%%" % mean_advantage)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) JS divergence comparison
    ax = axes[0]
    names = [r['name'] for r in results]
    js_cl = [r['js_classical'] for r in results]
    js_qu = [r['js_quantum'] for r in results]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, js_cl, w, label='Classical (average)',
                   color='#FF5722', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + w/2, js_qu, w, label='Quantum (interference)',
                   color='#2196F3', alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_ylabel('JS divergence from ground truth\n(lower = better)', fontsize=10)
    ax.set_title('(a) Prediction Quality\nClassical vs Quantum Mix',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (b) Advantage
    ax = axes[1]
    advantages = [r['advantage_pct'] for r in results]
    colors = ['#4CAF50' if a > 0 else '#F44336' for a in advantages]
    ax.bar(names, advantages, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Quantum advantage (%)', fontsize=11)
    ax.set_title('(b) Quantum Advantage\nGreen = Quantum Wins',
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.grid(alpha=0.3, axis='y')

    # (c) Summary
    ax = axes[2]
    ax.text(0.5, 0.65,
            'QUANTUM\nADVANTAGE',
            ha='center', va='center', fontsize=22, fontweight='bold',
            color='#4CAF50' if n_quantum_wins > len(results) / 2 else '#FF5722',
            transform=ax.transAxes)
    ax.text(0.5, 0.35,
            'Quantum wins: %d/%d tasks\n'
            'Mean advantage: %.1f%%\n\n'
            'Interference-based mixing\n'
            '%s classical averaging' % (
                n_quantum_wins, len(results),
                mean_advantage,
                'OUTPERFORMS' if mean_advantage > 0 else 'does not beat'),
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) Verdict', fontsize=12, fontweight='bold')

    plt.suptitle('Q105: Can Quantum Interference Improve Language Prediction?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q105_nlp_advantage.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    final_results = {
        'phase': 'Q105', 'name': 'Quantum Advantage in NLP',
        'n_quantum_wins': n_quantum_wins,
        'total_tasks': len(results),
        'mean_advantage_pct': float(mean_advantage),
        'tasks': results,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q105_nlp_advantage.json')
    with open(res_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return final_results


if __name__ == '__main__':
    main()
