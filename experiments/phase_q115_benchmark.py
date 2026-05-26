# -*- coding: utf-8 -*-
"""
Phase Q115: Real-World Quantum Benchmark
=========================================
The "Grok-Proof" benchmark: S-Qubit techniques applied to
practical NLP tasks with measurable real-world impact.

Tests:
1. Semantic clustering quality (vs classical k-means)
2. Analogy completion (king-man+woman=queen)
3. Zero-shot classification accuracy
4. Context disambiguation
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


def main():
    print("=" * 60)
    print("Phase Q115: Real-World Quantum Benchmark")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # ===== Test 1: Quantum vs Classical Semantic Clustering =====
    print("\n--- Test 1: Semantic Clustering ---")
    # Cluster words into categories using S-Qubit interference vs cosine
    categories = {
        'animals': ['cat', 'dog', 'elephant', 'whale', 'eagle'],
        'fruits': ['apple', 'banana', 'grape', 'mango', 'peach'],
        'colors': ['red', 'blue', 'green', 'yellow', 'purple'],
        'metals': ['gold', 'silver', 'iron', 'copper', 'zinc'],
    }

    # Get embeddings for all words
    all_words = []
    all_labels = []
    all_embeds = []
    for cat, words in categories.items():
        for w in words:
            all_words.append(w)
            all_labels.append(cat)
            # Get hidden state from model
            prompt = "The word '%s' means" % w
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu()
            all_embeds.append(h)

    embeds = torch.stack(all_embeds)
    n_words = len(all_words)

    # Method 1: Classical cosine similarity clustering
    embeds_norm = torch.nn.functional.normalize(embeds, dim=-1)
    cos_sim = embeds_norm @ embeds_norm.T

    # Method 2: S-Qubit interference clustering
    # Apply phase rotation based on semantic distance
    phases = torch.atan2(embeds[:, 1::2].sum(-1), embeds[:, ::2].sum(-1))
    phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)
    interference_sim = torch.cos(phase_diff)  # Interference pattern

    # Evaluate: for each word, does its nearest neighbor share the same category?
    classical_correct = 0
    quantum_correct = 0
    for i in range(n_words):
        # Classical: nearest by cosine
        cos_row = cos_sim[i].clone()
        cos_row[i] = -1  # Exclude self
        nn_classical = cos_row.argmax().item()

        # Quantum: nearest by interference
        int_row = interference_sim[i].clone()
        int_row[i] = -1
        nn_quantum = int_row.argmax().item()

        if all_labels[nn_classical] == all_labels[i]:
            classical_correct += 1
        if all_labels[nn_quantum] == all_labels[i]:
            quantum_correct += 1

    classical_acc = classical_correct / n_words
    quantum_acc = quantum_correct / n_words
    print("  Classical NN accuracy: %.1f%% (%d/%d)" %
          (classical_acc * 100, classical_correct, n_words))
    print("  Quantum NN accuracy: %.1f%% (%d/%d)" %
          (quantum_acc * 100, quantum_correct, n_words))

    # ===== Test 2: Analogy Completion =====
    print("\n--- Test 2: Analogy Completion ---")
    analogies = [
        ('king', 'man', 'woman', 'queen'),
        ('Paris', 'France', 'Japan', 'Tokyo'),
        ('big', 'small', 'hot', 'cold'),
        ('doctor', 'hospital', 'teacher', 'school'),
        ('day', 'night', 'summer', 'winter'),
    ]

    analogy_results = []
    for a, b, c, expected in analogies:
        # Get embeddings
        words_here = [a, b, c, expected]
        embs = {}
        for w in words_here:
            prompt = "The word '%s' represents" % w
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            embs[w] = out.hidden_states[-1][0, -1, :].float()

        # Classical: a - b + c should be close to expected
        classical_vec = embs[a] - embs[b] + embs[c]
        classical_cos = torch.nn.functional.cosine_similarity(
            classical_vec.unsqueeze(0), embs[expected].unsqueeze(0)).item()

        # Quantum: interference-based analogy
        # Phase of (a, b) relationship applied to c
        phase_ab = torch.atan2(
            (embs[a] - embs[b])[1::2].mean(),
            (embs[a] - embs[b])[::2].mean()).item()
        quantum_vec = embs[c] * np.cos(phase_ab) + \
                      torch.roll(embs[c], 1, dims=-1) * np.sin(phase_ab)
        quantum_cos = torch.nn.functional.cosine_similarity(
            quantum_vec.unsqueeze(0), embs[expected].unsqueeze(0)).item()

        analogy_results.append({
            'analogy': '%s:%s::%s:?' % (a, b, c),
            'expected': expected,
            'classical_cos': round(classical_cos, 4),
            'quantum_cos': round(quantum_cos, 4),
            'quantum_wins': quantum_cos > classical_cos
        })
        print("  %s:%s::%s:%s  classical=%.3f  quantum=%.3f  %s" %
              (a, b, c, expected, classical_cos, quantum_cos,
               "Q wins" if quantum_cos > classical_cos else "C wins"))

    n_quantum_wins = sum(1 for r in analogy_results if r['quantum_wins'])

    # ===== Test 3: Zero-Shot Classification =====
    print("\n--- Test 3: Zero-Shot Classification ---")
    test_sentences = [
        ("The stock market crashed today", "finance"),
        ("New species of frog discovered in Amazon", "science"),
        ("Team wins championship in overtime", "sports"),
        ("New album tops billboard charts", "entertainment"),
        ("Vaccine shows 95% efficacy in trials", "health"),
        ("Hurricane approaching the coast", "weather"),
    ]

    # Classify using S-Qubit phase matching
    label_set = ['finance', 'science', 'sports', 'entertainment', 'health', 'weather']
    label_embeds = {}
    for label in label_set:
        prompt = "This text is about %s:" % label
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        label_embeds[label] = out.hidden_states[-1][0, -1, :].float()

    classical_correct_zs = 0
    quantum_correct_zs = 0

    for sentence, true_label in test_sentences:
        inp = tok(sentence, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        sent_emb = out.hidden_states[-1][0, -1, :].float()

        # Classical: cosine similarity
        classical_scores = {}
        for label, lemb in label_embeds.items():
            classical_scores[label] = torch.nn.functional.cosine_similarity(
                sent_emb.unsqueeze(0), lemb.unsqueeze(0)).item()
        classical_pred = max(classical_scores, key=classical_scores.get)

        # Quantum: interference-based
        quantum_scores = {}
        for label, lemb in label_embeds.items():
            # Phase difference
            phase = torch.atan2(
                (sent_emb - lemb)[1::2].mean(),
                (sent_emb - lemb)[::2].mean()).item()
            # Constructive interference at phase=0
            quantum_scores[label] = np.cos(phase) * abs(classical_scores[label])
        quantum_pred = max(quantum_scores, key=quantum_scores.get)

        if classical_pred == true_label:
            classical_correct_zs += 1
        if quantum_pred == true_label:
            quantum_correct_zs += 1

    classical_acc_zs = classical_correct_zs / len(test_sentences)
    quantum_acc_zs = quantum_correct_zs / len(test_sentences)
    print("  Classical accuracy: %.1f%%" % (classical_acc_zs * 100))
    print("  Quantum accuracy: %.1f%%" % (quantum_acc_zs * 100))

    # ===== Save Results =====
    results = {
        'phase': 'Q115',
        'name': 'Real-World Quantum Benchmark',
        'clustering': {
            'classical_accuracy': round(classical_acc, 4),
            'quantum_accuracy': round(quantum_acc, 4),
            'quantum_advantage': round(quantum_acc - classical_acc, 4)
        },
        'analogy': {
            'quantum_wins': n_quantum_wins,
            'total': len(analogies),
            'results': analogy_results
        },
        'zero_shot': {
            'classical_accuracy': round(classical_acc_zs, 4),
            'quantum_accuracy': round(quantum_acc_zs, 4),
            'quantum_advantage': round(quantum_acc_zs - classical_acc_zs, 4)
        },
        'overall_quantum_advantage': round(
            (quantum_acc - classical_acc +
             quantum_acc_zs - classical_acc_zs +
             (n_quantum_wins / len(analogies) - 0.5)) / 3, 4),
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q115_benchmark.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Clustering
    ax = axes[0]
    ax.bar(['Classical', 'Quantum'], [classical_acc * 100, quantum_acc * 100],
           color=['#FF5722', '#2196F3'], edgecolor='black', alpha=0.85)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(a) Semantic Clustering\n(nearest-neighbor)')
    ax.set_ylim(0, 105)
    for i, v in enumerate([classical_acc, quantum_acc]):
        ax.text(i, v * 100 + 2, '%.1f%%' % (v * 100), ha='center',
                fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # (b) Analogy
    ax = axes[1]
    c_vals = [r['classical_cos'] for r in analogy_results]
    q_vals = [r['quantum_cos'] for r in analogy_results]
    x = np.arange(len(analogy_results))
    ax.bar(x - 0.2, c_vals, 0.4, label='Classical', color='#FF5722', alpha=0.85)
    ax.bar(x + 0.2, q_vals, 0.4, label='Quantum', color='#2196F3', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r['analogy'][:10] for r in analogy_results],
                       fontsize=7, rotation=15)
    ax.set_ylabel('Cosine to expected')
    ax.set_title('(b) Analogy Completion\n(Q wins: %d/%d)' %
                 (n_quantum_wins, len(analogies)))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # (c) Zero-shot
    ax = axes[2]
    ax.bar(['Classical', 'Quantum'],
           [classical_acc_zs * 100, quantum_acc_zs * 100],
           color=['#FF5722', '#2196F3'], edgecolor='black', alpha=0.85)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(c) Zero-Shot Classification')
    ax.set_ylim(0, 105)
    for i, v in enumerate([classical_acc_zs, quantum_acc_zs]):
        ax.text(i, v * 100 + 2, '%.1f%%' % (v * 100), ha='center',
                fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q115: Real-World Quantum Benchmark (Grok-Proof)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q115_benchmark.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ115 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
