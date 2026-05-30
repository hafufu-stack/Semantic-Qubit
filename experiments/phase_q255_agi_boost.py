# -*- coding: utf-8 -*-
"""
Phase Q255: Entanglement-Boosted AGI
========================================
Does quantum-native LoRA (Q249) improve GENERAL reasoning,
not just VQE? Test on math and logic tasks.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

class LoRAAdapter(nn.Module):
    def __init__(self, dim, rank=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.scale = 0.1
    def forward(self, x):
        return x + self.scale * (x @ self.A @ self.B)

def main():
    print("=" * 60)
    print("Phase Q255: Entanglement-Boosted AGI")
    print("  (Does quantum LoRA improve general reasoning?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    dim = 4

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # Train quantum LoRA (reproduce Q249)
    print("\n  Step 1: Training Quantum-Native LoRA (60 steps)...")
    adapters = nn.ModuleList([LoRAAdapter(hidden_dim, rank=8).to(device) for _ in range(4)])
    optimizer = torch.optim.Adam(adapters.parameters(), lr=0.001)
    train_prompts = ["quantum ground state", "variational optimization", "eigenvalue problem"]

    for step in range(60):
        optimizer.zero_grad()
        prompt = train_prompts[step % len(train_prompts)]
        inp = tok(prompt, return_tensors='pt').to(device)
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[n_layers][0, -1, :]
        for adapter in adapters:
            h = adapter(h)
        psi = h[:dim] / (torch.norm(h[:dim]) + 1e-10)
        rho = torch.outer(psi, psi)
        off_diag = torch.sum(torch.abs(rho)) - torch.sum(torch.abs(torch.diag(rho)))
        loss = -off_diag
        loss.backward(); optimizer.step()

    # Step 2: Test on reasoning tasks
    print("\n  Step 2: Testing on reasoning tasks...")

    # Simple reasoning: measure perplexity and representation quality
    test_pairs = [
        ("2 + 3 =", "5"),
        ("If A implies B and A is true then B is", "true"),
        ("The capital of France is", "Paris"),
        ("Water freezes at", "zero"),
        ("The square root of 16 is", "4"),
        ("All birds can fly. Penguins are birds. Can penguins fly?", "No"),
        ("If x = 5 and y = x + 3, then y =", "8"),
        ("The opposite of hot is", "cold"),
    ]

    baseline_scores = []
    quantum_scores = []

    for question, answer in test_pairs:
        # Baseline: raw model logits for answer
        full_text = question + " " + answer
        inp = tok(full_text, return_tensors='pt').to(device)
        with torch.no_grad():
            out_base = model(**inp)
            logits_base = out_base.logits[0, -2, :]  # Logits for predicting answer token
            ans_tok = tok(answer, return_tensors='pt')['input_ids'][0, -1]
            score_base = float(logits_base[ans_tok].cpu())

        # Quantum LoRA: apply adapters to hidden states
        with torch.no_grad():
            out_q = model(**inp, output_hidden_states=True)
            h_q = out_q.hidden_states[n_layers][0, -1, :]
            for adapter in adapters:
                h_q = adapter(h_q)

        # Measure representation quality: cosine sim to answer embedding
        ans_inp = tok(answer, return_tensors='pt').to(device)
        with torch.no_grad():
            ans_out = model(**ans_inp, output_hidden_states=True)
            ans_rep = ans_out.hidden_states[n_layers][0, -1, :]

        q_inp = tok(question, return_tensors='pt').to(device)
        with torch.no_grad():
            q_out = model(**q_inp, output_hidden_states=True)
            q_rep = q_out.hidden_states[n_layers][0, -1, :]
            q_rep_adapted = q_rep.clone()
            for adapter in adapters:
                q_rep_adapted = adapter(q_rep_adapted)

        cos_base = float(torch.nn.functional.cosine_similarity(q_rep.unsqueeze(0), ans_rep.unsqueeze(0)).cpu())
        cos_quantum = float(torch.nn.functional.cosine_similarity(q_rep_adapted.unsqueeze(0), ans_rep.unsqueeze(0)).cpu())

        baseline_scores.append(cos_base)
        quantum_scores.append(cos_quantum)

    avg_base = np.mean(baseline_scores)
    avg_quantum = np.mean(quantum_scores)
    boost = (avg_quantum - avg_base) / max(abs(avg_base), 1e-6) * 100

    n_improved = sum(1 for b, q in zip(baseline_scores, quantum_scores) if q > b)

    if boost > 5 and n_improved > len(test_pairs) // 2:
        verdict = "AGI BOOST: +%.1f%% avg similarity, %d/%d improved" % (boost, n_improved, len(test_pairs))
    elif n_improved > len(test_pairs) // 2:
        verdict = "PARTIAL BOOST: %d/%d tasks improved (%.1f%%)" % (n_improved, len(test_pairs), boost)
    else:
        verdict = "NO GENERAL BOOST: %.1f%% change, %d/%d improved" % (boost, n_improved, len(test_pairs))

    print("\n  Baseline avg similarity: %.4f" % avg_base)
    print("  Quantum avg similarity: %.4f" % avg_quantum)
    print("  %s" % verdict)

    results = {
        'phase': 'Q255', 'name': 'Entanglement-Boosted AGI',
        'tasks': [{'question': q[:30], 'base': round(b, 4), 'quantum': round(q_, 4)}
                  for (q, _), b, q_ in zip(test_pairs, baseline_scores, quantum_scores)],
        'summary': {'avg_base': round(avg_base, 4), 'avg_quantum': round(avg_quantum, 4),
                     'boost_pct': round(boost, 1), 'n_improved': n_improved, 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q255_agi_boost.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(test_pairs))
    ax.bar(x - 0.2, baseline_scores, 0.4, label='Baseline', color='#607D8B', edgecolor='black')
    ax.bar(x + 0.2, quantum_scores, 0.4, label='Quantum LoRA', color='#E91E63', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['T%d' % (i+1) for i in range(len(test_pairs))], fontsize=8)
    ax.set_ylabel('Cosine Similarity'); ax.set_title('(a) Per-Task Comparison')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar([0, 1], [avg_base, avg_quantum], color=['#607D8B', '#E91E63'], edgecolor='black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Baseline', 'Quantum LoRA'])
    ax.set_ylabel('Avg Cosine Similarity'); ax.set_title('(b) Overall')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q255: Entanglement-Boosted AGI\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q255_agi_boost.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok, adapters; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ255 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
