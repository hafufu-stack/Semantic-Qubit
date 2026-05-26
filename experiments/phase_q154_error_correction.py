# -*- coding: utf-8 -*-
"""
Phase Q154: LLM Quantum Error Correction
==========================================
Quantum error correction: lose some qubits, maintain computation.
LLM has 12 heads x 28 layers = 336 "qubits" of attention.

Test: knock out attention heads and measure how robust the
quantum state (hidden representation) is.

If robust -> LLM has natural error correction (like topological codes)
If fragile -> no error correction, just redundancy
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
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 60)
    print("Phase Q154: LLM Quantum Error Correction")
    print("  (Head Knockout Robustness)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = hidden_size // n_heads

    prompts = [
        "The ground state of hydrogen is",
        "Quantum entanglement between two particles creates",
        "The Hamiltonian of the system describes",
    ]

    # Step 1: Get baseline hidden states (no knockout)
    print("\n--- Baseline (no knockout) ---")
    baselines = {}
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        baselines[prompt] = out.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Step 2: Knockout experiments
    # Zero out specific heads in specific layers
    knockout_configs = [
        ('1 head (layer 0)', [(0, 0)]),
        ('1 head (mid)', [(n_layers // 2, 0)]),
        ('1 head (last)', [(n_layers - 1, 0)]),
        ('3 heads (layer 0)', [(0, h) for h in range(3)]),
        ('6 heads (layer 0)', [(0, h) for h in range(6)]),
        ('All heads (layer 0)', [(0, h) for h in range(n_heads)]),
        ('1 head per layer', [(li, 0) for li in range(n_layers)]),
        ('3 heads per layer', [(li, h) for li in range(n_layers) for h in range(3)]),
        ('6 heads per layer', [(li, h) for li in range(n_layers) for h in range(6)]),
    ]

    all_results = []

    for ko_name, ko_list in knockout_configs:
        # Store original weights to restore later
        saved_weights = {}
        for li, hi in ko_list:
            layer = model.model.layers[li]
            attn = layer.self_attn
            # Zero out the output projection for this head
            start = hi * head_dim
            end = start + head_dim
            key = (li, hi)
            saved_weights[key] = attn.o_proj.weight.data[:, start:end].clone()
            attn.o_proj.weight.data[:, start:end] = 0

        # Measure
        sims = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h_ko = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
            sim = cosine_sim(baselines[prompt], h_ko)
            sims.append(sim)

        # Restore weights
        for (li, hi), w in saved_weights.items():
            layer = model.model.layers[li]
            start = hi * head_dim
            end = start + head_dim
            layer.self_attn.o_proj.weight.data[:, start:end] = w

        avg_sim = float(np.mean(sims))
        n_knocked = len(ko_list)
        pct_knocked = n_knocked / (n_heads * n_layers) * 100

        result = {
            'config': ko_name,
            'n_heads_knocked': n_knocked,
            'pct_knocked': round(pct_knocked, 2),
            'avg_similarity': round(avg_sim, 6),
            'per_prompt': [round(s, 4) for s in sims],
        }
        all_results.append(result)
        print("  %-25s: %3d heads (%.1f%%) -> cos=%.4f" %
              (ko_name, n_knocked, pct_knocked, avg_sim))

    # Threshold analysis
    print("\n--- Error Correction Threshold ---")
    for r in all_results:
        status = "RECOVERED" if r['avg_similarity'] > 0.99 else \
                 "DEGRADED" if r['avg_similarity'] > 0.90 else \
                 "FAILED"
        print("  %-25s: %.4f -> %s" %
              (r['config'], r['avg_similarity'], status))

    # Save
    results = {
        'phase': 'Q154',
        'name': 'LLM Quantum Error Correction',
        'knockout_results': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q154_error_correction.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    names = [r['config'] for r in all_results]
    sims_plot = [r['avg_similarity'] for r in all_results]
    colors = ['#4CAF50' if s > 0.99 else '#FF9800' if s > 0.9 else '#F44336'
              for s in sims_plot]
    ax.barh(range(len(names)), sims_plot, color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0.99, color='green', ls='--', label='Recovery threshold')
    ax.axvline(0.90, color='orange', ls='--', label='Degradation threshold')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Cosine Similarity to Baseline')
    ax.set_title('(a) Head Knockout Robustness')
    ax.set_xlim(0.5, 1.02)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='x')

    ax = axes[1]
    pcts = [r['pct_knocked'] for r in all_results]
    ax.scatter(pcts, sims_plot, s=80, c=colors, edgecolor='black', zorder=3)
    ax.plot(pcts, sims_plot, '--', color='gray', alpha=0.5, zorder=2)
    ax.axhline(0.99, color='green', ls='--', alpha=0.5)
    ax.axhline(0.90, color='orange', ls='--', alpha=0.5)
    ax.set_xlabel('% of heads knocked out')
    ax.set_ylabel('Cosine similarity')
    ax.set_title('(b) Error Correction Curve')
    ax.grid(alpha=0.3)

    plt.suptitle('Q154: LLM Quantum Error Correction (Head Knockout)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q154_error_correction.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ154 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
