# -*- coding: utf-8 -*-
"""
Phase Q113: Semantic Bose-Einstein Condensation (BEC)
=====================================================
Tests whether lowering LLM "temperature" to near-zero causes
tokens to collapse into a single macroscopic quantum state
(semantic BEC).

Physical BEC: below critical temperature T_c, bosons occupy
the ground state macroscopically. We test the semantic analogue.
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
    print("Phase Q113: Semantic Bose-Einstein Condensation")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # Test prompts with multiple semantic tokens
    prompts = [
        "The universe is vast and infinite, containing galaxies, stars, planets, and moons",
        "Mathematics includes algebra, geometry, calculus, topology, and number theory",
        "Music encompasses melody, harmony, rhythm, dynamics, and timbre",
    ]

    # Temperature sweep: from hot to cold
    temperatures = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]

    all_condensation = []

    for pi, prompt in enumerate(prompts):
        print("\n  Prompt %d: %s..." % (pi + 1, prompt[:40]))
        inp = tok(prompt, return_tensors='pt').to(device)
        n_tokens = inp['input_ids'].shape[1]

        temp_results = []

        for temp in temperatures:
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            # Get final layer hidden states for ALL tokens
            h_final = out.hidden_states[-1][0, :, :].float()  # (seq, hidden)

            # Apply temperature scaling to logits
            logits = out.logits[0, :, :]  # (seq, vocab)
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)

            # Measure 1: Entropy of probability distribution (per token)
            entropy_per_token = -(probs * (probs + 1e-10).log()).sum(-1)
            mean_entropy = entropy_per_token.mean().item()

            # Measure 2: Ground state occupation
            # What fraction of probability is in the top-1 token?
            top1_probs = probs.max(dim=-1).values
            ground_state_frac = top1_probs.mean().item()

            # Measure 3: Condensation = how similar are all token states?
            # Pairwise cosine similarity between all token hidden states
            h_norm = torch.nn.functional.normalize(h_final, dim=-1)
            sim_matrix = h_norm @ h_norm.T
            # Off-diagonal mean
            mask = ~torch.eye(n_tokens, dtype=torch.bool, device=device)
            mean_similarity = sim_matrix[mask].mean().item()

            # Measure 4: Effective number of distinct states
            # SVD of token matrix -> how many significant singular values?
            U, S, V = torch.svd(h_final)
            S_norm = S / S.sum()
            effective_dim = torch.exp(-(S_norm * (S_norm + 1e-10).log()).sum()).item()
            condensation_ratio = 1.0 / max(effective_dim, 1.0)

            temp_results.append({
                'temperature': temp,
                'mean_entropy': round(mean_entropy, 4),
                'ground_state_fraction': round(ground_state_frac, 6),
                'mean_similarity': round(mean_similarity, 6),
                'effective_dim': round(effective_dim, 2),
                'condensation_ratio': round(condensation_ratio, 6)
            })
            print("    T=%.2f: entropy=%.2f, ground=%.4f, sim=%.4f, dim=%.1f" %
                  (temp, mean_entropy, ground_state_frac, mean_similarity, effective_dim))

        # Find critical temperature (steepest entropy drop)
        entropies = [tr['mean_entropy'] for tr in temp_results]
        entropy_diffs = [entropies[i] - entropies[i+1] for i in range(len(entropies)-1)]
        if entropy_diffs:
            max_drop_idx = np.argmax(entropy_diffs)
            t_c = (temperatures[max_drop_idx] + temperatures[max_drop_idx + 1]) / 2
        else:
            t_c = 0.5

        # Check for BEC: condensation ratio should spike below T_c
        low_t_cond = np.mean([tr['condensation_ratio'] for tr in temp_results
                              if tr['temperature'] < t_c])
        high_t_cond = np.mean([tr['condensation_ratio'] for tr in temp_results
                               if tr['temperature'] >= t_c])
        bec_detected = low_t_cond > 2 * high_t_cond

        all_condensation.append({
            'prompt': prompt[:50],
            'n_tokens': n_tokens,
            'critical_temperature': round(t_c, 4),
            'bec_detected': str(bec_detected),
            'low_t_condensation': round(low_t_cond, 6),
            'high_t_condensation': round(high_t_cond, 6),
            'temperature_sweep': temp_results
        })
        print("    T_c = %.3f, BEC = %s" % (t_c, bec_detected))

    # ===== Save Results =====
    n_bec = sum(1 for c in all_condensation if c['bec_detected'] == 'True')
    mean_tc = np.mean([c['critical_temperature'] for c in all_condensation])

    results = {
        'phase': 'Q113',
        'name': 'Semantic Bose-Einstein Condensation',
        'n_bec_detected': n_bec,
        'n_total': len(prompts),
        'mean_critical_temperature': round(mean_tc, 4),
        'condensation_data': all_condensation,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q113_bec.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Entropy vs temperature
    ax = axes[0]
    for ci, cond in enumerate(all_condensation):
        temps = [tr['temperature'] for tr in cond['temperature_sweep']]
        ents = [tr['mean_entropy'] for tr in cond['temperature_sweep']]
        ax.plot(temps, ents, 'o-', label='Prompt %d' % (ci + 1), markersize=5)
    ax.axvline(mean_tc, color='red', ls='--', alpha=0.5, label='Mean T_c=%.2f' % mean_tc)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean entropy')
    ax.set_xscale('log')
    ax.set_title('(a) Entropy vs Temperature')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # (b) Ground state fraction
    ax = axes[1]
    for ci, cond in enumerate(all_condensation):
        temps = [tr['temperature'] for tr in cond['temperature_sweep']]
        gf = [tr['ground_state_fraction'] for tr in cond['temperature_sweep']]
        ax.plot(temps, gf, 'o-', label='Prompt %d' % (ci + 1), markersize=5)
    ax.axvline(mean_tc, color='red', ls='--', alpha=0.5, label='T_c')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Ground state fraction')
    ax.set_xscale('log')
    ax.set_title('(b) Ground State Occupation')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # (c) Effective dimensionality
    ax = axes[2]
    for ci, cond in enumerate(all_condensation):
        temps = [tr['temperature'] for tr in cond['temperature_sweep']]
        ed = [tr['effective_dim'] for tr in cond['temperature_sweep']]
        ax.plot(temps, ed, 'o-', label='Prompt %d' % (ci + 1), markersize=5)
    ax.axvline(mean_tc, color='red', ls='--', alpha=0.5, label='T_c')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Effective dimensionality')
    ax.set_xscale('log')
    ax.set_title('(c) State Condensation\n(lower = more condensed)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    plt.suptitle('Q113: Semantic Bose-Einstein Condensation (BEC=%d/%d)' %
                 (n_bec, len(prompts)), fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q113_bec.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ113 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
