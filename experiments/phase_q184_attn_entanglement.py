# -*- coding: utf-8 -*-
"""
Phase Q184: Attention Head Entanglement Map
=============================================
Which layers create the most entanglement between concepts?

Method (revised - no output_attentions needed):
1. Compare hidden state correlations when concepts appear alone vs together
2. Measure how inter-token correlation grows through layers
3. Use representation similarity analysis (RSA) as entanglement proxy
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
    print("Phase Q184: Attention Head Entanglement Map")
    print("  (Which Layers Generate Quantum Entanglement?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = hidden_size // n_heads

    print("  Architecture: %d layers x %d heads (head_dim=%d)" %
          (n_layers, n_heads, head_dim))

    def get_all_hidden(prompt):
        """Get hidden states at every layer."""
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        return [out.hidden_states[li][0].float().cpu().numpy()
                for li in range(n_layers + 1)]

    # === Test 1: Context-Dependent Entanglement ===
    print("\n--- Test 1: Context-Dependent Entanglement ---")
    print("  (How does co-occurrence change representations?)")

    prompt_pairs = [
        ("The capital of France is", "The capital of Germany is", "Paris", "Berlin"),
        ("The sky is", "The grass is", "sky", "grass"),
        ("Fire is very", "Ice is very", "Fire", "Ice"),
        ("Cats are", "Dogs are", "Cats", "Dogs"),
        ("Summer is", "Winter is", "Summer", "Winter"),
    ]
    pair_labels = ['Capital', 'Color', 'Temperature', 'Animal', 'Season']

    entanglement_by_layer = []  # (n_pairs, n_layers+1)

    for (pA, pB, wordA, wordB), label in zip(prompt_pairs, pair_labels):
        print("\n  [%s] '%s' + '%s'" % (label, wordA, wordB))

        # Get hidden states for individual prompts
        h_A = get_all_hidden(pA)  # list of (seq, hidden) per layer
        h_B = get_all_hidden(pB)

        # Get hidden states for combined prompt
        combined = pA + " and " + pB
        h_AB = get_all_hidden(combined)

        # Measure entanglement at each layer:
        # Definition: how much does the representation of A change
        # when B is also present? (contextual shift)
        ent_per_layer = []

        for li in range(n_layers + 1):
            # Last token of individual vs corresponding token in combined
            h_a_alone = h_A[li][-1, :]  # last token of A alone
            h_b_alone = h_B[li][-1, :]  # last token of B alone

            # In combined prompt, get representations at different positions
            h_combined_last = h_AB[li][-1, :]

            # Entanglement = how much does A's representation in combined
            # context differ from A alone?
            # Use 1 - cosine_similarity as distance
            sim_a = cosine_sim(h_a_alone, h_combined_last)
            sim_b = cosine_sim(h_b_alone, h_combined_last)

            # Contextual entanglement: deviation from independent representation
            ent = 1.0 - (sim_a + sim_b) / 2.0
            ent_per_layer.append(max(0.0, ent))

        entanglement_by_layer.append(ent_per_layer)

        # Report key layers
        ent_arr = np.array(ent_per_layer)
        peak_layer = int(np.argmax(ent_arr))
        print("    Peak entanglement: Layer %d (ent=%.4f)" %
              (peak_layer, ent_arr[peak_layer]))
        print("    Layer 0: %.4f, Mid: %.4f, Final: %.4f" %
              (ent_arr[0], ent_arr[n_layers // 2], ent_arr[-1]))

    # === Test 2: Token-Token Correlation Matrix ===
    print("\n--- Test 2: Token-Token Correlation Growth ---")

    analysis_prompt = "Paris and Berlin are both capitals of Europe"
    h_all = get_all_hidden(analysis_prompt)
    inp_ids = tok(analysis_prompt, return_tensors='pt')['input_ids'][0]
    n_tokens = len(inp_ids)

    # Safe ASCII representation of tokens
    token_strs = []
    for tid in inp_ids:
        t = tok.decode([tid.item()])
        # Replace non-ASCII characters
        safe_t = ''.join(c if ord(c) < 128 else '?' for c in t)
        token_strs.append(safe_t.strip() if safe_t.strip() else '_')
    print("  Tokens (%d): %s" % (n_tokens, str(token_strs)))

    # Correlation matrix at each layer
    corr_matrices = []
    for li in range(n_layers + 1):
        h = h_all[li]  # (seq, hidden)
        # Pairwise cosine similarity
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        h_normed = h / norms
        corr = h_normed @ h_normed.T
        corr_matrices.append(corr)

    # Average off-diagonal correlation (entanglement strength)
    avg_corr_per_layer = []
    for corr in corr_matrices:
        mask = 1 - np.eye(n_tokens)
        avg_off_diag = float(np.mean(np.abs(corr * mask)))
        avg_corr_per_layer.append(avg_off_diag)

    print("  Avg off-diagonal correlation by layer:")
    for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
        print("    Layer %d: %.4f" % (li, avg_corr_per_layer[li]))

    # === Test 3: Head-Level Analysis via Hidden State Decomposition ===
    print("\n--- Test 3: Per-Head Entanglement (Hidden State Decomposition) ---")

    # Decompose hidden states into head-sized chunks
    # h[layer] has shape (seq, hidden) = (seq, n_heads * head_dim)
    head_entanglement = np.zeros((n_layers + 1, n_heads))

    for li in range(n_layers + 1):
        h = h_all[li]  # (seq, hidden)
        for hi in range(n_heads):
            start = hi * head_dim
            end = (hi + 1) * head_dim
            h_head = h[:, start:end]  # (seq, head_dim)

            # Correlation between first and last token in this head
            corr = cosine_sim(h_head[0], h_head[-1])
            head_entanglement[li, hi] = abs(corr)

    # Find top entangling heads (excluding layer 0)
    head_ent_no_l0 = head_entanglement[1:, :]
    flat_top = np.argsort(head_ent_no_l0.ravel())[::-1][:10]
    print("  Top 10 entangling (layer, head) pairs:")
    for idx in flat_top:
        li, hi = np.unravel_index(idx, head_ent_no_l0.shape)
        li += 1  # offset for removed layer 0
        print("    Layer %d Head %d: corr=%.4f" %
              (li, hi, head_entanglement[li, hi]))

    # Summary
    avg_ent_growth = np.array(avg_corr_per_layer)
    growth_ratio = avg_ent_growth[-1] / (avg_ent_growth[0] + 1e-10)
    peak_layer_avg = int(np.argmax(avg_ent_growth))

    print("\n--- Summary ---")
    print("  Entanglement growth ratio (last/first): %.2fx" % growth_ratio)
    print("  Peak entanglement layer: %d" % peak_layer_avg)
    print("  Pattern: %s" % (
        "Monotonic growth" if np.corrcoef(range(len(avg_ent_growth)),
                                          avg_ent_growth)[0, 1] > 0.5
        else "Non-monotonic"))

    # Save
    results = {
        'phase': 'Q184',
        'name': 'Attention Head Entanglement Map',
        'architecture': {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'head_dim': head_dim,
        },
        'context_entanglement': {
            label: [round(e, 4) for e in ent]
            for label, ent in zip(pair_labels, entanglement_by_layer)
        },
        'token_correlation_growth': [round(c, 4) for c in avg_corr_per_layer],
        'head_entanglement': head_entanglement.tolist(),
        'summary': {
            'growth_ratio': round(growth_ratio, 2),
            'peak_layer': peak_layer_avg,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q184_attn_entanglement.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Context-dependent entanglement per layer
    ax = axes[0]
    palette = plt.cm.tab10(np.linspace(0, 1, len(pair_labels)))
    for i, (label, ent) in enumerate(zip(pair_labels, entanglement_by_layer)):
        ax.plot(range(len(ent)), ent, '-', color=palette[i],
                linewidth=1.5, label=label)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Contextual Entanglement')
    ax.set_title('(a) Context-Dependent Entanglement\n(How co-occurrence changes representations)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Head entanglement heatmap
    ax = axes[1]
    im = ax.imshow(head_entanglement[1:].T, aspect='auto', cmap='hot',
                   origin='lower')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')
    ax.set_title('(b) Per-Head Entanglement\n(Token correlation by head)')
    plt.colorbar(im, ax=ax, label='Correlation')

    # (c) Average correlation growth
    ax = axes[2]
    ax.plot(range(len(avg_corr_per_layer)), avg_corr_per_layer,
            'o-', color='#E91E63', linewidth=2, markersize=4)
    ax.fill_between(range(len(avg_corr_per_layer)), avg_corr_per_layer,
                    alpha=0.2, color='#E91E63')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Off-Diagonal Correlation')
    ax.set_title('(c) Entanglement Growth\n(%.1fx increase through layers)' %
                growth_ratio)
    ax.grid(alpha=0.3)

    plt.suptitle('Q184: Attention Head Entanglement Map\n'
                 '(Entanglement grows %.1fx through %d layers, peak at L%d)' %
                 (growth_ratio, n_layers, peak_layer_avg),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q184_attn_entanglement.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ184 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
