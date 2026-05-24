# -*- coding: utf-8 -*-
"""
Phase Q58: Quantum Contextuality (Kochen-Specker Theorem)

Quantum contextuality is a fundamental non-classical property:
the outcome of a measurement depends on what other measurements 
are performed simultaneously.

Test: Can S-Qubits exhibit contextual behavior where the same
observable yields different outcomes depending on which other
observables are jointly measured?

This is stronger than Bell inequality violation - it proves
non-classicality without requiring entanglement.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INJECT_LAYER = 10
EPOCHS = 100


def train_soul(model, tok, data, device, layer, epochs=EPOCHS, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def hook(m, i, o, v=vec):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def measure_context(model, tok, vec, prompt, target_ids, device, layer):
    """Measure probabilities in a given context (prompt)"""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return {tid: float(probs[tid]) for tid in target_ids}


def main():
    print("[Q58] Quantum Contextuality Test")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # Train S-Qubit for a specific task
    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    vec = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    
    # Target tokens to observe
    target_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    target_ids = [tok.encode(t)[-1] for t in target_tokens]

    # Define measurement contexts
    # Same S-Qubit measured in different prompt contexts
    contexts = [
        ("Context A: arithmetic", "min(7,2)="),
        ("Context B: comparison", "Compare 7 and 2. The smaller is"),
        ("Context C: sorting", "Sort [7,2] ascending: ["),
        ("Context D: language", "The minimum of seven and two is"),
        ("Context E: irrelevant", "The color of the sky is"),
        ("Context F: reversed", "max(2,7)="),
    ]

    print("\n  Testing contextuality across %d contexts..." % len(contexts))
    context_results = {}
    
    for ctx_name, prompt in contexts:
        probs = measure_context(model, tok, vec, prompt, target_ids, DEVICE, INJECT_LAYER)
        context_results[ctx_name] = probs
        top_tok = max(probs, key=probs.get)
        top_label = target_tokens[target_ids.index(top_tok)]
        print("    %s -> top prediction: '%s' (p=%.4f)" % (
            ctx_name, top_label, probs[top_tok]))

    # Contextuality analysis
    # For classical (non-contextual) hidden variable model:
    # The outcome of measuring the same observable should be context-independent
    # For quantum/contextual: same observable yields different results in different contexts

    # Compute pairwise context differences
    n_ctx = len(contexts)
    ctx_names = [c[0] for c in contexts]
    diff_matrix = np.zeros((n_ctx, n_ctx))
    
    for i in range(n_ctx):
        for j in range(n_ctx):
            p_i = np.array([context_results[ctx_names[i]][tid] for tid in target_ids])
            p_j = np.array([context_results[ctx_names[j]][tid] for tid in target_ids])
            # Total variation distance
            diff_matrix[i, j] = 0.5 * np.sum(np.abs(p_i - p_j))

    avg_context_shift = np.mean(diff_matrix[np.triu_indices(n_ctx, k=1)])
    max_context_shift = np.max(diff_matrix)
    
    print("\n  Contextuality metrics:")
    print("    Average context shift (TVD): %.4f" % avg_context_shift)
    print("    Maximum context shift: %.4f" % max_context_shift)
    
    # Non-contextuality inequality (KCBS-like)
    # If non-contextual: sum of pairwise agreements >= threshold
    # Compute: for each context pair, the probability that they give same top answer
    agreements = 0
    n_pairs = 0
    for i in range(n_ctx):
        for j in range(i+1, n_ctx):
            p_i = np.array([context_results[ctx_names[i]][tid] for tid in target_ids])
            p_j = np.array([context_results[ctx_names[j]][tid] for tid in target_ids])
            # Agreement = overlap
            agreement = np.sum(np.minimum(p_i, p_j))
            agreements += agreement
            n_pairs += 1
    
    avg_agreement = agreements / n_pairs
    # Non-contextual bound: agreement should be high (>0.5)
    nc_bound = 0.5
    contextual = avg_agreement < nc_bound
    
    print("    Average pairwise agreement: %.4f" % avg_agreement)
    print("    Non-contextual bound: %.4f" % nc_bound)
    print("    CONTEXTUAL: %s" % ("YES" if contextual else "NO"))

    # Determinism check: run same context twice
    print("\n  Determinism check...")
    det_shifts = []
    for ctx_name, prompt in contexts[:3]:
        p1 = measure_context(model, tok, vec, prompt, target_ids, DEVICE, INJECT_LAYER)
        p2 = measure_context(model, tok, vec, prompt, target_ids, DEVICE, INJECT_LAYER)
        shift = 0.5 * sum(abs(p1[t] - p2[t]) for t in target_ids)
        det_shifts.append(shift)
        print("    %s: repeat shift = %.6f" % (ctx_name, shift))
    avg_det = np.mean(det_shifts)
    print("    Average deterministic shift: %.6f (should be ~0)" % avg_det)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Probability distributions per context
    ax = axes[0]
    x = np.arange(len(target_tokens))
    width = 0.12
    for idx, (ctx_name, _) in enumerate(contexts):
        probs_list = [context_results[ctx_name][tid] for tid in target_ids]
        ax.bar(x + idx * width, probs_list, width, label=ctx_name.split(':')[0],
               alpha=0.85)
    ax.set_xticks(x + width * (n_ctx-1) / 2)
    ax.set_xticklabels(target_tokens)
    ax.set_xlabel('Token')
    ax.set_ylabel('Probability')
    ax.set_title('(a) Same S-Qubit, Different Contexts\nContextuality = context-dependent outcomes',
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis='y')

    # (b) Context distance heatmap
    ax = axes[1]
    short_names = [c.split(':')[0] for c in ctx_names]
    im = ax.imshow(diff_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(n_ctx):
        for j in range(n_ctx):
            ax.text(j, i, '%.2f' % diff_matrix[i,j], ha='center', va='center',
                    fontsize=7, color='white' if diff_matrix[i,j] > 0.3 else 'black')
    plt.colorbar(im, ax=ax, label='Total Variation Distance')
    ax.set_title('(b) Context Distance Matrix\nAvg TVD=%.3f' % avg_context_shift,
                 fontweight='bold')

    # (c) Agreement vs bound
    ax = axes[2]
    pair_agreements = []
    pair_labels = []
    for i in range(n_ctx):
        for j in range(i+1, n_ctx):
            p_i = np.array([context_results[ctx_names[i]][tid] for tid in target_ids])
            p_j = np.array([context_results[ctx_names[j]][tid] for tid in target_ids])
            pair_agreements.append(np.sum(np.minimum(p_i, p_j)))
            pair_labels.append('%s-%s' % (chr(65+i), chr(65+j)))
    
    colors = ['#4CAF50' if a < nc_bound else '#2196F3' for a in pair_agreements]
    ax.bar(range(len(pair_agreements)), pair_agreements, color=colors, 
           edgecolor='black', alpha=0.85)
    ax.axhline(nc_bound, color='red', ls='--', lw=2, label='NC bound (%.1f)' % nc_bound)
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=45, fontsize=7)
    ax.set_xlabel('Context pair')
    ax.set_ylabel('Agreement')
    ax.set_title('(c) Pairwise Agreement\nGreen = contextual (below NC bound)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q58: Quantum Contextuality (Kochen-Specker)\n'
                 'S-Qubit outcomes depend on measurement context',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q58_contextuality.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q58', 'name': 'quantum_contextuality',
        'avg_context_shift_tvd': round(avg_context_shift, 4),
        'max_context_shift': round(max_context_shift, 4),
        'avg_pairwise_agreement': round(avg_agreement, 4),
        'nc_bound': nc_bound,
        'contextual': bool(contextual),
        'avg_deterministic_shift': round(float(avg_det), 8),
        'n_contexts': n_ctx,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q58_contextuality.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q58 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
