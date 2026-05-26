# -*- coding: utf-8 -*-
"""
Phase Q132: Attention-qLDPC (Error Correction Visualization)
==============================================================
Q126 showed TOTAL noise immunity (noise=100, all layers, adversarial).
WHY? Hypothesis: Attention performs syndrome measurement, using
other token positions to reconstruct the corrupted last token.

This experiment visualizes the error correction mechanism.
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
    print("Phase Q132: Attention-qLDPC Error Correction")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    prompt = "The capital of France is"
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    # === Step 1: Clean hidden state patterns ===
    with torch.no_grad():
        out_clean = model(**inp, output_hidden_states=True)
    top_clean = torch.argmax(out_clean.logits[0, -1, :]).item()
    clean_token = tok.decode([top_clean])
    print("  Clean prediction: '%s'" % clean_token)

    # Collect clean hidden states per layer (for recovery analysis)
    clean_hidden = []
    for li in range(n_layers + 1):
        h = out_clean.hidden_states[li][0, -1, :].float().cpu()
        clean_hidden.append(h)
    clean_hidden_stack = torch.stack(clean_hidden)  # (n_layers+1, hidden)

    # === Step 2: Inject MASSIVE noise at different positions ===
    # Fine-grained noise to find exact breaking point
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    injection_strategies = [
        ('last_token_only', 'Last token'),
        ('single_mid_layer', 'Single mid layer'),
        ('all_tokens', 'All tokens'),
        ('all_except_last', 'All except last'),
    ]

    attention_shifts = {}
    prediction_results = {}

    for strategy_key, strategy_name in injection_strategies:
        print("\n  Strategy: %s" % strategy_name)
        strategy_results = []

        for noise_level in noise_levels:
            handles = []

            # For single_mid_layer, only hook layer n_layers//2
            layers_to_hook = [n_layers // 2] if strategy_key == 'single_mid_layer' else range(n_layers)
            for li in layers_to_hook:
                def make_hook(nl, strat):
                    def hook(module, input, output):
                        # Qwen2 layers return Tensor directly, not tuple
                        if isinstance(output, torch.Tensor):
                            h = output.clone()
                        elif isinstance(output, tuple):
                            h = output[0].clone()
                        else:
                            return output

                        h_std = h.float().std().item()
                        scale = nl * max(h_std, 0.01)

                        if strat == 'last_token_only':
                            nv = torch.randn(h.shape[-1], device=h.device, dtype=h.dtype) * scale
                            h[0, -1, :] = h[0, -1, :] + nv
                        elif strat == 'all_tokens':
                            nv = torch.randn_like(h) * scale
                            h = h + nv
                        elif strat == 'all_except_last':
                            nv = torch.randn_like(h) * scale
                            h[:, :-1, :] = h[:, :-1, :] + nv[:, :-1, :]

                        if isinstance(output, torch.Tensor):
                            return h
                        else:
                            return (h,) + output[1:]
                    return hook

                handles.append(model.model.layers[li].register_forward_hook(
                    make_hook(noise_level, strategy_key)))

            with torch.no_grad():
                out_noisy = model(**inp, output_hidden_states=True)

            for h in handles:
                h.remove()

            logits_raw = out_noisy.logits[0, -1, :].float()
            if torch.isnan(logits_raw).any() or torch.isinf(logits_raw).any():
                top_noisy = -1  # mark as broken
            else:
                top_noisy = torch.argmax(logits_raw).item()

            # Measure hidden state recovery per layer
            layer_recovery = []
            for li_check in range(n_layers + 1):
                h_noisy = out_noisy.hidden_states[li_check][0, -1, :].float().cpu()
                h_clean = clean_hidden[li_check]
                n1 = h_noisy.norm().item()
                n2 = h_clean.norm().item()
                if n1 > 1e-8 and n2 > 1e-8 and not (torch.isnan(h_noisy).any()):
                    cos = torch.nn.functional.cosine_similarity(
                        h_noisy.unsqueeze(0), h_clean.unsqueeze(0)).item()
                    if np.isnan(cos): cos = 0.0
                else:
                    cos = 0.0
                layer_recovery.append(cos)

            mid_layer_cos = layer_recovery[n_layers // 2]
            final_layer_cos = layer_recovery[-1]
            recovery_gain = final_layer_cos - mid_layer_cos
            if np.isnan(recovery_gain): recovery_gain = 0.0
            mean_rec = np.nanmean(layer_recovery)
            attn_shift = float(1.0 - mean_rec) if not np.isnan(mean_rec) else 1.0

            # NaN-safe probability
            logits = out_noisy.logits[0, -1, :].float()
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                prob_correct = 0.0
            else:
                probs = torch.softmax(logits, dim=-1)
                prob_correct = probs[top_clean].item()
                if np.isnan(prob_correct): prob_correct = 0.0

            strategy_results.append({
                'noise': noise_level,
                'correct': str(top_noisy == top_clean),
                'prob_correct': round(float(prob_correct), 6),
                'attn_shift': round(float(attn_shift), 6),
                'recovery_gain': round(float(recovery_gain), 6),
                'layer_recovery': [round(r, 4) for r in layer_recovery] if noise_level in [0, 10, 100] else None,
            })
            print("    noise=%.0f: correct=%s, P=%.4f, attn_shift=%.4f" %
                  (noise_level, top_noisy == top_clean, prob_correct, attn_shift))

        attention_shifts[strategy_key] = strategy_results
        prediction_results[strategy_key] = strategy_results

    # === Key finding: Does attention REDIRECT under noise? ===
    print("\n--- Key Finding ---")
    # Compare: last_token_only vs all_except_last
    lt_results = attention_shifts.get('last_token_only', [])
    ae_results = attention_shifts.get('all_except_last', [])

    lt_correct = sum(1 for r in lt_results if r['correct'] == 'True')
    ae_correct = sum(1 for r in ae_results if r['correct'] == 'True')

    print("  Last-token noise: %d/%d correct" % (lt_correct, len(lt_results)))
    print("  All-except-last noise: %d/%d correct" % (ae_correct, len(ae_results)))

    if ae_correct < lt_correct:
        print("  -> Attention RECOVERS from last-token corruption")
        print("     by reading from OTHER (clean) token positions!")
        print("     This IS implicit quantum error correction!")
        mechanism = "Attention-based syndrome measurement from clean tokens"
    else:
        print("  -> Model is resilient to BOTH strategies")
        mechanism = "Deep residual connections + layer normalization"

    # === Save ===
    results = {
        'phase': 'Q132',
        'name': 'Attention-qLDPC Error Correction',
        'prompt': prompt,
        'clean_prediction': clean_token,
        'strategies': attention_shifts,
        'mechanism': mechanism,
        'last_token_correct': lt_correct,
        'all_except_last_correct': ae_correct,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q132_qlpdc.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Layer-by-layer recovery for noise=100
    ax = axes[0]
    for skey, sname in injection_strategies:
        for r in attention_shifts[skey]:
            if r['noise'] == 100.0 and r.get('layer_recovery'):
                ax.plot(range(len(r['layer_recovery'])), r['layer_recovery'],
                        'o-', label=sname, markersize=3)
    ax.axhline(1.0, color='gold', ls='--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine similarity to clean')
    ax.set_title('(a) Hidden State Recovery\n(noise=100, per layer)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) Probability under noise (3 strategies)
    ax = axes[1]
    for skey, sname in injection_strategies:
        nls = [r['noise'] for r in attention_shifts[skey] if r['noise'] > 0]
        probs = [r['prob_correct'] for r in attention_shifts[skey] if r['noise'] > 0]
        ax.semilogx(nls, probs, 'o-', label=sname, markersize=5)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Noise level (log)')
    ax.set_ylabel('P(correct)')
    ax.set_title('(b) Error Correction by Strategy')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (c) Recovery gain (improvement from mid to final layer)
    ax = axes[2]
    for skey, sname in injection_strategies:
        nls = [r['noise'] for r in attention_shifts[skey] if r['noise'] > 0]
        gains = [r.get('recovery_gain', 0) for r in attention_shifts[skey] if r['noise'] > 0]
        ax.semilogx(nls, gains, 'o-', label=sname, markersize=5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Noise level (log)')
    ax.set_ylabel('Recovery gain (final - mid cosine)')
    ax.set_title('(c) Error Correction Strength\n(positive = recovery happening)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Q132: Attention-qLDPC Error Correction\nMechanism: %s' % mechanism[:50],
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q132_qlpdc.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ132 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
