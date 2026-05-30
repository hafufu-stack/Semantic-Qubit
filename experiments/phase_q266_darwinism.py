# -*- coding: utf-8 -*-
"""
Phase Q266: Quantum Darwinism
================================
Why does "objective reality" emerge from quantum uncertainty?
Zurek's Quantum Darwinism: certain states get redundantly
copied across environment fragments (attention heads).
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
    print("Phase Q266: Quantum Darwinism")
    print("  (How objective reality emerges from quantum)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    prompts = [
        "The cat is sitting on the mat",
        "Quantum superposition of all possibilities",
        "The answer to the question is 42",
        "Schrodinger's cat is alive and dead",
    ]

    # Hook to capture attention patterns
    attn_data = {}
    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # output is (attn_output, attn_weights, ...) for some models
            # We'll capture via the hidden state agreement across heads
            pass
        return hook

    all_results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Use hidden states at different layers as "environment fragments"
        # Quantum Darwinism: how much info about the final state is in each fragment?
        h_final = out.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
        h_final_norm = h_final / (np.linalg.norm(h_final) + 1e-10)

        # Sample environment fragments: groups of layers
        n_frags = 6
        frag_size = max(1, n_layers // n_frags)
        mutual_infos = []
        cumulative_info = []

        for f in range(n_frags):
            frag_layers = list(range(f * frag_size, min((f + 1) * frag_size, n_layers)))
            # Information in fragment about system = agreement with final state
            frag_info = 0
            for li in frag_layers:
                if li <= n_layers:
                    h_frag = out.hidden_states[li][0, -1, :].float().cpu().numpy()
                    h_frag_norm = h_frag / (np.linalg.norm(h_frag) + 1e-10)
                    # Mutual information proxy: cosine similarity
                    cos_sim = float(np.dot(h_frag_norm, h_final_norm))
                    frag_info += abs(cos_sim)
            frag_info /= max(len(frag_layers), 1)
            mutual_infos.append(round(frag_info, 4))

            # Cumulative: info from first f+1 fragments
            all_frags = list(range(0, min((f + 1) * frag_size, n_layers)))
            cum_info = 0
            for li in all_frags:
                if li <= n_layers:
                    h_f = out.hidden_states[li][0, -1, :].float().cpu().numpy()
                    h_f /= np.linalg.norm(h_f) + 1e-10
                    cum_info += abs(float(np.dot(h_f, h_final_norm)))
            cum_info /= max(len(all_frags), 1)
            cumulative_info.append(round(cum_info, 4))

        # Darwinism signature: plateau in mutual info
        # (once you see enough fragments, you know everything)
        info_plateau = cumulative_info[-1] - cumulative_info[len(cumulative_info) // 2]
        redundancy = sum(1 for mi in mutual_infos if mi > 0.5) / len(mutual_infos)

        print("  '%s'..." % prompt[:35])
        print("    Fragment info: %s" % mutual_infos)
        print("    Redundancy: %.1f%%, Plateau: %.4f" % (redundancy * 100, info_plateau))

        all_results.append({
            'prompt': prompt[:35],
            'fragment_info': mutual_infos,
            'cumulative_info': cumulative_info,
            'redundancy': round(redundancy, 2),
            'plateau': round(info_plateau, 4),
        })

    avg_redundancy = np.mean([r['redundancy'] for r in all_results])
    avg_plateau = np.mean([r['plateau'] for r in all_results])

    if avg_redundancy > 0.5 and avg_plateau < 0.1:
        verdict = "DARWINISM: %.0f%% redundancy, plateau=%.3f (info broadcast to env)" % (
            avg_redundancy * 100, avg_plateau)
    elif avg_redundancy > 0.3:
        verdict = "PARTIAL DARWINISM: %.0f%% redundancy, plateau=%.3f" % (
            avg_redundancy * 100, avg_plateau)
    else:
        verdict = "NO DARWINISM: low redundancy %.0f%%" % (avg_redundancy * 100)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q266', 'name': 'Quantum Darwinism',
        'scenarios': all_results,
        'summary': {'avg_redundancy': round(avg_redundancy, 2),
                     'avg_plateau': round(avg_plateau, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q266_darwinism.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i, r in enumerate(all_results):
        ax.plot(range(len(r['fragment_info'])), r['fragment_info'], 'o-', lw=2,
                label=r['prompt'][:15])
    ax.set_xlabel('Fragment Index'); ax.set_ylabel('Mutual Info (proxy)')
    ax.set_title('(a) Per-Fragment Information'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for i, r in enumerate(all_results):
        ax.plot(range(len(r['cumulative_info'])), r['cumulative_info'], 'o-', lw=2,
                label=r['prompt'][:15])
    ax.set_xlabel('Fragments Observed'); ax.set_ylabel('Cumulative Info')
    ax.set_title('(b) Cumulative (Darwinism = plateau)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('Q266: Quantum Darwinism\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q266_darwinism.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ266 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
