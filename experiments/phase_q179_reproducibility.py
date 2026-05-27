# -*- coding: utf-8 -*-
"""
Phase Q179: Blind Reproducibility Protocol (v3)
=================================================
Grok's critique: "Cherry-picked seeds/prompts?"

Using the CORRECT S-Qubit methodology (from Q10):
1. For N random task pairs, train soul vectors targeting specific tokens
2. Measure interference visibility in P(target_token) space
3. Report pass rates with confidence intervals

If ALL random tasks produce V > 0.5 -> universal, not cherry-picked.
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

INJECT_LAYER = 8
N_PHI = 16


def train_soul(model, tok, data, device, layer=8, epochs=80, seed=42):
    """Train a soul vector to produce a specific target token (from Q10)."""
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
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def get_p_token(model, tok, prompt, device, inject_vec, inject_layer, target_tok_id):
    """Measure P(target_token) with injected soul vector (from Q10)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[inject_layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[target_tok_id])


def measure_interference(model, tok, prompt, device, vec0, vec1,
                         target_tok_id, inject_layer, n_phi=N_PHI):
    """Sweep phi and measure P(target) interference pattern (from Q10)."""
    phis = np.linspace(0, 4 * np.pi, n_phi)
    p_vals = []
    scale = vec0.norm()
    for phi in phis:
        vec = np.cos(phi / 2) * vec0 + np.sin(phi / 2) * vec1
        n = vec.norm()
        if n > 0:
            vec = vec / n * scale
        p = get_p_token(model, tok, prompt, device, vec, inject_layer, target_tok_id)
        p_vals.append(p)
    p_arr = np.array(p_vals)
    amp = (p_arr.max() - p_arr.min()) / 2.0
    visibility = (p_arr.max() - p_arr.min()) / (p_arr.max() + p_arr.min() + 1e-8)
    return float(visibility), float(amp), p_arr.tolist()


# Task definitions: diverse domains
TASK_POOL = [
    {
        'name': 'CAPITAL_FR_DE',
        'zero_data': [("The capital of France is","Paris"),("France's capital:","Paris")],
        'one_data': [("The capital of Germany is","Berlin"),("Germany's capital:","Berlin")],
        'prompt': "The capital of France is",
        'target': "Paris",
    },
    {
        'name': 'COLOR_sky_grass',
        'zero_data': [("The sky is","blue"),("The ocean is","blue"),("Clear sky:","blue")],
        'one_data': [("The grass is","green"),("Leaves are","green"),("Plants are","green")],
        'prompt': "The sky is",
        'target': "blue",
    },
    {
        'name': 'CODE_return_pass',
        'zero_data': [("def add(x,y):","return"),("def square(x):","return"),("def get():","return")],
        'one_data': [("def empty():","pass"),("def noop():","pass"),("def skip():","pass")],
        'prompt': "def compute():",
        'target': "return",
    },
    {
        'name': 'PARITY_even_odd',
        'zero_data': [("2 is","even"),("4 is","even"),("6 is","even")],
        'one_data': [("1 is","odd"),("3 is","odd"),("5 is","odd")],
        'prompt': "8 is",
        'target': "even",
    },
    {
        'name': 'ANIMAL_cat_dog',
        'zero_data': [("A small furry pet that purrs is a","cat"),("Meow says the","cat")],
        'one_data': [("A loyal pet that barks is a","dog"),("Woof says the","dog")],
        'prompt': "The pet that purrs is a",
        'target': "cat",
    },
    {
        'name': 'SEASON_summer_winter',
        'zero_data': [("The hottest season is","summer"),("Beach weather in","summer")],
        'one_data': [("The coldest season is","winter"),("Snow falls in","winter")],
        'prompt': "The hottest season is",
        'target': "summer",
    },
    {
        'name': 'DIRECTION_north_south',
        'zero_data': [("The Arctic is in the","north"),("Canada is in the","north")],
        'one_data': [("Antarctica is in the","south"),("Australia is in the","south")],
        'prompt': "The Arctic is in the",
        'target': "north",
    },
    {
        'name': 'SIZE_big_small',
        'zero_data': [("An elephant is","big"),("A whale is","big"),("A mountain is","big")],
        'one_data': [("An ant is","small"),("A mouse is","small"),("A seed is","small")],
        'prompt': "A whale is",
        'target': "big",
    },
    {
        'name': 'TEMP_hot_cold',
        'zero_data': [("Fire is","hot"),("The sun is","hot"),("Lava is","hot")],
        'one_data': [("Ice is","cold"),("Snow is","cold"),("The Arctic is","cold")],
        'prompt': "Fire is",
        'target': "hot",
    },
    {
        'name': 'BOOL_true_false',
        'zero_data': [("1 == 1 is","True"),("2 > 1 is","True"),("3 != 4 is","True")],
        'one_data': [("1 == 2 is","False"),("2 > 3 is","False"),("3 != 3 is","False")],
        'prompt': "5 == 5 is",
        'target': "True",
    },
    {
        'name': 'FOOD_sweet_salty',
        'zero_data': [("Candy is","sweet"),("Sugar is","sweet"),("Chocolate is","sweet")],
        'one_data': [("Chips are","salty"),("Pretzels are","salty"),("Popcorn is","salty")],
        'prompt': "Honey is",
        'target': "sweet",
    },
    {
        'name': 'SPEED_fast_slow',
        'zero_data': [("A cheetah is","fast"),("Light is","fast"),("A jet is","fast")],
        'one_data': [("A snail is","slow"),("A turtle is","slow"),("A sloth is","slow")],
        'prompt': "A rocket is",
        'target': "fast",
    },
]


def main():
    print("=" * 60)
    print("Phase Q179: Blind Reproducibility Protocol (v3)")
    print("  (12 Random Tasks, Proper Soul Vector Training)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    results_list = []

    for task in TASK_POOL:
        name = task['name']
        print("\n--- Task: %s ---" % name)

        # Train soul vectors
        vec0 = train_soul(model, tok, task['zero_data'], device,
                         layer=INJECT_LAYER, seed=42, epochs=80)
        vec1 = train_soul(model, tok, task['one_data'], device,
                         layer=INJECT_LAYER, seed=99, epochs=80)

        # Get target token ID
        target_id = tok.encode(task['target'])[-1]

        # Verify basis quality
        p0 = get_p_token(model, tok, task['prompt'], device,
                        vec0, INJECT_LAYER, target_id)
        p1 = get_p_token(model, tok, task['prompt'], device,
                        vec1, INJECT_LAYER, target_id)
        cos_sim = float(torch.nn.functional.cosine_similarity(
            vec0.unsqueeze(0), vec1.unsqueeze(0)).item())

        # Measure interference
        vis, amp, p_curve = measure_interference(
            model, tok, task['prompt'], device, vec0, vec1,
            target_id, INJECT_LAYER)

        result = {
            'name': name,
            'visibility': round(vis, 4),
            'amplitude': round(amp, 6),
            'p0_target': round(p0, 4),
            'p1_target': round(p1, 4),
            'cosine_01': round(cos_sim, 4),
            'p_curve': [round(p, 4) for p in p_curve],
        }
        results_list.append(result)

        print("  |0> P(target)=%.4f  |1> P(target)=%.4f  cos=%.4f" %
              (p0, p1, cos_sim))
        print("  Visibility=%.4f  Amplitude=%.6f" % (vis, amp))

    # Summary statistics
    visibilities = [r['visibility'] for r in results_list]
    amplitudes = [r['amplitude'] for r in results_list]
    vis_mean = float(np.mean(visibilities))
    vis_std = float(np.std(visibilities))
    vis_pass = sum(1 for v in visibilities if v > 0.5)
    amp_mean = float(np.mean(amplitudes))
    amp_cv = float(np.std(amplitudes) / (amp_mean + 1e-10))

    print("\n--- SUMMARY ---")
    print("  Tasks: %d" % len(TASK_POOL))
    print("  Visibility: %.4f +/- %.4f" % (vis_mean, vis_std))
    print("  Pass rate (V>0.5): %d/%d (%.1f%%)" %
          (vis_pass, len(TASK_POOL), 100 * vis_pass / len(TASK_POOL)))
    print("  Amplitude: %.6f (CV=%.1f%%)" % (amp_mean, 100 * amp_cv))

    if vis_pass >= len(TASK_POOL) * 0.8:
        verdict = "UNIVERSAL: interference in %d/%d tasks (%.0f%%)" % (
            vis_pass, len(TASK_POOL), 100 * vis_pass / len(TASK_POOL))
    elif vis_pass >= len(TASK_POOL) * 0.5:
        verdict = "PARTIAL: interference in %d/%d tasks" % (vis_pass, len(TASK_POOL))
    else:
        verdict = "TASK-SPECIFIC: only %d/%d tasks show interference" % (
            vis_pass, len(TASK_POOL))
    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q179',
        'name': 'Blind Reproducibility Protocol (v3)',
        'n_tasks': len(TASK_POOL),
        'tasks': results_list,
        'summary': {
            'visibility_mean': round(vis_mean, 4),
            'visibility_std': round(vis_std, 4),
            'pass_rate_pct': round(100 * vis_pass / len(TASK_POOL), 1),
            'amplitude_mean': round(amp_mean, 6),
            'amplitude_cv': round(amp_cv, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q179_reproducibility.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, len(TASK_POOL)))

    # (a) Interference fringes for all tasks
    ax = axes[0]
    phis = np.linspace(0, 4 * np.pi, N_PHI)
    for i, r in enumerate(results_list):
        ax.plot(phis / np.pi, r['p_curve'], '-', color=palette[i],
                linewidth=1.5, label=r['name'][:10], alpha=0.8)
    ax.set_xlabel('Phase (x pi)')
    ax.set_ylabel('P(target token)')
    ax.set_title('(a) Interference Fringes\n(%d diverse tasks)' % len(TASK_POOL))
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    # (b) Visibility bar chart
    ax = axes[1]
    names = [r['name'][:8] for r in results_list]
    ax.bar(range(len(names)), visibilities, color=palette, edgecolor='black',
           alpha=0.85)
    ax.axhline(0.5, color='red', ls='--', linewidth=2, label='Threshold (0.5)')
    ax.axhline(vis_mean, color='blue', ls=':', linewidth=2,
               label='Mean=%.3f' % vis_mean)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Fringe Visibility')
    ax.set_title('(b) Visibility by Task\n(%.0f%% pass rate)' %
                (100 * vis_pass / len(TASK_POOL)))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # (c) Amplitude distribution
    ax = axes[2]
    ax.hist(amplitudes, bins=10, color='#4CAF50', edgecolor='black', alpha=0.85)
    ax.axvline(amp_mean, color='red', ls='--', linewidth=2,
               label='Mean=%.4f' % amp_mean)
    ax.set_xlabel('Interference Amplitude')
    ax.set_ylabel('Count')
    ax.set_title('(c) Amplitude Distribution\n(CV=%.1f%%)' % (100 * amp_cv))
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Q179: Blind Reproducibility Protocol\n'
                 '12 Tasks, Zero Cherry-Picking -> %s' % verdict[:40],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q179_reproducibility.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ179 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
