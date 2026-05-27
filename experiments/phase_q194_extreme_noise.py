# -*- coding: utf-8 -*-
"""
Phase Q194: Extreme Noise Resilience
=======================================
Q190 showed V=1.000 at 50% noise. How far can we push it?

Test at 60%, 70%, 80%, 90%, 95%, 99% noise levels.
Also: test STRUCTURED noise (correlated, not random).

If V stays high at 90%+ noise -> superhuman error correction.
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


def get_p_token(model, tok, prompt, device, inject_vec, layer, target_id):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[target_id])


def add_extreme_noise(vec, noise_type, noise_level, device):
    v = vec.clone()
    n = v.shape[0]

    if noise_type == 'erasure':
        n_erase = max(1, int(n * noise_level))
        idx = torch.randperm(n)[:n_erase]
        v[idx] = 0.0

    elif noise_type == 'gaussian':
        v = v + torch.randn_like(v) * noise_level * torch.norm(v)

    elif noise_type == 'replacement':
        # Replace fraction of dims with random values
        n_replace = max(1, int(n * noise_level))
        idx = torch.randperm(n)[:n_replace]
        v[idx] = torch.randn(n_replace, device=device) * torch.std(v)

    elif noise_type == 'shuffle':
        # Shuffle fraction of dimensions
        n_shuffle = max(2, int(n * noise_level))
        idx = torch.randperm(n)[:n_shuffle]
        v[idx] = v[idx[torch.randperm(n_shuffle)]]

    elif noise_type == 'correlated':
        # Correlated noise: same noise added to blocks
        block_size = max(1, int(n * 0.1))
        n_blocks = max(1, int(n * noise_level / block_size))
        for _ in range(n_blocks):
            start = torch.randint(0, n - block_size, (1,)).item()
            noise_val = torch.randn(1, device=device) * torch.std(v) * noise_level
            v[start:start+block_size] = v[start:start+block_size] + noise_val

    elif noise_type == 'quantize':
        # Quantize to fewer bits (information loss)
        n_levels = max(2, int(1 / (noise_level + 1e-10)))
        v_min, v_max = v.min(), v.max()
        v = torch.round((v - v_min) / (v_max - v_min + 1e-10) * n_levels) / n_levels
        v = v * (v_max - v_min) + v_min

    return v


def measure_visibility(model, tok, prompt, device, vec0, vec1, target_id,
                        noise_type, noise_level):
    phis = np.linspace(0, 4 * np.pi, N_PHI)
    p_vals = []
    scale = vec0.norm()
    for phi in phis:
        vec = np.cos(phi/2) * vec0 + np.sin(phi/2) * vec1
        n = vec.norm()
        if n > 0: vec = vec / n * scale
        vec_noisy = add_extreme_noise(vec, noise_type, noise_level, device)
        p = get_p_token(model, tok, prompt, device, vec_noisy, INJECT_LAYER, target_id)
        p_vals.append(p)
    p_arr = np.array(p_vals)
    vis = (p_arr.max() - p_arr.min()) / (p_arr.max() + p_arr.min() + 1e-8)
    return float(vis)


def main():
    print("=" * 60)
    print("Phase Q194: Extreme Noise Resilience")
    print("  (Pushing to 99%% noise - how far can we go?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size

    print("  Training soul vectors...")
    vec0 = train_soul(model, tok,
                     [("The sky is","blue"),("The ocean is","blue")],
                     device, layer=INJECT_LAYER, seed=42)
    vec1 = train_soul(model, tok,
                     [("The grass is","green"),("Leaves are","green")],
                     device, layer=INJECT_LAYER, seed=99)

    target_id = tok.encode("blue")[-1]
    prompt = "The sky is"

    # Extended noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    noise_types = ['erasure', 'gaussian', 'replacement', 'shuffle',
                   'correlated', 'quantize']

    all_results = {}
    n_trials = 5

    for ntype in noise_types:
        print("\n--- %s ---" % ntype)
        visibilities = []
        for nl in noise_levels:
            trials = []
            for t in range(n_trials):
                torch.manual_seed(t * 1000 + int(nl * 100))
                vis = measure_visibility(model, tok, prompt, device,
                                        vec0, vec1, target_id, ntype, nl)
                trials.append(vis)
            avg = float(np.mean(trials))
            visibilities.append(avg)
            if nl >= 0.5:
                print("  noise=%.2f: V=%.4f" % (nl, avg))
        all_results[ntype] = visibilities

    # Find breaking point for each noise type
    breaking_points = {}
    for ntype, viss in all_results.items():
        bp = noise_levels[-1]
        for nl, v in zip(noise_levels, viss):
            if v < 0.5:
                bp = nl
                break
        breaking_points[ntype] = bp

    print("\n--- Breaking Points (V drops below 0.5) ---")
    for ntype, bp in breaking_points.items():
        if bp >= 0.99:
            print("  %s: INVINCIBLE (V>0.5 even at 99%%)" % ntype)
        else:
            print("  %s: breaks at %.0f%%" % (ntype, 100 * bp))

    avg_bp = float(np.mean(list(breaking_points.values())))
    n_invincible = sum(1 for bp in breaking_points.values() if bp >= 0.99)

    if n_invincible == len(noise_types):
        verdict = "INVINCIBLE: V>0.5 at 99%% noise for ALL %d noise types" % len(noise_types)
    elif avg_bp > 0.9:
        verdict = "EXTREME RESILIENCE: avg breaking point %.0f%%" % (100 * avg_bp)
    else:
        verdict = "MODERATE: avg breaking point %.0f%%" % (100 * avg_bp)
    print("\n  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q194',
        'name': 'Extreme Noise Resilience',
        'hidden_size': hidden_size,
        'noise_levels': noise_levels,
        'results': all_results,
        'breaking_points': breaking_points,
        'summary': {
            'avg_breaking_point': round(avg_bp, 4),
            'n_invincible': n_invincible,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q194_extreme_noise.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Visibility vs noise
    ax = axes[0]
    palette = plt.cm.Set1(np.linspace(0, 1, len(noise_types)))
    for i, ntype in enumerate(noise_types):
        ax.plot(noise_levels, all_results[ntype], 'o-', color=palette[i],
                linewidth=2, markersize=4, label=ntype)
    ax.axhline(0.5, color='red', ls=':', label='V=0.5')
    ax.set_xlabel('Noise Level (fraction)')
    ax.set_ylabel('Interference Visibility')
    ax.set_title('(a) Extreme Noise Resilience\n(6 noise types, up to 99%%)')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # (b) Breaking points
    ax = axes[1]
    bp_vals = [breaking_points[nt] for nt in noise_types]
    colors = ['#4CAF50' if bp >= 0.99 else '#FF9800' if bp >= 0.7
              else '#F44336' for bp in bp_vals]
    ax.barh(range(len(noise_types)), [100*bp for bp in bp_vals],
            color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(noise_types)))
    ax.set_yticklabels(noise_types)
    ax.set_xlabel('Breaking Point (%%)')
    ax.set_title('(b) Breaking Points\n(Max tolerable noise)')
    ax.axvline(99, color='green', ls='--', alpha=0.5)
    ax.grid(alpha=0.3, axis='x')

    # (c) Visibility at 90% noise
    ax = axes[2]
    idx_90 = noise_levels.index(0.9)
    v_at_90 = [all_results[nt][idx_90] for nt in noise_types]
    ax.bar(range(len(noise_types)), v_at_90,
           color=['#4CAF50' if v > 0.9 else '#FF9800' if v > 0.5
                  else '#F44336' for v in v_at_90],
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(noise_types)))
    ax.set_xticklabels(noise_types, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Visibility at 90%% noise')
    ax.set_title('(c) Performance at 90%% Noise\n(How much quantum info survives?)')
    ax.axhline(0.5, color='red', ls=':', label='Classical limit')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q194: Extreme Noise Resilience\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q194_extreme_noise.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ194 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
