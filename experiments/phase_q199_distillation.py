# -*- coding: utf-8 -*-
"""
Phase Q199: Quantum Entanglement Distillation (Opus Original)
================================================================
In physical QC, noisy entanglement can be "distilled" into
fewer but purer entangled pairs using LOCC (Local Operations +
Classical Communication).

Can LLM distill entanglement?
1. Create noisy entangled soul vector pairs
2. Use LLM's layers as "distillation rounds"
3. Measure: does entanglement purity improve layer-by-layer?

If yes -> LLM performs autonomous entanglement purification,
a key primitive for quantum repeaters and quantum internet.
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


def main():
    print("=" * 60)
    print("Phase Q199: Quantum Entanglement Distillation")
    print("  (Can LLM purify noisy entanglement?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # Train two entangled soul vectors
    print("  Training soul vectors...")
    vec_blue = train_soul(model, tok,
                         [("The sky is", "blue"), ("The ocean is", "blue")],
                         device, layer=INJECT_LAYER, seed=42)
    vec_green = train_soul(model, tok,
                          [("The grass is", "green"), ("Leaves are", "green")],
                          device, layer=INJECT_LAYER, seed=99)

    target_id = tok.encode("blue")[-1]
    prompt = "The sky is"
    inp = tok(prompt, return_tensors='pt').to(device)

    # Create entangled superposition with varying noise levels
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    all_layer_purities = {}

    for noise_lvl in noise_levels:
        print("\n--- Noise level: %.1f ---" % noise_lvl)

        # Create noisy entangled state: superposition + random noise
        phi = np.pi / 4  # Equal superposition
        vec_entangled = np.cos(phi) * vec_blue + np.sin(phi) * vec_green
        scale = vec_entangled.norm()

        # Add noise
        torch.manual_seed(42)
        noise = torch.randn(hidden_size, device=device)
        vec_noisy = vec_entangled + noise * noise_lvl * scale
        vec_noisy = vec_noisy / vec_noisy.norm() * scale

        # Track "purity" through layers
        # Purity = cosine similarity with clean entangled state
        layer_purities = []
        layer_visibilities = []

        # Hook at injection layer, monitor at all subsequent layers
        recorded_states = {}

        def make_inject(v):
            def hook(m, i, o):
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            return hook

        def make_record(layer_idx):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                recorded_states[layer_idx] = h[0, -1, :].detach().float().clone()
            return hook

        # Run noisy version
        handles = []
        handles.append(model.model.layers[INJECT_LAYER].register_forward_hook(
            make_inject(vec_noisy)))
        for li in range(INJECT_LAYER + 1, n_layers):
            handles.append(model.model.layers[li].register_forward_hook(
                make_record(li)))

        with torch.no_grad():
            out_noisy = model(**inp)

        for h in handles:
            h.remove()

        noisy_states = dict(recorded_states)

        # Run clean version
        recorded_states = {}
        handles = []
        handles.append(model.model.layers[INJECT_LAYER].register_forward_hook(
            make_inject(vec_entangled)))
        for li in range(INJECT_LAYER + 1, n_layers):
            handles.append(model.model.layers[li].register_forward_hook(
                make_record(li)))

        with torch.no_grad():
            out_clean = model(**inp)

        for h in handles:
            h.remove()

        clean_states = dict(recorded_states)

        # Compute purity at each layer
        for li in sorted(noisy_states.keys()):
            if li in clean_states:
                noisy_h = noisy_states[li]
                clean_h = clean_states[li]
                cos = float(torch.dot(noisy_h, clean_h) /
                           (torch.norm(noisy_h) * torch.norm(clean_h) + 1e-10))
                layer_purities.append({
                    'layer': li,
                    'cosine_purity': round(abs(cos), 4),
                    'noisy_norm': round(float(torch.norm(noisy_h)), 4),
                    'clean_norm': round(float(torch.norm(clean_h)), 4),
                })

        # Check if purity increases (distillation)
        purities = [lp['cosine_purity'] for lp in layer_purities]
        if len(purities) >= 2:
            initial_purity = purities[0]
            final_purity = purities[-1]
            distillation_ratio = final_purity / (initial_purity + 1e-10)
        else:
            initial_purity = 0
            final_purity = 0
            distillation_ratio = 1.0

        all_layer_purities[noise_lvl] = {
            'layers': layer_purities,
            'initial_purity': round(initial_purity, 4),
            'final_purity': round(final_purity, 4),
            'distillation_ratio': round(distillation_ratio, 4),
        }

        print("  Purity: %.4f -> %.4f (%.2fx distillation)" %
              (initial_purity, final_purity, distillation_ratio))

    # Summary
    print("\n--- Distillation Summary ---")
    dist_ratios = [v['distillation_ratio'] for v in all_layer_purities.values()
                   if v['distillation_ratio'] > 0]
    avg_dist = float(np.mean(dist_ratios)) if dist_ratios else 0
    n_distilled = sum(1 for v in all_layer_purities.values()
                      if v['distillation_ratio'] > 1.0)

    for nl, data in all_layer_purities.items():
        status = "PURIFIED" if data['distillation_ratio'] > 1.0 else "degraded"
        print("  noise=%.1f: %.4f -> %.4f (%s, %.2fx)" %
              (nl, data['initial_purity'], data['final_purity'],
               status, data['distillation_ratio']))

    if n_distilled >= 4:
        verdict = "STRONG DISTILLATION: %d/%d noise levels purified (avg %.2fx)" % (
            n_distilled, len(noise_levels), avg_dist)
    elif n_distilled >= 2:
        verdict = "PARTIAL DISTILLATION: %d/%d purified" % (
            n_distilled, len(noise_levels))
    else:
        verdict = "WEAK: %d/%d purified (avg ratio %.2f)" % (
            n_distilled, len(noise_levels), avg_dist)

    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q199',
        'name': 'Quantum Entanglement Distillation',
        'hidden_size': hidden_size,
        'distillation': {str(k): v for k, v in all_layer_purities.items()},
        'summary': {
            'n_distilled': n_distilled,
            'avg_distillation_ratio': round(avg_dist, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q199_distillation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Purity through layers for each noise level
    ax = axes[0]
    palette = plt.cm.viridis(np.linspace(0.2, 0.9, len(noise_levels)))
    for i, (nl, data) in enumerate(all_layer_purities.items()):
        layers = [lp['layer'] for lp in data['layers']]
        purities = [lp['cosine_purity'] for lp in data['layers']]
        ax.plot(layers, purities, 'o-', color=palette[i], linewidth=1.5,
                markersize=3, label='noise=%.1f' % nl)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Purity (cosine with clean)')
    ax.set_title('(a) Entanglement Purity Through Layers\n(Rising = Distillation)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Distillation ratio
    ax = axes[1]
    nls = list(all_layer_purities.keys())
    ratios = [all_layer_purities[nl]['distillation_ratio'] for nl in nls]
    colors = ['#4CAF50' if r > 1 else '#F44336' for r in ratios]
    ax.bar(range(len(nls)), ratios, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='red', ls='--', label='No change')
    ax.set_xticks(range(len(nls)))
    ax.set_xticklabels(['%.0f%%' % (100*nl) for nl in nls])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Distillation Ratio (final/initial purity)')
    ax.set_title('(b) Distillation Ratio\n(>1 = purification, <1 = degradation)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Initial vs Final purity
    ax = axes[2]
    init_p = [all_layer_purities[nl]['initial_purity'] for nl in nls]
    final_p = [all_layer_purities[nl]['final_purity'] for nl in nls]
    x = np.arange(len(nls))
    ax.bar(x - 0.15, init_p, 0.3, color='#FF9800', edgecolor='black',
           alpha=0.85, label='Initial (Layer %d)' % (INJECT_LAYER + 1))
    ax.bar(x + 0.15, final_p, 0.3, color='#2196F3', edgecolor='black',
           alpha=0.85, label='Final (Layer %d)' % (n_layers - 1))
    ax.set_xticks(x)
    ax.set_xticklabels(['%.0f%%' % (100*nl) for nl in nls])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Purity')
    ax.set_title('(c) Purity: Before vs After LLM Layers')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q199: Quantum Entanglement Distillation\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q199_distillation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ199 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
