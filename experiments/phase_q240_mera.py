# -*- coding: utf-8 -*-
"""
Phase Q240: MERA Structure Detection
========================================
Multi-scale Entanglement Renormalization Ansatz (MERA) is the
tensor network that describes critical quantum systems.
Does the Transformer's layer structure match MERA?

Test: entanglement entropy scaling at different coarse-graining levels.
MERA predicts log(L) scaling at criticality.
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

def coarse_grain(h, level):
    """Coarse-grain by averaging pairs of components."""
    for _ in range(level):
        if len(h) < 2: break
        h = (h[::2][:len(h)//2] + h[1::2][:len(h)//2]) / np.sqrt(2)
    return h

def entanglement_entropy(h, cut):
    if cut <= 0 or cut >= len(h): return 0
    psi = h / (np.linalg.norm(h) + 1e-10)
    psi_mat = psi[:cut * (len(psi) // cut)].reshape(cut, -1)
    s = np.linalg.svd(psi_mat, compute_uv=False)
    s2 = s**2; s2 = s2[s2 > 1e-12]; s2 /= s2.sum()
    return float(-np.sum(s2 * np.log2(s2)))

def main():
    print("=" * 60)
    print("Phase Q240: MERA Structure Detection")
    print("  (Is the Transformer a MERA tensor network?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    prompts = ["quantum critical point phase transition",
               "many-body entanglement structure", "conformal field theory",
               "renormalization group flow"]
    dim = 64
    cg_levels = [0, 1, 2, 3, 4]

    all_results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in [0, n_layers // 2, n_layers]:
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            level_data = []
            for cg in cg_levels:
                h_cg = coarse_grain(h.copy(), cg)
                if len(h_cg) >= 4:
                    S = entanglement_entropy(h_cg, len(h_cg) // 2)
                else:
                    S = 0
                level_data.append({'level': cg, 'dim': len(h_cg), 'entropy': round(S, 4)})

            all_results.append({
                'prompt': prompt[:30], 'layer': li, 'levels': level_data
            })

    # Check log scaling: S ~ log(L) at each layer
    fits = []
    for r in all_results:
        dims = [l['dim'] for l in r['levels'] if l['dim'] >= 4]
        Ss = [l['entropy'] for l in r['levels'] if l['dim'] >= 4]
        if len(dims) > 2 and np.std(Ss) > 0.01:
            log_dims = np.log2(dims)
            slope, _ = np.polyfit(log_dims, Ss, 1)
            fits.append(slope)

    avg_slope = np.mean(fits) if fits else 0
    if 0.1 < avg_slope < 2.0:
        verdict = "MERA-LIKE: S ~ %.2f * log(L) (consistent with critical MERA)" % avg_slope
    elif avg_slope > 0:
        verdict = "WEAK MERA: slope=%.3f" % avg_slope
    else:
        verdict = "NOT MERA: no log scaling"

    print("\n--- Summary ---")
    print("  Avg entropy-size slope: %.4f" % avg_slope)
    print("  %s" % verdict)

    results = {
        'phase': 'Q240', 'name': 'MERA Structure',
        'data': all_results, 'summary': {'avg_slope': round(avg_slope, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q240_mera.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for r in all_results[:4]:
        dims = [l['dim'] for l in r['levels'] if l['dim'] >= 2]
        Ss = [l['entropy'] for l in r['levels'] if l['dim'] >= 2]
        ax.plot(np.log2(dims), Ss, 'o-', label='L%d %s' % (r['layer'], r['prompt'][:15]), ms=4, alpha=0.7)
    ax.set_xlabel('log2(Subsystem Size)'); ax.set_ylabel('Entanglement Entropy')
    ax.set_title('Q240: MERA Structure\n%s' % verdict[:60])
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q240_mera.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ240 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
