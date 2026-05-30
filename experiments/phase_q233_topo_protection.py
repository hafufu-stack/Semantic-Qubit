# -*- coding: utf-8 -*-
"""
Phase Q233: Topological Protection Test
==========================================
Q229 showed ALL layers are topological. If true, the computation
should be PROTECTED against noise (topological error correction).

Test: inject increasing noise and measure if accuracy degrades
gracefully (topological) or catastrophically (non-topological).
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


def run_noisy_vqe(model, tok, device, H_np, dim, noise_level, n_steps=150, lr=0.005):
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigh(H_np)[0][0])
    embed_layer = model.model.embed_tokens
    prompt = "ground state:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        noisy_opt = opt + torch.randn_like(opt) * noise_level
        out = model(inputs_embeds=noisy_opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()

    with torch.no_grad():
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E_final = float(torch.dot(psi, H_torch @ psi))

    return abs(E_final - E_exact) * 1000


def main():
    print("=" * 60)
    print("Phase Q233: Topological Protection Test")
    print("  (Does topological phase protect computation?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dims = [4, 8, 16]
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    all_results = []
    for dim in dims:
        print("\n--- dim=%d ---" % dim)
        rng = np.random.RandomState(42 + dim)
        H = rng.randn(dim, dim).astype(np.float32) * 0.3
        H = (H + H.T) / 2

        dim_results = []
        for nl in noise_levels:
            err = run_noisy_vqe(model, tok, device, H, dim, nl)
            print("  noise=%.2f: error=%.4f mHa" % (nl, err))
            dim_results.append({'noise': nl, 'error_mHa': round(err, 4)})

        # Compute protection: how much does error grow per unit noise?
        errors = [r['error_mHa'] for r in dim_results]
        clean_err = max(errors[0], 0.0001)
        degradation = [e / clean_err for e in errors]

        # Threshold: noise at which error exceeds 1.6 mHa (chemical accuracy)
        threshold_noise = -1
        for r in dim_results:
            if r['error_mHa'] > 1.6:
                threshold_noise = r['noise']
                break

        all_results.append({
            'dim': dim,
            'results': dim_results,
            'degradation': [round(d, 2) for d in degradation],
            'threshold_noise': threshold_noise,
        })

    # Summary
    thresholds = [r['threshold_noise'] for r in all_results if r['threshold_noise'] > 0]
    avg_threshold = np.mean(thresholds) if thresholds else float('inf')

    if avg_threshold > 0.2:
        verdict = "STRONG PROTECTION: tolerates noise up to %.2f before losing chem accuracy" % avg_threshold
    elif avg_threshold > 0.05:
        verdict = "MODERATE PROTECTION: threshold=%.2f" % avg_threshold
    else:
        verdict = "WEAK PROTECTION: threshold=%.3f" % avg_threshold

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q233',
        'name': 'Topological Protection',
        'dimensions': all_results,
        'summary': {'avg_threshold': round(avg_threshold, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q233_topo_protection.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for r in all_results:
        nl = [d['noise'] for d in r['results']]
        errs = [d['error_mHa'] for d in r['results']]
        ax.semilogy(nl, [max(e, 0.0001) for e in errs], 'o-', label='dim=%d' % r['dim'], lw=2, ms=5)
    ax.axhline(1.6, color='red', ls='--', label='Chemical accuracy (1.6 mHa)')
    ax.set_xlabel('Noise Level'); ax.set_ylabel('Error (mHa)')
    ax.set_title('Q233: Topological Protection\n%s' % verdict[:60])
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q233_topo_protection.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ233 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
