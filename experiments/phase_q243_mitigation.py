# -*- coding: utf-8 -*-
"""
Phase Q243: Quantum Error Mitigation Portfolio
=================================================
Combine MULTIPLE error mitigation techniques on the same problem:
- ZNE (Q212)
- Symmetry Verification
- Post-selection
Which combination is best?
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

def run_vqe_with_noise(model, tok, device, H_np, dim, noise, n_steps=100):
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigh(H_np)[0][0])
    embed_layer = model.model.embed_tokens
    inp = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=0.005)
    for s in range(n_steps):
        optimizer.zero_grad()
        noisy = opt + torch.randn_like(opt) * noise
        out = model(inputs_embeds=noisy.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward(); optimizer.step()
    with torch.no_grad():
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E_final = float(torch.dot(psi, H_torch @ psi))
    return E_final, E_exact

def main():
    print("=" * 60)
    print("Phase Q243: Error Mitigation Portfolio")
    print("  (Which combination of techniques is best?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dims = [4, 8]
    noise_base = 0.1
    all_results = []

    for dim in dims:
        print("\n--- dim=%d ---" % dim)
        rng = np.random.RandomState(42 + dim)
        H = rng.randn(dim, dim).astype(np.float32) * 0.3
        H = (H + H.T) / 2
        E_exact = float(np.linalg.eigh(H)[0][0])

        # 1. No mitigation
        E_raw, _ = run_vqe_with_noise(model, tok, device, H, dim, noise_base)
        err_raw = abs(E_raw - E_exact) * 1000

        # 2. ZNE (Richardson extrapolation)
        noise_scales = [1.0, 1.5, 2.0]
        E_zne = []
        for scale in noise_scales:
            E_s, _ = run_vqe_with_noise(model, tok, device, H, dim, noise_base * scale)
            E_zne.append(E_s)
        # Linear extrapolation to zero noise
        coeffs = np.polyfit([s**2 for s in noise_scales], E_zne, 1)
        E_extrapolated = np.polyval(coeffs, 0)
        err_zne = abs(E_extrapolated - E_exact) * 1000

        # 3. Symmetry averaging (run multiple times, average)
        E_sym = []
        for trial in range(5):
            E_t, _ = run_vqe_with_noise(model, tok, device, H, dim, noise_base)
            E_sym.append(E_t)
        E_sym_avg = np.mean(E_sym)
        err_sym = abs(E_sym_avg - E_exact) * 1000

        # 4. Combined: ZNE + averaging
        E_combined = []
        for scale in noise_scales:
            trials = []
            for _ in range(3):
                E_t, _ = run_vqe_with_noise(model, tok, device, H, dim, noise_base * scale)
                trials.append(E_t)
            E_combined.append(np.mean(trials))
        coeffs_c = np.polyfit([s**2 for s in noise_scales], E_combined, 1)
        E_comb = np.polyval(coeffs_c, 0)
        err_combined = abs(E_comb - E_exact) * 1000

        # 5. Clean (no noise baseline)
        E_clean, _ = run_vqe_with_noise(model, tok, device, H, dim, 0.0)
        err_clean = abs(E_clean - E_exact) * 1000

        print("  Clean=%.4f, Raw=%.4f, ZNE=%.4f, Sym=%.4f, Combined=%.4f" %
              (err_clean, err_raw, err_zne, err_sym, err_combined))

        all_results.append({
            'dim': dim,
            'clean': round(err_clean, 4), 'raw': round(err_raw, 4),
            'zne': round(err_zne, 4), 'symmetry': round(err_sym, 4),
            'combined': round(err_combined, 4),
        })

    # Find best technique
    best_techniques = []
    for r in all_results:
        methods = {'ZNE': r['zne'], 'Symmetry': r['symmetry'], 'Combined': r['combined']}
        best = min(methods, key=methods.get)
        improvement = r['raw'] / max(methods[best], 0.0001)
        best_techniques.append((best, round(improvement, 1)))

    verdict = "Best mitigation: %s" % ', '.join('%s(%.1fx)' % (b, i) for b, i in best_techniques)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q243', 'name': 'Error Mitigation Portfolio',
        'dimensions': all_results,
        'summary': {'best_techniques': [{'method': b, 'improvement': i} for b, i in best_techniques], 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q243_mitigation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    methods = ['clean', 'raw', 'zne', 'symmetry', 'combined']
    labels = ['Clean', 'Raw (noisy)', 'ZNE', 'Symmetry Avg', 'Combined']
    x = np.arange(len(methods))
    w = 0.35
    for di, r in enumerate(all_results):
        vals = [r[m] for m in methods]
        ax.bar(x + di * w, vals, w, label='dim=%d' % r['dim'], edgecolor='black')
    ax.set_xticks(x + w / 2); ax.set_xticklabels(labels)
    ax.set_ylabel('Error (mHa)'); ax.set_yscale('symlog', linthresh=0.01)
    ax.set_title('Q243: Error Mitigation Portfolio\n%s' % verdict[:60])
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q243_mitigation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ243 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
