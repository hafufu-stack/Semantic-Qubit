# -*- coding: utf-8 -*-
"""
Phase Q242: Advantage vs Problem Conditioning
================================================
Q226 showed 9/9 wins. But WHY? Is it because S-Qubit handles
ill-conditioned problems better? Correlate advantage with
condition number of H.
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

def run_vqe(model, tok, device, H_np, dim, n_steps=150):
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigh(H_np)[0][0])
    embed_layer = model.model.embed_tokens
    inp = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=0.005)
    for s in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward(); optimizer.step()
    return max(abs(float(E.detach()) - E_exact) * 1000, 1e-6)

def run_sa(H_np, dim, seed=42):
    rng = np.random.RandomState(seed)
    E_exact = float(np.linalg.eigh(H_np)[0][0])
    psi = rng.randn(dim).astype(np.float64)
    psi /= np.linalg.norm(psi)
    E_best = float(psi @ H_np @ psi)
    for s in range(3000):
        T = max(0.01, 1.0 - s / 3000)
        psi_new = psi + rng.randn(dim) * 0.1
        psi_new /= np.linalg.norm(psi_new)
        E_new = float(psi_new @ H_np @ psi_new)
        if E_new < E_best or rng.rand() < np.exp(-(E_new - E_best) / max(T, 1e-10)):
            psi = psi_new; E_best = min(E_best, E_new)
    return max(abs(E_best - E_exact) * 1000, 1e-6)

def main():
    print("=" * 60)
    print("Phase Q242: Advantage vs Problem Conditioning")
    print("  (WHY does S-Qubit win?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    dim = 8

    conditions, advantages = [], []
    all_data = []

    for seed in range(20):
        rng = np.random.RandomState(seed * 13 + 7)
        # Vary condition number
        eigvals = rng.uniform(0.01, 1.0 + seed * 0.5, dim)
        eigvals = np.sort(eigvals)
        Q = np.linalg.qr(rng.randn(dim, dim))[0]
        H = (Q @ np.diag(eigvals) @ Q.T).astype(np.float32)

        cond = float(np.max(eigvals) / np.min(eigvals))
        gap = float(eigvals[1] - eigvals[0])

        vqe_err = run_vqe(model, tok, device, H, dim)
        sa_err = run_sa(H, dim, seed)
        ratio = sa_err / vqe_err

        conditions.append(cond)
        advantages.append(ratio)
        all_data.append({
            'seed': seed, 'condition': round(cond, 2), 'gap': round(gap, 4),
            'vqe_err': round(vqe_err, 4), 'sa_err': round(sa_err, 4),
            'advantage': round(ratio, 2),
        })
        if seed % 5 == 0:
            print("  seed=%d: cond=%.1f, gap=%.4f, ratio=%.0fx" % (seed, cond, gap, ratio))

    # Correlation
    corr_cond = float(np.corrcoef(conditions, advantages)[0, 1]) if np.std(conditions) > 0 else 0
    gaps = [d['gap'] for d in all_data]
    corr_gap = float(np.corrcoef(gaps, advantages)[0, 1]) if np.std(gaps) > 0 else 0

    verdict = "Advantage correlates with: condition r=%.2f, gap r=%.2f" % (corr_cond, corr_gap)
    if corr_cond > 0.3:
        verdict += " -> S-Qubit EXCELS on ill-conditioned problems"
    elif corr_gap < -0.3:
        verdict += " -> S-Qubit EXCELS on small-gap problems"

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q242', 'name': 'Advantage vs Conditioning',
        'data': all_data,
        'summary': {'corr_condition': round(corr_cond, 4), 'corr_gap': round(corr_gap, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q242_conditioning.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(conditions, advantages, c='#E91E63', s=40, edgecolors='black')
    ax.set_xlabel('Condition Number'); ax.set_ylabel('Advantage (SA/VQE)')
    ax.set_title('(a) vs Condition (r=%.2f)' % corr_cond); ax.set_yscale('log'); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.scatter(gaps, advantages, c='#2196F3', s=40, edgecolors='black')
    ax.set_xlabel('Spectral Gap'); ax.set_ylabel('Advantage (SA/VQE)')
    ax.set_title('(b) vs Gap (r=%.2f)' % corr_gap); ax.set_yscale('log'); ax.grid(alpha=0.3)
    plt.suptitle('Q242: Advantage vs Conditioning\n%s' % verdict[:70], fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q242_conditioning.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ242 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
