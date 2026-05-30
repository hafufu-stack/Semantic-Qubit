# -*- coding: utf-8 -*-
"""
Phase Q223: Entanglement During VQE Optimization
===================================================
Connect Q210 (layer entanglement) with Q211 (VQE success):
Does entanglement GROW during VQE optimization?
If so, entanglement is a computational RESOURCE being consumed.
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


def partial_transpose(rho, da, db):
    rho_pt = np.zeros_like(rho)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                for l in range(db):
                    rho_pt[i*db+k, j*db+l] = rho[i*db+l, j*db+k]
    return rho_pt


def measure_entanglement(h_np, dim_a=2, dim_b=2):
    dt = dim_a * dim_b
    h = h_np[:dt]
    h = h / (np.linalg.norm(h) + 1e-10)
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dt) / dt
    rho /= np.trace(rho)
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, dim_a, dim_b))
    neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
    log_neg = float(np.log2(2 * neg + 1)) if neg > 0 else 0
    return neg, log_neg


def main():
    print("=" * 60)
    print("Phase Q223: Entanglement During VQE Optimization")
    print("  (Is entanglement a computational resource?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dims = [4, 8, 16]
    n_steps = 200
    measure_every = 10

    all_results = []

    for dim in dims:
        print("\n--- dim=%d ---" % dim)
        rng = np.random.RandomState(42 + dim)
        H = rng.randn(dim, dim).astype(np.float32) * 0.3
        H = (H + H.T) / 2
        H_torch = torch.tensor(H, device=device)
        E_exact = float(np.linalg.eigh(H)[0][0])

        embed_layer = model.model.embed_tokens
        prompt = "ground state energy:"
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)

        trajectory = []

        for step in range(n_steps):
            optimizer.zero_grad()
            out = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :dim]
            psi = h / (torch.norm(h) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward()
            optimizer.step()

            if step % measure_every == 0:
                h_np = h.detach().float().cpu().numpy()
                neg, log_neg = measure_entanglement(h_np)
                error = abs(float(E.detach()) - E_exact) * 1000

                trajectory.append({
                    'step': step,
                    'energy_error_mHa': round(error, 4),
                    'negativity': round(neg, 6),
                    'log_negativity': round(log_neg, 6),
                })

                if step % 50 == 0:
                    print("  step=%d: error=%.4f mHa, neg=%.4f" % (step, error, neg))

        # Correlation between entanglement and accuracy
        steps = [t['step'] for t in trajectory]
        negs = [t['negativity'] for t in trajectory]
        errors = [t['energy_error_mHa'] for t in trajectory]

        # Does entanglement grow?
        if len(negs) > 2:
            trend = np.polyfit(range(len(negs)), negs, 1)[0]
        else:
            trend = 0

        # Correlation
        if len(negs) > 2 and np.std(negs) > 1e-8 and np.std(errors) > 1e-8:
            corr = float(np.corrcoef(negs, errors)[0, 1])
        else:
            corr = 0

        all_results.append({
            'dim': dim,
            'trajectory': trajectory,
            'entanglement_trend': round(trend, 8),
            'ent_error_correlation': round(corr, 4),
            'final_error': trajectory[-1]['energy_error_mHa'],
            'final_negativity': trajectory[-1]['negativity'],
        })

    # Summary
    avg_corr = np.mean([r['ent_error_correlation'] for r in all_results])
    trends = [r['entanglement_trend'] for r in all_results]
    growing = sum(1 for t in trends if t > 0.0001)

    if avg_corr < -0.3:
        verdict = "RESOURCE: entanglement anti-correlated with error (r=%.2f), %d/%d growing" % (
            avg_corr, growing, len(all_results))
    elif growing > 0:
        verdict = "GROWING ENTANGLEMENT: %d/%d dims show growth during VQE" % (growing, len(all_results))
    else:
        verdict = "NO CLEAR PATTERN: corr=%.2f" % avg_corr

    print("\n--- Summary ---")
    print("  Avg correlation: %.4f" % avg_corr)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q223',
        'name': 'Entanglement During VQE',
        'dimensions': all_results,
        'summary': {
            'avg_correlation': round(avg_corr, 4),
            'n_growing': growing,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q223_vqe_entanglement.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        traj = r['trajectory']
        steps = [t['step'] for t in traj]
        negs = [t['negativity'] for t in traj]
        errors = [t['energy_error_mHa'] for t in traj]

        ax2 = ax.twinx()
        l1 = ax.plot(steps, negs, 'o-', color='#E91E63', label='Negativity', ms=3)
        l2 = ax2.plot(steps, errors, 's--', color='#2196F3', label='Error (mHa)', ms=3, alpha=0.7)
        ax.set_xlabel('VQE Step')
        ax.set_ylabel('Negativity', color='#E91E63')
        ax2.set_ylabel('Error (mHa)', color='#2196F3')
        ax.set_title('dim=%d (corr=%.2f)' % (r['dim'], r['ent_error_correlation']))
        lines = l1 + l2
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Q223: Entanglement During VQE\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q223_vqe_entanglement.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ223 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
