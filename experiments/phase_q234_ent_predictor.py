# -*- coding: utf-8 -*-
"""
Phase Q234: Entanglement-Convergence Predictor
=================================================
Q223 showed r=-0.57 (entanglement anti-correlated with error).
Can we USE entanglement as an EARLY STOPPING criterion?
If entanglement plateaus, stop VQE early and save computation.
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


def measure_neg(h_np, dim_a=2, dim_b=2):
    dt = dim_a * dim_b
    h = h_np[:dt] / (np.linalg.norm(h_np[:dt]) + 1e-10)
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dt) / dt
    rho /= np.trace(rho)
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, dim_a, dim_b))
    return float(np.sum(np.abs(eigvals[eigvals < -1e-10])))


def main():
    print("=" * 60)
    print("Phase Q234: Entanglement-Convergence Predictor")
    print("  (Use entanglement as early stopping for VQE)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dims = [4, 8, 16, 32]
    n_steps = 300
    window = 20  # steps to check plateau

    all_results = []
    for dim in dims:
        print("\n--- dim=%d ---" % dim)
        rng = np.random.RandomState(42 + dim)
        H = rng.randn(dim, dim).astype(np.float32) * 0.3
        H = (H + H.T) / 2
        H_torch = torch.tensor(H, device=device)
        E_exact = float(np.linalg.eigh(H)[0][0])

        embed_layer = model.model.embed_tokens
        inp = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)

        traj = []
        early_stop_step = -1

        for step in range(n_steps):
            optimizer.zero_grad()
            out = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :dim]
            psi = h / (torch.norm(h) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward()
            optimizer.step()

            if step % 5 == 0:
                h_np = h.detach().float().cpu().numpy()
                neg = measure_neg(h_np)
                err = abs(float(E.detach()) - E_exact) * 1000
                traj.append({'step': step, 'error': round(err, 4), 'neg': round(neg, 6)})

                # Early stopping: if last `window` entanglement values are flat
                if early_stop_step < 0 and len(traj) > window:
                    recent_negs = [t['neg'] for t in traj[-window:]]
                    if np.std(recent_negs) < 0.001:
                        early_stop_step = step
                        print("  Early stop at step %d (neg plateau)" % step)

        final_err = traj[-1]['error']
        early_err = traj[early_stop_step // 5]['error'] if early_stop_step > 0 else final_err
        savings = (1 - early_stop_step / n_steps) * 100 if early_stop_step > 0 else 0

        print("  dim=%d: final=%.4f, early=%.4f, savings=%.0f%%" %
              (dim, final_err, early_err, savings))

        all_results.append({
            'dim': dim,
            'trajectory': traj,
            'early_stop_step': early_stop_step,
            'final_error': final_err,
            'early_error': round(early_err, 4),
            'savings_pct': round(savings, 1),
        })

    avg_savings = np.mean([r['savings_pct'] for r in all_results if r['savings_pct'] > 0])
    n_early = sum(1 for r in all_results if r['early_stop_step'] > 0)

    if n_early > 0 and avg_savings > 20:
        verdict = "PREDICTOR WORKS: %.0f%% compute savings (%d/%d dims)" % (avg_savings, n_early, len(all_results))
    elif n_early > 0:
        verdict = "MARGINAL: %.0f%% savings" % avg_savings
    else:
        verdict = "NO EARLY STOPPING POSSIBLE"

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q234', 'name': 'Entanglement-Convergence Predictor',
        'dimensions': all_results,
        'summary': {'avg_savings': round(avg_savings, 1) if n_early > 0 else 0, 'n_early': n_early, 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q234_ent_predictor.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, r in enumerate(all_results[:4]):
        ax = axes[idx // 2][idx % 2]
        steps = [t['step'] for t in r['trajectory']]
        errs = [t['error'] for t in r['trajectory']]
        negs = [t['neg'] for t in r['trajectory']]
        ax2 = ax.twinx()
        ax.semilogy(steps, [max(e, 0.0001) for e in errs], 'o-', color='#2196F3', ms=2, label='Error')
        ax2.plot(steps, negs, 's-', color='#E91E63', ms=2, label='Negativity')
        if r['early_stop_step'] > 0:
            ax.axvline(r['early_stop_step'], color='green', ls='--', lw=2, label='Early stop')
        ax.set_xlabel('Step'); ax.set_ylabel('Error (mHa)', color='#2196F3')
        ax2.set_ylabel('Negativity', color='#E91E63')
        ax.set_title('dim=%d (save %.0f%%)' % (r['dim'], r['savings_pct']))
        ax.grid(alpha=0.3)
    plt.suptitle('Q234: Entanglement-Convergence Predictor\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q234_ent_predictor.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ234 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
