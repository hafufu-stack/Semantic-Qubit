# -*- coding: utf-8 -*-
"""
Phase Q211: LLM-Seeded High-Dim VQE
======================================
Combine Q203's "LLM chemical intuition" with Q208's high-dim problem.
Q208 showed dim^5.58 scaling with random init. Can LLM-seeding break
the "curse of dimensionality"?
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


def build_hamiltonian(dim, seed=42):
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim) * 0.3
    H = (H + H.T) / 2
    H[np.diag_indices_from(H)] = np.sort(rng.randn(dim)) * 0.5
    return H


def run_vqe(model, tok, device, H_np, dim, seed_type='llm',
            n_steps=300, lr=0.005, rng_seed=0):
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    eigvals = np.linalg.eigh(H_np)[0]
    E_exact = eigvals[0]
    embed_layer = model.model.embed_tokens

    if seed_type == 'llm':
        prompt = "ground state energy dimension %d:" % dim
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp).detach().clone()
    else:
        prompt = "random:"
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp).detach().clone()
        rng = np.random.RandomState(rng_seed)
        with torch.no_grad():
            embeds[0, -1, :dim] = torch.tensor(
                rng.randn(dim).astype(np.float32), device=device) * 0.1

    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=lr)
    history = []

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()
        history.append(float(E.detach()))

    error_mha = abs(history[-1] - E_exact) * 1000
    conv = next((i for i, e in enumerate(history) if abs(e - E_exact) < 0.002), n_steps)
    return error_mha, conv, history


def main():
    print("=" * 60)
    print("Phase Q211: LLM-Seeded High-Dim VQE")
    print("  (Breaking the curse of dimensionality)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    dimensions = [4, 8, 16, 32, 64]
    n_random_trials = 3
    n_steps = 300
    all_results = []

    for dim in dimensions:
        print("\n--- Dimension %d ---" % dim)
        H = build_hamiltonian(dim, seed=42 + dim)

        # LLM-seeded
        llm_err, llm_conv, llm_hist = run_vqe(
            model, tok, device, H, dim, seed_type='llm', n_steps=n_steps)
        print("  LLM-seeded: error=%.4f mHa, conv@%d" % (llm_err, llm_conv))

        # Random-seeded (multiple trials)
        rand_errs, rand_convs = [], []
        rand_hists = []
        for trial in range(n_random_trials):
            r_err, r_conv, r_hist = run_vqe(
                model, tok, device, H, dim, seed_type='random',
                n_steps=n_steps, rng_seed=trial)
            rand_errs.append(r_err)
            rand_convs.append(r_conv)
            rand_hists.append(r_hist)

        avg_rand_err = np.mean(rand_errs)
        avg_rand_conv = np.mean(rand_convs)
        speedup = avg_rand_conv / max(llm_conv, 1)
        improvement = avg_rand_err / max(llm_err, 0.0001)

        print("  Random (avg): error=%.4f mHa, conv@%.0f" %
              (avg_rand_err, avg_rand_conv))
        print("  Speedup: %.1fx convergence, %.1fx error improvement" %
              (speedup, improvement))

        all_results.append({
            'dim': dim,
            'llm': {'error_mHa': round(llm_err, 4), 'conv': llm_conv,
                    'history': [round(h, 6) for h in llm_hist]},
            'random': {'avg_error_mHa': round(avg_rand_err, 4),
                       'avg_conv': round(avg_rand_conv, 1),
                       'errors': [round(e, 4) for e in rand_errs]},
            'speedup': round(speedup, 2),
            'error_improvement': round(improvement, 2),
        })

    # Scaling analysis
    dims_arr = np.array([r['dim'] for r in all_results], dtype=float)
    llm_errs = np.array([r['llm']['error_mHa'] for r in all_results])
    rand_errs_arr = np.array([r['random']['avg_error_mHa'] for r in all_results])

    # Fit scaling for LLM-seeded
    valid = llm_errs > 0
    if valid.sum() > 2:
        alpha_llm = np.polyfit(np.log(dims_arr[valid]), np.log(llm_errs[valid]), 1)[0]
    else:
        alpha_llm = 0

    valid_r = rand_errs_arr > 0
    if valid_r.sum() > 2:
        alpha_rand = np.polyfit(np.log(dims_arr[valid_r]), np.log(rand_errs_arr[valid_r]), 1)[0]
    else:
        alpha_rand = 0

    if alpha_llm < alpha_rand * 0.5:
        verdict = "CURSE BROKEN: LLM scales dim^%.1f vs Random dim^%.1f" % (alpha_llm, alpha_rand)
    elif alpha_llm < alpha_rand:
        verdict = "IMPROVED: LLM dim^%.1f vs Random dim^%.1f" % (alpha_llm, alpha_rand)
    else:
        verdict = "NO IMPROVEMENT: Both scale similarly"

    print("\n--- Summary ---")
    print("  LLM scaling: dim^%.2f" % alpha_llm)
    print("  Random scaling: dim^%.2f" % alpha_rand)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q211',
        'name': 'LLM-Seeded High-Dim VQE',
        'dimensions': all_results,
        'scaling': {
            'llm_exponent': round(alpha_llm, 4),
            'random_exponent': round(alpha_rand, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q211_highdim_vqe.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Error scaling comparison
    ax = axes[0]
    ax.loglog(dims_arr, llm_errs + 1e-6, 'o-', color='#E91E63', lw=2, label='LLM-seeded')
    ax.loglog(dims_arr, rand_errs_arr + 1e-6, 's--', color='#607D8B', lw=2, label='Random')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Error (mHa)')
    ax.set_title('(a) Error Scaling')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # (b) Convergence comparison
    ax = axes[1]
    llm_convs = [r['llm']['conv'] for r in all_results]
    rand_convs_avg = [r['random']['avg_conv'] for r in all_results]
    ax.plot(dims_arr, llm_convs, 'o-', color='#E91E63', lw=2, label='LLM-seeded')
    ax.plot(dims_arr, rand_convs_avg, 's--', color='#607D8B', lw=2, label='Random')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Steps to Converge')
    ax.set_title('(b) Convergence Speed')
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Speedup
    ax = axes[2]
    speedups = [r['speedup'] for r in all_results]
    colors = ['#4CAF50' if s > 1 else '#F44336' for s in speedups]
    ax.bar(range(len(dims_arr)), speedups, color=colors, edgecolor='black')
    ax.axhline(1, color='gray', ls='--')
    ax.set_xticks(range(len(dims_arr)))
    ax.set_xticklabels([str(int(d)) for d in dims_arr])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('(c) LLM Speedup Factor')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q211: LLM-Seeded High-Dim VQE\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q211_highdim_vqe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ211 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
