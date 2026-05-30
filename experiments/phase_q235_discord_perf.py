# -*- coding: utf-8 -*-
"""
Phase Q235: Discord-Performance Correlation
=============================================
Q225 showed 100% discord. Does the AMOUNT of discord predict
how well VQE performs? If discord ~ accuracy, discord is a
quantum computational resource distinct from entanglement.
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


def vn_entropy(rho):
    eigvals = np.real(np.linalg.eigvalsh(rho))
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log2(eigvals))) if len(eigvals) > 0 else 0

def partial_trace_b(rho, da, db):
    rho_a = np.zeros((da, da), dtype=complex)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                rho_a[i, j] += rho[i*db+k, j*db+k]
    return rho_a

def partial_trace_a(rho, da, db):
    rho_b = np.zeros((db, db), dtype=complex)
    for i in range(db):
        for j in range(db):
            for k in range(da):
                rho_b[i, j] += rho[k*db+i, k*db+j]
    return rho_b

def quick_discord(rho, da, db):
    S_ab = vn_entropy(rho)
    S_a = vn_entropy(partial_trace_b(rho, da, db))
    S_b = vn_entropy(partial_trace_a(rho, da, db))
    I_ab = S_a + S_b - S_ab
    # Simplified classical correlation
    rng = np.random.RandomState(42)
    max_J = 0
    for _ in range(10):
        v = rng.randn(db) + 1j * rng.randn(db)
        v /= np.linalg.norm(v)
        proj_0 = np.outer(v, v.conj())
        proj_1 = np.eye(db) - proj_0
        S_cond = 0
        for proj in [proj_0, proj_1]:
            M = np.kron(np.eye(da), proj)
            rho_after = M @ rho @ M
            p = max(np.real(np.trace(rho_after)), 1e-12)
            if p > 1e-10:
                rho_cond = rho_after / p
                S_cond += p * vn_entropy(partial_trace_b(rho_cond, da, db))
        max_J = max(max_J, S_a - S_cond)
    return max(0, I_ab - max_J)


def main():
    print("=" * 60)
    print("Phase Q235: Discord-Performance Correlation")
    print("  (Is discord a computational resource?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Run VQE with different Hamiltonians, measure discord at convergence
    n_hamiltonians = 12
    dim = 4
    da, db = 2, 2

    discord_vals = []
    vqe_errors = []

    for hi in range(n_hamiltonians):
        rng = np.random.RandomState(hi * 7 + 1)
        H = rng.randn(dim, dim).astype(np.float32) * 0.3 * (1 + hi * 0.2)
        H = (H + H.T) / 2
        H_torch = torch.tensor(H, device=device)
        E_exact = float(np.linalg.eigh(H)[0][0])

        embed_layer = model.model.embed_tokens
        inp = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)

        for step in range(150):
            optimizer.zero_grad()
            out = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :dim]
            psi = h / (torch.norm(h) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward()
            optimizer.step()

        h_np = h.detach().float().cpu().numpy()
        h_norm = h_np[:dim] / (np.linalg.norm(h_np[:dim]) + 1e-10)
        rho = np.outer(h_norm, h_norm.conj())
        rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
        rho /= np.trace(rho)

        disc = quick_discord(rho, da, db)
        err = abs(float(E.detach()) - E_exact) * 1000

        discord_vals.append(disc)
        vqe_errors.append(err)
        print("  H%d: discord=%.4f, error=%.4f mHa" % (hi, disc, err))

    # Correlation
    if np.std(discord_vals) > 1e-8 and np.std(vqe_errors) > 1e-8:
        corr = float(np.corrcoef(discord_vals, vqe_errors)[0, 1])
    else:
        corr = 0

    if corr < -0.3:
        verdict = "DISCORD IS RESOURCE: r=%.2f (more discord -> lower error)" % corr
    elif corr > 0.3:
        verdict = "DISCORD ANTI-RESOURCE: r=%.2f (more discord -> higher error)" % corr
    else:
        verdict = "NO CORRELATION: r=%.2f" % corr

    print("\n--- Summary ---")
    print("  Correlation(discord, error): %.4f" % corr)
    print("  %s" % verdict)

    results = {
        'phase': 'Q235', 'name': 'Discord-Performance Correlation',
        'data': [{'H_idx': i, 'discord': round(discord_vals[i], 6), 'error_mHa': round(vqe_errors[i], 4)}
                 for i in range(n_hamiltonians)],
        'summary': {'correlation': round(corr, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q235_discord_perf.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(discord_vals, vqe_errors, c='#E91E63', s=60, edgecolors='black', zorder=5)
    if abs(corr) > 0.1:
        z = np.polyfit(discord_vals, vqe_errors, 1)
        x_fit = np.linspace(min(discord_vals), max(discord_vals), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), '--', color='gray', label='r=%.2f' % corr)
    ax.set_xlabel('Quantum Discord'); ax.set_ylabel('VQE Error (mHa)')
    ax.set_title('Q235: Discord-Performance Correlation\n%s' % verdict[:60])
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q235_discord_perf.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ235 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
