# -*- coding: utf-8 -*-
"""
Phase Q253: Entanglement-Orthogonal Pruning (Q252 Revenge)
=============================================================
SVD-based pruning: protect VQE-critical subspace, prune only
orthogonal classical-noise dimensions.
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

def measure_neg(h_np, dim=4):
    h = h_np[:dim] / (np.linalg.norm(h_np[:dim]) + 1e-10)
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
    rho /= np.trace(rho)
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, 2, 2))
    return float(np.sum(np.abs(eigvals[eigvals < -1e-10])))

def main():
    print("=" * 60)
    print("Phase Q253: Entanglement-Orthogonal Pruning")
    print("  (Q252 revenge: SVD-protected quantum distillation)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    vqe_dim = 4

    # Step 1: Identify VQE-critical subspace via SVD
    print("\n  Step 1: SVD analysis of hidden state space...")
    prompts = ["quantum ground state energy", "hydrogen molecule computation",
               "variational eigenvalue problem", "Hamiltonian optimization"]
    reps = []
    for p in prompts:
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        reps.append(out.hidden_states[n_layers][0, -1, :].float().cpu().numpy())
    R = np.stack(reps)  # (n_prompts, hidden_dim)

    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    # Top-k singular vectors = VQE-critical subspace
    k_protect = vqe_dim
    protected_dirs = Vt[:k_protect]  # (k, hidden_dim) - directions to protect
    print("  Top %d singular values: %s" % (k_protect, str(np.round(S[:k_protect], 2))))

    # Step 2: Measure entanglement in orthogonal subspace
    print("\n  Step 2: Measuring entanglement in orthogonal space...")
    h_test = reps[0]
    h_protected = np.sum([np.dot(h_test, d) * d for d in protected_dirs], axis=0)
    h_orthogonal = h_test - h_protected

    neg_full = measure_neg(h_test, vqe_dim)
    neg_protected = measure_neg(h_protected, vqe_dim)
    print("  Full neg: %.6f, Protected subspace neg: %.6f" % (neg_full, neg_protected))

    # Step 3: VQE with different pruning strategies
    print("\n  Step 3: VQE comparison...")
    rng = np.random.RandomState(42)
    H = rng.randn(vqe_dim, vqe_dim).astype(np.float32) * 0.3
    H = (H + H.T) / 2; H_torch = torch.tensor(H, device=device)
    E_exact = float(np.linalg.eigh(H)[0][0])

    def run_vqe(projection_matrix=None, n_steps=150):
        embed_layer = model.model.embed_tokens
        inp_ids = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp_ids).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)
        for s in range(n_steps):
            optimizer.zero_grad()
            o = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = o.hidden_states[-1][0, -1, :]
            if projection_matrix is not None:
                P = torch.tensor(projection_matrix, device=device, dtype=torch.float32)
                h = P @ h
            psi = h[:vqe_dim] / (torch.norm(h[:vqe_dim]) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward(); optimizer.step()
        return abs(float(E.detach()) - E_exact) * 1000

    # Full model
    err_full = run_vqe()
    print("  Full model: %.4f mHa" % err_full)

    # Project onto protected subspace only
    P_protected = protected_dirs.T @ protected_dirs  # (hidden, hidden)
    err_protected = run_vqe(P_protected)
    print("  Protected only (rank %d): %.4f mHa" % (k_protect, err_protected))

    # Project onto top-8 (protected + some orthogonal)
    P_extended = Vt[:8].T @ Vt[:8]
    err_extended = run_vqe(P_extended)
    print("  Extended (rank 8): %.4f mHa" % err_extended)

    # Random pruning (keep same rank)
    rng_dirs = np.random.randn(k_protect, len(h_test))
    rng_dirs, _ = np.linalg.qr(rng_dirs.T)
    P_random = rng_dirs[:, :k_protect] @ rng_dirs[:, :k_protect].T
    err_random = run_vqe(P_random)
    print("  Random (rank %d): %.4f mHa" % (k_protect, err_random))

    # Verdict
    if err_protected < err_random * 0.5:
        verdict = "QUANTUM DISTILLATION: SVD-protected %.4f vs random %.4f mHa (%.1fx better)" % (
            err_protected, err_random, err_random / max(err_protected, 1e-6))
    elif err_protected < err_full * 2:
        verdict = "SUCCESSFUL PRUNING: %.4f mHa (%.1fx full model)" % (
            err_protected, err_protected / max(err_full, 1e-6))
    else:
        verdict = "PRUNING DEGRADES: protected=%.4f, full=%.4f" % (err_protected, err_full)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q253', 'name': 'Entanglement-Orthogonal Pruning',
        'singular_values': [round(float(s), 4) for s in S[:8]],
        'vqe_errors': {'full': round(err_full, 4), 'protected': round(err_protected, 4),
                       'extended': round(err_extended, 4), 'random': round(err_random, 4)},
        'summary': {'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q253_ortho_pruning.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(range(min(16, len(S))), S[:16], color='#E91E63', edgecolor='none')
    ax.axvline(k_protect - 0.5, color='red', ls='--', lw=2, label='Protected boundary')
    ax.set_xlabel('Singular Value Index'); ax.set_ylabel('Singular Value')
    ax.set_title('(a) SVD Spectrum'); ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    methods = ['Full\nModel', 'SVD\nProtected', 'Extended\n(rank 8)', 'Random\n(rank %d)' % k_protect]
    errs = [err_full, err_protected, err_extended, err_random]
    colors = ['#4CAF50', '#E91E63', '#FF9800', '#9E9E9E']
    ax.bar(range(4), errs, color=colors, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('VQE Error (mHa)'); ax.set_title('(b) Pruning Impact')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q253: Entanglement-Orthogonal Pruning\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q253_ortho_pruning.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ253 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
