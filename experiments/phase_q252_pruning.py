# -*- coding: utf-8 -*-
"""
Phase Q252: Entanglement-Guided Pruning
==========================================
MY IDEA: Use entanglement as a criterion for neuron importance.
Prune neurons with LOW entanglement and measure impact on VQE.
If entanglement = resource, then high-entanglement neurons are
the most important for quantum computation.
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


def main():
    print("=" * 60)
    print("Phase Q252: Entanglement-Guided Pruning")
    print("  (Are high-entanglement neurons essential?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    vqe_dim = 4

    # Step 1: Measure per-neuron entanglement contribution
    print("\n  Step 1: Computing per-neuron entanglement map...")
    prompt = "quantum ground state computation"
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    h_final = out.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
    hidden_dim = len(h_final)

    # Measure entanglement contribution of each neuron (top 64 neurons)
    n_test = min(64, hidden_dim)
    neuron_ent = np.zeros(n_test)

    for ni in range(n_test):
        h_masked = h_final[:vqe_dim].copy()
        if ni < vqe_dim:
            h_masked[ni] = 0  # Zero out this neuron
        h_masked /= np.linalg.norm(h_masked) + 1e-10
        rho = np.outer(h_masked, h_masked.conj())
        rho = 0.7 * rho + 0.3 * np.eye(vqe_dim) / vqe_dim
        rho /= np.trace(rho)
        eigvals = np.linalg.eigvalsh(partial_transpose(rho, 2, 2))
        neg_without = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))

        # Full entanglement
        h_full = h_final[:vqe_dim].copy()
        h_full /= np.linalg.norm(h_full) + 1e-10
        rho_full = np.outer(h_full, h_full.conj())
        rho_full = 0.7 * rho_full + 0.3 * np.eye(vqe_dim) / vqe_dim
        rho_full /= np.trace(rho_full)
        eigvals_full = np.linalg.eigvalsh(partial_transpose(rho_full, 2, 2))
        neg_full = float(np.sum(np.abs(eigvals_full[eigvals_full < -1e-10])))

        neuron_ent[ni] = neg_full - neg_without  # Positive = important

    # Rank neurons by entanglement contribution
    sorted_idx = np.argsort(neuron_ent)  # Low ent first
    high_ent_idx = sorted_idx[-n_test//2:]  # Top half
    low_ent_idx = sorted_idx[:n_test//2]    # Bottom half

    print("  Top 5 entanglement neurons: %s" % str(sorted_idx[-5:]))
    print("  Bottom 5 entanglement neurons: %s" % str(sorted_idx[:5]))

    # Step 2: VQE with pruning
    print("\n  Step 2: VQE with entanglement-guided pruning...")
    rng = np.random.RandomState(42)
    H = rng.randn(vqe_dim, vqe_dim).astype(np.float32) * 0.3
    H = (H + H.T) / 2
    H_torch = torch.tensor(H, device=device)
    E_exact = float(np.linalg.eigh(H)[0][0])

    def run_vqe_masked(mask_dims=None, n_steps=120):
        embed_layer = model.model.embed_tokens
        inp_ids = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp_ids).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)
        for s in range(n_steps):
            optimizer.zero_grad()
            o = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = o.hidden_states[-1][0, -1, :vqe_dim]
            if mask_dims is not None:
                mask = torch.ones(vqe_dim, device=device)
                for d in mask_dims:
                    if d < vqe_dim:
                        mask[d] = 0
                h = h * mask
            psi = h / (torch.norm(h) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward(); optimizer.step()
        return abs(float(E.detach()) - E_exact) * 1000

    # Full VQE
    err_full = run_vqe_masked()
    # Prune LOW entanglement neurons
    low_mask = [int(i) for i in low_ent_idx if i < vqe_dim]
    err_prune_low = run_vqe_masked(low_mask) if low_mask else err_full
    # Prune HIGH entanglement neurons
    high_mask = [int(i) for i in high_ent_idx if i < vqe_dim]
    err_prune_high = run_vqe_masked(high_mask) if high_mask else err_full
    # Random prune (control)
    random_mask = list(np.random.choice(vqe_dim, min(len(low_mask), vqe_dim), replace=False))
    err_prune_random = run_vqe_masked(random_mask)

    print("\n  Full: %.4f mHa" % err_full)
    print("  Prune LOW ent: %.4f mHa (should be small impact)" % err_prune_low)
    print("  Prune HIGH ent: %.4f mHa (should be big impact)" % err_prune_high)
    print("  Prune RANDOM: %.4f mHa" % err_prune_random)

    if err_prune_high > err_prune_low * 1.5:
        verdict = "ENTANGLEMENT GUIDES PRUNING: pruning high-ent neurons %.1fx worse than low-ent" % (
            err_prune_high / max(err_prune_low, 1e-6))
    else:
        verdict = "NO CLEAR GUIDANCE: high-ent prune=%.4f, low-ent prune=%.4f" % (err_prune_high, err_prune_low)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q252', 'name': 'Entanglement-Guided Pruning',
        'neuron_entanglement': [round(float(n), 6) for n in neuron_ent[:16]],
        'vqe_errors': {
            'full': round(err_full, 4),
            'prune_low_ent': round(err_prune_low, 4),
            'prune_high_ent': round(err_prune_high, 4),
            'prune_random': round(err_prune_random, 4),
        },
        'summary': {'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q252_pruning.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(range(n_test), neuron_ent, color='#E91E63', edgecolor='none', alpha=0.7)
    ax.set_xlabel('Neuron Index'); ax.set_ylabel('Entanglement Contribution')
    ax.set_title('(a) Per-Neuron Entanglement Map'); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    methods = ['Full', 'Prune Low\nEnt', 'Prune High\nEnt', 'Prune\nRandom']
    errs = [err_full, err_prune_low, err_prune_high, err_prune_random]
    colors = ['#4CAF50', '#8BC34A', '#F44336', '#FF9800']
    ax.bar(range(4), errs, color=colors, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('VQE Error (mHa)'); ax.set_title('(b) Pruning Impact')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q252: Entanglement-Guided Pruning\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q252_pruning.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ252 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
