# -*- coding: utf-8 -*-
"""
Phase Q249: Quantum-Native LoRA
==================================
Fine-tune LLM weights to MAXIMIZE entanglement.
Q223 proved entanglement = computational resource.
What happens when we train the model to be MORE entangled?
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
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
    da, db = 2, 2
    h = h_np[:dim] / (np.linalg.norm(h_np[:dim]) + 1e-10)
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
    rho /= np.trace(rho)
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, da, db))
    return float(np.sum(np.abs(eigvals[eigvals < -1e-10])))


class LoRAAdapter(nn.Module):
    """Minimal LoRA adapter for a single layer."""
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_dim) * 0.01)
        self.scale = 0.1

    def forward(self, x):
        return x + self.scale * (x @ self.A @ self.B)


def main():
    print("=" * 60)
    print("Phase Q249: Quantum-Native LoRA")
    print("  (Train model to maximize entanglement)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4
    hidden_dim = model.config.hidden_size

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # Add LoRA adapter to last 4 layers
    adapters = nn.ModuleList([
        LoRAAdapter(hidden_dim, hidden_dim, rank=8).to(device)
        for _ in range(4)
    ])
    optimizer = torch.optim.Adam(adapters.parameters(), lr=0.001)

    # Test prompts for VQE
    test_prompts = [
        "ground state energy of hydrogen",
        "quantum optimization problem",
        "variational eigenvalue computation",
    ]

    # Measure BEFORE training
    pre_negs = []
    for prompt in test_prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
        pre_negs.append(measure_neg(h, dim))
    avg_pre_neg = np.mean(pre_negs)
    print("  Pre-training avg negativity: %.6f" % avg_pre_neg)

    # Pre-training VQE test
    rng = np.random.RandomState(42)
    H = rng.randn(dim, dim).astype(np.float32) * 0.3
    H = (H + H.T) / 2
    H_torch = torch.tensor(H, device=device)
    E_exact = float(np.linalg.eigh(H)[0][0])

    def vqe_test(use_adapters=False):
        embed_layer = model.model.embed_tokens
        inp_ids = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp_ids).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        opt_vqe = torch.optim.Adam([opt], lr=0.005)
        for s in range(100):
            opt_vqe.zero_grad()
            out = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :]
            if use_adapters:
                for adapter in adapters:
                    h = adapter(h)
            h_dim = h[:dim]
            psi = h_dim / (torch.norm(h_dim) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward()
            opt_vqe.step()
        return abs(float(E.detach()) - E_exact) * 1000

    pre_vqe_err = vqe_test(use_adapters=False)
    print("  Pre-training VQE error: %.4f mHa" % pre_vqe_err)

    # Train LoRA to maximize entanglement
    print("\n  Training LoRA for entanglement maximization...")
    training_negs = []
    n_train_steps = 80

    for step in range(n_train_steps):
        optimizer.zero_grad()
        prompt = test_prompts[step % len(test_prompts)]
        inp = tok(prompt, return_tensors='pt').to(device)
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[n_layers][0, -1, :]

        # Apply adapters
        for adapter in adapters:
            h = adapter(h)

        # Loss: maximize entanglement (minimize -negativity proxy)
        h_dim = h[:dim]
        psi = h_dim / (torch.norm(h_dim) + 1e-10)
        rho = torch.outer(psi, psi)
        # Off-diagonal magnitude as entanglement proxy (differentiable)
        off_diag = torch.sum(torch.abs(rho)) - torch.sum(torch.abs(torch.diag(rho)))
        loss = -off_diag  # maximize off-diagonal = maximize coherence/entanglement

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            h_np = h.detach().float().cpu().numpy()
            neg = measure_neg(h_np, dim)
            training_negs.append({'step': step, 'neg': round(neg, 6)})
            print("    Step %d: loss=%.4f, neg=%.6f" % (step, float(loss), neg))

    # Measure AFTER training
    post_negs = []
    for prompt in test_prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[n_layers][0, -1, :]
            for adapter in adapters:
                h = adapter(h)
        h_np = h.float().cpu().numpy()
        post_negs.append(measure_neg(h_np, dim))
    avg_post_neg = np.mean(post_negs)
    print("\n  Post-training avg negativity: %.6f" % avg_post_neg)

    # Post-training VQE test
    post_vqe_err = vqe_test(use_adapters=True)
    print("  Post-training VQE error: %.4f mHa" % post_vqe_err)

    neg_change = (avg_post_neg - avg_pre_neg) / max(avg_pre_neg, 1e-6) * 100
    vqe_change = (pre_vqe_err - post_vqe_err) / max(pre_vqe_err, 1e-6) * 100

    if neg_change > 10 and vqe_change > 10:
        verdict = "QUANTUM LORA WORKS: +%.0f%% entanglement, +%.0f%% VQE accuracy" % (neg_change, vqe_change)
    elif neg_change > 10:
        verdict = "MORE ENTANGLED but VQE unchanged: +%.0f%% ent, %.0f%% VQE" % (neg_change, vqe_change)
    elif vqe_change > 10:
        verdict = "VQE IMPROVED but ent unchanged: %.0f%% ent, +%.0f%% VQE" % (neg_change, vqe_change)
    else:
        verdict = "MARGINAL EFFECT: %.0f%% ent, %.0f%% VQE" % (neg_change, vqe_change)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q249', 'name': 'Quantum-Native LoRA',
        'pre_neg': round(avg_pre_neg, 6), 'post_neg': round(avg_post_neg, 6),
        'pre_vqe': round(pre_vqe_err, 4), 'post_vqe': round(post_vqe_err, 4),
        'training': training_negs,
        'summary': {'neg_change_pct': round(neg_change, 1), 'vqe_change_pct': round(vqe_change, 1), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q249_quantum_lora.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    steps = [t['step'] for t in training_negs]
    negs = [t['neg'] for t in training_negs]
    ax.plot(steps, negs, 'o-', color='#E91E63', lw=2)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Negativity')
    ax.set_title('(a) Entanglement During Training'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar([0, 1], [avg_pre_neg, avg_post_neg], color=['#607D8B', '#E91E63'], edgecolor='black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Before', 'After'])
    ax.set_ylabel('Negativity'); ax.set_title('(b) Entanglement Change')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar([0, 1], [pre_vqe_err, post_vqe_err], color=['#607D8B', '#4CAF50'], edgecolor='black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Before', 'After'])
    ax.set_ylabel('VQE Error (mHa)'); ax.set_title('(c) VQE Accuracy Change')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q249: Quantum-Native LoRA\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q249_quantum_lora.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok, adapters; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ249 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
