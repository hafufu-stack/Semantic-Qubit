# -*- coding: utf-8 -*-
"""
Phase Q161: Embedding Space VQE (Continuous Optimization)
==========================================================
Q155 used brute-force prompt search (432 combos).
Q161: do GRADIENT DESCENT in continuous embedding space!

1. Start with a prompt embedding
2. Compute dE/d(embedding) via backprop
3. Move embedding toward lower energy
4. Project back to nearest token (optional)

This is VQE where the variational manifold is the LLM's embedding space!
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


def build_h2_hamiltonian(bond_length=0.74):
    dim = 16
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)
    def kron4(a, b, c, d):
        return np.kron(np.kron(np.kron(a, b), c), d)
    r = bond_length
    g0 = -0.5 - 0.2 * np.exp(-r)
    g1 = 0.2 * np.exp(-0.5 * r)
    g2 = 0.15 * np.exp(-0.3 * r)
    g3 = -0.1 * np.exp(-0.8 * r)
    H = np.real(
        g0 * kron4(I2, I2, I2, I2) +
        g1 * kron4(Z, I2, I2, I2) +
        g1 * kron4(I2, Z, I2, I2) +
        g2 * kron4(Z, Z, I2, I2) +
        g2 * kron4(I2, I2, Z, Z) +
        g3 * kron4(X, X, I2, I2) +
        g3 * kron4(I2, I2, X, X) +
        g3 * kron4(Z, I2, Z, I2) * 0.5 * g2 / g3 +
        g3 * kron4(I2, Z, I2, Z) * 0.5 * g2 / g3)
    return H


def main():
    print("=" * 60)
    print("Phase Q161: Embedding Space VQE")
    print("  (Gradient Descent in Token Space)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    dim = 16
    H_np = build_h2_hamiltonian(0.74)
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigvalsh(H_np)[0])
    print("  Exact ground state energy: %.6f" % E_exact)

    # Get embedding layer
    embed_layer = model.model.embed_tokens
    embed_weight = embed_layer.weight.detach()  # (vocab_size, hidden)

    # Start with semantic prompt
    seed_prompt = "Chemical bond ground state energy:"
    seed_ids = tok(seed_prompt, return_tensors='pt')['input_ids'].to(device)
    seed_embeds = embed_layer(seed_ids).detach().clone()  # (1, seq_len, hidden)

    # Make embeddings optimizable
    opt_embeds = seed_embeds.clone().requires_grad_(True)

    # Extraction function: run model with custom embeddings, extract psi
    def forward_and_energy(embeds):
        # Forward pass with custom embeddings (ensure float32)
        outputs = model(inputs_embeds=embeds.float(), output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :]  # last token, last layer

        # Extract psi from hidden state
        psi = h[:dim]
        psi_norm = psi / (torch.norm(psi) + 1e-10)

        # Compute energy <psi|H|psi>
        E = psi_norm @ H_torch @ psi_norm
        return E, psi_norm

    # Phase 1: Evaluate seed prompt (no optimization)
    with torch.no_grad():
        E_seed, _ = forward_and_energy(seed_embeds)
    E_seed = float(E_seed)
    print("  Seed prompt energy: %.6f (error: %.2f mHa)" %
          (E_seed, abs(E_seed - E_exact) * 1000))

    # Phase 2: Gradient descent in embedding space
    print("\n--- Embedding Space Optimization ---")
    opt_embeds = seed_embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_embeds], lr=0.001)
    
    energy_history = []
    best_E = E_seed
    best_embeds = seed_embeds.clone()

    for step in range(200):
        optimizer.zero_grad()
        E, psi = forward_and_energy(opt_embeds)
        E.backward()
        optimizer.step()

        E_val = float(E.detach())
        energy_history.append(E_val)
        
        if E_val < best_E:
            best_E = E_val
            best_embeds = opt_embeds.clone().detach()

        if step % 20 == 0:
            err = abs(E_val - E_exact) * 1000
            print("  Step %3d: E=%.6f, error=%.2f mHa" % (step, E_val, err))

    print("  Best optimized: E=%.6f, error=%.2f mHa" %
          (best_E, abs(best_E - E_exact) * 1000))

    # Phase 3: Project back to nearest tokens
    print("\n--- Token Projection ---")
    with torch.no_grad():
        best_token_ids = []
        for pos in range(best_embeds.shape[1]):
            emb = best_embeds[0, pos, :]
            dists = torch.norm(embed_weight - emb.unsqueeze(0), dim=1)
            nearest_id = int(torch.argmin(dists))
            best_token_ids.append(nearest_id)

        projected_text = tok.decode(best_token_ids)
        print("  Projected prompt: '%s'" % projected_text[:60])

        # Evaluate projected prompt
        proj_ids = torch.tensor([best_token_ids], device=device)
        proj_embeds = embed_layer(proj_ids)
        E_proj, _ = forward_and_energy(proj_embeds)
        E_proj = float(E_proj)
        print("  Projected energy: %.6f, error: %.2f mHa" %
              (E_proj, abs(E_proj - E_exact) * 1000))

    # Phase 4: Random baseline
    rand_errors = []
    for _ in range(100):
        psi_r = np.random.randn(dim)
        psi_r /= np.linalg.norm(psi_r)
        E_r = float(np.real(psi_r @ H_np @ psi_r))
        rand_errors.append(abs(E_r - E_exact) * 1000)

    # Phase 5: Multiple random starts
    print("\n--- Multi-Start Optimization ---")
    multi_results = []
    prompts_to_try = [
        "Chemical bond ground state energy:",
        "Hydrogen molecule wavefunction:",
        "Quantum chemistry calculation:",
        "The cat sat on the mat:",
        "Hello world program:",
    ]
    for p in prompts_to_try:
        p_ids = tok(p, return_tensors='pt')['input_ids'].to(device)
        p_embeds = embed_layer(p_ids).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([p_embeds], lr=0.001)

        best_this = float('inf')
        for step in range(100):
            opt.zero_grad()
            E, _ = forward_and_energy(p_embeds)
            E.backward()
            opt.step()
            E_val = float(E.detach())
            if E_val < best_this:
                best_this = E_val

        err = abs(best_this - E_exact) * 1000
        multi_results.append({
            'prompt': p[:30],
            'final_error': round(err, 2),
        })
        print("  '%s': %.2f mHa" % (p[:25], err))

    # Summary
    print("\n--- Summary ---")
    seed_err = abs(E_seed - E_exact) * 1000
    opt_err = abs(best_E - E_exact) * 1000
    proj_err = abs(E_proj - E_exact) * 1000
    rand_mean = float(np.mean(rand_errors))

    print("  Seed (no opt):      %.2f mHa" % seed_err)
    print("  Optimized (cont.):  %.2f mHa" % opt_err)
    print("  Projected (tokens): %.2f mHa" % proj_err)
    print("  Random (mean):      %.2f mHa" % rand_mean)
    print("  Improvement: %.1fx (seed->opt)" % (seed_err / max(opt_err, 0.001)))

    # Save
    results = {
        'phase': 'Q161',
        'name': 'Embedding Space VQE',
        'exact_energy': round(E_exact, 6),
        'seed_error_mha': round(seed_err, 4),
        'optimized_error_mha': round(opt_err, 4),
        'projected_error_mha': round(proj_err, 4),
        'projected_prompt': projected_text[:60],
        'random_mean_mha': round(rand_mean, 4),
        'improvement_factor': round(seed_err / max(opt_err, 0.001), 2),
        'multi_start': multi_results,
        'n_steps': 200,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q161_embedding_vqe.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(energy_history, color='#E91E63', linewidth=1.5)
    ax.axhline(E_exact, color='green', ls='--', linewidth=2, label='Exact E0')
    ax.axhline(E_seed, color='orange', ls=':', label='Seed prompt')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Energy')
    ax.set_title('(a) Embedding VQE Convergence')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    methods = ['Seed\n(no opt)', 'Embedding\nVQE', 'Token\nProjection',
               'Random\n(mean)']
    errors = [seed_err, opt_err, proj_err, rand_mean]
    colors = ['#FF9800', '#4CAF50', '#2196F3', '#F44336']
    ax.bar(range(len(methods)), errors, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Error (mHa)')
    ax.set_title('(b) Method Comparison')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    mp = [r['prompt'][:20] for r in multi_results]
    me = [r['final_error'] for r in multi_results]
    c3 = ['#4CAF50' if e < rand_mean else '#F44336' for e in me]
    ax.barh(range(len(mp)), me, color=c3, edgecolor='black', alpha=0.85)
    ax.axvline(rand_mean, color='red', ls='--', label='Random mean')
    ax.set_yticks(range(len(mp)))
    ax.set_yticklabels(mp, fontsize=7)
    ax.set_xlabel('Error (mHa)')
    ax.set_title('(c) Multi-Start: Does Seed Matter?')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='x')

    plt.suptitle('Q161: Embedding Space VQE (Words -> Quantum States via Gradient Descent)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q161_embedding_vqe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ161 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
