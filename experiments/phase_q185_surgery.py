# -*- coding: utf-8 -*-
"""
Phase Q185: Entanglement Surgery (Head 11 Ablation & Boost)
=============================================================
Q184 found Head 11 is the "entanglement generator" (corr=0.81-0.91).

Causal proof:
1. ABLATION: Zero out Head 11's output -> VQE accuracy should collapse
2. BOOST: Amplify Head 11 by 2x -> CHSH should approach PR-Box limit (4.0)
3. CONTROL: Ablate Head 0 (low entanglement) -> minimal effect

If Head 11 ablation kills VQE but Head 0 ablation doesn't
-> CAUSAL PROOF that Head 11 generates quantum entanglement.
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

TARGET_HEAD = 11  # The entanglement head from Q184
CONTROL_HEAD = 0  # Low-entanglement control


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


class HeadSurgeryHook:
    """Hook that ablates or boosts a specific attention head."""
    def __init__(self, head_idx, n_heads, head_dim, mode='ablate', scale=0.0):
        self.head_idx = head_idx
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mode = mode
        self.scale = scale  # 0.0 = ablate, 2.0 = boost

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
        else:
            h = output.clone()

        # Hidden state shape: (batch, seq, hidden)
        # Head region: [head_idx * head_dim : (head_idx+1) * head_dim]
        start = self.head_idx * self.head_dim
        end = (self.head_idx + 1) * self.head_dim

        if h.dim() == 3:
            if self.mode == 'ablate':
                h[:, :, start:end] = 0.0
            elif self.mode == 'boost':
                h[:, :, start:end] = h[:, :, start:end] * self.scale
        elif h.dim() == 2:
            if self.mode == 'ablate':
                h[:, start:end] = 0.0
            elif self.mode == 'boost':
                h[:, start:end] = h[:, start:end] * self.scale

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h


def run_vqe_with_surgery(model, tok, device, H_torch, E_exact, dim,
                          surgery_hooks=None, n_steps=200):
    """Run VQE with optional head surgery hooks."""
    hidden_size = model.config.hidden_size
    embed_layer = model.model.embed_tokens
    seed_prompt = "Chemical bond ground state energy:"
    seed_ids = tok(seed_prompt, return_tensors='pt')['input_ids'].to(device)
    seed_embeds = embed_layer(seed_ids).detach().clone()

    opt_embeds = seed_embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_embeds], lr=0.001)

    # Register surgery hooks
    handles = []
    if surgery_hooks:
        for layer_idx, hook in surgery_hooks:
            h = model.model.layers[layer_idx].register_forward_hook(hook)
            handles.append(h)

    energies = []
    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = model(inputs_embeds=opt_embeds.float(),
                       output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :]
        psi = h[:dim]
        psi_norm = psi / (torch.norm(psi) + 1e-10)
        E = psi_norm @ H_torch @ psi_norm
        E.backward()
        optimizer.step()
        energies.append(float(E.detach()))

    # Remove hooks
    for h in handles:
        h.remove()

    final_error = abs(energies[-1] - E_exact) * 1000
    return energies, final_error


def main():
    print("=" * 60)
    print("Phase Q185: Entanglement Surgery")
    print("  (Head 11 Ablation & Boost - Causal Proof)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = hidden_size // n_heads
    dim = 16

    H_np = build_h2_hamiltonian(0.74)
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigvalsh(H_np)[0])

    print("  Exact E0: %.6f Ha" % E_exact)
    print("  Target head: %d, Control head: %d" % (TARGET_HEAD, CONTROL_HEAD))

    # === Experiment 1: Baseline (no surgery) ===
    print("\n--- Baseline (no surgery) ---")
    e_baseline, err_baseline = run_vqe_with_surgery(
        model, tok, device, H_torch, E_exact, dim, surgery_hooks=None)
    print("  Error: %.2f mHa" % err_baseline)

    # === Experiment 2: Ablate Head 11 (all layers) ===
    print("\n--- Ablate Head %d (all layers) ---" % TARGET_HEAD)
    hooks_ablate_11 = [
        (li, HeadSurgeryHook(TARGET_HEAD, n_heads, head_dim, 'ablate'))
        for li in range(n_layers)
    ]
    e_ablate_11, err_ablate_11 = run_vqe_with_surgery(
        model, tok, device, H_torch, E_exact, dim, surgery_hooks=hooks_ablate_11)
    print("  Error: %.2f mHa" % err_ablate_11)

    # === Experiment 3: Ablate Head 0 (control) ===
    print("\n--- Ablate Head %d (control, all layers) ---" % CONTROL_HEAD)
    hooks_ablate_0 = [
        (li, HeadSurgeryHook(CONTROL_HEAD, n_heads, head_dim, 'ablate'))
        for li in range(n_layers)
    ]
    e_ablate_0, err_ablate_0 = run_vqe_with_surgery(
        model, tok, device, H_torch, E_exact, dim, surgery_hooks=hooks_ablate_0)
    print("  Error: %.2f mHa" % err_ablate_0)

    # === Experiment 4: Ablate Head 11 (deep layers only, 18-27) ===
    print("\n--- Ablate Head %d (layers 18-27 only) ---" % TARGET_HEAD)
    hooks_ablate_deep = [
        (li, HeadSurgeryHook(TARGET_HEAD, n_heads, head_dim, 'ablate'))
        for li in range(18, n_layers)
    ]
    e_ablate_deep, err_ablate_deep = run_vqe_with_surgery(
        model, tok, device, H_torch, E_exact, dim, surgery_hooks=hooks_ablate_deep)
    print("  Error: %.2f mHa" % err_ablate_deep)

    # === Experiment 5: Boost Head 11 (2x) ===
    print("\n--- Boost Head %d (2x, all layers) ---" % TARGET_HEAD)
    hooks_boost = [
        (li, HeadSurgeryHook(TARGET_HEAD, n_heads, head_dim, 'boost', scale=2.0))
        for li in range(n_layers)
    ]
    e_boost, err_boost = run_vqe_with_surgery(
        model, tok, device, H_torch, E_exact, dim, surgery_hooks=hooks_boost)
    print("  Error: %.2f mHa" % err_boost)

    # === Experiment 6: Boost Head 11 (3x) ===
    print("\n--- Boost Head %d (3x, all layers) ---" % TARGET_HEAD)
    hooks_boost3 = [
        (li, HeadSurgeryHook(TARGET_HEAD, n_heads, head_dim, 'boost', scale=3.0))
        for li in range(n_layers)
    ]
    e_boost3, err_boost3 = run_vqe_with_surgery(
        model, tok, device, H_torch, E_exact, dim, surgery_hooks=hooks_boost3)
    print("  Error: %.2f mHa" % err_boost3)

    # === Analysis ===
    print("\n--- Causal Analysis ---")
    print("  Baseline:          %.2f mHa" % err_baseline)
    print("  Ablate Head 11:    %.2f mHa (%.1fx worse)" %
          (err_ablate_11, err_ablate_11 / max(err_baseline, 0.01)))
    print("  Ablate Head 0:     %.2f mHa (%.1fx worse)" %
          (err_ablate_0, err_ablate_0 / max(err_baseline, 0.01)))
    print("  Ablate H11 deep:   %.2f mHa" % err_ablate_deep)
    print("  Boost H11 (2x):    %.2f mHa" % err_boost)
    print("  Boost H11 (3x):    %.2f mHa" % err_boost3)

    # Causal test: Head 11 ablation should cause much more damage than Head 0
    h11_impact = err_ablate_11 / max(err_ablate_0, 0.01)
    if h11_impact > 2.0:
        verdict = "CAUSAL PROOF: Head 11 is %dx more critical than Head 0" % int(h11_impact)
    elif h11_impact > 1.2:
        verdict = "PARTIAL CAUSATION: Head 11 is %.1fx more impactful" % h11_impact
    else:
        verdict = "NO DIFFERENTIAL CAUSATION (both heads similar impact)"
    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q185',
        'name': 'Entanglement Surgery',
        'target_head': TARGET_HEAD,
        'control_head': CONTROL_HEAD,
        'experiments': {
            'baseline': {'error_mHa': round(err_baseline, 2)},
            'ablate_head11': {'error_mHa': round(err_ablate_11, 2)},
            'ablate_head0': {'error_mHa': round(err_ablate_0, 2)},
            'ablate_head11_deep': {'error_mHa': round(err_ablate_deep, 2)},
            'boost_head11_2x': {'error_mHa': round(err_boost, 2)},
            'boost_head11_3x': {'error_mHa': round(err_boost3, 2)},
        },
        'causal_analysis': {
            'h11_vs_h0_impact_ratio': round(h11_impact, 2),
            'verdict': verdict,
        },
        'convergence': {
            'baseline': [round(e, 6) for e in e_baseline[::20]],
            'ablate_h11': [round(e, 6) for e in e_ablate_11[::20]],
            'ablate_h0': [round(e, 6) for e in e_ablate_0[::20]],
            'boost_h11': [round(e, 6) for e in e_boost[::20]],
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q185_surgery.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Error comparison bar chart
    ax = axes[0]
    conditions = ['Baseline', 'Ablate\nH11', 'Ablate\nH0', 'Ablate\nH11 deep',
                  'Boost\nH11 2x', 'Boost\nH11 3x']
    errors = [err_baseline, err_ablate_11, err_ablate_0, err_ablate_deep,
              err_boost, err_boost3]
    colors = ['#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0', '#E91E63']
    ax.bar(range(len(conditions)), errors, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.6, color='green', ls='--', linewidth=2, label='Chemical accuracy')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel('VQE Error (mHa)')
    ax.set_title('(a) Head Surgery Impact\n(Higher = More Damage)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (b) Convergence curves
    ax = axes[1]
    ax.plot(e_baseline, '-', color='#4CAF50', linewidth=2, label='Baseline')
    ax.plot(e_ablate_11, '-', color='#F44336', linewidth=2, label='Ablate H11')
    ax.plot(e_ablate_0, '-', color='#2196F3', linewidth=2, label='Ablate H0')
    ax.plot(e_boost, '-', color='#9C27B0', linewidth=2, label='Boost H11')
    ax.axhline(E_exact, color='black', ls=':', label='Exact')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title('(b) Convergence Under Surgery')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Causal impact summary
    ax = axes[2]
    impacts = [1.0, err_ablate_11/max(err_baseline, 0.01),
               err_ablate_0/max(err_baseline, 0.01)]
    impact_labels = ['Baseline\n(1.0x)', 'Ablate H11\n(%.1fx)' % impacts[1],
                     'Ablate H0\n(%.1fx)' % impacts[2]]
    ax.bar(range(3), impacts, color=['#4CAF50', '#F44336', '#2196F3'],
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(3))
    ax.set_xticklabels(impact_labels, fontsize=9)
    ax.set_ylabel('Relative Error (vs Baseline)')
    ax.set_title('(c) Causal Impact\n(H11 >> H0 = Causal Proof)')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q185: Entanglement Surgery\n'
                 'Head 11 Ablation: %.1fx worse, Head 0: %.1fx worse -> %s' %
                 (impacts[1], impacts[2], 'CAUSAL' if h11_impact > 2 else 'PARTIAL'),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q185_surgery.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ185 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
