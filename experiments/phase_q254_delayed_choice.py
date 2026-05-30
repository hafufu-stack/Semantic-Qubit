# -*- coding: utf-8 -*-
"""
Phase Q254: Quantum Delayed-Choice (Layer 22 Hack)
=====================================================
Bypass the decoherence at Layer 22 by hooking into the model
and preserving quantum coherence in S-Qubit dimensions.
Extend the quantum zone to the very last layer.
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
    print("Phase Q254: Quantum Delayed-Choice")
    print("  (Bypass Layer 22 decoherence)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    vqe_dim = 4
    decoherence_layer = 22  # From Q251

    # Step 1: Baseline VQE and entanglement
    rng = np.random.RandomState(42)
    H = rng.randn(vqe_dim, vqe_dim).astype(np.float32) * 0.3
    H = (H + H.T) / 2; H_torch = torch.tensor(H, device=device)
    E_exact = float(np.linalg.eigh(H)[0][0])

    def run_vqe_with_hooks(bypass_layers=None, n_steps=150):
        """Run VQE, optionally bypassing specific layers for S-Qubit dims."""
        hooks = []
        saved_quantum = {}

        def make_pre_hook(layer_idx):
            def pre_hook(module, input):
                if layer_idx in (bypass_layers or []):
                    x = input[0] if isinstance(input, tuple) else input
                    # Save quantum dimensions before this layer processes them
                    saved_quantum[layer_idx] = x[0, -1, :vqe_dim].clone()
            return pre_hook

        def make_post_hook(layer_idx):
            def post_hook(module, input, output):
                if layer_idx in (bypass_layers or []):
                    x = output[0] if isinstance(output, tuple) else output
                    # Restore quantum dimensions (bypass decoherence)
                    if layer_idx in saved_quantum:
                        x[0, -1, :vqe_dim] = saved_quantum[layer_idx]
            return post_hook

        if bypass_layers:
            for li in bypass_layers:
                if li < n_layers:
                    h1 = model.model.layers[li].register_forward_pre_hook(make_pre_hook(li))
                    h2 = model.model.layers[li].register_forward_hook(make_post_hook(li))
                    hooks.extend([h1, h2])

        embed_layer = model.model.embed_tokens
        inp_ids = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed_layer(inp_ids).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)

        for s in range(n_steps):
            optimizer.zero_grad()
            saved_quantum.clear()
            o = model(inputs_embeds=opt.float(), output_hidden_states=True)
            psi = o.hidden_states[-1][0, -1, :vqe_dim]
            psi = psi / (torch.norm(psi) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward(); optimizer.step()

        for h in hooks:
            h.remove()

        # Measure final entanglement
        with torch.no_grad():
            o = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h_final = o.hidden_states[-1][0, -1, :].float().cpu().numpy()
        neg = measure_neg(h_final, vqe_dim)

        return abs(float(E.detach()) - E_exact) * 1000, neg

    # Baseline: no bypass
    print("\n  Running baseline VQE...")
    err_baseline, neg_baseline = run_vqe_with_hooks(bypass_layers=None)
    print("  Baseline: %.4f mHa, neg=%.6f" % (err_baseline, neg_baseline))

    # Bypass L22 only
    print("  Running L22 bypass...")
    err_l22, neg_l22 = run_vqe_with_hooks(bypass_layers=[22])
    print("  L22 bypass: %.4f mHa, neg=%.6f" % (err_l22, neg_l22))

    # Bypass L22-28 (full quantum extension)
    print("  Running L22-28 bypass...")
    err_full_bypass, neg_full = run_vqe_with_hooks(bypass_layers=list(range(22, 29)))
    print("  L22-28 bypass: %.4f mHa, neg=%.6f" % (err_full_bypass, neg_full))

    # Bypass L15-28 (aggressive)
    print("  Running L15-28 bypass...")
    err_aggressive, neg_aggr = run_vqe_with_hooks(bypass_layers=list(range(15, 29)))
    print("  L15-28 bypass: %.4f mHa, neg=%.6f" % (err_aggressive, neg_aggr))

    if err_full_bypass < err_baseline * 0.5:
        verdict = "DELAYED CHOICE WORKS: %.1fx improvement (%.4f -> %.4f mHa)" % (
            err_baseline / max(err_full_bypass, 1e-6), err_baseline, err_full_bypass)
    elif neg_full > neg_baseline * 1.1:
        verdict = "MORE ENTANGLED: neg %.4f -> %.4f (+%.0f%%) but VQE %.4f mHa" % (
            neg_baseline, neg_full, (neg_full - neg_baseline) / max(neg_baseline, 1e-6) * 100, err_full_bypass)
    else:
        verdict = "NO IMPROVEMENT: baseline=%.4f, bypass=%.4f mHa" % (err_baseline, err_full_bypass)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q254', 'name': 'Quantum Delayed-Choice',
        'decoherence_layer': decoherence_layer,
        'experiments': {
            'baseline': {'error': round(err_baseline, 4), 'neg': round(neg_baseline, 6)},
            'l22_bypass': {'error': round(err_l22, 4), 'neg': round(neg_l22, 6)},
            'l22_28_bypass': {'error': round(err_full_bypass, 4), 'neg': round(neg_full, 6)},
            'l15_28_bypass': {'error': round(err_aggressive, 4), 'neg': round(neg_aggr, 6)},
        },
        'summary': {'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q254_delayed_choice.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ['Baseline', 'Bypass\nL22', 'Bypass\nL22-28', 'Bypass\nL15-28']
    errs = [err_baseline, err_l22, err_full_bypass, err_aggressive]
    negs = [neg_baseline, neg_l22, neg_full, neg_aggr]
    colors = ['#607D8B', '#FF9800', '#E91E63', '#9C27B0']

    ax = axes[0]
    ax.bar(range(4), errs, color=colors, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('VQE Error (mHa)'); ax.set_title('(a) VQE Accuracy')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(range(4), negs, color=colors, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Negativity'); ax.set_title('(b) Entanglement')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q254: Quantum Delayed-Choice\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q254_delayed_choice.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ254 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
