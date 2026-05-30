# -*- coding: utf-8 -*-
"""
Phase Q225: Quantum Discord
==============================
Discord = quantum correlation that exists EVEN WITHOUT entanglement.
Q219 showed non-contextuality, Q209 showed entanglement.
Discord bridges the gap: it's the "weakest" quantum signature.

Discord(A:B) = I(A:B) - J(A:B) where J is classical mutual info.
If Discord > 0, the state has irreducibly quantum correlations.
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
    if len(eigvals) == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log2(eigvals)))


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


def quantum_discord(rho, da, db, n_measurements=10):
    """Estimate quantum discord via optimization over measurements on B."""
    S_ab = vn_entropy(rho)
    rho_a = partial_trace_b(rho, da, db)
    rho_b = partial_trace_a(rho, da, db)
    S_a = vn_entropy(rho_a)
    S_b = vn_entropy(rho_b)

    # Quantum mutual information
    I_ab = S_a + S_b - S_ab

    # Classical correlation: maximize over measurement bases on B
    rng = np.random.RandomState(42)
    max_J = 0

    for _ in range(n_measurements):
        # Random measurement basis on B
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)

        # Projectors for qubit measurement
        if db == 2:
            proj_0 = np.array([[np.cos(theta/2)**2, np.cos(theta/2)*np.sin(theta/2)*np.exp(-1j*phi)],
                               [np.cos(theta/2)*np.sin(theta/2)*np.exp(1j*phi), np.sin(theta/2)**2]])
            proj_1 = np.eye(2) - proj_0
            projectors = [proj_0, proj_1]
        else:
            # Random projector for larger dims
            v = rng.randn(db) + 1j * rng.randn(db)
            v /= np.linalg.norm(v)
            proj_0 = np.outer(v, v.conj())
            proj_1 = np.eye(db) - proj_0
            projectors = [proj_0, proj_1]

        # Conditional entropy S(A|B_measurement)
        S_cond = 0
        for proj in projectors:
            # Apply measurement on B
            M = np.kron(np.eye(da), proj)
            rho_after = M @ rho @ M
            p = max(np.real(np.trace(rho_after)), 1e-12)
            if p > 1e-10:
                rho_cond = rho_after / p
                rho_a_cond = partial_trace_b(rho_cond, da, db)
                S_cond += p * vn_entropy(rho_a_cond)

        J = S_a - S_cond
        max_J = max(max_J, J)

    discord = I_ab - max_J
    return max(0, discord), I_ab, max_J


def main():
    print("=" * 60)
    print("Phase Q225: Quantum Discord")
    print("  (Quantum correlations without entanglement)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    da, db = 2, 2
    dt = da * db

    prompts = [
        "quantum entanglement between particles",
        "classical correlation coin flip",
        "Bell state maximally entangled",
        "thermal noise random process",
        "superposition of quantum states",
        "separable mixed quantum state",
    ]

    # Measure discord at key layers
    key_layers = list(range(0, n_layers + 1, 4))
    all_results = []

    for prompt in prompts:
        print("\n--- %s ---" % prompt[:35])
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layer_discords = []
        for li in key_layers:
            if li >= len(out.hidden_states):
                continue
            h = out.hidden_states[li][0, -1, :dt].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dt) / dt
            rho /= np.trace(rho)

            discord, mi, classical = quantum_discord(rho, da, db, n_measurements=20)

            layer_discords.append({
                'layer': li,
                'discord': round(discord, 6),
                'mutual_info': round(mi, 6),
                'classical_corr': round(classical, 6),
            })

            if li % 8 == 0:
                print("  L%d: discord=%.4f, MI=%.4f, classical=%.4f" %
                      (li, discord, mi, classical))

        all_results.append({
            'prompt': prompt[:35],
            'layers': layer_discords,
        })

    # Summary: average discord
    all_discords = [d['discord'] for r in all_results for d in r['layers']]
    avg_discord = np.mean(all_discords) if all_discords else 0
    n_nonzero = sum(1 for d in all_discords if d > 0.001)

    if avg_discord > 0.01:
        verdict = "QUANTUM DISCORD: avg=%.4f (%d/%d nonzero)" % (
            avg_discord, n_nonzero, len(all_discords))
    elif n_nonzero > 0:
        verdict = "WEAK DISCORD: avg=%.4f (%d/%d nonzero)" % (
            avg_discord, n_nonzero, len(all_discords))
    else:
        verdict = "NO DISCORD: states are classically correlated"

    print("\n--- Summary ---")
    print("  Avg discord: %.6f" % avg_discord)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q225',
        'name': 'Quantum Discord',
        'prompts': all_results,
        'summary': {
            'avg_discord': round(avg_discord, 6),
            'n_nonzero': n_nonzero,
            'total': len(all_discords),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q225_discord.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, r in enumerate(all_results[:6]):
        ax = axes_flat[idx]
        layers = [d['layer'] for d in r['layers']]
        discords = [d['discord'] for d in r['layers']]
        mis = [d['mutual_info'] for d in r['layers']]
        classicals = [d['classical_corr'] for d in r['layers']]

        ax.plot(layers, mis, 'o-', color='#2196F3', label='MI', ms=3)
        ax.plot(layers, classicals, 's--', color='#FF9800', label='Classical', ms=3)
        ax.fill_between(layers, classicals, mis, alpha=0.2, color='#E91E63',
                        label='Discord')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Information (bits)')
        ax.set_title(r['prompt'][:30], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Q225: Quantum Discord\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q225_discord.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ225 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
