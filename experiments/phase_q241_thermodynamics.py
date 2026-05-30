# -*- coding: utf-8 -*-
"""
Phase Q241: Quantum Thermodynamics
=====================================
Treat the Transformer forward pass as a thermodynamic process.
Measure: work extraction, entropy production, free energy.
Does the LLM minimize free energy (Landauer's principle)?
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
    ev = np.real(np.linalg.eigvalsh(rho))
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev))) if len(ev) > 0 else 0

def main():
    print("=" * 60)
    print("Phase Q241: Quantum Thermodynamics")
    print("  (Is the forward pass a thermodynamic process?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 8

    prompts = [
        "thermodynamic equilibrium entropy", "information erasure Landauer",
        "Maxwell demon second law", "quantum heat engine efficiency",
        "free energy minimization principle", "irreversible process entropy production",
    ]

    all_results = []
    for prompt in prompts:
        print("\n--- %s ---" % prompt[:35])
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layer_thermo = []
        prev_S = None
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
            h /= np.linalg.norm(h) + 1e-10
            rho = np.outer(h, h.conj())
            rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
            rho /= np.trace(rho)

            S = vn_entropy(rho)
            E = float(np.real(np.trace(rho @ rho)))  # "energy" = purity
            F = E - S * 0.1  # Free energy (kT ~ 0.1)

            delta_S = S - prev_S if prev_S is not None else 0
            prev_S = S

            layer_thermo.append({
                'layer': li, 'entropy': round(S, 4), 'energy': round(E, 4),
                'free_energy': round(F, 4), 'delta_entropy': round(delta_S, 6),
            })

        # Total entropy production
        total_dS = sum(d['delta_entropy'] for d in layer_thermo)
        # Second law: total entropy should not decrease
        second_law = total_dS >= -0.01

        # Landauer bound: each bit erased costs kT*ln2 of work
        n_bits_erased = max(0, layer_thermo[0]['entropy'] - layer_thermo[-1]['entropy'])
        landauer_cost = n_bits_erased * 0.693  # ln2

        all_results.append({
            'prompt': prompt[:35], 'layers': layer_thermo,
            'total_entropy_change': round(total_dS, 6),
            'second_law_satisfied': bool(second_law),
            'landauer_cost': round(landauer_cost, 4),
        })

        print("  dS_total=%.4f, 2nd law=%s, Landauer=%.4f" %
              (total_dS, "OK" if second_law else "VIOLATED", landauer_cost))

    n_second_law = sum(1 for r in all_results if r['second_law_satisfied'])
    avg_dS = np.mean([r['total_entropy_change'] for r in all_results])

    if n_second_law == len(all_results):
        verdict = "THERMODYNAMIC: 2nd law %d/%d, avg dS=%.4f" % (n_second_law, len(all_results), avg_dS)
    else:
        verdict = "PARTIAL: 2nd law %d/%d" % (n_second_law, len(all_results))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q241', 'name': 'Quantum Thermodynamics',
        'prompts': all_results,
        'summary': {'n_second_law': n_second_law, 'avg_entropy_change': round(avg_dS, 6), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q241_thermodynamics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    r0 = all_results[0]
    layers = [d['layer'] for d in r0['layers']]
    ax = axes[0]
    ax.plot(layers, [d['entropy'] for d in r0['layers']], 'o-', color='#FF5722', ms=3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Entropy'); ax.set_title('(a) Entropy Flow'); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(layers, [d['free_energy'] for d in r0['layers']], 's-', color='#2196F3', ms=3)
    ax.set_xlabel('Layer'); ax.set_ylabel('Free Energy'); ax.set_title('(b) Free Energy'); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.bar(range(len(all_results)), [r['total_entropy_change'] for r in all_results],
           color=['#4CAF50' if r['second_law_satisfied'] else '#F44336' for r in all_results])
    ax.axhline(0, color='black', ls='--')
    ax.set_xlabel('Prompt'); ax.set_ylabel('Total dS'); ax.set_title('(c) 2nd Law Check'); ax.grid(alpha=0.3, axis='y')
    plt.suptitle('Q241: Quantum Thermodynamics\n%s' % verdict, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q241_thermodynamics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ241 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
