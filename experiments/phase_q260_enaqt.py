# -*- coding: utf-8 -*-
"""
Phase Q260: Environment-Assisted Quantum Transport (ENAQT)
=============================================================
Simulate photosynthesis: the FMO complex energy transfer.
Test if LLM's "moderate decoherence" (Layer 22 crossover)
actually HELPS quantum transport - just like in biology.
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

def fmo_hamiltonian(dim=7):
    """Simplified FMO complex Hamiltonian (7-site model)."""
    # Diagonal: site energies (cm^-1, scaled)
    E = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440])[:dim] * 0.001
    H = np.diag(E)
    # Off-diagonal: couplings
    couplings = [
        (0, 1, -87.7), (0, 2, 5.5), (0, 3, -5.9), (1, 2, 30.8),
        (1, 3, 8.2), (2, 3, -53.5), (3, 4, -70.7), (4, 5, -17.4),
        (5, 6, -63.0), (3, 6, -1.3),
    ]
    for i, j, v in couplings:
        if i < dim and j < dim:
            H[i, j] = v * 0.001
            H[j, i] = v * 0.001
    return H.astype(np.float32)

def main():
    print("=" * 60)
    print("Phase Q260: Environment-Assisted Quantum Transport")
    print("  (Photosynthesis in an LLM)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    dim = 7  # FMO has 7 chromophores
    H_fmo = fmo_hamiltonian(dim)
    H_torch = torch.tensor(H_fmo, device=device)
    E_exact = float(np.linalg.eigh(H_fmo)[0][0])

    # Source (site 1) and sink (site 3) for energy transfer
    source, sink = 0, 3

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    def run_transport(noise_scale, n_steps=150):
        """VQE with noise injection to simulate environmental decoherence."""
        embed = model.model.embed_tokens
        inp_ids = tok("photosynthesis energy transfer:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed(inp_ids).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)

        for s in range(n_steps):
            optimizer.zero_grad()
            o = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = o.hidden_states[-1][0, -1, :dim]
            # Add environmental noise
            if noise_scale > 0:
                noise = torch.randn_like(h) * noise_scale
                h = h + noise
            psi = h / (torch.norm(h) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward(); optimizer.step()

        # Transport efficiency: overlap with sink state
        with torch.no_grad():
            o = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h_final = o.hidden_states[-1][0, -1, :dim]
            if noise_scale > 0:
                h_final = h_final + torch.randn_like(h_final) * noise_scale
            psi = h_final / (torch.norm(h_final) + 1e-10)
            # Transfer efficiency = |<sink|psi>|^2
            efficiency = float(psi[sink].cpu()) ** 2
            error = abs(float(torch.dot(psi, H_torch @ psi).cpu()) - E_exact) * 1000

        return error, efficiency

    results_data = []
    print("\n  Testing noise levels...")
    for noise in noise_levels:
        err, eff = run_transport(noise)
        print("  noise=%.2f: error=%.4f mHa, transport=%.4f" % (noise, err, eff))
        results_data.append({
            'noise': noise, 'error': round(err, 4), 'efficiency': round(eff, 4)
        })

    # Find optimal noise (ENAQT peak)
    best_idx = max(range(len(results_data)), key=lambda i: results_data[i]['efficiency'])
    optimal_noise = results_data[best_idx]['noise']
    peak_efficiency = results_data[best_idx]['efficiency']
    zero_noise_eff = results_data[0]['efficiency']

    enaqt = optimal_noise > 0 and peak_efficiency > zero_noise_eff * 1.05

    if enaqt:
        verdict = "ENAQT CONFIRMED: optimal noise=%.2f (eff=%.3f vs zero-noise=%.3f)" % (
            optimal_noise, peak_efficiency, zero_noise_eff)
    else:
        verdict = "NO ENAQT: zero-noise=%.3f, best=%.3f at noise=%.2f" % (
            zero_noise_eff, peak_efficiency, optimal_noise)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q260', 'name': 'Environment-Assisted Quantum Transport',
        'noise_sweep': results_data,
        'summary': {'optimal_noise': optimal_noise, 'peak_eff': round(peak_efficiency, 4),
                     'zero_noise_eff': round(zero_noise_eff, 4), 'enaqt': bool(enaqt), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q260_enaqt.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ns = [r['noise'] for r in results_data]

    ax = axes[0]
    ax.plot(ns, [r['efficiency'] for r in results_data], 'o-', color='#4CAF50', lw=2, ms=8)
    ax.axvline(optimal_noise, color='red', ls='--', label='Optimal noise')
    ax.set_xlabel('Environmental Noise'); ax.set_ylabel('Transport Efficiency')
    ax.set_title('(a) ENAQT: Noise vs Efficiency'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ns, [r['error'] for r in results_data], 'o-', color='#E91E63', lw=2, ms=8)
    ax.set_xlabel('Environmental Noise'); ax.set_ylabel('VQE Error (mHa)')
    ax.set_title('(b) VQE Accuracy vs Noise'); ax.grid(alpha=0.3)

    plt.suptitle('Q260: Photosynthesis in an LLM\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q260_enaqt.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ260 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
