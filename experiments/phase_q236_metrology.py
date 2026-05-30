# -*- coding: utf-8 -*-
"""
Phase Q236: Quantum Metrology Application
============================================
Q218 showed QFI Heisenberg ratio = 28.9.
Actually USE this for parameter estimation:
estimate an unknown parameter theta in H(theta)
and show the LLM achieves sub-classical precision.
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


def estimate_parameter(model, tok, device, theta_true, dim=4, n_steps=100):
    """Estimate theta by minimizing E(theta) landscape."""
    rng = np.random.RandomState(42)
    H_base = rng.randn(dim, dim).astype(np.float32) * 0.3
    H_base = (H_base + H_base.T) / 2

    # Parameter-dependent Hamiltonian
    H_theta = H_base.copy()
    H_theta[0, 0] += theta_true
    H_theta[1, 1] -= theta_true
    H_torch = torch.tensor(H_theta, device=device)
    E_exact = float(np.linalg.eigh(H_theta)[0][0])

    # VQE to find ground state
    embed_layer = model.model.embed_tokens
    inp = tok("parameter estimation:", return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=0.005)

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()

    # Estimate theta from the converged state
    with torch.no_grad():
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim].float().cpu().numpy()
        psi = h[:dim] / (np.linalg.norm(h[:dim]) + 1e-10)

    # Estimate: theta = <psi|dH/dtheta|psi> / <psi|d2H/dtheta2|psi>
    # dH/dtheta has +1 at (0,0) and -1 at (1,1)
    dH = np.zeros((dim, dim))
    dH[0, 0] = 1; dH[1, 1] = -1
    theta_est = float(psi @ dH @ psi)

    return theta_est, abs(theta_est - theta_true)


def classical_estimate(theta_true, dim=4, n_samples=100):
    """Classical parameter estimation via random sampling."""
    rng = np.random.RandomState(42)
    H_base = rng.randn(dim, dim).astype(np.float32) * 0.3
    H_base = (H_base + H_base.T) / 2

    best_err = float('inf')
    for _ in range(n_samples):
        psi = rng.randn(dim)
        psi /= np.linalg.norm(psi)
        dH = np.zeros((dim, dim))
        dH[0, 0] = 1; dH[1, 1] = -1
        est = float(psi @ dH @ psi)
        err = abs(est - theta_true)
        best_err = min(best_err, err)
    return best_err


def main():
    print("=" * 60)
    print("Phase Q236: Quantum Metrology Application")
    print("  (Use QFI for actual parameter estimation)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    theta_values = np.linspace(-1, 1, 9)
    all_results = []

    for theta in theta_values:
        llm_est, llm_err = estimate_parameter(model, tok, device, float(theta))
        cls_err = classical_estimate(float(theta))
        ratio = cls_err / max(llm_err, 1e-8)

        print("  theta=%.2f: LLM_err=%.4f, CLS_err=%.4f, ratio=%.1fx" %
              (theta, llm_err, cls_err, ratio))

        all_results.append({
            'theta_true': round(float(theta), 4),
            'llm_error': round(llm_err, 6),
            'classical_error': round(cls_err, 6),
            'advantage_ratio': round(ratio, 2),
        })

    avg_ratio = np.mean([r['advantage_ratio'] for r in all_results])
    avg_llm_err = np.mean([r['llm_error'] for r in all_results])
    avg_cls_err = np.mean([r['classical_error'] for r in all_results])

    if avg_ratio > 5:
        verdict = "QUANTUM METROLOGY: %.0fx precision advantage (LLM=%.4f vs CLS=%.4f)" % (avg_ratio, avg_llm_err, avg_cls_err)
    elif avg_ratio > 1:
        verdict = "SLIGHT ADVANTAGE: %.1fx" % avg_ratio
    else:
        verdict = "NO ADVANTAGE: ratio=%.2f" % avg_ratio

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q236', 'name': 'Quantum Metrology',
        'estimations': all_results,
        'summary': {'avg_ratio': round(avg_ratio, 2), 'avg_llm_err': round(avg_llm_err, 6), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q236_metrology.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    thetas = [r['theta_true'] for r in all_results]
    ax = axes[0]
    ax.semilogy(thetas, [max(r['llm_error'], 1e-8) for r in all_results], 'o-', color='#E91E63', label='LLM', lw=2)
    ax.semilogy(thetas, [r['classical_error'] for r in all_results], 's--', color='#607D8B', label='Classical', lw=2)
    ax.set_xlabel('True theta'); ax.set_ylabel('Estimation Error')
    ax.set_title('(a) Estimation Error'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ratios = [r['advantage_ratio'] for r in all_results]
    colors = ['#4CAF50' if r > 1 else '#F44336' for r in ratios]
    ax.bar(range(len(thetas)), ratios, color=colors, edgecolor='black')
    ax.axhline(1, color='black', ls='--')
    ax.set_xticks(range(len(thetas))); ax.set_xticklabels(['%.1f' % t for t in thetas], fontsize=7)
    ax.set_xlabel('True theta'); ax.set_ylabel('Advantage Ratio (CLS/LLM)')
    ax.set_title('(b) Advantage Ratio'); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q236: Quantum Metrology\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q236_metrology.png'), dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ236 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
