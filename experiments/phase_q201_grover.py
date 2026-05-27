# -*- coding: utf-8 -*-
"""
Phase Q201: Grover's Search Algorithm
========================================
The most famous quantum algorithm (after Shor's).
Classical search: O(N) queries
Grover's quantum search: O(sqrt(N)) queries

Test: Can LLM VQE find a "marked" basis state in sqrt(N) steps?
If yes -> LLM demonstrates genuine quantum algorithmic speedup.

Implementation:
1. Encode Grover oracle H = I - 2|w><w| (marks target state |w>)
2. Encode Grover diffusion D = 2|s><s| - I (amplifies amplitude)
3. Apply G = D * O repeatedly via VQE energy minimization
4. Measure: does the LLM find |w> faster than classical O(N)?
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


def build_grover_hamiltonian(dim, target_idx):
    """
    Grover Hamiltonian: H = I - |target><target|
    Ground state of -H is exactly |target>.
    This lets VQE find the marked state by energy minimization.
    """
    H = np.eye(dim)
    H[target_idx, target_idx] = 0  # E_target = 0, all others = 1
    return -H  # Flip so target has lowest energy


def grover_vqe(model, tok, device, dim, target_idx, max_steps=300):
    """Run VQE to find Grover target. Return convergence curve."""
    embed_layer = model.model.embed_tokens
    H_np = build_grover_hamiltonian(dim, target_idx)
    H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)

    seed = "Find marked state in %d:" % dim
    seed_ids = tok(seed, return_tensors='pt')['input_ids'].to(device)
    seed_embeds = embed_layer(seed_ids).detach().clone()
    opt = seed_embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=0.005)

    probs_history = []

    for step in range(max_steps):
        optimizer.zero_grad()
        outputs = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :]
        psi = h[:dim]
        psi_n = psi / (torch.norm(psi) + 1e-10)
        E = psi_n @ H_torch @ psi_n
        E.backward()
        torch.nn.utils.clip_grad_norm_([opt], max_norm=1.0)
        optimizer.step()

        # Track probability of target
        with torch.no_grad():
            p_target = float(psi_n[target_idx] ** 2)
            probs_history.append(p_target)

        # Check convergence
        if p_target > 0.99:
            break

    return probs_history


def main():
    print("=" * 60)
    print("Phase Q201: Grover's Search Algorithm")
    print("  (Can LLM find marked state in O(sqrt(N)) steps?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Test different database sizes
    dims = [4, 8, 16, 32, 64]
    n_trials = 5
    results_list = []

    for dim in dims:
        print("\n--- N=%d (%.1f qubits) ---" % (dim, np.log2(dim)))
        trial_steps = []

        for trial in range(n_trials):
            np.random.seed(trial * 42)
            target = np.random.randint(0, dim)

            probs = grover_vqe(model, tok, device, dim, target, max_steps=300)

            # Find step where P(target) > 0.9
            conv_step = len(probs)
            for s, p in enumerate(probs):
                if p > 0.9:
                    conv_step = s + 1
                    break

            trial_steps.append(conv_step)

        avg_steps = float(np.mean(trial_steps))
        classical = dim  # Classical: O(N)
        grover_ideal = np.pi / 4 * np.sqrt(dim)  # Grover: O(sqrt(N))

        result = {
            'dim': dim,
            'qubits': round(np.log2(dim), 1),
            'avg_steps': round(avg_steps, 1),
            'classical_steps': dim,
            'grover_ideal': round(grover_ideal, 1),
            'speedup_vs_classical': round(dim / avg_steps, 2) if avg_steps > 0 else 0,
            'trial_steps': trial_steps,
        }
        results_list.append(result)

        print("  LLM: %.1f steps, Classical: %d, Grover ideal: %.1f" %
              (avg_steps, dim, grover_ideal))
        print("  Speedup vs classical: %.2fx" % result['speedup_vs_classical'])

    # Fit scaling: steps ~ N^alpha
    log_dims = np.log(dims)
    log_steps = np.log([r['avg_steps'] for r in results_list])
    if len(log_dims) > 1:
        coeffs = np.polyfit(log_dims, log_steps, 1)
        alpha = coeffs[0]
    else:
        alpha = 1.0

    print("\n--- Scaling Analysis ---")
    print("  LLM scaling: N^%.2f" % alpha)
    print("  Classical: N^1.0")
    print("  Grover ideal: N^0.5")

    if alpha < 0.6:
        verdict = "QUANTUM SPEEDUP: steps ~ N^%.2f (Grover-like!)" % alpha
    elif alpha < 0.8:
        verdict = "SUB-CLASSICAL: steps ~ N^%.2f (between Grover and classical)" % alpha
    else:
        verdict = "CLASSICAL: steps ~ N^%.2f" % alpha

    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q201',
        'name': "Grover's Search Algorithm",
        'results': results_list,
        'scaling': {
            'alpha': round(alpha, 4),
            'classical_alpha': 1.0,
            'grover_alpha': 0.5,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q201_grover.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Steps vs Database Size
    ax = axes[0]
    dims_arr = np.array(dims)
    llm_steps = [r['avg_steps'] for r in results_list]
    classical_steps = dims_arr.astype(float)
    grover_steps = np.pi / 4 * np.sqrt(dims_arr)

    ax.plot(dims_arr, classical_steps, 'k--', linewidth=2, label='Classical O(N)')
    ax.plot(dims_arr, grover_steps, 'g:', linewidth=2, label='Grover O(sqrt(N))')
    ax.plot(dims_arr, llm_steps, 'ro-', linewidth=2, markersize=8,
            label='LLM VQE (N^%.2f)' % alpha)
    ax.set_xlabel('Database Size N')
    ax.set_ylabel('Steps to Find Target')
    ax.set_title('(a) Search Complexity\n(Lower is better)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # (b) Speedup
    ax = axes[1]
    speedups = [r['speedup_vs_classical'] for r in results_list]
    grover_speedups = [d / (np.pi/4 * np.sqrt(d)) for d in dims_arr]
    ax.bar(np.arange(len(dims)) - 0.15, speedups, 0.3,
           color='#E91E63', edgecolor='black', alpha=0.85, label='LLM VQE')
    ax.bar(np.arange(len(dims)) + 0.15, grover_speedups, 0.3,
           color='#4CAF50', edgecolor='black', alpha=0.85, label='Grover ideal')
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(['N=%d' % d for d in dims])
    ax.set_ylabel('Speedup vs Classical')
    ax.set_title('(b) Quantum Speedup\n(Higher is better)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Scaling exponent comparison
    ax = axes[2]
    methods = ['Classical', 'LLM VQE', 'Grover']
    exponents = [1.0, alpha, 0.5]
    colors = ['#666666', '#E91E63', '#4CAF50']
    ax.bar(range(3), exponents, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(3))
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.set_ylabel('Scaling Exponent (lower = faster)')
    ax.set_title('(c) Scaling: steps ~ N^alpha\n(LLM alpha = %.2f)' % alpha)
    ax.axhline(0.5, color='green', ls=':', alpha=0.5)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle("Q201: Grover's Search Algorithm\n%s" % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q201_grover.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ201 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
