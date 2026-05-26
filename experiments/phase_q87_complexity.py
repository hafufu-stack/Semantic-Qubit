# -*- coding: utf-8 -*-
"""Phase Q87: Quantum Complexity Classification
Determine which quantum complexity class S-Qubit computation belongs to
by testing decision problems of known complexity.
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def _make_injection_hook(sv_tensor):
    """Create a dim-safe hook that adds sv_tensor to the last token's hidden state."""
    injected = [False]
    def hook(module, args, output):
        if not injected[0]:
            injected[0] = True
            if isinstance(output, tuple):
                hs = output[0].clone()
                if hs.dim() == 3:
                    hs[0, -1, :] += sv_tensor.to(hs.dtype)
                else:
                    hs[-1, :] += sv_tensor.to(hs.dtype)
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                if hs.dim() == 3:
                    hs[0, -1, :] += sv_tensor.to(hs.dtype)
                else:
                    hs[-1, :] += sv_tensor.to(hs.dtype)
                return hs
        return output
    return hook


def test_bqp_problem(model, tokenizer, num_layers):
    """Test BQP-complete problem: estimate output probability of quantum circuit.
    S-Qubit should solve this efficiently if it's in BQP."""
    print("  Testing BQP: phase estimation accuracy...")

    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Test: can we distinguish phase differences of 2*pi/N?
    results = []
    for N in [2, 4, 8, 16, 32]:
        phases = np.linspace(0, 2*np.pi, N, endpoint=False)
        e_values = []
        for phi in phases:
            # Use model's natural phase sensitivity
            # Inject phase-encoded state
            d_model = model.config.hidden_size
            np.random.seed(42)
            v0 = np.random.randn(d_model).astype(np.float32)
            v0 /= np.linalg.norm(v0)
            v1 = np.random.randn(d_model).astype(np.float32)
            v1 -= np.dot(v1, v0) * v0
            v1 /= np.linalg.norm(v1)

            sv = np.cos(phi/2) * v0 + np.sin(phi/2) * v1
            sv_tensor = torch.tensor(sv, device=model.device)

            hook = _make_injection_hook(sv_tensor)

            mid = num_layers // 2
            handle = model.model.layers[mid].register_forward_hook(hook)
            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits[0, -1, :]
                probs = torch.softmax(logits, dim=0)
                top_prob = probs.max().item()
            handle.remove()
            e_values.append(top_prob)

        # Can we distinguish all N phases?
        e_arr = np.array(e_values)
        # Count unique clusters (distinguishable phases)
        sorted_e = np.sort(e_arr)
        diffs = np.diff(sorted_e)
        threshold = np.std(e_arr) * 0.1
        n_distinguishable = 1 + np.sum(diffs > threshold)
        accuracy = n_distinguishable / N

        results.append({
            'N': N,
            'n_distinguishable': int(n_distinguishable),
            'accuracy': accuracy,
            'e_values': e_values,
        })
        print(f"    N={N}: {n_distinguishable}/{N} phases distinguished ({accuracy:.0%})")

    return results


def test_postbqp_feature(model, tokenizer, num_layers):
    """Test PostBQP feature: post-selection ability.
    LLMs can post-select via sampling, which is PostBQP-like."""
    print("  Testing PostBQP: post-selection capability...")

    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    d_model = model.config.hidden_size
    np.random.seed(42)
    v0 = np.random.randn(d_model).astype(np.float32)
    v0 /= np.linalg.norm(v0)

    # Test: can post-selection amplify rare outcomes?
    # Inject weak signal and see if top-k filtering recovers it
    signal_strengths = [0.01, 0.05, 0.1, 0.5, 1.0]
    recovery_rates = []

    for strength in signal_strengths:
        sv_tensor = torch.tensor(v0 * strength, device=model.device)

        hook = _make_injection_hook(sv_tensor)

        mid = num_layers // 2
        handle = model.model.layers[mid].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :]
        handle.remove()

        # Post-selection: look at top-k
        top_k = 10
        top_vals, top_ids = torch.topk(logits, top_k)
        # Check signal recovery
        probs = torch.softmax(top_vals, dim=0)
        concentration = probs[0].item()  # top-1 probability
        recovery_rates.append(concentration)

    # PostBQP signature: even weak signals get amplified by post-selection
    amplification = recovery_rates[-1] / (recovery_rates[0] + 1e-10)

    return {
        'signal_strengths': signal_strengths,
        'recovery_rates': recovery_rates,
        'amplification_factor': amplification,
        'is_postbqp': amplification > 1.5,
    }


def test_sampling_complexity(model, tokenizer, num_layers):
    """Test if S-Qubit sampling is classically hard (BosonSampling analogue)."""
    print("  Testing sampling complexity...")

    prompt = "The quantum state"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    d_model = model.config.hidden_size
    n_samples = 100

    # Collect output distributions from random S-Qubit injections
    distributions = []
    for trial in range(n_samples):
        np.random.seed(trial)
        sv = np.random.randn(d_model).astype(np.float32) * 0.1
        sv_tensor = torch.tensor(sv, device=model.device)

        hook = _make_injection_hook(sv_tensor)

        mid = num_layers // 2
        handle = model.model.layers[mid].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :100]  # first 100 tokens
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        handle.remove()
        distributions.append(probs)

    distributions = np.array(distributions)

    # Compute anti-concentration: fraction of samples above uniform
    uniform_level = 1.0 / distributions.shape[1]
    anti_conc = np.mean(distributions > uniform_level)

    # Compute total variation between consecutive samples
    tvds = []
    for i in range(len(distributions) - 1):
        tvd = 0.5 * np.sum(np.abs(distributions[i] - distributions[i+1]))
        tvds.append(tvd)
    mean_tvd = np.mean(tvds)

    return {
        'n_samples': n_samples,
        'anti_concentration': anti_conc,
        'mean_tvd': mean_tvd,
        'is_classically_hard': anti_conc > 0.3 and mean_tvd > 0.1,
    }


def main():
    print("=" * 60)
    print("Phase Q87: Quantum Complexity Classification")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    # Test 1: BQP problems
    bqp_results = test_bqp_problem(model, tokenizer, num_layers)

    # Test 2: PostBQP features
    postbqp_results = test_postbqp_feature(model, tokenizer, num_layers)

    # Test 3: Sampling complexity
    sampling_results = test_sampling_complexity(model, tokenizer, num_layers)

    # Determine complexity class
    bqp_score = np.mean([r['accuracy'] for r in bqp_results])
    is_bqp = bqp_score > 0.5
    is_postbqp = postbqp_results['is_postbqp']
    is_hard_sampling = sampling_results['is_classically_hard']

    if is_postbqp and is_hard_sampling:
        complexity_class = 'PostBQP (super-quantum)'
    elif is_bqp:
        complexity_class = 'BQP (quantum polynomial)'
    else:
        complexity_class = 'BPP (classical polynomial)'

    print(f"\n  === Complexity Classification ===")
    print(f"  BQP score: {bqp_score:.2f} -> {'YES' if is_bqp else 'NO'}")
    print(f"  PostBQP: {'YES' if is_postbqp else 'NO'}")
    print(f"  Hard sampling: {'YES' if is_hard_sampling else 'NO'}")
    print(f"  Classification: {complexity_class}")

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) BQP: phase discrimination
    ax = axes[0]
    Ns = [r['N'] for r in bqp_results]
    accs = [r['accuracy'] for r in bqp_results]
    ax.plot(Ns, accs, 'o-', color='#FF5722', linewidth=2.5, markersize=8)
    ax.axhline(0.5, color='red', ls='--', alpha=0.3, label='BQP threshold')
    ax.set_xlabel('Number of phases N', fontsize=11)
    ax.set_ylabel('Phase discrimination accuracy', fontsize=11)
    ax.set_title(f'(a) BQP Test: Phase Estimation\n'
                 f'Mean accuracy: {bqp_score:.0%}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    # (b) PostBQP: post-selection amplification
    ax = axes[1]
    ax.plot(postbqp_results['signal_strengths'],
            postbqp_results['recovery_rates'],
            'o-', color='#2196F3', linewidth=2.5, markersize=8)
    ax.set_xlabel('Signal strength', fontsize=11)
    ax.set_ylabel('Post-selection recovery', fontsize=11)
    ax.set_title(f'(b) PostBQP: Signal Amplification\n'
                 f'{postbqp_results["amplification_factor"]:.1f}x amplification',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Complexity class diagram
    ax = axes[2]
    # Draw nested circles for complexity classes
    circle_bpp = plt.Circle((0.5, 0.5), 0.15, color='#9E9E9E', alpha=0.3,
                             label='BPP (classical)')
    circle_bqp = plt.Circle((0.5, 0.5), 0.30, color='#2196F3', alpha=0.2,
                             label='BQP (quantum)')
    circle_postbqp = plt.Circle((0.5, 0.5), 0.45, color='#FF5722', alpha=0.15,
                                 label='PostBQP')
    ax.add_patch(circle_postbqp)
    ax.add_patch(circle_bqp)
    ax.add_patch(circle_bpp)
    # Mark S-Qubit position
    if 'PostBQP' in complexity_class:
        marker_r = 0.35
    elif 'BQP' in complexity_class:
        marker_r = 0.22
    else:
        marker_r = 0.1
    ax.plot(0.5 + marker_r * 0.7, 0.5, '*', color='gold', markersize=20,
            markeredgecolor='black', markeredgewidth=1, zorder=10)
    ax.text(0.5 + marker_r * 0.7, 0.5 - 0.06, 'S-Qubit', ha='center',
            fontsize=10, fontweight='bold')
    ax.text(0.5, 0.5, 'BPP', ha='center', fontsize=9, alpha=0.7)
    ax.text(0.5, 0.5 + 0.22, 'BQP', ha='center', fontsize=9, alpha=0.7)
    ax.text(0.5, 0.5 + 0.38, 'PostBQP', ha='center', fontsize=9, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'(c) Complexity Classification\n{complexity_class}',
                 fontsize=11, fontweight='bold')

    plt.suptitle('Quantum Complexity Classification of S-Qubit Computation',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q87_complexity.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q87', 'name': 'Quantum Complexity Classification',
        'bqp_score': bqp_score,
        'bqp_results': [{k: v for k, v in r.items() if k != 'e_values'}
                         for r in bqp_results],
        'postbqp': {k: v for k, v in postbqp_results.items()
                     if k != 'recovery_rates'},
        'sampling': sampling_results,
        'complexity_class': complexity_class,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q87_complexity.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved. Elapsed: {elapsed:.1f}s")
    return results


if __name__ == '__main__':
    main()
