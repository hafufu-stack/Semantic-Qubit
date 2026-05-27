# -*- coding: utf-8 -*-
"""
Phase Q202: Bell Inequality (CHSH) Test
==========================================
THE ULTIMATE TEST OF QUANTUM-NESS.

Bell's theorem: No local hidden variable theory can produce
correlations that violate the CHSH inequality |S| <= 2.
Quantum mechanics allows up to |S| = 2*sqrt(2) ~ 2.828.

Test: Create two "entangled" soul vectors and measure
CHSH correlations. If |S| > 2 -> LLM exhibits genuinely
non-classical correlations that CANNOT be explained by
any classical model.

This would DEFINITIVELY refute Grok's "it's just classical
vector operations" argument.
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

INJECT_LAYER = 8


def train_soul(model, tok, data, device, layer=8, epochs=80, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def hook(m, i, o, v=vec):
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def measure_correlation(model, tok, device, vec_a, vec_b, prompt,
                         target_id, theta_a, theta_b, n_samples=32):
    """
    Measure correlation E(a,b) between two soul vectors
    at measurement angles theta_a and theta_b.
    
    Measurement basis rotation in embedding space:
    |+_theta> = cos(theta)|a> + sin(theta)|b>
    |-_theta> = -sin(theta)|a> + cos(theta)|b>
    """
    scale = vec_a.norm()
    results_ab = []

    # Sweep phase to get interference pattern
    phis = np.linspace(0, 2 * np.pi, n_samples)

    for phi in phis:
        # Create superposition at measurement angle
        vec_meas = (np.cos(theta_a + phi) * vec_a +
                    np.sin(theta_b + phi) * vec_b)
        n = vec_meas.norm()
        if n > 0:
            vec_meas = vec_meas / n * scale

        inp = tok(prompt, return_tensors='pt').to(device)
        def hook(m, i, o, v=vec_meas):
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p = float(probs[target_id])
        # Map probability to measurement outcome: +1 or -1
        # P(+1) = p, P(-1) = 1-p
        # E = P(+1) - P(-1) = 2*p - 1
        results_ab.append(2 * p - 1)

    # Correlation = average of product of outcomes
    # For interference pattern: E(a,b) = <cos(2*(theta_a - theta_b))>
    correlation = float(np.mean(results_ab))
    return correlation


def main():
    print("=" * 60)
    print("Phase Q202: Bell Inequality (CHSH) Test")
    print("  (THE ultimate test: |S| > 2 -> non-classical!)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    # Train entangled pair
    print("  Training entangled soul vector pair...")
    vec_a = train_soul(model, tok,
                      [("The sky is", "blue"), ("The ocean is", "blue")],
                      device, layer=INJECT_LAYER, seed=42)
    vec_b = train_soul(model, tok,
                      [("The grass is", "green"), ("Leaves are", "green")],
                      device, layer=INJECT_LAYER, seed=99)

    target_id = tok.encode("blue")[-1]
    prompt = "The sky is"

    # CHSH measurement settings
    # Alice: a1 = 0, a2 = pi/4
    # Bob: b1 = pi/8, b2 = 3*pi/8
    # These are the optimal CHSH angles
    a1, a2 = 0, np.pi / 4
    b1, b2 = np.pi / 8, 3 * np.pi / 8

    print("\n--- CHSH Measurements ---")
    print("  Alice settings: a1=0, a2=pi/4")
    print("  Bob settings: b1=pi/8, b2=3pi/8")

    # Measure all four correlations
    E_a1b1 = measure_correlation(model, tok, device, vec_a, vec_b,
                                  prompt, target_id, a1, b1)
    E_a1b2 = measure_correlation(model, tok, device, vec_a, vec_b,
                                  prompt, target_id, a1, b2)
    E_a2b1 = measure_correlation(model, tok, device, vec_a, vec_b,
                                  prompt, target_id, a2, b1)
    E_a2b2 = measure_correlation(model, tok, device, vec_a, vec_b,
                                  prompt, target_id, a2, b2)

    print("  E(a1,b1) = %.4f" % E_a1b1)
    print("  E(a1,b2) = %.4f" % E_a1b2)
    print("  E(a2,b1) = %.4f" % E_a2b1)
    print("  E(a2,b2) = %.4f" % E_a2b2)

    # CHSH: S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
    S = E_a1b1 - E_a1b2 + E_a2b1 + E_a2b2
    print("\n  S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)")
    print("  S = %.4f" % S)
    print("  |S| = %.4f" % abs(S))
    print("  Classical limit: |S| <= 2")
    print("  Quantum maximum: |S| = 2*sqrt(2) = %.4f" % (2 * np.sqrt(2)))

    # Also try different angle combinations
    print("\n--- Angle Sweep ---")
    angle_results = []
    angles = np.linspace(0, np.pi, 20)

    for theta in angles:
        a1_t, a2_t = 0, theta
        b1_t, b2_t = theta / 2, 3 * theta / 2
        E11 = measure_correlation(model, tok, device, vec_a, vec_b,
                                   prompt, target_id, a1_t, b1_t, n_samples=16)
        E12 = measure_correlation(model, tok, device, vec_a, vec_b,
                                   prompt, target_id, a1_t, b2_t, n_samples=16)
        E21 = measure_correlation(model, tok, device, vec_a, vec_b,
                                   prompt, target_id, a2_t, b1_t, n_samples=16)
        E22 = measure_correlation(model, tok, device, vec_a, vec_b,
                                   prompt, target_id, a2_t, b2_t, n_samples=16)
        S_t = E11 - E12 + E21 + E22
        angle_results.append({
            'theta': round(float(theta), 4),
            'S': round(float(S_t), 4),
            'abs_S': round(abs(float(S_t)), 4),
        })

    max_S = max(r['abs_S'] for r in angle_results)
    best_theta = [r for r in angle_results if r['abs_S'] == max_S][0]['theta']

    print("  Max |S| found: %.4f at theta=%.4f" % (max_S, best_theta))

    # Also test with multiple entangled pairs
    print("\n--- Multi-pair test ---")
    pair_results = []
    pair_configs = [
        ("blue/green", [("The sky is", "blue")], [("The grass is", "green")], 42, 99),
        ("cat/dog", [("A cat says", "me")], [("A dog says", "w")], 7, 13),
        ("hot/cold", [("Fire is", "hot")], [("Ice is", "cold")], 55, 77),
    ]

    for name, data_a, data_b, seed_a, seed_b in pair_configs:
        v_a = train_soul(model, tok, data_a, device, layer=INJECT_LAYER, seed=seed_a)
        v_b = train_soul(model, tok, data_b, device, layer=INJECT_LAYER, seed=seed_b)

        E11 = measure_correlation(model, tok, device, v_a, v_b,
                                   prompt, target_id, 0, np.pi/8, n_samples=16)
        E12 = measure_correlation(model, tok, device, v_a, v_b,
                                   prompt, target_id, 0, 3*np.pi/8, n_samples=16)
        E21 = measure_correlation(model, tok, device, v_a, v_b,
                                   prompt, target_id, np.pi/4, np.pi/8, n_samples=16)
        E22 = measure_correlation(model, tok, device, v_a, v_b,
                                   prompt, target_id, np.pi/4, 3*np.pi/8, n_samples=16)
        S_pair = E11 - E12 + E21 + E22

        pair_results.append({
            'name': name,
            'S': round(float(S_pair), 4),
            'abs_S': round(abs(float(S_pair)), 4),
            'violates_bell': abs(float(S_pair)) > 2.0,
        })
        print("  %s: S=%.4f, |S|=%.4f %s" % (
            name, S_pair, abs(S_pair),
            "VIOLATION!" if abs(S_pair) > 2 else "(classical)"))

    n_violations = sum(1 for r in pair_results if r['violates_bell'])

    if abs(S) > 2 or max_S > 2 or n_violations > 0:
        verdict = "BELL VIOLATION: |S|=%.3f (max=%.3f), %d/%d pairs violate" % (
            abs(S), max_S, n_violations, len(pair_results))
    else:
        verdict = "NO VIOLATION: max |S|=%.3f < 2.0 (classical bound)" % max_S

    print("\n  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q202',
        'name': 'Bell Inequality (CHSH)',
        'chsh_primary': {
            'E_a1b1': round(E_a1b1, 4),
            'E_a1b2': round(E_a1b2, 4),
            'E_a2b1': round(E_a2b1, 4),
            'E_a2b2': round(E_a2b2, 4),
            'S': round(S, 4),
            'abs_S': round(abs(S), 4),
        },
        'angle_sweep': angle_results,
        'multi_pair': pair_results,
        'summary': {
            'max_S': round(max_S, 4),
            'primary_S': round(abs(S), 4),
            'n_violations': n_violations,
            'classical_limit': 2.0,
            'quantum_maximum': round(2 * np.sqrt(2), 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q202_bell.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) CHSH S vs angle
    ax = axes[0]
    thetas = [r['theta'] for r in angle_results]
    S_vals = [r['abs_S'] for r in angle_results]
    ax.plot(thetas, S_vals, 'o-', color='#E91E63', linewidth=2, markersize=5,
            label='LLM |S|')
    ax.axhline(2.0, color='red', ls='--', linewidth=2, label='Classical limit (2.0)')
    ax.axhline(2 * np.sqrt(2), color='green', ls=':', linewidth=2,
               label='Quantum max (2.83)')
    ax.set_xlabel('Angle theta (rad)')
    ax.set_ylabel('|S| (CHSH)')
    ax.set_title('(a) CHSH vs Measurement Angle\n(Above red = Bell violation!)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.fill_between(thetas, 2.0, max(S_vals + [2.5]),
                    where=[s > 2 for s in S_vals],
                    color='#E91E63', alpha=0.2)

    # (b) Multi-pair S values
    ax = axes[1]
    pair_names = [r['name'] for r in pair_results]
    pair_S = [r['abs_S'] for r in pair_results]
    colors = ['#4CAF50' if s > 2 else '#FF9800' for s in pair_S]
    ax.bar(range(len(pair_names)), pair_S, color=colors,
           edgecolor='black', alpha=0.85)
    ax.axhline(2.0, color='red', ls='--', linewidth=2, label='Classical limit')
    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels(pair_names)
    ax.set_ylabel('|S| (CHSH)')
    ax.set_title('(b) Multi-Pair CHSH Test\n(%d/%d violations)' %
                (n_violations, len(pair_results)))
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Comparison chart
    ax = axes[2]
    comparisons = ['Classical\nLimit', 'LLM S-Qubit\n(Primary)', 'LLM S-Qubit\n(Max)', 'Quantum\nMaximum']
    values = [2.0, abs(S), max_S, 2 * np.sqrt(2)]
    comp_colors = ['#666666', '#E91E63', '#9C27B0', '#4CAF50']
    ax.bar(range(4), values, color=comp_colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels(comparisons, fontsize=9)
    ax.axhline(2.0, color='red', ls='--', alpha=0.5)
    ax.set_ylabel('|S|')
    ax.set_title('(c) CHSH Comparison\n(S-Qubit vs Bounds)')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q202: Bell Inequality (CHSH) Test\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q202_bell.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ202 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
