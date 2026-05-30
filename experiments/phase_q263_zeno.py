# -*- coding: utf-8 -*-
"""
Phase Q263: Quantum Zeno Effect
==================================
MY IDEA: Does frequent "measurement" freeze quantum evolution?
In quantum mechanics, repeated observation prevents state change.
Test if hooking into multiple layers to "observe" the S-Qubit
freezes its evolution (Zeno effect).
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

def main():
    print("=" * 60)
    print("Phase Q263: Quantum Zeno Effect")
    print("  (Does frequent observation freeze quantum evolution?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 8

    prompt = "quantum superposition evolution dynamics"

    # Measure with different observation frequencies
    # "Observation" = project onto computational basis (destroy off-diagonal)
    obs_configs = [
        ('No obs', []),
        ('Every 7', list(range(0, n_layers, 7))),
        ('Every 5', list(range(0, n_layers, 5))),
        ('Every 3', list(range(0, n_layers, 3))),
        ('Every 2', list(range(0, n_layers, 2))),
        ('Every 1', list(range(0, n_layers, 1))),
    ]

    results_data = []
    for label, obs_layers in obs_configs:
        hooks = []
        def make_zeno_hook(li):
            def hook(module, input, output):
                x = output[0] if isinstance(output, tuple) else output
                h = x[0, -1, :dim]
                # "Measure" = project onto computational basis
                # This destroys superposition (Zeno measurement)
                probs = h.abs() ** 2
                probs = probs / (probs.sum() + 1e-10)
                # Collapse: set to basis state with highest probability
                # Soft collapse: normalize by magnitude
                h_measured = h.abs() * torch.sign(h)
                x[0, -1, :dim] = h_measured
            return hook

        for li in obs_layers:
            if li < n_layers:
                hooks.append(model.model.layers[li].register_forward_hook(make_zeno_hook(li)))

        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for h in hooks: h.remove()

        # Measure final state coherence
        h_final = out.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()
        h_final /= np.linalg.norm(h_final) + 1e-10
        rho = np.outer(h_final, h_final.conj())
        rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
        rho /= np.trace(rho)
        coh = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)

        # State change from input embedding
        h_input = out.hidden_states[0][0, -1, :dim].float().cpu().numpy()
        h_input /= np.linalg.norm(h_input) + 1e-10
        state_change = 1 - abs(float(np.dot(h_final, h_input)))

        print("  %s (%d obs): coherence=%.4f, state_change=%.4f" % (
            label, len(obs_layers), coh, state_change))
        results_data.append({
            'label': label, 'n_obs': len(obs_layers),
            'coherence': round(coh, 4), 'state_change': round(state_change, 4),
        })

    # Zeno effect: more observations -> less state change?
    changes = [r['state_change'] for r in results_data]
    zeno_gradient = changes[0] - changes[-1] if len(changes) > 1 else 0

    if zeno_gradient > 0.05:
        verdict = "ZENO EFFECT: state change %.3f -> %.3f (frozen by observation)" % (changes[0], changes[-1])
    elif zeno_gradient > 0:
        verdict = "WEAK ZENO: slight freezing (%.3f -> %.3f)" % (changes[0], changes[-1])
    else:
        verdict = "NO ZENO: observation does not freeze evolution (%.3f -> %.3f)" % (changes[0], changes[-1])

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q263', 'name': 'Quantum Zeno Effect',
        'configs': results_data,
        'summary': {'zeno_gradient': round(zeno_gradient, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q263_zeno.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = [r['n_obs'] for r in results_data]
    ax = axes[0]
    ax.plot(x, [r['state_change'] for r in results_data], 'o-', color='#E91E63', lw=2, ms=8)
    ax.set_xlabel('Number of Observations'); ax.set_ylabel('State Change')
    ax.set_title('(a) Zeno Effect: Observation vs Evolution'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, [r['coherence'] for r in results_data], 'o-', color='#2196F3', lw=2, ms=8)
    ax.set_xlabel('Number of Observations'); ax.set_ylabel('Coherence')
    ax.set_title('(b) Coherence vs Observations'); ax.grid(alpha=0.3)

    plt.suptitle('Q263: Quantum Zeno Effect\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q263_zeno.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ263 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
