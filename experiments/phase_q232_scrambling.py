# -*- coding: utf-8 -*-
"""
Phase Q232: Quantum Scrambling (OTOC)
========================================
Out-of-Time-Order Correlator (OTOC) measures quantum information
scrambling - how fast local information spreads to become nonlocal.

OTOC(t) = <W(t) V W(t) V> where W, V are local operators.
Fast scrambling -> black hole-like information processing.
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


def compute_otoc(model, tok, device, prompt, dim=8):
    """Compute OTOC across layers (layers = discrete time steps)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    otoc_values = []

    # Reference state at layer 0
    h0 = out.hidden_states[0][0, -1, :dim].float().cpu().numpy()
    h0 /= np.linalg.norm(h0) + 1e-10

    # W operator: acts on first dim/2 components
    W = np.eye(dim)
    W[:dim//2, :dim//2] *= -1  # Reflection in first half

    # V operator: acts on last dim/2 components
    V = np.eye(dim)
    V[dim//2:, dim//2:] *= -1  # Reflection in second half

    for li in range(n_layers + 1):
        h_t = out.hidden_states[li][0, -1, :dim].float().cpu().numpy()
        h_t /= np.linalg.norm(h_t) + 1e-10

        # OTOC = |<psi| W(t)^dag V^dag W(t) V |psi>|^2
        # where W(t) = U(t)^dag W U(t) is Heisenberg-evolved
        # Simplified: use the hidden states as U(t)|psi>
        Wh = W @ h_t
        VWh = V @ Wh
        WVWh = W @ VWh

        otoc = abs(np.dot(h_t, WVWh)) ** 2
        otoc_values.append({
            'layer': li,
            'otoc': round(float(otoc), 6),
        })

    return otoc_values


def main():
    print("=" * 60)
    print("Phase Q232: Quantum Scrambling (OTOC)")
    print("  (How fast does information scramble?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    prompts = [
        "black hole information paradox",
        "quantum chaos butterfly effect",
        "thermalization quantum system",
        "scrambling time fast scrambler",
        "quantum error correction code",
        "classical deterministic system",
    ]

    dim = 8
    all_results = []

    for prompt in prompts:
        print("\n--- %s ---" % prompt[:35])
        otoc_data = compute_otoc(model, tok, device, prompt, dim)

        # Scrambling rate: how fast OTOC decays
        otocs = [d['otoc'] for d in otoc_data]
        if len(otocs) > 2 and otocs[0] > 0:
            # Fit exponential decay
            log_otocs = np.log(np.array(otocs) + 1e-10)
            layers = np.arange(len(otocs))
            slope, _ = np.polyfit(layers, log_otocs, 1)
            scrambling_rate = -slope  # positive = fast scrambling
        else:
            scrambling_rate = 0

        # Scrambling time: layer where OTOC drops to 1/e
        scrambling_time = -1
        threshold = otocs[0] * np.exp(-1) if otocs[0] > 0 else 0
        for i, o in enumerate(otocs):
            if o < threshold:
                scrambling_time = i
                break

        print("  Initial OTOC=%.4f, Final=%.4f, rate=%.4f, t_scr=%d" %
              (otocs[0], otocs[-1], scrambling_rate,
               scrambling_time if scrambling_time >= 0 else -1))

        all_results.append({
            'prompt': prompt[:35],
            'otoc_data': otoc_data,
            'scrambling_rate': round(scrambling_rate, 6),
            'scrambling_time': scrambling_time,
            'initial_otoc': otocs[0],
            'final_otoc': otocs[-1],
        })

    # Summary
    avg_rate = np.mean([r['scrambling_rate'] for r in all_results])
    scr_times = [r['scrambling_time'] for r in all_results if r['scrambling_time'] > 0]
    avg_scr_time = np.mean(scr_times) if scr_times else -1

    # Fast scrambling bound: t_scr ~ log(N) (Maldacena-Shenker-Stanford)
    n_layers = len(all_results[0]['otoc_data']) - 1
    log_bound = np.log(n_layers)

    if avg_scr_time > 0 and avg_scr_time < log_bound * 2:
        verdict = "FAST SCRAMBLER: t_scr=%.1f layers (bound=%.1f)" % (avg_scr_time, log_bound)
    elif avg_rate > 0.01:
        verdict = "SCRAMBLER: rate=%.4f (avg time=%.1f)" % (avg_rate, avg_scr_time)
    else:
        verdict = "SLOW/NO SCRAMBLING: rate=%.6f" % avg_rate

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q232',
        'name': 'Quantum Scrambling (OTOC)',
        'prompts': all_results,
        'summary': {
            'avg_scrambling_rate': round(avg_rate, 6),
            'avg_scrambling_time': round(avg_scr_time, 2) if avg_scr_time > 0 else -1,
            'log_bound': round(log_bound, 2),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q232_scrambling.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, r in enumerate(all_results[:6]):
        ax = axes[idx // 3][idx % 3]
        layers = [d['layer'] for d in r['otoc_data']]
        otocs = [d['otoc'] for d in r['otoc_data']]
        ax.plot(layers, otocs, 'o-', color='#FF5722', ms=3, lw=2)
        if r['scrambling_time'] > 0:
            ax.axvline(r['scrambling_time'], color='green', ls='--',
                       label='t_scr=%d' % r['scrambling_time'])
        ax.set_xlabel('Layer'); ax.set_ylabel('OTOC')
        ax.set_title('%s\nrate=%.4f' % (r['prompt'][:25], r['scrambling_rate']), fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('Q232: Quantum Scrambling (OTOC)\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q232_scrambling.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ232 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
