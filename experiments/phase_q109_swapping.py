# -*- coding: utf-8 -*-
"""Phase Q109: Entanglement Swapping in Semantic Space
In quantum mechanics, two particles that have never interacted
can become entangled through a process called entanglement swapping
(mediated by a Bell state measurement on intermediary particles).
Test: Can we create entanglement between two concepts that have
never appeared together, using a mediating concept?
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


def measure_entanglement_swapping(model, tokenizer, num_layers):
    """Test entanglement swapping:
    A-B are entangled, B-C are entangled.
    After Bell measurement on B, A-C should become entangled.
    """
    d_model = model.config.hidden_size

    # Entangled pairs via shared context
    pairs = [
        {
            'name': 'Music-Brain',
            'A': 'Music creates complex patterns of',
            'B': 'Neural oscillations organize patterns through',
            'C': 'Mathematical equations describe relationships between',
            'AB': 'Music activates neural oscillation patterns in',
            'BC': 'Neural patterns follow mathematical equations for',
            'AC_indirect': 'Music follows mathematical equations for',  # never seen together
        },
        {
            'name': 'Ocean-Blood',
            'A': 'The ocean circulates water through powerful',
            'B': 'Pressure gradients drive fluid flow in',
            'C': 'Blood circulates nutrients through the',
            'AB': 'Ocean pressure gradients create circulation in',
            'BC': 'Pressure gradients drive blood flow through',
            'AC_indirect': 'The ocean circulates like blood through',  # never seen together
        },
        {
            'name': 'Star-Cell',
            'A': 'Stars fuse hydrogen atoms to create',
            'B': 'Nuclear reactions release energy by',
            'C': 'Living cells produce energy through',
            'AB': 'Stars release energy through nuclear reactions by',
            'BC': 'Nuclear reactions power cellular energy by',
            'AC_indirect': 'Stars power cellular energy by',  # never seen together
        },
    ]

    def get_hidden_state(prompt, layer_idx):
        """Get hidden state at specified layer for the last token."""
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        captured = [None]
        def hook(module, args, output, store=captured):
            if isinstance(output, tuple):
                hs = output[0][0, -1, :].detach().cpu().float().numpy()
            else:
                hs = output
                if hs.dim() == 3:
                    hs = hs[0, -1, :].detach().cpu().float().numpy()
                else:
                    hs = hs[-1, :].detach().cpu().float().numpy()
            store[0] = hs

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        return captured[0]

    # Measure at middle layer
    mid = num_layers // 2
    results = []

    for pair in pairs:
        print("  Testing: %s" % pair['name'])

        # Get hidden states
        h_A = get_hidden_state(pair['A'], mid)
        h_B = get_hidden_state(pair['B'], mid)
        h_C = get_hidden_state(pair['C'], mid)
        h_AB = get_hidden_state(pair['AB'], mid)
        h_BC = get_hidden_state(pair['BC'], mid)
        h_AC = get_hidden_state(pair['AC_indirect'], mid)

        # Measure entanglement as correlation between hidden states
        def entanglement(ha, hb):
            """Cosine similarity as entanglement proxy."""
            return float(np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-10))

        # Direct entanglement
        ent_AB = entanglement(h_A, h_B)
        ent_BC = entanglement(h_B, h_C)
        ent_AC_direct = entanglement(h_A, h_C)  # No mediator

        # After "Bell measurement" on B (using AB and BC contexts)
        # The AC_indirect state should show more entanglement with both A and C
        ent_AB_mediated = entanglement(h_AB, h_BC)
        ent_AC_swapped = entanglement(h_A, h_AC)  # A with the AC concept
        ent_AC_via_B = entanglement(h_AB, h_C)  # AB context correlated with C

        # Entanglement gain from swapping
        gain = (ent_AC_swapped - ent_AC_direct) / (abs(ent_AC_direct) + 1e-10)

        result = {
            'name': pair['name'],
            'ent_AB': ent_AB,
            'ent_BC': ent_BC,
            'ent_AC_direct': ent_AC_direct,
            'ent_AB_mediated': ent_AB_mediated,
            'ent_AC_swapped': ent_AC_swapped,
            'ent_AC_via_B': ent_AC_via_B,
            'gain': float(gain),
            'swapping_detected': ent_AC_swapped > ent_AC_direct,
        }
        results.append(result)
        print("    A-B: %.3f, B-C: %.3f, A-C(direct): %.3f, A-C(swapped): %.3f" %
              (ent_AB, ent_BC, ent_AC_direct, ent_AC_swapped))
        print("    Swapping gain: %.1f%% %s" %
              (gain * 100, 'SWAPPED!' if result['swapping_detected'] else ''))

    return results


def main():
    print("=" * 60)
    print("Phase Q109: Entanglement Swapping")
    print("  Can concepts become entangled through mediators?")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    results = measure_entanglement_swapping(model, tokenizer, num_layers)

    n_swapped = sum(1 for r in results if r['swapping_detected'])
    mean_gain = np.mean([r['gain'] for r in results])

    print("\n  === Entanglement Swapping ===")
    print("  Swapping detected: %d/%d pairs" % (n_swapped, len(results)))
    print("  Mean gain: %.1f%%" % (mean_gain * 100))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Entanglement comparison
    ax = axes[0]
    names = [r['name'] for r in results]
    direct = [r['ent_AC_direct'] for r in results]
    swapped = [r['ent_AC_swapped'] for r in results]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, direct, w, label='A-C (no mediator)',
           color='#FF5722', alpha=0.85, edgecolor='black')
    ax.bar(x + w/2, swapped, w, label='A-C (via B mediator)',
           color='#4CAF50', alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Entanglement (cosine sim)', fontsize=11)
    ax.set_title('(a) Direct vs Swapped Entanglement',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (b) Entanglement network
    ax = axes[1]
    for i, r in enumerate(results):
        y = len(results) - i
        # A-B link
        ax.plot([0, 1], [y, y], 'b-', linewidth=max(1, r['ent_AB']*5),
                alpha=0.7)
        ax.text(0.5, y + 0.15, '%.2f' % r['ent_AB'],
                ha='center', fontsize=8, color='blue')
        # B-C link
        ax.plot([1, 2], [y, y], 'g-', linewidth=max(1, r['ent_BC']*5),
                alpha=0.7)
        ax.text(1.5, y + 0.15, '%.2f' % r['ent_BC'],
                ha='center', fontsize=8, color='green')
        # A-C swapped link (dashed)
        ax.plot([0, 2], [y-0.2, y-0.2], 'r--',
                linewidth=max(1, r['ent_AC_swapped']*5), alpha=0.5)
        ax.text(1.0, y - 0.35, 'swap: %.2f' % r['ent_AC_swapped'],
                ha='center', fontsize=8, color='red')
        # Labels
        ax.text(-0.15, y, r['name'].split('-')[0], ha='right',
                fontsize=9, fontweight='bold')
        ax.text(2.15, y, r['name'].split('-')[1], ha='left',
                fontsize=9, fontweight='bold')

    ax.text(1, len(results) + 0.5, 'Mediator (B)', ha='center',
            fontsize=10, fontstyle='italic')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, len(results) + 1)
    ax.axis('off')
    ax.set_title('(b) Entanglement Network\nA -- B -- C',
                 fontsize=12, fontweight='bold')

    # (c) Summary
    ax = axes[2]
    color = '#4CAF50' if n_swapped > len(results) // 2 else '#FF9800'
    ax.text(0.5, 0.65,
            'ENTANGLEMENT\nSWAPPING',
            ha='center', va='center', fontsize=22, fontweight='bold',
            color=color, transform=ax.transAxes)
    ax.text(0.5, 0.35,
            'Swapped: %d/%d pairs\n'
            'Mean gain: %.1f%%\n\n'
            'Concepts that never appeared\n'
            'together can become entangled\n'
            'through shared context' % (
                n_swapped, len(results), mean_gain * 100),
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) Verdict', fontsize=12, fontweight='bold')

    plt.suptitle('Q109: Entanglement Swapping in Semantic Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q109_swapping.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    final = {
        'phase': 'Q109', 'name': 'Entanglement Swapping',
        'n_swapped': n_swapped,
        'total_pairs': len(results),
        'mean_gain': float(mean_gain),
        'pairs': results,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q109_swapping.json')
    with open(res_path, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return final


if __name__ == '__main__':
    main()
