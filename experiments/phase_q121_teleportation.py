# -*- coding: utf-8 -*-
"""
Phase Q121: Quantum Teleportation Protocol
===========================================
Tests whether semantic states can be "teleported" between
non-adjacent layers using entanglement as a resource.

Protocol:
1. Create entangled S-Qubit pair at layers A and B
2. Measure (destructively) at layer A with "message" state
3. Apply correction at layer B
4. Verify the teleported state matches the original

This is the semantic analogue of quantum teleportation,
exploiting the entanglement confirmed in earlier phases.
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
    print("Phase Q121: Quantum Teleportation Protocol")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # Messages to teleport
    messages = [
        "The capital of France is Paris",
        "Water boils at 100 degrees",
        "Pi equals 3.14159",
        "Gravity pulls objects toward Earth",
        "DNA carries genetic information",
    ]

    # Layer pairs for teleportation (source, destination)
    layer_pairs = [
        (2, n_layers - 2),    # early -> late
        (5, n_layers // 2),   # early -> middle
        (n_layers // 2, n_layers - 1),  # middle -> late
        (3, 10),              # nearby layers
        (1, n_layers - 1),    # extreme: first -> last
    ]

    all_results = []

    for msg in messages:
        print("\n  Message: '%s'" % msg[:40])
        inp = tok(msg, return_tensors='pt').to(device)

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        msg_results = []
        for src, dst in layer_pairs:
            if src >= n_layers or dst >= n_layers:
                continue

            # Step 1: Get "message" hidden state at source layer
            h_src = out.hidden_states[src + 1][0, -1, :].float()

            # Step 2: Get entangled pair (A, B) states
            # A = source layer representation, B = destination layer representation
            h_a = out.hidden_states[src + 1][0, -1, :].float()
            h_b = out.hidden_states[dst + 1][0, -1, :].float()

            # Step 3: "Bell measurement" at source
            # Project message onto A's basis -> get measurement outcome
            cos_ma = torch.nn.functional.cosine_similarity(
                h_src.unsqueeze(0), h_a.unsqueeze(0)).item()

            # Step 4: Apply "correction" at destination
            # Teleported state = B adjusted by measurement outcome
            # In real QT: apply Pauli correction based on Bell measurement
            # Semantic analogue: scale B by the phase relationship
            phase_correction = torch.atan2(
                (h_src - h_a)[1::2].mean(),
                (h_src - h_a)[::2].mean()).item()

            h_teleported = h_b * np.cos(phase_correction) + \
                           torch.roll(h_b, 1, dims=-1) * np.sin(phase_correction)

            # Step 5: Verify - how similar is teleported state to original message?
            fidelity = torch.nn.functional.cosine_similarity(
                h_src.unsqueeze(0), h_teleported.unsqueeze(0)).item()

            # Classical baseline: just use destination layer directly (no teleportation)
            classical_fidelity = torch.nn.functional.cosine_similarity(
                h_src.unsqueeze(0), h_b.unsqueeze(0)).item()

            # Quantum advantage: did teleportation improve fidelity?
            advantage = fidelity - classical_fidelity

            msg_results.append({
                'src_layer': src,
                'dst_layer': dst,
                'distance': dst - src,
                'fidelity': round(fidelity, 6),
                'classical_fidelity': round(classical_fidelity, 6),
                'advantage': round(advantage, 6),
                'phase_correction': round(phase_correction, 4)
            })

        all_results.append({
            'message': msg[:40],
            'teleportations': msg_results
        })

        # Best result for this message
        if msg_results:
            best = max(msg_results, key=lambda x: x['fidelity'])
            print("    Best: L%d->L%d fidelity=%.4f (classical=%.4f, adv=%.4f)" %
                  (best['src_layer'], best['dst_layer'], best['fidelity'],
                   best['classical_fidelity'], best['advantage']))

    # ===== Distance-Fidelity Analysis =====
    print("\n--- Distance-Fidelity Analysis ---")
    all_pairs = []
    for mr in all_results:
        all_pairs.extend(mr['teleportations'])

    # Group by distance
    from collections import defaultdict
    dist_groups = defaultdict(list)
    for p in all_pairs:
        dist_groups[p['distance']].append(p)

    distance_fidelity = []
    for dist in sorted(dist_groups.keys()):
        fids = [p['fidelity'] for p in dist_groups[dist]]
        advs = [p['advantage'] for p in dist_groups[dist]]
        distance_fidelity.append({
            'distance': dist,
            'mean_fidelity': round(float(np.mean(fids)), 4),
            'mean_advantage': round(float(np.mean(advs)), 4),
            'n_samples': len(fids)
        })
        print("  Distance %d: fidelity=%.4f, advantage=%.4f" %
              (dist, np.mean(fids), np.mean(advs)))

    # Overall statistics
    mean_fidelity = float(np.mean([p['fidelity'] for p in all_pairs]))
    mean_advantage = float(np.mean([p['advantage'] for p in all_pairs]))
    teleportation_success = sum(1 for p in all_pairs if p['fidelity'] > 0.5)

    print("\n  Overall fidelity: %.4f" % mean_fidelity)
    print("  Teleportation success (F>0.5): %d/%d" %
          (teleportation_success, len(all_pairs)))

    # ===== Save Results =====
    results = {
        'phase': 'Q121',
        'name': 'Quantum Teleportation Protocol',
        'messages': all_results,
        'distance_fidelity': distance_fidelity,
        'mean_fidelity': round(mean_fidelity, 4),
        'mean_advantage': round(mean_advantage, 4),
        'teleportation_success_rate': round(teleportation_success / max(len(all_pairs), 1), 4),
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q121_teleportation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Fidelity by layer pair
    ax = axes[0]
    for i, mr in enumerate(all_results):
        dists = [t['distance'] for t in mr['teleportations']]
        fids = [t['fidelity'] for t in mr['teleportations']]
        ax.plot(dists, fids, 'o-', label='Msg %d' % (i+1), markersize=5, alpha=0.7)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Success threshold')
    ax.set_xlabel('Layer distance')
    ax.set_ylabel('Teleportation fidelity')
    ax.set_title('(a) Fidelity vs Distance')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Quantum vs Classical fidelity
    ax = axes[1]
    q_fids = [p['fidelity'] for p in all_pairs]
    c_fids = [p['classical_fidelity'] for p in all_pairs]
    ax.scatter(c_fids, q_fids, c='#4CAF50', alpha=0.6, s=40, edgecolors='black')
    lims = [min(min(q_fids), min(c_fids)) - 0.05, max(max(q_fids), max(c_fids)) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='x=y')
    ax.set_xlabel('Classical fidelity')
    ax.set_ylabel('Teleported fidelity')
    ax.set_title('(b) Teleported vs Classical\n(above line = QT wins)')
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Distance-averaged fidelity
    ax = axes[2]
    dists = [d['distance'] for d in distance_fidelity]
    mean_f = [d['mean_fidelity'] for d in distance_fidelity]
    mean_a = [d['mean_advantage'] for d in distance_fidelity]
    ax.bar(range(len(dists)), mean_f, color='#2196F3', alpha=0.85, label='Fidelity')
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(['d=%d' % d for d in dists], fontsize=8)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5)
    ax.set_ylabel('Mean fidelity')
    ax.set_title('(c) Distance-Averaged Fidelity')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q121: Semantic Quantum Teleportation (F=%.3f)' % mean_fidelity,
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q121_teleportation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ121 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
