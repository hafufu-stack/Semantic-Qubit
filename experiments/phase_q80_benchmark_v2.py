# -*- coding: utf-8 -*-
"""
Phase Q80: Grand Benchmark v2 (Season 6 Integration)
=======================================================
Update the QAS score to include all Season 6 experiments (Q68-Q79).
Comprehensive scoring of all quantum phenomena demonstrated.
"""
import torch, json, os, gc, numpy as np, time, sys, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("[Q80] Grand Benchmark v2 - Season 6 Integration")
    start = time.time()

    # Load all results
    all_results = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, 'phase_q*.json'))):
        with open(f) as fh:
            data = json.load(fh)
            phase = data.get('phase', os.path.basename(f).replace('.json', ''))
            all_results[phase] = data

    print("  Loaded %d phase results" % len(all_results))

    # Score each quantum phenomenon category
    categories = {
        'Superposition & Interference': {
            'phases': ['Q1', 'Q3', 'Q4', 'Q46'],
            'description': 'Wave-like behavior, Hadamard, destructive interference, parallelism',
            'max_score': 10,
        },
        'Entanglement & Bell': {
            'phases': ['Q2', 'Q5', 'Q8', 'Q10', 'Q11', 'Q16', 'Q27'],
            'description': 'Bell tests, CHSH violation, entanglement probes',
            'max_score': 15,
        },
        'Quantum Gates & Circuits': {
            'phases': ['Q12', 'Q12v2', 'Q12v3', 'Q14', 'Q15', 'Q17', 'Q22'],
            'description': 'Two-qubit gates, Toffoli, NQPU circuits',
            'max_score': 10,
        },
        'Quantum Algorithms': {
            'phases': ['Q18', 'Q23', 'Q35', 'Q39', 'Q41', 'Q42', 'Q44', 'Q47'],
            'description': 'Grover, Deutsch-Jozsa, BV, Simon, QPE, QAOA',
            'max_score': 15,
        },
        'Quantum Communication': {
            'phases': ['Q24', 'Q30', 'Q31', 'Q37', 'Q40'],
            'description': 'Teleportation, entanglement swap, superdense, BB84',
            'max_score': 10,
        },
        'Decoherence & Error Correction': {
            'phases': ['Q6', 'Q6v2', 'Q7', 'Q13', 'Q21', 'Q28', 'Q52', 'Q63', 'Q70', 'Q76'],
            'description': 'Collapse, decoherence, QEC, qLDPC',
            'max_score': 10,
        },
        'Quantum Information Theory': {
            'phases': ['Q19', 'Q43', 'Q49', 'Q56', 'Q65'],
            'description': 'No-cloning, SWAP test, tomography, Holevo, channels',
            'max_score': 10,
        },
        'Advanced Quantum Physics': {
            'phases': ['Q36', 'Q38', 'Q53', 'Q58', 'Q59', 'Q62', 'Q64', 'Q66', 'Q67'],
            'description': 'Dimension law, GHZ, non-unitary, contextuality, DFS, adiabatic, tunneling',
            'max_score': 10,
        },
        'Season 6: Neural-Quantum Bridge': {
            'phases': ['Q68', 'Q69', 'Q71', 'Q72', 'Q73', 'Q74', 'Q77', 'Q78', 'Q79'],
            'description': 'QRAM, theta, NQU, unification, Darwinism, Zeno, universality, uncertainty',
            'max_score': 10,
        },
    }

    scores = {}
    total_score = 0
    total_max = 0

    for cat_name, cat_info in categories.items():
        present = sum(1 for p in cat_info['phases']
                      if any(p.lower() in k.lower() for k in all_results.keys()))
        total = len(cat_info['phases'])
        ratio = present / total if total > 0 else 0
        score = ratio * cat_info['max_score']
        scores[cat_name] = {
            'score': round(score, 1),
            'max': cat_info['max_score'],
            'present': present,
            'total': total,
            'ratio': round(ratio, 3),
        }
        total_score += score
        total_max += cat_info['max_score']
        print("  %s: %.1f/%.0f (%d/%d phases)" % (
            cat_name, score, cat_info['max_score'], present, total))

    qas = round(total_score / total_max * 100, 1) if total_max > 0 else 0
    print("\n  === QUANTUM ADVANTAGE SCORE (QAS) ===")
    print("  QAS = %.1f / 100" % qas)
    print("  Total: %.1f / %.0f" % (total_score, total_max))
    print("  Phases completed: %d" % len(all_results))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) Category radar chart (as bar chart for clarity)
    ax = axes[0]
    cat_names = list(scores.keys())
    cat_scores = [scores[c]['score'] for c in cat_names]
    cat_maxes = [scores[c]['max'] for c in cat_names]
    y = np.arange(len(cat_names))

    colors_bar = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0',
                  '#FF9800', '#00BCD4', '#E91E63', '#607D8B', '#FFEB3B']
    bars_max = ax.barh(y, cat_maxes, color='#E0E0E0', edgecolor='gray', height=0.6)
    bars_score = ax.barh(y, cat_scores, color=[colors_bar[i % len(colors_bar)]
                         for i in range(len(cat_names))],
                         edgecolor='black', height=0.6, alpha=0.85)
    for i, (s, m) in enumerate(zip(cat_scores, cat_maxes)):
        ax.text(m + 0.2, i, '%.1f/%d' % (s, m), va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace(' & ', '\n& ').replace(': ', ':\n')
                        for n in cat_names], fontsize=7)
    ax.set_xlabel('Score')
    ax.set_title('(a) Category Breakdown\nQAS = %.1f/100' % qas, fontweight='bold')
    ax.set_xlim(0, max(cat_maxes) + 3)

    # (b) Timeline of phase completions
    ax = axes[1]
    phase_nums = []
    for k in all_results:
        try:
            num = int(''.join(c for c in k if c.isdigit()))
            phase_nums.append(num)
        except:
            pass
    phase_nums.sort()
    cumulative = np.arange(1, len(phase_nums) + 1)
    ax.plot(phase_nums, cumulative, '-', color='#FF5722', linewidth=2)
    ax.fill_between(phase_nums, cumulative, alpha=0.15, color='#FF5722')
    ax.set_xlabel('Phase number')
    ax.set_ylabel('Cumulative experiments')
    ax.set_title('(b) Experiment Timeline\n%d phases completed' % len(phase_nums),
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) QAS gauge
    ax = axes[2]
    theta = np.linspace(0, np.pi, 100)
    # Draw gauge arc
    for r in [0.8, 1.0]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), '-', color='#E0E0E0', linewidth=3)
    # Color regions
    for i, (t0, t1, color) in enumerate([
        (0, 0.3, '#F44336'), (0.3, 0.6, '#FF9800'),
        (0.6, 0.8, '#FFC107'), (0.8, 1.0, '#4CAF50')]):
        t_range = np.linspace(np.pi * (1 - t1), np.pi * (1 - t0), 50)
        ax.fill_between(0.9 * np.cos(t_range), 0.9 * np.sin(t_range),
                         np.zeros_like(t_range), alpha=0.3, color=color)
    # Needle
    needle_angle = np.pi * (1 - qas / 100)
    ax.plot([0, 0.75 * np.cos(needle_angle)],
            [0, 0.75 * np.sin(needle_angle)],
            '-', color='red', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=8)
    ax.text(0, -0.15, 'QAS = %.1f' % qas, ha='center',
            fontsize=18, fontweight='bold', color='#FF5722')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) Quantum Advantage Score', fontweight='bold')

    plt.suptitle('Phase Q80: Grand Benchmark v2\n'
                 'Season 6 Complete - %d Quantum Experiments' % len(all_results),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q80_benchmark_v2.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q80', 'name': 'grand_benchmark_v2',
        'qas_score': qas,
        'total_score': round(total_score, 1),
        'total_max': total_max,
        'total_phases': len(all_results),
        'categories': scores,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q80_benchmark_v2.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q80 completed in %.0fs" % (time.time() - start))


if __name__ == '__main__':
    main()
