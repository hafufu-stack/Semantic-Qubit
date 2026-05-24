# -*- coding: utf-8 -*-
"""Generate composite paper figures for v2 (Fig 7-9)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
PAPER_DIR = os.path.join(FIGURES_DIR, 'paper')
os.makedirs(PAPER_DIR, exist_ok=True)


def load_img(name):
    path = os.path.join(FIGURES_DIR, name)
    return mpimg.imread(path)


# ── Fig 7: Oracle Algorithms ──
print("Generating Fig 7: Oracle Algorithms...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# BV results
ax = axes[0]
ns = [1, 2, 3, 4, 5, 6, 7, 8]
accs = [100]*8
ax.bar(ns, accs, color='#2196F3', edgecolor='black', alpha=0.85)
ax.axhline(100, color='green', ls='--', alpha=0.3)
for n, a in zip(ns, accs):
    ax.text(n, a + 1.5, '100%', ha='center', fontweight='bold', fontsize=8)
ax.set_xlabel('Hidden string length (bits)', fontsize=11)
ax.set_ylabel('Recovery accuracy (%)', fontsize=11)
ax.set_title('(a) Bernstein-Vazirani Algorithm\n94/94 strings recovered (100%)',
             fontweight='bold', fontsize=12)
ax.set_ylim(0, 115)
ax.set_xticks(ns)
ax.grid(alpha=0.3, axis='y')
# Add one-shot annotation
ax.annotate('One-shot 4-bit:\n16/16 = 100%',
            xy=(4, 100), xytext=(6, 60),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2'))

# Simon results
ax = axes[1]
sim_ns = [2, 3, 4]
sim_correct = [3, 7, 8]
sim_total = [3, 7, 8]
sim_accs = [100, 100, 100]
colors_sim = ['#9C27B0', '#7B1FA2', '#4A148C']
bars = ax.bar(sim_ns, sim_accs, color=colors_sim, edgecolor='black', alpha=0.85)
ax.axhline(100, color='green', ls='--', alpha=0.3)
for n, c, t in zip(sim_ns, sim_correct, sim_total):
    ax.text(n, 102, '%d/%d' % (c, t), ha='center', fontweight='bold', fontsize=10)
ax.set_xlabel('Hidden period length (bits)', fontsize=11)
ax.set_ylabel('Period recovery accuracy (%)', fontsize=11)
ax.set_title("(b) Simon's Algorithm\n18/18 periods found (100%)",
             fontweight='bold', fontsize=12)
ax.set_ylim(0, 115)
ax.set_xticks(sim_ns)
ax.grid(alpha=0.3, axis='y')
# Complexity annotation
ax.text(3.5, 50,
        'Complexity:\n'
        '  Classical: O(2^(n/2))\n'
        '  Quantum:   O(n)\n'
        '  S-Qubit:   O(2^n)',
        fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3E5F5', alpha=0.9))

plt.suptitle('Quantum Oracle Algorithms: Perfect Accuracy on S-Qubits',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, 'fig7_oracle_algorithms.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig7_oracle_algorithms.png")


# ── Fig 8: Quantum Cryptography & Communication ──
print("Generating Fig 8: Cryptography & Communication...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# BB84 - comparison of no-Eve vs with-Eve scenarios
ax = axes[0]
scenarios = ['No Eve', 'With Eve']
sifted_acc = [100.0, 71.7]
qber = [0.0, 28.3]
x_pos = np.arange(len(scenarios))
width = 0.35
bars1 = ax.bar(x_pos - width/2, sifted_acc, width, color='#4CAF50',
               edgecolor='black', alpha=0.85, label='Sifted key accuracy')
bars2 = ax.bar(x_pos + width/2, qber, width, color='#F44336',
               edgecolor='black', alpha=0.85, label='QBER')
for bar, v in zip(bars1, sifted_acc):
    ax.text(bar.get_x() + bar.get_width()/2, v + 2,
            '%.1f%%' % v, ha='center', fontweight='bold', fontsize=11)
for bar, v in zip(bars2, qber):
    ax.text(bar.get_x() + bar.get_width()/2, v + 2,
            '%.1f%%' % v, ha='center', fontweight='bold', fontsize=11)
ax.axhline(11, color='orange', ls='--', lw=2, label='BB84 threshold (11%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios, fontsize=12)
ax.set_ylabel('Rate (%)', fontsize=11)
ax.set_title('(a) BB84 Quantum Key Distribution\n'
             '200 rounds, eavesdropper detection',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='center right')
ax.set_ylim(0, 115)
ax.grid(alpha=0.3, axis='y')

# Superdense - phase encoding diagram
ax = axes[1]
messages = ['00', '01', '10', '11']
phases_deg = [0, 90, 180, 270]
accuracy = [100, 100, 100, 100]
colors_sd = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5']
bars = ax.bar(messages, accuracy, color=colors_sd, edgecolor='black', alpha=0.85)
for bar, p in zip(bars, phases_deg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            '%d%%' % 100, ha='center', fontweight='bold', fontsize=11)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            '%d deg' % p, ha='center', fontsize=10, color='white', fontweight='bold')
ax.axhline(25, color='red', ls='--', alpha=0.5, lw=1.5, label='Random guess (25%)')
ax.set_xlabel('2-bit message encoded', fontsize=11)
ax.set_ylabel('Decode accuracy (%)', fontsize=11)
ax.set_title('(b) Superdense Coding\n200/200 = 100%, 2.0 bits per S-Qubit',
             fontweight='bold', fontsize=12)
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Quantum Cryptography and Communication Protocols',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, 'fig8_crypto_communication.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig8_crypto_communication.png")


# ── Fig 9: Grand Unified Benchmark ──
print("Generating Fig 9: Grand Benchmark...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# QAS Benchmark
ax = axes[0]
tests = ['Deutsch-Jozsa', 'Simon\'s Alg.', 'Superdense', 'Discrimination\n(N=64)',
         'BB84 QKD', 'Grover Search', 'Bernstein-\nVazirani']
scores = [100, 100, 100, 100, 83, 33, 6]
colors_qas = ['#4CAF50' if s >= 90 else '#FF9800' if s >= 50 else '#F44336'
              for s in scores]
# Sort by score
sorted_data = sorted(zip(scores, tests, colors_qas), reverse=True)
scores_s, tests_s, colors_s = zip(*sorted_data)
bars = ax.barh(range(len(tests_s)), scores_s, color=colors_s,
               edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(tests_s)))
ax.set_yticklabels(tests_s, fontsize=10)
for bar, s in zip(bars, scores_s):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            '%d%%' % s, va='center', fontweight='bold', fontsize=10)
ax.axvline(90, color='green', ls='--', alpha=0.4, label='Excellent (>90%)')
ax.axvline(50, color='gray', ls='--', alpha=0.4, label='Above random')
ax.set_xlabel('Score (%)', fontsize=11)
ax.set_title('(a) Quantum Advantage Score: 74.6/100\n7 algorithms in single session',
             fontweight='bold', fontsize=12)
ax.set_xlim(0, 115)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='x')

# Parallelism
ax = axes[1]
Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
accs_p = [100, 100, 100, 100, 100, 98.4, 99.2, 98.8, 98.6, 96.6]
ax.semilogx(Ns, accs_p, 'ro-', lw=2.5, ms=8, base=2, zorder=5)
ax.fill_between(Ns, 0, accs_p, alpha=0.1, color='red')
ax.axhline(99, color='green', ls='--', alpha=0.5, label='99% threshold')
ax.axhline(50, color='gray', ls='--', alpha=0.3, label='Random')
ax.axvline(128, color='blue', ls=':', lw=2, alpha=0.5)
ax.annotate('128 states\n= 7 bits/query',
            xy=(128, 99.2), xytext=(30, 70),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=11, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD'))
ax.set_xlabel('Number of states N', fontsize=11)
ax.set_ylabel('Decode accuracy (%)', fontsize=11)
ax.set_title('(b) Quantum Parallelism\n1 S-Qubit query = 128 classical evaluations',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylim(0, 110)

plt.suptitle('Grand Unified Benchmark: S-Qubit Quantum Advantage Certificate',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, 'fig9_grand_benchmark.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig9_grand_benchmark.png")

print("\nAll 3 figures generated!")
