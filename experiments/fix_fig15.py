# -*- coding: utf-8 -*-
"""Regenerate fig15 with correct JSON keys for Q102."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r'c:\Users\kyjan\研究\Semantic-Qubit\results'
FIGURES_DIR = r'c:\Users\kyjan\研究\Semantic-Qubit\figures\paper'

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Cross-architecture universality (Q101) - already works
ax = axes[0]
with open(os.path.join(RESULTS_DIR, 'phase_q101_universality.json')) as f:
    d = json.load(f)
models = d.get('models', [])
names = [m['model_name'] for m in models]
scores = [m['n_confirmed'] for m in models]
colors = ['#4CAF50' if s >= 4 else '#FF9800' if s >= 3 else '#F44336' for s in scores]
ax.barh(range(len(names)), scores, color=colors, edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Confirmed quantum properties (out of 5)', fontsize=10)
ax.set_xlim(0, 5.5)
for i, s in enumerate(scores):
    ax.text(s + 0.1, i, '%d/5' % s, va='center', fontsize=10, fontweight='bold')
ax.set_title('(a) Cross-Architecture Universality\n6 models tested',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# (b) Hippocampal bridge - MD/LD phase alignment (Q102) - FIXED
ax = axes[1]
with open(os.path.join(RESULTS_DIR, 'phase_q102_hippocampal.json')) as f:
    d = json.load(f)
phase_data = d['phase_data']
layers = [p['layer'] for p in phase_data]
cosines = [p['cosine'] for p in phase_data]
firing = [p['dg_firing'] for p in phase_data]

bar_colors = ['#4CAF50' if f else '#2196F3' for f in firing]
bars = ax.bar(layers, cosines, color=bar_colors, edgecolor='black', alpha=0.85, width=0.8)
ax.axhline(0.5, color='red', ls='--', linewidth=2, alpha=0.7, label='Firing threshold (0.5)')

# Mark the first firing layer
first_fire = next(i for i, f in enumerate(firing) if f)
ax.annotate('First firing!\nL%d' % layers[first_fire],
            xy=(layers[first_fire], cosines[first_fire]),
            xytext=(layers[first_fire] - 6, cosines[first_fire] + 0.15),
            fontsize=10, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('MD-LD cosine similarity', fontsize=11)
ax.set_title('(b) Hippocampal Bridge (Q102)\nPhase alignment fires at L18+',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3, axis='y')

# Add legend for bar colors
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#4CAF50', label='Firing (cos > 0.5)'),
    Patch(facecolor='#2196F3', label='Not firing'),
    plt.Line2D([0],[0], color='red', ls='--', lw=2, label='Threshold')
], fontsize=8, loc='upper left')

# (c) Scaling laws (Q103) - already works
ax = axes[2]
with open(os.path.join(RESULTS_DIR, 'phase_q103_scaling.json')) as f:
    d = json.load(f)
exponents = d.get('scaling_exponents', {})
props = list(exponents.keys())
alphas = [exponents[p]['alpha'] for p in props]
labels_map = {'phi': 'Phi', 'entanglement': 'Entangle', 'interference': 'Interfere',
              'n_confirmed': 'Total'}
display = [labels_map.get(p, p) for p in props]
colors = ['#4CAF50' if a > 0 else '#2196F3' for a in alphas]
ax.barh(range(len(display)), alphas, color=colors, edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(display)))
ax.set_yticklabels(display, fontsize=10)
ax.set_xlabel('Scaling exponent (alpha)', fontsize=10)
ax.axvline(0, color='black', linewidth=0.5)
for i, a in enumerate(alphas):
    ax.text(a + 0.02 if a >= 0 else a - 0.15, i, '%.2f' % a,
            va='center', fontsize=10, fontweight='bold')
ax.set_title('(c) Scaling Laws (Q103)\nTotal properties: alpha ~= 0',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

plt.suptitle('Season 10: Universality, Origins, and Scaling Laws (Q101--Q110)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig15_universality_scaling.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("fig15 regenerated with correct hippocampal data!")
