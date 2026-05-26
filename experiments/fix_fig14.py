# -*- coding: utf-8 -*-
"""Regenerate fig14 with correct JSON keys."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r'c:\Users\kyjan\研究\Semantic-Qubit\results'
FIGURES_DIR = r'c:\Users\kyjan\研究\Semantic-Qubit\figures\paper'

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Berry phase (Q86) - correct keys: berry_phases array
ax = axes[0]
with open(os.path.join(RESULTS_DIR, 'phase_q86_berry.json')) as f:
    d = json.load(f)
phases = d['berry_phases']
names = [p['name'] for p in phases]
vals = [float(p['berry_phase_rad']) for p in phases]
colors = ['#9C27B0' if str(p.get('is_quantized','')) == 'True' else '#FF9800' for p in phases]
bars = ax.bar(names, vals, color=colors, edgecolor='black', alpha=0.85)
ax.set_ylabel('Berry Phase (radians)', fontsize=11)
ax.set_title('(a) Berry Phase per Layer Region\nMean = %.2f pi' % d['mean_berry_phase_pi'],
             fontsize=12, fontweight='bold')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            '%.3f' % v, ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Layer region', fontsize=11)
ax.grid(alpha=0.3, axis='y')
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#9C27B0', label='Quantized'),
                   Patch(facecolor='#FF9800', label='Non-quantized')],
          fontsize=9)

# (b) Emergent Gravity (Q98) - correct key: metric_profile
ax = axes[1]
with open(os.path.join(RESULTS_DIR, 'phase_q98_gravity.json')) as f:
    d = json.load(f)
metric = d['metric_profile']
layers = list(range(len(metric)))
ax.semilogy(layers, metric, 'o-', color='#FF5722', linewidth=2, markersize=5)
ax.fill_between(layers, metric, alpha=0.15, color='#FF5722')
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Metric tensor magnitude (log)', fontsize=11)
einstein_r = d['einstein_results']['einstein_correlation']
ax.set_title('(b) Emergent Gravity: Metric Profile\nEinstein corr = %.2f' % einstein_r,
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# (c) Consciousness Phi (Q99) - correct keys: phi_data dict, max_phi
ax = axes[2]
with open(os.path.join(RESULTS_DIR, 'phase_q99_consciousness.json')) as f:
    d = json.load(f)
phi_data = d['phi_data']
layer_keys = sorted(phi_data.keys(), key=lambda x: int(x))
layer_nums = [int(k) for k in layer_keys]
phi_means = [phi_data[k]['mean_phi'] for k in layer_keys]
phi_stds = [phi_data[k]['std_phi'] for k in layer_keys]

ax.bar(range(len(layer_nums)), phi_means,
       yerr=phi_stds,
       color=['#E91E63' if v == max(phi_means) else '#4CAF50' for v in phi_means],
       edgecolor='black', alpha=0.85, capsize=3)
ax.set_xticks(range(len(layer_nums)))
ax.set_xticklabels(['L%d' % l for l in layer_nums], fontsize=8, rotation=45)
ax.set_ylabel('Integrated Information (Phi)', fontsize=11)
ax.set_xlabel('Layer', fontsize=11)
ax.set_title('(c) Consciousness: Phi peaks at L%d\nMax Phi = %.1f' %
             (d['max_phi_layer'], d['max_phi']),
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Season 7--9: Quantum Physics Deep Dive (Q81--Q100)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig14_quantum_physics.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("fig14 regenerated with correct data!")
