# -*- coding: utf-8 -*-
import json, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r'c:\Users\kyjan\研究\Semantic-Qubit\results'
FIGURES_DIR = r'c:\Users\kyjan\研究\Semantic-Qubit\figures'

with open(os.path.join(RESULTS_DIR, 'phase_q8_multilayer_bell_test.json')) as f:
    q8 = json.load(f)

# Results from Q9 log output (computation completed, only plot crashed)
all_results = {
    8:  {'peak_layer': 8,  'peak_amp': 0.4996, 'self_amp': q8['layer_amplitudes'][8]},
    4:  {'peak_layer': 4,  'peak_amp': 0.4981, 'self_amp': 0.4981},
    12: {'peak_layer': 12, 'peak_amp': 0.4998, 'self_amp': 0.4998},
    20: {'peak_layer': 20, 'peak_amp': 0.4888, 'self_amp': 0.4888},
}
train_ls = sorted(all_results.keys())
colors = {8: '#E91E63', 4: '#2196F3', 12: '#4CAF50', 20: '#FF9800'}
num_layers = 28

print('[Q9] VERDICT:')
for tl in [4, 12, 20]:
    res = all_results[tl]
    follow = (res['peak_layer'] == tl)
    tag = 'FOLLOWS TRAINING' if follow else 'L8 UNIVERSAL'
    print('  Train@L%d -> Peak@L%d (amp=%.4f)  %s' % (tl, res['peak_layer'], res['peak_amp'], tag))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Q8 profile with horizontal lines for other layers
ax = axes[0]
ax.bar(range(num_layers), q8['layer_amplitudes'],
       color=['#E91E63' if i == 8 else '#90CAF9' for i in range(num_layers)],
       edgecolor='none', alpha=0.85)
ax.axhline(0.4981, color='#2196F3', linestyle='--', lw=2, label='Train@L4 self=0.498')
ax.axhline(0.4998, color='#4CAF50', linestyle=':', lw=2, label='Train@L12 self=0.500')
ax.axhline(0.4888, color='#FF9800', linestyle='-.', lw=2, label='Train@L20 self=0.489')
ax.set_xlabel('Injection Layer', fontsize=11)
ax.set_ylabel('Interference Amplitude', fontsize=11)
ax.set_title('Q8 (Train@L8) Amplitude Profile\n+ Self-amp for L4/L12/L20', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Panel 2: Self-amp bar chart
ax = axes[1]
x = np.arange(len(train_ls))
self_amps = [all_results[tl]['self_amp'] for tl in train_ls]
peak_amps = [all_results[tl]['peak_amp'] for tl in train_ls]
w = 0.35
ax.bar(x - w/2, self_amps, w, color=[colors[tl] for tl in train_ls],
       label='Self-injection', edgecolor='black', alpha=0.9)
ax.bar(x + w/2, peak_amps, w, color=[colors[tl] for tl in train_ls],
       label='Overall peak', edgecolor='black', alpha=0.5, hatch='//')
for i, tl in enumerate(train_ls):
    ax.text(i, max(self_amps[i], peak_amps[i]) + 0.005,
            'peak@L%d' % all_results[tl]['peak_layer'],
            ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Train@L%d' % tl for tl in train_ls], fontsize=11)
ax.set_ylabel('Interference Amplitude', fontsize=11)
ax.set_ylim(0, 0.6)
ax.set_title('Peak FOLLOWS Training Layer\nEvery layer equally quantum (~0.49-0.50)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.axhline(0.5, color='red', linestyle='--', lw=1.5, alpha=0.5)

plt.suptitle('Phase Q9: Is Layer 8 Universally Special?\nResult: NO - All middle layers have equal quantum capacity!',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'phase_q9_training_layer_sweep.png'),
            dpi=150, bbox_inches='tight')
plt.close()

output = {
    'phase': 'Q9', 'name': 'training_layer_sweep',
    'num_layers': num_layers, 'n_phi': 37, 'epochs': 75,
    'results': {str(tl): all_results[tl] for tl in train_ls},
    'verdict': 'FOLLOWS_TRAINING',
    'conclusion': 'L8 is NOT a unique quantum bottleneck. Interference is a property of the trained vector, not the layer. All middle layers have similar quantum capacity (~0.49-0.50).',
}
with open(os.path.join(RESULTS_DIR, 'phase_q9_training_layer_sweep.json'), 'w') as f:
    json.dump(output, f, indent=2)
print('Q9 saved OK!')
