# -*- coding: utf-8 -*-
"""
Phase Q9: Is Layer 8 Universally Special? (Training Layer Sweep)

Q8 trained at L8, swept injection: peak at L8.
Q9 asks: if we train at L4, L12, L20, does the peak FOLLOW the training layer?

Experiment:
  For each training layer T in [4, 12, 20]:
    - Train min_vec, max_vec with training at layer T (75 epochs)
    - Sweep injection at all 28 layers, measure interference amplitude
    - Compare: does peak occur at T (follows training) or always at L8?

Result:
  - If peak always at L8 -> L8 is a universal "quantum bottleneck"
  - If peak follows T   -> interference is a training artifact (less interesting)
  - Mixed result        -> L8 has a resonance, other layers have weaker peaks
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_PHI = 37
EPOCHS = 75  # faster than Q8's 150


def train_soul(model, tok, data, device, layer, epochs=75, seed=42):
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


def get_p_min(model, tok, prompt, device, inject_vec, inject_layer, min_tok_id):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[inject_layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[min_tok_id])


def sweep_all_layers(model, tok, prompt, device, min_vec, max_vec,
                      min_tok_id, num_layers, n_phi=N_PHI):
    """For each injection layer, sweep phi and compute interference amplitude."""
    phis = np.linspace(0, 4 * np.pi, n_phi)
    amplitudes = []
    for inject_li in range(num_layers):
        p_vals = []
        for phi in phis:
            vec = np.cos(phi / 2) * min_vec + np.sin(phi / 2) * max_vec
            norm_val = vec.norm()
            if norm_val > 0:
                vec = vec / norm_val * min_vec.norm()
            p_vals.append(get_p_min(model, tok, prompt, device, vec, inject_li, min_tok_id))
        amp = (max(p_vals) - min(p_vals)) / 2.0
        amplitudes.append(amp)
    return amplitudes


def main():
    print("[Q9] Is Layer 8 Universally Special? (Training Layer Sweep)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    num_layers = len(model.model.layers)
    print("  Model: %d layers" % num_layers)

    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"),
    ]
    prompt = "min(7,2)="
    min_tok_id = tok.encode("2")[-1]

    # Training layers to test: L4, L12, L20 (L8 already done in Q8)
    train_layers_to_test = [4, 12, 20]

    all_results = {}
    # Also include Q8 result as baseline
    q8_amps = None
    try:
        with open(os.path.join(RESULTS_DIR, 'phase_q8_multilayer_bell_test.json')) as f:
            import json as _json
            q8 = _json.load(f)
            q8_amps = q8['layer_amplitudes']
            print("  Loaded Q8 (train@L8) amplitudes as baseline")
            all_results[8] = {'amps': q8_amps, 'peak_layer': q8['peak_layer'],
                               'peak_amp': q8['peak_amplitude'],
                               'self_amp': q8_amps[8]}  # add self_amp
    except Exception as e:
        print("  Could not load Q8: %s" % e)

    for train_layer in train_layers_to_test:
        print("\n  === Training at Layer %d (%d epochs) ===" % (train_layer, EPOCHS))
        min_vec = train_soul(model, tok, min_data, DEVICE, layer=train_layer, seed=42, epochs=EPOCHS)
        max_vec = train_soul(model, tok, max_data, DEVICE, layer=train_layer, seed=99, epochs=EPOCHS)
        print("  Sweeping all %d injection layers..." % num_layers)
        amps = sweep_all_layers(model, tok, prompt, DEVICE, min_vec, max_vec,
                                 min_tok_id, num_layers)
        peak_li = int(np.argmax(amps))
        peak_amp = amps[peak_li]
        print("  Train@L%d -> Peak@L%d (amp=%.4f)  [self=%.4f]" % (
            train_layer, peak_li, peak_amp, amps[train_layer]))
        all_results[train_layer] = {
            'amps': [round(a, 6) for a in amps],
            'peak_layer': peak_li,
            'peak_amp': round(peak_amp, 6),
            'self_amp': round(amps[train_layer], 6),
        }

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === PLOT ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    layer_ids = list(range(num_layers))
    colors = {8: '#E91E63', 4: '#2196F3', 12: '#4CAF50', 20: '#FF9800'}
    linestyles = {8: '-', 4: '--', 12: ':', 20: '-.'}

    # Panel 1: Amplitude profiles for all training layers
    ax = axes[0]
    for tl, res in sorted(all_results.items()):
        ax.plot(layer_ids, res['amps'],
                color=colors.get(tl, 'gray'),
                linestyle=linestyles.get(tl, '-'),
                lw=2, label='Train@L%d (peak=L%d, %.3f)' % (
                    tl, res['peak_layer'], res['peak_amp']))
        ax.axvline(res['peak_layer'], color=colors.get(tl, 'gray'),
                   alpha=0.3, linewidth=0.8)
    ax.axvline(8, color='black', linestyle='--', lw=1, alpha=0.5, label='L8 reference')
    ax.set_xlabel('Injection Layer', fontsize=11)
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax.set_title('Bell Test: Does Peak Follow Training Layer?\n'
                 'If peak stays at L8 -> L8 is universal quantum seat',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Summary - self-amp vs cross-amp at L8
    ax = axes[1]
    train_ls = sorted(all_results.keys())
    self_amps = [all_results[tl]['self_amp'] for tl in train_ls]
    l8_amps = [all_results[tl]['amps'][8] for tl in train_ls]
    peak_layers = [all_results[tl]['peak_layer'] for tl in train_ls]

    x = np.arange(len(train_ls))
    w = 0.3
    bars1 = ax.bar(x - w/2, self_amps, w, color=[colors.get(tl, 'gray') for tl in train_ls],
                   label='Self (inject at train layer)', edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x + w/2, l8_amps, w, color=[colors.get(tl, 'gray') for tl in train_ls],
                   label='At L8', edgecolor='black', alpha=0.4, hatch='//')

    for i, (tl, pl) in enumerate(zip(train_ls, peak_layers)):
        ax.text(i, max(self_amps[i], l8_amps[i]) + 0.01,
                'peak@L%d' % pl, ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Train@L%d' % tl for tl in train_ls])
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax.set_title('Self-Injection vs L8-Injection\nAmplitude Comparison',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Phase Q9: Is Layer 8 Universally the Quantum Seat?\n'
        'Training Layer Sweep: Does peak follow training?',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q9_training_layer_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    verdict = []
    for tl, res in sorted(all_results.items()):
        if tl == 8:
            continue
        if res['peak_layer'] == 8:
            verdict.append("L%d->peak@L8 (UNIVERSAL)" % tl)
        elif res['peak_layer'] == tl:
            verdict.append("L%d->peak@L%d (FOLLOWS TRAINING)" % (tl, tl))
        else:
            verdict.append("L%d->peak@L%d (PARTIAL)" % (tl, res['peak_layer']))
    print("\n  VERDICT: %s" % ' | '.join(verdict))

    output = {
        'phase': 'Q9', 'name': 'training_layer_sweep',
        'num_layers': num_layers,
        'n_phi': N_PHI,
        'epochs': EPOCHS,
        'results': {str(tl): v for tl, v in all_results.items()},
        'verdict': verdict,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q9_training_layer_sweep.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q9 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
