# -*- coding: utf-8 -*-
"""
Phase Q69: Quantum Theta Rhythm (Master's Thesis Full Analogy)
================================================================
BRIDGE: Master's Thesis LD/MD Phase-Frequency Coupling <-> S-Qubit

Master's thesis discovered: LD (non-spatial) input at specific
phase/frequency matching MD (spatial) burst input -> EPSP summation
-> enhanced pattern separation (firing).

S-Qubit analogy: Oscillate S-Qubit phase across layers (like theta
rhythm) and find the resonant injection frequency that maximizes
task performance.
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
INJECT_LAYER = 10
EPOCHS = 100


def train_soul(model, tok, data, device, layer, epochs=EPOCHS, seed=42):
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
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def main():
    print("[Q69] Quantum Theta Rhythm")
    print("  BRIDGE: Master's Thesis LD/MD Phase Coupling")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.model.layers)
    hs = model.config.hidden_size

    # Train two orthogonal S-Qubits (|0> and |1>)
    v0 = train_soul(model, tok,
                    [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")],
                    DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok,
                    [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")],
                    DEVICE, INJECT_LAYER, EPOCHS, 99)

    target_0 = tok.encode("2")[-1]  # min answer
    prompt = "min(7,2)="
    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Experiment: Multi-layer injection with phase oscillation
    # Like theta rhythm: phase rotates as we go deeper in layers
    # theta(layer) = 2*pi*f * (layer - inject_layer) / n_layers
    
    # Test different "theta frequencies"
    frequencies = np.linspace(0, 3.0, 30)  # cycles across the network
    inject_range = range(INJECT_LAYER, min(INJECT_LAYER + 12, n_layers))
    n_inject_layers = len(list(inject_range))

    print("  Testing %d theta frequencies across %d injection layers..." % (
        len(frequencies), n_inject_layers))

    freq_results = []

    for freq in frequencies:
        handles = []

        for layer_idx in inject_range:
            depth = layer_idx - INJECT_LAYER
            theta = 2 * np.pi * freq * depth / n_inject_layers
            # Phase-modulated superposition
            v_theta = np.cos(theta) * v0 + np.sin(theta) * v1

            def make_hook(v, amp=0.3):
                def hook(m, i, o, vec=v, amplitude=amp):
                    h = (o[0] if isinstance(o, tuple) else o).clone()
                    # Mix with existing activation (not replace)
                    h[0, -1, :] = (1 - amplitude) * h[0, -1, :] + amplitude * vec.to(h.dtype)
                    return (h,) + o[1:] if isinstance(o, tuple) else h
                return hook

            handle = model.model.layers[layer_idx].register_forward_hook(make_hook(v_theta))
            handles.append(handle)

        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()

        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_correct = float(probs[target_0])

        # Compute entropy
        top_probs = probs.topk(100).values
        top_probs = top_probs[top_probs > 1e-10]
        entropy = -float(torch.sum(top_probs * torch.log2(top_probs)))

        freq_results.append({
            'freq': float(freq),
            'p_correct': p_correct,
            'entropy': entropy,
        })

    # Single-layer injection baseline
    def hook_single(m, i, o, v=v0):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook_single)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    baseline_prob = float(torch.softmax(out.logits[0, -1, :].float(), dim=-1)[target_0])

    # Find resonant frequency
    p_values = [r['p_correct'] for r in freq_results]
    best_idx = np.argmax(p_values)
    resonant_freq = freq_results[best_idx]['freq']
    peak_prob = freq_results[best_idx]['p_correct']
    resonance_gain = peak_prob / (baseline_prob + 1e-10)

    print("\n  RESULTS:")
    print("    Baseline (single inject): p=%.4f" % baseline_prob)
    print("    Resonant frequency: %.2f cycles" % resonant_freq)
    print("    Peak probability: %.4f" % peak_prob)
    print("    Resonance gain: %.2fx" % resonance_gain)
    print("    -> Like theta rhythm: specific frequency maximizes pattern separation!")

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Frequency sweep
    ax = axes[0]
    freqs = [r['freq'] for r in freq_results]
    probs_plot = [r['p_correct'] for r in freq_results]
    ax.plot(freqs, probs_plot, 'o-', color='#FF5722', linewidth=2, markersize=5)
    ax.axhline(baseline_prob, color='green', ls='--', alpha=0.5,
               label='Single injection (%.3f)' % baseline_prob)
    ax.axvline(resonant_freq, color='blue', ls=':', alpha=0.5,
               label='Resonant f=%.2f' % resonant_freq)
    ax.set_xlabel('Theta frequency (cycles across layers)')
    ax.set_ylabel('P(correct)')
    ax.set_title('(a) Theta Rhythm Frequency Sweep\n'
                 'Peak at f=%.2f (%.1fx gain)' % (resonant_freq, resonance_gain),
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Entropy vs frequency
    ax = axes[1]
    entropies = [r['entropy'] for r in freq_results]
    ax.plot(freqs, entropies, 'o-', color='#9C27B0', linewidth=2, markersize=5)
    ax.axvline(resonant_freq, color='blue', ls=':', alpha=0.5)
    ax.set_xlabel('Theta frequency (cycles)')
    ax.set_ylabel('Decision entropy (bits)')
    ax.set_title('(b) Entropy Landscape\nLow entropy = high confidence',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Brain theta analogy
    ax = axes[2]
    # Visualize phase modulation pattern at resonant frequency
    layers_plot = list(inject_range)
    phases = [2 * np.pi * resonant_freq * (l - INJECT_LAYER) / n_inject_layers
              for l in layers_plot]
    cos_phases = [np.cos(p) for p in phases]
    sin_phases = [np.sin(p) for p in phases]
    ax.plot(layers_plot, cos_phases, 'o-', color='#2196F3', linewidth=2,
            label='cos(theta) -> |0> weight')
    ax.plot(layers_plot, sin_phases, 's-', color='#FF5722', linewidth=2,
            label='sin(theta) -> |1> weight')
    ax.fill_between(layers_plot, -1, 1, alpha=0.05, color='gray')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Phase weight')
    ax.set_title('(c) Resonant Phase Pattern\n'
                 'Like LD/MD burst coupling in hippocampus',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Phase Q69: Quantum Theta Rhythm\n"
                 "Master's thesis LD/MD coupling recreated in LLM layers",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q69_theta.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q69', 'name': 'quantum_theta_rhythm',
        'baseline_prob': round(float(baseline_prob), 4),
        'resonant_freq': round(float(resonant_freq), 4),
        'peak_prob': round(float(peak_prob), 4),
        'resonance_gain': round(float(resonance_gain), 2),
        'n_inject_layers': n_inject_layers,
        'bridge': "Master's Thesis (LD/MD coupling) -> Quantum Theta Rhythm",
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q69_theta.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q69 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
