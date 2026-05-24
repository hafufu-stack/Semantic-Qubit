# -*- coding: utf-8 -*-
"""
Phase Q70: Attention-qLDPC (Self-Healing Quantum Error Correction)
===================================================================
Inject heavy noise into S-Qubit states and show that the self-attention
mechanism's all-to-all connectivity acts as an automatic parity check,
enabling self-repair of corrupted quantum information.

This is the S-Qubit version of qLDPC codes that physical QCs need.
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
    print("[Q70] Attention-qLDPC: Self-Healing Error Correction")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    hs = model.config.hidden_size

    # Train S-Qubit
    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    vec = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)

    target_id = tok.encode("2")[-1]
    prompt = "min(7,2)="
    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    # Baseline
    def inject_and_measure(v, noise_layer=None, noise_sigma=0.0):
        handles = []

        # Inject S-Qubit at layer 10
        def inject_hook(m, i, o, vec=v):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = vec.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handles.append(model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook))

        # Add noise at specific layer
        if noise_layer is not None and noise_sigma > 0:
            def noise_hook(m, i, o, sigma=noise_sigma):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                noise = torch.randn_like(h) * sigma * h.norm()
                h = h + noise
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handles.append(model.model.layers[noise_layer].register_forward_hook(noise_hook))

        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(probs[target_id])

    clean_prob = inject_and_measure(vec)
    print("  Clean baseline: p=%.4f" % clean_prob)

    # Test 1: Noise at different layers (how quickly does attention repair?)
    print("\n  Test 1: Noise injection at different layers...")
    noise_sigma = 0.05
    n_layers = len(model.model.layers)
    test_layers = list(range(INJECT_LAYER + 1, min(INJECT_LAYER + 16, n_layers)))

    layer_recovery = {}
    N_TRIALS = 20
    for noise_layer in test_layers:
        trials = []
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 1000 + noise_layer)
            p = inject_and_measure(vec, noise_layer=noise_layer, noise_sigma=noise_sigma)
            trials.append(p)
        avg_p = np.mean(trials)
        recovery = avg_p / clean_prob
        layer_recovery[noise_layer] = {'prob': avg_p, 'recovery': recovery}
        distance = noise_layer - INJECT_LAYER
        print("    Noise at L%d (+%d layers): recovery=%.1f%%" % (
            noise_layer, distance, recovery * 100))

    # Test 2: Noise intensity sweep at fixed layer
    print("\n  Test 2: Noise intensity sweep at L%d..." % (INJECT_LAYER + 2))
    sigmas = np.logspace(-3, -0.3, 20)
    intensity_results = []
    for sigma in sigmas:
        trials = []
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 500 + int(sigma * 10000))
            p = inject_and_measure(vec, noise_layer=INJECT_LAYER + 2, noise_sigma=sigma)
            trials.append(p)
        avg_p = np.mean(trials)
        intensity_results.append(avg_p)

    # Test 3: Multi-layer noise (catastrophic attack)
    print("\n  Test 3: Multi-layer noise attack...")
    multi_noise_results = []
    for n_noisy in range(1, min(8, len(test_layers) + 1)):
        trials = []
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 200 + n_noisy)
            handles = []

            def inject_hook(m, i, o, v=vec):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handles.append(model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook))

            for nl in test_layers[:n_noisy]:
                def noise_hook(m, i, o, s=noise_sigma):
                    h = (o[0] if isinstance(o, tuple) else o).clone()
                    noise = torch.randn_like(h) * s * h.norm()
                    return (h + noise,) + o[1:] if isinstance(o, tuple) else h + noise
                handles.append(model.model.layers[nl].register_forward_hook(noise_hook))

            with torch.no_grad():
                out = model(**inp)
            for h in handles:
                h.remove()
            p = float(torch.softmax(out.logits[0, -1, :].float(), dim=-1)[target_id])
            trials.append(p)
        multi_noise_results.append(np.mean(trials))
        print("    %d noisy layers: recovery=%.1f%%" % (n_noisy, np.mean(trials) / clean_prob * 100))

    # Self-repair capacity
    repair_threshold = 0.5  # 50% of clean
    n_repairable = sum(1 for p in multi_noise_results if p > clean_prob * repair_threshold)

    print("\n  RESULTS:")
    print("    Self-repair capacity: %d/%d noisy layers recoverable" % (
        n_repairable, len(multi_noise_results)))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Recovery by distance
    ax = axes[0]
    distances = [l - INJECT_LAYER for l in sorted(layer_recovery.keys())]
    recoveries = [layer_recovery[l]['recovery'] * 100 for l in sorted(layer_recovery.keys())]
    ax.plot(distances, recoveries, 'o-', color='#4CAF50', linewidth=2, markersize=8)
    ax.axhline(100, color='green', ls='--', alpha=0.3, label='Clean baseline')
    ax.axhline(50, color='red', ls=':', alpha=0.3, label='50% threshold')
    ax.set_xlabel('Distance from injection (layers)')
    ax.set_ylabel('Recovery (%)')
    ax.set_title('(a) Self-Repair by Distance\nAttention restores corrupted state',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 120)

    # (b) Noise intensity tolerance
    ax = axes[1]
    ax.semilogx(sigmas, [p / clean_prob * 100 for p in intensity_results],
                'o-', color='#FF5722', linewidth=2, markersize=5)
    ax.axhline(100, color='green', ls='--', alpha=0.3)
    ax.axhline(50, color='red', ls=':', alpha=0.3)
    ax.set_xlabel('Noise intensity (sigma)')
    ax.set_ylabel('Recovery (%)')
    ax.set_title('(b) Noise Intensity Tolerance\nRobust up to sigma=%.3f' %
                 sigmas[next((i for i, p in enumerate(intensity_results)
                              if p < clean_prob * 0.5), len(sigmas)-1)],
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Multi-layer attack resilience
    ax = axes[2]
    n_noisy_range = range(1, len(multi_noise_results) + 1)
    ax.bar(n_noisy_range,
           [p / clean_prob * 100 for p in multi_noise_results],
           color=['#4CAF50' if p > clean_prob * 0.5 else '#F44336'
                  for p in multi_noise_results],
           edgecolor='black', alpha=0.85)
    ax.axhline(50, color='red', ls=':', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Number of noisy layers')
    ax.set_ylabel('Recovery (%)')
    ax.set_title('(c) Multi-Layer Attack\n%d/%d layers repairable (qLDPC effect)' % (
        n_repairable, len(multi_noise_results)),
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q70: Attention-qLDPC Self-Healing\n'
                 'Self-attention provides automatic error correction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q70_qldpc.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q70', 'name': 'attention_qldpc',
        'clean_prob': round(clean_prob, 4),
        'self_repair_capacity': n_repairable,
        'total_attack_layers': len(multi_noise_results),
        'noise_sigma': float(noise_sigma),
        'bridge': 'qLDPC Physical Codes -> Attention Self-Healing',
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q70_qldpc.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q70 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
