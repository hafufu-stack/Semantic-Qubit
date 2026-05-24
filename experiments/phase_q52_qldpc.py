# -*- coding: utf-8 -*-
"""
Phase Q52: Attention-qLDPC (Software-Defined Quantum Error Correction)

Physical quantum computers need ~1000 physical qubits per logical qubit
for error correction. We test if the Attention mechanism acts as a
natural error-correcting code (qLDPC analogue).

Experiment:
  1. Inject soul vector into one position (target)
  2. Inject CORRECT soul vectors into neighboring positions (anchors)
  3. Add heavy noise to the target
  4. Measure: does Attention "repair" the target using anchor context?
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
    print("[Q52] Attention-qLDPC Error Correction")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)

    prompt = "min(7,2)="
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    seq_len = inp['input_ids'].shape[1]

    # Reference E (no noise)
    def measure_with_noise(noise_sigma, n_anchors, inject_layer):
        """Inject v0 + noise at last position, with n_anchors clean v0 copies."""
        def hook(m, i, o, v0=v0, sigma=noise_sigma, anchors=n_anchors):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            # Add noise to the target (last position)
            noisy_v = v0 + torch.randn_like(v0) * sigma
            h[0, -1, :] = noisy_v.to(h.dtype)
            # Inject clean anchors at prior positions
            for a in range(1, min(anchors + 1, h.shape[1])):
                pos = -1 - a
                if -pos <= h.shape[1]:
                    h[0, pos, :] = v0.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[inject_layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(probs[min_tok]) - float(probs[max_tok])

    # Clean reference
    E_clean = measure_with_noise(0, 0, INJECT_LAYER)
    print("  Clean reference E: %.4f" % E_clean)

    # Test: noise levels x anchor counts
    noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    anchor_counts = [0, 1, 2, 3]
    N_trials = 5

    results = {}
    print("\n  Testing error correction...")
    for n_anchors in anchor_counts:
        results[n_anchors] = []
        for sigma in noise_levels:
            Es = []
            for _ in range(N_trials):
                E = measure_with_noise(sigma, n_anchors, INJECT_LAYER)
                Es.append(E)
            fidelity = 1 - abs(np.mean(Es) - E_clean) / (abs(E_clean) + 1e-10)
            fidelity = max(0, min(1, fidelity))
            results[n_anchors].append({
                'sigma': sigma,
                'E_mean': round(float(np.mean(Es)), 4),
                'E_std': round(float(np.std(Es)), 4),
                'fidelity': round(float(fidelity), 4),
            })
            print("    anchors=%d, sigma=%.2f -> E=%.4f +/- %.4f, fidelity=%.3f" % (
                n_anchors, sigma, np.mean(Es), np.std(Es), fidelity))

    # Compute correction gain
    print("\n  ERROR CORRECTION SUMMARY:")
    for sigma in [0.1, 0.5, 1.0]:
        idx = noise_levels.index(sigma)
        fid_0 = results[0][idx]['fidelity']
        fid_3 = results[3][idx]['fidelity']
        gain = fid_3 / (fid_0 + 1e-10)
        print("    sigma=%.1f: 0 anchors=%.3f, 3 anchors=%.3f, gain=%.2fx" % (
            sigma, fid_0, fid_3, gain))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for n_a in anchor_counts:
        fids = [r['fidelity'] for r in results[n_a]]
        ax.plot(noise_levels, fids, 'o-', lw=2, ms=6,
                label='%d anchors' % n_a)
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('Fidelity')
    ax.set_title('(a) Error Correction via Attention\nFidelity vs noise level',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xscale('log')

    ax = axes[1]
    sigmas_plot = [0.05, 0.1, 0.2, 0.5]
    x_pos = np.arange(len(sigmas_plot))
    width = 0.2
    for i, n_a in enumerate(anchor_counts):
        fids_plot = []
        for sigma in sigmas_plot:
            idx = noise_levels.index(sigma)
            fids_plot.append(results[n_a][idx]['fidelity'])
        ax.bar(x_pos + i * width, fids_plot, width, label='%d anchors' % n_a,
               alpha=0.85, edgecolor='black')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(['sigma=%.2f' % s for s in sigmas_plot])
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) Anchor Effect\nMore anchors = better correction',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q52: Attention-qLDPC Error Correction\n'
                 'Self-Attention as software-defined quantum error correction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q52_qlpdc.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q52', 'name': 'attention_qldpc',
        'E_clean': round(float(E_clean), 4),
        'results': {str(k): v for k, v in results.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q52_qldpc.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q52 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
