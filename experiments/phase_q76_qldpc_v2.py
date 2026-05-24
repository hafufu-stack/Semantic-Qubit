# -*- coding: utf-8 -*-
"""
Phase Q76: Attention-qLDPC v2 (Additive Noise Self-Repair)
============================================================
Q70 used multiplicative noise (sigma * norm) which was too aggressive.
This version uses careful additive noise at the embedding level
to test self-attention's error correction capacity more precisely.

Also tests: if we corrupt SOME dimensions of the S-Qubit itself,
can the transformer's attention heads + residual connections
recover the information?
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
    print("[Q76] Attention-qLDPC v2: Dimension Corruption Self-Repair")
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
    def inject_and_measure(v):
        def hook(m, i, o, vec=v):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = vec.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(probs[target_id])

    clean_prob = inject_and_measure(vec)
    print("  Clean baseline: p=%.4f" % clean_prob)

    # Test 1: Dimension zeroing (erase parts of S-Qubit)
    print("\n  Test 1: Dimension zeroing (erase dimensions)...")
    zero_fractions = np.linspace(0, 0.99, 30)
    zero_results = []
    N_TRIALS = 10

    for frac in zero_fractions:
        n_zero = int(frac * hs)
        trials = []
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 100 + int(frac * 1000))
            v_corrupted = vec.clone()
            # Randomly zero out dimensions
            perm = torch.randperm(hs, device=DEVICE)
            v_corrupted[perm[:n_zero]] = 0
            p = inject_and_measure(v_corrupted)
            trials.append(p)
        zero_results.append({
            'frac': float(frac),
            'n_zero': n_zero,
            'avg_prob': float(np.mean(trials)),
            'recovery': float(np.mean(trials)) / clean_prob,
        })

    # Find resilience threshold (where performance drops to 50%)
    threshold_50 = next((r['frac'] for r in zero_results if r['recovery'] < 0.5), 1.0)
    threshold_90 = next((r['frac'] for r in zero_results if r['recovery'] < 0.9), 1.0)

    print("    50%% performance threshold: %.0f%% dims zeroed" % (threshold_50 * 100))
    print("    90%% performance threshold: %.0f%% dims zeroed" % (threshold_90 * 100))

    # Test 2: Dimension shuffling (scramble parts of S-Qubit)
    print("\n  Test 2: Dimension shuffling...")
    shuffle_fractions = np.linspace(0, 0.99, 20)
    shuffle_results = []

    for frac in shuffle_fractions:
        n_shuffle = int(frac * hs)
        trials = []
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 200 + int(frac * 1000))
            v_shuffled = vec.clone()
            # Randomly shuffle some dimensions
            perm = torch.randperm(hs, device=DEVICE)
            idx_to_shuffle = perm[:n_shuffle]
            shuffled_vals = v_shuffled[idx_to_shuffle][torch.randperm(n_shuffle, device=DEVICE)]
            v_shuffled[idx_to_shuffle] = shuffled_vals
            p = inject_and_measure(v_shuffled)
            trials.append(p)
        shuffle_results.append({
            'frac': float(frac),
            'avg_prob': float(np.mean(trials)),
            'recovery': float(np.mean(trials)) / clean_prob,
        })

    # Test 3: Gaussian noise addition at various scales
    print("\n  Test 3: Additive Gaussian noise...")
    vec_norm = float(vec.norm())
    noise_ratios = np.logspace(-3, 1, 25)  # noise/signal ratio
    noise_results = []

    for ratio in noise_ratios:
        sigma = ratio * vec_norm / np.sqrt(hs)
        trials = []
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 300 + int(ratio * 100))
            noise = torch.randn(hs, device=DEVICE) * sigma
            v_noisy = vec + noise.to(vec.dtype)
            p = inject_and_measure(v_noisy)
            trials.append(p)
        noise_results.append({
            'ratio': float(ratio),
            'sigma': float(sigma),
            'avg_prob': float(np.mean(trials)),
            'recovery': float(np.mean(trials)) / clean_prob,
        })

    noise_threshold = next((r['ratio'] for r in noise_results if r['recovery'] < 0.5), 10.0)
    print("    Noise tolerance (50%% threshold): ratio=%.3f" % noise_threshold)

    # Compute effective code rate (bits of info per dimension)
    bits_per_dim = -np.log2(1 - threshold_50 + 1e-10) if threshold_50 < 1 else 0
    print("\n  RESULTS:")
    print("    Zeroing resilience: %.0f%% dims can be erased" % (threshold_50 * 100))
    print("    Noise tolerance ratio: %.3f" % noise_threshold)
    print("    Effective redundancy: %.1f%% of dims are redundant" % (threshold_50 * 100))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Dimension zeroing
    ax = axes[0]
    fracs = [r['frac'] * 100 for r in zero_results]
    recovs = [r['recovery'] * 100 for r in zero_results]
    ax.plot(fracs, recovs, 'o-', color='#FF5722', linewidth=2, markersize=4)
    ax.axhline(50, color='red', ls=':', alpha=0.5, label='50%% threshold')
    ax.axhline(90, color='green', ls='--', alpha=0.3, label='90%% threshold')
    ax.axvline(threshold_50 * 100, color='blue', ls=':', alpha=0.5,
               label='%.0f%% dims erasable' % (threshold_50 * 100))
    ax.set_xlabel('Dimensions zeroed (%)')
    ax.set_ylabel('Recovery (%)')
    ax.set_title('(a) Dimension Erasure Resilience\n'
                 '%.0f%% dims can be erased at 50%% perf' % (threshold_50 * 100),
                 fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    # (b) Dimension shuffling
    ax = axes[1]
    fracs_s = [r['frac'] * 100 for r in shuffle_results]
    recovs_s = [r['recovery'] * 100 for r in shuffle_results]
    ax.plot(fracs_s, recovs_s, 's-', color='#9C27B0', linewidth=2, markersize=5)
    ax.axhline(50, color='red', ls=':', alpha=0.5)
    ax.set_xlabel('Dimensions shuffled (%)')
    ax.set_ylabel('Recovery (%)')
    ax.set_title('(b) Dimension Shuffle Resilience\n'
                 'Position information in S-Qubit',
                 fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    # (c) Additive noise tolerance
    ax = axes[2]
    ratios = [r['ratio'] for r in noise_results]
    recovs_n = [r['recovery'] * 100 for r in noise_results]
    ax.semilogx(ratios, recovs_n, 'o-', color='#2196F3', linewidth=2, markersize=5)
    ax.axhline(50, color='red', ls=':', alpha=0.5)
    ax.axvline(noise_threshold, color='blue', ls=':', alpha=0.5,
               label='Threshold: %.3f' % noise_threshold)
    ax.set_xlabel('Noise/Signal ratio')
    ax.set_ylabel('Recovery (%)')
    ax.set_title('(c) Additive Noise Tolerance\n'
                 'Robust up to %.2fx noise' % noise_threshold,
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 110)

    plt.suptitle('Phase Q76: Attention-qLDPC v2 (Dimension-Level Error Correction)\n'
                 'S-Qubit information is distributed across redundant dimensions',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q76_qldpc_v2.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q76', 'name': 'attention_qldpc_v2',
        'clean_prob': round(float(clean_prob), 4),
        'zero_threshold_50pct': round(float(threshold_50), 3),
        'zero_threshold_90pct': round(float(threshold_90), 3),
        'noise_threshold': round(float(noise_threshold), 3),
        'effective_redundancy_pct': round(float(threshold_50 * 100), 1),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q76_qldpc_v2.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q76 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
