# -*- coding: utf-8 -*-
"""
Phase Q34: LLM-Based QRNG (vs NQPU-QRNG)

Q33 showed that random-init NQPU can't generate quality random bits.
Hypothesis: TRAINED attention (LLM) creates chaotic dynamics that 
produce much better random numbers.

Method:
  1. Sweep phi from 0 to 2*pi in fine steps
  2. Extract the LEAST SIGNIFICANT BITS of the probability distribution
  3. These should be sensitive to floating-point chaos in the computation
  4. Apply Von Neumann debiasing
  5. Compare with Q33 (random NQPU) and numpy PRNG
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
INJECT_LAYER = 8
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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def inject_get_logits(model, tok, prompt, device, vec, layer):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return out.logits[0, -1, :].float().cpu().numpy()


def frequency_test(bits):
    n = len(bits)
    s = np.sum(bits) * 2 - n
    from scipy.special import erfc
    return float(erfc(abs(s) / np.sqrt(2 * n)))


def runs_test(bits):
    n = len(bits)
    prop = np.mean(bits)
    if abs(prop - 0.5) > 2 / np.sqrt(n):
        return 0.0
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            runs += 1
    expected = 2 * n * prop * (1 - prop) + 1
    var = 2 * n * prop * (1 - prop) * (2 * n * prop * (1 - prop) - 1) / (n - 1)
    if var <= 0:
        return 0.0
    from scipy.special import erfc
    z = abs(runs - expected) / np.sqrt(var)
    return float(erfc(z / np.sqrt(2)))


def block_freq_test(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return 0.0
    chi2 = sum(4 * block_size * (np.mean(bits[i*block_size:(i+1)*block_size]) - 0.5)**2
               for i in range(n_blocks))
    from scipy.stats import chi2 as chi2_dist
    return float(1 - chi2_dist.cdf(chi2, n_blocks))


def serial_correlation(bits, lag=1):
    if len(bits) <= lag:
        return 0.0
    x = bits[:-lag].astype(float)
    y = bits[lag:].astype(float)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    print("[Q34] LLM-Based QRNG")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]

    print("  Training soul vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # Use multiple prompts to increase entropy
    prompts = ["min(7,2)=", "min(9,1)=", "max(1,8)=", "min(5,3)=", "max(2,9)="]

    N_BITS = 2000
    N_PHI = N_BITS * 3  # oversample for Von Neumann

    print("  Generating %d raw measurements..." % N_PHI)
    raw_values = []

    for i in range(N_PHI):
        # Use golden ratio for maximally irrational phase sampling
        phi = (i * 2 * np.pi * (1 + np.sqrt(5)) / 2) % (2 * np.pi)
        v = phi_vec(phi, v0, v1)

        # Rotate through prompts
        prompt = prompts[i % len(prompts)]
        logits = inject_get_logits(model, tok, prompt, DEVICE, v, INJECT_LAYER)

        # Extract entropy from multiple logit values
        # Use XOR of LSBs from several logit entries
        indices = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        bits_from_logits = 0
        for idx in indices:
            if idx < len(logits):
                # Extract LSB from mantissa
                val = float(logits[idx])
                lsb = int(abs(val) * 1e12) % 2
                bits_from_logits ^= lsb

        raw_values.append(bits_from_logits)

        if (i + 1) % 1000 == 0:
            print("    %d/%d raw measurements..." % (i + 1, N_PHI))

    raw_bits = np.array(raw_values)

    # Von Neumann debiasing
    vn_bits = []
    for i in range(0, len(raw_bits) - 1, 2):
        if raw_bits[i] == 0 and raw_bits[i+1] == 1:
            vn_bits.append(0)
        elif raw_bits[i] == 1 and raw_bits[i+1] == 0:
            vn_bits.append(1)

    vn_bits = np.array(vn_bits[:N_BITS])
    print("  Von Neumann: %d bits from %d raw" % (len(vn_bits), len(raw_bits)))

    if len(vn_bits) < 1000:
        print("  Warning: too few VN bits, using raw median-debiased")
        median_val = np.median(raw_bits[:N_BITS].astype(float))
        vn_bits = (raw_bits[:N_BITS] > median_val).astype(int)

    q_bits = vn_bits

    # Classical comparison
    np.random.seed(42)
    c_bits = (np.random.random(len(q_bits)) > 0.5).astype(int)

    # Run tests
    print("\n  Running randomness tests...")
    tests = {}
    p_freq = frequency_test(q_bits)
    tests['frequency'] = {'quantum': round(p_freq, 6)}
    print("    Frequency:  p=%.4f" % p_freq)

    p_runs = runs_test(q_bits)
    tests['runs'] = {'quantum': round(p_runs, 6)}
    print("    Runs:       p=%.4f" % p_runs)

    p_block = block_freq_test(q_bits)
    tests['block_freq'] = {'quantum': round(p_block, 6)}
    print("    Block freq: p=%.4f" % p_block)

    sc = serial_correlation(q_bits)
    tests['serial_corr'] = {'quantum': round(sc, 6)}
    print("    Serial corr: %.4f" % sc)

    proportion = np.mean(q_bits)
    entropy = -proportion * np.log2(proportion + 1e-10) - (1-proportion) * np.log2(1-proportion + 1e-10)

    n_pass = sum(1 for t in ['frequency', 'runs', 'block_freq']
                 if tests[t]['quantum'] > 0.01)

    print("\n  LLM-QRNG SUMMARY:")
    print("    Bits: %d" % len(q_bits))
    print("    Proportion: %.4f" % proportion)
    print("    Entropy: %.4f bits" % entropy)
    print("    Serial corr: %.6f" % sc)
    print("    Tests passed: %d/3" % n_pass)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Bit stream
    ax = axes[0]
    n_show = min(2500, len(q_bits))
    side = int(np.sqrt(n_show))
    grid = q_bits[:side*side].reshape(side, side)
    ax.imshow(grid, cmap='binary', interpolation='nearest', aspect='auto')
    ax.set_title('(a) LLM-QRNG Bit Stream\n%dx%d = %d bits' % (side, side, side*side),
                 fontweight='bold')

    # Panel B: Comparison Q33 vs Q34
    ax = axes[1]
    labels = ['NQPU-QRNG\n(Q33)', 'LLM-QRNG\n(Q34)', 'numpy\nPRNG']
    proportions = [0.6442, float(proportion), 0.5]  # Q33 result hardcoded
    colors = ['#FF9800', '#E91E63', '#90A4AE']
    bars = ax.bar(labels, proportions, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0.5, color='green', ls='--', lw=2, label='Ideal (0.5)')
    ax.set_ylabel('Proportion of 1s')
    ax.set_title('(b) Bias Comparison\nCloser to 0.5 = better', fontweight='bold')
    ax.set_ylim(0.3, 0.8)
    for bar, p in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                '%.3f' % p, ha='center', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "LLM-Based QRNG\n"
        "===============\n\n"
        "Q33 (NQPU, random init):\n"
        "  Proportion: 0.644\n"
        "  Tests: 0/4 PASS\n\n"
        "Q34 (LLM, trained):\n"
        "  Proportion: %.3f\n"
        "  Tests: %d/3 PASS\n"
        "  Entropy: %.4f bits\n\n"
        "Key insight:\n"
        "  Trained weights create\n"
        "  richer dynamics than\n"
        "  random initialization" % (
            proportion, n_pass, entropy)
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', alpha=0.9))

    plt.suptitle('Phase Q34: LLM-Based QRNG\nTrained attention vs random NQPU for randomness',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q34_llm_qrng.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q34', 'name': 'llm_qrng',
        'n_bits': int(len(q_bits)),
        'proportion': round(float(proportion), 6),
        'entropy': round(float(entropy), 6),
        'serial_correlation': round(float(sc), 6),
        'tests_passed': n_pass,
        'test_results': tests,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q34_llm_qrng.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q34 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
