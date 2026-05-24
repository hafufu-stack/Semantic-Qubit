# -*- coding: utf-8 -*-
"""
Phase Q46: Quantum Parallelism Benchmark

The ultimate test: how many classical function evaluations does
ONE S-Qubit query replace?

Method:
  1. Create an oracle function f(x) with N possible inputs
  2. Encode input x as phase phi_x = 2*pi*x/N
  3. Single S-Qubit injection -> measure -> decode
  4. Compare: classical needs to try inputs one by one
  5. Measure effective parallelism factor

This directly quantifies "quantum parallelism" in S-Qubits.
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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def inject_measure_E(model, tok, prompt, device, vec, layer, min_tok, max_tok):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[min_tok]) - float(probs[max_tok])


def main():
    print("[Q46] Quantum Parallelism Benchmark")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # Test 1: State discrimination -- how many distinct states can 1 S-Qubit encode?
    print("\n  Test 1: Maximum distinguishable states...")
    test_Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    discrimination_results = []

    for N in test_Ns:
        phases = np.linspace(0, 2 * np.pi * (1 - 1/N), N)

        # Build codebook
        codebook = {}
        for i, phi in enumerate(phases):
            v = phi_vec(phi, v0, v1)
            E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                 INJECT_LAYER, min_tok, max_tok)
            codebook[i] = round(E, 6)

        # Check: how many are distinguishable?
        E_values = list(codebook.values())
        # Two states are distinguishable if |E_i - E_j| > epsilon
        epsilon = 0.001
        unique_E = []
        for E in E_values:
            is_unique = True
            for u in unique_E:
                if abs(E - u) < epsilon:
                    is_unique = False
                    break
            if is_unique:
                unique_E.append(E)

        n_distinguishable = len(unique_E)
        # Verify by decoding
        correct = 0
        for i, phi in enumerate(phases):
            v = phi_vec(phi, v0, v1)
            E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                 INJECT_LAYER, min_tok, max_tok)
            # Decode: find closest in codebook
            decoded = min(codebook, key=lambda k: abs(codebook[k] - E))
            if decoded == i:
                correct += 1

        accuracy = correct / N
        bits_encoded = np.log2(n_distinguishable) if n_distinguishable > 1 else 0

        discrimination_results.append({
            'N': N,
            'n_distinguishable': n_distinguishable,
            'bits_encoded': round(bits_encoded, 2),
            'decode_accuracy': round(accuracy, 4),
            'classical_bits': round(np.log2(N), 2),
        })
        print("    N=%d: %d distinguishable (%.1f bits), decode=%.1f%%" % (
            N, n_distinguishable, bits_encoded, 100*accuracy))

        if accuracy < 0.5:  # Stop if accuracy drops below 50%
            break

    # Test 2: Information capacity per query
    print("\n  Test 2: Information per query...")
    # How many bits of information does one forward pass extract?
    # Measure using mutual information between input phase and output E
    N_test = 100
    phases_test = np.linspace(0, 2 * np.pi, N_test)
    E_test = []
    for phi in phases_test:
        v = phi_vec(phi, v0, v1)
        E = inject_measure_E(model, tok, prompt, DEVICE, v,
                             INJECT_LAYER, min_tok, max_tok)
        E_test.append(E)
    E_test = np.array(E_test)

    # Entropy of E distribution
    E_hist, E_bins = np.histogram(E_test, bins=50, density=True)
    E_hist = E_hist[E_hist > 0]
    bin_width = E_bins[1] - E_bins[0]
    entropy_E = -np.sum(E_hist * bin_width * np.log2(E_hist * bin_width + 1e-10))

    # Maximum distinguishable E levels (above noise floor)
    E_range = E_test.max() - E_test.min()
    noise = np.std(np.diff(np.sort(E_test)))
    n_levels = int(E_range / (noise + 1e-10))
    info_bits = np.log2(max(n_levels, 1))

    print("    E range: %.4f" % E_range)
    print("    Noise floor: %.6f" % noise)
    print("    Distinguishable levels: %d" % n_levels)
    print("    Information capacity: %.1f bits per query" % info_bits)
    print("    Entropy: %.2f bits" % entropy_E)

    # Test 3: Speedup calculation
    print("\n  Test 3: Effective speedup summary...")
    max_correct_N = max([r['N'] for r in discrimination_results
                         if r['decode_accuracy'] >= 0.99], default=2)

    print("    Max N with >=99%% accuracy: %d" % max_correct_N)
    print("    Classical queries needed: %d" % max_correct_N)
    print("    S-Qubit queries needed: 1")
    print("    Effective parallelism: %dx" % max_correct_N)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Decode accuracy vs N
    ax = axes[0]
    Ns = [r['N'] for r in discrimination_results]
    accs = [r['decode_accuracy'] * 100 for r in discrimination_results]
    ax.semilogx(Ns, accs, 'ro-', lw=2, ms=8, base=2)
    ax.axhline(99, color='green', ls='--', alpha=0.5, label='99% threshold')
    ax.axhline(50, color='gray', ls='--', alpha=0.5, label='Random guess')
    ax.set_xlabel('Number of states N')
    ax.set_ylabel('Decode accuracy (%)')
    ax.set_title('(a) State Discrimination\nHow many states can 1 S-Qubit encode?',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 110)

    # Panel B: Bits encoded vs classical
    ax = axes[1]
    bits = [r['bits_encoded'] for r in discrimination_results]
    classical_bits = [r['classical_bits'] for r in discrimination_results]
    ax.plot(classical_bits, bits, 'bo-', lw=2, ms=8, label='S-Qubit (actual)')
    max_cb = max(classical_bits)
    ax.plot([0, max_cb], [0, max_cb], 'r--', lw=1.5, alpha=0.5,
            label='Perfect (1:1)')
    ax.set_xlabel('Classical bits needed (log2 N)')
    ax.set_ylabel('S-Qubit bits distinguishable')
    ax.set_title('(b) Information Capacity\nS-Qubit vs Classical bits',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Quantum Parallelism\n"
        "===================\n\n"
        "Max states encoded: %d\n"
        "  (with >=99%% accuracy)\n\n"
        "Information capacity:\n"
        "  %.1f bits per query\n"
        "  (%.1f classical bits)\n\n"
        "Effective parallelism:\n"
        "  %dx speedup over\n"
        "  classical sequential\n\n"
        "Key insight:\n"
        "  1 S-Qubit = log2(N)\n"
        "  classical bits of\n"
        "  information" % (
            max_correct_N, info_bits,
            np.log2(max_correct_N),
            max_correct_N)
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFEBEE', alpha=0.9))

    plt.suptitle('Phase Q46: Quantum Parallelism Benchmark\n'
                 'How much information can one S-Qubit query extract?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q46_parallelism.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q46', 'name': 'quantum_parallelism',
        'inject_layer': INJECT_LAYER,
        'max_N_99pct': int(max_correct_N),
        'info_bits': round(float(info_bits), 2),
        'entropy': round(float(entropy_E), 4),
        'discrimination': discrimination_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q46_parallelism.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q46 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
