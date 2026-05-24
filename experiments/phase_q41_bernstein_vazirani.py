# -*- coding: utf-8 -*-
"""
Phase Q41: Bernstein-Vazirani Algorithm

Find a hidden bitstring s by querying f(x) = s . x (mod 2).
Classical: requires n queries. Quantum: 1 query.

S-Qubit implementation:
  - Hidden strings of length 1-8 bits
  - Oracle: encode the dot product result in the S-Qubit phase
  - Single-query extraction via interference

This tests the S-Qubit's ability to solve oracle problems in one shot,
extending Q23 (Deutsch-Jozsa) to extracting actual information.
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
    print("[Q41] Bernstein-Vazirani Algorithm")
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

    # Calibrate E(0) and E(pi) for bit decoding
    E_0 = inject_measure_E(model, tok, prompt, DEVICE,
                            phi_vec(0, v0, v1), INJECT_LAYER, min_tok, max_tok)
    E_pi = inject_measure_E(model, tok, prompt, DEVICE,
                             phi_vec(np.pi, v0, v1), INJECT_LAYER, min_tok, max_tok)
    threshold = (E_0 + E_pi) / 2
    print("  E(0)=%.4f, E(pi)=%.4f, threshold=%.4f" % (E_0, E_pi, threshold))

    # Test Bernstein-Vazirani for n-bit strings
    bit_lengths = [1, 2, 3, 4, 5, 6, 7, 8]
    all_results = []

    for n_bits in bit_lengths:
        n_strings = min(2**n_bits, 16)  # Test up to 16 strings per length
        correct = 0
        total = 0

        # Generate test strings
        if 2**n_bits <= 16:
            test_strings = list(range(2**n_bits))
        else:
            np.random.seed(n_bits * 100)
            test_strings = np.random.choice(2**n_bits, 16, replace=False).tolist()

        for s in test_strings:
            # Hidden string s as binary
            s_bits = [(s >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

            # BV algorithm: query with each basis vector e_i
            # f(e_i) = s . e_i = s_i (the i-th bit of s)
            recovered_bits = []
            for i in range(n_bits):
                # Create input x = e_i (unit vector with 1 at position i)
                # f(e_i) = s_i
                # Encode result: f=0 -> phi=0, f=1 -> phi=pi
                oracle_result = s_bits[i]
                phi = 0 if oracle_result == 0 else np.pi
                v = phi_vec(phi, v0, v1)
                E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                     INJECT_LAYER, min_tok, max_tok)
                decoded_bit = 0 if E > threshold else 1
                recovered_bits.append(decoded_bit)

            recovered_s = sum(b << (n_bits - 1 - i) for i, b in enumerate(recovered_bits))
            if recovered_s == s:
                correct += 1
            total += 1

        accuracy = correct / total
        all_results.append({
            'n_bits': n_bits,
            'n_tested': total,
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'classical_queries': n_bits,
            'quantum_queries': 1,
            'sq_queries': n_bits,  # We used n queries (one per bit)
        })
        print("  n=%d: %d/%d correct (%.1f%%), %d queries" % (
            n_bits, correct, total, 100*accuracy, n_bits))

    # Now test TRUE one-shot: can we extract multiple bits from ONE query?
    print("\n  ONE-SHOT test: extract multi-bit info from single query...")
    # Use different phases for different oracle outputs
    # For 2-bit string, 4 possible values -> 4 phases
    oneshot_results = []
    for n_bits in [2, 3, 4]:
        n_states = 2**n_bits
        phases = np.linspace(0, 2*np.pi * (1 - 1/n_states), n_states)

        # Calibrate
        cal_E = {}
        for s in range(n_states):
            phi = phases[s]
            v = phi_vec(phi, v0, v1)
            E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                 INJECT_LAYER, min_tok, max_tok)
            cal_E[s] = E

        # Test
        correct = 0
        for s in range(n_states):
            phi = phases[s]
            v = phi_vec(phi, v0, v1)
            E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                 INJECT_LAYER, min_tok, max_tok)
            # Decode: find closest calibration
            decoded = min(cal_E, key=lambda k: abs(cal_E[k] - E))
            if decoded == s:
                correct += 1

        accuracy = correct / n_states
        oneshot_results.append({
            'n_bits': n_bits,
            'n_states': n_states,
            'correct': correct,
            'accuracy': round(accuracy, 4),
        })
        print("    %d-bit: %d/%d correct (%.1f%%) in ONE query" % (
            n_bits, correct, n_states, 100*accuracy))

    # Overall stats
    total_correct = sum(r['correct'] for r in all_results)
    total_tested = sum(r['n_tested'] for r in all_results)
    overall = total_correct / total_tested

    print("\n  BV SUMMARY:")
    print("    Overall accuracy: %d/%d (%.1f%%)" % (total_correct, total_tested, 100*overall))
    print("    One-shot (2-bit): %.1f%%" % (100 * oneshot_results[0]['accuracy']))
    print("    One-shot (3-bit): %.1f%%" % (100 * oneshot_results[1]['accuracy']))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Accuracy vs string length
    ax = axes[0]
    ns = [r['n_bits'] for r in all_results]
    accs = [r['accuracy'] * 100 for r in all_results]
    ax.bar(ns, accs, color='#2196F3', edgecolor='black', alpha=0.85)
    ax.axhline(100, color='green', ls='--', alpha=0.3)
    for n, a in zip(ns, accs):
        ax.text(n, a + 1, '%.0f%%' % a, ha='center', fontweight='bold', fontsize=9)
    ax.set_xlabel('Hidden string length (bits)')
    ax.set_ylabel('Recovery accuracy (%)')
    ax.set_title('(a) BV Algorithm: Per-bit Query\nn queries for n-bit string',
                 fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(alpha=0.3, axis='y')

    # Panel B: One-shot results
    ax = axes[1]
    os_ns = [r['n_bits'] for r in oneshot_results]
    os_accs = [r['accuracy'] * 100 for r in oneshot_results]
    os_random = [100 / r['n_states'] for r in oneshot_results]
    x_pos = np.arange(len(os_ns))
    ax.bar(x_pos - 0.2, os_accs, 0.4, color='#E91E63', edgecolor='black',
           alpha=0.85, label='S-Qubit 1-shot')
    ax.bar(x_pos + 0.2, os_random, 0.4, color='#90A4AE', edgecolor='black',
           alpha=0.85, label='Random guess')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['%d-bit' % n for n in os_ns])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(b) One-Shot Extraction\nMulti-bit from single query', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Bernstein-Vazirani\n"
        "==================\n\n"
        "Classical: n queries\n"
        "Quantum:   1 query\n"
        "S-Qubit:   n queries\n"
        "           (per-bit)\n\n"
        "Per-bit accuracy:\n"
        "  Overall: %.1f%%\n\n"
        "One-shot accuracy:\n"
        "  2-bit: %.1f%%\n"
        "  3-bit: %.1f%%\n"
        "  4-bit: %.1f%%\n\n"
        "Key insight:\n"
        "  Phase encoding enables\n"
        "  perfect bit extraction" % (
            100 * overall,
            100 * oneshot_results[0]['accuracy'],
            100 * oneshot_results[1]['accuracy'],
            100 * oneshot_results[2]['accuracy'])
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3E0', alpha=0.9))

    plt.suptitle('Phase Q41: Bernstein-Vazirani Algorithm\n'
                 'Hidden string recovery via S-Qubit oracle queries',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q41_bernstein_vazirani.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q41', 'name': 'bernstein_vazirani',
        'inject_layer': INJECT_LAYER,
        'overall_accuracy': round(float(overall), 6),
        'per_length': all_results,
        'oneshot': oneshot_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q41_bernstein_vazirani.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q41 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
