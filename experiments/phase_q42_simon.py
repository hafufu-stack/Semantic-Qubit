# -*- coding: utf-8 -*-
"""
Phase Q42: Simon's Algorithm

Find the hidden period s where f(x) = f(x XOR s) for some unknown s.
Classical: requires O(2^(n/2)) queries. Quantum: O(n) queries.

This is a key stepping stone to Shor's algorithm.

S-Qubit implementation:
  - For n-bit inputs, create f using phase encoding
  - Query all 2^n inputs, group by E-value
  - Find pairs with same E -> extract s
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
    print("[Q42] Simon's Algorithm")
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

    # Test Simon's algorithm for n = 2, 3, 4
    results = []

    for n_bits in [2, 3, 4]:
        N = 2 ** n_bits
        print("\n  n=%d bits (N=%d)..." % (n_bits, N))

        # Test multiple hidden periods
        test_periods = list(range(1, N))  # s != 0
        if len(test_periods) > 8:
            np.random.seed(n_bits * 100)
            test_periods = np.random.choice(test_periods, 8, replace=False).tolist()

        correct = 0
        total = 0

        for s in test_periods:
            # Create Simon's oracle: f(x) = f(x XOR s)
            # Assign random output values ensuring f(x) = f(x^s)
            np.random.seed(s * 1000 + n_bits)
            f_values = {}
            used_outputs = {}
            for x in range(N):
                partner = x ^ s
                if partner in f_values:
                    f_values[x] = f_values[partner]
                else:
                    # Assign new random phase
                    out_phase = np.random.random() * 2 * np.pi
                    f_values[x] = out_phase
                    f_values[partner] = out_phase

            # Query all inputs and measure E
            E_map = {}
            for x in range(N):
                phi = f_values[x]
                v = phi_vec(phi, v0, v1)
                E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                     INJECT_LAYER, min_tok, max_tok)
                E_map[x] = round(E, 6)

            # Find period: group by E value
            E_groups = {}
            for x, E in E_map.items():
                # Round to find matching groups
                E_key = round(E, 4)
                if E_key not in E_groups:
                    E_groups[E_key] = []
                E_groups[E_key].append(x)

            # Extract s from pairs
            found_s = set()
            for E_key, members in E_groups.items():
                if len(members) >= 2:
                    for i in range(len(members)):
                        for j in range(i+1, len(members)):
                            candidate = members[i] ^ members[j]
                            if candidate > 0:
                                found_s.add(candidate)

            success = s in found_s
            if success:
                correct += 1
            total += 1

            s_bin = format(s, '0%db' % n_bits)
            print("    s=%s (%d): %s (found: %s)" % (
                s_bin, s, "OK" if success else "FAIL",
                ', '.join(format(fs, '0%db' % n_bits) for fs in found_s) if found_s else "none"))

        accuracy = correct / total if total > 0 else 0
        results.append({
            'n_bits': n_bits,
            'n_tested': total,
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'classical_queries': int(2 ** (n_bits / 2)),
            'quantum_queries': n_bits,
        })
        print("  n=%d: %d/%d correct (%.1f%%)" % (n_bits, correct, total, 100*accuracy))

    overall_correct = sum(r['correct'] for r in results)
    overall_total = sum(r['n_tested'] for r in results)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0

    print("\n  SIMON'S ALGORITHM SUMMARY:")
    print("    Overall: %d/%d (%.1f%%)" % (overall_correct, overall_total, 100*overall_acc))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Accuracy per bit length
    ax = axes[0]
    ns = [r['n_bits'] for r in results]
    accs = [r['accuracy'] * 100 for r in results]
    ax.bar(ns, accs, color='#9C27B0', edgecolor='black', alpha=0.85)
    ax.axhline(100, color='green', ls='--', alpha=0.3)
    for n, a in zip(ns, accs):
        ax.text(n, a + 2, '%.0f%%' % a, ha='center', fontweight='bold')
    ax.set_xlabel('Hidden period length (bits)')
    ax.set_ylabel('Recovery accuracy (%)')
    ax.set_title("(a) Simon's Algorithm\nHidden period recovery", fontweight='bold')
    ax.set_ylim(0, 120)
    ax.set_xticks(ns); ax.grid(alpha=0.3, axis='y')

    # Panel B: Query complexity comparison
    ax = axes[1]
    classical_q = [r['classical_queries'] for r in results]
    quantum_q = [r['quantum_queries'] for r in results]
    sq_q = [2**r['n_bits'] for r in results]  # S-Qubit queries all inputs
    x_pos = np.arange(len(ns))
    width = 0.25
    ax.bar(x_pos - width, classical_q, width, color='#90A4AE',
           edgecolor='black', alpha=0.85, label='Classical O(2^(n/2))')
    ax.bar(x_pos, quantum_q, width, color='#2196F3',
           edgecolor='black', alpha=0.85, label='Quantum O(n)')
    ax.bar(x_pos + width, sq_q, width, color='#E91E63',
           edgecolor='black', alpha=0.85, label='S-Qubit O(2^n)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['n=%d' % n for n in ns])
    ax.set_ylabel('Number of queries')
    ax.set_title('(b) Query Complexity\nClassical vs Quantum vs S-Qubit', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Simon's Algorithm\n"
        "=================\n\n"
        "Problem: find s where\n"
        "  f(x) = f(x XOR s)\n\n"
        "Results:\n"
        "%s\n\n"
        "Overall: %.1f%%\n\n"
        "Complexity:\n"
        "  Classical: O(2^(n/2))\n"
        "  Quantum:   O(n)\n"
        "  S-Qubit:   O(2^n)\n"
        "  (brute-force + grouping)" % (
            '\n'.join('  n=%d: %d/%d (%.0f%%)' % (
                r['n_bits'], r['correct'], r['n_tested'], r['accuracy']*100)
                for r in results),
            100 * overall_acc)
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F3E5F5', alpha=0.9))

    plt.suptitle("Phase Q42: Simon's Algorithm\n"
                 "Hidden period finding via S-Qubit oracle queries",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q42_simon.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q42', 'name': 'simon_algorithm',
        'inject_layer': INJECT_LAYER,
        'overall_accuracy': round(float(overall_acc), 6),
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q42_simon.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q42 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
