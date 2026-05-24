# -*- coding: utf-8 -*-
"""
Phase Q40: BB84 Quantum Key Distribution

Simulate the BB84 quantum key distribution protocol using S-Qubits.
BB84 is the first quantum cryptography protocol (Bennett & Brassard 1984).

Protocol:
  1. Alice randomly chooses bits and bases (Z or X)
  2. Alice encodes: Z-basis: |0> or |1>, X-basis: |+> or |->
  3. Bob randomly chooses measurement basis
  4. When bases match, Bob recovers Alice's bit (sifted key)
  5. Test: eavesdropper (Eve) intercepting causes detectable errors

S-Qubit encoding:
  Z-basis: phi=0 (|0>) or phi=pi (|1>)
  X-basis: phi=pi/2 (|+>) or phi=3pi/2 (|->)
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
    print("[Q40] BB84 Quantum Key Distribution")
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

    # BB84 encoding phases
    # Z-basis: bit 0 -> phi=0, bit 1 -> phi=pi
    # X-basis: bit 0 -> phi=pi/2, bit 1 -> phi=3*pi/2
    Z_phases = {0: 0, 1: np.pi}
    X_phases = {0: np.pi/2, 1: 3*np.pi/2}

    # Calibrate: measure E for each encoding
    print("\n  Calibrating encodings...")
    cal = {}
    for basis_name, phases in [('Z', Z_phases), ('X', X_phases)]:
        for bit, phi in phases.items():
            v = phi_vec(phi, v0, v1)
            E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                 INJECT_LAYER, min_tok, max_tok)
            cal[(basis_name, bit)] = E
            print("    %s-basis, bit=%d, phi=%.2f*pi -> E=%.4f" % (
                basis_name, bit, phi/np.pi, E))

    # Decision thresholds
    Z_threshold = (cal[('Z', 0)] + cal[('Z', 1)]) / 2
    X_threshold = (cal[('X', 0)] + cal[('X', 1)]) / 2

    # Run BB84 protocol
    N_ROUNDS = 200
    np.random.seed(42)

    print("\n  Running BB84 (%d rounds)..." % N_ROUNDS)

    # === Scenario 1: No eavesdropper ===
    alice_bits = np.random.randint(0, 2, N_ROUNDS)
    alice_bases = np.random.randint(0, 2, N_ROUNDS)  # 0=Z, 1=X
    bob_bases = np.random.randint(0, 2, N_ROUNDS)

    bob_results = []
    for i in range(N_ROUNDS):
        # Alice encodes
        if alice_bases[i] == 0:
            phi = Z_phases[alice_bits[i]]
        else:
            phi = X_phases[alice_bits[i]]
        v = phi_vec(phi, v0, v1)
        E = inject_measure_E(model, tok, prompt, DEVICE, v,
                             INJECT_LAYER, min_tok, max_tok)

        # Bob decodes with his basis
        if bob_bases[i] == 0:
            bob_bit = 0 if E > Z_threshold else 1
        else:
            bob_bit = 0 if E > X_threshold else 1
        bob_results.append(bob_bit)

    bob_results = np.array(bob_results)

    # Sifting: keep only matching bases
    matching = alice_bases == bob_bases
    sifted_alice = alice_bits[matching]
    sifted_bob = bob_results[matching]
    sifted_correct = np.sum(sifted_alice == sifted_bob)
    sifted_total = len(sifted_alice)
    sifted_rate = sifted_correct / sifted_total if sifted_total > 0 else 0

    # Wrong basis: should be ~50% correct (random)
    wrong_mask = ~matching
    wrong_alice = alice_bits[wrong_mask]
    wrong_bob = bob_results[wrong_mask]
    wrong_correct = np.sum(wrong_alice == wrong_bob)
    wrong_total = len(wrong_alice)
    wrong_rate = wrong_correct / wrong_total if wrong_total > 0 else 0

    print("\n  NO EAVESDROPPER:")
    print("    Sifted key: %d/%d bits correct (%.1f%%)" % (
        sifted_correct, sifted_total, 100*sifted_rate))
    print("    Wrong basis: %d/%d (%.1f%%, expected ~50%%)" % (
        wrong_correct, wrong_total, 100*wrong_rate))

    # === Scenario 2: With eavesdropper (Eve) ===
    print("\n  Running with eavesdropper (Eve)...")
    eve_bases = np.random.randint(0, 2, N_ROUNDS)
    eve_bob_results = []

    for i in range(N_ROUNDS):
        # Alice encodes
        if alice_bases[i] == 0:
            phi = Z_phases[alice_bits[i]]
        else:
            phi = X_phases[alice_bits[i]]
        v_alice = phi_vec(phi, v0, v1)

        # Eve intercepts: measures in her random basis
        E_eve = inject_measure_E(model, tok, prompt, DEVICE, v_alice,
                                  INJECT_LAYER, min_tok, max_tok)
        if eve_bases[i] == 0:
            eve_bit = 0 if E_eve > Z_threshold else 1
        else:
            eve_bit = 0 if E_eve > X_threshold else 1

        # Eve re-encodes in HER basis (this is where disturbance happens)
        if eve_bases[i] == 0:
            phi_eve = Z_phases[eve_bit]
        else:
            phi_eve = X_phases[eve_bit]
        v_eve = phi_vec(phi_eve, v0, v1)

        # Bob measures Eve's re-encoded state
        E_bob = inject_measure_E(model, tok, prompt, DEVICE, v_eve,
                                  INJECT_LAYER, min_tok, max_tok)
        if bob_bases[i] == 0:
            bob_bit = 0 if E_bob > Z_threshold else 1
        else:
            bob_bit = 0 if E_bob > X_threshold else 1
        eve_bob_results.append(bob_bit)

    eve_bob_results = np.array(eve_bob_results)

    # Sifted key with Eve
    eve_sifted_alice = alice_bits[matching]
    eve_sifted_bob = eve_bob_results[matching]
    eve_correct = np.sum(eve_sifted_alice == eve_sifted_bob)
    eve_rate = eve_correct / sifted_total if sifted_total > 0 else 0

    # QBER (Quantum Bit Error Rate)
    qber_no_eve = 1 - sifted_rate
    qber_with_eve = 1 - eve_rate

    print("  WITH EAVESDROPPER:")
    print("    Sifted key: %d/%d bits correct (%.1f%%)" % (
        eve_correct, sifted_total, 100*eve_rate))
    print("    QBER (no Eve):   %.1f%%" % (100 * qber_no_eve))
    print("    QBER (with Eve): %.1f%%" % (100 * qber_with_eve))
    print("    Eve detectable: %s" % ("YES" if qber_with_eve > qber_no_eve + 0.05 else "NO"))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Key agreement comparison
    ax = axes[0]
    labels = ['No Eve\n(sifted)', 'No Eve\n(wrong basis)', 'With Eve\n(sifted)']
    rates = [sifted_rate * 100, wrong_rate * 100, eve_rate * 100]
    colors = ['#4CAF50', '#FF9800', '#F44336']
    bars = ax.bar(labels, rates, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(100, color='green', ls='--', alpha=0.3, label='Perfect')
    ax.axhline(50, color='gray', ls='--', alpha=0.3, label='Random')
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                '%.1f%%' % r, ha='center', fontweight='bold')
    ax.set_ylabel('Agreement rate (%)')
    ax.set_title('(a) BB84 Key Agreement\nSifted key accuracy', fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # Panel B: QBER comparison
    ax = axes[1]
    qber_labels = ['No Eve', 'With Eve']
    qber_vals = [qber_no_eve * 100, qber_with_eve * 100]
    qber_colors = ['#4CAF50', '#F44336']
    bars = ax.bar(qber_labels, qber_vals, color=qber_colors,
                  edgecolor='black', alpha=0.85, width=0.5)
    ax.axhline(11, color='red', ls='--', lw=2, label='BB84 security threshold (11%)')
    for bar, q in zip(bars, qber_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                '%.1f%%' % q, ha='center', fontweight='bold')
    ax.set_ylabel('QBER (%)')
    ax.set_title('(b) Quantum Bit Error Rate\nEve detection via QBER increase',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "BB84 QKD Protocol\n"
        "=================\n\n"
        "Encoding:\n"
        "  Z: |0>=phi_0, |1>=phi_pi\n"
        "  X: |+>=phi_pi/2, |->=phi_3pi/2\n\n"
        "Results:\n"
        "  Rounds: %d\n"
        "  Sifted: %d bits\n"
        "  No Eve:  %.1f%% match\n"
        "  Eve:     %.1f%% match\n\n"
        "  QBER(clean): %.1f%%\n"
        "  QBER(Eve):   %.1f%%\n\n"
        "Eve detectable: %s" % (
            N_ROUNDS, sifted_total,
            100*sifted_rate, 100*eve_rate,
            100*qber_no_eve, 100*qber_with_eve,
            "YES!" if qber_with_eve > qber_no_eve + 0.05 else "NO")
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8EAF6', alpha=0.9))

    plt.suptitle('Phase Q40: BB84 Quantum Key Distribution\n'
                 'Secure key exchange with eavesdropper detection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q40_bb84_qkd.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q40', 'name': 'bb84_qkd',
        'inject_layer': INJECT_LAYER,
        'n_rounds': N_ROUNDS,
        'sifted_bits': int(sifted_total),
        'sifted_accuracy_no_eve': round(float(sifted_rate), 6),
        'sifted_accuracy_with_eve': round(float(eve_rate), 6),
        'wrong_basis_rate': round(float(wrong_rate), 6),
        'qber_no_eve': round(float(qber_no_eve), 6),
        'qber_with_eve': round(float(qber_with_eve), 6),
        'eve_detectable': bool(qber_with_eve > qber_no_eve + 0.05),
        'calibration': {
            'Z0': round(cal[('Z', 0)], 6), 'Z1': round(cal[('Z', 1)], 6),
            'X0': round(cal[('X', 0)], 6), 'X1': round(cal[('X', 1)], 6),
        },
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q40_bb84_qkd.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q40 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
