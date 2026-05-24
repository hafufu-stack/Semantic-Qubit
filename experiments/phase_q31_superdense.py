# -*- coding: utf-8 -*-
"""
Phase Q31: Superdense Coding

Encode 2 classical bits using 1 S-Qubit manipulation.

Protocol (physical QC):
  1. Alice and Bob share an entangled pair
  2. Alice applies one of 4 operations (I, X, Z, XZ) to encode 2 bits
  3. Bob measures and recovers both bits

S-Qubit implementation:
  1. Train entangled pair (SQ1@L8, SQ2@L20)
  2. Alice encodes message by choosing phase: {00: 0, 01: pi/2, 10: pi, 11: 3pi/2}
  3. Bob decodes by measuring E(SQ1, SQ2)
  4. Test all 4 messages x 50 trials = 200 total
  5. Compare: 1 classical bit per qubit vs 2 bits per S-Qubit
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
SQ1_LAYER = 8
SQ2_LAYER = 20
EPOCHS = 100


def train_soul(model, tok, data, device, layer, pos=-1, epochs=EPOCHS, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            seq_len = inp['input_ids'].shape[1]
            actual_pos = pos if pos >= 0 else seq_len + pos
            def hook(m, i, o, v=vec, p=actual_pos):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, p, :] = v.to(h.dtype)
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


def inject_two(model, tok, prompt, device, v1, l1, p1, v2, l2, p2):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    ap1 = p1 if p1 >= 0 else seq_len + p1
    ap2 = p2 if p2 >= 0 else seq_len + p2
    def make_hook(v, p):
        def hook(m, i, o):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, p, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        return hook
    h1 = model.model.layers[l1].register_forward_hook(make_hook(v1, ap1))
    h2 = model.model.layers[l2].register_forward_hook(make_hook(v2, ap2))
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def main():
    print("[Q31] Superdense Coding")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training entangled pair...")
    sq1_v0 = train_soul(model, tok, min_data, DEVICE, SQ1_LAYER, -1, EPOCHS, 42)
    sq1_v1 = train_soul(model, tok, max_data, DEVICE, SQ1_LAYER, -1, EPOCHS, 99)
    sq2_v0 = train_soul(model, tok, min_data, DEVICE, SQ2_LAYER, -2, EPOCHS, 10)
    sq2_v1 = train_soul(model, tok, max_data, DEVICE, SQ2_LAYER, -2, EPOCHS, 20)

    # Encoding scheme: 2 bits -> phase angle
    # We use 4 well-separated phases for maximum distinguishability
    messages = {
        '00': 0,
        '01': np.pi / 2,
        '10': np.pi,
        '11': 3 * np.pi / 2,
    }

    # Step 1: Calibrate -- measure E for each encoding
    print("\n  Calibrating encoding...")
    calibration = {}
    for msg, phi in messages.items():
        v_alice = phi_vec(phi, sq1_v0, sq1_v1)
        v_bob = phi_vec(0, sq2_v0, sq2_v1)  # Bob's qubit at |0>
        probs = inject_two(model, tok, prompt, DEVICE,
                          v_alice, SQ1_LAYER, -1, v_bob, SQ2_LAYER, -2)
        E = float(probs[min_tok]) - float(probs[max_tok])
        calibration[msg] = E
        print("    msg=%s, phi=%.2f*pi -> E=%.4f" % (msg, phi/np.pi, E))

    # Find optimal decision boundaries
    E_values = sorted(calibration.items(), key=lambda x: x[1])
    print("  E ordering:", [(m, round(e, 4)) for m, e in E_values])

    # Step 2: Communication test -- 50 trials per message
    N_TRIALS = 50
    print("\n  Running %d trials per message..." % N_TRIALS)

    total_correct = 0
    total_trials = 0
    per_msg_results = {}

    for msg, phi in messages.items():
        correct = 0
        E_list = []
        for trial in range(N_TRIALS):
            # Alice encodes
            v_alice = phi_vec(phi, sq1_v0, sq1_v1)
            v_bob = phi_vec(0, sq2_v0, sq2_v1)

            # Bob measures
            probs = inject_two(model, tok, prompt, DEVICE,
                              v_alice, SQ1_LAYER, -1, v_bob, SQ2_LAYER, -2)
            E_measured = float(probs[min_tok]) - float(probs[max_tok])
            E_list.append(E_measured)

            # Bob decodes: find closest calibration value
            best_msg = min(calibration, key=lambda m: abs(calibration[m] - E_measured))
            if best_msg == msg:
                correct += 1

        accuracy = correct / N_TRIALS
        per_msg_results[msg] = {
            'accuracy': round(accuracy, 4),
            'E_mean': round(float(np.mean(E_list)), 6),
            'E_std': round(float(np.std(E_list)), 6),
        }
        total_correct += correct
        total_trials += N_TRIALS
        print("    msg=%s: accuracy=%.1f%% (E_mean=%.4f +/- %.4f)" % (
            msg, 100*accuracy, np.mean(E_list), np.std(E_list)))

    overall_accuracy = total_correct / total_trials
    bits_per_qubit = overall_accuracy * 2  # 2 bits encoded, weighted by accuracy

    print("\n  SUPERDENSE CODING SUMMARY:")
    print("    Overall accuracy: %.1f%%" % (100 * overall_accuracy))
    print("    Effective bits/qubit: %.2f (classical limit: 1.0)" % bits_per_qubit)
    print("    Beats classical: %s" % ("YES" if bits_per_qubit > 1.0 else "NO"))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: E values for each message
    ax = axes[0]
    msgs = list(messages.keys())
    E_means = [per_msg_results[m]['E_mean'] for m in msgs]
    E_stds = [per_msg_results[m]['E_std'] for m in msgs]
    colors_msg = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']
    bars = ax.bar(msgs, E_means, yerr=E_stds, color=colors_msg,
                  edgecolor='black', alpha=0.85, capsize=5)
    ax.set_xlabel('2-bit Message')
    ax.set_ylabel('Measured E')
    ax.set_title('(a) Superdense Encoding\n4 messages via 1 S-Qubit phase',
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel B: Accuracy per message
    ax = axes[1]
    accs = [per_msg_results[m]['accuracy'] * 100 for m in msgs]
    ax.bar(msgs, accs, color=colors_msg, edgecolor='black', alpha=0.85)
    ax.axhline(25, color='gray', ls='--', lw=1.5, label='Random guess (25%)')
    ax.axhline(50, color='orange', ls='--', lw=1.5, label='Classical 1-bit (50%)')
    for i, acc in enumerate(accs):
        ax.text(i, acc + 1, '%.0f%%' % acc, ha='center', fontweight='bold')
    ax.set_xlabel('2-bit Message')
    ax.set_ylabel('Decoding Accuracy (%)')
    ax.set_title('(b) Decoding Accuracy\n%.1f%% overall' % (100 * overall_accuracy),
                 fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Superdense Coding\n"
        "=================\n\n"
        "Protocol:\n"
        "  Alice encodes 2 bits\n"
        "  into 1 S-Qubit phase\n"
        "  Bob decodes via E\n\n"
        "Results:\n"
        "  Accuracy: %.1f%%\n"
        "  Bits/qubit: %.2f\n"
        "  Classical max: 1.0\n"
        "  Beat classical: %s\n\n"
        "Encoding:\n"
        "  00 -> phi=0\n"
        "  01 -> phi=pi/2\n"
        "  10 -> phi=pi\n"
        "  11 -> phi=3pi/2" % (
            100 * overall_accuracy, bits_per_qubit,
            "YES!" if bits_per_qubit > 1.0 else "NO")
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F3E5F5', alpha=0.9))

    plt.suptitle('Phase Q31: Superdense Coding\n'
                 '2 classical bits encoded in 1 S-Qubit',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q31_superdense.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q31', 'name': 'superdense_coding',
        'sq1_layer': SQ1_LAYER, 'sq2_layer': SQ2_LAYER,
        'n_trials_per_msg': N_TRIALS,
        'overall_accuracy': round(overall_accuracy, 6),
        'bits_per_qubit': round(bits_per_qubit, 4),
        'beat_classical': bits_per_qubit > 1.0,
        'per_message': per_msg_results,
        'calibration_E': {k: round(v, 6) for k, v in calibration.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q31_superdense.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q31 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
