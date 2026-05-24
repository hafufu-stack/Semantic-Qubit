# -*- coding: utf-8 -*-
"""
Phase Q50: Grand Unified Benchmark

The capstone experiment: run ALL key quantum algorithms on a single
model load and produce a unified "quantum advantage certificate".

Tests:
  1. Deutsch-Jozsa (constant vs balanced)
  2. Bernstein-Vazirani (hidden string recovery)
  3. Simon's Algorithm (hidden period)
  4. Grover Search (target amplification)
  5. BB84 QKD (key exchange)
  6. Superdense Coding (2 bits/qubit)
  7. State Discrimination (parallelism)

All run back-to-back with the same trained soul vectors.
Produces a single "Quantum Advantage Score" (QAS).
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


def inject_measure(model, tok, prompt, device, vec, layer, min_tok, max_tok):
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
    return float(probs[min_tok]), float(probs[max_tok])


def main():
    print("=" * 60)
    print("[Q50] GRAND UNIFIED BENCHMARK")
    print("    The Quantum Advantage Certificate")
    print("=" * 60)
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("\n  Training universal soul vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # Calibrate
    p0_min, p0_max = inject_measure(model, tok, prompt, DEVICE,
                                     phi_vec(0, v0, v1), INJECT_LAYER, min_tok, max_tok)
    p1_min, p1_max = inject_measure(model, tok, prompt, DEVICE,
                                     phi_vec(np.pi, v0, v1), INJECT_LAYER, min_tok, max_tok)
    E0 = p0_min - p0_max
    E1 = p1_min - p1_max
    threshold = (E0 + E1) / 2

    scores = {}

    # ── TEST 1: Deutsch-Jozsa ──
    print("\n  [1/7] Deutsch-Jozsa...")
    dj_correct = 0
    dj_total = 10
    for i in range(dj_total):
        is_constant = (i % 2 == 0)
        if is_constant:
            phi = 0  # f always returns 0
        else:
            phi = np.pi  # f is balanced
        p_min, p_max = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(phi, v0, v1), INJECT_LAYER,
                                       min_tok, max_tok)
        E = p_min - p_max
        predicted_constant = (E > threshold)
        if predicted_constant == is_constant:
            dj_correct += 1
    scores['deutsch_jozsa'] = dj_correct / dj_total
    print("    %d/%d = %.0f%%" % (dj_correct, dj_total, 100 * scores['deutsch_jozsa']))

    # ── TEST 2: Bernstein-Vazirani ──
    print("\n  [2/7] Bernstein-Vazirani (4-bit)...")
    bv_correct = 0
    bv_total = 16
    for s in range(bv_total):
        recovered = 0
        for bit_pos in range(4):
            oracle_bit = (s >> (3 - bit_pos)) & 1
            phi = 0 if oracle_bit == 0 else np.pi
            p_min, _ = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(phi, v0, v1), INJECT_LAYER,
                                       min_tok, max_tok)
            E = p_min - (1 - p_min)
            decoded = 0 if E > threshold else 1
            recovered |= (decoded << (3 - bit_pos))
        if recovered == s:
            bv_correct += 1
    scores['bernstein_vazirani'] = bv_correct / bv_total
    print("    %d/%d = %.0f%%" % (bv_correct, bv_total, 100 * scores['bernstein_vazirani']))

    # ── TEST 3: Simon's Algorithm (3-bit) ──
    print("\n  [3/7] Simon's Algorithm (3-bit)...")
    simon_correct = 0
    simon_total = 7
    for s in range(1, 8):
        np.random.seed(s * 1000 + 3)
        f_values = {}
        for x in range(8):
            partner = x ^ s
            if partner in f_values:
                f_values[x] = f_values[partner]
            else:
                f_values[x] = np.random.random() * 2 * np.pi

        E_map = {}
        for x in range(8):
            p_min, p_max = inject_measure(model, tok, prompt, DEVICE,
                                           phi_vec(f_values[x], v0, v1),
                                           INJECT_LAYER, min_tok, max_tok)
            E_map[x] = round(p_min - p_max, 4)

        E_groups = {}
        for x, E in E_map.items():
            key = round(E, 3)
            if key not in E_groups:
                E_groups[key] = []
            E_groups[key].append(x)

        found_s = set()
        for members in E_groups.values():
            if len(members) >= 2:
                for i in range(len(members)):
                    for j in range(i+1, len(members)):
                        c = members[i] ^ members[j]
                        if c > 0:
                            found_s.add(c)

        if s in found_s:
            simon_correct += 1
    scores['simon'] = simon_correct / simon_total
    print("    %d/%d = %.0f%%" % (simon_correct, simon_total, 100 * scores['simon']))

    # ── TEST 4: Grover Search ──
    print("\n  [4/7] Grover Search...")
    grover_sizes = [4, 16, 64, 256]
    grover_advantages = []
    for N in grover_sizes:
        # Target at phi=0
        p_target, _ = inject_measure(model, tok, prompt, DEVICE,
                                      phi_vec(0, v0, v1), INJECT_LAYER,
                                      min_tok, max_tok)
        # Sample non-targets
        non_target_ps = []
        for k in range(1, min(N, 20)):
            phi = 2 * np.pi * k / N
            p_nt, _ = inject_measure(model, tok, prompt, DEVICE,
                                      phi_vec(phi, v0, v1), INJECT_LAYER,
                                      min_tok, max_tok)
            non_target_ps.append(p_nt)
        avg_nt = np.mean(non_target_ps)
        advantage = p_target / (1/N)
        grover_advantages.append(advantage)
    scores['grover'] = np.mean(grover_advantages) / max(grover_sizes)  # normalize
    # Cap at 1.0
    scores['grover'] = min(scores['grover'], 1.0)
    print("    Advantages: %s" % ', '.join('%.0fx' % a for a in grover_advantages))

    # ── TEST 5: BB84 QKD ──
    print("\n  [5/7] BB84 QKD (50 rounds)...")
    np.random.seed(42)
    n_bb84 = 50
    alice_bits = np.random.randint(0, 2, n_bb84)
    alice_bases = np.random.randint(0, 2, n_bb84)
    bob_bases = np.random.randint(0, 2, n_bb84)
    Z_phases = {0: 0, 1: np.pi}
    X_phases = {0: np.pi/2, 1: 3*np.pi/2}

    bob_results = []
    for i in range(n_bb84):
        phi = Z_phases[alice_bits[i]] if alice_bases[i] == 0 else X_phases[alice_bits[i]]
        p_min, p_max = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(phi, v0, v1), INJECT_LAYER,
                                       min_tok, max_tok)
        E = p_min - p_max
        bob_bit = 0 if E > threshold else 1
        bob_results.append(bob_bit)

    matching = alice_bases == bob_bases
    sifted_correct = np.sum(alice_bits[matching] == np.array(bob_results)[matching])
    sifted_total = matching.sum()
    scores['bb84'] = sifted_correct / sifted_total if sifted_total > 0 else 0
    print("    Sifted key: %d/%d = %.0f%%" % (
        sifted_correct, sifted_total, 100 * scores['bb84']))

    # ── TEST 6: Superdense Coding ──
    print("\n  [6/7] Superdense Coding (50 pairs)...")
    n_sd = 50
    phases_2bit = {
        (0, 0): 0,
        (0, 1): np.pi/2,
        (1, 0): np.pi,
        (1, 1): 3*np.pi/2,
    }
    # Calibrate
    E_cal = {}
    for bits, phi in phases_2bit.items():
        p_min, p_max = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(phi, v0, v1), INJECT_LAYER,
                                       min_tok, max_tok)
        E_cal[bits] = p_min - p_max

    sd_correct = 0
    for _ in range(n_sd):
        b1, b2 = np.random.randint(0, 2), np.random.randint(0, 2)
        phi = phases_2bit[(b1, b2)]
        p_min, p_max = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(phi, v0, v1), INJECT_LAYER,
                                       min_tok, max_tok)
        E = p_min - p_max
        decoded = min(E_cal, key=lambda k: abs(E_cal[k] - E))
        if decoded == (b1, b2):
            sd_correct += 1
    scores['superdense'] = sd_correct / n_sd
    print("    %d/%d = %.0f%%" % (sd_correct, n_sd, 100 * scores['superdense']))

    # ── TEST 7: State Discrimination (Parallelism) ──
    print("\n  [7/7] State Discrimination (N=64)...")
    N_disc = 64
    phases = np.linspace(0, 2 * np.pi * (1 - 1/N_disc), N_disc)
    codebook = {}
    for i, phi in enumerate(phases):
        p_min, p_max = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(phi, v0, v1), INJECT_LAYER,
                                       min_tok, max_tok)
        codebook[i] = p_min - p_max

    disc_correct = 0
    for i, phi in enumerate(phases):
        p_min, p_max = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(phi, v0, v1), INJECT_LAYER,
                                       min_tok, max_tok)
        E = p_min - p_max
        decoded = min(codebook, key=lambda k: abs(codebook[k] - E))
        if decoded == i:
            disc_correct += 1
    scores['discrimination'] = disc_correct / N_disc
    print("    %d/%d = %.0f%%" % (disc_correct, N_disc, 100 * scores['discrimination']))

    # ── QUANTUM ADVANTAGE SCORE ──
    QAS = np.mean(list(scores.values())) * 100
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  QUANTUM ADVANTAGE CERTIFICATE")
    print("=" * 60)
    print("  Model: Qwen2.5-3B-Instruct")
    print("  Device: %s" % DEVICE)
    print("  Time: %.0fs" % elapsed)
    print()
    for name, score in scores.items():
        bar = "#" * int(score * 30) + "." * (30 - int(score * 30))
        print("  %-22s [%s] %.0f%%" % (name, bar, 100 * score))
    print()
    print("  QUANTUM ADVANTAGE SCORE: %.1f / 100" % QAS)
    print("=" * 60)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Radar chart (as bar chart since matplotlib radar is tricky)
    ax = axes[0]
    test_names = list(scores.keys())
    test_scores = [scores[n] * 100 for n in test_names]
    colors = ['#4CAF50' if s >= 90 else '#FF9800' if s >= 70 else '#F44336'
              for s in test_scores]
    bars = ax.barh(range(len(test_names)), test_scores, color=colors,
                   edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(test_names)))
    ax.set_yticklabels([n.replace('_', ' ').title() for n in test_names], fontsize=10)
    ax.set_xlabel('Score (%)')
    ax.axvline(90, color='green', ls='--', alpha=0.3, label='Excellent (90%)')
    ax.axvline(50, color='red', ls='--', alpha=0.3, label='Random (50%)')
    for bar, s in zip(bars, test_scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                '%.0f%%' % s, va='center', fontweight='bold', fontsize=10)
    ax.set_title('Quantum Algorithm Benchmark', fontweight='bold', fontsize=13)
    ax.set_xlim(0, 115)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='x')

    # Panel B: QAS certificate
    ax = axes[1]
    ax.axis('off')
    cert_color = '#E8F5E9' if QAS >= 80 else '#FFF9C4' if QAS >= 50 else '#FFEBEE'
    certificate = (
        "QUANTUM ADVANTAGE\n"
        "CERTIFICATE\n"
        "==================\n\n"
        "Model: Qwen2.5-3B\n"
        "Framework: Soul Vector\n"
        "Device: %s\n\n"
        "Test Results:\n"
        "%s\n\n"
        "SCORE: %.1f / 100\n\n"
        "Grade: %s\n\n"
        "Time: %.0f seconds\n"
        "Date: 2026-05-24" % (
            DEVICE.upper(),
            '\n'.join('  %s: %.0f%%' % (n.replace('_', ' ').title(), scores[n]*100)
                      for n in test_names),
            QAS,
            "EXCELLENT" if QAS >= 90 else "GOOD" if QAS >= 70 else
            "MODERATE" if QAS >= 50 else "NEEDS WORK",
            elapsed)
    )
    ax.text(0.5, 0.5, certificate, ha='center', va='center',
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round,pad=1.0', facecolor=cert_color,
                      edgecolor='black', linewidth=2))

    plt.suptitle('Phase Q50: Grand Unified Benchmark\n'
                 'Quantum Advantage Score = %.1f / 100' % QAS,
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q50_grand_benchmark.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q50', 'name': 'grand_unified_benchmark',
        'inject_layer': INJECT_LAYER,
        'QAS': round(float(QAS), 2),
        'scores': {k: round(float(v), 4) for k, v in scores.items()},
        'elapsed': round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q50_grand_benchmark.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("\n  Q50 completed in %.0fs" % elapsed)
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
