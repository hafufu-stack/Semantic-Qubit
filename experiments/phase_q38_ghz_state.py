# -*- coding: utf-8 -*-
"""
Phase Q38: GHZ State -- Multi-Qubit Entanglement

Create a 3-qubit GHZ state |000> + |111> using S-Qubits.
GHZ states are maximally entangled and violate Mermin's inequality.

Physical QC: |GHZ> = (|000> + |111>) / sqrt(2)
- Measuring any qubit collapses ALL others

S-Qubit implementation:
  1. Train 3 soul vectors at SAME layer, different positions
  2. Phase-lock them to create correlated states
  3. Test Mermin inequality: M = <XXY> - <XYX> - <YXX> - <YYY>
     Classical: |M| <= 2,  Quantum: |M| = 4
  4. Test all-or-nothing correlation (measuring one predicts others)
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


def inject_multi_measure(model, tok, prompt, device, injections, layer):
    """Inject multiple vectors at same layer, different positions."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    positions = []
    vectors = []
    for pos, vec in injections:
        actual_pos = pos if pos >= 0 else seq_len + pos
        positions.append(actual_pos)
        vectors.append(vec)

    def hook(m, i, o):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        for p, v in zip(positions, vectors):
            h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return probs


def main():
    print("[Q38] GHZ State -- Multi-Qubit Entanglement")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    # Train 3 pairs of soul vectors at different positions
    print("  Training 3 qubit soul vectors...")
    # Qubit A at pos -1, Qubit B at pos -2, Qubit C at pos -3
    qA_v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, -1, EPOCHS, 42)
    qA_v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, -1, EPOCHS, 99)
    qB_v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, -2, EPOCHS, 10)
    qB_v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, -2, EPOCHS, 20)
    qC_v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, -3, EPOCHS, 30)
    qC_v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, -3, EPOCHS, 40)

    # Test 1: GHZ correlation -- all same phase
    print("\n  Test 1: GHZ-like phase correlation...")
    n_phi = 37
    phis = np.linspace(0, 2 * np.pi, n_phi)

    # When all 3 qubits are at the same phase, measure E
    ghz_same = []
    for phi in phis:
        vA = phi_vec(phi, qA_v0, qA_v1)
        vB = phi_vec(phi, qB_v0, qB_v1)
        vC = phi_vec(phi, qC_v0, qC_v1)
        probs = inject_multi_measure(model, tok, prompt, DEVICE,
                                      [(-1, vA), (-2, vB), (-3, vC)], INJECT_LAYER)
        E = float(probs[min_tok]) - float(probs[max_tok])
        ghz_same.append(E)

    # When only qubit A varies, B and C fixed at |0>
    ghz_a_only = []
    for phi in phis:
        vA = phi_vec(phi, qA_v0, qA_v1)
        vB = phi_vec(0, qB_v0, qB_v1)
        vC = phi_vec(0, qC_v0, qC_v1)
        probs = inject_multi_measure(model, tok, prompt, DEVICE,
                                      [(-1, vA), (-2, vB), (-3, vC)], INJECT_LAYER)
        E = float(probs[min_tok]) - float(probs[max_tok])
        ghz_a_only.append(E)

    # Test 2: Mermin inequality
    # M = E(X,X,Y) - E(X,Y,X) - E(Y,X,X) - E(Y,Y,Y)
    # X basis: phi=0, Y basis: phi=pi/2
    print("  Test 2: Mermin inequality...")
    X_phase = 0
    Y_phase = np.pi / 2

    def measure_3qubit(phi_a, phi_b, phi_c):
        vA = phi_vec(phi_a, qA_v0, qA_v1)
        vB = phi_vec(phi_b, qB_v0, qB_v1)
        vC = phi_vec(phi_c, qC_v0, qC_v1)
        probs = inject_multi_measure(model, tok, prompt, DEVICE,
                                      [(-1, vA), (-2, vB), (-3, vC)], INJECT_LAYER)
        return float(probs[min_tok]) - float(probs[max_tok])

    E_XXY = measure_3qubit(X_phase, X_phase, Y_phase)
    E_XYX = measure_3qubit(X_phase, Y_phase, X_phase)
    E_YXX = measure_3qubit(Y_phase, X_phase, X_phase)
    E_YYY = measure_3qubit(Y_phase, Y_phase, Y_phase)

    Mermin = E_XXY - E_XYX - E_YXX - E_YYY
    print("    E(XXY) = %.4f" % E_XXY)
    print("    E(XYX) = %.4f" % E_XYX)
    print("    E(YXX) = %.4f" % E_YXX)
    print("    E(YYY) = %.4f" % E_YYY)
    print("    Mermin M = %.4f (classical <= 2, quantum = 4)" % Mermin)

    # Test 3: Full angle sweep for Mermin-like
    print("  Test 3: Mermin vs angle sweep...")
    sweep_phis = np.linspace(0, np.pi, 19)
    mermin_values = []
    for phi_offset in sweep_phis:
        e1 = measure_3qubit(0, 0, phi_offset)
        e2 = measure_3qubit(0, phi_offset, 0)
        e3 = measure_3qubit(phi_offset, 0, 0)
        e4 = measure_3qubit(phi_offset, phi_offset, phi_offset)
        M = e1 - e2 - e3 - e4
        mermin_values.append(M)

    # Test 4: All-or-nothing correlation
    print("  Test 4: All-or-nothing correlation test...")
    n_configs = 8  # 2^3 configurations
    config_E = []
    for cfg in range(n_configs):
        phi_a = 0 if (cfg & 4) == 0 else np.pi
        phi_b = 0 if (cfg & 2) == 0 else np.pi
        phi_c = 0 if (cfg & 1) == 0 else np.pi
        E = measure_3qubit(phi_a, phi_b, phi_c)
        label = "%d%d%d" % ((cfg>>2)&1, (cfg>>1)&1, cfg&1)
        config_E.append((label, E))
        print("    |%s>: E=%.4f" % (label, E))

    ghz_same = np.array(ghz_same)
    ghz_a_only = np.array(ghz_a_only)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: GHZ phase sweep
    ax = axes[0]
    ax.plot(phis / np.pi, ghz_same, 'r-', lw=2, ms=4, label='All 3 same phase')
    ax.plot(phis / np.pi, ghz_a_only, 'b--', lw=2, label='Only A varies')
    ax.set_xlabel('Phase (x pi)')
    ax.set_ylabel('E = P(min) - P(max)')
    ax.set_title('(a) GHZ Phase Correlation\n3-qubit coherent phase sweep',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: Mermin sweep
    ax = axes[1]
    ax.plot(sweep_phis / np.pi, mermin_values, 'go-', lw=2, ms=6)
    ax.axhline(2.0, color='gray', ls='--', lw=1.5, label='Classical bound (2)')
    ax.axhline(-2.0, color='gray', ls='--', lw=1.5)
    ax.axhline(4.0, color='blue', ls='--', lw=1.5, label='Quantum bound (4)')
    ax.set_xlabel('Offset angle (x pi)')
    ax.set_ylabel('Mermin M')
    ax.set_title('(b) Mermin Inequality\nM=%.2f at pi/2' % Mermin, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel C: Configuration energies
    ax = axes[2]
    labels = [c[0] for c in config_E]
    Es = [c[1] for c in config_E]
    colors = ['#E91E63' if l in ['000', '111'] else '#90A4AE' for l in labels]
    ax.bar(labels, Es, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xlabel('3-qubit configuration')
    ax.set_ylabel('E = P(min) - P(max)')
    ax.set_title('(c) All-or-Nothing Test\nGHZ: |000> and |111> should dominate',
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q38: GHZ State (3-Qubit Entanglement)\n'
                 'Mermin inequality and multi-qubit correlations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q38_ghz_state.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q38', 'name': 'ghz_state',
        'inject_layer': INJECT_LAYER,
        'mermin_M': round(float(Mermin), 4),
        'mermin_classical_bound': 2.0,
        'mermin_quantum_bound': 4.0,
        'E_XXY': round(float(E_XXY), 4),
        'E_XYX': round(float(E_XYX), 4),
        'E_YXX': round(float(E_YXX), 4),
        'E_YYY': round(float(E_YYY), 4),
        'config_E': {c[0]: round(c[1], 4) for c in config_E},
        'ghz_amplitude': round(float((ghz_same.max() - ghz_same.min()) / 2), 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q38_ghz_state.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q38 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
