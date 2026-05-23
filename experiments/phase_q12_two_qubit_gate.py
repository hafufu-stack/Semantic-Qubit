# -*- coding: utf-8 -*-
"""
Phase Q12: Two-Qubit S-Gate (Neural Quantum Entanglement)

Define TWO independent S-Qubits:
  SQ1 at Layer 8  (L8):  |MIN1> / |MAX1> from min/max task
  SQ2 at Layer 16 (L16): |MIN2> / |MAX2> from a DIFFERENT domain

Create superposition of each, inject simultaneously.
Measure correlations between output probabilities to detect entanglement.

Entanglement criterion:
  If P(outcome1 | outcome2) != P(outcome1) -> correlated -> entangled
  CHSH-like: S12 > 2 for joint measurements

Operations:
  1. Product state: |00>, |01>, |10>, |11>  (inject independently)
  2. Bell state: (|00> + |11>) / sqrt(2)   (both in superposition)
  3. CNOT-analog: flip SQ2 based on SQ1 state
  4. Joint phi sweep: vary phi1 and phi2, measure joint P(min1, min2)

This simulates a 2-qubit quantum circuit in neural activation space.
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

SQ1_LAYER = 8   # First qubit injection layer
SQ2_LAYER = 16  # Second qubit injection layer
N_PHI = 13      # Coarser grid for 2D joint sweep (13x13=169 points)


def train_soul(model, tok, data, device, layer, epochs=100, seed=42):
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
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def two_qubit_forward(model, tok, prompt, device, vec1, vec2, layer1, layer2):
    """Inject vec1 at layer1, vec2 at layer2 in a single forward pass."""
    inp = tok(prompt, return_tensors='pt').to(device)

    def hook1(m, i, o, v=vec1):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h

    def hook2(m, i, o, v=vec2):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h

    h1 = model.model.layers[layer1].register_forward_hook(hook1)
    h2 = model.model.layers[layer2].register_forward_hook(hook2)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return probs


def phi_vec(phi, vec0, vec1):
    v = np.cos(phi / 2) * vec0 + np.sin(phi / 2) * vec1
    n = v.norm()
    if n > 0:
        v = v / n * vec0.norm()
    return v


def main():
    print("[Q12] Two-Qubit S-Gate: Neural Quantum Entanglement")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print("  SQ1@L%d, SQ2@L%d" % (SQ1_LAYER, SQ2_LAYER))

    # SQ1: min/max task (same as Q1-Q11)
    sq1_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                  ("min(4,6)=","4"),("min(9,3)=","3")]
    sq1_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                  ("min(4,6)=","6"),("min(9,3)=","9")]

    # SQ2: color task (different domain → different basis vectors)
    sq2_0_data = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                  ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_1_data = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                  ("The forest is","green"),("Grass color is","green")]

    print("  Training SQ1 basis (L%d, 100 epochs)..." % SQ1_LAYER)
    sq1_min = train_soul(model, tok, sq1_0_data, DEVICE, SQ1_LAYER, 100, 42)
    sq1_max = train_soul(model, tok, sq1_1_data, DEVICE, SQ1_LAYER, 100, 99)

    print("  Training SQ2 basis (L%d, 100 epochs)..." % SQ2_LAYER)
    sq2_blue  = train_soul(model, tok, sq2_0_data, DEVICE, SQ2_LAYER, 100, 42)
    sq2_green = train_soul(model, tok, sq2_1_data, DEVICE, SQ2_LAYER, 100, 99)

    prompt = "min(7,2)="  # Use min/max prompt to observe SQ1 outcome
    sq1_tok_min = tok.encode("2")[-1]
    sq1_tok_max = tok.encode("7")[-1]

    print("  Basis trained. Computing product states...")

    # 1. Product states |00>, |01>, |10>, |11>
    product_states = {
        '|00>': (sq1_min, sq2_blue),    # |MIN>|BLUE>
        '|01>': (sq1_min, sq2_green),   # |MIN>|GREEN>
        '|10>': (sq1_max, sq2_blue),    # |MAX>|BLUE>
        '|11>': (sq1_max, sq2_green),   # |MAX>|GREEN>
    }
    product_results = {}
    for state_name, (v1, v2) in product_states.items():
        probs = two_qubit_forward(model, tok, prompt, DEVICE, v1, v2, SQ1_LAYER, SQ2_LAYER)
        p_min = float(probs[sq1_tok_min])
        p_max = float(probs[sq1_tok_max])
        product_results[state_name] = {'P_min': round(p_min, 4), 'P_max': round(p_max, 4),
                                        'E': round(p_min - p_max, 4)}
        print("    %s: P(min)=%.4f  P(max)=%.4f  E=%.4f" % (
            state_name, p_min, p_max, p_min - p_max))

    # 2. Bell state (|00> + |11>) / sqrt(2)
    print("  Computing Bell state |Phi+> = (|00>+|11>)/sqrt2...")
    v1_bell = (sq1_min + sq1_max) / np.sqrt(2)
    n1 = v1_bell.norm()
    if n1 > 0: v1_bell = v1_bell / n1 * sq1_min.norm()
    v2_bell = (sq2_blue + sq2_green) / np.sqrt(2)
    n2 = v2_bell.norm()
    if n2 > 0: v2_bell = v2_bell / n2 * sq2_blue.norm()

    probs_bell = two_qubit_forward(model, tok, prompt, DEVICE, v1_bell, v2_bell,
                                    SQ1_LAYER, SQ2_LAYER)
    p_min_bell = float(probs_bell[sq1_tok_min])
    p_max_bell = float(probs_bell[sq1_tok_max])
    bell_E = p_min_bell - p_max_bell
    print("    |Phi+>: P(min)=%.4f  P(max)=%.4f  E=%.4f" % (p_min_bell, p_max_bell, bell_E))

    # 3. Joint phi sweep: phi1 x phi2 → 2D interference map
    print("  2D joint phi sweep (%dx%d)..." % (N_PHI, N_PHI))
    phis = np.linspace(0, 2 * np.pi, N_PHI)
    joint_E = np.zeros((N_PHI, N_PHI))
    for i, phi1 in enumerate(phis):
        for j, phi2 in enumerate(phis):
            v1 = phi_vec(phi1, sq1_min, sq1_max)
            v2 = phi_vec(phi2, sq2_blue, sq2_green)
            probs = two_qubit_forward(model, tok, prompt, DEVICE, v1, v2,
                                       SQ1_LAYER, SQ2_LAYER)
            joint_E[i, j] = float(probs[sq1_tok_min]) - float(probs[sq1_tok_max])
    print("    Joint sweep done. E range: [%.4f, %.4f]" % (joint_E.min(), joint_E.max()))

    # Compute 2-qubit CHSH
    # E(phi1, phi2) and test CHSH at 4 angle pairs
    # Standard CHSH: S = |E(0,0) - E(0,pi/2) + E(pi/2,0) + E(pi/2,pi/2)|
    phi1_idx = {0: 0, int(N_PHI//4): int(N_PHI//4)}
    E00 = joint_E[0, 0]
    E01 = joint_E[0, N_PHI//4]
    E10 = joint_E[N_PHI//4, 0]
    E11 = joint_E[N_PHI//4, N_PHI//4]
    S_joint = abs(E00 - E01 + E10 + E11)
    print("  2-Qubit CHSH S = %.4f (classical<=2, quantum<=%.3f)" % (
        S_joint, 2*np.sqrt(2)))

    # 4. Check SQ2 influence on SQ1 (cross-qubit correlation)
    # Vary phi2 with SQ1 fixed at |+> = (|0>+|1>)/sqrt2
    sq1_plus = (sq1_min + sq1_max) / np.sqrt(2)
    n = sq1_plus.norm()
    if n > 0: sq1_plus = sq1_plus / n * sq1_min.norm()

    print("  Cross-qubit influence (SQ1=|+>, vary SQ2)...")
    phi2_sweep = np.linspace(0, 2 * np.pi, 25)
    cross_E = []
    for phi2 in phi2_sweep:
        v2 = phi_vec(phi2, sq2_blue, sq2_green)
        probs = two_qubit_forward(model, tok, prompt, DEVICE, sq1_plus, v2,
                                   SQ1_LAYER, SQ2_LAYER)
        cross_E.append(float(probs[sq1_tok_min]) - float(probs[sq1_tok_max]))

    cross_E = np.array(cross_E)
    cross_amp = (cross_E.max() - cross_E.min()) / 2.0
    print("  SQ2->SQ1 influence amplitude: %.4f" % cross_amp)

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Product states vs Bell state
    ax = axes[0]
    state_names = list(product_results.keys()) + ['|Phi+>']
    E_vals = [product_results[s]['E'] for s in product_results] + [bell_E]
    bar_colors = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF5722']
    bars = ax.bar(range(len(state_names)), E_vals, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    for bar, val in zip(bars, E_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.02 if val >= 0 else -0.05),
                '%.3f' % val, ha='center', fontsize=10)
    ax.set_xticks(range(len(state_names)))
    ax.set_xticklabels(state_names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('E = P(min) - P(max)', fontsize=11)
    ax.set_title('2-Qubit Product States vs Bell State\n'
                 'E should differ between product & Bell', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: 2D Joint phi sweep heatmap
    ax = axes[1]
    im = ax.imshow(joint_E, aspect='auto', cmap='RdBu', vmin=-1, vmax=1,
                   extent=[0, 2, 0, 2])
    plt.colorbar(im, ax=ax, label='E(phi1, phi2)')
    ax.set_xlabel('phi2 (SQ2 angle, x pi)', fontsize=11)
    ax.set_ylabel('phi1 (SQ1 angle, x pi)', fontsize=11)
    ax.set_title('2D Joint Interference Map\n'
                 'Entanglement: diagonal structure!', fontweight='bold')

    # Panel 3: SQ2 influence on SQ1
    ax = axes[2]
    ax.plot(phi2_sweep / np.pi, cross_E, '#9C27B0', lw=2, marker='o', ms=5)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(cross_E.mean(), color='red', linestyle='--', lw=1.5,
               label='mean=%.4f' % cross_E.mean())
    ax.fill_between(phi2_sweep / np.pi, cross_E, cross_E.mean(),
                    alpha=0.2, color='#9C27B0')
    ax.set_xlabel('phi2 (SQ2 angle, x pi)', fontsize=11)
    ax.set_ylabel('E(SQ1) = P(min1) - P(max1)', fontsize=11)
    ax.set_title('SQ2 Influence on SQ1\namp=%.4f (0=no influence)' % cross_amp,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle(
        'Phase Q12: Two-Qubit S-Gate\n'
        'SQ1@L%d (min/max) x SQ2@L%d (blue/green)' % (SQ1_LAYER, SQ2_LAYER),
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q12_two_qubit_gate.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q12', 'name': 'two_qubit_gate',
        'sq1_layer': SQ1_LAYER, 'sq2_layer': SQ2_LAYER,
        'product_states': product_results,
        'bell_state': {'P_min': round(p_min_bell, 4), 'P_max': round(p_max_bell, 4),
                       'E': round(bell_E, 4)},
        'joint_S_chsh': round(S_joint, 6),
        'cross_qubit_amplitude': round(float(cross_amp), 6),
        'joint_E_min': round(float(joint_E.min()), 4),
        'joint_E_max': round(float(joint_E.max()), 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q12_two_qubit_gate.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q12 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
