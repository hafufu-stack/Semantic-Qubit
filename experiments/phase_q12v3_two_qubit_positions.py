# -*- coding: utf-8 -*-
"""
Phase Q12v3: Two-Qubit S-Gate (Separate Token Positions)

Q12/Q12v2 failure analysis:
  - Q12:   SQ2@L16 replaces same position as SQ1@L8 -> wipes out SQ1
  - Q12v2: Additive injection with alpha=0.5 -> too weak (~1% of hidden state)

Q12v3 Correct Design:
  SQ1 at L8:  inject into position -1 (last token)    REPLACEMENT
  SQ2 at L16: inject into position -2 (2nd-to-last)   REPLACEMENT

  L16's injection at position -2 does NOT overwrite L8's injection at position -1!
  The self-attention mechanism at L16-L27 allows position -2 to influence position -1.

  This creates a true 2-qubit system:
    - SQ1 is encoded in position -1 at L8
    - SQ2 is encoded in position -2 at L16
    - Self-attention couples the two positions (analogous to qubit-qubit interaction)
    - Output at position -1 reflects both qubits and their coupling

Entanglement signature:
  If E(phi1, phi2) != E1(phi1) * E2_marginal -> non-separable (entangled)
  SQ2 varying phi2 while SQ1 is fixed -> does P(min_at_pos-1) change?
  Yes -> coupling exists -> qubit interaction detected!
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
SQ2_LAYER = 16
N_PHI = 17   # 17x17 = 289 joint measurements


def train_soul(model, tok, data, device, layer, pos=-1, epochs=100, seed=42):
    """Train soul vector, injecting at given layer and token position."""
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


def two_qubit_pos(model, tok, prompt, device, vec1, vec2,
                   layer1, pos1, layer2, pos2):
    """Two separate position injections in one forward pass."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    p1 = pos1 if pos1 >= 0 else seq_len + pos1
    p2 = pos2 if pos2 >= 0 else seq_len + pos2

    def hook1(m, i, o, v=vec1, p=p1):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    def hook2(m, i, o, v=vec2, p=p2):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    h1 = model.model.layers[layer1].register_forward_hook(hook1)
    h2 = model.model.layers[layer2].register_forward_hook(hook2)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def single_pos(model, tok, prompt, device, vec, layer, pos):
    """Single injection at given position."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    p = pos if pos >= 0 else seq_len + pos
    def hook(m, i, o, v=vec, pp=p):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, pp, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0:
        v = v / n * v0.norm()
    return v


def main():
    print("[Q12v3] Two-Qubit S-Gate (Separate Token Positions)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print("  SQ1@L%d pos=-1, SQ2@L%d pos=-2" % (SQ1_LAYER, SQ2_LAYER))

    prompt = "min(7,2)="
    seq_len = tok(prompt, return_tensors='pt')['input_ids'].shape[1]
    print("  Prompt '%s' -> %d tokens (pos-1=%d, pos-2=%d)" % (
        prompt, seq_len, seq_len-1, seq_len-2))

    sq1_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                  ("min(4,6)=","4"),("min(9,3)=","3")]
    sq1_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                  ("min(4,6)=","6"),("min(9,3)=","9")]
    sq2_0_data = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                  ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_1_data = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                  ("The forest is","green"),("Grass color is","green")]

    print("  Training SQ1 at L%d pos=-1..." % SQ1_LAYER)
    sq1_min = train_soul(model, tok, sq1_0_data, DEVICE, SQ1_LAYER, pos=-1, epochs=100, seed=42)
    sq1_max = train_soul(model, tok, sq1_1_data, DEVICE, SQ1_LAYER, pos=-1, epochs=100, seed=99)

    print("  Training SQ2 at L%d pos=-2..." % SQ2_LAYER)
    sq2_0 = train_soul(model, tok, sq2_0_data, DEVICE, SQ2_LAYER, pos=-2, epochs=100, seed=42)
    sq2_1 = train_soul(model, tok, sq2_1_data, DEVICE, SQ2_LAYER, pos=-2, epochs=100, seed=99)

    sq1_tok = tok.encode("2")[-1]
    sq1_tok_1 = tok.encode("7")[-1]

    # 1. Single-qubit marginals (separate positions, no coupling)
    print("  Single-qubit marginals...")
    phis_1d = np.linspace(0, 4 * np.pi, 37)
    E1_phi, E2_phi = [], []
    for phi in phis_1d:
        v1 = phi_vec(phi, sq1_min, sq1_max)
        p1 = single_pos(model, tok, prompt, DEVICE, v1, SQ1_LAYER, -1)
        E1_phi.append(float(p1[sq1_tok]) - float(p1[sq1_tok_1]))

        v2 = phi_vec(phi, sq2_0, sq2_1)
        # SQ2 at pos=-2, measure output at pos=-1
        # Does changing pos-2 affect pos-1?
        p2 = single_pos(model, tok, prompt, DEVICE, v2, SQ2_LAYER, -2)
        E2_phi.append(float(p2[sq1_tok]) - float(p2[sq1_tok_1]))

    E1_phi = np.array(E1_phi)
    E2_phi = np.array(E2_phi)
    amp1 = (E1_phi.max() - E1_phi.min()) / 2
    amp2 = (E2_phi.max() - E2_phi.min()) / 2
    print("    SQ1 amp=%.4f  SQ2 cross-coupling amp=%.4f" % (amp1, amp2))
    print("    (SQ2 amp>0 -> pos-2 injection affects pos-1 output via attention!)")

    # 2. 2D joint phi sweep
    print("  2D joint sweep (%dx%d)..." % (N_PHI, N_PHI))
    phis_2d = np.linspace(0, 2 * np.pi, N_PHI)
    joint_E = np.zeros((N_PHI, N_PHI))
    for i, phi1 in enumerate(phis_2d):
        for j, phi2 in enumerate(phis_2d):
            v1 = phi_vec(phi1, sq1_min, sq1_max)
            v2 = phi_vec(phi2, sq2_0, sq2_1)
            probs = two_qubit_pos(model, tok, prompt, DEVICE,
                                   v1, v2, SQ1_LAYER, -1, SQ2_LAYER, -2)
            joint_E[i, j] = float(probs[sq1_tok]) - float(probs[sq1_tok_1])

    print("    Joint E range: [%.4f, %.4f]" % (joint_E.min(), joint_E.max()))

    # Separability test: E_joint vs E1(phi1) * E2(phi2)
    E1_2d = np.array([((E1_phi[k] + 1)/2) for k in range(N_PHI)])  # rough marginal
    E2_2d = np.array([((E2_phi[k] + 1)/2) for k in range(N_PHI)])
    product_approx = np.outer(
        E1_phi[:N_PHI] / (abs(E1_phi[:N_PHI]).max() + 1e-6),
        E2_phi[:N_PHI] / (abs(E2_phi[:N_PHI]).max() + 1e-6)
    ) * abs(joint_E).max()
    residual = joint_E - product_approx
    residual_norm = np.std(residual)
    print("    Separability residual std=%.6f" % residual_norm)

    # SQ2 cross-coupling: fix SQ1=|0>, sweep SQ2
    print("  SQ2 cross-coupling on SQ1 readout (SQ1=|0> fixed)...")
    phi2_sweep = np.linspace(0, 2 * np.pi, 25)
    cross_E_sq1fixed = []
    for phi2 in phi2_sweep:
        v2 = phi_vec(phi2, sq2_0, sq2_1)
        probs = two_qubit_pos(model, tok, prompt, DEVICE,
                               sq1_min, v2, SQ1_LAYER, -1, SQ2_LAYER, -2)
        cross_E_sq1fixed.append(float(probs[sq1_tok]) - float(probs[sq1_tok_1]))
    cross_E_sq1fixed = np.array(cross_E_sq1fixed)
    cross_amp_sq1fixed = (cross_E_sq1fixed.max() - cross_E_sq1fixed.min()) / 2
    print("    SQ2->SQ1 coupling amplitude (SQ1=|0>): %.4f" % cross_amp_sq1fixed)

    # CHSH from joint sweep
    E00 = joint_E[0, 0]
    E01 = joint_E[0, N_PHI//4]
    E10 = joint_E[N_PHI//4, 0]
    E11 = joint_E[N_PHI//4, N_PHI//4]
    S_joint = abs(E00 - E01 + E10 + E11)
    print("  2-Qubit CHSH S = %.4f" % S_joint)

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    phis_plot = np.linspace(0, 4 * np.pi, 37)
    ax.plot(phis_plot / np.pi, E1_phi, '#E91E63', lw=2, label='SQ1@L%d pos=-1 (amp=%.3f)' % (SQ1_LAYER, amp1))
    ax.plot(phis_plot / np.pi, E2_phi, '#2196F3', lw=2, label='SQ2@L%d pos=-2 on pos-1 (amp=%.3f)' % (SQ2_LAYER, amp2), linestyle='--')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Phase phi (x pi)', fontsize=11)
    ax.set_ylabel('E = P(min) - P(max)', fontsize=11)
    ax.set_title('Single-Qubit Marginals\nSQ2 at pos-2 leaks to pos-1 via attention', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    vmax = max(abs(joint_E.min()), abs(joint_E.max()), 0.01)
    im = ax.imshow(joint_E, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax,
                   extent=[0, 2, 0, 2])
    plt.colorbar(im, ax=ax, label='E(phi1,phi2)')
    ax.set_xlabel('phi2 SQ2 (x pi)', fontsize=11)
    ax.set_ylabel('phi1 SQ1 (x pi)', fontsize=11)
    ax.set_title('2D Joint Interference\nSQ1@pos-1 x SQ2@pos-2', fontweight='bold')

    ax = axes[2]
    phi2_plot = np.linspace(0, 2 * np.pi, 25)
    ax.plot(phi2_plot / np.pi, cross_E_sq1fixed, '#9C27B0', lw=2, marker='o', ms=5)
    ax.axhline(cross_E_sq1fixed.mean(), color='red', linestyle='--', lw=1.5,
               label='Mean=%.4f' % cross_E_sq1fixed.mean())
    ax.fill_between(phi2_plot / np.pi, cross_E_sq1fixed, cross_E_sq1fixed.mean(),
                    alpha=0.2, color='#9C27B0')
    ax.set_xlabel('phi2 (SQ2 angle, x pi)', fontsize=11)
    ax.set_ylabel('E at pos-1 (SQ1 fixed to |0>)', fontsize=11)
    ax.set_title('Cross-Qubit Coupling\namp=%.4f (via self-attention)' % cross_amp_sq1fixed, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle(
        'Phase Q12v3: Two-Qubit S-Gate (Separate Token Positions)\n'
        'SQ1@L%d,pos=-1  x  SQ2@L%d,pos=-2  |  CHSH S=%.4f' % (SQ1_LAYER, SQ2_LAYER, S_joint),
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q12v3_two_qubit_positions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q12v3', 'name': 'two_qubit_separate_positions',
        'sq1_layer': SQ1_LAYER, 'sq1_pos': -1,
        'sq2_layer': SQ2_LAYER, 'sq2_pos': -2,
        'sq1_amp': round(float(amp1), 6),
        'sq2_cross_amp': round(float(amp2), 6),
        'cross_amp_sq1fixed': round(float(cross_amp_sq1fixed), 6),
        'joint_E_min': round(float(joint_E.min()), 4),
        'joint_E_max': round(float(joint_E.max()), 4),
        'separability_residual_std': round(float(residual_norm), 6),
        'joint_S_chsh': round(float(S_joint), 6),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q12v3_two_qubit_positions.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q12v3 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
