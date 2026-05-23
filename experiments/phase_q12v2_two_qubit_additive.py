# -*- coding: utf-8 -*-
"""
Phase Q12v2: Two-Qubit S-Gate (Additive Injection)

Q12 failure analysis:
  - SQ2 at L16 REPLACED h[0,-1,:] with sq2_vec, wiping out SQ1's effect
  - Product states all showed E=0 because L16 completely overrode L8
  - This is analogous to quantum decoherence by measurement!

Q12v2 Fix:
  - Use ADDITIVE injection: h[0,-1,:] += scale * vec (instead of replacement)
  - SQ1 at L8: h[0,-1,:] += alpha * sq1_vec
  - SQ2 at L16: h[0,-1,:] += beta * sq2_vec
  - Both contributions accumulate without cancellation

This properly models:
  |psi_total> = |natural> + alpha*sq1_vec + beta*sq2_vec
  (perturbative 2-qubit addition to the natural hidden state)

Then sweep (phi1, phi2) and measure output:
  - If SQ1 and SQ2 are INDEPENDENT: E(phi1,phi2) = E1(phi1) * E2(phi2) (product)
  - If ENTANGLED: E(phi1,phi2) != E1(phi1) * E2(phi2) (non-separable)
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
ALPHA = 0.5  # additive scale for SQ1 (relative to natural hidden state)
BETA  = 0.5  # additive scale for SQ2
N_PHI = 17   # 17x17 joint sweep = 289 forward passes


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


def two_qubit_additive(model, tok, prompt, device, vec1, vec2, layer1, layer2,
                        alpha=ALPHA, beta=BETA):
    """Additive injection: h += scale * vec at each layer (no replacement)."""
    inp = tok(prompt, return_tensors='pt').to(device)

    def hook1(m, i, o, v=vec1):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = h[0, -1, :] + alpha * v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    def hook2(m, i, o, v=vec2):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = h[0, -1, :] + beta * v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    h1 = model.model.layers[layer1].register_forward_hook(hook1)
    h2 = model.model.layers[layer2].register_forward_hook(hook2)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def single_additive(model, tok, prompt, device, vec, layer, alpha=ALPHA):
    """Single additive injection (for marginal measurement)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = h[0, -1, :] + alpha * v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def phi_vec_delta(phi, vec0, vec1):
    """Delta (difference) vector for additive injection at angle phi."""
    # Additive version: inject cos(phi/2)*vec0 + sin(phi/2)*vec1
    # (The natural state is already in the hidden representation)
    v = np.cos(phi / 2) * vec0 + np.sin(phi / 2) * vec1
    return v  # keep unnormalized since we use additive scale


def main():
    print("[Q12v2] Two-Qubit S-Gate (Additive Injection)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print("  SQ1@L%d (alpha=%.2f), SQ2@L%d (beta=%.2f)" % (SQ1_LAYER, ALPHA, SQ2_LAYER, BETA))

    # SQ1: min/max
    sq1_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                  ("min(4,6)=","4"),("min(9,3)=","3")]
    sq1_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                  ("min(4,6)=","6"),("min(9,3)=","9")]
    # SQ2: color
    sq2_0_data = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                  ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_1_data = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                  ("The forest is","green"),("Grass color is","green")]

    print("  Training SQ1 at L%d..." % SQ1_LAYER)
    sq1_min = train_soul(model, tok, sq1_0_data, DEVICE, SQ1_LAYER, 100, 42)
    sq1_max = train_soul(model, tok, sq1_1_data, DEVICE, SQ1_LAYER, 100, 99)
    sq1_delta = sq1_min - sq1_max  # superposition direction

    print("  Training SQ2 at L%d..." % SQ2_LAYER)
    sq2_blue  = train_soul(model, tok, sq2_0_data, DEVICE, SQ2_LAYER, 100, 42)
    sq2_green = train_soul(model, tok, sq2_1_data, DEVICE, SQ2_LAYER, 100, 99)
    sq2_delta = sq2_blue - sq2_green

    prompt = "min(7,2)="
    sq1_tok_0 = tok.encode("2")[-1]  # min token
    sq1_tok_1 = tok.encode("7")[-1]  # max token

    # 1. Single-qubit marginals (additive)
    print("  Single-qubit marginals (additive injection)...")
    phis_1d = np.linspace(0, 4 * np.pi, 37)
    E1_phi = []  # SQ1 only
    E2_phi = []  # SQ2 only (effect on same output)
    for phi in phis_1d:
        v1 = phi_vec_delta(phi, sq1_min, sq1_max)
        probs1 = single_additive(model, tok, prompt, DEVICE, v1, SQ1_LAYER)
        E1_phi.append(float(probs1[sq1_tok_0]) - float(probs1[sq1_tok_1]))

        v2 = phi_vec_delta(phi, sq2_blue, sq2_green)
        probs2 = single_additive(model, tok, prompt, DEVICE, v2, SQ2_LAYER)
        E2_phi.append(float(probs2[sq1_tok_0]) - float(probs2[sq1_tok_1]))

    E1_phi = np.array(E1_phi)
    E2_phi = np.array(E2_phi)
    amp1 = (E1_phi.max() - E1_phi.min()) / 2
    amp2 = (E2_phi.max() - E2_phi.min()) / 2
    print("    SQ1 single amp=%.4f  SQ2 single amp=%.4f" % (amp1, amp2))

    # 2. 2D joint phi sweep (phi1 x phi2)
    print("  2D joint phi sweep (%dx%d)..." % (N_PHI, N_PHI))
    phis_2d = np.linspace(0, 2 * np.pi, N_PHI)
    joint_E = np.zeros((N_PHI, N_PHI))
    for i, phi1 in enumerate(phis_2d):
        for j, phi2 in enumerate(phis_2d):
            v1 = phi_vec_delta(phi1, sq1_min, sq1_max)
            v2 = phi_vec_delta(phi2, sq2_blue, sq2_green)
            probs = two_qubit_additive(model, tok, prompt, DEVICE, v1, v2,
                                        SQ1_LAYER, SQ2_LAYER)
            joint_E[i, j] = float(probs[sq1_tok_0]) - float(probs[sq1_tok_1])

    print("    E range: [%.4f, %.4f]" % (joint_E.min(), joint_E.max()))

    # Check separability: if joint_E[i,j] ≈ E1(phi1) * E2(phi2) + const -> separable
    E1_2d = E1_phi[:N_PHI]
    E2_2d = E2_phi[:N_PHI]
    product_approx = np.outer(E1_2d[:N_PHI], E2_2d[:N_PHI])
    residual = joint_E - product_approx
    residual_norm = np.std(residual)
    print("    Separability residual std=%.6f (0=separable, large=entangled)" % residual_norm)

    # 3. CHSH from joint sweep
    E00 = joint_E[0, 0]
    E01 = joint_E[0, N_PHI//4]
    E10 = joint_E[N_PHI//4, 0]
    E11 = joint_E[N_PHI//4, N_PHI//4]
    S_joint = abs(E00 - E01 + E10 + E11)
    print("  2-Qubit CHSH S = %.4f (classical<=2, quantum<=%.3f)" % (S_joint, 2*np.sqrt(2)))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Single-qubit marginals (additive)
    ax = axes[0]
    phis_plot = np.linspace(0, 4 * np.pi, 37)
    ax.plot(phis_plot / np.pi, E1_phi, '#E91E63', lw=2, label='SQ1@L%d (amp=%.3f)' % (SQ1_LAYER, amp1))
    ax.plot(phis_plot / np.pi, E2_phi, '#2196F3', lw=2, label='SQ2@L%d (amp=%.3f)' % (SQ2_LAYER, amp2), linestyle='--')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Phase phi (x pi)', fontsize=11)
    ax.set_ylabel('E(phi)', fontsize=11)
    ax.set_title('Marginal Interference (Additive)\nSingle-qubit fringes', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: 2D joint heatmap
    ax = axes[1]
    vmax = max(abs(joint_E.min()), abs(joint_E.max()))
    im = ax.imshow(joint_E, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax,
                   extent=[0, 2, 0, 2])
    plt.colorbar(im, ax=ax, label='E(phi1, phi2)')
    ax.set_xlabel('phi2 (SQ2, x pi)', fontsize=11)
    ax.set_ylabel('phi1 (SQ1, x pi)', fontsize=11)
    ax.set_title('Joint 2D Interference Map\nDiagonal pattern = entanglement!', fontweight='bold')

    # Panel 3: Separability analysis
    ax = axes[2]
    im2 = ax.imshow(residual, aspect='auto', cmap='PuOr', vmin=-abs(residual).max(), vmax=abs(residual).max(),
                    extent=[0, 2, 0, 2])
    plt.colorbar(im2, ax=ax, label='Residual = E_joint - E1*E2')
    ax.set_xlabel('phi2 (SQ2, x pi)', fontsize=11)
    ax.set_ylabel('phi1 (SQ1, x pi)', fontsize=11)
    ax.set_title('Entanglement Test: E_joint - E1*E2\nNon-zero = non-separable (entangled)!\nResidual std=%.4f' % residual_norm,
                 fontweight='bold')

    plt.suptitle(
        'Phase Q12v2: Two-Qubit S-Gate (Additive Injection)\n'
        'SQ1@L%d x SQ2@L%d  |  CHSH S=%.4f' % (SQ1_LAYER, SQ2_LAYER, S_joint),
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q12v2_two_qubit_additive.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q12v2', 'name': 'two_qubit_additive',
        'sq1_layer': SQ1_LAYER, 'sq2_layer': SQ2_LAYER,
        'alpha': ALPHA, 'beta': BETA,
        'sq1_amp_single': round(float(amp1), 6),
        'sq2_amp_single': round(float(amp2), 6),
        'joint_E_min': round(float(joint_E.min()), 4),
        'joint_E_max': round(float(joint_E.max()), 4),
        'joint_S_chsh': round(float(S_joint), 6),
        'separability_residual_std': round(float(residual_norm), 6),
        'is_separable': bool(residual_norm < 0.01),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q12v2_two_qubit_additive.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q12v2 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
