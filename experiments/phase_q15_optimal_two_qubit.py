# -*- coding: utf-8 -*-
"""
Phase Q15: Full 2-Qubit Bell Test at Optimal Configuration (SQ1@L8, SQ2@L20)

Q14 finding: PEAK cross-coupling at SQ2@L20 (dist=12, amp=0.8194)
  - L20 is 2 layers BEFORE the wavefunction collapse zone (L22-L26)
  - This is the "pre-collapse sweet spot" for 2-qubit interaction

Q15: Complete 2-qubit characterization at the optimal L8×L20 configuration:
  1. Full 2D phi sweep (phi1 x phi2): 17x17 = 289 measurements
  2. Proper CHSH Bell test at optimal angles
  3. Entanglement verification (separability residual)
  4. "CNOT-like" gate: E(SQ1 | SQ2=|0>) vs E(SQ1 | SQ2=|1>)
  5. Quantum state tomography of joint 2D interference map

Expected: if cross_amp=0.82, then E(phi1,phi2) shows strong 2D modulation
with residual >> 0.27 (Q12v3 result), supporting non-separability.
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
SQ2_LAYER = 20  # optimal from Q14
SQ1_POS   = -1
SQ2_POS   = -2
N_PHI_2D  = 21   # 21x21 = 441 joint measurements (finer than Q12v3's 17x17)
N_PHI_1D  = 37


def train_soul(model, tok, data, device, layer, pos=-1, epochs=100, seed=42):
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


def two_qubit_forward(model, tok, prompt, device, v1, v2, l1, p1, l2, p2):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    a1 = p1 if p1 >= 0 else seq_len + p1
    a2 = p2 if p2 >= 0 else seq_len + p2
    def hook1(m, i, o, v=v1, p=a1):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    def hook2(m, i, o, v=v2, p=a2):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    h1 = model.model.layers[l1].register_forward_hook(hook1)
    h2 = model.model.layers[l2].register_forward_hook(hook2)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def single_forward(model, tok, prompt, device, v, layer, pos):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    p = pos if pos >= 0 else seq_len + pos
    def hook(m, i, o, vv=v, pp=p):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, pp, :] = vv.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def main():
    print("[Q15] Full 2-Qubit Bell Test: SQ1@L%d,pos=%d x SQ2@L%d,pos=%d" % (
        SQ1_LAYER, SQ1_POS, SQ2_LAYER, SQ2_POS))
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    prompt = "min(7,2)="
    sq1_tok = tok.encode("2")[-1]
    sq1_tok_1 = tok.encode("7")[-1]

    sq1_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                  ("min(4,6)=","4"),("min(9,3)=","3")]
    sq1_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                  ("min(4,6)=","6"),("min(9,3)=","9")]
    sq2_0_data = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                  ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_1_data = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                  ("The forest is","green"),("Grass color is","green")]

    print("  Training SQ1@L%d pos=%d (150 epochs)..." % (SQ1_LAYER, SQ1_POS))
    sq1_min = train_soul(model, tok, sq1_0_data, DEVICE, SQ1_LAYER, SQ1_POS, 150, 42)
    sq1_max = train_soul(model, tok, sq1_1_data, DEVICE, SQ1_LAYER, SQ1_POS, 150, 99)

    print("  Training SQ2@L%d pos=%d (150 epochs)..." % (SQ2_LAYER, SQ2_POS))
    sq2_0 = train_soul(model, tok, sq2_0_data, DEVICE, SQ2_LAYER, SQ2_POS, 150, 42)
    sq2_1 = train_soul(model, tok, sq2_1_data, DEVICE, SQ2_LAYER, SQ2_POS, 150, 99)

    # cos similarity between basis pairs
    cos_sq1 = float(torch.nn.functional.cosine_similarity(sq1_min.unsqueeze(0), sq1_max.unsqueeze(0)))
    cos_sq2 = float(torch.nn.functional.cosine_similarity(sq2_0.unsqueeze(0), sq2_1.unsqueeze(0)))
    print("  cos(SQ1|0>,SQ1|1>) = %.4f  cos(SQ2|0>,SQ2|1>) = %.4f" % (cos_sq1, cos_sq2))

    # 1. Single-qubit marginals
    print("  Single-qubit marginals (%d phi points)..." % N_PHI_1D)
    phis_1d = np.linspace(0, 4 * np.pi, N_PHI_1D)
    E1_phi, E2_phi = [], []
    for phi in phis_1d:
        v1 = phi_vec(phi, sq1_min, sq1_max)
        p1 = single_forward(model, tok, prompt, DEVICE, v1, SQ1_LAYER, SQ1_POS)
        E1_phi.append(float(p1[sq1_tok]) - float(p1[sq1_tok_1]))

        v2 = phi_vec(phi, sq2_0, sq2_1)
        p2 = single_forward(model, tok, prompt, DEVICE, v2, SQ2_LAYER, SQ2_POS)
        E2_phi.append(float(p2[sq1_tok]) - float(p2[sq1_tok_1]))

    E1_phi = np.array(E1_phi)
    E2_phi = np.array(E2_phi)
    amp1 = (E1_phi.max() - E1_phi.min()) / 2
    amp2 = (E2_phi.max() - E2_phi.min()) / 2
    print("    SQ1 amp=%.4f  SQ2 cross-coupling amp=%.4f" % (amp1, amp2))

    # 2. CNOT-like gate: E(SQ1|phi1) with SQ2 fixed at |0> vs |1>
    print("  CNOT-like gate analysis...")
    E_sq1_cond0, E_sq1_cond1 = [], []
    for phi1 in phis_1d:
        v1 = phi_vec(phi1, sq1_min, sq1_max)
        # SQ2 = |0> (BLUE)
        p_c0 = two_qubit_forward(model, tok, prompt, DEVICE,
                                  v1, sq2_0, SQ1_LAYER, SQ1_POS, SQ2_LAYER, SQ2_POS)
        E_sq1_cond0.append(float(p_c0[sq1_tok]) - float(p_c0[sq1_tok_1]))
        # SQ2 = |1> (GREEN)
        p_c1 = two_qubit_forward(model, tok, prompt, DEVICE,
                                  v1, sq2_1, SQ1_LAYER, SQ1_POS, SQ2_LAYER, SQ2_POS)
        E_sq1_cond1.append(float(p_c1[sq1_tok]) - float(p_c1[sq1_tok_1]))

    E_sq1_cond0 = np.array(E_sq1_cond0)
    E_sq1_cond1 = np.array(E_sq1_cond1)
    # Conditional amplitudes
    amp_cond0 = (E_sq1_cond0.max() - E_sq1_cond0.min()) / 2
    amp_cond1 = (E_sq1_cond1.max() - E_sq1_cond1.min()) / 2
    # Shift in E baseline (SQ2 state changes SQ1 bias)
    bias_shift = float(E_sq1_cond0.mean() - E_sq1_cond1.mean())
    print("    E(SQ1 | SQ2=|0>) amp=%.4f  mean=%.4f" % (amp_cond0, E_sq1_cond0.mean()))
    print("    E(SQ1 | SQ2=|1>) amp=%.4f  mean=%.4f" % (amp_cond1, E_sq1_cond1.mean()))
    print("    CNOT bias shift: %.4f" % bias_shift)

    # 3. Full 2D joint phi sweep
    print("  2D joint phi sweep (%dx%d = %d points)..." % (N_PHI_2D, N_PHI_2D, N_PHI_2D**2))
    phis_2d = np.linspace(0, 2 * np.pi, N_PHI_2D)
    joint_E = np.zeros((N_PHI_2D, N_PHI_2D))
    for i, phi1 in enumerate(phis_2d):
        for j, phi2 in enumerate(phis_2d):
            v1 = phi_vec(phi1, sq1_min, sq1_max)
            v2 = phi_vec(phi2, sq2_0, sq2_1)
            probs = two_qubit_forward(model, tok, prompt, DEVICE,
                                       v1, v2, SQ1_LAYER, SQ1_POS, SQ2_LAYER, SQ2_POS)
            joint_E[i, j] = float(probs[sq1_tok]) - float(probs[sq1_tok_1])

    print("    Joint E range: [%.4f, %.4f]" % (joint_E.min(), joint_E.max()))

    # 4. Separability test
    # Compare to product approximation
    E1_2d = E1_phi[:N_PHI_2D]  # marginal SQ1
    E2_2d = E2_phi[:N_PHI_2D]  # marginal SQ2
    # Normalize to max 1
    E1_n = E1_2d / (abs(E1_2d).max() + 1e-6)
    E2_n = E2_2d / (abs(E2_2d).max() + 1e-6)
    product_approx = np.outer(E1_n, E2_n) * abs(joint_E).max()
    residual = joint_E - product_approx
    sep_residual = float(np.std(residual))
    print("    Separability residual std=%.6f (Q12v3 was 0.2687)" % sep_residual)

    # 5. CHSH grid search (4-angle CHSH from joint_E)
    best_S = 0
    for i1 in range(N_PHI_2D):
        for i2 in range(N_PHI_2D):
            for j1 in range(N_PHI_2D):
                for j2 in range(N_PHI_2D):
                    S = abs(joint_E[i1,j1] - joint_E[i1,j2] + joint_E[i2,j1] + joint_E[i2,j2])
                    if S > best_S:
                        best_S = S
                        best_idx = (i1, i2, j1, j2)
    print("    Best CHSH S = %.4f (classical<=2, quantum max=%.3f)" % (best_S, 2*np.sqrt(2)))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Conditional SQ1 (CNOT effect)
    ax = axes[0]
    ax.plot(phis_1d / np.pi, E_sq1_cond0, '#E91E63', lw=2,
            label='SQ2=|0>(BLUE)  amp=%.3f' % amp_cond0)
    ax.plot(phis_1d / np.pi, E_sq1_cond1, '#2196F3', lw=2, linestyle='--',
            label='SQ2=|1>(GREEN) amp=%.3f' % amp_cond1)
    ax.axhline(0, color='black', lw=0.8)
    ax.fill_between(phis_1d / np.pi, E_sq1_cond0, E_sq1_cond1,
                    alpha=0.15, color='gray')
    ax.set_xlabel('phi1 (SQ1 angle, x pi)', fontsize=11)
    ax.set_ylabel('E(SQ1)', fontsize=11)
    ax.set_title('CNOT-Like Gate\nSQ2 state controls SQ1 amplitude\nbias_shift=%.4f' % bias_shift,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: 2D joint heatmap
    ax = axes[1]
    vmax = max(abs(joint_E.min()), abs(joint_E.max()))
    im = ax.imshow(joint_E, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax,
                   extent=[0, 2, 0, 2])
    cb = plt.colorbar(im, ax=ax, label='E(phi1,phi2)')
    ax.set_xlabel('phi2 / pi (SQ2@L%d)' % SQ2_LAYER, fontsize=11)
    ax.set_ylabel('phi1 / pi (SQ1@L%d)' % SQ1_LAYER, fontsize=11)
    ax.set_title('2D Quantum Interference Map\nSQ1@L8,pos=-1 x SQ2@L20,pos=-2\n'
                 'CHSH S=%.4f  Sep.residual=%.4f' % (best_S, sep_residual),
                 fontweight='bold')

    # Panel 3: Separability residual heatmap
    ax = axes[2]
    vmax_r = abs(residual).max()
    im2 = ax.imshow(residual, aspect='auto', cmap='PuOr', vmin=-vmax_r, vmax=vmax_r,
                    extent=[0, 2, 0, 2])
    plt.colorbar(im2, ax=ax, label='Residual = E_joint - E1*E2')
    ax.set_xlabel('phi2 / pi (SQ2@L%d)' % SQ2_LAYER, fontsize=11)
    ax.set_ylabel('phi1 / pi (SQ1@L%d)' % SQ1_LAYER, fontsize=11)
    ax.set_title('Entanglement Test: E_joint - E1*E2\nNon-zero = Non-separable!\nstd=%.4f (Q12v3: 0.2687)' % sep_residual,
                 fontweight='bold')

    plt.suptitle(
        'Phase Q15: Optimal 2-Qubit Bell Test (SQ1@L%d,pos=-1  x  SQ2@L%d,pos=-2)\n'
        '"Pre-collapse sweet spot coupling: SQ2 at the edge of the wavefunction collapse zone"' % (SQ1_LAYER, SQ2_LAYER),
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q15_optimal_two_qubit.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q15', 'name': 'optimal_two_qubit_bell',
        'sq1_layer': SQ1_LAYER, 'sq1_pos': SQ1_POS,
        'sq2_layer': SQ2_LAYER, 'sq2_pos': SQ2_POS,
        'cos_sq1': round(cos_sq1, 4), 'cos_sq2': round(cos_sq2, 4),
        'sq1_amp': round(float(amp1), 6),
        'sq2_cross_amp': round(float(amp2), 6),
        'cnot_amp_cond0': round(float(amp_cond0), 6),
        'cnot_amp_cond1': round(float(amp_cond1), 6),
        'cnot_bias_shift': round(float(bias_shift), 6),
        'joint_E_min': round(float(joint_E.min()), 4),
        'joint_E_max': round(float(joint_E.max()), 4),
        'separability_residual_std': round(float(sep_residual), 6),
        'chsh_S_best': round(float(best_S), 6),
        'classical_bound': 2.0,
        'quantum_max': round(2*np.sqrt(2), 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q15_optimal_two_qubit.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q15 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
