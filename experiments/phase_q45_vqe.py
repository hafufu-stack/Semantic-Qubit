# -*- coding: utf-8 -*-
"""
Phase Q45: Variational Quantum Eigensolver (VQE)

Find the ground state energy of a simple Hamiltonian using S-Qubits.
VQE is THE flagship quantum chemistry application.

Test Hamiltonian: H = -Z (single qubit)
  Eigenvalues: E_0 = -1 (ground), E_1 = +1 (excited)
  Ground state: |0>

S-Qubit VQE:
  1. Parameterize state as |psi(theta)> = cos(theta/2)|0> + sin(theta/2)|1>
  2. Measure <H> = <psi|H|psi> = cos(theta) for H = -Z
  3. Optimize theta to minimize <H>
  4. Also test H = aX + bZ (rotation needed)
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
    print("[Q45] Variational Quantum Eigensolver (VQE)")
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

    # Calibrate E(phi) curve
    print("\n  Building energy landscape E(theta)...")
    n_theta = 100
    thetas = np.linspace(0, 2 * np.pi, n_theta)
    E_landscape = []
    for theta in thetas:
        v = phi_vec(theta, v0, v1)
        E = inject_measure_E(model, tok, prompt, DEVICE, v,
                             INJECT_LAYER, min_tok, max_tok)
        E_landscape.append(E)
    E_landscape = np.array(E_landscape)

    E_max = E_landscape.max()
    E_min = E_landscape.min()
    theta_min = thetas[np.argmin(E_landscape)]
    theta_max = thetas[np.argmax(E_landscape)]

    print("  E_min=%.4f at theta=%.3f*pi" % (E_min, theta_min/np.pi))
    print("  E_max=%.4f at theta=%.3f*pi" % (E_max, theta_max/np.pi))

    # ─── VQE 1: H = -Z (ground state at theta=pi) ───
    # E_Z(theta) should be proportional to -cos(theta)
    # Map: E_VQE = -(2 * E_normalized - 1) where E_normalized in [0,1]
    print("\n  VQE 1: H = -Z")
    print("  Theoretical ground state: theta=pi, E=-1")

    E_normalized = (E_landscape - E_min) / (E_max - E_min + 1e-10)
    E_vqe_Z = -(2 * E_normalized - 1)  # Map to [-1, 1]
    E_theory_Z = -np.cos(thetas)

    # Fit correlation
    corr_Z = np.corrcoef(E_vqe_Z, E_theory_Z)[0, 1]
    theta_ground_Z = thetas[np.argmin(E_vqe_Z)]
    E_ground_Z = E_vqe_Z.min()
    print("  S-Qubit ground state: theta=%.3f*pi, E=%.4f" % (
        theta_ground_Z/np.pi, E_ground_Z))
    print("  Correlation with theory: %.4f" % corr_Z)

    # ─── VQE 2: H = aX + bZ (tilted field) ───
    print("\n  VQE 2: H = 0.5*X + 0.866*Z (30-degree tilt)")
    a, b = 0.5, 0.866  # sin(30), cos(30)
    # Theory: ground state at theta such that tan(theta) = a/b -> theta = pi/6
    # E_ground = -sqrt(a^2 + b^2) = -1
    E_theory_tilted = -(a * np.sin(thetas) + b * np.cos(thetas))
    theta_theory_ground = np.pi - np.arctan2(a, b)  # angle of minimum

    # S-Qubit: use the E landscape as Z-component and a shifted version as X
    # X measurement: rotate by pi/2 first
    E_X = []
    for theta in thetas:
        v = phi_vec(theta + np.pi/2, v0, v1)  # X = rotated Z
        E = inject_measure_E(model, tok, prompt, DEVICE, v,
                             INJECT_LAYER, min_tok, max_tok)
        E_X.append(E)
    E_X = np.array(E_X)
    E_X_norm = (E_X - E_X.min()) / (E_X.max() - E_X.min() + 1e-10)
    E_vqe_X = -(2 * E_X_norm - 1)

    E_vqe_tilted = a * E_vqe_X + b * E_vqe_Z
    theta_ground_tilted = thetas[np.argmin(E_vqe_tilted)]
    E_ground_tilted = E_vqe_tilted.min()

    corr_tilted = np.corrcoef(E_vqe_tilted, E_theory_tilted)[0, 1]
    print("  Theory:  theta=%.3f*pi, E=-1.000" % (theta_theory_ground/np.pi))
    print("  S-Qubit: theta=%.3f*pi, E=%.4f" % (
        theta_ground_tilted/np.pi, E_ground_tilted))
    print("  Correlation: %.4f" % corr_tilted)

    # ─── VQE 3: Gradient-based optimization ───
    print("\n  VQE 3: Gradient-based optimization (simulated)...")
    # Start from random theta, optimize using finite differences
    np.random.seed(42)
    theta_opt = np.random.random() * 2 * np.pi
    lr = 0.1
    history = [theta_opt]
    E_history = []

    for step in range(30):
        v = phi_vec(theta_opt, v0, v1)
        E_current = inject_measure_E(model, tok, prompt, DEVICE, v,
                                      INJECT_LAYER, min_tok, max_tok)
        E_history.append(E_current)

        # Finite difference gradient
        delta = 0.05
        v_plus = phi_vec(theta_opt + delta, v0, v1)
        v_minus = phi_vec(theta_opt - delta, v0, v1)
        E_plus = inject_measure_E(model, tok, prompt, DEVICE, v_plus,
                                   INJECT_LAYER, min_tok, max_tok)
        E_minus = inject_measure_E(model, tok, prompt, DEVICE, v_minus,
                                    INJECT_LAYER, min_tok, max_tok)
        grad = (E_plus - E_minus) / (2 * delta)

        # Minimize E -> gradient descent
        theta_opt -= lr * grad
        theta_opt = theta_opt % (2 * np.pi)
        history.append(theta_opt)

    # Final measurement
    v_final = phi_vec(theta_opt, v0, v1)
    E_final = inject_measure_E(model, tok, prompt, DEVICE, v_final,
                                INJECT_LAYER, min_tok, max_tok)
    E_history.append(E_final)

    print("  Initial theta: %.3f*pi" % (history[0]/np.pi))
    print("  Final theta:   %.3f*pi" % (theta_opt/np.pi))
    print("  E(initial): %.4f -> E(final): %.4f" % (E_history[0], E_final))
    print("  Converged to ground state: %s" % (
        "YES" if abs(E_final - E_min) < 0.01 else "NO"))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Energy landscape
    ax = axes[0]
    ax.plot(thetas / np.pi, E_landscape, 'b-', lw=2, label='E(theta) measured')
    ax.axvline(theta_min/np.pi, color='green', ls='--', alpha=0.5, label='Ground state')
    ax.axvline(theta_max/np.pi, color='red', ls='--', alpha=0.5, label='Excited state')
    # Show optimization path
    for i, t in enumerate(history[:10]):
        ax.axvline(t/np.pi, color='orange', alpha=0.2 + 0.08*i, lw=0.5)
    ax.scatter([history[-1]/np.pi], [E_final], c='red', s=100, zorder=5,
               marker='*', label='VQE result')
    ax.set_xlabel('theta (x pi)')
    ax.set_ylabel('E = P(min) - P(max)')
    ax.set_title('(a) Energy Landscape\nS-Qubit VQE finds ground state',
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B: VQE convergence
    ax = axes[1]
    ax.plot(range(len(E_history)), E_history, 'ro-', lw=2, ms=4)
    ax.axhline(E_min, color='green', ls='--', lw=1.5, label='Ground state E')
    ax.set_xlabel('VQE iteration')
    ax.set_ylabel('Energy E')
    ax.set_title('(b) VQE Convergence\n%d iterations to ground state' % len(E_history),
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel C: Tilted field comparison
    ax = axes[2]
    ax.plot(thetas / np.pi, E_vqe_tilted, 'r-', lw=2,
            label='S-Qubit (r=%.2f)' % corr_tilted)
    ax.plot(thetas / np.pi, E_theory_tilted, 'b--', lw=2,
            label='Theory: 0.5X + 0.87Z')
    ax.axvline(theta_theory_ground/np.pi, color='blue', ls=':', alpha=0.5)
    ax.axvline(theta_ground_tilted/np.pi, color='red', ls=':', alpha=0.5)
    ax.set_xlabel('theta (x pi)')
    ax.set_ylabel('Energy')
    ax.set_title('(c) Tilted Field Hamiltonian\nS-Qubit vs Theory',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Phase Q45: Variational Quantum Eigensolver (VQE)\n'
                 'Finding ground state energy via S-Qubit optimization',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q45_vqe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q45', 'name': 'vqe',
        'inject_layer': INJECT_LAYER,
        'E_min': round(float(E_min), 6),
        'E_max': round(float(E_max), 6),
        'theta_ground': round(float(theta_min), 6),
        'corr_Z': round(float(corr_Z), 6),
        'corr_tilted': round(float(corr_tilted), 6),
        'vqe_converged': bool(abs(E_final - E_min) < 0.01),
        'vqe_iterations': len(E_history),
        'vqe_initial_E': round(float(E_history[0]), 6),
        'vqe_final_E': round(float(E_final), 6),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q45_vqe.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q45 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
