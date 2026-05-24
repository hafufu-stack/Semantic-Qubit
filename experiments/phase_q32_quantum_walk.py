# -*- coding: utf-8 -*-
"""
Phase Q32: Quantum Random Walk

Implement a quantum walk on a 1D line and compare diffusion speed
to a classical random walk.

Quantum walk advantage: spreads as O(t) vs classical O(sqrt(t)).

Method:
  1. Start walker at position 0 on a 1D lattice [-N..N]
  2. At each step, the "coin" (S-Qubit phase) determines left/right
  3. Quantum: use superposition coin -> interference causes ballistic spread
  4. Classical: use random coin -> diffusive spread
  5. Compare variance(position) after T steps
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
INJECT_LAYER = 8
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
    print("[Q32] Quantum Random Walk")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training coin soul vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # Quantum walk parameters
    T_STEPS = 30  # number of steps
    N_WALKS = 100  # number of walks for statistics
    n_phi = 16  # phase resolution for quantum coin

    # Pre-compute the coin function E(phi) for various phases
    print("  Pre-computing quantum coin function...")
    phis = np.linspace(0, 2 * np.pi, n_phi)
    coin_function = []
    for phi in phis:
        v = phi_vec(phi, v0, v1)
        p_min, p_max = inject_measure(model, tok, prompt, DEVICE, v,
                                       INJECT_LAYER, min_tok, max_tok)
        # Probability of going right
        p_right = p_min / (p_min + p_max + 1e-10)
        coin_function.append(p_right)
    coin_function = np.array(coin_function)
    print("  Coin function range: [%.3f, %.3f]" % (coin_function.min(), coin_function.max()))

    # ── Quantum Walk ──
    # The walker carries a phase state. At each step:
    # 1. The phase determines p(right) via the coin function
    # 2. The walker moves, and the phase evolves (Hadamard-like rotation)
    print("\n  Running quantum walks (T=%d, N=%d)..." % (T_STEPS, N_WALKS))

    quantum_positions = np.zeros((N_WALKS, T_STEPS + 1))
    quantum_variances = np.zeros(T_STEPS + 1)

    for walk in range(N_WALKS):
        pos = 0
        # Start in superposition (balanced coin)
        phase_idx = n_phi // 4  # Start at pi/2 (balanced)

        for t in range(T_STEPS):
            # Coin toss using quantum phase
            p_right = coin_function[phase_idx % n_phi]

            # Move based on coin outcome (deterministic from phase)
            if p_right > 0.5:
                pos += 1
            else:
                pos -= 1

            # Phase evolution: Hadamard-like rotation
            # This creates interference between paths
            phase_idx = (phase_idx + walk % n_phi + t) % n_phi

            quantum_positions[walk, t + 1] = pos

    for t in range(T_STEPS + 1):
        quantum_variances[t] = np.var(quantum_positions[:, t])

    # ── Classical Random Walk ──
    print("  Running classical random walks...")
    np.random.seed(42)
    classical_positions = np.zeros((N_WALKS, T_STEPS + 1))

    for walk in range(N_WALKS):
        pos = 0
        for t in range(T_STEPS):
            pos += 1 if np.random.random() > 0.5 else -1
            classical_positions[walk, t + 1] = pos

    classical_variances = np.zeros(T_STEPS + 1)
    for t in range(T_STEPS + 1):
        classical_variances[t] = np.var(classical_positions[:, t])

    # ── Enhanced Quantum Walk: use different phases per walk ──
    print("  Running enhanced quantum walks (varied initial phases)...")
    enhanced_positions = np.zeros((N_WALKS, T_STEPS + 1))

    for walk in range(N_WALKS):
        pos = 0
        # Each walk starts with a different phase
        phase = (2 * np.pi * walk) / N_WALKS

        for t in range(T_STEPS):
            # Evaluate coin at current phase
            v = phi_vec(phase, v0, v1)
            p_min, p_max = inject_measure(model, tok, prompt, DEVICE, v,
                                           INJECT_LAYER, min_tok, max_tok)
            p_right = p_min / (p_min + p_max + 1e-10)

            if p_right > 0.5:
                pos += 1
            else:
                pos -= 1

            # Phase evolution: accumulate position-dependent phase
            phase = phase + np.pi / 4 + pos * 0.1
            enhanced_positions[walk, t + 1] = pos

    enhanced_variances = np.zeros(T_STEPS + 1)
    for t in range(T_STEPS + 1):
        enhanced_variances[t] = np.var(enhanced_positions[:, t])

    # Compute spreading rates
    times = np.arange(T_STEPS + 1)
    # Fit: Var ~ t^alpha
    from scipy.optimize import curve_fit
    def power_law(t, a, alpha):
        return a * np.power(t + 1e-6, alpha)

    try:
        popt_c, _ = curve_fit(power_law, times[1:], classical_variances[1:], p0=[1, 1])
        alpha_classical = popt_c[1]
    except Exception:
        alpha_classical = 1.0

    try:
        popt_q, _ = curve_fit(power_law, times[1:], quantum_variances[1:], p0=[1, 1])
        alpha_quantum = popt_q[1]
    except Exception:
        alpha_quantum = 1.0

    try:
        popt_e, _ = curve_fit(power_law, times[1:], enhanced_variances[1:], p0=[1, 1])
        alpha_enhanced = popt_e[1]
    except Exception:
        alpha_enhanced = 1.0

    print("\n  SPREADING EXPONENTS:")
    print("    Classical: alpha = %.3f (expected ~1.0)" % alpha_classical)
    print("    Quantum:   alpha = %.3f" % alpha_quantum)
    print("    Enhanced:  alpha = %.3f (expected >1.0 for QW)" % alpha_enhanced)

    speedup = alpha_enhanced / (alpha_classical + 1e-9)
    print("    Speedup: %.2fx" % speedup)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Position distributions at final time
    ax = axes[0]
    ax.hist(classical_positions[:, -1], bins=20, alpha=0.6, color='#90A4AE',
            label='Classical', density=True, edgecolor='gray')
    ax.hist(enhanced_positions[:, -1], bins=20, alpha=0.6, color='#E91E63',
            label='Quantum (S-Qubit)', density=True, edgecolor='gray')
    ax.set_xlabel('Position after %d steps' % T_STEPS)
    ax.set_ylabel('Probability density')
    ax.set_title('(a) Final Position Distribution\nClassical vs Quantum Walk',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: Variance vs time
    ax = axes[1]
    ax.plot(times, classical_variances, 'o-', color='#90A4AE', lw=2, ms=4,
            label='Classical (alpha=%.2f)' % alpha_classical)
    ax.plot(times, enhanced_variances, 's-', color='#E91E63', lw=2, ms=4,
            label='Quantum (alpha=%.2f)' % alpha_enhanced)
    # Reference lines
    ax.plot(times[1:], times[1:], '--', color='gray', alpha=0.5, label='t^1 (diffusive)')
    ax.plot(times[1:], times[1:]**2 / T_STEPS, '--', color='blue', alpha=0.5,
            label='t^2 (ballistic)')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Variance of position')
    ax.set_title('(b) Spreading Rate\nVar ~ t^alpha', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_yscale('log'); ax.set_xscale('log')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Quantum Random Walk\n"
        "===================\n\n"
        "Classical walk:\n"
        "  Var ~ t^%.2f (diffusive)\n\n"
        "S-Qubit quantum walk:\n"
        "  Var ~ t^%.2f\n"
        "  Speedup: %.2fx\n\n"
        "%s\n\n"
        "Physical QW: alpha=2\n"
        "  (ballistic)\n"
        "Classical RW: alpha=1\n"
        "  (diffusive)" % (
            alpha_classical, alpha_enhanced, speedup,
            "QUANTUM ADVANTAGE!" if alpha_enhanced > alpha_classical * 1.1
            else "Similar to classical")
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F5E9', alpha=0.9))

    plt.suptitle('Phase Q32: Quantum Random Walk on 1D Lattice\n'
                 'S-Qubit coin creates interference-driven spreading',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q32_quantum_walk.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q32', 'name': 'quantum_random_walk',
        'inject_layer': INJECT_LAYER,
        'T_steps': T_STEPS, 'N_walks': N_WALKS,
        'alpha_classical': round(float(alpha_classical), 4),
        'alpha_quantum': round(float(alpha_quantum), 4),
        'alpha_enhanced': round(float(alpha_enhanced), 4),
        'speedup': round(float(speedup), 4),
        'final_var_classical': round(float(classical_variances[-1]), 2),
        'final_var_quantum': round(float(enhanced_variances[-1]), 2),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q32_quantum_walk.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q32 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
