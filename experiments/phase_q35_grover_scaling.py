# -*- coding: utf-8 -*-
"""
Phase Q35: Grover Scaling Analysis

Q18 showed 4631x amplification for a single search.
Now test how the Grover-like amplification SCALES with "database size".

Key question: does S-Qubit Grover show the quadratic speedup O(sqrt(N))?

Method:
  1. Create "databases" of size N = {4, 8, 16, 32, 64, 128, 256}
  2. For each N, embed N items as soul vectors at different phases
  3. Mark 1 target item
  4. Measure amplification ratio (target vs average non-target)
  5. Plot amplification vs N and fit scaling law
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


def inject_measure(model, tok, prompt, device, vec, layer):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def main():
    print("[Q35] Grover Scaling Analysis")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training soul vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # Database sizes to test
    db_sizes = [4, 8, 16, 32, 64, 128, 256]
    results = []

    for N in db_sizes:
        print("\n  Database size N=%d..." % N)

        # Target: |0> (phi=0)
        # Database items: phi = 2*pi*k/N for k=0..N-1
        target_phi = 0.0
        target_probs = inject_measure(model, tok, prompt, DEVICE,
                                       phi_vec(target_phi, v0, v1), INJECT_LAYER)
        target_p = float(target_probs[min_tok])

        # Measure all other items
        non_target_probs = []
        for k in range(1, N):
            phi = 2 * np.pi * k / N
            probs = inject_measure(model, tok, prompt, DEVICE,
                                    phi_vec(phi, v0, v1), INJECT_LAYER)
            non_target_probs.append(float(probs[min_tok]))

        avg_non_target = np.mean(non_target_probs) if non_target_probs else 1e-10
        max_non_target = max(non_target_probs) if non_target_probs else 1e-10
        min_non_target = min(non_target_probs) if non_target_probs else 1e-10

        # Amplification ratio
        amp_ratio = target_p / (avg_non_target + 1e-10)
        contrast = (target_p - avg_non_target) / (target_p + avg_non_target + 1e-10)

        # Classical baseline: random guess = 1/N
        classical_p = 1.0 / N
        quantum_advantage = target_p / (classical_p + 1e-10)

        results.append({
            'N': N,
            'target_p': round(target_p, 6),
            'avg_non_target': round(avg_non_target, 6),
            'max_non_target': round(max_non_target, 6),
            'min_non_target': round(min_non_target, 6),
            'amp_ratio': round(float(amp_ratio), 4),
            'contrast': round(float(contrast), 6),
            'classical_p': round(classical_p, 6),
            'quantum_advantage': round(float(quantum_advantage), 4),
        })
        print("    target=%.4f, avg_other=%.4f, ratio=%.1fx, advantage=%.1fx" % (
            target_p, avg_non_target, amp_ratio, quantum_advantage))

    # Fit scaling law: advantage ~ N^alpha
    from scipy.optimize import curve_fit
    Ns = np.array([r['N'] for r in results], dtype=float)
    advantages = np.array([r['quantum_advantage'] for r in results])

    def power_law(N, a, alpha):
        return a * np.power(N, alpha)

    try:
        popt, _ = curve_fit(power_law, Ns, advantages, p0=[1, 0.5], maxfev=5000)
        alpha_fit = popt[1]
        a_fit = popt[0]
    except Exception:
        alpha_fit = 0.0
        a_fit = 1.0

    print("\n  GROVER SCALING SUMMARY:")
    print("    Advantage ~ N^%.3f" % alpha_fit)
    print("    Quantum speedup (ideal): N^0.5")
    print("    Classical: N^0 (no speedup)")

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Target vs non-target probability
    ax = axes[0]
    ax.plot(Ns, [r['target_p'] for r in results], 'ro-', lw=2, ms=8,
            label='Target |0>')
    ax.plot(Ns, [r['avg_non_target'] for r in results], 'bs-', lw=2, ms=6,
            label='Avg non-target')
    ax.fill_between(Ns, [r['min_non_target'] for r in results],
                    [r['max_non_target'] for r in results],
                    alpha=0.2, color='blue')
    ax.set_xlabel('Database size N')
    ax.set_ylabel('P(min token)')
    ax.set_title('(a) Target Amplification\nvs Database Size', fontweight='bold')
    ax.set_xscale('log', base=2); ax.legend(); ax.grid(alpha=0.3)

    # Panel B: Quantum advantage scaling
    ax = axes[1]
    ax.loglog(Ns, advantages, 'ro-', lw=2, ms=8, label='S-Qubit measured')
    # Reference lines
    ax.loglog(Ns, Ns**0.5, 'b--', lw=1.5, label='sqrt(N) (Grover)', alpha=0.7)
    ax.loglog(Ns, Ns**1.0, 'g--', lw=1.5, label='N (max possible)', alpha=0.7)
    ax.loglog(Ns, power_law(Ns, a_fit, alpha_fit), 'r:', lw=2,
              label='Fit: N^%.2f' % alpha_fit)
    ax.set_xlabel('Database size N')
    ax.set_ylabel('Quantum advantage (x)')
    ax.set_title('(b) Scaling Law\nAdvantage ~ N^%.2f' % alpha_fit, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Grover Scaling Analysis\n"
        "=======================\n\n"
        "Physical Grover: O(sqrt(N))\n"
        "S-Qubit Grover:  O(N^%.2f)\n\n"
        "Database sizes: %s\n\n"
        "Best result:\n"
        "  N=%d: %.1fx advantage\n\n"
        "%s" % (
            alpha_fit,
            str([r['N'] for r in results]),
            results[-1]['N'], results[-1]['quantum_advantage'],
            "SUPER-GROVER!" if alpha_fit > 0.5
            else "Sub-Grover but\nstill shows scaling" if alpha_fit > 0.1
            else "Minimal scaling")
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F5E9', alpha=0.9))

    plt.suptitle('Phase Q35: Grover Scaling Analysis\n'
                 'How does S-Qubit search advantage scale with database size?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q35_grover_scaling.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q35', 'name': 'grover_scaling',
        'db_sizes': db_sizes,
        'scaling_exponent': round(float(alpha_fit), 6),
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q35_grover_scaling.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q35 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
