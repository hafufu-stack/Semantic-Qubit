# -*- coding: utf-8 -*-
"""
Phase Q2: Bell Test - Semantic Interference Fringes
The quantum Bell test analog: does injecting
  |psi(theta)> = cos(theta/2)*|MIN> + sin(theta/2)*e^{i*phi}*|MAX>
produce INTERFERENCE FRINGES in output probability as theta varies?

Classical mixing (mixed state): P(MIN) = cos^2(theta/2), P(MAX) = sin^2(theta/2)
  -> probabilities follow smooth monotone curve

Quantum interference (pure state): P(MIN) oscillates with SINE WAVE pattern
  driven by the phase phi, NOT just amplitude.

This is the key distinction: if output probs vary SINUSOIDALLY with phi
at FIXED amplitude, that is neural quantum interference.
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
LAYER = 8


def train_soul(model, tok, data, device, layer, epochs=150, lr=0.01, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=lr)
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


def probe_phase_fringes(model, tok, min_vec, max_vec, prompt, device, layer,
                         theta, phi_range):
    """
    Fix theta (amplitude mixing), sweep phi (phase).
    psi = cos(theta/2)*|MIN> + sin(theta/2) * phase_rotated(|MAX>)

    Phase rotation: We rotate MAX_vec in the 2D plane defined by MIN and MAX
    orthogonal component. Real approximation of e^{i*phi} using Givens rotation.
    """
    # Orthogonalize: max_perp = MAX - (MAX.MIN/|MIN|^2)*MIN
    min_n = min_vec / (min_vec.norm() + 1e-8)
    max_perp = max_vec - (max_vec @ min_n) * min_n
    max_perp_n = max_perp / (max_perp.norm() + 1e-8)

    results = []
    amp_min = float(np.cos(theta / 2))
    amp_max = float(np.sin(theta / 2))

    for phi in phi_range:
        # Givens rotation in (min_n, max_perp_n) plane by angle phi
        rotated_max = (np.cos(phi) * max_vec +
                       np.sin(phi) * max_perp_n * max_vec.norm())
        psi = amp_min * min_vec + amp_max * rotated_max
        psi = psi / (psi.norm() + 1e-8) * min_vec.norm()  # renormalize to |MIN| scale

        def hook(m, i, o, v=psi):
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h

        inp = tok(prompt, return_tensors='pt').to(device)
        handle = model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()

        probs = torch.softmax(out.logits[0, -1, :], dim=-1)
        tok_min = tok.encode('2')[-1]
        tok_max = tok.encode('7')[-1]
        results.append({
            'phi': round(float(phi), 4),
            'phi_deg': round(float(phi * 180 / np.pi), 1),
            'prob_min': round(float(probs[tok_min]), 5),
            'prob_max': round(float(probs[tok_max]), 5),
            'entropy': round(float(-(probs * (probs + 1e-12).log()).sum()), 4),
        })
    return results


def main():
    print("[Q2] Bell Test - Semantic Interference Fringes")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"), ("min(7,4)=", "4"),
        ("min(6,1)=", "1"), ("min(2,8)=", "2"), ("min(5,9)=", "5"),
        ("min(1,3)=", "1"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"), ("min(7,4)=", "7"),
        ("min(6,1)=", "6"), ("min(2,8)=", "8"), ("min(5,9)=", "9"),
        ("min(1,3)=", "3"),
    ]

    print("  Training basis vectors...")
    min_vec = train_soul(model, tok, min_data, DEVICE, LAYER, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, LAYER, seed=99)

    # Sweep phi from 0 to 2*pi (full revolution) at equal amplitude (theta = pi/2)
    phi_range = np.linspace(0, 2 * np.pi, 37)  # 0, 10, 20, ..., 360 degrees
    prompt = "min(7,2)="

    print("  Sweeping phase phi (0 -> 2*pi) at theta=pi/2 (equal amplitude)...")
    fringes = probe_phase_fringes(
        model, tok, min_vec, max_vec, prompt, DEVICE, LAYER,
        theta=np.pi / 2, phi_range=phi_range
    )

    # Check for periodicity: fit sine wave to prob_min
    prob_min_arr = np.array([r['prob_min'] for r in fringes])
    phi_arr = np.array([r['phi'] for r in fringes])
    # Amplitude of variation (peak-to-peak / 2)
    variation_amplitude = (prob_min_arr.max() - prob_min_arr.min()) / 2
    print("  Phase sweep variation amplitude: %.4f" % variation_amplitude)
    print("  (>0.05 suggests interference; <0.01 suggests classical mixing)")

    # Also sweep at theta=pi/4 (asymmetric amplitude)
    print("  Sweeping at theta=pi/4 (asymmetric)...")
    fringes_asym = probe_phase_fringes(
        model, tok, min_vec, max_vec, prompt, DEVICE, LAYER,
        theta=np.pi / 4, phi_range=phi_range
    )
    prob_min_asym = np.array([r['prob_min'] for r in fringes_asym])
    var_amp_asym = (prob_min_asym.max() - prob_min_asym.min()) / 2
    print("  Asymmetric amplitude variation: %.4f" % var_amp_asym)

    # Classical prediction (no interference)
    classical_equal = np.ones(len(phi_range)) * prob_min_arr.mean()

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Phase fringes at equal amplitude
    ax = axes[0]
    phi_deg = [r['phi_deg'] for r in fringes]
    prob_min_list = [r['prob_min'] for r in fringes]
    prob_max_list = [r['prob_max'] for r in fringes]
    ax.plot(phi_deg, prob_min_list, 'o-', color='#E91E63', lw=2, label='P(min answer=2)')
    ax.plot(phi_deg, prob_max_list, 's-', color='#2196F3', lw=2, label='P(max answer=7)')
    ax.axhline(np.mean(prob_min_list), color='#E91E63', linestyle='--', alpha=0.4,
               label='Classical prediction')
    ax.set_xlabel('Phase phi (degrees)')
    ax.set_ylabel('Token Probability')
    ax.set_title('Phase Sweep at Equal Amplitude\ntheta=pi/2', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks([0, 90, 180, 270, 360])

    # Panel 2: Variation amplitude comparison
    ax = axes[1]
    thetas = ['theta=pi/2\n(equal amp)', 'theta=pi/4\n(asymmetric)']
    var_amps = [variation_amplitude, var_amp_asym]
    colors = ['#9C27B0' if v > 0.05 else '#FF9800' for v in var_amps]
    bars = ax.bar(thetas, var_amps, color=colors, edgecolor='black', width=0.4)
    for bar, v in zip(bars, var_amps):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                '%.4f' % v, ha='center', fontweight='bold')
    ax.axhline(0.05, color='red', linestyle='--', label='Interference threshold (0.05)')
    ax.set_ylabel('Phase-induced Variation Amplitude')
    ax.set_title('Interference Strength\nPeak-to-Peak / 2', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel 3: Entropy variation
    ax = axes[2]
    entropies = [r['entropy'] for r in fringes]
    ax.plot(phi_deg, entropies, 'D-', color='#4CAF50', lw=2)
    ax.set_xlabel('Phase phi (degrees)')
    ax.set_ylabel('Output Entropy (nats)')
    ax.set_title('Output Entropy vs Phase\n"Uncertainty landscape"', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks([0, 90, 180, 270, 360])

    verdict = "INTERFERENCE DETECTED!" if variation_amplitude > 0.05 else "Classical mixing (no fringes)"
    plt.suptitle(
        'Phase Q2: Bell Test - Semantic Interference Fringes\n'
        'Verdict: %s (amp=%.4f)' % (verdict, variation_amplitude),
        fontsize=12, fontweight='bold',
        color='#1B5E20' if variation_amplitude > 0.05 else '#BF360C'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q2_bell_test.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q2', 'name': 'bell_test',
        'variation_amplitude_equal': float(variation_amplitude),
        'variation_amplitude_asym': float(var_amp_asym),
        'interference_detected': bool(variation_amplitude > 0.05),
        'phase_fringes_equal': fringes,
        'phase_fringes_asym': fringes_asym,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q2_bell_test.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Q2 completed in %.0fs" % (time.time() - start))
    print("  RESULT: %s" % ("INTERFERENCE FRINGES DETECTED!" if variation_amplitude > 0.05
                             else "Classical mixing (no quantum interference)"))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
