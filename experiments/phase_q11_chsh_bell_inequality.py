# -*- coding: utf-8 -*-
"""
Phase Q11: CHSH-Style Bell Inequality Test

Standard quantum Bell inequality (CHSH):
  S = |<AB> - <AB'> + <A'B> + <A'B'>| <= 2 (classical)
  S > 2 (quantum, max = 2*sqrt(2) ≈ 2.828)

For S-Qubits, we define:
  A, A': two measurement bases for the first qubit (different phi angles)
  B, B': two measurement bases for the second qubit (different phi angles)
  <AB> = correlation between measurement outcomes

Neural adaptation:
  - We only have ONE injection point (L8), not two separate qubits
  - But we can define "two qubits" using DIFFERENT tasks on DIFFERENT tokens
  - Qubit 1: min/max measured at position -1 (last token)
  - Qubit 2: inject at position -2 (second-to-last token) with different basis

Simpler CHSH analog (single qubit, two angles):
  Measure P(correct | phi_a), P(correct | phi_b) for angles a in {0, pi/4}
  and b in {pi/8, 3*pi/8}
  E(a,b) = 2*P(same outcome | a,b) - 1  (correlation)
  S = |E(0, pi/8) - E(0, 3pi/8) + E(pi/4, pi/8) + E(pi/4, 3pi/8)|
  Classical bound: S <= 2
  Quantum bound:   S <= 2*sqrt(2)

We compute this for ALL combinations and see if S > 2.
Also compute a simpler fringe-based CHSH from the phi sweep data.
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
N_PHI_FINE = 73  # finer resolution for precise CHSH


def train_soul(model, tok, data, device, layer=8, epochs=150, seed=42):
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


def inject_and_get_probs(model, tok, prompt, device, inject_vec, inject_layer):
    """Return full probability distribution."""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[inject_layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def phi_to_vec(phi, vec0, vec1):
    """Superposition state at angle phi."""
    vec = np.cos(phi / 2) * vec0 + np.sin(phi / 2) * vec1
    n = vec.norm()
    if n > 0:
        vec = vec / n * vec0.norm()
    return vec


def measure_outcome(probs, tok_id_0, tok_id_1):
    """
    Binarize: +1 if tok_id_0 > tok_id_1 (outcome=0), -1 otherwise (outcome=1).
    Returns E = p0 - p1 (expectation value in [-1, 1]).
    """
    p0 = float(probs[tok_id_0])
    p1 = float(probs[tok_id_1])
    # E(phi) = p0 - p1 (analogous to <sigma_z>)
    return p0 - p1


def compute_chsh(E_vals, angle_pairs):
    """
    Compute CHSH S from E(a,b) values at 4 angle combinations.
    angle_pairs: [(a1,b1), (a1,b2), (a2,b1), (a2,b2)]
    S = |E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)|
    """
    E11, E12, E21, E22 = [E_vals[k] for k in angle_pairs]
    S = abs(E11 - E12 + E21 + E22)
    return S


def main():
    print("[Q11] CHSH-Style Bell Inequality Test")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print("  Model loaded. Inject layer=%d" % INJECT_LAYER)

    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"),
    ]
    print("  Training basis vectors (150 epochs)...")
    min_vec = train_soul(model, tok, min_data, DEVICE, layer=INJECT_LAYER, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, layer=INJECT_LAYER, seed=99)

    prompt = "min(7,2)="
    min_tok_id = tok.encode("2")[-1]
    max_tok_id = tok.encode("7")[-1]
    print("  Basis trained. min_tok=%d, max_tok=%d" % (min_tok_id, max_tok_id))

    # === Method 1: Fine-grained phi sweep for fringe analysis ===
    print("  Fine-grained phi sweep (%d points)..." % N_PHI_FINE)
    phis_fine = np.linspace(0, 4 * np.pi, N_PHI_FINE)
    E_phi = []  # E(phi) = P(min) - P(max)
    P_min_phi = []
    P_max_phi = []
    for phi in phis_fine:
        vec = phi_to_vec(phi, min_vec, max_vec)
        probs = inject_and_get_probs(model, tok, prompt, DEVICE, vec, INJECT_LAYER)
        E = measure_outcome(probs, min_tok_id, max_tok_id)
        E_phi.append(E)
        P_min_phi.append(float(probs[min_tok_id]))
        P_max_phi.append(float(probs[max_tok_id]))

    E_phi = np.array(E_phi)
    P_min_phi = np.array(P_min_phi)
    P_max_phi = np.array(P_max_phi)

    # === Method 2: CHSH S value at specific angle combinations ===
    # Standard CHSH angles: a=0, a'=pi/2, b=pi/4, b'=-pi/4
    # Map to our phi angles:
    # Alice angles: phi_a1=0, phi_a2=pi/2
    # Bob angles:   phi_b1=pi/4, phi_b2=-pi/4 (= 7*pi/4)
    chsh_angles = {
        'a1': 0,
        'a2': np.pi / 2,
        'b1': np.pi / 4,
        'b2': 7 * np.pi / 4,
        # Additional CHSH optimal angles
        'a1_opt': 0,
        'a2_opt': np.pi / 2,
        'b1_opt': np.pi / 8,
        'b2_opt': 3 * np.pi / 8,
    }

    print("  Computing E(phi) at CHSH angle combinations...")
    E_at = {}
    for angle_name, phi_val in chsh_angles.items():
        vec = phi_to_vec(phi_val, min_vec, max_vec)
        probs = inject_and_get_probs(model, tok, prompt, DEVICE, vec, INJECT_LAYER)
        E_at[angle_name] = measure_outcome(probs, min_tok_id, max_tok_id)
        print("    phi=%s (%.3f*pi): E=%.4f" % (angle_name, phi_val/np.pi, E_at[angle_name]))

    # Standard CHSH
    S_std = abs(E_at['a1']*E_at['b1'] - E_at['a1']*E_at['b2'] +
                E_at['a2']*E_at['b1'] + E_at['a2']*E_at['b2'])
    # Optimal CHSH
    S_opt = abs(E_at['a1_opt']*E_at['b1_opt'] - E_at['a1_opt']*E_at['b2_opt'] +
                E_at['a2_opt']*E_at['b1_opt'] + E_at['a2_opt']*E_at['b2_opt'])

    # Method 3: Grid CHSH search over angle pairs
    # Search for phi angles that maximize S
    print("  Grid search for maximum CHSH S...")
    test_phis = np.linspace(0, 2 * np.pi, 17)
    # Get E for all test_phis
    E_grid = []
    for phi in test_phis:
        vec = phi_to_vec(phi, min_vec, max_vec)
        probs = inject_and_get_probs(model, tok, prompt, DEVICE, vec, INJECT_LAYER)
        E_grid.append(measure_outcome(probs, min_tok_id, max_tok_id))

    # Find max S over all angle combinations
    best_S = 0
    best_combo = None
    for i1, pa1 in enumerate(test_phis):
        for i2, pa2 in enumerate(test_phis):
            for i3, pb1 in enumerate(test_phis):
                for i4, pb2 in enumerate(test_phis):
                    S = abs(E_grid[i1]*E_grid[i3] - E_grid[i1]*E_grid[i4] +
                            E_grid[i2]*E_grid[i3] + E_grid[i2]*E_grid[i4])
                    if S > best_S:
                        best_S = S
                        best_combo = (pa1, pa2, pb1, pb2)

    print("  Best CHSH S = %.4f (classical bound=2.000, quantum max=%.3f)" % (
        best_S, 2 * np.sqrt(2)))
    print("  S_std = %.4f, S_opt = %.4f" % (S_std, S_opt))

    # Fit sinusoid to E(phi)
    from numpy.fft import fft
    fft_E = np.abs(fft(E_phi))
    dominant_freq = np.argmax(fft_E[1:N_PHI_FINE//2]) + 1
    print("  Dominant interference frequency: %d cycles" % dominant_freq)

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: E(phi) = P(min) - P(max) interference fringe
    ax = axes[0]
    ax.plot(phis_fine / np.pi, E_phi, '#E91E63', lw=2, label='E(phi)=P(min)-P(max)')
    ax.plot(phis_fine / np.pi, P_min_phi, 'b--', lw=1.5, alpha=0.7, label='P(min)')
    ax.plot(phis_fine / np.pi, P_max_phi, 'g--', lw=1.5, alpha=0.7, label='P(max)')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Phase phi (x pi)', fontsize=11)
    ax.set_ylabel('E(phi)', fontsize=11)
    ax.set_title('Expectation Value E(phi)\nFine-grained fringe', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: CHSH S values
    ax = axes[1]
    S_vals = [S_std, S_opt, best_S]
    S_labels = ['Standard\nangles', 'Optimal\nangles', 'Grid\nsearch max']
    colors_bar = ['#2196F3', '#9C27B0', '#E91E63']
    bars = ax.bar(range(3), S_vals, color=colors_bar, edgecolor='black', alpha=0.85)
    ax.axhline(2.0, color='red', linestyle='--', lw=2, label='Classical bound S=2')
    ax.axhline(2 * np.sqrt(2), color='green', linestyle=':', lw=2,
               label='Quantum max S=2sqrt2=%.3f' % (2*np.sqrt(2)))
    for bar, val in zip(bars, S_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                '%.4f' % val, ha='center', fontweight='bold', fontsize=11)
    ax.set_xticks(range(3))
    ax.set_xticklabels(S_labels, fontsize=10)
    ax.set_ylabel('CHSH S value', fontsize=11)
    ax.set_title('CHSH Bell Inequality Test\nS>2 = quantum violation!', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 3.2)
    ax.grid(alpha=0.3, axis='y')

    # Panel 3: FFT of E(phi)
    ax = axes[2]
    freqs = np.arange(1, N_PHI_FINE // 2)
    fft_power = fft_E[1:N_PHI_FINE//2]
    ax.bar(freqs, fft_power, color='#FF9800', edgecolor='none', alpha=0.8)
    ax.axvline(dominant_freq, color='red', lw=2, label='Dominant freq=%d' % dominant_freq)
    ax.set_xlabel('Frequency (cycles per 4*pi)', fontsize=11)
    ax.set_ylabel('FFT Power', fontsize=11)
    ax.set_title('Frequency Spectrum of E(phi)\n(Quantum: expect freq=1 or 2)',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle(
        'Phase Q11: CHSH-Style Bell Inequality Test\n'
        '"Does the S-Qubit violate classical bounds?"',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q11_chsh_bell_inequality.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q11', 'name': 'chsh_bell_inequality',
        'inject_layer': INJECT_LAYER,
        'n_phi_fine': N_PHI_FINE,
        'E_phi': [round(e, 6) for e in E_phi.tolist()],
        'P_min_phi': [round(p, 6) for p in P_min_phi.tolist()],
        'P_max_phi': [round(p, 6) for p in P_max_phi.tolist()],
        'E_at_angles': {k: round(v, 6) for k, v in E_at.items()},
        'S_std': round(S_std, 6),
        'S_opt': round(S_opt, 6),
        'S_grid_max': round(best_S, 6),
        'classical_bound': 2.0,
        'quantum_max': round(2 * np.sqrt(2), 6),
        'quantum_violation_std': bool(S_std > 2.0),
        'quantum_violation_opt': bool(S_opt > 2.0),
        'quantum_violation_grid': bool(best_S > 2.0),
        'dominant_freq': int(dominant_freq),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q11_chsh_bell_inequality.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q11 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
