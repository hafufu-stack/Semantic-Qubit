# -*- coding: utf-8 -*-
"""
Phase Q13: Temperature & Decoherence Study

In quantum mechanics, decoherence destroys superposition.
Neural analog: adding noise to the hidden state = "thermal decoherence"

Experiments:
1. Noise-induced decoherence: add Gaussian noise sigma to injected soul vector
   -> Measure interference amplitude vs noise sigma
   -> Find decoherence threshold (amplitude drops to 50%)

2. Temperature analog: instead of fixed vec(phi), use softmax temperature T
   on the output logits to simulate "thermal measurement"
   -> Measure amplitude vs T (1.0=standard, 0.1=cold/sharp, 5.0=hot/diffuse)

3. Layer decoherence distance: inject at L8, measure amplitude when noise
   is added at intermediate layers L9, L10, ..., L16
   -> Does amplitude decay with "distance" from injection point?
   -> Analogous to quantum coherence length

This characterizes the "fragility" of the S-Qubit superposition.
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
N_PHI = 25


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


def measure_amplitude(model, tok, prompt, device, vec0, vec1, inject_layer,
                       tok_id, n_phi=N_PHI, noise_sigma=0.0, noise_layer=None,
                       temperature=1.0):
    """Sweep phi and compute interference amplitude with optional noise/temperature."""
    phis = np.linspace(0, 4 * np.pi, n_phi)
    p_vals = []
    scale = vec0.norm()
    for phi in phis:
        vec = np.cos(phi / 2) * vec0 + np.sin(phi / 2) * vec1
        n = vec.norm()
        if n > 0:
            vec = vec / n * scale

        # Add noise to the injection vector
        if noise_sigma > 0:
            noise = torch.randn_like(vec) * noise_sigma * scale
            vec = vec + noise

        def inj_hook(m, i, o, v=vec):
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h

        inp = tok(prompt, return_tensors='pt').to(device)
        handles = [model.model.layers[inject_layer].register_forward_hook(inj_hook)]

        # Add intermediate noise at noise_layer
        if noise_layer is not None and noise_layer != inject_layer and noise_sigma > 0:
            hs_size = vec0.shape[0]
            noise_vec = torch.randn(hs_size, device=device) * noise_sigma * scale
            def noise_hook(m, i, o, nv=noise_vec):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = h[0, -1, :] + nv.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handles.append(model.model.layers[noise_layer].register_forward_hook(noise_hook))

        with torch.no_grad():
            out = model(**inp)
        for h in handles:
            h.remove()

        # Apply temperature
        logits = out.logits[0, -1, :].float() / temperature
        probs = torch.softmax(logits, dim=-1)
        p_vals.append(float(probs[tok_id]))

    p_arr = np.array(p_vals)
    return (p_arr.max() - p_arr.min()) / 2.0


def main():
    print("[Q13] Temperature & Decoherence Study")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    hs = model.config.hidden_size
    print("  Model: hidden_size=%d" % hs)

    min_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                ("min(4,6)=","4"),("min(9,3)=","3")]
    max_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                ("min(4,6)=","6"),("min(9,3)=","9")]
    print("  Training basis vectors...")
    min_vec = train_soul(model, tok, min_data, DEVICE, layer=INJECT_LAYER, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, layer=INJECT_LAYER, seed=99)

    prompt = "min(7,2)="
    tok_id = tok.encode("2")[-1]

    # === Experiment 1: Noise-induced decoherence ===
    print("\n  [Exp1] Noise-induced decoherence (noise on injection vector)...")
    noise_sigmas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    noise_amps = []
    for sigma in noise_sigmas:
        # Average over 5 noise seeds
        amps = []
        for seed in range(5):
            torch.manual_seed(seed)
            amp = measure_amplitude(model, tok, prompt, DEVICE, min_vec, max_vec,
                                     INJECT_LAYER, tok_id, noise_sigma=sigma)
            amps.append(amp)
        avg_amp = np.mean(amps)
        noise_amps.append(avg_amp)
        print("    sigma=%.2f: amp=%.4f" % (sigma, avg_amp))

    # Find decoherence threshold (50% amplitude drop)
    baseline_amp = noise_amps[0]
    half_amp = baseline_amp / 2
    decoherence_sigma = None
    for i, (s, a) in enumerate(zip(noise_sigmas, noise_amps)):
        if a <= half_amp:
            decoherence_sigma = s
            break

    # === Experiment 2: Temperature sensitivity ===
    print("\n  [Exp2] Temperature sensitivity (output logit temperature)...")
    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    temp_amps = []
    for T in temperatures:
        amp = measure_amplitude(model, tok, prompt, DEVICE, min_vec, max_vec,
                                 INJECT_LAYER, tok_id, temperature=T)
        temp_amps.append(amp)
        print("    T=%.1f: amp=%.4f" % (T, amp))

    # === Experiment 3: Decoherence by distance ===
    print("\n  [Exp3] Decoherence by layer distance...")
    sigma_mid = 0.2  # medium noise
    dist_layers = list(range(INJECT_LAYER + 1, INJECT_LAYER + 10))  # L9 to L17
    dist_amps = []
    for nl in dist_layers:
        amps = []
        for seed in range(3):
            torch.manual_seed(seed)
            amp = measure_amplitude(model, tok, prompt, DEVICE, min_vec, max_vec,
                                     INJECT_LAYER, tok_id, noise_sigma=sigma_mid,
                                     noise_layer=nl)
            amps.append(amp)
        avg_amp = np.mean(amps)
        dist_amps.append(avg_amp)
        print("    noise@L%d (dist=%d): amp=%.4f" % (nl, nl - INJECT_LAYER, avg_amp))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Noise decoherence curve
    ax = axes[0]
    ax.semilogx(noise_sigmas, noise_amps, '#E91E63', lw=2, marker='o', ms=8)
    ax.axhline(baseline_amp / 2, color='gray', linestyle='--', lw=1.5,
               label='50%% threshold (%.4f)' % (baseline_amp/2))
    if decoherence_sigma:
        ax.axvline(decoherence_sigma, color='red', linestyle=':', lw=2,
                   label='Decoherence sigma=%.2f' % decoherence_sigma)
    ax.set_xlabel('Noise sigma (log scale)', fontsize=11)
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax.set_title('Noise-Induced Decoherence\n"Quantum -> Classical transition"', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Temperature sensitivity
    ax = axes[1]
    ax.plot(temperatures, temp_amps, '#9C27B0', lw=2, marker='s', ms=8)
    ax.axvline(1.0, color='gray', linestyle='--', lw=1.5, label='T=1.0 (standard)')
    ax.set_xlabel('Output Temperature', fontsize=11)
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax.set_title('Temperature Sensitivity\nCold=sharp, Hot=diffuse', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 3: Decoherence by distance
    ax = axes[2]
    distances = [nl - INJECT_LAYER for nl in dist_layers]
    ax.plot(distances, dist_amps, '#2196F3', lw=2, marker='^', ms=8)
    ax.axhline(noise_amps[0], color='gray', linestyle='--', lw=1.5,
               label='No distance noise (sig=0)')
    # Fit exponential decay
    if len(dist_amps) > 2 and dist_amps[0] > 0:
        try:
            log_amps = np.log(np.array(dist_amps) + 1e-6)
            coeffs = np.polyfit(distances, log_amps, 1)
            decay_rate = -coeffs[0]
            ax.plot(distances, np.exp(np.polyval(coeffs, distances)),
                    'r--', lw=1.5, label='Exp decay rate=%.3f/layer' % decay_rate)
        except Exception:
            decay_rate = 0
    ax.set_xlabel('Distance from injection (layers)', fontsize=11)
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax.set_title('Coherence Length\n"How far does superposition survive?"', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        'Phase Q13: Temperature & Decoherence Study\n'
        'Characterizing S-Qubit fragility',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q13_decoherence_study.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q13', 'name': 'decoherence_study',
        'inject_layer': INJECT_LAYER,
        'exp1_noise': {'sigmas': noise_sigmas,
                       'amplitudes': [round(a, 6) for a in noise_amps],
                       'decoherence_sigma': decoherence_sigma,
                       'baseline_amp': round(baseline_amp, 6)},
        'exp2_temperature': {'temperatures': temperatures,
                              'amplitudes': [round(a, 6) for a in temp_amps]},
        'exp3_distance': {'layers': dist_layers,
                          'distances': distances,
                          'amplitudes': [round(a, 6) for a in dist_amps]},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q13_decoherence_study.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("\n  Q13 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
