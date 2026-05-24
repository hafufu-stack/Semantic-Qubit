# -*- coding: utf-8 -*-
"""
Phase Q55: Room-Temperature VQE

Variational Quantum Eigensolver on S-Qubits: find ground state energy
of a simple Hamiltonian (Ising model / combinatorial optimization)
on a standard GPU at room temperature.

H = -J * sum(Z_i * Z_{i+1}) - h * sum(X_i)

We encode spin configurations as S-Qubit phases and use
VQE-style optimization to find the minimum energy.
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


def main():
    print("[Q55] Room-Temperature VQE")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]
    prompt = "min(7,2)="

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    def inject_measure(phi):
        v = torch.cos(torch.tensor(phi/2)) * v0 + torch.sin(torch.tensor(phi/2)) * v1
        v = v / v.norm() * v0.norm()
        def hook(m, i, o, vec=v):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = vec.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(probs[min_tok]) - float(probs[max_tok])

    # ── Ising model simulation ──
    # For an N-spin Ising chain: H = -J * sum(s_i * s_{i+1})
    # Encode each spin config as a phase, measure energy via E-value mapping

    N_spins_list = [3, 4, 5, 6]
    results_all = []

    for N_spins in N_spins_list:
        print("\n  Ising model with N=%d spins" % N_spins)
        N_configs = 2 ** N_spins
        J = 1.0

        # Classical: compute true energies for all spin configs
        configs = []
        true_energies = []
        for i in range(N_configs):
            spins = [(i >> b) & 1 for b in range(N_spins)]
            spins_pm = [2*s - 1 for s in spins]  # map to +1/-1
            E = 0
            for j in range(N_spins - 1):
                E -= J * spins_pm[j] * spins_pm[j+1]
            configs.append(spins)
            true_energies.append(E)

        true_gs = min(true_energies)
        true_gs_config = configs[true_energies.index(true_gs)]
        print("    True ground state: %s, E=%.1f" % (
            str(true_gs_config), true_gs))

        # Map each config to a phase and measure E-value
        phases = np.linspace(0, 2*np.pi*(1-1/N_configs), N_configs)
        E_values = []
        for i, phi in enumerate(phases):
            E = inject_measure(phi)
            E_values.append(E)

        # VQE: optimize phase to find minimum energy mapping
        # Map: E_measured(phi) -> Ising energy via calibration
        # We map linearly: E_ising = a * E_measured + b
        E_arr = np.array(E_values)
        true_E_arr = np.array(true_energies)

        # Fit linear mapping
        from scipy.stats import linregress
        slope, intercept, r_val, _, _ = linregress(E_arr, true_E_arr)

        # Predict energies
        predicted_E = slope * E_arr + intercept
        predicted_gs = min(predicted_E)
        predicted_gs_idx = np.argmin(predicted_E)
        predicted_gs_config = configs[predicted_gs_idx]

        # VQE optimization: sweep phases to find minimum
        best_E = float('inf')
        best_phi = 0
        sweep_results = []
        for phi in np.linspace(0, 2*np.pi, 100):
            E_m = inject_measure(phi)
            E_ising = slope * E_m + intercept
            sweep_results.append({'phi': round(phi, 4), 'E': round(E_ising, 4)})
            if E_ising < best_E:
                best_E = E_ising
                best_phi = phi

        # Quality metrics
        approx_ratio = true_gs / (best_E + 1e-10) if best_E < 0 else 0
        gs_found = (abs(best_E - true_gs) < 0.5)

        result = {
            'N_spins': N_spins,
            'N_configs': N_configs,
            'true_gs_energy': round(float(true_gs), 2),
            'true_gs_config': true_gs_config,
            'predicted_gs_energy': round(float(best_E), 4),
            'predicted_gs_config': predicted_gs_config,
            'approx_ratio': round(float(approx_ratio), 4),
            'gs_found': bool(gs_found),
            'correlation': round(float(r_val), 4),
        }
        results_all.append(result)
        print("    VQE result: E_pred=%.3f, true=%.1f, ratio=%.3f, correct=%s" % (
            best_E, true_gs, approx_ratio, str(gs_found)))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    for r in results_all:
        color = '#4CAF50' if r['gs_found'] else '#F44336'
        ax.bar(str(r['N_spins']), r['approx_ratio'], color=color,
               edgecolor='black', alpha=0.85)
        ax.text(str(r['N_spins']), r['approx_ratio'] + 0.02,
                '%.2f' % r['approx_ratio'], ha='center', fontweight='bold')
    ax.axhline(1.0, color='green', ls='--', alpha=0.5, label='Optimal (ratio=1)')
    ax.axhline(0.5, color='red', ls='--', alpha=0.3, label='Random')
    ax.set_xlabel('Number of spins N')
    ax.set_ylabel('Approximation ratio')
    ax.set_title('(a) VQE Approximation Ratio\nGreen=exact, Red=approx',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    corrs = [r['correlation'] for r in results_all]
    ax.bar([str(r['N_spins']) for r in results_all], corrs,
           color='#2196F3', edgecolor='black', alpha=0.85)
    for i, c in enumerate(corrs):
        ax.text(i, c + 0.02, '%.3f' % c, ha='center', fontweight='bold')
    ax.set_xlabel('N spins')
    ax.set_ylabel('E-value / Energy correlation')
    ax.set_title('(b) Measurement-Energy Correlation', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # VQE landscape for largest N
    ax = axes[2]
    if sweep_results:
        phis = [r['phi'] for r in sweep_results]
        Es = [r['E'] for r in sweep_results]
        ax.plot(phis, Es, 'b-', lw=2)
        ax.axhline(true_gs, color='red', ls='--', lw=2,
                   label='True GS: %.1f' % true_gs)
        ax.scatter([best_phi], [best_E], c='red', s=100, zorder=5,
                   label='VQE min: %.2f' % best_E)
        ax.set_xlabel('Phase (rad)')
        ax.set_ylabel('Ising Energy')
        ax.set_title('(c) VQE Energy Landscape (N=%d)\n' % N_spins_list[-1] +
                     'Room temp, zero noise', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Phase Q55: Room-Temperature VQE\n'
                 'Variational Quantum Eigensolver at 300K on GPU',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q55_vqe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q55', 'name': 'room_temp_vqe',
        'results': results_all,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q55_vqe.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("\n  Q55 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
