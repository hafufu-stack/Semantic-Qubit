# -*- coding: utf-8 -*-
"""
Phase Q36: Dimension-Coherence Universal Law

Q17 established that d >= 1024 enables quantum-like behavior.
Q29 confirmed this with the NQPU.
Now: systematic study of the dimension-coherence relationship
across ALL S-Qubit metrics to find a universal scaling law.

Tests: interference visibility, CHSH S, fidelity, and entanglement
as a function of hidden dimension (using layers at different depths).

CPU-ONLY: Uses NQPU architecture for systematic dimension sweep.
"""
import json, os, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


class NQPU:
    """NQPU with configurable dimension."""
    def __init__(self, d=256, n_heads=4, n_layers=4, seed=42):
        np.random.seed(seed)
        self.d = d
        self.n_heads = min(n_heads, max(1, d // 16))
        self.head_d = d // self.n_heads
        self.n_layers = n_layers
        self.Wq = [np.random.randn(d, d) / np.sqrt(d) for _ in range(n_layers)]
        self.Wk = [np.random.randn(d, d) / np.sqrt(d) for _ in range(n_layers)]
        self.Wv = [np.random.randn(d, d) / np.sqrt(d) for _ in range(n_layers)]
        self.Wo = [np.random.randn(d, d) / np.sqrt(d) for _ in range(n_layers)]
        self.W1 = [np.random.randn(d, d*4) / np.sqrt(d) for _ in range(n_layers)]
        self.W2 = [np.random.randn(d*4, d) / np.sqrt(d*4) for _ in range(n_layers)]

    def forward(self, x):
        for l in range(self.n_layers):
            Q = x @ self.Wq[l]; K = x @ self.Wk[l]; V = x @ self.Wv[l]
            seq_len = x.shape[0]
            Q = Q.reshape(seq_len, self.n_heads, self.head_d)
            K = K.reshape(seq_len, self.n_heads, self.head_d)
            V = V.reshape(seq_len, self.n_heads, self.head_d)
            out_heads = []
            for h in range(self.n_heads):
                scores = Q[:, h, :] @ K[:, h, :].T / np.sqrt(self.head_d)
                attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
                attn /= attn.sum(axis=-1, keepdims=True)
                out_heads.append(attn @ V[:, h, :])
            attn_out = np.concatenate(out_heads, axis=-1) @ self.Wo[l]
            x = x + attn_out
            h = np.maximum(0, x @ self.W1[l])
            x = x + h @ self.W2[l]
        return x


def main():
    print("[Q36] Dimension-Coherence Universal Law")
    start = time.time()

    # Extended dimension sweep
    dims = [16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048]
    n_seeds = 5  # average over random seeds
    n_phi = 37
    phis = np.linspace(0, 2 * np.pi, n_phi)
    seq_len = 8

    all_results = []

    for d in dims:
        print("\n  Testing d=%d..." % d)
        seed_results = []

        for seed in range(n_seeds):
            nqpu = NQPU(d=d, n_heads=4, n_layers=4, seed=seed * 100 + 42)

            np.random.seed(seed * 100 + 123)
            base_input = np.random.randn(seq_len, d) * 0.1
            basis_0 = np.random.randn(d); basis_0 /= np.linalg.norm(basis_0)
            basis_1 = np.random.randn(d)
            basis_1 -= np.dot(basis_1, basis_0) * basis_0
            basis_1 /= np.linalg.norm(basis_1)

            # 1. Interference sweep
            E_vals = []
            for phi in phis:
                sv = np.cos(phi / 2) * basis_0 + np.sin(phi / 2) * basis_1
                x = base_input.copy(); x[-1, :] = sv
                out = nqpu.forward(x)
                state = out[-1, :]
                E = (np.dot(state, basis_0) - np.dot(state, basis_1)) / (np.linalg.norm(state) + 1e-10)
                E_vals.append(E)
            E_arr = np.array(E_vals)
            visibility = (E_arr.max() - E_arr.min()) / (abs(E_arr.max()) + abs(E_arr.min()) + 1e-10)
            amplitude = (E_arr.max() - E_arr.min()) / 2

            # Cosine fit R^2
            from scipy.optimize import curve_fit
            try:
                def cos_model(x, A, B, C):
                    return A * np.cos(x + B) + C
                popt, _ = curve_fit(cos_model, phis, E_arr, p0=[amplitude, 0, E_arr.mean()])
                E_fit = cos_model(phis, *popt)
                ss_res = np.sum((E_arr - E_fit)**2)
                ss_tot = np.sum((E_arr - E_arr.mean())**2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
            except Exception:
                r2 = 0.0

            # 2. State fidelity (inject same state twice, compare outputs)
            sv_test = (basis_0 + basis_1) / np.sqrt(2)
            x1 = base_input.copy(); x1[-1, :] = sv_test
            out1 = nqpu.forward(x1)
            x2 = base_input.copy(); x2[-1, :] = sv_test
            out2 = nqpu.forward(x2)
            fidelity = np.dot(out1[-1, :], out2[-1, :]) / (
                np.linalg.norm(out1[-1, :]) * np.linalg.norm(out2[-1, :]) + 1e-10)

            # 3. Two-qubit CHSH (simplified: 4 angle pairs)
            basis_0b = np.random.randn(d); basis_0b /= np.linalg.norm(basis_0b)
            basis_1b = np.random.randn(d)
            basis_1b -= np.dot(basis_1b, basis_0b) * basis_0b
            basis_1b /= np.linalg.norm(basis_1b)

            chsh_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            E_joint = np.zeros((4, 4))
            for ia, phi_a in enumerate(chsh_angles):
                for ib, phi_b in enumerate(chsh_angles):
                    sv_a = np.cos(phi_a/2) * basis_0 + np.sin(phi_a/2) * basis_1
                    sv_b = np.cos(phi_b/2) * basis_0b + np.sin(phi_b/2) * basis_1b
                    x = base_input.copy()
                    x[-1, :] = sv_a; x[-2, :] = sv_b
                    out = nqpu.forward(x)
                    state = out[-1, :]
                    E_joint[ia, ib] = (np.dot(state, basis_0) - np.dot(state, basis_1)) / (
                        np.linalg.norm(state) + 1e-10)

            best_S = 0
            for ia1 in range(4):
                for ia2 in range(4):
                    for ib1 in range(4):
                        for ib2 in range(4):
                            S = E_joint[ia1,ib1] - E_joint[ia1,ib2] + E_joint[ia2,ib1] + E_joint[ia2,ib2]
                            if abs(S) > abs(best_S):
                                best_S = S

            # 4. Entanglement measure: mutual information between positions
            n_mi = 20
            mi_phis = np.linspace(0, 2*np.pi, n_mi)
            proj_a, proj_b = [], []
            for phi in mi_phis:
                sv = np.cos(phi/2) * basis_0 + np.sin(phi/2) * basis_1
                x = base_input.copy(); x[-1, :] = sv
                out = nqpu.forward(x)
                proj_a.append(np.dot(out[-1, :], basis_0))
                proj_b.append(np.dot(out[-2, :], basis_0b))
            corr_ab = abs(np.corrcoef(proj_a, proj_b)[0, 1])

            seed_results.append({
                'visibility': visibility,
                'amplitude': amplitude,
                'cos_r2': r2,
                'fidelity': fidelity,
                'chsh_S': best_S,
                'cross_corr': corr_ab,
            })

        # Average over seeds
        avg = {}
        std = {}
        for key in seed_results[0]:
            vals = [r[key] for r in seed_results]
            avg[key] = float(np.mean(vals))
            std[key] = float(np.std(vals))

        result = {'d': d, 'avg': avg, 'std': std}
        all_results.append(result)
        print("    vis=%.3f+/-%.3f, R2=%.3f, S=%.3f, fid=%.4f" % (
            avg['visibility'], std['visibility'], avg['cos_r2'],
            avg['chsh_S'], avg['fidelity']))

    # ── PLOT ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ds = [r['d'] for r in all_results]

    # Panel A: Visibility vs dimension
    ax = axes[0, 0]
    vis_avg = [r['avg']['visibility'] for r in all_results]
    vis_std = [r['std']['visibility'] for r in all_results]
    ax.errorbar(ds, vis_avg, yerr=vis_std, fmt='ro-', lw=2, ms=8, capsize=4)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Interference Visibility')
    ax.set_title('(a) Visibility vs Dimension', fontweight='bold')
    ax.set_xscale('log', base=2); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Panel B: Cosine R^2 vs dimension
    ax = axes[0, 1]
    r2_avg = [r['avg']['cos_r2'] for r in all_results]
    r2_std = [r['std']['cos_r2'] for r in all_results]
    ax.errorbar(ds, r2_avg, yerr=r2_std, fmt='bs-', lw=2, ms=8, capsize=4)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Cosine Fit R^2')
    ax.set_title('(b) Coherence (R^2) vs Dimension', fontweight='bold')
    ax.set_xscale('log', base=2); ax.grid(alpha=0.3)
    ax.axhline(0.9, color='green', ls='--', alpha=0.5, label='High coherence threshold')
    ax.legend()

    # Panel C: CHSH S vs dimension
    ax = axes[1, 0]
    chsh_avg = [r['avg']['chsh_S'] for r in all_results]
    chsh_std = [r['std']['chsh_S'] for r in all_results]
    ax.errorbar(ds, chsh_avg, yerr=chsh_std, fmt='go-', lw=2, ms=8, capsize=4)
    ax.axhline(2.0, color='gray', ls='--', lw=1.5, label='Classical bound')
    ax.axhline(2.83, color='blue', ls='--', lw=1.5, label='Tsirelson bound')
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('CHSH S-value')
    ax.set_title('(c) CHSH S vs Dimension', fontweight='bold')
    ax.set_xscale('log', base=2); ax.legend(); ax.grid(alpha=0.3)

    # Panel D: Cross-position correlation
    ax = axes[1, 1]
    corr_avg = [r['avg']['cross_corr'] for r in all_results]
    corr_std = [r['std']['cross_corr'] for r in all_results]
    ax.errorbar(ds, corr_avg, yerr=corr_std, fmt='ms-', lw=2, ms=8, capsize=4)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Cross-position correlation')
    ax.set_title('(d) Entanglement Proxy vs Dimension', fontweight='bold')
    ax.set_xscale('log', base=2); ax.grid(alpha=0.3)

    plt.suptitle('Phase Q36: Dimension-Coherence Universal Law\n'
                 'Systematic study of quantum-like behavior across dimensions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q36_dimension_law.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q36', 'name': 'dimension_coherence_law',
        'dims_tested': dims,
        'n_seeds': n_seeds,
        'results': [{
            'd': r['d'],
            'visibility': round(r['avg']['visibility'], 6),
            'cos_r2': round(r['avg']['cos_r2'], 6),
            'chsh_S': round(r['avg']['chsh_S'], 4),
            'fidelity': round(r['avg']['fidelity'], 6),
            'cross_corr': round(r['avg']['cross_corr'], 6),
        } for r in all_results],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q36_dimension_law.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("\n  Q36 completed in %.0fs" % (time.time() - start))


if __name__ == '__main__':
    main()
