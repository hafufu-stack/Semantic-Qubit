# -*- coding: utf-8 -*-
"""
Phase Q75: Entanglement Entropy Scaling (Area Law vs Volume Law)
=================================================================
In physical quantum systems, entanglement entropy follows an
"area law" for ground states. Critical systems show volume-law scaling.

Test: How does entanglement entropy scale with the number of 
S-Qubit dimensions involved? Does it follow area or volume law?
This has deep implications for the computational power of S-Qubits.
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
    print("[Q75] Entanglement Entropy Scaling (Area vs Volume Law)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    hs = model.config.hidden_size

    # Train multiple S-Qubits
    task_data = [
        ([("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")], 42),
        ([("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")], 99),
        ([("2+3=", "5"), ("1+4=", "5"), ("3+2=", "5")], 77),
        ([("7-3=", "4"), ("9-5=", "4"), ("6-2=", "4")], 33),
    ]
    vecs = []
    for data, seed in task_data:
        v = train_soul(model, tok, data, DEVICE, INJECT_LAYER, EPOCHS, seed)
        vecs.append(v)

    # Stack vectors into a matrix
    V = torch.stack(vecs)  # (n_tasks, hs)

    # Compute entanglement entropy for different subsystem sizes
    # Treat the hidden dimensions as a bipartite system A|B
    # Entropy of reduced density matrix rho_A = Tr_B(|psi><psi|)

    subsystem_sizes = np.unique(np.logspace(0, np.log10(hs//2), 30).astype(int))
    entropy_results = []

    print("  Computing entanglement entropy for %d subsystem sizes..." % len(subsystem_sizes))

    for L in subsystem_sizes:
        L = int(L)
        # Take the first L dimensions as subsystem A
        V_A = V[:, :L]  # (n_tasks, L)
        V_B = V[:, L:]   # (n_tasks, hs-L)

        # Compute reduced density matrix via SVD of the "state"
        # Reshape V as a bipartite state: (L, hs-L) per task
        # Average over tasks to get ensemble entropy
        entropies = []
        for v in vecs:
            # Reshape vector as matrix (L, hs-L)
            v_np = v.cpu().float().numpy()
            # Create bipartite state
            v_A = v_np[:L]
            v_B = v_np[L:]

            # Schmidt decomposition via SVD of reshaped state
            min_dim = min(L, len(v_B))
            if min_dim < 2:
                continue

            # Create correlation matrix
            C = np.outer(v_A[:min_dim], v_B[:min_dim])
            try:
                U, sigma, Vt = np.linalg.svd(C, full_matrices=False)
                # Normalize singular values to get Schmidt coefficients
                sigma_sq = sigma ** 2
                sigma_sq = sigma_sq / (sigma_sq.sum() + 1e-30)
                sigma_sq = sigma_sq[sigma_sq > 1e-15]
                # Von Neumann entropy
                S = -np.sum(sigma_sq * np.log2(sigma_sq + 1e-30))
                entropies.append(S)
            except:
                pass

        if entropies:
            avg_entropy = np.mean(entropies)
            entropy_results.append({
                'L': int(L),
                'entropy': float(avg_entropy),
                'log_L': float(np.log2(L)),
            })

    # Fit area law (S ~ log(L)) and volume law (S ~ L)
    if len(entropy_results) > 5:
        Ls = np.array([r['L'] for r in entropy_results])
        Ss = np.array([r['entropy'] for r in entropy_results])

        # Area law fit: S = a * log(L) + b
        log_Ls = np.log2(Ls)
        area_fit = np.polyfit(log_Ls, Ss, 1)
        area_pred = np.polyval(area_fit, log_Ls)
        area_r2 = 1 - np.sum((Ss - area_pred)**2) / np.sum((Ss - np.mean(Ss))**2)

        # Volume law fit: S = a * L + b
        vol_fit = np.polyfit(Ls, Ss, 1)
        vol_pred = np.polyval(vol_fit, Ls)
        vol_r2 = 1 - np.sum((Ss - vol_pred)**2) / np.sum((Ss - np.mean(Ss))**2)

        print("\n  RESULTS:")
        print("    Area law (S ~ log L): R2=%.4f, slope=%.4f" % (area_r2, area_fit[0]))
        print("    Volume law (S ~ L): R2=%.4f, slope=%.6f" % (vol_r2, vol_fit[0]))
        if area_r2 > vol_r2:
            print("    -> AREA LAW (ground state behavior)")
            scaling = 'area'
        else:
            print("    -> VOLUME LAW (critical/thermal behavior)")
            scaling = 'volume'
    else:
        area_r2, vol_r2 = 0, 0
        area_fit, vol_fit = [0, 0], [0, 0]
        scaling = 'unknown'

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    Ls_plot = [r['L'] for r in entropy_results]
    Ss_plot = [r['entropy'] for r in entropy_results]

    # (a) Entropy vs subsystem size
    ax = axes[0]
    ax.plot(Ls_plot, Ss_plot, 'o-', color='#FF5722', linewidth=2, markersize=5)
    ax.set_xlabel('Subsystem size L (dimensions)')
    ax.set_ylabel('Entanglement entropy S (bits)')
    ax.set_title('(a) Entropy vs Subsystem Size\n'
                 'S(L) reveals quantum structure',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # (b) Area law test (S vs log L)
    ax = axes[1]
    log_Ls_plot = np.log2(np.array(Ls_plot) + 1)
    ax.plot(log_Ls_plot, Ss_plot, 'o', color='#FF5722', markersize=6)
    if len(entropy_results) > 5:
        x_fit = np.linspace(min(log_Ls_plot), max(log_Ls_plot), 100)
        ax.plot(x_fit, np.polyval(area_fit, x_fit), '--', color='#2196F3',
                linewidth=2, label='Area law fit (R2=%.3f)' % area_r2)
    ax.set_xlabel('log2(L)')
    ax.set_ylabel('Entropy S (bits)')
    ax.set_title('(b) Area Law Test\nS ~ log(L) for ground states',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Volume law test (S vs L)
    ax = axes[2]
    ax.plot(Ls_plot, Ss_plot, 'o', color='#FF5722', markersize=6)
    if len(entropy_results) > 5:
        x_fit = np.linspace(min(Ls_plot), max(Ls_plot), 100)
        ax.plot(x_fit, np.polyval(vol_fit, x_fit), '--', color='#4CAF50',
                linewidth=2, label='Volume law fit (R2=%.3f)' % vol_r2)
    ax.set_xlabel('Subsystem size L')
    ax.set_ylabel('Entropy S (bits)')
    ax.set_title('(c) Volume Law Test\n'
                 'Result: %s law (R2=%.3f)' % (scaling.upper(),
                 max(area_r2, vol_r2)),
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle('Phase Q75: Entanglement Entropy Scaling\n'
                 'Area vs Volume law in S-Qubit space',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q75_entropy_scaling.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q75', 'name': 'entanglement_entropy_scaling',
        'area_law_r2': round(float(area_r2), 4),
        'volume_law_r2': round(float(vol_r2), 4),
        'scaling_type': scaling,
        'max_entropy': round(float(max(Ss_plot)) if Ss_plot else 0, 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q75_entropy_scaling.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q75 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
