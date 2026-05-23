# -*- coding: utf-8 -*-
"""
Phase Q16: Statistical Validation of Super-Quantum CHSH (S > 2.828)

Q15 found S = 3.4060 with a specific random seed.
Q16 validates this across multiple SQ2 training seeds.

Protocol:
  - Fix SQ1 (trained at L8, seed=42/99) - same as Q15
  - Train SQ2 at L20 with 5 different seed pairs
  - For each seed: run compact 2D phi sweep (13x13=169 points)
  - Report mean S +/- std across seeds
  - Statistical significance: is S > 2.828 in all/most seeds?
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
SQ2_LAYER = 20
SQ1_POS   = -1
SQ2_POS   = -2
N_PHI_2D  = 13  # compact for speed
N_SEEDS    = 5  # number of SQ2 seed pairs to test


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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def compute_chsh_from_joint(joint_E, n_phi):
    """Grid search CHSH S over all angle combinations."""
    best_S = 0
    for i1 in range(n_phi):
        for i2 in range(n_phi):
            for j1 in range(n_phi):
                for j2 in range(n_phi):
                    S = abs(joint_E[i1,j1] - joint_E[i1,j2] + joint_E[i2,j1] + joint_E[i2,j2])
                    if S > best_S:
                        best_S = S
    return float(best_S)


def run_one_seed(model, tok, prompt, sq1_min, sq1_max, sq2_0_data, sq2_1_data,
                  sq1_tok, sq1_tok_1, seed0, seed1, n_phi=N_PHI_2D, epochs=100):
    sq2_0 = train_soul(model, tok, sq2_0_data, DEVICE, SQ2_LAYER, SQ2_POS, epochs, seed0)
    sq2_1 = train_soul(model, tok, sq2_1_data, DEVICE, SQ2_LAYER, SQ2_POS, epochs, seed1)

    phis = np.linspace(0, 2 * np.pi, n_phi)
    joint_E = np.zeros((n_phi, n_phi))
    for i, phi1 in enumerate(phis):
        for j, phi2 in enumerate(phis):
            v1 = phi_vec(phi1, sq1_min, sq1_max)
            v2 = phi_vec(phi2, sq2_0, sq2_1)
            probs = two_qubit_forward(model, tok, prompt, DEVICE,
                                       v1, v2, SQ1_LAYER, SQ1_POS, SQ2_LAYER, SQ2_POS)
            joint_E[i, j] = float(probs[sq1_tok]) - float(probs[sq1_tok_1])

    S = compute_chsh_from_joint(joint_E, n_phi)
    E_range = (float(joint_E.min()), float(joint_E.max()))
    return S, E_range, joint_E


def main():
    print("[Q16] Statistical Validation: Is S > 2.828 reproducible?")
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

    # Fixed SQ1 (same as Q15)
    print("  Training SQ1@L%d (fixed, 150 epochs)..." % SQ1_LAYER)
    sq1_min = train_soul(model, tok, sq1_0_data, DEVICE, SQ1_LAYER, SQ1_POS, 150, 42)
    sq1_max = train_soul(model, tok, sq1_1_data, DEVICE, SQ1_LAYER, SQ1_POS, 150, 99)

    # Test SQ2 with different seeds
    seed_pairs = [(42, 99), (1, 2), (10, 20), (100, 200), (7, 77)]
    S_values = []
    all_joint_E = []
    results_by_seed = {}

    for i, (s0, s1) in enumerate(seed_pairs):
        print("  Seed pair (%d,%d)..." % (s0, s1))
        S, E_range, joint_E = run_one_seed(
            model, tok, prompt, sq1_min, sq1_max,
            sq2_0_data, sq2_1_data, sq1_tok, sq1_tok_1,
            s0, s1, n_phi=N_PHI_2D, epochs=100)
        S_values.append(S)
        all_joint_E.append(joint_E)
        print("    S=%.4f  E:[%.4f,%.4f]  > 2.828: %s  > 2.000: %s" % (
            S, E_range[0], E_range[1], S > 2.828, S > 2.0))
        results_by_seed[str(i)] = {
            'seeds': [s0, s1], 'S': round(S, 6),
            'E_min': round(E_range[0], 4), 'E_max': round(E_range[1], 4),
            'violates_quantum': bool(S > 2.828), 'violates_classical': bool(S > 2.0),
        }

    S_arr = np.array(S_values)
    mean_S = float(S_arr.mean())
    std_S  = float(S_arr.std())
    min_S  = float(S_arr.min())
    max_S  = float(S_arr.max())
    n_quantum_violated = int((S_arr > 2.828).sum())
    n_classical_violated = int((S_arr > 2.0).sum())

    print()
    print("  STATISTICAL SUMMARY:")
    print("    S: mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (mean_S, std_S, min_S, max_S))
    print("    Quantum violation (S>2.828): %d/%d seeds" % (n_quantum_violated, N_SEEDS))
    print("    Classical violation (S>2.000): %d/%d seeds" % (n_classical_violated, N_SEEDS))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: S values by seed (bar chart)
    ax = axes[0]
    bars = ax.bar(range(N_SEEDS), S_values,
                  color=['#E91E63' if s > 2.828 else '#2196F3' if s > 2.0 else '#9E9E9E'
                         for s in S_values],
                  edgecolor='black', alpha=0.85, width=0.6)
    ax.axhline(2.0, color='blue', linestyle='--', lw=2, label='Classical bound S=2')
    ax.axhline(2 * np.sqrt(2), color='green', linestyle=':', lw=2,
               label='Quantum max S=2sqrt2=%.3f' % (2*np.sqrt(2)))
    ax.axhline(mean_S, color='red', linestyle='-', lw=1.5,
               label='Mean S=%.4f +/- %.4f' % (mean_S, std_S))
    for bar, val in zip(bars, S_values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                '%.3f' % val, ha='center', fontweight='bold', fontsize=10)
    ax.set_xticks(range(N_SEEDS))
    ax.set_xticklabels(['Seeds\n%d,%d' % p for p in seed_pairs], fontsize=9)
    ax.set_ylabel('CHSH S value', fontsize=11)
    ax.set_title('CHSH S Across %d SQ2 Seeds\nRed=super-quantum, Blue=classical violation' % N_SEEDS,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 4.5)
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Best joint E heatmap
    best_idx = int(np.argmax(S_values))
    ax = axes[1]
    vmax = max(abs(all_joint_E[best_idx].min()), abs(all_joint_E[best_idx].max()))
    im = ax.imshow(all_joint_E[best_idx], aspect='auto', cmap='RdBu',
                   vmin=-vmax, vmax=vmax, extent=[0, 2, 0, 2])
    plt.colorbar(im, ax=ax, label='E(phi1,phi2)')
    ax.set_xlabel('phi2 / pi (SQ2@L20)', fontsize=11)
    ax.set_ylabel('phi1 / pi (SQ1@L8)', fontsize=11)
    ax.set_title('Best 2D Interference Map (S=%.4f)\nSeed pair: %s' % (
        S_values[best_idx], str(seed_pairs[best_idx])), fontweight='bold')

    # Panel 3: Summary table
    ax = axes[2]
    ax.axis('off')
    table_data = [['Seed pair', 'S value', '>2.000', '>2.828']]
    for sp, S in zip(seed_pairs, S_values):
        table_data.append([
            str(sp), '%.4f' % S,
            'YES' if S > 2.0 else 'NO',
            'YES' if S > 2.828 else 'NO',
        ])
    table_data.append(['MEAN', '%.4f +/- %.4f' % (mean_S, std_S),
                       '%d/%d' % (n_classical_violated, N_SEEDS),
                       '%d/%d' % (n_quantum_violated, N_SEEDS)])
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r == len(table_data) - 1:
            cell.set_facecolor('#FFF9C4')
        elif c in [2, 3]:
            text = cell.get_text().get_text()
            if text == 'YES':
                cell.set_facecolor('#E8F5E9')
    ax.set_title('Statistical Validation Summary\n"%d/%d seeds show S > 2.828"' % (
        n_quantum_violated, N_SEEDS), fontweight='bold', y=0.97)

    plt.suptitle(
        'Phase Q16: Statistical Validation of Super-Quantum CHSH\n'
        'SQ1@L8,pos=-1  x  SQ2@L20,pos=-2  |  Mean S=%.4f +/- %.4f' % (mean_S, std_S),
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q16_chsh_statistical_validation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q16', 'name': 'chsh_statistical_validation',
        'sq1_layer': SQ1_LAYER, 'sq2_layer': SQ2_LAYER,
        'n_seeds': N_SEEDS, 'n_phi': N_PHI_2D,
        'S_values': [round(s, 6) for s in S_values],
        'mean_S': round(mean_S, 6), 'std_S': round(std_S, 6),
        'min_S': round(min_S, 6), 'max_S': round(max_S, 6),
        'classical_violations': n_classical_violated,
        'quantum_violations': n_quantum_violated,
        'seed_results': results_by_seed,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q16_chsh_statistical_validation.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q16 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
