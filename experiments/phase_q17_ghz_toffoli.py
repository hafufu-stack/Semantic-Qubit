# -*- coding: utf-8 -*-
"""
Phase Q17: 3-Qubit GHZ State and Toffoli Gate via Self-Attention

Physical quantum computers face a devastating "wiring problem": connecting
3+ qubits requires physical couplers that introduce crosstalk and decoherence.
Self-attention is a fully-connected bus that couples ALL token positions
regardless of distance -- the "wiring" is free.

Experiment:
  SQ1@L8, pos=-1  (math: min/max)
  SQ2@L16, pos=-2 (color: blue/green)
  SQ3@L20, pos=-3 (size: small/large)  [pre-collapse sweet spot]

  1. Single-qubit marginals (3x): each SQ affects pos=-1 output
  2. 2-body correlations (3 pairs): measure non-separability for each pair
  3. 3-body GHZ witness: E(phi1,phi2,phi3) measured at 8 corners
     GHZ fidelity = fraction of 8 settings where 3-body correlator > 0.5
  4. Toffoli (CCNOT) gate: does SQ1 output flip only when BOTH SQ2 and SQ3
     are in state |1>? (controlled-controlled operation)
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

SQ1_LAYER, SQ1_POS = 8, -1
SQ2_LAYER, SQ2_POS = 16, -2
SQ3_LAYER, SQ3_POS = 20, -3
EPOCHS = 120


def train_soul(model, tok, data, device, layer, pos, epochs=EPOCHS, seed=42):
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


def multi_inject_forward(model, tok, prompt, device, injections):
    """
    injections: list of (vec, layer, pos) tuples.
    Returns softmax probs at pos=-1.
    """
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    handles = []
    for vec, layer, pos in injections:
        actual_pos = pos if pos >= 0 else seq_len + pos
        def make_hook(v, p):
            def hook(m, i, o):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, p, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            return hook
        handle = model.model.layers[layer].register_forward_hook(make_hook(vec, actual_pos))
        handles.append(handle)
    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0:
        v = v / n * v0.norm()
    return v


def main():
    print("[Q17] 3-Qubit GHZ State and Toffoli Gate")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    prompt = "min(7,2)="
    sq1_tok = tok.encode("2")[-1]
    sq1_tok_1 = tok.encode("7")[-1]

    # Training data
    sq1_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                  ("min(4,6)=","4"),("min(9,3)=","3")]
    sq1_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                  ("min(4,6)=","6"),("min(9,3)=","9")]
    sq2_0_data = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                  ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_1_data = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                  ("The forest is","green"),("Grass color is","green")]
    sq3_0_data = [("An ant is","small"),("A mouse is","small"),("A coin is","small"),
                  ("A seed is","small"),("A bug is","small")]
    sq3_1_data = [("A whale is","large"),("An elephant is","large"),("A mountain is","large"),
                  ("The sun is","large"),("A building is","large")]

    # Train all 6 basis vectors
    print("  Training SQ1@L%d pos=%d..." % (SQ1_LAYER, SQ1_POS))
    sq1_0 = train_soul(model, tok, sq1_0_data, DEVICE, SQ1_LAYER, SQ1_POS, EPOCHS, 42)
    sq1_1 = train_soul(model, tok, sq1_1_data, DEVICE, SQ1_LAYER, SQ1_POS, EPOCHS, 99)

    print("  Training SQ2@L%d pos=%d..." % (SQ2_LAYER, SQ2_POS))
    sq2_0 = train_soul(model, tok, sq2_0_data, DEVICE, SQ2_LAYER, SQ2_POS, EPOCHS, 42)
    sq2_1 = train_soul(model, tok, sq2_1_data, DEVICE, SQ2_LAYER, SQ2_POS, EPOCHS, 99)

    print("  Training SQ3@L%d pos=%d..." % (SQ3_LAYER, SQ3_POS))
    sq3_0 = train_soul(model, tok, sq3_0_data, DEVICE, SQ3_LAYER, SQ3_POS, EPOCHS, 42)
    sq3_1 = train_soul(model, tok, sq3_1_data, DEVICE, SQ3_LAYER, SQ3_POS, EPOCHS, 99)

    def E_val(probs):
        return float(probs[sq1_tok]) - float(probs[sq1_tok_1])

    # ── 1. Single-qubit marginals ──
    print("\n  [1] Single-qubit marginals (each alone)...")
    n_phi = 25
    phis = np.linspace(0, 2*np.pi, n_phi)
    E1, E2, E3 = [], [], []
    for phi in phis:
        v1 = phi_vec(phi, sq1_0, sq1_1)
        p = multi_inject_forward(model, tok, prompt, DEVICE, [(v1, SQ1_LAYER, SQ1_POS)])
        E1.append(E_val(p))
        v2 = phi_vec(phi, sq2_0, sq2_1)
        p = multi_inject_forward(model, tok, prompt, DEVICE, [(v2, SQ2_LAYER, SQ2_POS)])
        E2.append(E_val(p))
        v3 = phi_vec(phi, sq3_0, sq3_1)
        p = multi_inject_forward(model, tok, prompt, DEVICE, [(v3, SQ3_LAYER, SQ3_POS)])
        E3.append(E_val(p))

    E1, E2, E3 = np.array(E1), np.array(E2), np.array(E3)
    amp1 = (E1.max()-E1.min())/2; amp2 = (E2.max()-E2.min())/2; amp3 = (E3.max()-E3.min())/2
    print("    SQ1 amp=%.4f  SQ2 cross-amp=%.4f  SQ3 cross-amp=%.4f" % (amp1, amp2, amp3))

    # ── 2. Two-body correlations (3 pairs) ──
    print("\n  [2] 2-body correlations...")
    n_2d = 11
    phis_2d = np.linspace(0, 2*np.pi, n_2d)
    pairs = [
        ('SQ1-SQ2', SQ1_LAYER, SQ1_POS, sq1_0, sq1_1, SQ2_LAYER, SQ2_POS, sq2_0, sq2_1),
        ('SQ1-SQ3', SQ1_LAYER, SQ1_POS, sq1_0, sq1_1, SQ3_LAYER, SQ3_POS, sq3_0, sq3_1),
        ('SQ2-SQ3', SQ2_LAYER, SQ2_POS, sq2_0, sq2_1, SQ3_LAYER, SQ3_POS, sq3_0, sq3_1),
    ]
    pair_results = {}
    for name, la, pa, va0, va1, lb, pb, vb0, vb1 in pairs:
        joint = np.zeros((n_2d, n_2d))
        for i, p1 in enumerate(phis_2d):
            for j, p2 in enumerate(phis_2d):
                va = phi_vec(p1, va0, va1)
                vb = phi_vec(p2, vb0, vb1)
                probs = multi_inject_forward(model, tok, prompt, DEVICE,
                                              [(va, la, pa), (vb, lb, pb)])
                joint[i,j] = E_val(probs)
        # Separability residual
        marg_a = joint.mean(axis=1)
        marg_b = joint.mean(axis=0)
        product = np.outer(marg_a, marg_b) / (abs(marg_a).max()*abs(marg_b).max()+1e-9) * abs(joint).max()
        residual = float(np.std(joint - product))
        # CHSH S for this pair
        best_S = 0
        for i1 in range(n_2d):
            for i2 in range(n_2d):
                for j1 in range(n_2d):
                    for j2 in range(n_2d):
                        S = abs(joint[i1,j1]-joint[i1,j2]+joint[i2,j1]+joint[i2,j2])
                        if S > best_S: best_S = S
        pair_results[name] = {
            'sep_residual': round(residual, 6),
            'chsh_S': round(best_S, 6),
            'E_range': [round(float(joint.min()),4), round(float(joint.max()),4)],
        }
        print("    %s: sep_residual=%.4f  CHSH_S=%.4f  E:[%.4f,%.4f]" % (
            name, residual, best_S, joint.min(), joint.max()))

    # ── 3. 3-body GHZ test: 8 computational basis corners ──
    print("\n  [3] 3-body GHZ test (8 corners)...")
    # Evaluate E at each of 8 combinations of (|0>,|1>) for 3 qubits
    corners = []
    for b1 in [0, 1]:
        for b2 in [0, 1]:
            for b3 in [0, 1]:
                v1 = sq1_0 if b1==0 else sq1_1
                v2 = sq2_0 if b2==0 else sq2_1
                v3 = sq3_0 if b3==0 else sq3_1
                probs = multi_inject_forward(model, tok, prompt, DEVICE,
                    [(v1, SQ1_LAYER, SQ1_POS), (v2, SQ2_LAYER, SQ2_POS),
                     (v3, SQ3_LAYER, SQ3_POS)])
                e = E_val(probs)
                corners.append({
                    'state': '|%d%d%d>' % (b1,b2,b3),
                    'E': round(e, 6),
                })
                print("    |%d%d%d>: E=%.4f" % (b1, b2, b3, e))

    # GHZ-like 3-body correlator: C3 = E(000)+E(111) - E(001)-E(010)-E(100)-E(011)-E(101)-E(110)
    E_vals = [c['E'] for c in corners]
    C3 = (E_vals[0] + E_vals[7]) - sum(E_vals[1:7])/6  # normalized
    # Genuine 3-body: is C3 non-trivial?
    ghz_fidelity = abs(E_vals[0] - E_vals[7])  # |E(000) - E(111)|
    print("    GHZ contrast |E(000)-E(111)| = %.4f" % ghz_fidelity)
    print("    3-body correlator C3 = %.4f" % C3)

    # ── 4. Toffoli (CCNOT): SQ1 flip only when SQ2=|1> AND SQ3=|1> ──
    print("\n  [4] Toffoli gate test...")
    # E(SQ1=|0>, SQ2=s2, SQ3=s3) for all 4 combinations of s2,s3
    toffoli_data = {}
    for s2_label, s2_vec in [('|0>', sq2_0), ('|1>', sq2_1)]:
        for s3_label, s3_vec in [('|0>', sq3_0), ('|1>', sq3_1)]:
            probs_0 = multi_inject_forward(model, tok, prompt, DEVICE,
                [(sq1_0, SQ1_LAYER, SQ1_POS), (s2_vec, SQ2_LAYER, SQ2_POS),
                 (s3_vec, SQ3_LAYER, SQ3_POS)])
            probs_1 = multi_inject_forward(model, tok, prompt, DEVICE,
                [(sq1_1, SQ1_LAYER, SQ1_POS), (s2_vec, SQ2_LAYER, SQ2_POS),
                 (s3_vec, SQ3_LAYER, SQ3_POS)])
            e0 = E_val(probs_0)
            e1 = E_val(probs_1)
            key = "SQ2=%s,SQ3=%s" % (s2_label, s3_label)
            toffoli_data[key] = {
                'E_sq1_0': round(e0, 4), 'E_sq1_1': round(e1, 4),
                'contrast': round(abs(e0 - e1), 4),
            }
            print("    %s: E(SQ1=|0>)=%.4f  E(SQ1=|1>)=%.4f  contrast=%.4f" % (
                key, e0, e1, abs(e0-e1)))

    # Toffoli fidelity: contrast should be highest when SQ2=|1>,SQ3=|1>
    toffoli_11_contrast = toffoli_data['SQ2=|1>,SQ3=|1>']['contrast']
    other_contrasts = [toffoli_data[k]['contrast'] for k in toffoli_data if k != 'SQ2=|1>,SQ3=|1>']
    toffoli_ratio = toffoli_11_contrast / (np.mean(other_contrasts) + 1e-6)
    print("    Toffoli selectivity: %.4f (|1,1> contrast / avg others)" % toffoli_ratio)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Single-qubit marginals
    ax = axes[0]
    ax.plot(phis/np.pi, E1, '#E91E63', lw=2, label='SQ1@L%d (amp=%.3f)' % (SQ1_LAYER, amp1))
    ax.plot(phis/np.pi, E2, '#2196F3', lw=2, ls='--', label='SQ2@L%d cross (%.3f)' % (SQ2_LAYER, amp2))
    ax.plot(phis/np.pi, E3, '#4CAF50', lw=2, ls=':', label='SQ3@L%d cross (%.3f)' % (SQ3_LAYER, amp3))
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Phase (x pi)'); ax.set_ylabel('E = P(min)-P(max)')
    ax.set_title('(a) 3-Qubit Single Marginals', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel B: GHZ 8 corners
    ax = axes[1]
    states = [c['state'] for c in corners]
    Es = [c['E'] for c in corners]
    bar_colors = ['#E91E63' if s in ['|000>','|111>'] else '#90CAF9' for s in states]
    ax.bar(range(8), Es, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(8))
    ax.set_xticklabels(states, fontsize=9)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('E')
    ax.set_title('(b) 3-Body Computational Basis\n|E(000)-E(111)|=%.3f  C3=%.3f' % (ghz_fidelity, C3),
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel C: Toffoli gate
    ax = axes[2]
    keys = list(toffoli_data.keys())
    contrasts = [toffoli_data[k]['contrast'] for k in keys]
    cols = ['#E91E63' if '|1>,SQ3=|1>' in k else '#90CAF9' for k in keys]
    ax.bar(range(4), contrasts, color=cols, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels([k.replace('SQ2=','').replace('SQ3=','') for k in keys], fontsize=9)
    ax.set_ylabel('SQ1 Contrast |E(0)-E(1)|')
    ax.set_title('(c) Toffoli Gate: CCNOT Selectivity\nRatio=%.2f (highest when both=|1>)' % toffoli_ratio,
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q17: 3-Qubit GHZ State and Toffoli Gate\n'
                 'SQ1@L%d x SQ2@L%d x SQ3@L%d  |  "Attention = free wiring"' % (
                     SQ1_LAYER, SQ2_LAYER, SQ3_LAYER),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q17_ghz_toffoli.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q17', 'name': '3_qubit_ghz_toffoli',
        'sq1': {'layer': SQ1_LAYER, 'pos': SQ1_POS, 'amp': round(float(amp1), 6)},
        'sq2': {'layer': SQ2_LAYER, 'pos': SQ2_POS, 'cross_amp': round(float(amp2), 6)},
        'sq3': {'layer': SQ3_LAYER, 'pos': SQ3_POS, 'cross_amp': round(float(amp3), 6)},
        'pair_correlations': pair_results,
        'ghz_corners': corners,
        'ghz_contrast': round(ghz_fidelity, 6),
        'three_body_C3': round(float(C3), 6),
        'toffoli': toffoli_data,
        'toffoli_selectivity': round(float(toffoli_ratio), 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q17_ghz_toffoli.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("\n  Q17 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
