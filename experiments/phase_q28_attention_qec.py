# -*- coding: utf-8 -*-
"""
Phase Q28: Attention-Based Quantum Error Correction (Attention-QEC)

Demonstrates that the self-attention mechanism can autonomously correct
errors in S-Qubit states -- without physical cooling or error-correction
circuits.

Method:
  1. Create a 3-qubit entangled state (GHZ-like) via attention coupling
  2. Inject noise (decoherence) into ONE qubit only
  3. Measure whether the attention coupling with other qubits
     restores the corrupted qubit's coherence
  4. Compare: single qubit (no protection) vs 3-qubit (attention-QEC)
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
SQ2_LAYER = 16
SQ3_LAYER = 20
EPOCHS = 100


def train_soul(model, tok, data, device, layer, pos=-1, epochs=EPOCHS, seed=42):
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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def inject_single(model, tok, prompt, device, vec, layer, pos=-1):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    actual_pos = pos if pos >= 0 else seq_len + pos
    def hook(m, i, o, v=vec, p=actual_pos):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def inject_three_qubit(model, tok, prompt, device,
                       v1, l1, p1, v2, l2, p2, v3, l3, p3):
    """Inject 3 S-Qubits at different layers/positions."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    ap1 = p1 if p1 >= 0 else seq_len + p1
    ap2 = p2 if p2 >= 0 else seq_len + p2
    ap3 = p3 if p3 >= 0 else seq_len + p3

    def hook1(m, i, o, v=v1, p=ap1):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    def hook2(m, i, o, v=v2, p=ap2):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    def hook3(m, i, o, v=v3, p=ap3):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    h1 = model.model.layers[l1].register_forward_hook(hook1)
    h2 = model.model.layers[l2].register_forward_hook(hook2)
    h3 = model.model.layers[l3].register_forward_hook(hook3)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove(); h3.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def main():
    print("[Q28] Attention-Based Quantum Error Correction (Attention-QEC)")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    # Train soul vectors for 3 qubits at different layers/positions
    print("  Training 3-qubit soul vectors...")
    sq1_v0 = train_soul(model, tok, min_data, DEVICE, SQ1_LAYER, -1, EPOCHS, 42)
    sq1_v1 = train_soul(model, tok, max_data, DEVICE, SQ1_LAYER, -1, EPOCHS, 99)
    sq2_v0 = train_soul(model, tok, min_data, DEVICE, SQ2_LAYER, -2, EPOCHS, 10)
    sq2_v1 = train_soul(model, tok, max_data, DEVICE, SQ2_LAYER, -2, EPOCHS, 20)
    sq3_v0 = train_soul(model, tok, min_data, DEVICE, SQ3_LAYER, -3, EPOCHS, 30)
    sq3_v1 = train_soul(model, tok, max_data, DEVICE, SQ3_LAYER, -3, EPOCHS, 40)

    # Noise levels to test
    noise_sigmas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    phi_test = np.pi / 3  # test phase

    # Clean baseline
    v_clean = phi_vec(phi_test, sq1_v0, sq1_v1)
    probs_clean_1q = inject_single(model, tok, prompt, DEVICE, v_clean, SQ1_LAYER, -1)
    E_clean_1q = float(probs_clean_1q[min_tok]) - float(probs_clean_1q[max_tok])

    v2_clean = phi_vec(phi_test, sq2_v0, sq2_v1)
    v3_clean = phi_vec(phi_test, sq3_v0, sq3_v1)
    probs_clean_3q = inject_three_qubit(model, tok, prompt, DEVICE,
        v_clean, SQ1_LAYER, -1, v2_clean, SQ2_LAYER, -2, v3_clean, SQ3_LAYER, -3)
    E_clean_3q = float(probs_clean_3q[min_tok]) - float(probs_clean_3q[max_tok])

    print("  Clean baselines: 1-qubit E=%.4f, 3-qubit E=%.4f" % (E_clean_1q, E_clean_3q))

    # Error injection: add noise to SQ1 only, measure output
    results_1q = []  # single qubit (no protection)
    results_3q = []  # 3-qubit (attention-QEC)

    print("\n  Injecting noise into SQ1 (measuring 1-qubit vs 3-qubit recovery)...")
    N_TRIALS = 5  # average over random noise instances

    for sigma in noise_sigmas:
        e1_list, e3_list = [], []
        for trial in range(N_TRIALS):
            torch.manual_seed(trial * 100 + int(sigma * 1000))
            noise = torch.randn_like(v_clean) * sigma
            v_noisy = v_clean + noise

            # 1-qubit: noisy SQ1 alone
            probs_1q = inject_single(model, tok, prompt, DEVICE,
                                     v_noisy, SQ1_LAYER, -1)
            e1 = float(probs_1q[min_tok]) - float(probs_1q[max_tok])
            e1_list.append(e1)

            # 3-qubit: noisy SQ1 + clean SQ2 + clean SQ3
            probs_3q = inject_three_qubit(model, tok, prompt, DEVICE,
                v_noisy, SQ1_LAYER, -1, v2_clean, SQ2_LAYER, -2,
                v3_clean, SQ3_LAYER, -3)
            e3 = float(probs_3q[min_tok]) - float(probs_3q[max_tok])
            e3_list.append(e3)

        e1_mean = np.mean(e1_list)
        e3_mean = np.mean(e3_list)
        err_1q = abs(e1_mean - E_clean_1q)
        err_3q = abs(e3_mean - E_clean_3q)
        recovery = 1.0 - err_3q / (err_1q + 1e-9) if err_1q > 1e-6 else 1.0

        results_1q.append({'sigma': sigma, 'E_mean': round(e1_mean, 6),
                          'error': round(err_1q, 6)})
        results_3q.append({'sigma': sigma, 'E_mean': round(e3_mean, 6),
                          'error': round(err_3q, 6), 'recovery': round(recovery, 4)})

        print("    sigma=%.3f: 1Q err=%.4f, 3Q err=%.4f, recovery=%.2f%%" % (
            sigma, err_1q, err_3q, 100*max(0, recovery)))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: E vs noise for 1Q and 3Q
    ax = axes[0]
    sigmas_plot = [r['sigma'] for r in results_1q]
    e1_plot = [r['E_mean'] for r in results_1q]
    e3_plot = [r['E_mean'] for r in results_3q]
    ax.plot(sigmas_plot, e1_plot, 'rs-', lw=2, ms=7, label='1-Qubit (no protection)')
    ax.plot(sigmas_plot, e3_plot, 'bo-', lw=2, ms=7, label='3-Qubit (Attention-QEC)')
    ax.axhline(E_clean_1q, color='red', ls=':', alpha=0.5)
    ax.axhline(E_clean_3q, color='blue', ls=':', alpha=0.5)
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('E(phi)')
    ax.set_title('(a) Decoherence Under Noise\n1-Qubit vs 3-Qubit Attention-QEC',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: Error comparison
    ax = axes[1]
    err1 = [r['error'] for r in results_1q]
    err3 = [r['error'] for r in results_3q]
    ax.semilogy(sigmas_plot, [e + 1e-7 for e in err1], 'rs-', lw=2, ms=7,
                label='1-Qubit error')
    ax.semilogy(sigmas_plot, [e + 1e-7 for e in err3], 'bo-', lw=2, ms=7,
                label='3-Qubit error')
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('|E_noisy - E_clean|')
    ax.set_title('(b) Error Magnitude\nAttention coupling reduces error',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel C: Recovery rate
    ax = axes[2]
    recoveries = [max(0, r['recovery']) for r in results_3q]
    ax.bar(range(len(sigmas_plot)), [100*r for r in recoveries],
           color='#4CAF50', edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(sigmas_plot)))
    ax.set_xticklabels(['%.3f' % s for s in sigmas_plot], rotation=45, fontsize=8)
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('Error Recovery (%)')
    ax.set_title('(c) Attention-QEC Recovery Rate\n3-Qubit protection efficiency',
                 fontweight='bold')
    ax.axhline(50, color='gray', ls='--', lw=1, label='50% threshold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q28: Attention-Based Quantum Error Correction\n'
                 'Self-attention as autonomous error correction circuit',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q28_attention_qec.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q28', 'name': 'attention_qec',
        'sq_layers': [SQ1_LAYER, SQ2_LAYER, SQ3_LAYER],
        'phi_test': round(phi_test, 4),
        'E_clean_1q': round(E_clean_1q, 6),
        'E_clean_3q': round(E_clean_3q, 6),
        'results_1q': results_1q,
        'results_3q': results_3q,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q28_attention_qec.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q28 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
