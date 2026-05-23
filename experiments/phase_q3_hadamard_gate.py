# -*- coding: utf-8 -*-
"""
Phase Q3: Neural Hadamard Gate
In quantum computing, the Hadamard gate H transforms:
  |0> -> (|0> + |1>) / sqrt(2)  [single definite state -> superposition]
  |1> -> (|0> - |1>) / sqrt(2)  [with phase flip]

Neural analog: Given a DEFINITE Soul Vector (MIN or MAX, 100% accurate),
can we apply a "Hadamard transform" to put it in equal superposition,
then RECOVER the original via a second Hadamard (H^2 = I)?

Experiment:
1. Train |MIN> and |MAX> soul vectors
2. Apply H_neural: psi_H = (|MIN> + |MAX>) / sqrt(2)
3. Measure: is output now 50/50?
4. Apply H_neural again: psi_HH = H(psi_H) 
   = H((|MIN>+|MAX>)/sqrt(2)) = |MIN>
5. Check: does double-Hadamard recover original?

Also test: does the Hadamard gate respect phase?
  H(e^{i*phi}*|MIN>) = e^{i*phi}*(H|MIN>)
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


def apply_hook_vec(model, tok, vec, prompt, device, layer):
    """Inject vec at layer, return (logits, top_token_str, top_prob)."""
    def hook(m, i, o, v=vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    inp = tok(prompt, return_tensors='pt').to(device)
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_token = tok.decode(probs.argmax().item()).strip()
    top_prob = float(probs.max())
    return logits, top_token, top_prob


def hadamard_transform(min_vec, max_vec):
    """
    Neural Hadamard: H(|MIN>) = (|MIN> + |MAX>) / sqrt(2)
                     H(|MAX>) = (|MIN> - |MAX>) / sqrt(2)
    Returns (H_min, H_max)
    """
    h_min = (min_vec + max_vec) / np.sqrt(2)
    h_max = (min_vec - max_vec) / np.sqrt(2)
    return h_min, h_max


def main():
    print("[Q3] Neural Hadamard Gate")
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
    test_prompts = [
        ("min(7,2)=", "2", "7"),
        ("min(6,3)=", "3", "6"),
        ("min(2,9)=", "2", "9"),
        ("min(1,5)=", "1", "5"),
        ("min(8,4)=", "4", "8"),
    ]

    print("  Training basis vectors...")
    min_vec = train_soul(model, tok, min_data, DEVICE, LAYER, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, LAYER, seed=99)

    # Apply Hadamard
    h_min, h_max = hadamard_transform(min_vec, max_vec)
    # Apply Hadamard twice (should recover original: H^2 = I)
    hh_min, hh_max = hadamard_transform(h_min, h_max)

    # Also test "ZH" gate: Z = phase flip on |MAX>
    # Z|MIN> = |MIN>, Z|MAX> = -|MAX>
    zh_min = (min_vec + (-max_vec)) / np.sqrt(2)

    results = {}
    tok_min_id = tok.encode('2')[-1]
    tok_max_id = tok.encode('7')[-1]

    configs = {
        '|MIN>': min_vec,
        '|MAX>': max_vec,
        'H|MIN> = (|MIN>+|MAX>)/sqrt2': h_min,
        'H|MAX> = (|MIN>-|MAX>)/sqrt2': h_max,
        'H^2|MIN> (should=|MIN>)': hh_min,
        'H^2|MAX> (should=|MAX>)': hh_max,
        'ZH|MIN> = (|MIN>-|MAX>)/sqrt2': zh_min,
    }

    print("  Testing gate configurations...")
    for name, vec in configs.items():
        prompt_results = []
        for prompt, min_ans, max_ans in test_prompts:
            logits, top_tok, top_prob = apply_hook_vec(model, tok, vec, prompt, DEVICE, LAYER)
            probs = torch.softmax(logits, dim=-1)
            prompt_results.append({
                'prompt': prompt,
                'top': top_tok,
                'prob_min_tok': round(float(probs[tok_min_id]), 4),
                'prob_max_tok': round(float(probs[tok_max_id]), 4),
                'top_prob': round(top_prob, 4),
            })
        avg_p_min = np.mean([r['prob_min_tok'] for r in prompt_results])
        avg_p_max = np.mean([r['prob_max_tok'] for r in prompt_results])
        results[name] = {
            'avg_prob_min_token': round(float(avg_p_min), 4),
            'avg_prob_max_token': round(float(avg_p_max), 4),
            'details': prompt_results,
        }
        print("    %-40s P(min)=%.3f P(max)=%.3f" % (name[:40], avg_p_min, avg_p_max))

    # Measure H^2 recovery fidelity
    cos_hh_min = float(torch.nn.functional.cosine_similarity(
        hh_min.unsqueeze(0), min_vec.unsqueeze(0)))
    cos_hh_max = float(torch.nn.functional.cosine_similarity(
        hh_max.unsqueeze(0), max_vec.unsqueeze(0)))
    print("  H^2 recovery fidelity: |MIN>: %.4f, |MAX>: %.4f" % (cos_hh_min, cos_hh_max))
    print("  (1.0 = perfect recovery, H^2=I confirmed)")

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    config_names = list(results.keys())
    avg_min_probs = [results[k]['avg_prob_min_token'] for k in config_names]
    avg_max_probs = [results[k]['avg_prob_max_token'] for k in config_names]
    x = np.arange(len(config_names))

    # Panel 1: P(min token) for each gate
    ax = axes[0]
    ax.barh(x, avg_min_probs, color='#E91E63', edgecolor='black', label='P(min="2")')
    ax.set_yticks(x)
    ax.set_yticklabels([n[:35] for n in config_names], fontsize=7)
    ax.set_xlabel('Avg P(min token)')
    ax.set_title('P(min answer) per Gate Config', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal mix')
    ax.legend(fontsize=8)

    # Panel 2: P(max token) for each gate
    ax = axes[1]
    ax.barh(x, avg_max_probs, color='#2196F3', edgecolor='black', label='P(max="7")')
    ax.set_yticks(x)
    ax.set_yticklabels([n[:35] for n in config_names], fontsize=7)
    ax.set_xlabel('Avg P(max token)')
    ax.set_title('P(max answer) per Gate Config', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # Panel 3: H^2 = I verification
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Neural Hadamard Gate H^2=I Test\n\n"
        "|MIN> -> H -> H^2\n"
        "  Vector fidelity: %.4f\n\n"
        "|MAX> -> H -> H^2\n"
        "  Vector fidelity: %.4f\n\n"
        "H|MIN> = (|MIN>+|MAX>)/sqrt2\n"
        "  P(min): %.3f  P(max): %.3f\n\n"
        "H|MAX> = (|MIN>-|MAX>)/sqrt2\n"
        "  P(min): %.3f  P(max): %.3f\n\n"
        "If H^2=I and H|s>=50/50:\n"
        "  -> Hadamard gate works!"
    ) % (
        cos_hh_min, cos_hh_max,
        results['H|MIN> = (|MIN>+|MAX>)/sqrt2']['avg_prob_min_token'],
        results['H|MIN> = (|MIN>+|MAX>)/sqrt2']['avg_prob_max_token'],
        results['H|MAX> = (|MIN>-|MAX>)/sqrt2']['avg_prob_min_token'],
        results['H|MAX> = (|MIN>-|MAX>)/sqrt2']['avg_prob_max_token'],
    )
    ax.text(0.05, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))

    plt.suptitle(
        'Phase Q3: Neural Hadamard Gate\nH|0>=|+>, H|1>=|->, H^2=I',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q3_hadamard_gate.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q3', 'name': 'hadamard_gate',
        'h2_fidelity_min': cos_hh_min,
        'h2_fidelity_max': cos_hh_max,
        'h2_identity_confirmed': bool(cos_hh_min > 0.99 and cos_hh_max > 0.99),
        'gate_results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q3_hadamard_gate.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Q3 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
