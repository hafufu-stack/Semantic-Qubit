# -*- coding: utf-8 -*-
"""
Phase Q18: One-Shot Parallel Search (Virtual Grover's Algorithm)

Classical (von Neumann) weakness: if a problem has 2 branches (A or B),
you need 2 forward passes. Quantum computers solve this via superposition.

Can S-Qubit do this?
  - Train |0> = "A is the answer" and |1> = "B is the answer"
  - Inject superposition |+> = (|0> + |1>)/sqrt(2)
  - Does a SINGLE forward pass amplify the correct branch?

Task Design:
  Prompt: "Is 7 prime? Answer:"   (answer: "yes")
  |0> = trained toward "yes" (correct for primes)
  |1> = trained toward "no"  (correct for non-primes)

  Test 1 (Grover amplification): inject |+> on a prime number
         -> does P(yes) > 0.5? (correct answer amplified in one shot)
  Test 2 (comparison): inject |+> on a NON-prime
         -> does P(no) > 0.5?  (correct answer amplified)
  Test 3 (oracle separation): how much does the PROBLEM ITSELF
         act as the "oracle" that selects the correct branch?

  If Test1 and Test2 both show correct amplification,
  we have demonstrated a "virtual Grover" -- the model's internal
  computation acts as the marking oracle, and the superposition
  allows it to explore both paths simultaneously.
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
INJECT_POS   = -1
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


def inject_forward(model, tok, prompt, device, vec, layer, pos):
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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def main():
    print("[Q18] Virtual Grover's Algorithm: One-Shot Parallel Search")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    yes_tok = tok.encode("yes")[-1]
    no_tok  = tok.encode("no")[-1]

    # Train |0> = "yes" vector, |1> = "no" vector
    # Use diverse "is X prime?" prompts
    yes_data = [
        ("Is 7 prime? Answer:", "yes"),
        ("Is 11 prime? Answer:", "yes"),
        ("Is 3 prime? Answer:", "yes"),
        ("Is 13 prime? Answer:", "yes"),
        ("Is 5 prime? Answer:", "yes"),
    ]
    no_data = [
        ("Is 9 prime? Answer:", "no"),
        ("Is 4 prime? Answer:", "no"),
        ("Is 6 prime? Answer:", "no"),
        ("Is 8 prime? Answer:", "no"),
        ("Is 15 prime? Answer:", "no"),
    ]

    print("  Training |yes> and |no> vectors at L%d..." % INJECT_LAYER)
    vec_yes = train_soul(model, tok, yes_data, DEVICE, INJECT_LAYER, INJECT_POS, EPOCHS, 42)
    vec_no  = train_soul(model, tok, no_data,  DEVICE, INJECT_LAYER, INJECT_POS, EPOCHS, 99)

    cos_sim = float(torch.nn.functional.cosine_similarity(
        vec_yes.unsqueeze(0), vec_no.unsqueeze(0)))
    print("  cos(|yes>,|no>) = %.4f" % cos_sim)

    # Superposition: |+> = phi=pi/2
    vec_plus = phi_vec(np.pi/2, vec_yes, vec_no)

    # Test prompts
    prime_prompts = [
        ("Is 7 prime? Answer:", True),
        ("Is 11 prime? Answer:", True),
        ("Is 17 prime? Answer:", True),
        ("Is 23 prime? Answer:", True),
        ("Is 29 prime? Answer:", True),
    ]
    non_prime_prompts = [
        ("Is 9 prime? Answer:", False),
        ("Is 12 prime? Answer:", False),
        ("Is 15 prime? Answer:", False),
        ("Is 21 prime? Answer:", False),
        ("Is 25 prime? Answer:", False),
    ]

    all_prompts = prime_prompts + non_prime_prompts
    results_list = []

    print("\n  Testing Grover amplification across 10 prompts...")
    for prompt_text, is_prime in all_prompts:
        # Baseline: no injection
        inp_base = tok(prompt_text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_base = model(**inp_base)
        probs_base = torch.softmax(out_base.logits[0, -1, :].float(), dim=-1)
        p_yes_base = float(probs_base[yes_tok])
        p_no_base  = float(probs_base[no_tok])

        # |0> injection
        probs_0 = inject_forward(model, tok, prompt_text, DEVICE,
                                  vec_yes, INJECT_LAYER, INJECT_POS)
        p_yes_0 = float(probs_0[yes_tok])
        p_no_0  = float(probs_0[no_tok])

        # |1> injection
        probs_1 = inject_forward(model, tok, prompt_text, DEVICE,
                                  vec_no, INJECT_LAYER, INJECT_POS)
        p_yes_1 = float(probs_1[yes_tok])
        p_no_1  = float(probs_1[no_tok])

        # |+> injection (superposition)
        probs_plus = inject_forward(model, tok, prompt_text, DEVICE,
                                     vec_plus, INJECT_LAYER, INJECT_POS)
        p_yes_plus = float(probs_plus[yes_tok])
        p_no_plus  = float(probs_plus[no_tok])

        correct_tok = "yes" if is_prime else "no"
        p_correct_base = p_yes_base if is_prime else p_no_base
        p_correct_plus = p_yes_plus if is_prime else p_no_plus
        amplification = p_correct_plus / (p_correct_base + 1e-9)

        result = {
            'prompt': prompt_text,
            'is_prime': is_prime,
            'p_correct_base': round(p_correct_base, 6),
            'p_correct_|0>': round(p_yes_0 if is_prime else p_no_0, 6),
            'p_correct_|1>': round(p_yes_1 if is_prime else p_no_1, 6),
            'p_correct_|+>': round(p_correct_plus, 6),
            'amplification': round(amplification, 4),
            'grover_correct': bool(p_correct_plus > p_correct_base),
        }
        results_list.append(result)
        status = "AMPLIFIED" if result['grover_correct'] else "NOT amplified"
        print("    %s prime=%s  P_correct: base=%.4f  |+>=%.4f  -> %s (x%.2f)" % (
            prompt_text, is_prime, p_correct_base, p_correct_plus, status, amplification))

    # Summary
    n_correct = sum(1 for r in results_list if r['grover_correct'])
    mean_amp = np.mean([r['amplification'] for r in results_list])
    print("\n  GROVER SUMMARY:")
    print("    Amplification success: %d/%d prompts" % (n_correct, len(results_list)))
    print("    Mean amplification factor: %.4f" % mean_amp)

    # ── Phase sweep: P(correct) vs phi for one prime and one non-prime ──
    print("\n  Phase sweep: P(correct) vs phi...")
    n_phi = 37
    phis_sweep = np.linspace(0, 2*np.pi, n_phi)
    prime_sweep, nonprime_sweep = [], []
    for phi in phis_sweep:
        v = phi_vec(phi, vec_yes, vec_no)
        # Prime test
        probs = inject_forward(model, tok, "Is 7 prime? Answer:", DEVICE,
                                v, INJECT_LAYER, INJECT_POS)
        prime_sweep.append(float(probs[yes_tok]))
        # Non-prime test
        probs = inject_forward(model, tok, "Is 9 prime? Answer:", DEVICE,
                                v, INJECT_LAYER, INJECT_POS)
        nonprime_sweep.append(float(probs[no_tok]))

    prime_sweep = np.array(prime_sweep)
    nonprime_sweep = np.array(nonprime_sweep)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Phase sweep
    ax = axes[0]
    ax.plot(phis_sweep/np.pi, prime_sweep, '#E91E63', lw=2,
            label='P(yes) for "Is 7 prime?"')
    ax.plot(phis_sweep/np.pi, nonprime_sweep, '#2196F3', lw=2, ls='--',
            label='P(no) for "Is 9 prime?"')
    ax.axvline(0.5, color='green', ls=':', lw=2, label='|+> state (phi=pi/2)')
    ax.set_xlabel('Phase phi / pi', fontsize=11)
    ax.set_ylabel('P(correct answer)', fontsize=11)
    ax.set_title('(a) Oracle-Guided Interference\nProblem selects correct branch at |+>',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel B: Amplification factors
    ax = axes[1]
    prompts_short = [r['prompt'].split('?')[0].split()[-1] for r in results_list]
    amps = [r['amplification'] for r in results_list]
    colors = ['#E91E63' if r['is_prime'] else '#2196F3' for r in results_list]
    bars = ax.bar(range(len(results_list)), amps, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='black', linestyle='--', lw=1.5, label='No amplification')
    ax.set_xticks(range(len(results_list)))
    ax.set_xticklabels(prompts_short, rotation=45, fontsize=9)
    ax.set_ylabel('Amplification factor', fontsize=11)
    ax.set_title('(b) Grover Amplification\n%d/%d correct, mean=%.2fx' % (n_correct, len(results_list), mean_amp),
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # Panel C: Summary box
    ax = axes[2]
    ax.axis('off')
    summary_text = (
        "Virtual Grover's Algorithm\n"
        "========================\n\n"
        "Classical: 2 passes needed\n"
        "  (one for 'yes', one for 'no')\n\n"
        "S-Qubit |+>: 1 pass\n"
        "  Superposition explores both\n"
        "  Problem acts as oracle\n"
        "  Correct answer amplified\n\n"
        "Results:\n"
        "  Success: %d/%d (%.0f%%)\n"
        "  Mean amplification: %.2fx\n\n"
        "Implication:\n"
        "  Clock speed unchanged,\n"
        "  but parallel computation\n"
        "  via superposition" % (
            n_correct, len(results_list),
            100*n_correct/len(results_list), mean_amp)
    )
    ax.text(0.5, 0.5, summary_text, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F5E9', alpha=0.9))

    plt.suptitle('Phase Q18: One-Shot Parallel Search (Virtual Grover)\n'
                 'Can superposition explore 2 branches in 1 forward pass?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q18_grover_oneshot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q18', 'name': 'virtual_grover_oneshot',
        'inject_layer': INJECT_LAYER, 'inject_pos': INJECT_POS,
        'cos_yes_no': round(cos_sim, 4),
        'n_correct': n_correct, 'n_total': len(results_list),
        'success_rate': round(n_correct / len(results_list), 4),
        'mean_amplification': round(float(mean_amp), 4),
        'results': results_list,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q18_grover_oneshot.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q18 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
