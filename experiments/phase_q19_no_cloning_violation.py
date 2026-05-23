# -*- coding: utf-8 -*-
"""
Phase Q19: No-Cloning Theorem Violation (Quantum Save & Load)

Physical quantum mechanics forbids copying an unknown quantum state
(no-cloning theorem). S-Qubits are digital tensors -> we can COPY them.

Experiment:
  1. Prepare superposition |psi(phi)> at L8
  2. Extract hidden state at L20 (pre-collapse) -> SAVE to memory (clone)
  3. Continue forward pass to completion -> "observe" (collapse)
  4. RELOAD the saved L20 state -> run a SECOND completion
  5. Compare: do the two completions yield different but valid results?

  This demonstrates "branching universes" -- from one quantum state,
  we fork into multiple measurement outcomes. IMPOSSIBLE in physics.

  Practical value: quantum debugging. You can inspect the "wavefunction"
  at any layer, save it, modify it, and replay -- like a video game save.
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
SAVE_LAYERS  = [10, 14, 18, 20, 22, 24, 26]  # where to save state
INJECT_POS   = -1
EPOCHS = 120


def train_soul(model, tok, data, device, layer, pos, epochs, seed):
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


def capture_and_complete(model, tok, prompt, device, inject_vec, inject_layer,
                          inject_pos, save_layer):
    """
    1. Forward with injection at inject_layer, capturing hidden state at save_layer
    2. Return: final probs, saved hidden state tensor
    """
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    actual_pos = inject_pos if inject_pos >= 0 else seq_len + inject_pos

    saved_state = {}

    def inject_hook(m, i, o, v=inject_vec, p=actual_pos):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    def save_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        saved_state['hidden'] = h.detach().clone()

    h1 = model.model.layers[inject_layer].register_forward_hook(inject_hook)
    h2 = model.model.layers[save_layer].register_forward_hook(save_hook)

    with torch.no_grad():
        out = model(**inp)

    h1.remove(); h2.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return probs, saved_state['hidden']


def replay_from_saved(model, tok, prompt, device, saved_hidden, save_layer):
    """
    Replay from saved hidden state at save_layer.
    Inject the saved hidden state at save_layer, run layers save_layer+1..end.
    """
    inp = tok(prompt, return_tensors='pt').to(device)

    def replay_hook(m, i, o, saved=saved_hidden):
        # Replace entire hidden state with saved copy
        if isinstance(o, tuple):
            return (saved.to(o[0].dtype),) + o[1:]
        return saved.to(o.dtype)

    handle = model.model.layers[save_layer].register_forward_hook(replay_hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def main():
    print("[Q19] No-Cloning Theorem Violation: Quantum Save & Load")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    prompt = "min(7,2)="
    sq_tok = tok.encode("2")[-1]
    sq_tok_1 = tok.encode("7")[-1]

    sq_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                 ("min(4,6)=","4"),("min(9,3)=","3")]
    sq_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                 ("min(4,6)=","6"),("min(9,3)=","9")]

    print("  Training basis vectors at L%d..." % INJECT_LAYER)
    vec_0 = train_soul(model, tok, sq_0_data, DEVICE, INJECT_LAYER, INJECT_POS, EPOCHS, 42)
    vec_1 = train_soul(model, tok, sq_1_data, DEVICE, INJECT_LAYER, INJECT_POS, EPOCHS, 99)

    # Test at multiple superposition angles and save layers
    test_phis = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    phi_labels = ['0', 'pi/4', 'pi/2', '3pi/4', 'pi']

    results = {}

    for save_layer in SAVE_LAYERS:
        layer_results = []
        print("\n  Save layer L%d:" % save_layer)

        for phi, phi_label in zip(test_phis, phi_labels):
            vec = phi_vec(phi, vec_0, vec_1)

            # Original forward pass (with injection + capture)
            probs_orig, saved_hidden = capture_and_complete(
                model, tok, prompt, DEVICE, vec, INJECT_LAYER, INJECT_POS, save_layer)
            E_orig = float(probs_orig[sq_tok]) - float(probs_orig[sq_tok_1])

            # Clone and replay (no injection, just use saved state)
            probs_replay = replay_from_saved(
                model, tok, prompt, DEVICE, saved_hidden, save_layer)
            E_replay = float(probs_replay[sq_tok]) - float(probs_replay[sq_tok_1])

            # Fidelity: how close are replay probs to original?
            fidelity = float(torch.sum(torch.sqrt(probs_orig * probs_replay + 1e-12)))

            # Clone divergence test: add small perturbation to clone
            perturbed_hidden = saved_hidden.clone()
            noise = torch.randn_like(perturbed_hidden) * 0.01
            perturbed_hidden = perturbed_hidden + noise

            probs_perturbed = replay_from_saved(
                model, tok, prompt, DEVICE, perturbed_hidden, save_layer)
            E_perturbed = float(probs_perturbed[sq_tok]) - float(probs_perturbed[sq_tok_1])

            lr = {
                'phi': phi_label, 'E_orig': round(E_orig, 6),
                'E_replay': round(E_replay, 6), 'E_perturbed': round(E_perturbed, 6),
                'replay_fidelity': round(fidelity, 6),
                'clone_divergence': round(abs(E_replay - E_perturbed), 6),
            }
            layer_results.append(lr)
            print("    phi=%s: E_orig=%.4f  E_replay=%.4f  fid=%.4f  E_pert=%.4f" % (
                phi_label, E_orig, E_replay, fidelity, E_perturbed))

        results[str(save_layer)] = layer_results

    # Summary statistics
    perfect_replays = 0
    total_tests = 0
    for sl, lr_list in results.items():
        for lr in lr_list:
            total_tests += 1
            if abs(lr['E_orig'] - lr['E_replay']) < 0.001:
                perfect_replays += 1

    print("\n  NO-CLONING VIOLATION SUMMARY:")
    print("    Perfect replays: %d/%d (%.1f%%)" % (
        perfect_replays, total_tests, 100*perfect_replays/total_tests))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: E_orig vs E_replay at each save layer
    ax = axes[0]
    for sl in SAVE_LAYERS:
        lr_list = results[str(sl)]
        E_origs = [lr['E_orig'] for lr in lr_list]
        E_replays = [lr['E_replay'] for lr in lr_list]
        ax.scatter(E_origs, E_replays, s=40, alpha=0.7, label='L%d' % sl)
    ax.plot([-1, 1], [-1, 1], 'k--', lw=1.5, alpha=0.5, label='Perfect clone')
    ax.set_xlabel('E (original forward pass)', fontsize=11)
    ax.set_ylabel('E (replayed from saved state)', fontsize=11)
    ax.set_title('(a) Clone Fidelity\nSaved state -> identical output?', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Panel B: Replay fidelity vs save layer
    ax = axes[1]
    fidelities = []
    for sl in SAVE_LAYERS:
        lr_list = results[str(sl)]
        mean_fid = np.mean([lr['replay_fidelity'] for lr in lr_list])
        fidelities.append(mean_fid)
    ax.bar(SAVE_LAYERS, fidelities, color='#9C27B0', edgecolor='black', alpha=0.85, width=1.5)
    ax.axhline(1.0, color='red', linestyle='--', lw=1.5, label='Perfect fidelity')
    ax.axvspan(22, 26, alpha=0.15, color='orange', label='Collapse zone')
    ax.set_xlabel('Save Layer', fontsize=11)
    ax.set_ylabel('Mean Replay Fidelity', fontsize=11)
    ax.set_title('(b) Where to Save?\nFidelity vs layer position', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # Panel C: Clone divergence (perturbation sensitivity)
    ax = axes[2]
    divergences = []
    for sl in SAVE_LAYERS:
        lr_list = results[str(sl)]
        mean_div = np.mean([lr['clone_divergence'] for lr in lr_list])
        divergences.append(mean_div)
    ax.bar(SAVE_LAYERS, divergences, color='#FF9800', edgecolor='black', alpha=0.85, width=1.5)
    ax.axvspan(22, 26, alpha=0.15, color='orange', label='Collapse zone')
    ax.set_xlabel('Save Layer', fontsize=11)
    ax.set_ylabel('Mean |E_clone - E_perturbed|', fontsize=11)
    ax.set_title('(c) Perturbation Sensitivity\n"Quantum debugging": save + modify + replay',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q19: No-Cloning Theorem Violation\n'
                 '"Copy, save, and replay quantum states -- impossible in physics, trivial in S-Qubit"',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q19_no_cloning.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q19', 'name': 'no_cloning_violation',
        'inject_layer': INJECT_LAYER,
        'save_layers': SAVE_LAYERS,
        'perfect_replays': perfect_replays,
        'total_tests': total_tests,
        'perfect_ratio': round(perfect_replays/total_tests, 4),
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q19_no_cloning.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q19 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
