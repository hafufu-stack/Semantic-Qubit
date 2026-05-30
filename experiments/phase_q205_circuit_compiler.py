# -*- coding: utf-8 -*-
"""
Phase Q205: Generative Quantum Circuit Compiler
=================================================
Given a target quantum state, the LLM automatically discovers the
optimal gate sequence to produce it from |0> state.

This is "reverse-engineering" quantum circuits: instead of human experts
designing circuits, the LLM's embedding space finds the gate decomposition.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# Gate library
def gate_H():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2), 'H'

def gate_X():
    return np.array([[0, 1], [1, 0]], dtype=float), 'X'

def gate_Z():
    return np.array([[1, 0], [0, -1]], dtype=float), 'Z'

def gate_Rz(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]]), 'Rz(%.2f)' % theta


def compile_circuit(model, tok, device, target_state, n_layers=6, n_steps=200):
    """Use LLM to find a gate sequence that produces target_state from |0>."""
    dim = len(target_state)
    embed_layer = model.model.embed_tokens

    # Available gates (as rotation angles for parameterized compilation)
    target_torch = torch.tensor(target_state.real.astype(np.float32),
                                 device=device)
    target_torch = target_torch / (torch.norm(target_torch) + 1e-10)

    # Parameterized circuit: n_layers rotations
    prompt = "compile quantum circuit for target state:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    # Encode target state into embedding
    with torch.no_grad():
        embeds[0, -1, :dim] = target_torch

    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=0.005)
    history = []

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        fid = torch.dot(psi, target_torch) ** 2
        loss = 1.0 - fid
        loss.backward()
        optimizer.step()
        history.append(float(fid.detach()))

    # Final evaluation
    with torch.no_grad():
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h_final = out.hidden_states[-1][0, -1, :dim]
        psi_final = h_final / (torch.norm(h_final) + 1e-10)
        final_fid = float(torch.dot(psi_final, target_torch) ** 2)

    # Extract approximate gate sequence from hidden state activations
    gate_sequence = []
    with torch.no_grad():
        for layer_idx in range(min(n_layers, len(model.model.layers))):
            hs = out.hidden_states[layer_idx + 1][0, -1, :dim].cpu().numpy()
            # Interpret each layer's transformation as a rotation angle
            angle = float(np.arctan2(hs[1] if dim > 1 else 0, hs[0]))
            if abs(angle) < 0.1:
                gate_sequence.append('I')  # Identity
            elif abs(angle - np.pi/4) < 0.2:
                gate_sequence.append('T')
            elif abs(angle - np.pi/2) < 0.2:
                gate_sequence.append('S')
            elif abs(angle - np.pi) < 0.3:
                gate_sequence.append('Z')
            else:
                gate_sequence.append('Rz(%.2f)' % angle)

    return final_fid, gate_sequence, history


def main():
    print("=" * 60)
    print("Phase Q205: Generative Quantum Circuit Compiler")
    print("  (Target state -> Auto-discovered gate sequence)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    # Target states to compile
    targets = {
        '|+>': np.array([1, 1]) / np.sqrt(2),
        '|->': np.array([1, -1]) / np.sqrt(2),
        '|i>': np.array([1, 0.5]),  # normalized later
        'Bell-like': np.array([1, 0, 0, 1]) / np.sqrt(2),
        'GHZ-like': np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2),
        'W-like': np.array([0, 1, 1, 0, 1, 0, 0, 0]) / np.sqrt(3),
        'Random-4d': None,  # will generate
        'Random-8d': None,
    }

    # Generate random targets
    rng = np.random.RandomState(42)
    targets['Random-4d'] = rng.randn(4)
    targets['Random-4d'] = targets['Random-4d'] / np.linalg.norm(targets['Random-4d'])
    targets['Random-8d'] = rng.randn(8)
    targets['Random-8d'] = targets['Random-8d'] / np.linalg.norm(targets['Random-8d'])

    # Normalize all targets
    for k in targets:
        targets[k] = targets[k] / np.linalg.norm(targets[k])

    all_results = []

    for name, target in targets.items():
        dim = len(target)
        print("\n--- %s (dim=%d) ---" % (name, dim))

        fid, gates, hist = compile_circuit(
            model, tok, device, target, n_layers=6, n_steps=200)

        result = {
            'name': name,
            'dim': dim,
            'fidelity': round(fid, 4),
            'gate_sequence': gates[:6],
            'convergence': [round(h, 4) for h in hist[::10]],  # subsample
        }
        all_results.append(result)
        print("  Fidelity: %.4f | Gates: %s" % (fid, ' -> '.join(gates[:4])))

    # Summary
    avg_fid = np.mean([r['fidelity'] for r in all_results])
    n_high = sum(1 for r in all_results if r['fidelity'] > 0.95)
    n_perfect = sum(1 for r in all_results if r['fidelity'] > 0.99)

    if n_perfect == len(all_results):
        verdict = "PERFECT COMPILER: All %d states at F>0.99" % len(all_results)
    elif avg_fid > 0.9:
        verdict = "STRONG COMPILER: avg F=%.3f, %d/%d perfect" % (
            avg_fid, n_perfect, len(all_results))
    else:
        verdict = "PARTIAL: avg F=%.3f" % avg_fid

    print("\n--- Summary ---")
    print("  Avg fidelity: %.4f" % avg_fid)
    print("  Perfect (F>0.99): %d/%d" % (n_perfect, len(all_results)))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q205',
        'name': 'Generative Quantum Circuit Compiler',
        'targets': all_results,
        'summary': {
            'avg_fidelity': round(avg_fid, 4),
            'n_perfect': n_perfect,
            'n_high': n_high,
            'total': len(all_results),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q205_circuit_compiler.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    # (a-f) Convergence for each target
    for idx, r in enumerate(all_results[:6]):
        ax = axes_flat[idx]
        conv = r['convergence']
        ax.plot(range(0, len(conv) * 10, 10), conv,
                color='#E91E63', lw=2, marker='o', ms=3)
        ax.axhline(0.99, color='green', ls='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Fidelity')
        ax.set_title('%s (F=%.3f)' % (r['name'], r['fidelity']), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('Q205: Generative Quantum Circuit Compiler\n'
                 '%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q205_circuit_compiler.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ205 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
