# -*- coding: utf-8 -*-
"""
Phase Q218: Quantum Fisher Information
=========================================
QFI measures the sensitivity of a quantum state to parameter changes.
High QFI = high precision in quantum metrology (Heisenberg limit).

If the LLM's hidden states have high QFI, they can be used as
quantum sensors with super-classical precision.
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


def compute_qfi(model, tok, device, prompt, layer, dim=8, delta=0.01):
    """Compute QFI by finite-difference of fidelity."""
    embed_layer = model.model.embed_tokens
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    # Get state at parameter theta
    def get_state(theta_shift):
        e = embeds.clone()
        with torch.no_grad():
            e[0, -1, 0] += theta_shift
        out = model(inputs_embeds=e.float(), output_hidden_states=True)
        h = out.hidden_states[layer][0, -1, :dim].float().cpu().numpy()
        return h / (np.linalg.norm(h) + 1e-10)

    with torch.no_grad():
        psi_0 = get_state(0)
        psi_plus = get_state(delta)
        psi_minus = get_state(-delta)

    # Fidelity
    F_plus = np.abs(np.dot(psi_0, psi_plus)) ** 2
    F_minus = np.abs(np.dot(psi_0, psi_minus)) ** 2

    # QFI = 8(1 - sqrt(F)) / delta^2 for pure states
    qfi_plus = 8 * (1 - np.sqrt(max(F_plus, 0))) / (delta ** 2)
    qfi_minus = 8 * (1 - np.sqrt(max(F_minus, 0))) / (delta ** 2)
    qfi = (qfi_plus + qfi_minus) / 2

    # Classical Fisher Info (diagonal of density matrix)
    probs = psi_0 ** 2
    probs_plus = psi_plus ** 2
    cfi = np.sum((probs_plus - probs) ** 2 / (probs + 1e-10)) / (delta ** 2)

    # Heisenberg ratio: QFI/CFI > 1 means quantum advantage
    heisenberg_ratio = qfi / max(cfi, 1e-10)

    return {
        'qfi': float(qfi),
        'cfi': float(cfi),
        'heisenberg_ratio': float(heisenberg_ratio),
        'fidelity_sensitivity': float(1 - F_plus),
    }


def main():
    print("=" * 60)
    print("Phase Q218: Quantum Fisher Information")
    print("  (Can LLM states be quantum sensors?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    dim = 8

    prompts = [
        "quantum sensing magnetic field",
        "gravitational wave detector",
        "precision measurement of time",
        "atomic clock frequency standard",
    ]

    # Measure QFI at each layer
    layer_qfi = []
    for li in range(0, n_layers + 1, 2):
        qfi_vals, cfi_vals, hr_vals = [], [], []
        for prompt in prompts:
            result = compute_qfi(model, tok, device, prompt, li, dim=dim)
            qfi_vals.append(result['qfi'])
            cfi_vals.append(result['cfi'])
            hr_vals.append(result['heisenberg_ratio'])

        avg_data = {
            'layer': li,
            'avg_qfi': round(np.mean(qfi_vals), 4),
            'avg_cfi': round(np.mean(cfi_vals), 4),
            'avg_heisenberg_ratio': round(np.mean(hr_vals), 4),
        }
        layer_qfi.append(avg_data)

        if li % 4 == 0:
            print("  L%d: QFI=%.4f, CFI=%.4f, H-ratio=%.2f" %
                  (li, avg_data['avg_qfi'], avg_data['avg_cfi'],
                   avg_data['avg_heisenberg_ratio']))

    # Find peak QFI
    qfis = [d['avg_qfi'] for d in layer_qfi]
    peak_layer = layer_qfi[int(np.argmax(qfis))]['layer']
    peak_qfi = max(qfis)

    # Heisenberg limit check
    n_super_heisenberg = sum(1 for d in layer_qfi if d['avg_heisenberg_ratio'] > 1)
    avg_hr = np.mean([d['avg_heisenberg_ratio'] for d in layer_qfi])

    if avg_hr > 2:
        verdict = "QUANTUM SENSOR: avg Heisenberg ratio=%.1f (peak QFI at L%d)" % (avg_hr, peak_layer)
    elif avg_hr > 1:
        verdict = "ENHANCED SENSOR: avg H-ratio=%.2f, %d/%d layers super-classical" % (
            avg_hr, n_super_heisenberg, len(layer_qfi))
    else:
        verdict = "CLASSICAL SENSOR: H-ratio=%.2f" % avg_hr

    print("\n--- Summary ---")
    print("  Peak QFI: Layer %d (%.4f)" % (peak_layer, peak_qfi))
    print("  Avg Heisenberg ratio: %.4f" % avg_hr)
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q218',
        'name': 'Quantum Fisher Information',
        'layers': layer_qfi,
        'summary': {
            'peak_layer': peak_layer,
            'peak_qfi': round(peak_qfi, 4),
            'avg_heisenberg_ratio': round(avg_hr, 4),
            'n_super_heisenberg': n_super_heisenberg,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q218_qfi.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers_x = [d['layer'] for d in layer_qfi]

    ax = axes[0]
    ax.plot(layers_x, qfis, 'o-', color='#E91E63', lw=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('QFI')
    ax.set_title('(a) Quantum Fisher Information')
    ax.grid(alpha=0.3)

    ax = axes[1]
    cfis = [d['avg_cfi'] for d in layer_qfi]
    ax.plot(layers_x, qfis, 'o-', color='#E91E63', lw=2, label='QFI')
    ax.plot(layers_x, cfis, 's--', color='#2196F3', lw=2, label='CFI')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Fisher Information')
    ax.set_title('(b) QFI vs CFI')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    hrs = [d['avg_heisenberg_ratio'] for d in layer_qfi]
    colors = ['#4CAF50' if h > 1 else '#F44336' for h in hrs]
    ax.bar(range(len(layers_x)), hrs, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(1, color='black', ls='--', lw=1, label='Classical limit')
    ax.set_xticks(range(0, len(layers_x), 2))
    ax.set_xticklabels([str(layers_x[i]) for i in range(0, len(layers_x), 2)], fontsize=7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('QFI/CFI (Heisenberg Ratio)')
    ax.set_title('(c) Heisenberg Ratio')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q218: Quantum Fisher Information\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q218_qfi.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ218 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
