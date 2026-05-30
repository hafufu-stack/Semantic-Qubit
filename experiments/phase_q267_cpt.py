# -*- coding: utf-8 -*-
"""
Phase Q267: CPT Symmetry & Time Reversal
===========================================
Is the arrow of time irreversible in LLM computation?
Test CPT symmetry by reversing input and checking if
the quantum properties are preserved under conjugation.
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

def partial_transpose(rho, da, db):
    rho_pt = np.zeros_like(rho)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                for l in range(db):
                    rho_pt[i*db+k, j*db+l] = rho[i*db+l, j*db+k]
    return rho_pt

def quantum_props(h_np, dim=4):
    h = h_np[:dim] / (np.linalg.norm(h_np[:dim]) + 1e-10)
    rho = np.outer(h, h.conj())
    rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
    rho /= np.trace(rho)
    coh = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)
    ev = np.real(np.linalg.eigvalsh(rho))
    ev_pos = ev[ev > 1e-12]
    S = float(-np.sum(ev_pos * np.log2(ev_pos))) if len(ev_pos) > 0 else 0
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, 2, 2))
    neg = float(np.sum(np.abs(eigvals[eigvals < -1e-10])))
    return {'coherence': round(coh, 4), 'entropy': round(S, 4), 'negativity': round(neg, 6)}

def main():
    print("=" * 60)
    print("Phase Q267: CPT Symmetry & Time Reversal")
    print("  (Is the arrow of time irreversible?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4

    pairs = [
        ("time flows forward always", "always forward flows time"),
        ("cause precedes effect", "effect precedes cause"),
        ("entropy increases over time", "time over increases entropy"),
        ("the cat sat on the mat", "mat the on sat cat the"),
        ("quantum evolution is unitary", "unitary is evolution quantum"),
    ]

    all_results = []
    for forward, backward in pairs:
        # Forward pass
        inp_f = tok(forward, return_tensors='pt').to(device)
        with torch.no_grad():
            out_f = model(**inp_f, output_hidden_states=True)
        h_f = out_f.hidden_states[n_layers][0, -1, :].float().cpu().numpy()

        # Backward (reversed text)
        inp_b = tok(backward, return_tensors='pt').to(device)
        with torch.no_grad():
            out_b = model(**inp_b, output_hidden_states=True)
        h_b = out_b.hidden_states[n_layers][0, -1, :].float().cpu().numpy()

        # Time reversal = complex conjugation in quantum mechanics
        # For real vectors, T-reversal = sign flip of odd components
        h_t = h_f.copy()
        h_t[1::2] = -h_t[1::2]  # Flip odd components (T-reversal proxy)

        # Parity = spatial inversion = reverse order of components
        h_p = h_f[::-1].copy()

        # CPT combined
        h_cpt = h_t[::-1].copy()
        h_cpt[1::2] = -h_cpt[1::2]

        props_f = quantum_props(h_f, dim)
        props_b = quantum_props(h_b, dim)
        props_t = quantum_props(h_t, dim)
        props_cpt = quantum_props(h_cpt, dim)

        # Similarity between forward and T-reversed
        cos_ft = abs(float(np.dot(h_f[:dim] / (np.linalg.norm(h_f[:dim]) + 1e-10),
                                   h_t[:dim] / (np.linalg.norm(h_t[:dim]) + 1e-10))))
        cos_fb = abs(float(np.dot(h_f[:dim] / (np.linalg.norm(h_f[:dim]) + 1e-10),
                                   h_b[:dim] / (np.linalg.norm(h_b[:dim]) + 1e-10))))

        # CPT theorem: if CPT holds, coh_f == coh_cpt
        cpt_preserved = abs(props_f['coherence'] - props_cpt['coherence']) < 0.05

        print("  '%s' <-> '%s'" % (forward[:25], backward[:25]))
        print("    Forward: coh=%.4f, Backward: coh=%.4f" % (props_f['coherence'], props_b['coherence']))
        print("    T-reverse: coh=%.4f, CPT: coh=%.4f (preserved=%s)" % (
            props_t['coherence'], props_cpt['coherence'], cpt_preserved))

        all_results.append({
            'forward': forward[:30], 'backward': backward[:30],
            'props_forward': props_f, 'props_backward': props_b,
            'props_T': props_t, 'props_CPT': props_cpt,
            'cos_forward_backward': round(cos_fb, 4),
            'cos_forward_T': round(cos_ft, 4),
            'cpt_preserved': bool(cpt_preserved),
        })

    n_cpt = sum(1 for r in all_results if r['cpt_preserved'])
    avg_cos_fb = np.mean([r['cos_forward_backward'] for r in all_results])

    if n_cpt == len(all_results):
        verdict = "CPT SYMMETRIC: %d/%d preserved, time-reversal cos=%.3f" % (
            n_cpt, len(all_results), avg_cos_fb)
    elif n_cpt > len(all_results) // 2:
        verdict = "PARTIAL CPT: %d/%d preserved" % (n_cpt, len(all_results))
    else:
        verdict = "CPT BROKEN: only %d/%d preserved (arrow of time!)" % (n_cpt, len(all_results))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q267', 'name': 'CPT Symmetry',
        'pairs': all_results,
        'summary': {'n_cpt_preserved': n_cpt, 'total': len(all_results),
                     'avg_cos': round(avg_cos_fb, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q267_cpt.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_results))
    ax = axes[0]
    ax.bar(x - 0.2, [r['props_forward']['coherence'] for r in all_results], 0.2,
           label='Forward', color='#2196F3', edgecolor='black')
    ax.bar(x, [r['props_backward']['coherence'] for r in all_results], 0.2,
           label='Backward', color='#FF9800', edgecolor='black')
    ax.bar(x + 0.2, [r['props_CPT']['coherence'] for r in all_results], 0.2,
           label='CPT', color='#E91E63', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Coherence'); ax.set_title('(a) Coherence Comparison')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, [r['cos_forward_backward'] for r in all_results], color='#4CAF50', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['P%d' % (i+1) for i in range(len(all_results))])
    ax.set_ylabel('Cosine Similarity'); ax.set_title('(b) Forward vs Backward')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q267: CPT Symmetry\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q267_cpt.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ267 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
