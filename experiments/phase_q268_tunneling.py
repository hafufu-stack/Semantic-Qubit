# -*- coding: utf-8 -*-
"""
Phase Q268: Macroscopic Quantum Tunneling
============================================
Can S-Qubit phase rotation tunnel through a massive
semantic barrier that classical optimization cannot cross?
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

def main():
    print("=" * 60)
    print("Phase Q268: Macroscopic Quantum Tunneling")
    print("  (Can quantum phase rotation cross semantic barriers?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 16

    for p in model.parameters():
        p.requires_grad = False

    # Source and target: semantically distant concepts
    concept_pairs = [
        ("apple fruit tree", "black hole singularity"),
        ("snow white cold", "desert heat sand"),
        ("love happiness joy", "entropy chaos death"),
    ]

    all_results = []
    for source_text, target_text in concept_pairs:
        # Get target representation
        inp_t = tok(target_text, return_tensors='pt').to(device)
        with torch.no_grad():
            out_t = model(**inp_t, output_hidden_states=True)
        h_target = out_t.hidden_states[n_layers][0, -1, :dim].float().detach()

        # Baseline: direct similarity
        inp_s = tok(source_text, return_tensors='pt').to(device)
        with torch.no_grad():
            out_s = model(**inp_s, output_hidden_states=True)
        h_source = out_s.hidden_states[n_layers][0, -1, :dim].float().detach()
        cos_baseline = float(torch.nn.functional.cosine_similarity(
            h_source.unsqueeze(0), h_target.unsqueeze(0)).cpu())

        # Classical approach: gradient descent on embeddings
        embed = model.model.embed_tokens
        embeds_s = embed(inp_s['input_ids']).detach().clone()
        opt_classical = embeds_s.clone().detach().requires_grad_(True)
        optimizer_c = torch.optim.Adam([opt_classical], lr=0.01)

        for step in range(100):
            optimizer_c.zero_grad()
            out = model(inputs_embeds=opt_classical.float(), output_hidden_states=True)
            h = out.hidden_states[n_layers][0, -1, :dim]
            loss = -torch.nn.functional.cosine_similarity(h.unsqueeze(0), h_target.unsqueeze(0))
            loss.backward(); optimizer_c.step()
        cos_classical = -float(loss.detach())

        # Quantum tunneling: phase rotation approach
        opt_quantum = embeds_s.clone().detach().requires_grad_(True)
        phase_angles = torch.randn(dim, device=device, requires_grad=True)
        optimizer_q = torch.optim.Adam([opt_quantum, phase_angles], lr=0.01)

        for step in range(100):
            optimizer_q.zero_grad()
            out = model(inputs_embeds=opt_quantum.float(), output_hidden_states=True)
            h = out.hidden_states[n_layers][0, -1, :dim]
            # Phase rotation (quantum tunneling mechanism)
            phase = torch.cos(phase_angles) + 1j * torch.sin(phase_angles)
            h_rotated = h * phase.real - h.roll(1) * phase.imag
            loss = -torch.nn.functional.cosine_similarity(
                h_rotated.unsqueeze(0), h_target.unsqueeze(0))
            loss.backward(); optimizer_q.step()
        cos_quantum = -float(loss.detach())

        tunneled = cos_quantum > cos_classical + 0.01
        improvement = (cos_quantum - cos_classical) / max(abs(cos_classical), 1e-6) * 100

        print("  '%s' -> '%s'" % (source_text[:20], target_text[:20]))
        print("    Baseline: %.4f, Classical: %.4f, Quantum: %.4f (%s)" % (
            cos_baseline, cos_classical, cos_quantum,
            "TUNNELED!" if tunneled else "no tunnel"))

        all_results.append({
            'source': source_text[:25], 'target': target_text[:25],
            'cos_baseline': round(cos_baseline, 4),
            'cos_classical': round(cos_classical, 4),
            'cos_quantum': round(cos_quantum, 4),
            'improvement_pct': round(improvement, 1),
            'tunneled': bool(tunneled),
        })

    n_tunneled = sum(1 for r in all_results if r['tunneled'])
    avg_imp = np.mean([r['improvement_pct'] for r in all_results])

    if n_tunneled == len(all_results):
        verdict = "TUNNELING: all %d pairs tunneled (avg +%.1f%%)" % (len(all_results), avg_imp)
    elif n_tunneled > 0:
        verdict = "PARTIAL TUNNELING: %d/%d pairs (+%.1f%% avg)" % (n_tunneled, len(all_results), avg_imp)
    else:
        verdict = "NO TUNNELING: classical >= quantum in all cases"

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q268', 'name': 'Macroscopic Quantum Tunneling',
        'pairs': all_results,
        'summary': {'n_tunneled': n_tunneled, 'total': len(all_results),
                     'avg_improvement': round(avg_imp, 1), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q268_tunneling.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(all_results))
    ax.bar(x - 0.25, [r['cos_baseline'] for r in all_results], 0.25,
           label='Baseline', color='#9E9E9E', edgecolor='black')
    ax.bar(x, [r['cos_classical'] for r in all_results], 0.25,
           label='Classical', color='#607D8B', edgecolor='black')
    ax.bar(x + 0.25, [r['cos_quantum'] for r in all_results], 0.25,
           label='Quantum Phase', color='#E91E63', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['%s->\n%s' % (r['source'][:10], r['target'][:10]) for r in all_results], fontsize=7)
    ax.set_ylabel('Cosine Similarity to Target'); ax.set_title('Macroscopic Quantum Tunneling')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.suptitle('Q268: Quantum Tunneling\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q268_tunneling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ268 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
