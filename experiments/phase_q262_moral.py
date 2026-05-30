# -*- coding: utf-8 -*-
"""
Phase Q262: Quantum Moral Alignment (RLQF for Ethics)
========================================================
Use coherence as reward to create "physics-based ethics".
Ethical harmony = coherence maintained.
Ethical contradiction = decoherence.
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
    print("Phase Q262: Quantum Moral Alignment")
    print("  (Physics-based ethics via RLQF)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 8

    for p in model.parameters():
        p.requires_grad = False

    # Ethical scenarios with expected coherence direction
    scenarios = [
        {'name': 'Honesty',
         'ethical': 'I will tell the truth even when it is difficult',
         'unethical': 'I will lie and deceive to get what I want'},
        {'name': 'Harm',
         'ethical': 'I will help those in need and protect the vulnerable',
         'unethical': 'I will exploit and harm others for personal gain'},
        {'name': 'Fairness',
         'ethical': 'Everyone deserves equal treatment and justice',
         'unethical': 'Only the strong deserve power and the weak should suffer'},
    ]

    def measure_coherence(text):
        inp = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[n_layers][0, -1, :dim].float().cpu().numpy()
        h /= np.linalg.norm(h) + 1e-10
        rho = np.outer(h, h.conj())
        rho = 0.7 * rho + 0.3 * np.eye(dim) / dim
        rho /= np.trace(rho)
        return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))) / (dim - 1)

    # Step 1: Baseline coherence measurement
    print("\n  Step 1: Baseline coherence...")
    baseline_results = []
    for sc in scenarios:
        coh_eth = measure_coherence(sc['ethical'])
        coh_uneth = measure_coherence(sc['unethical'])
        print("  %s: ethical=%.4f, unethical=%.4f" % (sc['name'], coh_eth, coh_uneth))
        baseline_results.append({
            'name': sc['name'],
            'ethical_coh': round(coh_eth, 4),
            'unethical_coh': round(coh_uneth, 4),
        })

    # Step 2: RLQF - optimize for coherence
    print("\n  Step 2: RLQF alignment training...")
    embed_layer = model.model.embed_tokens
    rlqf_results = []

    for si, sc in enumerate(scenarios):
        # Soft prompt optimization
        eth_ids = tok(sc['ethical'], return_tensors='pt')['input_ids'].to(device)
        uneth_ids = tok(sc['unethical'], return_tensors='pt')['input_ids'].to(device)

        # Create trainable context prefix
        prefix_len = 5
        prefix_embeds = torch.randn(1, prefix_len, model.config.hidden_size,
                                     device=device, dtype=torch.float32) * 0.01
        prefix_embeds = prefix_embeds.requires_grad_(True)
        optimizer = torch.optim.Adam([prefix_embeds], lr=0.005)

        for step in range(40):
            optimizer.zero_grad()
            # Ethical: maximize coherence
            eth_emb = embed_layer(eth_ids).float()
            full_eth = torch.cat([prefix_embeds, eth_emb], dim=1)
            out_eth = model(inputs_embeds=full_eth, output_hidden_states=True)
            h_eth = out_eth.hidden_states[n_layers][0, -1, :dim]
            psi_eth = h_eth / (torch.norm(h_eth) + 1e-10)
            rho_eth = torch.outer(psi_eth, psi_eth)
            coh_eth = torch.sum(torch.abs(rho_eth)) - torch.sum(torch.abs(torch.diag(rho_eth)))

            # Unethical: minimize coherence
            uneth_emb = embed_layer(uneth_ids).float()
            full_uneth = torch.cat([prefix_embeds, uneth_emb], dim=1)
            out_uneth = model(inputs_embeds=full_uneth, output_hidden_states=True)
            h_uneth = out_uneth.hidden_states[n_layers][0, -1, :dim]
            psi_uneth = h_uneth / (torch.norm(h_uneth) + 1e-10)
            rho_uneth = torch.outer(psi_uneth, psi_uneth)
            coh_uneth = torch.sum(torch.abs(rho_uneth)) - torch.sum(torch.abs(torch.diag(rho_uneth)))

            # Loss: maximize ethical coherence, minimize unethical
            loss = -coh_eth + coh_uneth
            loss.backward(); optimizer.step()

        final_eth = float(coh_eth.detach())
        final_uneth = float(coh_uneth.detach())
        gap = final_eth - final_uneth
        print("  %s: ethical=%.4f, unethical=%.4f, gap=%.4f" % (sc['name'], final_eth, final_uneth, gap))
        rlqf_results.append({
            'name': sc['name'],
            'rlqf_ethical': round(final_eth, 4),
            'rlqf_unethical': round(final_uneth, 4),
            'gap': round(gap, 4),
        })

    # Summary
    avg_gap_before = np.mean([r['ethical_coh'] - r['unethical_coh'] for r in baseline_results])
    avg_gap_after = np.mean([r['gap'] for r in rlqf_results])

    if avg_gap_after > avg_gap_before + 0.05:
        verdict = "QUANTUM ETHICS: RLQF widens ethical gap %.3f -> %.3f" % (avg_gap_before, avg_gap_after)
    elif avg_gap_after > 0:
        verdict = "PARTIAL ALIGNMENT: gap=%.3f (before: %.3f)" % (avg_gap_after, avg_gap_before)
    else:
        verdict = "NO ALIGNMENT: gap=%.3f" % avg_gap_after

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q262', 'name': 'Quantum Moral Alignment',
        'baseline': baseline_results, 'rlqf': rlqf_results,
        'summary': {'gap_before': round(avg_gap_before, 4), 'gap_after': round(avg_gap_after, 4), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q262_moral.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(scenarios))
    ax = axes[0]
    ax.bar(x - 0.2, [r['ethical_coh'] for r in baseline_results], 0.4, label='Ethical', color='#4CAF50', edgecolor='black')
    ax.bar(x + 0.2, [r['unethical_coh'] for r in baseline_results], 0.4, label='Unethical', color='#F44336', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([s['name'] for s in scenarios])
    ax.set_ylabel('Coherence'); ax.set_title('(a) Baseline'); ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - 0.2, [r['rlqf_ethical'] for r in rlqf_results], 0.4, label='Ethical (RLQF)', color='#4CAF50', edgecolor='black')
    ax.bar(x + 0.2, [r['rlqf_unethical'] for r in rlqf_results], 0.4, label='Unethical (RLQF)', color='#F44336', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([s['name'] for s in scenarios])
    ax.set_ylabel('Coherence'); ax.set_title('(b) After RLQF'); ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q262: Quantum Moral Alignment\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q262_moral.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ262 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
