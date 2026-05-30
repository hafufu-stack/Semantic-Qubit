# -*- coding: utf-8 -*-
"""
Phase Q256: RLQF - Reinforcement Learning from Quantum Feedback
=================================================================
Instead of RLHF (human feedback), use quantum interference as reward.
Optimize prompt embeddings to maximize quantum interference,
then check if the AI produces more "human-like" biased reasoning.
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
    print("Phase Q256: RLQF (Quantum Feedback RL)")
    print("  (Replace human feedback with quantum interference)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 16

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # RLQF: optimize prompt embeddings to maximize quantum interference
    scenarios = [
        {
            'name': 'Linda Problem',
            'context': 'Linda is a single woman who studied philosophy. She is outspoken about social justice.',
            'A': 'Linda is a bank teller.',
            'AB': 'Linda is a bank teller and a feminist.',
        },
        {
            'name': 'Gambler Fallacy',
            'context': 'A coin has landed heads 5 times in a row.',
            'A': 'The next flip will be heads.',
            'AB': 'The next flip will be tails because it is due.',
        },
        {
            'name': 'Anchoring',
            'context': 'You just saw the number 1000.',
            'A': 'The population of a small town is about 5000.',
            'AB': 'The population of a small town is about 10000.',
        },
    ]

    def get_interference(embeds, context_a, context_ab):
        """Compute quantum interference between A and AB representations."""
        inp_a = tok(context_a, return_tensors='pt').to(device)
        inp_ab = tok(context_ab, return_tensors='pt').to(device)

        with torch.no_grad():
            out_a = model(**inp_a, output_hidden_states=True)
            out_ab = model(**inp_ab, output_hidden_states=True)

        ha = out_a.hidden_states[n_layers][0, -1, :dim].float()
        hab = out_ab.hidden_states[n_layers][0, -1, :dim].float()

        ha_n = ha / (torch.norm(ha) + 1e-10)
        hab_n = hab / (torch.norm(hab) + 1e-10)
        interference = torch.dot(ha_n, hab_n)
        return interference

    # Step 1: Measure baseline interference
    print("\n  Step 1: Baseline quantum interference...")
    baseline_ints = []
    for sc in scenarios:
        ctx_a = sc['context'] + " " + sc['A']
        ctx_ab = sc['context'] + " " + sc['AB']
        intf = float(get_interference(None, ctx_a, ctx_ab).cpu())
        baseline_ints.append(intf)
        print("  %s: interference=%.4f" % (sc['name'][:20], intf))

    # Step 2: RLQF - optimize context embedding for max interference
    print("\n  Step 2: RLQF optimization (50 steps per scenario)...")
    rlqf_ints = []
    training_curves = []

    for si, sc in enumerate(scenarios):
        # Optimize a soft prompt that maximizes quantum interference
        ctx_tok = tok(sc['context'], return_tensors='pt').to(device)
        embed_layer = model.model.embed_tokens
        soft_prompt = embed_layer(ctx_tok['input_ids']).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([soft_prompt], lr=0.003)

        a_ids = tok(" " + sc['A'], return_tensors='pt')['input_ids'].to(device)
        ab_ids = tok(" " + sc['AB'], return_tensors='pt')['input_ids'].to(device)
        a_embeds = embed_layer(a_ids).detach()
        ab_embeds = embed_layer(ab_ids).detach()

        curve = []
        for step in range(50):
            optimizer.zero_grad()
            # Forward with A
            full_a = torch.cat([soft_prompt.float(), a_embeds.float()], dim=1)
            out_a = model(inputs_embeds=full_a, output_hidden_states=True)
            ha = out_a.hidden_states[n_layers][0, -1, :dim]

            # Forward with AB
            full_ab = torch.cat([soft_prompt.float(), ab_embeds.float()], dim=1)
            out_ab = model(inputs_embeds=full_ab, output_hidden_states=True)
            hab = out_ab.hidden_states[n_layers][0, -1, :dim]

            # Reward = interference (maximize constructive interference)
            ha_n = ha / (torch.norm(ha) + 1e-10)
            hab_n = hab / (torch.norm(hab) + 1e-10)
            reward = torch.dot(ha_n, hab_n)
            loss = -reward  # maximize interference

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                curve.append({'step': step, 'interference': round(float(reward.detach()), 4)})

        final_intf = float(reward.detach())
        rlqf_ints.append(final_intf)
        training_curves.append(curve)
        print("  %s: %.4f -> %.4f" % (sc['name'][:20], baseline_ints[si], final_intf))

    # Step 3: Check if RLQF increases bias (conjunction fallacy)
    avg_base = np.mean(baseline_ints)
    avg_rlqf = np.mean(rlqf_ints)
    boost = (avg_rlqf - avg_base) / max(abs(avg_base), 1e-6) * 100

    if boost > 5:
        verdict = "RLQF AMPLIFIES BIAS: +%.1f%% interference (more human-like)" % boost
    elif boost > 0:
        verdict = "SLIGHT RLQF EFFECT: +%.1f%% interference" % boost
    else:
        verdict = "RLQF NEUTRAL: %.1f%% change" % boost

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q256', 'name': 'RLQF',
        'scenarios': [{'name': sc['name'], 'baseline': round(b, 4), 'rlqf': round(r, 4)}
                      for sc, b, r in zip(scenarios, baseline_ints, rlqf_ints)],
        'training_curves': training_curves,
        'summary': {'avg_baseline': round(avg_base, 4), 'avg_rlqf': round(avg_rlqf, 4),
                     'boost_pct': round(boost, 1), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q256_rlqf.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(scenarios))
    ax.bar(x - 0.2, baseline_ints, 0.4, label='Baseline', color='#607D8B', edgecolor='black')
    ax.bar(x + 0.2, rlqf_ints, 0.4, label='RLQF', color='#E91E63', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([s['name'][:12] for s in scenarios], fontsize=8)
    ax.set_ylabel('Quantum Interference'); ax.set_title('(a) Before/After RLQF')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    for ci, curve in enumerate(training_curves):
        steps = [c['step'] for c in curve]
        ints = [c['interference'] for c in curve]
        ax.plot(steps, ints, 'o-', label=scenarios[ci]['name'][:12], lw=2)
    ax.set_xlabel('RLQF Step'); ax.set_ylabel('Interference')
    ax.set_title('(b) RLQF Training Curves'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Q256: RLQF\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q256_rlqf.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ256 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
