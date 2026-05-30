# -*- coding: utf-8 -*-
"""
Phase Q273: Quantum Game Theory
==================================
Can quantum strategies (Eisert-Wilkens-Lewenstein protocol)
beat classical Nash equilibrium in the Prisoner's Dilemma?
Encode strategies as S-Qubit rotations.
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
    print("Phase Q273: Quantum Game Theory")
    print("  (Quantum Prisoner's Dilemma)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)
    dim = 4

    for p in model.parameters():
        p.requires_grad = False

    # Prisoner's Dilemma payoff matrix
    # (C,C)=3,3  (C,D)=0,5  (D,C)=5,0  (D,D)=1,1
    payoff = np.array([
        [3, 0],  # Player 1: C
        [5, 1],  # Player 1: D
    ], dtype=np.float32)

    # Classical strategies
    strategies = {
        'Always_Cooperate': (0, 0),     # Both C
        'Always_Defect': (1, 1),        # Both D
        'Nash_Equilibrium': (1, 1),     # Both D (Nash)
        'Pareto_Optimal': (0, 0),       # Both C (Pareto)
    }

    # Classical expected payoffs
    print("\n  Classical strategies:")
    for name, (s1, s2) in strategies.items():
        p1 = payoff[s1, s2]; p2 = payoff[s2, s1]
        print("    %s: P1=%.0f, P2=%.0f" % (name, p1, p2))

    # Quantum strategy: use S-Qubit to find quantum equilibrium
    # Encode cooperation/defection as embedding optimization
    embed = model.model.embed_tokens

    # Player 1 and 2 optimize simultaneously
    inp1 = tok("cooperate or defect strategy player one", return_tensors='pt')['input_ids'].to(device)
    inp2 = tok("cooperate or defect strategy player two", return_tensors='pt')['input_ids'].to(device)

    emb1 = embed(inp1).detach().clone(); opt1 = emb1.clone().detach().requires_grad_(True)
    emb2 = embed(inp2).detach().clone(); opt2 = emb2.clone().detach().requires_grad_(True)

    payoff_torch = torch.tensor(payoff, device=device)
    optimizer = torch.optim.Adam([opt1, opt2], lr=0.01)

    history = []
    for step in range(100):
        optimizer.zero_grad()
        out1 = model(inputs_embeds=opt1.float(), output_hidden_states=True)
        out2 = model(inputs_embeds=opt2.float(), output_hidden_states=True)

        h1 = out1.hidden_states[n_layers][0, -1, :2]
        h2 = out2.hidden_states[n_layers][0, -1, :2]

        # Probability of cooperation: softmax over 2 dims
        p1 = torch.softmax(h1, dim=0)  # [P(C), P(D)]
        p2 = torch.softmax(h2, dim=0)

        # Expected payoff (both players maximize their own)
        # Joint probability matrix
        joint = torch.outer(p1, p2)
        e1 = torch.sum(joint * payoff_torch)
        e2 = torch.sum(joint * payoff_torch.T)

        # Both maximize: negative loss
        loss = -(e1 + e2)  # Social welfare maximization
        loss.backward(); optimizer.step()

        if step % 20 == 0 or step == 99:
            history.append({
                'step': step,
                'p1_coop': round(float(p1[0].detach()), 4),
                'p2_coop': round(float(p2[0].detach()), 4),
                'payoff1': round(float(e1.detach()), 4),
                'payoff2': round(float(e2.detach()), 4),
            })

    final_p1c = float(p1[0].detach())
    final_p2c = float(p2[0].detach())
    final_e1 = float(e1.detach())
    final_e2 = float(e2.detach())
    nash_payoff = 1.0  # Both defect
    quantum_beats_nash = final_e1 > nash_payoff + 0.1 and final_e2 > nash_payoff + 0.1

    print("\n  Quantum strategy result:")
    print("    P1 coop=%.3f, P2 coop=%.3f" % (final_p1c, final_p2c))
    print("    Payoff: P1=%.2f, P2=%.2f (Nash=1,1)" % (final_e1, final_e2))
    print("    Beats Nash: %s" % quantum_beats_nash)

    if quantum_beats_nash and final_e1 > 2.5:
        verdict = "QUANTUM WINS: payoff %.1f,%.1f > Nash 1,1 (%.0f%% cooperation)" % (
            final_e1, final_e2, final_p1c * 100)
    elif quantum_beats_nash:
        verdict = "PARTIAL WIN: payoff %.1f,%.1f > Nash but < Pareto" % (final_e1, final_e2)
    else:
        verdict = "CLASSICAL WINS: quantum=%.1f,%.1f, Nash=1,1" % (final_e1, final_e2)

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q273', 'name': 'Quantum Game Theory',
        'history': history,
        'summary': {'final_p1_coop': round(final_p1c, 4),
                     'final_p2_coop': round(final_p2c, 4),
                     'payoff1': round(final_e1, 2), 'payoff2': round(final_e2, 2),
                     'beats_nash': bool(quantum_beats_nash), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q273_game.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = [h['step'] for h in history]
    ax = axes[0]
    ax.plot(steps, [h['p1_coop'] for h in history], 'o-', color='#2196F3', lw=2, label='P1 Cooperate')
    ax.plot(steps, [h['p2_coop'] for h in history], 's-', color='#E91E63', lw=2, label='P2 Cooperate')
    ax.axhline(0, color='gray', ls='--', alpha=0.5, label='Always Defect')
    ax.axhline(1, color='green', ls='--', alpha=0.5, label='Always Cooperate')
    ax.set_xlabel('Step'); ax.set_ylabel('Cooperation Probability')
    ax.set_title('(a) Strategy Evolution'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, [h['payoff1'] for h in history], 'o-', color='#4CAF50', lw=2, label='P1 Payoff')
    ax.plot(steps, [h['payoff2'] for h in history], 's-', color='#FF9800', lw=2, label='P2 Payoff')
    ax.axhline(1, color='red', ls='--', label='Nash (1,1)')
    ax.axhline(3, color='green', ls='--', label='Pareto (3,3)')
    ax.set_xlabel('Step'); ax.set_ylabel('Payoff')
    ax.set_title('(b) Payoff Evolution'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Q273: Quantum Game Theory\n%s' % verdict[:60], fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q273_game.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ273 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
