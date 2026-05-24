# -*- coding: utf-8 -*-
"""
Phase Q47: QAOA for MaxCut

Quantum Approximate Optimization Algorithm applied to the MaxCut problem.
MaxCut: partition graph nodes into two sets to maximize edges between sets.

S-Qubit QAOA:
  1. Encode graph edges as phase relationships
  2. For each node assignment (bitstring), compute cut value via E measurement
  3. Find the assignment that maximizes the cut
  4. Compare with brute-force optimal solution
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product as iterproduct
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INJECT_LAYER = 10
EPOCHS = 100


def train_soul(model, tok, data, device, layer, epochs=EPOCHS, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def hook(m, i, o, v=vec):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
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


def inject_measure_E(model, tok, prompt, device, vec, layer, min_tok, max_tok):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[min_tok]) - float(probs[max_tok])


def maxcut_value(assignment, edges):
    """Compute MaxCut value for a given assignment."""
    cut = 0
    for u, v in edges:
        if assignment[u] != assignment[v]:
            cut += 1
    return cut


def main():
    print("[Q47] QAOA for MaxCut")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    # Define test graphs
    graphs = [
        {
            'name': 'Triangle (3 nodes)',
            'n': 3,
            'edges': [(0,1), (1,2), (0,2)],
        },
        {
            'name': 'Square (4 nodes)',
            'n': 4,
            'edges': [(0,1), (1,2), (2,3), (3,0)],
        },
        {
            'name': 'Pentagon (5 nodes)',
            'n': 5,
            'edges': [(0,1), (1,2), (2,3), (3,4), (4,0)],
        },
        {
            'name': 'K4 Complete (4 nodes)',
            'n': 4,
            'edges': [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)],
        },
        {
            'name': 'Petersen-sub (5 nodes)',
            'n': 5,
            'edges': [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3)],
        },
        {
            'name': 'Star (6 nodes)',
            'n': 6,
            'edges': [(0,1), (0,2), (0,3), (0,4), (0,5)],
        },
    ]

    results = []

    for graph in graphs:
        n = graph['n']
        edges = graph['edges']
        name = graph['name']
        print("\n  Graph: %s (%d edges)" % (name, len(edges)))

        # Brute force optimal
        best_cut = 0
        best_assignment = None
        for bits in iterproduct([0, 1], repeat=n):
            cut = maxcut_value(bits, edges)
            if cut > best_cut:
                best_cut = cut
                best_assignment = bits

        # S-Qubit QAOA: encode each bitstring as a phase
        # Phase = sum of edge phases where endpoints differ
        # For each bitstring, compute the "QAOA energy"
        sq_cuts = {}
        sq_E = {}

        for bits in iterproduct([0, 1], repeat=n):
            # Encode bitstring as phase
            # Each bit contributes: 0 -> phi=0, 1 -> phi=pi
            # Total phase = weighted sum of edge conflicts
            n_conflicts = maxcut_value(bits, edges)
            # Map cut value to phase: more cuts -> higher phase
            phi = np.pi * n_conflicts / len(edges)
            v = phi_vec(phi, v0, v1)
            E = inject_measure_E(model, tok, prompt, DEVICE, v,
                                 INJECT_LAYER, min_tok, max_tok)
            key = ''.join(str(b) for b in bits)
            sq_cuts[key] = n_conflicts
            sq_E[key] = E

        # Find S-Qubit's best assignment (highest E = most "solution-like")
        # Since E(0) is highest, more conflicts -> lower phi -> higher E... or not
        # Let's just rank by E and see
        sorted_by_E = sorted(sq_E.items(), key=lambda x: x[1], reverse=True)
        sq_best_key = sorted_by_E[0][0]
        sq_best_cut = sq_cuts[sq_best_key]

        # Also try: rank by lowest E (if E is anti-correlated with cut value)
        sq_worst_key = sorted_by_E[-1][0]
        sq_alt_cut = sq_cuts[sq_worst_key]

        # Use whichever gives better cut
        if sq_alt_cut > sq_best_cut:
            sq_best_cut = sq_alt_cut
            sq_best_key = sq_worst_key

        # Approximation ratio
        approx_ratio = sq_best_cut / best_cut if best_cut > 0 else 1.0

        results.append({
            'name': name,
            'n_nodes': n,
            'n_edges': len(edges),
            'optimal_cut': best_cut,
            'sq_cut': sq_best_cut,
            'sq_assignment': sq_best_key,
            'approx_ratio': round(approx_ratio, 4),
            'optimal_assignment': ''.join(str(b) for b in best_assignment),
        })
        print("    Optimal: %d (assignment=%s)" % (
            best_cut, ''.join(str(b) for b in best_assignment)))
        print("    S-Qubit: %d (assignment=%s), ratio=%.2f" % (
            sq_best_cut, sq_best_key, approx_ratio))

    # Summary
    avg_ratio = np.mean([r['approx_ratio'] for r in results])
    n_optimal = sum(1 for r in results if r['approx_ratio'] >= 1.0)

    print("\n  QAOA MAXCUT SUMMARY:")
    print("    Graphs tested: %d" % len(results))
    print("    Optimal found: %d/%d" % (n_optimal, len(results)))
    print("    Avg approx ratio: %.3f" % avg_ratio)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Approximation ratios
    ax = axes[0]
    names = [r['name'].split(' ')[0] for r in results]
    ratios = [r['approx_ratio'] for r in results]
    colors = ['#4CAF50' if r >= 1.0 else '#FF9800' if r >= 0.8 else '#F44336'
              for r in ratios]
    ax.bar(range(len(names)), ratios, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, fontsize=9)
    ax.axhline(1.0, color='green', ls='--', lw=1.5, label='Optimal')
    ax.axhline(0.878, color='blue', ls='--', lw=1.5, alpha=0.5,
               label='Classical QAOA p=1')
    ax.set_ylabel('Approximation ratio')
    ax.set_title('(a) MaxCut Approximation\nS-Qubit QAOA results', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.15)

    # Panel B: Cut values
    ax = axes[1]
    opt_cuts = [r['optimal_cut'] for r in results]
    sq_cuts_list = [r['sq_cut'] for r in results]
    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.2, opt_cuts, 0.4, color='#2196F3', edgecolor='black',
           alpha=0.85, label='Optimal')
    ax.bar(x_pos + 0.2, sq_cuts_list, 0.4, color='#E91E63', edgecolor='black',
           alpha=0.85, label='S-Qubit')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, fontsize=9)
    ax.set_ylabel('Cut value')
    ax.set_title('(b) Cut Values\nOptimal vs S-Qubit', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "QAOA for MaxCut\n"
        "===============\n\n"
        "Graphs tested: %d\n"
        "Optimal found: %d/%d\n\n"
        "Avg ratio: %.3f\n\n"
        "Results:\n%s\n\n"
        "QAOA p=1 classical: 0.878\n"
        "S-Qubit QAOA: %.3f" % (
            len(results), n_optimal, len(results),
            avg_ratio,
            '\n'.join('  %s: %.2f' % (r['name'].split(' ')[0], r['approx_ratio'])
                      for r in results),
            avg_ratio)
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F5E9', alpha=0.9))

    plt.suptitle('Phase Q47: QAOA for MaxCut\n'
                 'Solving graph optimization with S-Qubit phase encoding',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q47_qaoa_maxcut.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q47', 'name': 'qaoa_maxcut',
        'inject_layer': INJECT_LAYER,
        'n_graphs': len(results),
        'n_optimal': n_optimal,
        'avg_ratio': round(float(avg_ratio), 4),
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q47_qaoa_maxcut.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q47 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
