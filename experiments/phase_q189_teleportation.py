# -*- coding: utf-8 -*-
"""
Phase Q189: Quantum Teleportation Protocol (Original idea by Opus)
====================================================================
Can we "teleport" a trained soul vector from one prompt context to another
without direct transfer, using entanglement as the channel?

Protocol:
1. Train soul vectors in Context A ("France is")
2. Create entangled pair between Context A and Context B ("Germany is")
3. Measure in Context A (Bell measurement)
4. Apply correction in Context B
5. Verify: Context B now produces Context A's output

If fidelity > 0.9 -> genuine quantum teleportation of semantic states!
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

INJECT_LAYER = 8


def train_soul(model, tok, data, device, layer=8, epochs=80, seed=42):
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
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def get_logits(model, tok, prompt, device, inject_vec, layer):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def kl_divergence(p, q):
    """KL(p||q)"""
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    return float(torch.sum(p * torch.log(p / q)))


def main():
    print("=" * 60)
    print("Phase Q189: Quantum Teleportation Protocol")
    print("  (Can Entanglement Teleport Semantic States?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size

    teleportation_tasks = [
        {
            'name': 'Capital',
            'source_data': [("The capital of France is","Paris")],
            'source_prompt': "The capital of France is",
            'target_prompt': "The capital of Japan is",
            'source_target': "Paris",
            'target_target': "Tokyo",
        },
        {
            'name': 'Color',
            'source_data': [("The sky is","blue"),("The ocean is","blue")],
            'source_prompt': "The sky is",
            'target_prompt': "Grass is",
            'source_target': "blue",
            'target_target': "green",
        },
        {
            'name': 'Animal',
            'source_data': [("A pet that purrs is a","cat")],
            'source_prompt': "A pet that purrs is a",
            'target_prompt': "A pet that barks is a",
            'source_target': "cat",
            'target_target': "dog",
        },
    ]

    results_list = []

    for task in teleportation_tasks:
        name = task['name']
        print("\n--- Teleportation: %s ---" % name)

        # Step 1: Train source soul vector
        print("  Step 1: Train source soul vector...")
        v_source = train_soul(model, tok, task['source_data'], device,
                             layer=INJECT_LAYER, epochs=80, seed=42)

        # Get source's output distribution
        probs_source = get_logits(model, tok, task['source_prompt'],
                                  device, v_source, INJECT_LAYER)
        source_token_id = tok.encode(task['source_target'])[-1]
        p_source_correct = float(probs_source[source_token_id])
        print("    Source P('%s')=%.4f" % (task['source_target'], p_source_correct))

        # Step 2: Create entangled pair via attention coupling
        # We create a "Bell pair" by training a joint soul vector
        print("  Step 2: Create entangled pair...")
        combined_prompt = task['source_prompt'] + " " + task['target_prompt']
        inp_combined = tok(combined_prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            out = model(**inp_combined, output_hidden_states=True)
            h_entangled = out.hidden_states[INJECT_LAYER][0, -1, :].float()

        # The entangled state captures correlations between source and target
        v_entangled = h_entangled.clone()

        # Step 3: "Teleport" - inject source vector but measure in target context
        print("  Step 3: Teleportation...")

        # Bell measurement: project source onto entangled basis
        # Correction: v_teleported = v_source projected through entangled channel
        overlap = torch.dot(v_source, v_entangled) / (
            torch.norm(v_entangled) + 1e-10)
        v_teleported = v_source - overlap * v_entangled / (
            torch.norm(v_entangled) + 1e-10)
        v_teleported = v_teleported / (torch.norm(v_teleported) + 1e-10) * torch.norm(v_source)

        # Measure teleported state in TARGET context
        probs_teleported = get_logits(model, tok, task['target_prompt'],
                                       device, v_teleported, INJECT_LAYER)
        p_teleported = float(probs_teleported[source_token_id])

        # Control: inject source vector directly into target (no entanglement)
        probs_direct = get_logits(model, tok, task['target_prompt'],
                                   device, v_source, INJECT_LAYER)
        p_direct = float(probs_direct[source_token_id])

        # Control 2: no injection (baseline)
        inp_target = tok(task['target_prompt'], return_tensors='pt').to(device)
        with torch.no_grad():
            out_baseline = model(**inp_target)
        probs_baseline = torch.softmax(out_baseline.logits[0, -1, :].float(), dim=-1)
        p_baseline = float(probs_baseline[source_token_id])

        # Fidelity: how similar is teleported distribution to source?
        fidelity = float(torch.sum(torch.sqrt(probs_source * probs_teleported)))
        kl = kl_divergence(probs_source, probs_teleported)

        result = {
            'name': name,
            'p_source_correct': round(p_source_correct, 4),
            'p_teleported': round(p_teleported, 4),
            'p_direct': round(p_direct, 4),
            'p_baseline': round(p_baseline, 4),
            'fidelity': round(fidelity, 4),
            'kl_divergence': round(kl, 4),
        }
        results_list.append(result)

        print("    Source P('%s')=%.4f" % (task['source_target'], p_source_correct))
        print("    Teleported P('%s')=%.4f" % (task['source_target'], p_teleported))
        print("    Direct inject P('%s')=%.4f" % (task['source_target'], p_direct))
        print("    Baseline P('%s')=%.4f" % (task['source_target'], p_baseline))
        print("    Fidelity=%.4f, KL=%.4f" % (fidelity, kl))

    # Summary
    avg_fidelity = float(np.mean([r['fidelity'] for r in results_list]))
    print("\n--- Summary ---")
    print("  Avg fidelity: %.4f" % avg_fidelity)

    if avg_fidelity > 0.9:
        verdict = "TELEPORTATION SUCCESS: avg fidelity=%.4f" % avg_fidelity
    elif avg_fidelity > 0.5:
        verdict = "PARTIAL TELEPORTATION: avg fidelity=%.4f" % avg_fidelity
    else:
        verdict = "TELEPORTATION FAILED: avg fidelity=%.4f" % avg_fidelity

    # Save
    results = {
        'phase': 'Q189',
        'name': 'Quantum Teleportation Protocol',
        'tasks': results_list,
        'summary': {
            'avg_fidelity': round(avg_fidelity, 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q189_teleportation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Probability comparison
    ax = axes[0]
    x = np.arange(len(results_list))
    w = 0.2
    ax.bar(x - 1.5*w, [r['p_source_correct'] for r in results_list], w,
           color='#4CAF50', label='Source', edgecolor='black')
    ax.bar(x - 0.5*w, [r['p_teleported'] for r in results_list], w,
           color='#2196F3', label='Teleported', edgecolor='black')
    ax.bar(x + 0.5*w, [r['p_direct'] for r in results_list], w,
           color='#FF9800', label='Direct', edgecolor='black')
    ax.bar(x + 1.5*w, [r['p_baseline'] for r in results_list], w,
           color='#9E9E9E', label='Baseline', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results_list])
    ax.set_ylabel('P(source token)')
    ax.set_title('(a) Teleportation Success\n(Higher = Better Transfer)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # (b) Fidelity
    ax = axes[1]
    fids = [r['fidelity'] for r in results_list]
    ax.bar(x, fids, color='#E91E63', edgecolor='black', alpha=0.85)
    ax.axhline(0.9, color='green', ls='--', label='Success threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results_list])
    ax.set_ylabel('Fidelity')
    ax.set_title('(b) Distribution Fidelity\n(Teleported vs Source)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Protocol diagram
    ax = axes[2]
    ax.text(0.5, 0.85, 'Quantum Teleportation Protocol', fontsize=14,
            ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.7, '1. Train |soul> in Context A', fontsize=11,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.55, '2. Create Bell pair via Attention', fontsize=11,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, '3. Bell measurement (projection)', fontsize=11,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, '4. Apply correction in Context B', fontsize=11,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.1, 'Avg Fidelity: %.4f' % avg_fidelity, fontsize=13,
            ha='center', fontweight='bold', color='#E91E63',
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(c) Protocol Steps')

    plt.suptitle('Q189: Quantum Teleportation Protocol\n%s' % verdict,
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q189_teleportation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ189 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
