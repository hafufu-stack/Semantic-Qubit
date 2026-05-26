# -*- coding: utf-8 -*-
"""
Phase Q112: Dentate-Resonance Memory
=====================================
Direct implementation of the author's 2015 Master's thesis parameters.

Master's thesis finding:
  "LD (non-spatial) at 20-30Hz and MD (spatial/context) at 5Hz burst
   produce maximal EPSP summation and pattern separation in dentate gyrus"

Translation to S-Qubit:
  - LD = S-Qubit phase rotation (abstract semantic info) at 20-30 layer period
  - MD = Token context (spatial/sequential info) at 5 layer burst intervals
  - Pattern separation = perfect fact retrieval from long context

We test if applying these biological resonance parameters to S-Qubit
injection produces superior retrieval compared to uniform injection.
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
    print("Phase Q112: Dentate-Resonance Memory")
    print("  (Master's Thesis Parameters: LD=20-30Hz, MD=5Hz)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # Define test facts embedded in long context
    test_cases = [
        {
            'name': 'Scientific Fact',
            'context': "The theory of relativity was proposed by Einstein. "
                       "Newton discovered gravity. Darwin wrote Origin of Species. "
                       "The speed of light is exactly 299792458 meters per second. "
                       "Quantum mechanics describes atomic behavior. "
                       "The human genome has approximately 3 billion base pairs.",
            'query': "What is the speed of light in meters per second?",
            'target_token': '299',
        },
        {
            'name': 'Historical Date',
            'context': "Rome was founded in 753 BC. The French Revolution began in 1789. "
                       "World War I started in 1914. The moon landing was in 1969. "
                       "The Berlin Wall fell in 1989. The internet was invented in 1969.",
            'query': "When did World War I start?",
            'target_token': '1914',
        },
        {
            'name': 'Numerical Recall',
            'context': "Pi equals 3.14159. Euler number e equals 2.71828. "
                       "Golden ratio phi equals 1.61803. Square root of 2 is 1.41421. "
                       "Avogadro number is 6.022e23. Boltzmann constant is 1.38e-23.",
            'query': "What is the golden ratio phi?",
            'target_token': '1',  # 1.61803
        },
    ]

    # ===== Resonance parameters from Master's thesis =====
    # LD frequency: 20-30 Hz -> in layer space, period = n_layers / 25 ~ 1.1 layers
    # MD frequency: 5 Hz -> period = n_layers / 5 ~ 5.6 layers
    ld_period = max(1, n_layers // 25)  # ~1 layer (high frequency)
    md_period = max(1, n_layers // 5)   # ~5-6 layers (burst)

    print("  Resonance params: LD period=%d layers, MD period=%d layers" %
          (ld_period, md_period))
    print("  Total layers: %d, hidden: %d" % (n_layers, hidden))

    # Test three injection strategies
    strategies = {
        'uniform': list(range(0, n_layers, 2)),           # Every 2 layers
        'ld_only': list(range(0, n_layers, ld_period)),   # LD frequency
        'md_only': list(range(0, n_layers, md_period)),   # MD burst
        'resonance': [],  # LD + MD combined (thesis optimal)
    }

    # Build resonance pattern: LD oscillation modulated by MD burst envelope
    for l in range(n_layers):
        ld_phase = np.sin(2 * np.pi * l / max(ld_period, 1))
        md_envelope = np.cos(2 * np.pi * l / max(md_period, 1))
        # Fire when both LD and MD are positive (phase alignment)
        if ld_phase > 0 and md_envelope > 0.3:
            strategies['resonance'].append(l)

    print("  Strategy layer counts: %s" %
          {k: len(v) for k, v in strategies.items()})

    # ===== Run retrieval tests =====
    all_results = []

    for tc in test_cases:
        print("\n  Test: %s" % tc['name'])
        full_prompt = tc['context'] + " " + tc['query']
        inp = tok(full_prompt, return_tensors='pt').to(device)

        strategy_scores = {}
        for strat_name, inject_layers in strategies.items():
            # Create S-Qubit vector for target
            target_ids = tok(tc['target_token'], add_special_tokens=False)['input_ids']
            target_embed = model.model.embed_tokens(
                torch.tensor([target_ids[0]], device=device)).squeeze(0).float()

            # Apply injection at specified layers
            handles = []
            for li in inject_layers:
                if li < n_layers:
                    phase = 2 * np.pi * li / n_layers  # Phase rotation
                    amplitude = 0.01  # Small perturbation
                    perturbation = amplitude * torch.cos(
                        torch.tensor(phase)) * target_embed[:hidden].to(torch.float16)

                    def make_hook(pert):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0].clone()
                                if h.dim() == 3:
                                    h[0, -1, :] += pert.to(h.device, h.dtype)
                                return (h,) + output[1:]
                            return output
                        return hook

                    h = model.model.layers[li].register_forward_hook(make_hook(perturbation))
                    handles.append(h)

            # Forward pass with injection
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Measure target token probability
            target_prob = probs[target_ids[0]].item()

            # Clean up hooks
            for h in handles:
                h.remove()

            strategy_scores[strat_name] = target_prob
            print("    %s: target_prob=%.6f" % (strat_name, target_prob))

        # Baseline (no injection)
        with torch.no_grad():
            out_base = model(**inp)
        base_prob = torch.softmax(out_base.logits[0, -1, :], dim=-1)[target_ids[0]].item()

        # Calculate advantages
        resonance_gain = strategy_scores['resonance'] / max(base_prob, 1e-10)
        best_strat = max(strategy_scores, key=strategy_scores.get)

        all_results.append({
            'name': tc['name'],
            'baseline_prob': round(base_prob, 8),
            'strategy_scores': {k: round(v, 8) for k, v in strategy_scores.items()},
            'resonance_gain': round(resonance_gain, 4),
            'best_strategy': best_strat
        })

    # ===== Phase alignment analysis =====
    print("\n--- Phase Alignment Analysis ---")
    # Measure MD/LD phase alignment across layers (from Q102)
    prompts_md = ["The cat sat on the mat in the room"]
    prompts_ld = ["Abstract mathematical concept of infinity"]

    inp_md = tok(prompts_md[0], return_tensors='pt').to(device)
    inp_ld = tok(prompts_ld[0], return_tensors='pt').to(device)

    with torch.no_grad():
        out_md = model(**inp_md, output_hidden_states=True)
        out_ld = model(**inp_ld, output_hidden_states=True)

    phase_alignment = []
    for li in range(n_layers):
        h_md = out_md.hidden_states[li + 1][0, -1, :].float()
        h_ld = out_ld.hidden_states[li + 1][0, -1, :].float()
        cos = torch.nn.functional.cosine_similarity(
            h_md.unsqueeze(0), h_ld.unsqueeze(0)).item()

        # Check if this layer is in resonance pattern
        in_resonance = li in strategies['resonance']
        phase_alignment.append({
            'layer': li,
            'cosine': round(cos, 4),
            'in_resonance': in_resonance,
            'dg_firing': cos > 0.5
        })

    n_firing = sum(1 for p in phase_alignment if p['dg_firing'])
    n_resonance_firing = sum(1 for p in phase_alignment
                             if p['in_resonance'] and p['dg_firing'])
    print("  Firing layers: %d/%d" % (n_firing, n_layers))
    print("  Resonance layers that fire: %d/%d" %
          (n_resonance_firing, len(strategies['resonance'])))

    # ===== Save Results =====
    results = {
        'phase': 'Q112',
        'name': 'Dentate-Resonance Memory',
        'ld_period': ld_period,
        'md_period': md_period,
        'resonance_layers': strategies['resonance'],
        'retrieval_results': all_results,
        'phase_alignment': phase_alignment,
        'n_firing': n_firing,
        'n_resonance_firing': n_resonance_firing,
        'resonance_win_count': sum(1 for r in all_results
                                    if r['best_strategy'] == 'resonance'),
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q112_dentate_resonance.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Strategy comparison
    ax = axes[0]
    strat_names = ['baseline', 'uniform', 'ld_only', 'md_only', 'resonance']
    for i, tc_result in enumerate(all_results):
        vals = [tc_result['baseline_prob']] + \
               [tc_result['strategy_scores'][s] for s in strat_names[1:]]
        ax.plot(strat_names, vals, 'o-', label=tc_result['name'], markersize=6)
    ax.set_ylabel('Target token probability')
    ax.set_title('(a) Retrieval by Strategy')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(alpha=0.3, axis='y')

    # (b) Phase alignment profile
    ax = axes[1]
    layers_pa = [p['layer'] for p in phase_alignment]
    cosines_pa = [p['cosine'] for p in phase_alignment]
    colors_pa = ['#4CAF50' if p['in_resonance'] else '#2196F3' for p in phase_alignment]
    ax.bar(layers_pa, cosines_pa, color=colors_pa, alpha=0.85)
    ax.axhline(0.5, color='red', ls='--', label='Firing threshold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('MD-LD cosine similarity')
    ax.set_title('(b) Phase Alignment\n(green = resonance layers)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # (c) Resonance pattern
    ax = axes[2]
    layer_x = np.arange(n_layers)
    ld_signal = np.sin(2 * np.pi * layer_x / max(ld_period, 1))
    md_signal = np.cos(2 * np.pi * layer_x / max(md_period, 1))
    combined = ld_signal * np.where(md_signal > 0.3, 1, 0)
    ax.plot(layer_x, ld_signal, alpha=0.4, label='LD (high freq)', color='#2196F3')
    ax.plot(layer_x, md_signal, alpha=0.4, label='MD (burst)', color='#FF5722')
    ax.fill_between(layer_x, 0, combined, alpha=0.3, color='#4CAF50', label='Resonance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Signal amplitude')
    ax.set_title('(c) LD x MD Resonance Pattern\n(Master thesis params)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Q112: Dentate-Resonance Memory (2015 Thesis Bridge)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q112_dentate_resonance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ112 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
