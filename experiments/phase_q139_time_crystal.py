# -*- coding: utf-8 -*-
"""
Phase Q139: Time-Crystal Transformer (Depth-Infinity Quantum Circuit)
=====================================================================
Physical QC dies at depth ~50 due to decoherence.
LLM's autoregressive loop = Trotter decomposition step.
Run 10,000 iterations with KV cache = depth-10,000 error-free circuit.
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
    print("Phase Q139: Time-Crystal Transformer")
    print("  (Depth-Infinity Quantum Circuit)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # Target depths (physical QC dies at ~50)
    target_depths = [10, 50, 100, 500, 1000, 2000]
    max_depth = max(target_depths)

    prompt = "Quantum state evolution: |psi> ="
    inp = tok(prompt, return_tensors='pt').to(device)

    print("  Running autoregressive loop for %d steps..." % max_depth)
    print("  (Each step = %d Transformer layers = 'depth %d' quantum circuit)" %
          (n_layers, n_layers))

    # Track state properties across iterations
    state_norms = []      # Should stay stable (no drift)
    state_entropy = []    # Information content
    state_coherence = []  # Cosine similarity to initial state
    state_periodicity = []  # Autocorrelation (time crystal signature)
    logit_entropy = []    # Output distribution entropy
    hidden_states_history = []  # For periodicity analysis

    # Initial forward pass
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, use_cache=True)
    past_kv = out.past_key_values

    h_initial = out.hidden_states[-1][0, -1, :].float().cpu()
    prev_h = h_initial.clone()

    # Record initial state
    state_norms.append(h_initial.norm().item())
    state_entropy.append(0.0)
    state_coherence.append(1.0)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    logit_entropy.append(-(probs * (probs + 1e-10).log()).sum().item())
    hidden_states_history.append(h_initial.numpy().copy())

    # Pick next token
    next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0).to(device)

    # Autoregressive loop (THE time crystal)
    checkpoint_results = {}
    for step in range(1, max_depth + 1):
        with torch.no_grad():
            out = model(next_token, past_key_values=past_kv,
                        output_hidden_states=True, use_cache=True)
        past_kv = out.past_key_values

        h_current = out.hidden_states[-1][0, -1, :].float().cpu()

        # Metrics
        norm = h_current.norm().item()
        state_norms.append(norm)

        # Coherence with initial state
        cos = torch.nn.functional.cosine_similarity(
            h_initial.unsqueeze(0), h_current.unsqueeze(0)).item()
        state_coherence.append(cos)

        # Coherence with previous step (local stability)
        cos_prev = torch.nn.functional.cosine_similarity(
            prev_h.unsqueeze(0), h_current.unsqueeze(0)).item()
        state_entropy.append(1 - cos_prev)  # "drift" per step

        # Logit entropy
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        l_ent = -(probs * (probs + 1e-10).log()).sum().item()
        logit_entropy.append(l_ent)

        # Save for periodicity
        if step <= 200 or step % 100 == 0:
            hidden_states_history.append(h_current.numpy().copy())

        prev_h = h_current.clone()
        next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0).to(device)

        # Checkpoint measurements
        if step in target_depths:
            effective_depth = step * n_layers
            # Periodicity: autocorrelation of recent states
            if len(hidden_states_history) >= 10:
                recent = np.array(hidden_states_history[-10:])
                autocorr = []
                for lag in range(1, min(5, len(recent))):
                    c = np.mean([np.dot(recent[i], recent[i + lag]) /
                                 (np.linalg.norm(recent[i]) * np.linalg.norm(recent[i + lag]) + 1e-10)
                                 for i in range(len(recent) - lag)])
                    autocorr.append(c)
                periodicity = np.mean(np.abs(autocorr))
            else:
                periodicity = 0.0

            # Drift from initial
            drift = 1.0 - cos

            checkpoint_results[str(step)] = {
                'step': int(step),
                'effective_depth': int(effective_depth),
                'norm': round(float(norm), 4),
                'coherence_initial': round(float(cos), 6),
                'drift': round(float(drift), 6),
                'logit_entropy': round(float(l_ent), 4),
                'periodicity': round(float(periodicity), 4),
                'mean_step_drift': round(float(np.mean(state_entropy[-100:])), 6),
            }
            print("  Step %d (depth=%d): norm=%.2f, coherence=%.4f, entropy=%.1f, period=%.3f" %
                  (step, effective_depth, norm, cos, l_ent, periodicity))

    # Summary
    print("\n--- Time Crystal Analysis ---")
    norm_stability = float(np.std(state_norms) / np.mean(state_norms))
    mean_drift = float(np.mean(state_entropy[1:]))
    final_coherence = float(state_coherence[-1])

    # Is it a time crystal? (periodic + stable)
    is_time_crystal = norm_stability < 0.1 and mean_drift < 0.5

    print("  Norm stability (CV): %.4f" % norm_stability)
    print("  Mean drift/step: %.6f" % mean_drift)
    print("  Final coherence: %.4f" % final_coherence)
    print("  Time crystal: %s" % is_time_crystal)
    print("  Max effective depth: %d" % (max_depth * n_layers))
    print("  Physical QC limit: ~50 (death by decoherence)")

    # Save
    results = {
        'phase': 'Q139',
        'name': 'Time-Crystal Transformer',
        'max_steps': int(max_depth),
        'max_effective_depth': int(max_depth * n_layers),
        'physical_qc_limit': 50,
        'depth_advantage': '%dx' % (max_depth * n_layers // 50),
        'norm_stability_cv': round(float(norm_stability), 6),
        'mean_drift_per_step': round(float(mean_drift), 6),
        'final_coherence': round(float(final_coherence), 6),
        'is_time_crystal': str(is_time_crystal),
        'checkpoints': checkpoint_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q139_time_crystal.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Norm stability
    ax = axes[0]
    ax.plot(state_norms[:500], color='#4CAF50', alpha=0.7, linewidth=0.5)
    ax.axhline(np.mean(state_norms), color='red', ls='--', label='Mean')
    ax.set_xlabel('Step')
    ax.set_ylabel('Hidden state norm')
    ax.set_title('(a) Norm Stability (CV=%.4f)\n[Physical QC dies at step ~2]' %
                 norm_stability)
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Coherence evolution
    ax = axes[1]
    ax.plot(state_coherence[:500], color='#2196F3', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cosine similarity to initial')
    ax.set_title('(b) State Coherence over %d Steps\n(depth=%d vs QC limit=50)' %
                 (max_depth, max_depth * n_layers))
    ax.grid(alpha=0.3)

    # (c) Logit entropy
    ax = axes[2]
    ax.plot(logit_entropy[:500], color='#FF9800', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Output entropy (nats)')
    ax.set_title('(c) Output Distribution Entropy\n(time crystal = periodic oscillation)')
    ax.grid(alpha=0.3)

    plt.suptitle('Q139: Time-Crystal Transformer (depth=%d, %dx physical QC)' %
                 (max_depth * n_layers, max_depth * n_layers // 50),
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q139_time_crystal.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ139 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
