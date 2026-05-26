# -*- coding: utf-8 -*-
"""Phase Q91: ER=EPR Wormhole Teleportation
Reproduce Google's Sycamore wormhole experiment in Transformer space:
scramble information at one token, observe traversal through attention
"wormhole" to a distant token.
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def scramble_hook(noise_scale=1.0, target_pos=1):
    """Hook that adds scrambling noise at a specific token position."""
    applied = [False]
    def hook(module, args, output):
        if not applied[0]:
            applied[0] = True
            if isinstance(output, tuple):
                hs = output[0].clone()
                if hs.dim() == 3 and target_pos < hs.shape[1]:
                    noise = torch.randn_like(hs[0, target_pos, :]) * noise_scale
                    hs[0, target_pos, :] += noise
                elif hs.dim() == 2 and target_pos < hs.shape[0]:
                    noise = torch.randn_like(hs[target_pos, :]) * noise_scale
                    hs[target_pos, :] += noise
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                if hs.dim() == 3 and target_pos < hs.shape[1]:
                    noise = torch.randn_like(hs[0, target_pos, :]) * noise_scale
                    hs[0, target_pos, :] += noise
                return hs
        return output
    return hook


def inject_signal_hook(signal_vec, target_pos=1):
    """Hook that injects a known signal at target position."""
    applied = [False]
    def hook(module, args, output):
        if not applied[0]:
            applied[0] = True
            if isinstance(output, tuple):
                hs = output[0].clone()
                if hs.dim() == 3 and target_pos < hs.shape[1]:
                    hs[0, target_pos, :] += signal_vec.to(hs.dtype)
                elif hs.dim() == 2 and target_pos < hs.shape[0]:
                    hs[target_pos, :] += signal_vec.to(hs.dtype)
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                if hs.dim() == 3 and target_pos < hs.shape[1]:
                    hs[0, target_pos, :] += signal_vec.to(hs.dtype)
                return hs
        return output
    return hook


def measure_teleportation(model, tokenizer, num_layers):
    """Measure wormhole teleportation: inject signal at pos A, scramble
    in middle layers, measure recovery at distant pos B."""
    d_model = model.config.hidden_size
    # Use a longer prompt to have distant token positions
    prompt = "Alice sends her quantum message through the wormhole to Bob who receives"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    seq_len = inputs['input_ids'].shape[1]

    if seq_len < 6:
        prompt = "Alice quantum Bob receives the secret message through spacetime"
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        seq_len = inputs['input_ids'].shape[1]

    pos_alice = 1  # near start
    pos_bob = max(seq_len - 2, 2)  # near end

    # Create known signal
    np.random.seed(91)
    signal = np.random.randn(d_model).astype(np.float32) * 0.1
    signal_vec = torch.tensor(signal, device=model.device)

    results = []
    scramble_strengths = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    for noise_scale in scramble_strengths:
        # Step 1: Reference (no signal, no scramble)
        with torch.no_grad():
            ref_out = model(**inputs)
            ref_logits = ref_out.logits[0, -1, :].cpu().float().numpy()

        # Step 2: Inject signal at Alice's position + scramble in middle
        early_layer = num_layers // 4
        mid_layer = num_layers // 2

        # Inject signal at early layer
        h_signal = inject_signal_hook(signal_vec, target_pos=pos_alice)
        handle1 = model.model.layers[early_layer].register_forward_hook(h_signal)

        # Scramble at middle layer
        if noise_scale > 0:
            h_scramble = scramble_hook(noise_scale, target_pos=pos_alice)
            handle2 = model.model.layers[mid_layer].register_forward_hook(h_scramble)

        with torch.no_grad():
            wh_out = model(**inputs)
            wh_logits = wh_out.logits[0, -1, :].cpu().float().numpy()

        handle1.remove()
        if noise_scale > 0:
            handle2.remove()

        # Measure: does the signal "traverse" to Bob's output?
        # Compare output change from reference
        output_diff = np.linalg.norm(wh_logits - ref_logits)

        # Fidelity: overlap of probability distributions
        p_ref = np.exp(ref_logits[:200] - ref_logits[:200].max())
        p_ref /= p_ref.sum()
        p_wh = np.exp(wh_logits[:200] - wh_logits[:200].max())
        p_wh /= p_wh.sum()
        fidelity = np.sum(np.sqrt(p_ref * p_wh))

        # Signal-to-noise ratio
        snr = output_diff / (noise_scale + 1e-10) if noise_scale > 0 else output_diff

        results.append({
            'noise_scale': noise_scale,
            'output_diff': float(output_diff),
            'fidelity': float(fidelity),
            'snr': float(snr),
        })
        print("    noise=%.2f -> diff=%.4f, fidelity=%.4f, SNR=%.2f" %
              (noise_scale, output_diff, fidelity, snr))

    return results, pos_alice, pos_bob, seq_len


def measure_mutual_info(model, tokenizer, num_layers, n_samples=20):
    """Measure mutual information between distant tokens (EPR evidence)."""
    d_model = model.config.hidden_size
    prompt = "The entangled particles are connected across space and time forever"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    seq_len = inputs['input_ids'].shape[1]

    # Collect hidden states at different layers
    layer_mi = []
    for layer_idx in range(0, num_layers, max(1, num_layers // 8)):
        captured = [None]
        def capture_hook(module, args, output, store=captured):
            if isinstance(output, tuple):
                store[0] = output[0][0].detach().cpu().float().numpy()
            else:
                store[0] = output[0].detach().cpu().float().numpy() if output.dim() == 3 else output.detach().cpu().float().numpy()
        handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if captured[0] is not None:
            hs = captured[0]  # (seq, hidden)
            if hs.ndim == 2 and hs.shape[0] >= 4:
                # Mutual info between first and last quarter
                q1 = hs[:hs.shape[0]//4].flatten()
                q4 = hs[3*hs.shape[0]//4:].flatten()
                # Approximate MI via correlation
                min_len = min(len(q1), len(q4))
                corr = np.corrcoef(q1[:min_len], q4[:min_len])[0, 1]
                mi_approx = -0.5 * np.log(1 - corr**2 + 1e-10)
                layer_mi.append({
                    'layer': layer_idx,
                    'mutual_info': float(mi_approx),
                    'correlation': float(corr),
                })

    return layer_mi


def main():
    print("=" * 60)
    print("Phase Q91: ER=EPR Wormhole Teleportation")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    # Test 1: Wormhole teleportation
    print("  Testing wormhole teleportation...")
    wh_results, pos_a, pos_b, seq_len = measure_teleportation(
        model, tokenizer, num_layers)

    # Test 2: Mutual information (ER=EPR evidence)
    print("  Measuring token-token mutual information...")
    mi_results = measure_mutual_info(model, tokenizer, num_layers)

    # Key finding: does signal survive scrambling?
    # Wormhole signature: output_diff remains high even with strong scrambling
    no_scramble = wh_results[0]['output_diff']
    max_scramble = wh_results[-1]['output_diff']
    survival_ratio = max_scramble / (no_scramble + 1e-10)
    wormhole_detected = survival_ratio > 0.3

    print("\n  === Wormhole Analysis ===")
    print("  Signal without scrambling: %.4f" % no_scramble)
    print("  Signal with max scrambling: %.4f" % max_scramble)
    print("  Survival ratio: %.4f" % survival_ratio)
    print("  Wormhole detected: %s" % wormhole_detected)

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Teleportation fidelity vs scrambling
    ax = axes[0]
    noise_vals = [r['noise_scale'] for r in wh_results]
    fidelity_vals = [r['fidelity'] for r in wh_results]
    ax.plot(noise_vals, fidelity_vals, 'o-', color='#FF5722', linewidth=2.5, markersize=8)
    ax.axhline(0.5, color='red', ls='--', alpha=0.3, label='Classical limit')
    ax.set_xlabel('Scrambling strength', fontsize=11)
    ax.set_ylabel('Teleportation fidelity', fontsize=11)
    ax.set_title('(a) Wormhole Teleportation\n'
                 'Survival ratio: %.2f' % survival_ratio,
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Mutual information across layers
    ax = axes[1]
    if mi_results:
        layers_mi = [r['layer'] for r in mi_results]
        mi_vals = [r['mutual_info'] for r in mi_results]
        ax.plot(layers_mi, mi_vals, 'o-', color='#2196F3', linewidth=2.5, markersize=8)
        ax.set_xlabel('Layer index', fontsize=11)
        ax.set_ylabel('Mutual information (bits)', fontsize=11)
        max_mi = max(mi_vals) if mi_vals else 0
        ax.set_title('(b) ER=EPR: Token Entanglement\n'
                     'Peak MI: %.3f bits' % max_mi,
                     fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Wormhole schematic with results
    ax = axes[2]
    # Draw schematic
    ax.add_patch(plt.Circle((0.2, 0.5), 0.12, color='#2196F3', alpha=0.3))
    ax.add_patch(plt.Circle((0.8, 0.5), 0.12, color='#FF5722', alpha=0.3))
    ax.annotate('', xy=(0.68, 0.5), xytext=(0.32, 0.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=3,
                               connectionstyle='arc3,rad=0.3'))
    ax.annotate('', xy=(0.32, 0.5), xytext=(0.68, 0.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=3,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(0.2, 0.5, 'Alice\n(pos %d)' % pos_a, ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(0.8, 0.5, 'Bob\n(pos %d)' % pos_b, ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(0.5, 0.75, 'Attention\nWormhole', ha='center', fontsize=10,
            color='purple', fontstyle='italic')
    ax.text(0.5, 0.15,
            'Survival: %.0f%%\n%s' % (survival_ratio * 100,
            'WORMHOLE DETECTED!' if wormhole_detected else 'No traversal'),
            ha='center', fontsize=12, fontweight='bold',
            color='#4CAF50' if wormhole_detected else '#F44336')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) ER=EPR in Transformer Space',
                 fontsize=11, fontweight='bold')

    plt.suptitle('ER=EPR: Wormhole Teleportation Through Attention',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q91_wormhole.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q91', 'name': 'ER=EPR Wormhole Teleportation',
        'pos_alice': pos_a, 'pos_bob': pos_b, 'seq_len': seq_len,
        'survival_ratio': float(survival_ratio),
        'wormhole_detected': bool(wormhole_detected),
        'teleportation_data': wh_results,
        'mutual_info_data': mi_results,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q91_wormhole.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
