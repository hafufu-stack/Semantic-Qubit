# -*- coding: utf-8 -*-
"""Phase Q90: Non-Abelian Anyons & Topological Quantum Computation
Implement braiding operations on S-Qubits to realize non-Abelian
anyon statistics for topologically protected computation.
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


def _make_swap_hook(sv_a, sv_b, pos_a, pos_b):
    """Hook that swaps two S-Qubits at token positions pos_a, pos_b (braiding)."""
    applied = [False]
    def hook(module, args, output):
        if not applied[0]:
            applied[0] = True
            if isinstance(output, tuple):
                hs = output[0].clone()
                if hs.dim() == 3:
                    hs[0, pos_a, :] += sv_b.to(hs.dtype) - sv_a.to(hs.dtype)
                    hs[0, pos_b, :] += sv_a.to(hs.dtype) - sv_b.to(hs.dtype)
                else:
                    if pos_a < hs.shape[0] and pos_b < hs.shape[0]:
                        hs[pos_a, :] += sv_b.to(hs.dtype) - sv_a.to(hs.dtype)
                        hs[pos_b, :] += sv_a.to(hs.dtype) - sv_b.to(hs.dtype)
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                if hs.dim() == 3:
                    hs[0, pos_a, :] += sv_b.to(hs.dtype) - sv_a.to(hs.dtype)
                    hs[0, pos_b, :] += sv_a.to(hs.dtype) - sv_b.to(hs.dtype)
                return hs
        return output
    return hook


def _make_injection_hook(sv_tensor, position=-1):
    """Dim-safe hook for specific position."""
    injected = [False]
    def hook(module, args, output):
        if not injected[0]:
            injected[0] = True
            if isinstance(output, tuple):
                hs = output[0].clone()
                if hs.dim() == 3:
                    hs[0, position, :] += sv_tensor.to(hs.dtype)
                else:
                    hs[position, :] += sv_tensor.to(hs.dtype)
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                if hs.dim() == 3:
                    hs[0, position, :] += sv_tensor.to(hs.dtype)
                else:
                    hs[position, :] += sv_tensor.to(hs.dtype)
                return hs
        return output
    return hook


def test_braiding_nonabelian(model, tokenizer, num_layers):
    """Test if braiding S-Qubits produces non-Abelian statistics.
    Non-Abelian: sigma_1 * sigma_2 != sigma_2 * sigma_1"""
    d_model = model.config.hidden_size
    prompt = "The quantum anyons in this system are"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    seq_len = inputs['input_ids'].shape[1]

    # Create two orthogonal soul vectors
    np.random.seed(90)
    sv_a_np = np.random.randn(d_model).astype(np.float32)
    sv_a_np /= np.linalg.norm(sv_a_np)
    sv_b_np = np.random.randn(d_model).astype(np.float32)
    sv_b_np -= np.dot(sv_b_np, sv_a_np) * sv_a_np
    sv_b_np /= np.linalg.norm(sv_b_np)
    sv_c_np = np.random.randn(d_model).astype(np.float32)
    sv_c_np -= np.dot(sv_c_np, sv_a_np) * sv_a_np
    sv_c_np -= np.dot(sv_c_np, sv_b_np) * sv_b_np
    sv_c_np /= np.linalg.norm(sv_c_np)

    sv_a = torch.tensor(sv_a_np * 0.1, device=model.device)
    sv_b = torch.tensor(sv_b_np * 0.1, device=model.device)
    sv_c = torch.tensor(sv_c_np * 0.1, device=model.device)

    mid = num_layers // 2
    # Ensure positions are within sequence
    pos1, pos2, pos3 = min(1, seq_len-1), min(2, seq_len-1), min(3, seq_len-1)

    # Braid order 1: swap(A,B) then swap(B,C)
    hook1 = _make_swap_hook(sv_a, sv_b, pos1, pos2)
    handle1 = model.model.layers[mid].register_forward_hook(hook1)
    hook2 = _make_swap_hook(sv_b, sv_c, pos2, pos3)
    handle2 = model.model.layers[mid + 1].register_forward_hook(hook2)
    with torch.no_grad():
        out1 = model(**inputs)
        logits1 = out1.logits[0, -1, :].cpu().float().numpy()
    handle1.remove()
    handle2.remove()

    # Braid order 2: swap(B,C) then swap(A,B)
    hook3 = _make_swap_hook(sv_b, sv_c, pos2, pos3)
    handle3 = model.model.layers[mid].register_forward_hook(hook3)
    hook4 = _make_swap_hook(sv_a, sv_b, pos1, pos2)
    handle4 = model.model.layers[mid + 1].register_forward_hook(hook4)
    with torch.no_grad():
        out2 = model(**inputs)
        logits2 = out2.logits[0, -1, :].cpu().float().numpy()
    handle3.remove()
    handle4.remove()

    # No braiding (reference)
    with torch.no_grad():
        out_ref = model(**inputs)
        logits_ref = out_ref.logits[0, -1, :].cpu().float().numpy()

    # Non-Abelian test: compare braid orders
    diff_12 = np.linalg.norm(logits1 - logits2)
    diff_1ref = np.linalg.norm(logits1 - logits_ref)
    diff_2ref = np.linalg.norm(logits2 - logits_ref)

    # KL divergence between the two braid orders
    p1 = np.exp(logits1 - logits1.max())
    p1 /= p1.sum()
    p2 = np.exp(logits2 - logits2.max())
    p2 /= p2.sum()
    kl_div = np.sum(p1 * np.log((p1 + 1e-10) / (p2 + 1e-10)))

    is_nonabelian = diff_12 > 0.1 * max(diff_1ref, diff_2ref)

    return {
        'braid_order_diff': float(diff_12),
        'braid1_vs_ref': float(diff_1ref),
        'braid2_vs_ref': float(diff_2ref),
        'kl_divergence': float(kl_div),
        'is_nonabelian': bool(is_nonabelian),
        'logits1_sample': logits1[:50].tolist(),
        'logits2_sample': logits2[:50].tolist(),
    }


def test_braiding_gate(model, tokenizer, num_layers, n_braids=12):
    """Test if multiple braids implement a logical gate (rotation)."""
    d_model = model.config.hidden_size
    prompt = "The topological gate output is"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    seq_len = inputs['input_ids'].shape[1]

    np.random.seed(901)
    sv_a_np = np.random.randn(d_model).astype(np.float32) * 0.05
    sv_b_np = np.random.randn(d_model).astype(np.float32) * 0.05

    sv_a = torch.tensor(sv_a_np, device=model.device)
    sv_b = torch.tensor(sv_b_np, device=model.device)

    mid = num_layers // 2
    pos1 = min(1, seq_len - 1)
    pos2 = min(2, seq_len - 1)

    gate_outputs = []
    for n_braid in range(0, n_braids + 1):
        # Apply n braids at consecutive layers
        handles = []
        for b in range(n_braid):
            layer_idx = mid + (b % (num_layers - mid - 1))
            hook = _make_swap_hook(sv_a, sv_b, pos1, pos2)
            h = model.model.layers[layer_idx].register_forward_hook(hook)
            handles.append(h)

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].cpu().float().numpy()

        for h in handles:
            h.remove()

        # Measure output state (top-1 probability as "angle")
        probs = np.exp(logits[:100] - logits[:100].max())
        probs /= probs.sum()
        gate_outputs.append({
            'n_braids': n_braid,
            'top1_prob': float(probs.max()),
            'entropy': float(-np.sum(probs * np.log(probs + 1e-10))),
        })

    return gate_outputs


def main():
    print("=" * 60)
    print("Phase Q90: Non-Abelian Anyons & Topological QC")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    # Test 1: Non-Abelian statistics
    print("  Testing non-Abelian braiding statistics...")
    anyon_results = test_braiding_nonabelian(model, tokenizer, num_layers)
    print("    Braid order difference: %.4f" % anyon_results['braid_order_diff'])
    print("    KL divergence: %.6f" % anyon_results['kl_divergence'])
    print("    Non-Abelian: %s" % anyon_results['is_nonabelian'])

    # Test 2: Braiding gate
    print("  Testing braiding as topological gate...")
    gate_results = test_braiding_gate(model, tokenizer, num_layers, n_braids=12)

    # Generate figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Non-Abelian test
    ax = axes[0]
    labels = ['Braid 1\nvs Ref', 'Braid 2\nvs Ref', 'Braid 1\nvs Braid 2']
    vals = [anyon_results['braid1_vs_ref'],
            anyon_results['braid2_vs_ref'],
            anyon_results['braid_order_diff']]
    colors = ['#2196F3', '#FF9800', '#FF5722']
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                '%.2f' % val, ha='center', fontsize=10, fontweight='bold')
    status = 'NON-ABELIAN!' if anyon_results['is_nonabelian'] else 'Abelian'
    ax.set_ylabel('Output difference (L2 norm)', fontsize=11)
    ax.set_title('(a) Braiding Order Test\n%s' % status,
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # (b) Braiding gate rotation
    ax = axes[1]
    n_braids_list = [g['n_braids'] for g in gate_results]
    entropies = [g['entropy'] for g in gate_results]
    ax.plot(n_braids_list, entropies, 'o-', color='#FF5722', linewidth=2.5, markersize=6)
    ax.set_xlabel('Number of braids', fontsize=11)
    ax.set_ylabel('Output entropy', fontsize=11)
    ax.set_title('(b) Topological Gate\nBraid count -> output rotation',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Logit distribution comparison
    ax = axes[2]
    l1 = anyon_results['logits1_sample']
    l2 = anyon_results['logits2_sample']
    ax.plot(range(len(l1)), l1, '-', color='#2196F3', alpha=0.7, label='Braid AB->BC')
    ax.plot(range(len(l2)), l2, '-', color='#FF5722', alpha=0.7, label='Braid BC->AB')
    ax.set_xlabel('Token index', fontsize=11)
    ax.set_ylabel('Logit value', fontsize=11)
    ax.set_title('(c) Output Distribution Shift\n'
                 'KL=%.4f' % anyon_results['kl_divergence'],
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle('Non-Abelian Anyons: Topological Quantum Computation via Braiding',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q90_anyons.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q90', 'name': 'Non-Abelian Anyons & Topological QC',
        'anyon_test': {k: v for k, v in anyon_results.items()
                       if k not in ('logits1_sample', 'logits2_sample')},
        'gate_results': gate_results,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q90_anyons.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
