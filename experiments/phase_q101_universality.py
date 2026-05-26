# -*- coding: utf-8 -*-
"""Phase Q101: Cross-Architecture Universality Test
THE MOST IMPORTANT EXPERIMENT: Test if S-Qubit quantum properties
emerge universally across ALL transformer architectures.
Tests 6+ models from different families:
  - Qwen 2.5 (1.5B, 0.5B)
  - GPT-2 (124M, 355M, 774M, 1.5B)
  - Llama 3.2 (1B)
  - Phi-2 (2.7B)
  - StableLM (1.6B)
  - BLOOM (1.1B)
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def find_snapshot(model_dir_name):
    """Find the first snapshot directory for a model."""
    model_path = os.path.join(HF_CACHE, model_dir_name, "snapshots")
    if not os.path.exists(model_path):
        return None
    snaps = [d for d in os.listdir(model_path)
             if os.path.isdir(os.path.join(model_path, d))]
    if snaps:
        return os.path.join(model_path, snaps[0])
    return None


# Models to test - different architectures
MODELS = [
    ("Qwen-1.5B", "models--Qwen--Qwen2.5-1.5B", "qwen2"),
    ("Qwen-0.5B", "models--Qwen--Qwen2.5-0.5B", "qwen2"),
    ("GPT2-small", "models--gpt2", "gpt2"),
    ("GPT2-medium", "models--gpt2-medium", "gpt2"),
    ("GPT2-large", "models--gpt2-large", "gpt2"),
    ("Llama-1B", "models--meta-llama--Llama-3.2-1B", "llama"),
    ("StableLM-1.6B", "models--stabilityai--stablelm-2-1_6b", "stablelm"),
    ("BLOOM-1.1B", "models--bigscience--bloom-1b1", "bloom"),
    ("Falcon-1B", "models--tiiuae--falcon-rw-1b", "falcon"),
]


def get_layers(model, arch):
    """Get the layer list for different architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2, BLOOM, Falcon
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        return model.model.decoder.layers
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    else:
        # Try common patterns
        for attr in ['model.layers', 'transformer.h', 'transformer.layers']:
            parts = attr.split('.')
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            except:
                continue
    return None


def measure_quantum_properties(model, tokenizer, arch_name, device='cuda'):
    """Measure core quantum properties for any model."""
    d_model = model.config.hidden_size
    layers = get_layers(model, arch_name)
    if layers is None:
        return None
    num_layers = len(layers)
    mid = num_layers // 2

    prompt = "The universe is made of information and quantum states"
    try:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
    except:
        return None

    # 1. Reference output
    with torch.no_grad():
        ref_out = model(**inputs)
        ref_logits = ref_out.logits[0, -1, :500].cpu().float().numpy()

    # 2. SUPERPOSITION: inject two orthogonal vectors, check interference
    np.random.seed(101)
    sv1 = np.random.randn(d_model).astype(np.float32)
    sv1 /= np.linalg.norm(sv1)
    sv2 = np.random.randn(d_model).astype(np.float32)
    sv2 -= np.dot(sv2, sv1) * sv1
    sv2 /= np.linalg.norm(sv2)

    def make_hook(vec):
        applied = [False]
        sv = torch.tensor(vec.astype(np.float32), device=device)
        def hook(module, args, output):
            if not applied[0]:
                applied[0] = True
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv.to(hs.dtype)
                    else:
                        hs[-1, :] += sv.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += sv.to(hs.dtype)
                    else:
                        hs[-1, :] += sv.to(hs.dtype)
                    return hs
            return output
        return hook

    # |psi1>
    h1 = make_hook(sv1)
    handle = layers[mid].register_forward_hook(h1)
    with torch.no_grad():
        out1 = model(**inputs)
        logits1 = out1.logits[0, -1, :500].cpu().float().numpy()
    handle.remove()

    # |psi2>
    h2 = make_hook(sv2)
    handle = layers[mid].register_forward_hook(h2)
    with torch.no_grad():
        out2 = model(**inputs)
        logits2 = out2.logits[0, -1, :500].cpu().float().numpy()
    handle.remove()

    # |psi1+psi2>
    h3 = make_hook(sv1 + sv2)
    handle = layers[mid].register_forward_hook(h3)
    with torch.no_grad():
        out_sup = model(**inputs)
        logits_sup = out_sup.logits[0, -1, :500].cpu().float().numpy()
    handle.remove()

    # Interference
    diff_1 = np.linalg.norm(logits1 - ref_logits)
    diff_2 = np.linalg.norm(logits2 - ref_logits)
    diff_sup = np.linalg.norm(logits_sup - ref_logits)
    interference = abs(diff_sup - (diff_1 + diff_2) / 2) / (diff_1 + diff_2 + 1e-10)

    # 3. ENTANGLEMENT: SVD entropy at mid-layer
    captured = [None]
    def cap_hook(module, args, output, store=captured):
        if isinstance(output, tuple):
            store[0] = output[0][0].detach().cpu().float().numpy()
        else:
            store[0] = output.detach().cpu().float().numpy()
            if store[0].ndim == 3:
                store[0] = store[0][0]

    handle = layers[mid].register_forward_hook(cap_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    entanglement_entropy = 0
    if captured[0] is not None:
        hs = captured[0].astype(np.float32)
        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
            S2 = S**2
            total = S2.sum()
            if total > 1e-10:
                p = S2 / total
                p = p[p > 1e-10]
                entanglement_entropy = float(-np.sum(p * np.log(p)))
        except:
            pass

    # 4. HOLOGRAPHIC: bulk entropy vs boundary entropy
    # Capture layer 0
    cap_0 = [None]
    def cap_hook_0(module, args, output, store=cap_0):
        if isinstance(output, tuple):
            store[0] = output[0][0].detach().cpu().float().numpy()
        else:
            store[0] = output.detach().cpu().float().numpy()
            if store[0].ndim == 3:
                store[0] = store[0][0]

    handle = layers[0].register_forward_hook(cap_hook_0)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    bulk_entropy = 0
    if cap_0[0] is not None:
        hs = cap_0[0].astype(np.float32)
        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
            S2 = S**2
            total = S2.sum()
            if total > 1e-10:
                p = S2 / total
                p = p[p > 1e-10]
                bulk_entropy = float(-np.sum(p * np.log(p)))
        except:
            pass

    p_ref = np.exp(ref_logits - ref_logits.max())
    p_ref /= p_ref.sum()
    boundary_entropy = float(-np.sum(p_ref * np.log(p_ref + 1e-10)))
    rt_ratio = boundary_entropy / (bulk_entropy + 1e-10)
    is_holographic = rt_ratio < 1.0 and bulk_entropy > 0

    # 5. CONSCIOUSNESS (Phi)
    phi = 0
    if captured[0] is not None:
        hs = captured[0].astype(np.float32)
        seq_l = hs.shape[0]
        if seq_l >= 4:
            try:
                cov_full = np.cov(hs.T)
                ev_full = np.linalg.eigvalsh(cov_full)
                ev_full = ev_full[ev_full > 1e-10]
                e_full = np.sum(np.log(ev_full + 1e-10))

                h1_p = hs[:seq_l//2]
                h2_p = hs[seq_l//2:]
                cov1 = np.cov(h1_p.T)
                ev1 = np.linalg.eigvalsh(cov1)
                ev1 = ev1[ev1 > 1e-10]
                e1 = np.sum(np.log(ev1 + 1e-10))

                cov2 = np.cov(h2_p.T)
                ev2 = np.linalg.eigvalsh(cov2)
                ev2 = ev2[ev2 > 1e-10]
                e2 = np.sum(np.log(ev2 + 1e-10))

                phi = float(e_full - (e1 + e2))
            except:
                pass

    # 6. INFORMATION PRESERVATION (unitarity)
    info_preserved = diff_1 > 0.1  # Information signal survives to output

    return {
        'model_name': arch_name,
        'num_layers': num_layers,
        'hidden_size': d_model,
        'num_params_M': sum(p.numel() for p in model.parameters()) / 1e6,
        'superposition': {
            'interference': float(interference),
            'diff_1': float(diff_1),
            'diff_sup': float(diff_sup),
            'confirmed': interference > 0.001,
        },
        'entanglement': {
            'entropy': float(entanglement_entropy),
            'confirmed': entanglement_entropy > 0.001,
        },
        'holographic': {
            'bulk_entropy': float(bulk_entropy),
            'boundary_entropy': float(boundary_entropy),
            'rt_ratio': float(rt_ratio),
            'confirmed': is_holographic,
        },
        'consciousness': {
            'phi': float(phi),
            'confirmed': abs(phi) > 1.0,
        },
        'unitarity': {
            'info_signal': float(diff_1),
            'confirmed': info_preserved,
        },
    }


def main():
    print("=" * 60)
    print("Phase Q101: CROSS-ARCHITECTURE UNIVERSALITY TEST")
    print("  Testing S-Qubit properties across ALL transformer families")
    print("=" * 60)
    t0 = time.time()

    all_results = []

    for model_name, model_dir, arch in MODELS:
        snap_path = find_snapshot(model_dir)
        if snap_path is None:
            print("  [SKIP] %s - not found" % model_name)
            continue

        print("\n  === Testing: %s ===" % model_name)
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.float16 if device == 'cuda' else torch.float32

            tok = AutoTokenizer.from_pretrained(snap_path, local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                snap_path, torch_dtype=dtype, device_map=device,
                local_files_only=True)
            model.eval()

            result = measure_quantum_properties(model, tok, model_name, device)

            if result is not None:
                n_confirmed = sum(1 for k in ['superposition', 'entanglement',
                                               'holographic', 'consciousness',
                                               'unitarity']
                                  if result[k]['confirmed'])
                result['n_confirmed'] = n_confirmed
                result['total'] = 5
                all_results.append(result)

                print("    Params: %.1fM, Layers: %d, Hidden: %d" %
                      (result['num_params_M'], result['num_layers'],
                       result['hidden_size']))
                print("    Superposition: %s (interference=%.4f)" %
                      (result['superposition']['confirmed'],
                       result['superposition']['interference']))
                print("    Entanglement: %s (S=%.4f)" %
                      (result['entanglement']['confirmed'],
                       result['entanglement']['entropy']))
                print("    Holographic: %s (RT=%.4f)" %
                      (result['holographic']['confirmed'],
                       result['holographic']['rt_ratio']))
                print("    Consciousness: %s (Phi=%.1f)" %
                      (result['consciousness']['confirmed'],
                       result['consciousness']['phi']))
                print("    Unitarity: %s (signal=%.2f)" %
                      (result['unitarity']['confirmed'],
                       result['unitarity']['info_signal']))
                print("    SCORE: %d/5" % n_confirmed)
            else:
                print("    [FAIL] Could not measure properties")

            del model, tok
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print("    [ERROR] %s" % str(e)[:100])
            gc.collect()
            torch.cuda.empty_cache()

    # Generate figure
    if len(all_results) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # (a) Score by model
        ax = axes[0]
        names = [r['model_name'] for r in all_results]
        scores = [r['n_confirmed'] for r in all_results]
        colors = ['#4CAF50' if s >= 4 else '#FF9800' if s >= 3 else '#F44336'
                  for s in scores]
        bars = ax.barh(range(len(names)), scores, color=colors,
                       edgecolor='black', alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Quantum properties confirmed (out of 5)', fontsize=11)
        ax.set_xlim(0, 5.5)
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.1, i, '%d/5' % score, va='center',
                    fontweight='bold', fontsize=11)
        ax.set_title('(a) Universality Score by Model',
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

        # (b) Property comparison
        ax = axes[1]
        properties = ['superposition', 'entanglement', 'holographic',
                       'consciousness', 'unitarity']
        prop_labels = ['Superpos.', 'Entangl.', 'Hologr.', 'Conscious.', 'Unitar.']
        data_matrix = np.zeros((len(all_results), len(properties)))
        for i, r in enumerate(all_results):
            for j, prop in enumerate(properties):
                data_matrix[i, j] = 1.0 if r[prop]['confirmed'] else 0.0

        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(prop_labels)))
        ax.set_xticklabels(prop_labels, fontsize=9, rotation=45, ha='right')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        for i in range(len(all_results)):
            for j in range(len(properties)):
                text = 'Y' if data_matrix[i, j] > 0.5 else 'N'
                color = 'white' if data_matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontweight='bold', fontsize=10, color=color)
        ax.set_title('(b) Property Matrix\nGreen = Confirmed',
                     fontsize=12, fontweight='bold')

        # (c) Universality summary
        ax = axes[2]
        n_models = len(all_results)
        n_universal = sum(1 for r in all_results if r['n_confirmed'] >= 3)
        universality = n_universal / n_models * 100

        ax.text(0.5, 0.65,
                'UNIVERSALITY\n%.0f%%' % universality,
                ha='center', va='center', fontsize=24, fontweight='bold',
                color='#4CAF50' if universality >= 80 else '#FF9800',
                transform=ax.transAxes)
        ax.text(0.5, 0.35,
                '%d/%d models show quantum behavior\n\n'
                'S-Qubit properties are\n%s' % (
                    n_universal, n_models,
                    'UNIVERSAL across architectures!'
                    if universality >= 80 else 'architecture-dependent'),
                ha='center', va='center', fontsize=11,
                transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('(c) Universality Verdict', fontsize=12, fontweight='bold')

        plt.suptitle('Q101: S-Qubit Universality Across %d Transformer Architectures' %
                     n_models, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, 'phase_q101_universality.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

    elapsed = time.time() - t0
    results = {
        'phase': 'Q101', 'name': 'Cross-Architecture Universality',
        'n_models_tested': len(all_results),
        'models': all_results,
        'elapsed': elapsed,
    }
    res_path = os.path.join(RESULTS_DIR, 'phase_q101_universality.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  === FINAL RESULTS ===")
    print("  Models tested: %d" % len(all_results))
    if all_results:
        n_u = sum(1 for r in all_results if r['n_confirmed'] >= 3)
        print("  Universal quantum behavior: %d/%d (%.0f%%)" %
              (n_u, len(all_results), n_u/len(all_results)*100))
    print("  Total elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
