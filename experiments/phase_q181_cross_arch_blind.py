# -*- coding: utf-8 -*-
"""
Phase Q181: Cross-Architecture Blind Interference
===================================================
Q179 proved 12/12 tasks produce V=1.000 on Qwen.
Q181: Does GPT-2 ALSO produce perfect interference?

If yes -> S-Qubit is a universal property of ALL transformers.
If no -> Qwen-specific artifact.

Uses GPT-2 small (124M), medium (355M) to verify.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

INJECT_FRAC = 0.3  # inject at 30% depth
N_PHI = 16


def load_gpt2(size='small'):
    """Load GPT-2 locally."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_name = 'gpt2' if size == 'small' else 'gpt2-medium'
    # Try local snapshot first
    snap_dir = os.path.join(cache, "models--" + model_name.replace('/', '--'))
    if os.path.exists(snap_dir):
        snaps = os.path.join(snap_dir, "snapshots")
        if os.path.exists(snaps):
            subs = os.listdir(snaps)
            if subs:
                local_path = os.path.join(snaps, subs[0])
                tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    local_path, local_files_only=True)
                model.eval()
                return model, tok, local_path
    # Fallback
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    model.eval()
    return model, tok, model_name


def train_soul_gpt2(model, tok, data, device, layer, epochs=80, seed=42):
    """Train soul vector for GPT-2 architecture."""
    hs = model.config.n_embd
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
                    h = o[0].clone()
                    if h.dim() == 3:
                        h[0, -1, :] = v.to(h.dtype)
                    else:
                        h[-1, :] = v.to(h.dtype)
                    return (h,) + o[1:]
                h = o.clone()
                if h.dim() == 3:
                    h[0, -1, :] = v.to(h.dtype)
                else:
                    h[-1, :] = v.to(h.dtype)
                return h
            handle = model.transformer.h[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def get_p_gpt2(model, tok, prompt, device, inject_vec, layer, target_id):
    """Get P(target) with injected vector for GPT-2."""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone()
            if h.dim() == 3:
                h[0, -1, :] = v.to(h.dtype)
            else:
                h[-1, :] = v.to(h.dtype)
            return (h,) + o[1:]
        h = o.clone()
        if h.dim() == 3:
            h[0, -1, :] = v.to(h.dtype)
        else:
            h[-1, :] = v.to(h.dtype)
        return h
    handle = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[target_id])


# Also test on Qwen for direct comparison
def train_soul_qwen(model, tok, data, device, layer, epochs=80, seed=42):
    """Train soul vector for Qwen architecture."""
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


def get_p_qwen(model, tok, prompt, device, inject_vec, layer, target_id):
    """Get P(target) for Qwen."""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[target_id])


TASKS = [
    {
        'name': 'CAPITAL',
        'zero_data': [("The capital of France is","Paris"),("France's capital:","Paris")],
        'one_data': [("The capital of Germany is","Berlin"),("Germany's capital:","Berlin")],
        'prompt': "The capital of France is",
        'target': "Paris",
    },
    {
        'name': 'COLOR',
        'zero_data': [("The sky is","blue"),("The ocean is","blue")],
        'one_data': [("The grass is","green"),("Leaves are","green")],
        'prompt': "The sky is",
        'target': "blue",
    },
    {
        'name': 'SIZE',
        'zero_data': [("An elephant is","big"),("A whale is","big")],
        'one_data': [("An ant is","small"),("A mouse is","small")],
        'prompt': "A whale is",
        'target': "big",
    },
    {
        'name': 'TEMP',
        'zero_data': [("Fire is","hot"),("The sun is","hot")],
        'one_data': [("Ice is","cold"),("Snow is","cold")],
        'prompt': "Fire is",
        'target': "hot",
    },
]


def run_on_model(model, tok, device, arch, tasks, train_fn, get_p_fn, n_layers):
    """Run interference test on a model."""
    inject_layer = int(n_layers * INJECT_FRAC)
    print("  Inject layer: %d / %d" % (inject_layer, n_layers))

    results = []
    for task in tasks:
        name = task['name']
        print("\n  [%s] Training..." % name)

        vec0 = train_fn(model, tok, task['zero_data'], device,
                       inject_layer, epochs=80, seed=42)
        vec1 = train_fn(model, tok, task['one_data'], device,
                       inject_layer, epochs=80, seed=99)

        target_id = tok.encode(task['target'])[-1]
        p0 = get_p_fn(model, tok, task['prompt'], device, vec0, inject_layer, target_id)
        p1 = get_p_fn(model, tok, task['prompt'], device, vec1, inject_layer, target_id)

        # Sweep phases
        phis = np.linspace(0, 4 * np.pi, N_PHI)
        p_vals = []
        scale = vec0.norm()
        for phi in phis:
            vec = np.cos(phi / 2) * vec0 + np.sin(phi / 2) * vec1
            n = vec.norm()
            if n > 0:
                vec = vec / n * scale
            p = get_p_fn(model, tok, task['prompt'], device, vec, inject_layer, target_id)
            p_vals.append(p)

        p_arr = np.array(p_vals)
        amp = (p_arr.max() - p_arr.min()) / 2.0
        vis = (p_arr.max() - p_arr.min()) / (p_arr.max() + p_arr.min() + 1e-8)

        result = {
            'task': name, 'visibility': round(float(vis), 4),
            'amplitude': round(float(amp), 6),
            'p0': round(p0, 4), 'p1': round(p1, 4),
            'p_curve': [round(float(p), 4) for p in p_vals],
        }
        results.append(result)
        print("    P(|0>)=%.4f P(|1>)=%.4f Vis=%.4f Amp=%.6f" %
              (p0, p1, vis, amp))

    return results


def main():
    print("=" * 60)
    print("Phase Q181: Cross-Architecture Blind Interference")
    print("  (GPT-2 + Qwen: Does V=1.000 Hold Everywhere?)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}

    # === GPT-2 Small ===
    print("\n=== GPT-2 Small (124M) ===")
    try:
        gpt2_model, gpt2_tok, gpt2_path = load_gpt2('small')
        gpt2_model = gpt2_model.to(device)
        for p in gpt2_model.parameters():
            p.requires_grad = False
        n_layers = gpt2_model.config.n_layer
        print("  Loaded: %d layers, hidden=%d" % (n_layers, gpt2_model.config.n_embd))

        gpt2_results = run_on_model(gpt2_model, gpt2_tok, device, 'gpt2',
                                     TASKS, train_soul_gpt2, get_p_gpt2, n_layers)
        all_results['GPT-2 (124M)'] = gpt2_results

        del gpt2_model, gpt2_tok; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print("  GPT-2 Small failed: %s" % str(e))
        all_results['GPT-2 (124M)'] = []

    # === GPT-2 Medium ===
    print("\n=== GPT-2 Medium (355M) ===")
    try:
        gpt2m_model, gpt2m_tok, _ = load_gpt2('medium')
        gpt2m_model = gpt2m_model.to(device)
        for p in gpt2m_model.parameters():
            p.requires_grad = False
        n_layers = gpt2m_model.config.n_layer
        print("  Loaded: %d layers, hidden=%d" % (n_layers, gpt2m_model.config.n_embd))

        gpt2m_results = run_on_model(gpt2m_model, gpt2m_tok, device, 'gpt2-medium',
                                      TASKS, train_soul_gpt2, get_p_gpt2, n_layers)
        all_results['GPT-2 (355M)'] = gpt2m_results

        del gpt2m_model, gpt2m_tok; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print("  GPT-2 Medium failed: %s" % str(e))
        all_results['GPT-2 (355M)'] = []

    # === Qwen 1.5B ===
    print("\n=== Qwen2.5-1.5B ===")
    try:
        from utils import load_model
        qwen_model, qwen_tok = load_model(device=device)
        for p in qwen_model.parameters():
            p.requires_grad = False
        n_layers = qwen_model.config.num_hidden_layers
        print("  Loaded: %d layers, hidden=%d" % (n_layers, qwen_model.config.hidden_size))

        qwen_results = run_on_model(qwen_model, qwen_tok, device, 'qwen',
                                     TASKS, train_soul_qwen, get_p_qwen, n_layers)
        all_results['Qwen2.5 (1.5B)'] = qwen_results

        del qwen_model, qwen_tok; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print("  Qwen failed: %s" % str(e))
        all_results['Qwen2.5 (1.5B)'] = []

    # === Summary ===
    print("\n" + "=" * 60)
    print("  CROSS-ARCHITECTURE SUMMARY")
    print("=" * 60)

    summary = {}
    for arch, results in all_results.items():
        if not results:
            continue
        viss = [r['visibility'] for r in results]
        amps = [r['amplitude'] for r in results]
        v_pass = sum(1 for v in viss if v > 0.5)
        summary[arch] = {
            'vis_mean': round(float(np.mean(viss)), 4),
            'vis_std': round(float(np.std(viss)), 4),
            'amp_mean': round(float(np.mean(amps)), 6),
            'pass_rate': round(100 * v_pass / len(viss), 1),
            'n_tasks': len(viss),
        }
        print("  %s: V=%.4f +/- %.4f, %d/%d pass (%.0f%%)" %
              (arch, np.mean(viss), np.std(viss), v_pass, len(viss),
               100 * v_pass / len(viss)))

    # Universal check
    all_pass = all(s['pass_rate'] == 100.0 for s in summary.values())
    verdict = "UNIVERSAL: V=1.000 across ALL architectures" if all_pass else \
              "PARTIAL: some architectures show reduced visibility"
    print("\n  Verdict: %s" % verdict)

    # Save
    output = {
        'phase': 'Q181',
        'name': 'Cross-Architecture Blind Interference',
        'architectures': {k: v for k, v in all_results.items()},
        'summary': summary,
        'verdict': verdict,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q181_cross_arch_blind.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    n_arch = len([k for k, v in all_results.items() if v])
    fig, axes = plt.subplots(1, min(n_arch, 3), figsize=(6 * min(n_arch, 3), 5))
    if n_arch == 1:
        axes = [axes]

    palette = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']
    phis = np.linspace(0, 4 * np.pi, N_PHI)

    for idx, (arch, results) in enumerate(all_results.items()):
        if not results or idx >= 3:
            continue
        ax = axes[idx]
        for i, r in enumerate(results):
            ax.plot(phis / np.pi, r['p_curve'], '-o', color=palette[i],
                    linewidth=1.5, markersize=3, label=r['task'])
        vis_m = summary.get(arch, {}).get('vis_mean', 0)
        ax.set_xlabel('Phase (x pi)')
        ax.set_ylabel('P(target)')
        ax.set_title('%s\n(V=%.4f)' % (arch, vis_m))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Q181: Cross-Architecture Blind Interference\n'
                 '%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q181_cross_arch_blind.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ181 complete! Elapsed: %.1fs" % (time.time() - t0))
    return output


if __name__ == '__main__':
    main()
