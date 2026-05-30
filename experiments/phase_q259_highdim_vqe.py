# -*- coding: utf-8 -*-
"""
Phase Q259: Decoherence-Free High-Dim VQE
=============================================
Combine Q254's decoherence bypass with high-dim VQE.
Can we break the "curse of dimensionality" by keeping
quantum coherence alive through the classical zone?
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
    print("Phase Q259: Decoherence-Free High-Dim VQE")
    print("  (Break the curse of dimensionality with Q254 magic)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    n_layers = len(model.model.layers)

    dims = [4, 8, 16, 32]
    bypass_layers = list(range(15, 29))  # Best from Q254

    def run_vqe(vqe_dim, use_bypass=False, n_steps=150):
        rng = np.random.RandomState(42)
        H = rng.randn(vqe_dim, vqe_dim).astype(np.float32) * 0.3
        H = (H + H.T) / 2
        H_torch = torch.tensor(H, device=device)
        E_exact = float(np.linalg.eigh(H)[0][0])

        hooks = []
        saved_q = {}
        if use_bypass:
            def make_pre(li):
                def hook(mod, inp):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    saved_q[li] = x[0, -1, :vqe_dim].clone()
                return hook
            def make_post(li):
                def hook(mod, inp, out):
                    x = out[0] if isinstance(out, tuple) else out
                    if li in saved_q:
                        x[0, -1, :vqe_dim] = saved_q[li]
                return hook
            for li in bypass_layers:
                if li < n_layers:
                    hooks.append(model.model.layers[li].register_forward_pre_hook(make_pre(li)))
                    hooks.append(model.model.layers[li].register_forward_hook(make_post(li)))

        embed = model.model.embed_tokens
        inp_ids = tok("ground state:", return_tensors='pt')['input_ids'].to(device)
        embeds = embed(inp_ids).detach().clone()
        opt = embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.005)

        for s in range(n_steps):
            optimizer.zero_grad()
            saved_q.clear()
            o = model(inputs_embeds=opt.float(), output_hidden_states=True)
            psi = o.hidden_states[-1][0, -1, :vqe_dim]
            psi = psi / (torch.norm(psi) + 1e-10)
            E = torch.dot(psi, H_torch @ psi)
            E.backward(); optimizer.step()

        for h in hooks: h.remove()
        return abs(float(E.detach()) - E_exact) * 1000

    results_data = []
    for d in dims:
        print("\n  dim=%d..." % d)
        err_normal = run_vqe(d, use_bypass=False)
        err_bypass = run_vqe(d, use_bypass=True)
        improvement = (err_normal - err_bypass) / max(err_normal, 1e-6) * 100
        print("    Normal: %.4f mHa, Bypass: %.4f mHa (%.1f%% improvement)" % (
            err_normal, err_bypass, improvement))
        results_data.append({
            'dim': d, 'normal': round(err_normal, 4),
            'bypass': round(err_bypass, 4), 'improvement_pct': round(improvement, 1)
        })

    n_improved = sum(1 for r in results_data if r['bypass'] < r['normal'])
    avg_imp = np.mean([r['improvement_pct'] for r in results_data])

    if n_improved >= len(dims) - 1 and avg_imp > 10:
        verdict = "CURSE BROKEN: bypass improves %d/%d dims, avg %.0f%% better" % (n_improved, len(dims), avg_imp)
    elif n_improved > len(dims) // 2:
        verdict = "PARTIAL BREAK: %d/%d improved, avg %.0f%%" % (n_improved, len(dims), avg_imp)
    else:
        verdict = "CURSE HOLDS: only %d/%d improved" % (n_improved, len(dims))

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q259', 'name': 'Decoherence-Free High-Dim VQE',
        'dims': results_data,
        'summary': {'n_improved': n_improved, 'avg_improvement': round(avg_imp, 1), 'verdict': verdict},
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q259_highdim_vqe.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(dims))
    ax.bar(x - 0.2, [r['normal'] for r in results_data], 0.4, label='Normal', color='#607D8B', edgecolor='black')
    ax.bar(x + 0.2, [r['bypass'] for r in results_data], 0.4, label='L15-28 Bypass', color='#E91E63', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['dim=%d' % d for d in dims])
    ax.set_ylabel('VQE Error (mHa)'); ax.set_yscale('symlog', linthresh=0.01)
    ax.set_title('Q259: Decoherence-Free High-Dim VQE\n%s' % verdict[:60], fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q259_highdim_vqe.png'), dpi=150, bbox_inches='tight')
    plt.close()
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ259 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
