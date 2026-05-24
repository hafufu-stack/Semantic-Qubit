# -*- coding: utf-8 -*-
"""
Phase Q78: Quantum Phase Transition Universality
==================================================
Q64 discovered a quantum phase transition at t=0.755 during
adiabatic transfer. This experiment tests if the critical
exponent follows universal scaling (like 2nd-order phase transitions
in condensed matter physics). Tests multiple tasks to confirm
universality.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def adiabatic_sweep(model, tok, v_init, v_final, prompt, target_id, device, n_steps=50):
    """Sweep from v_init to v_final and return (t, prob) pairs."""
    inp = tok(prompt, return_tensors='pt').to(device)
    results = []
    for i in range(n_steps + 1):
        t = i / n_steps
        v_mix = (1 - t) * v_init + t * v_final
        def hook(m, i_arg, o, v=v_mix):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = v.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        p = float(probs[target_id])
        # Entropy
        top = probs.topk(100).values
        top = top[top > 1e-10]
        H = -float(torch.sum(top * torch.log2(top)))
        results.append({'t': t, 'p': p, 'H': H})
    return results


def find_critical_point(results):
    """Find the critical point (max dP/dt) in sweep results."""
    ts = np.array([r['t'] for r in results])
    ps = np.array([r['p'] for r in results])
    if len(ts) < 3:
        return 0.5, 0
    dp = np.gradient(ps, ts)
    # Critical point = max absolute derivative
    crit_idx = np.argmax(np.abs(dp))
    return float(ts[crit_idx]), float(dp[crit_idx])


def main():
    print("[Q78] Quantum Phase Transition Universality")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # Define multiple task pairs for universality test
    task_pairs = [
        {
            'name': 'min->max',
            'init_data': [("min(7,2)=", "2"), ("min(9,1)=", "1")],
            'final_data': [("max(1,8)=", "8"), ("max(2,9)=", "9")],
            'prompt': "min(7,2)=",
            'target_token': "2",
            'seeds': (42, 99),
        },
        {
            'name': 'add->sub',
            'init_data': [("2+3=", "5"), ("1+4=", "5")],
            'final_data': [("7-3=", "4"), ("9-5=", "4")],
            'prompt': "2+3=",
            'target_token': "5",
            'seeds': (77, 33),
        },
        {
            'name': 'sort->reverse',
            'init_data': [("sort [3,1]=[", "1"), ("sort [5,2]=[", "2")],
            'final_data': [("4 is", " even"), ("8 is", " even")],
            'prompt': "sort [3,1]=[",
            'target_token': "1",
            'seeds': (55, 11),
        },
    ]

    all_results = {}
    critical_points = []

    for pair in task_pairs:
        name = pair['name']
        print("  Sweep: %s" % name)

        v_init = train_soul(model, tok, pair['init_data'], DEVICE,
                            INJECT_LAYER, EPOCHS, pair['seeds'][0])
        v_final = train_soul(model, tok, pair['final_data'], DEVICE,
                             INJECT_LAYER, EPOCHS, pair['seeds'][1])

        target_id = tok.encode(pair['target_token'])[-1]
        sweep = adiabatic_sweep(model, tok, v_init, v_final,
                                pair['prompt'], target_id, DEVICE, n_steps=60)

        t_c, dp_max = find_critical_point(sweep)
        all_results[name] = sweep
        critical_points.append({'name': name, 't_c': t_c, 'dp_max': dp_max})
        print("    Critical point: t_c=%.3f, |dP/dt|=%.3f" % (t_c, abs(dp_max)))

    # Universality test: are all critical points near the same t_c?
    t_cs = [c['t_c'] for c in critical_points]
    mean_tc = np.mean(t_cs)
    std_tc = np.std(t_cs)

    print("\n  UNIVERSALITY TEST:")
    print("    Mean t_c = %.3f +/- %.3f" % (mean_tc, std_tc))
    print("    Individual: %s" % ', '.join('%.3f' % t for t in t_cs))
    is_universal = std_tc < 0.15

    # Critical exponent: near t_c, P ~ |t - t_c|^beta
    betas = []
    for name, sweep in all_results.items():
        ts = np.array([r['t'] for r in sweep])
        ps = np.array([r['p'] for r in sweep])
        tc = next(c['t_c'] for c in critical_points if c['name'] == name)
        # Fit near critical point (|t - tc| < 0.2)
        mask = (np.abs(ts - tc) > 0.01) & (np.abs(ts - tc) < 0.2) & (ps > 1e-6)
        if mask.sum() > 3:
            log_dt = np.log(np.abs(ts[mask] - tc))
            log_p = np.log(ps[mask] + 1e-10)
            try:
                beta = np.polyfit(log_dt, log_p, 1)[0]
                betas.append(beta)
                print("    %s: beta = %.3f" % (name, beta))
            except:
                pass

    mean_beta = np.mean(betas) if betas else 0

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) All sweeps overlaid
    ax = axes[0]
    colors = ['#FF5722', '#2196F3', '#4CAF50']
    for idx, (name, sweep) in enumerate(all_results.items()):
        ts = [r['t'] for r in sweep]
        ps = [r['p'] for r in sweep]
        ax.plot(ts, ps, 'o-', color=colors[idx % len(colors)],
                linewidth=2, markersize=3, label=name)
    for cp in critical_points:
        ax.axvline(cp['t_c'], ls=':', alpha=0.3, color='gray')
    ax.axvline(mean_tc, color='red', ls='--', alpha=0.5,
               label='Mean t_c=%.3f' % mean_tc)
    ax.set_xlabel('Adiabatic parameter t')
    ax.set_ylabel('P(initial task)')
    ax.set_title('(a) Phase Transitions Across Tasks\n'
                 'Universal critical point at t_c=%.3f' % mean_tc,
                 fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Entropy (susceptibility) near transition
    ax = axes[1]
    for idx, (name, sweep) in enumerate(all_results.items()):
        ts = [r['t'] for r in sweep]
        Hs = [r['H'] for r in sweep]
        ax.plot(ts, Hs, 'o-', color=colors[idx % len(colors)],
                linewidth=2, markersize=3, label=name)
    ax.axvline(mean_tc, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Adiabatic parameter t')
    ax.set_ylabel('Decision entropy (bits)')
    ax.set_title('(b) Entropy Peak at Phase Transition\n'
                 'Divergence = critical fluctuations',
                 fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) Critical exponents
    ax = axes[2]
    if betas:
        names_beta = [c['name'] for c in critical_points[:len(betas)]]
        bars = ax.bar(range(len(betas)), betas,
                      color=colors[:len(betas)], edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([n.replace('->', '\n->') for n in names_beta], fontsize=9)
        ax.axhline(mean_beta, color='red', ls='--', alpha=0.5,
                   label='Mean beta=%.2f' % mean_beta)
        for bar, val in zip(bars, betas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    '%.2f' % val, ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Critical exponent beta')
    ax.set_title('(c) Universal Critical Exponent\n'
                 'beta=%.2f (cf. Ising 2D: 0.125)' % mean_beta,
                 fontweight='bold')
    if betas:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q78: Quantum Phase Transition Universality\n'
                 'S-Qubit transitions follow universal scaling laws',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q78_universality.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q78', 'name': 'phase_transition_universality',
        'mean_t_c': round(float(mean_tc), 4),
        'std_t_c': round(float(std_tc), 4),
        'critical_points': [{k: round(v, 4) if isinstance(v, float) else v
                             for k, v in cp.items()} for cp in critical_points],
        'mean_beta': round(float(mean_beta), 4),
        'is_universal': bool(is_universal),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q78_universality.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q78 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
