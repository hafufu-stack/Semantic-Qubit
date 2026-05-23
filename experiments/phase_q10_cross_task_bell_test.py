# -*- coding: utf-8 -*-
"""
Phase Q10: Cross-Task Bell Test (Universal Quantum Coherence?)

Q2/Q8 used min/max task vectors for interference.
Q10 asks: is the interference pattern TASK-SPECIFIC or UNIVERSAL?

For each of 5 task domains:
  - Define domain-specific |ZERO> and |ONE> basis states (trained soul vectors)
  - Sweep phi: vec(phi) = cos(phi/2)*ZERO + sin(phi/2)*ONE at L8
  - Measure P(correct_for_task) as function of phi
  - Compute interference amplitude and fringe visibility

Tasks:
  - MATH:    |SMALL>=3, |LARGE>=7  (min(3,7)=?)
  - CAPITAL: |FRANCE>=Paris, |GERMANY>=Berlin
  - COLOR:   |SKY>=blue, |GRASS>=green  (The ___ is ?)
  - CODE:    |RETURN>=return, |PASS>=pass  (def f(): ?)
  - LOGIC:   |TRUE>=True, |FALSE>=False

If all tasks show similar interference amplitude at L8
-> L8 is a UNIVERSAL quantum coherence layer (domain-invariant)
If amplitudes vary wildly -> task-specific phenomenon
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
N_PHI = 37
INJECT_LAYER = 8


def train_soul(model, tok, data, device, layer=8, epochs=100, seed=42):
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


def get_p_token(model, tok, prompt, device, inject_vec, inject_layer, target_tok_id):
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=inject_vec):
        if isinstance(o, tuple):
            h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
        h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
    handle = model.model.layers[inject_layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(probs[target_tok_id])


def run_bell_test(model, tok, prompt, device, vec0, vec1, target_tok_id,
                   inject_layer=INJECT_LAYER, n_phi=N_PHI):
    phis = np.linspace(0, 4 * np.pi, n_phi)
    p_vals = []
    scale = vec0.norm()
    for phi in phis:
        vec = np.cos(phi / 2) * vec0 + np.sin(phi / 2) * vec1
        n = vec.norm()
        if n > 0:
            vec = vec / n * scale
        p = get_p_token(model, tok, prompt, device, vec, inject_layer, target_tok_id)
        p_vals.append(p)
    p_arr = np.array(p_vals)
    amp = (p_arr.max() - p_arr.min()) / 2.0
    # Fringe visibility = (max-min)/(max+min)
    visibility = (p_arr.max() - p_arr.min()) / (p_arr.max() + p_arr.min() + 1e-8)
    # Try to fit sinusoid to get frequency
    return phis, p_arr, float(amp), float(visibility)


def main():
    print("[Q10] Cross-Task Bell Test: Is L8 Coherence Universal?")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print("  Model loaded. INJECT_LAYER=%d" % INJECT_LAYER)

    # Define task domains
    # Each: (name, |0> data, |1> data, test_prompt, target_0_token, description)
    TASKS = [
        {
            'name': 'MATH_min',
            'zero_name': '|SMALL>=3',
            'one_name':  '|LARGE>=7',
            'zero_data': [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                          ("min(4,6)=","4"),("min(9,3)=","3")],
            'one_data':  [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                          ("min(4,6)=","6"),("min(9,3)=","9")],
            'prompt':    "min(7,2)=",
            'target':    "2",  # correct answer (SMALL/min)
        },
        {
            'name': 'CAPITAL',
            'zero_name': '|PARIS>',
            'one_name':  '|BERLIN>',
            'zero_data': [("The capital of France is","Paris"),
                          ("France's capital city is","Paris"),
                          ("Paris is the capital of","France"),
                          ("The capital city: France ->","Paris"),
                          ("Capital of France:","Paris")],
            'one_data':  [("The capital of Germany is","Berlin"),
                          ("Germany's capital city is","Berlin"),
                          ("Berlin is the capital of","Germany"),
                          ("The capital city: Germany ->","Berlin"),
                          ("Capital of Germany:","Berlin")],
            'prompt':    "The capital of France is",
            'target':    "Paris",
        },
        {
            'name': 'COLOR',
            'zero_name': '|BLUE>',
            'one_name':  '|GREEN>',
            'zero_data': [("The sky is","blue"),("The ocean is","blue"),
                          ("Clear water is","blue"),("The sky color is","blue"),
                          ("A clear sky is","blue")],
            'one_data':  [("The grass is","green"),("Leaves are","green"),
                          ("Plants are","green"),("The forest is","green"),
                          ("Grass color is","green")],
            'prompt':    "The sky is",
            'target':    "blue",
        },
        {
            'name': 'CODE_keyword',
            'zero_name': '|RETURN>',
            'one_name':  '|PASS>',
            'zero_data': [("def add(x,y):","return"),("def square(x):","return"),
                          ("def get_value():","return"),("def compute():","return"),
                          ("def result():","return")],
            'one_data':  [("def empty():","pass"),("def placeholder():","pass"),
                          ("def noop():","pass"),("def skip():","pass"),
                          ("def todo():","pass")],
            'prompt':    "def add(x,y):",
            'target':    "return",
        },
        {
            'name': 'NUMBER_parity',
            'zero_name': '|EVEN>',
            'one_name':  '|ODD>',
            'zero_data': [("2 is","even"),("4 is","even"),("6 is","even"),
                          ("8 is","even"),("10 is","even")],
            'one_data':  [("1 is","odd"),("3 is","odd"),("5 is","odd"),
                          ("7 is","odd"),("9 is","odd")],
            'prompt':    "2 is",
            'target':    "even",
        },
    ]

    results = {}
    all_phi_curves = {}

    for task in TASKS:
        name = task['name']
        print("\n  === Task: %s ===" % name)
        print("    Training |0>=%s..." % task['zero_name'])
        vec0 = train_soul(model, tok, task['zero_data'], DEVICE,
                           layer=INJECT_LAYER, seed=42, epochs=100)
        print("    Training |1>=%s..." % task['one_name'])
        vec1 = train_soul(model, tok, task['one_data'], DEVICE,
                           layer=INJECT_LAYER, seed=99, epochs=100)

        # Verify basis quality
        target_id = tok.encode(task['target'])[-1]
        p0 = get_p_token(model, tok, task['prompt'], DEVICE, vec0, INJECT_LAYER, target_id)
        p1 = get_p_token(model, tok, task['prompt'], DEVICE, vec1, INJECT_LAYER, target_id)
        cos_sim = float(torch.nn.functional.cosine_similarity(
            vec0.unsqueeze(0), vec1.unsqueeze(0)).item())
        print("    |0> P(target)=%.4f  |1> P(target)=%.4f  cos(0,1)=%.4f" % (p0, p1, cos_sim))

        # Bell test
        phis, p_arr, amp, vis = run_bell_test(
            model, tok, task['prompt'], DEVICE, vec0, vec1, target_id)
        all_phi_curves[name] = p_arr.tolist()
        results[name] = {
            'amplitude': round(amp, 6),
            'visibility': round(vis, 6),
            'p0_baseline': round(p0, 4),
            'p1_baseline': round(p1, 4),
            'cosine_01': round(cos_sim, 4),
            'p_max': round(float(p_arr.max()), 4),
            'p_min': round(float(p_arr.min()), 4),
        }
        print("    Bell amp=%.4f  visibility=%.4f  P:[%.4f, %.4f]" % (
            amp, vis, p_arr.min(), p_arr.max()))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    task_names = list(results.keys())
    palette = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800']
    phis_plot = np.linspace(0, 4 * np.pi, N_PHI)

    # Panel 1: Interference fringes for all tasks
    ax = axes[0]
    for i, (name, p_arr) in enumerate(all_phi_curves.items()):
        ax.plot(phis_plot / np.pi, p_arr, color=palette[i], lw=2,
                label='%s (amp=%.3f)' % (name, results[name]['amplitude']))
    ax.set_xlabel('Phase phi (x pi)', fontsize=11)
    ax.set_ylabel('P(target token)', fontsize=11)
    ax.set_title('Cross-Task Interference Fringes\nat Layer 8', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Amplitude comparison bar chart
    ax = axes[1]
    amps = [results[n]['amplitude'] for n in task_names]
    viss = [results[n]['visibility'] for n in task_names]
    x = np.arange(len(task_names))
    ax.bar(x, amps, color=palette[:len(task_names)], edgecolor='black', alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(x, viss, 'k^--', lw=2, ms=8, label='Visibility')
    ax.axhline(np.mean(amps), color='red', linestyle='--', lw=1.5,
               label='Mean amp=%.3f' % np.mean(amps))
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax2.set_ylabel('Fringe Visibility', fontsize=11)
    ax.set_title('Bell Amplitude by Task\n(Universal = all similar)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel 3: Summary table
    ax = axes[2]
    ax.axis('off')
    rows = [['Task', 'Amplitude', 'Visibility', 'cos(|0>,|1>)']]
    for name in task_names:
        r = results[name]
        rows.append([name, '%.4f' % r['amplitude'], '%.4f' % r['visibility'],
                     '%.4f' % r['cosine_01']])
    rows.append(['MEAN', '%.4f' % np.mean(amps), '%.4f' % np.mean(viss), '—'])
    rows.append(['STD',  '%.4f' % np.std(amps),  '%.4f' % np.std(viss),  '—'])
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc='center', loc='center',
                     bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r == len(rows) - 2:
            cell.set_facecolor('#E3F2FD')
        elif r == len(rows) - 1:
            cell.set_facecolor('#FFF9C4')
    ax.set_title('Cross-Task Summary\n'
                 '"Amplitude similar -> L8 universal quantum seat"',
                 fontweight='bold', y=0.97)

    plt.suptitle(
        'Phase Q10: Cross-Task Bell Test\n'
        '"Is Layer 8 a Universal Quantum Coherence Layer?"',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q10_cross_task_bell_test.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    mean_amp = np.mean(amps)
    std_amp = np.std(amps)
    universal = std_amp / mean_amp < 0.3  # CV < 30% = universal

    print("\n  VERDICT: mean_amp=%.4f  std=%.4f  CV=%.1f%%  universal=%s" % (
        mean_amp, std_amp, 100*std_amp/mean_amp, universal))

    output = {
        'phase': 'Q10', 'name': 'cross_task_bell_test',
        'inject_layer': INJECT_LAYER,
        'n_phi': N_PHI,
        'results': results,
        'summary': {
            'mean_amplitude': round(mean_amp, 6),
            'std_amplitude': round(std_amp, 6),
            'cv_amplitude': round(std_amp / mean_amp, 4),
            'universal_coherence': bool(universal),
        },
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q10_cross_task_bell_test.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q10 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
