# -*- coding: utf-8 -*-
"""
Phase Q20: Model Size Scaling -- Is S-Qubit Universal?

Key question: are S-Qubit properties (interference, CHSH violation)
a fundamental architectural feature or an artifact of Qwen2.5-1.5B?

Test with Qwen2.5-0.5B (hidden_size=896, 24 layers):
  1. Single-qubit interference (Q2 replication)
  2. Cross-task universality (Q10 replication, 3 tasks)
  3. 2-Qubit CHSH at optimal layers (Q15 replication)

If results hold -> S-Qubit is a universal transformer property
If results degrade -> dimensionality matters (useful for NQPU design)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import load path for 0.5B
_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
_SNAP_0B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                          "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")
EPOCHS = 100


def load_05b(device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tok = AutoTokenizer.from_pretrained(_SNAP_0B5, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        _SNAP_0B5, torch_dtype=dtype, device_map=device, local_files_only=True)
    model.eval()
    return model, tok


def train_soul(model, tok, data, device, layer, pos=-1, epochs=EPOCHS, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for prompt, target_token in data:
            target_id = tok.encode(target_token)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            seq_len = inp['input_ids'].shape[1]
            actual_pos = pos if pos >= 0 else seq_len + pos
            def hook(m, i, o, v=vec, p=actual_pos):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, p, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def inject_forward(model, tok, prompt, device, vec, layer, pos=-1):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    actual_pos = pos if pos >= 0 else seq_len + pos
    def hook(m, i, o, v=vec, p=actual_pos):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def two_qubit_forward(model, tok, prompt, device, v1, v2, l1, p1, l2, p2):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    a1 = p1 if p1 >= 0 else seq_len + p1
    a2 = p2 if p2 >= 0 else seq_len + p2
    def hook1(m, i, o, v=v1, p=a1):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    def hook2(m, i, o, v=v2, p=a2):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, p, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    h1 = model.model.layers[l1].register_forward_hook(hook1)
    h2 = model.model.layers[l2].register_forward_hook(hook2)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def main():
    print("[Q20] Model Size Scaling: Qwen2.5-0.5B (896d, 24L)")
    start = time.time()

    if not os.path.exists(_SNAP_0B5):
        print("  ERROR: Qwen2.5-0.5B not found at %s" % _SNAP_0B5)
        print("  Skipping Q20.")
        return

    model, tok = load_05b(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.model.layers)
    hs = model.config.hidden_size
    print("  Model: %d layers, hidden_size=%d" % (n_layers, hs))

    # === Test 1: Single qubit interference ===
    # Use L6 as injection layer (scaled from L8 in 28-layer model: 8/28*24 ~ 6.9)
    sq_layer = 6
    prompt = "min(7,2)="
    sq_tok = tok.encode("2")[-1]
    sq_tok_1 = tok.encode("7")[-1]

    sq_0_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                 ("min(4,6)=","4"),("min(9,3)=","3")]
    sq_1_data = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                 ("min(4,6)=","6"),("min(9,3)=","9")]

    print("  [Test 1] Single-qubit interference at L%d..." % sq_layer)
    vec_0 = train_soul(model, tok, sq_0_data, DEVICE, sq_layer, -1, EPOCHS, 42)
    vec_1 = train_soul(model, tok, sq_1_data, DEVICE, sq_layer, -1, EPOCHS, 99)

    n_phi = 37
    phis = np.linspace(0, 4*np.pi, n_phi)
    E_phi = []
    for phi in phis:
        v = phi_vec(phi, vec_0, vec_1)
        probs = inject_forward(model, tok, prompt, DEVICE, v, sq_layer, -1)
        E_phi.append(float(probs[sq_tok]) - float(probs[sq_tok_1]))
    E_phi = np.array(E_phi)
    amp_05b = (E_phi.max() - E_phi.min()) / 2
    print("    Amplitude = %.4f (1.5B was 0.498)" % amp_05b)

    # === Test 2: Cross-task universality (3 tasks) ===
    print("  [Test 2] Cross-task universality...")
    tasks = {
        'MATH': {'data0': sq_0_data, 'data1': sq_1_data,
                 'prompt': prompt, 'tok0': sq_tok, 'tok1': sq_tok_1},
        'CAPITAL': {
            'data0': [("The capital of France is","Paris"),("The capital of Japan is","Tokyo"),
                      ("The capital of Italy is","Rome"),("The capital of Spain is","Madrid"),
                      ("The capital of Germany is","Berlin")],
            'data1': [("The capital of France is","London"),("The capital of Japan is","Seoul"),
                      ("The capital of Italy is","Berlin"),("The capital of Spain is","Lisbon"),
                      ("The capital of Germany is","Paris")],
            'prompt': "The capital of France is",
            'tok0': tok.encode("Paris")[-1], 'tok1': tok.encode("London")[-1],
        },
        'COLOR': {
            'data0': [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                      ("The sky color is","blue"),("A clear sky is","blue")],
            'data1': [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                      ("The forest is","green"),("Grass color is","green")],
            'prompt': "The sky is",
            'tok0': tok.encode("blue")[-1], 'tok1': tok.encode("green")[-1],
        },
    }

    task_results = {}
    for task_name, task in tasks.items():
        v0 = train_soul(model, tok, task['data0'], DEVICE, sq_layer, -1, EPOCHS, 42)
        v1 = train_soul(model, tok, task['data1'], DEVICE, sq_layer, -1, EPOCHS, 99)
        E_arr = []
        for phi in np.linspace(0, 4*np.pi, 25):
            v = phi_vec(phi, v0, v1)
            probs = inject_forward(model, tok, task['prompt'], DEVICE, v, sq_layer, -1)
            E_arr.append(float(probs[task['tok0']]) - float(probs[task['tok1']]))
        E_arr = np.array(E_arr)
        amp = (E_arr.max() - E_arr.min()) / 2
        task_results[task_name] = round(float(amp), 6)
        print("    %s: amp=%.4f" % (task_name, amp))

    # === Test 3: 2-Qubit CHSH (scaled layers) ===
    # L8 -> L6, L20 -> L17 (proportional scaling)
    sq1_layer, sq2_layer = 6, 17
    print("  [Test 3] 2-Qubit CHSH: SQ1@L%d,pos=-1 x SQ2@L%d,pos=-2..." % (sq1_layer, sq2_layer))
    sq1_0 = train_soul(model, tok, sq_0_data, DEVICE, sq1_layer, -1, EPOCHS, 42)
    sq1_1 = train_soul(model, tok, sq_1_data, DEVICE, sq1_layer, -1, EPOCHS, 99)
    sq2_0_data = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                  ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_1_data = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                  ("The forest is","green"),("Grass color is","green")]
    sq2_0 = train_soul(model, tok, sq2_0_data, DEVICE, sq2_layer, -2, EPOCHS, 42)
    sq2_1 = train_soul(model, tok, sq2_1_data, DEVICE, sq2_layer, -2, EPOCHS, 99)

    n_2d = 13
    phis_2d = np.linspace(0, 2*np.pi, n_2d)
    joint_E = np.zeros((n_2d, n_2d))
    for i, p1 in enumerate(phis_2d):
        for j, p2 in enumerate(phis_2d):
            v1 = phi_vec(p1, sq1_0, sq1_1)
            v2 = phi_vec(p2, sq2_0, sq2_1)
            probs = two_qubit_forward(model, tok, prompt, DEVICE,
                                       v1, v2, sq1_layer, -1, sq2_layer, -2)
            joint_E[i,j] = float(probs[sq_tok]) - float(probs[sq_tok_1])

    best_S = 0
    for i1 in range(n_2d):
        for i2 in range(n_2d):
            for j1 in range(n_2d):
                for j2 in range(n_2d):
                    S = abs(joint_E[i1,j1]-joint_E[i1,j2]+joint_E[i2,j1]+joint_E[i2,j2])
                    if S > best_S: best_S = S

    print("    Joint E range: [%.4f, %.4f]" % (joint_E.min(), joint_E.max()))
    print("    CHSH S_best = %.4f (1.5B was 3.41)" % best_S)

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(phis/np.pi, E_phi, '#E91E63', lw=2,
            label='0.5B (d=896, amp=%.3f)' % amp_05b)
    phi_theory = np.linspace(0, 4*np.pi, 200)
    ax.plot(phi_theory/np.pi, amp_05b*np.cos(phi_theory), 'k--', lw=1.5, alpha=0.5,
            label='cos(phi) fit')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Phase (x pi)'); ax.set_ylabel('E')
    ax.set_title('(a) 0.5B Single-Qubit Interference\n1.5B amp=0.498 vs 0.5B amp=%.3f' % amp_05b,
                 fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[1]
    amps_05 = list(task_results.values())
    amps_15 = [0.4993, 0.4983, 0.4999]  # Q10 1.5B values
    x = np.arange(3)
    ax.bar(x-0.2, amps_15, 0.35, color='#2196F3', edgecolor='black', alpha=0.85, label='1.5B')
    ax.bar(x+0.2, amps_05, 0.35, color='#E91E63', edgecolor='black', alpha=0.85, label='0.5B')
    ax.set_xticks(x)
    ax.set_xticklabels(list(task_results.keys()))
    ax.set_ylabel('Amplitude'); ax.set_ylim(0, 0.55)
    ax.set_title('(b) Cross-Task Universality\n1.5B vs 0.5B', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    comparison = [('1.5B\n(d=1536)', 3.41, '#2196F3'),
                  ('0.5B\n(d=896)', best_S, '#E91E63')]
    bars = ax.bar([c[0] for c in comparison], [c[1] for c in comparison],
                  color=[c[2] for c in comparison], edgecolor='black', alpha=0.85)
    ax.axhline(2.0, color='blue', ls='--', lw=2, label='Classical S=2')
    ax.axhline(2*np.sqrt(2), color='green', ls=':', lw=2, label='Quantum S=2sqrt2')
    for bar, (_, val, _) in zip(bars, comparison):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.05, '%.3f' % val,
                ha='center', fontweight='bold')
    ax.set_ylabel('CHSH S'); ax.set_ylim(0, 4.2)
    ax.set_title('(c) 2-Qubit CHSH: Model Scaling\nDoes S-Qubit survive smaller models?',
                 fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q20: Model Size Scaling (0.5B vs 1.5B)\n'
                 '"Is S-Qubit a universal transformer property?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q20_model_scaling.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q20', 'name': 'model_scaling',
        'model': 'Qwen2.5-0.5B', 'hidden_size': hs, 'n_layers': n_layers,
        'single_qubit_amp': round(float(amp_05b), 6),
        'task_universality': task_results,
        'two_qubit_chsh_S': round(float(best_S), 6),
        'sq1_layer': sq1_layer, 'sq2_layer': sq2_layer,
        'comparison_1_5B': {
            'single_qubit_amp': 0.4984,
            'chsh_S': 3.406006,
        },
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q20_model_scaling.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q20 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
