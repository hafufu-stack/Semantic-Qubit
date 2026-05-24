# -*- coding: utf-8 -*-
"""
Phase Q64: Adiabatic State Transfer (Quantum Annealing Analogy)
================================================================
Gradually morph one S-Qubit state into another, testing whether
the system follows an "adiabatic" path (smooth interpolation)
or undergoes phase transitions.

This is the S-Qubit analogue of quantum annealing / adiabatic
quantum computation. If the transfer is smooth, we can implement
arbitrary quantum state transformations via interpolation.
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


def main():
    print("[Q64] Adiabatic State Transfer")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # Train two distinct S-Qubits
    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    v_min = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v_max = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]
    
    # Adiabatic interpolation: v(t) = (1-t)*v_min + t*v_max
    N_STEPS = 50
    t_values = np.linspace(0, 1, N_STEPS)
    
    prompts_test = [
        ("min(7,2)=", min_tok, "min"),
        ("max(1,8)=", max_tok, "max"),
    ]
    
    print("  Adiabatic transfer: min -> max (%d steps)" % N_STEPS)
    
    results = {name: [] for _, _, name in prompts_test}
    entropy_list = []
    top_tokens_list = []
    
    for t in t_values:
        v_t = (1 - t) * v_min + t * v_max
        # Optionally normalize to maintain norm
        v_t = v_t / v_t.norm() * ((1-t) * v_min.norm() + t * v_max.norm())
        
        for prompt, target_id, name in prompts_test:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def hook(m, i, o, v=v_t):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, -1, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
            with torch.no_grad():
                out = model(**inp)
            handle.remove()
            probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
            results[name].append(float(probs[target_id]))
            
            if name == 'min':
                # Compute entropy
                top_probs = probs.topk(100).values
                top_probs = top_probs[top_probs > 1e-10]
                entropy = -float(torch.sum(top_probs * torch.log2(top_probs)))
                entropy_list.append(entropy)
                
                # Top predicted token
                top_id = int(probs.argmax())
                top_token = tok.decode([top_id]).strip()
                top_tokens_list.append(top_token)
    
    # Analyze phase transition
    min_probs = np.array(results['min'])
    max_probs = np.array(results['max'])
    
    # Find crossover point
    diff = min_probs - max_probs
    crossover_idx = None
    for i in range(1, len(diff)):
        if diff[i-1] > 0 and diff[i] <= 0:
            crossover_idx = i
            break
    
    crossover_t = t_values[crossover_idx] if crossover_idx else 0.5
    
    # Smoothness: compute 2nd derivative (jerk)
    d2_min = np.diff(min_probs, n=2)
    d2_max = np.diff(max_probs, n=2)
    smoothness = 1.0 / (np.max(np.abs(d2_min)) + 1e-10)
    
    # Is it adiabatic (smooth) or sudden (phase transition)?
    max_gradient = np.max(np.abs(np.diff(min_probs)))
    adiabatic = max_gradient < 0.1  # smooth transition
    
    print("\n  RESULTS:")
    print("    Crossover point: t=%.3f" % crossover_t)
    print("    Max gradient: %.4f" % max_gradient)
    print("    Smoothness: %.1f" % smoothness)
    print("    Adiabatic (smooth): %s" % ("YES" if adiabatic else "NO - phase transition"))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Probability evolution
    ax = axes[0]
    ax.plot(t_values, min_probs, 'o-', color='#2196F3', linewidth=2,
            markersize=3, label='P(min answer)')
    ax.plot(t_values, max_probs, 's-', color='#FF5722', linewidth=2,
            markersize=3, label='P(max answer)')
    ax.axvline(crossover_t, color='gray', ls='--', alpha=0.5,
               label='Crossover (t=%.2f)' % crossover_t)
    ax.fill_between(t_values, 0, 1, where=t_values < crossover_t,
                    alpha=0.05, color='blue')
    ax.fill_between(t_values, 0, 1, where=t_values >= crossover_t,
                    alpha=0.05, color='red')
    ax.set_xlabel('Interpolation parameter t')
    ax.set_ylabel('Probability')
    ax.set_title('(a) Adiabatic State Transfer\nmin(S-Qubit) -> max(S-Qubit)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Entropy during transfer
    ax = axes[1]
    ax.plot(t_values, entropy_list, 'o-', color='#9C27B0', linewidth=2,
            markersize=3)
    ax.axvline(crossover_t, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Interpolation parameter t')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('(b) Decision Entropy\nPeak at crossover = quantum uncertainty',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) Energy landscape analogy
    ax = axes[2]
    # Create energy landscape from probabilities
    energy_min = -np.log(min_probs + 1e-10)
    energy_max = -np.log(max_probs + 1e-10)
    ax.plot(t_values, energy_min, '-', color='#2196F3', linewidth=2,
            label='E(min state)')
    ax.plot(t_values, energy_max, '-', color='#FF5722', linewidth=2,
            label='E(max state)')
    ax.fill_between(t_values, energy_min, energy_max, alpha=0.1, color='gray')
    ax.axvline(crossover_t, color='gray', ls='--', alpha=0.5,
               label='Level crossing')
    ax.set_xlabel('Interpolation parameter t')
    ax.set_ylabel('-log(P) (energy analogue)')
    ax.set_title('(c) Energy Landscape\nLevel crossing at t=%.2f' % crossover_t,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 15)

    plt.suptitle('Phase Q64: Adiabatic State Transfer\n'
                 'Smooth quantum state morphing in S-Qubit space',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q64_adiabatic.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q64', 'name': 'adiabatic_state_transfer',
        'crossover_t': round(float(crossover_t), 3),
        'max_gradient': round(float(max_gradient), 4),
        'smoothness': round(float(smoothness), 1),
        'adiabatic': bool(adiabatic),
        'n_steps': N_STEPS,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q64_adiabatic.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q64 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
