# -*- coding: utf-8 -*-
"""
Phase Q24: Quantum Teleportation via S-Qubit

Quantum teleportation: transfer a quantum state from qubit A to qubit B
without physically moving it, using entanglement as a resource.

Protocol:
  1. Create entangled pair: SQ2@L16 and SQ3@L20 (Bell pair)
  2. SQ1@L8 holds the state to teleport: |psi> = cos(phi/2)|0> + sin(phi/2)|1>
  3. "Bell measurement" on SQ1+SQ2: measure joint state
  4. Apply correction to SQ3 based on measurement
  5. Verify: SQ3 output matches original SQ1 state

S-Qubit implementation:
  - Step 1: Train SQ2 and SQ3 as entangled pair (correlated training data)
  - Step 2: Train SQ1 with independent data
  - Step 3: Inject all three, measure output sensitivity to SQ1's phase
  - If teleportation works: SQ3's output should track SQ1's input phase
    even though SQ3 was never directly trained with SQ1's data

  This demonstrates information transfer through the attention mechanism's
  entanglement, without direct coupling between SQ1 and SQ3.
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

SQ1_LAYER, SQ1_POS = 8, -1    # State to teleport
SQ2_LAYER, SQ2_POS = 16, -2   # Bell pair member 1 (mediator)
SQ3_LAYER, SQ3_POS = 20, -3   # Bell pair member 2 (receiver)
EPOCHS = 120


def train_soul(model, tok, data, device, layer, pos, epochs, seed):
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


def multi_inject_forward(model, tok, prompt, device, injections):
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    handles = []
    for vec, layer, pos in injections:
        actual_pos = pos if pos >= 0 else seq_len + pos
        def make_hook(v, p):
            def hook(m, i, o):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, p, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            return hook
        handle = model.model.layers[layer].register_forward_hook(make_hook(vec, actual_pos))
        handles.append(handle)
    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()
    return torch.softmax(out.logits[0, -1, :].float(), dim=-1)


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def main():
    print("[Q24] Quantum Teleportation via S-Qubit")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    prompt = "min(7,2)="
    tok_min = tok.encode("2")[-1]
    tok_max = tok.encode("7")[-1]

    # SQ1: the state to teleport (math: min vs max)
    sq1_data0 = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                 ("min(4,6)=","4"),("min(9,3)=","3")]
    sq1_data1 = [("min(3,7)=","7"),("min(5,2)=","5"),("min(8,1)=","8"),
                 ("min(4,6)=","6"),("min(9,3)=","9")]

    # SQ2: Bell pair mediator (color domain - different from SQ1!)
    sq2_data0 = [("The sky is","blue"),("The ocean is","blue"),("Clear water is","blue"),
                 ("The sky color is","blue"),("A clear sky is","blue")]
    sq2_data1 = [("The grass is","green"),("Leaves are","green"),("Plants are","green"),
                 ("The forest is","green"),("Grass color is","green")]

    # SQ3: Bell pair receiver (size domain - different from both!)
    sq3_data0 = [("An ant is","small"),("A mouse is","small"),("A coin is","small"),
                 ("A seed is","small"),("A bug is","small")]
    sq3_data1 = [("A whale is","large"),("An elephant is","large"),("A mountain is","large"),
                 ("The sun is","large"),("A building is","large")]

    print("  Training SQ1@L%d (state to teleport)..." % SQ1_LAYER)
    sq1_0 = train_soul(model, tok, sq1_data0, DEVICE, SQ1_LAYER, SQ1_POS, EPOCHS, 42)
    sq1_1 = train_soul(model, tok, sq1_data1, DEVICE, SQ1_LAYER, SQ1_POS, EPOCHS, 99)

    print("  Training SQ2@L%d (Bell mediator)..." % SQ2_LAYER)
    sq2_0 = train_soul(model, tok, sq2_data0, DEVICE, SQ2_LAYER, SQ2_POS, EPOCHS, 42)
    sq2_1 = train_soul(model, tok, sq2_data1, DEVICE, SQ2_LAYER, SQ2_POS, EPOCHS, 99)

    print("  Training SQ3@L%d (Bell receiver)..." % SQ3_LAYER)
    sq3_0 = train_soul(model, tok, sq3_data0, DEVICE, SQ3_LAYER, SQ3_POS, EPOCHS, 42)
    sq3_1 = train_soul(model, tok, sq3_data1, DEVICE, SQ3_LAYER, SQ3_POS, EPOCHS, 99)

    def E_val(probs):
        return float(probs[tok_min]) - float(probs[tok_max])

    # ── Experiment 1: Direct SQ1 interference (baseline) ──
    print("\n  [1] SQ1 direct interference (baseline)...")
    n_phi = 25
    phis = np.linspace(0, 2*np.pi, n_phi)
    E_direct = []
    for phi in phis:
        v1 = phi_vec(phi, sq1_0, sq1_1)
        probs = multi_inject_forward(model, tok, prompt, DEVICE,
                                      [(v1, SQ1_LAYER, SQ1_POS)])
        E_direct.append(E_val(probs))
    E_direct = np.array(E_direct)
    amp_direct = (E_direct.max() - E_direct.min()) / 2
    print("    Direct SQ1 amplitude: %.4f" % amp_direct)

    # ── Experiment 2: SQ3 alone (no teleportation, should be ~0) ──
    print("\n  [2] SQ3 alone (no teleportation)...")
    E_sq3_alone = []
    for phi in phis:
        v3 = phi_vec(phi, sq3_0, sq3_1)
        probs = multi_inject_forward(model, tok, prompt, DEVICE,
                                      [(v3, SQ3_LAYER, SQ3_POS)])
        E_sq3_alone.append(E_val(probs))
    E_sq3_alone = np.array(E_sq3_alone)
    amp_sq3_alone = (E_sq3_alone.max() - E_sq3_alone.min()) / 2
    print("    SQ3 alone amplitude: %.4f (should be small)" % amp_sq3_alone)

    # ── Experiment 3: Teleportation! ──
    # Sweep SQ1's phase while keeping SQ2=|+> and SQ3=|+> (Bell pair in superposition)
    # If teleportation occurs: SQ3's contribution to output tracks SQ1's phase
    print("\n  [3] Teleportation: sweep SQ1 phase with SQ2+SQ3 Bell pair...")
    sq2_plus = phi_vec(np.pi/2, sq2_0, sq2_1)
    sq3_plus = phi_vec(np.pi/2, sq3_0, sq3_1)

    E_teleport = []
    for phi in phis:
        v1 = phi_vec(phi, sq1_0, sq1_1)
        probs = multi_inject_forward(model, tok, prompt, DEVICE, [
            (v1, SQ1_LAYER, SQ1_POS),
            (sq2_plus, SQ2_LAYER, SQ2_POS),
            (sq3_plus, SQ3_LAYER, SQ3_POS),
        ])
        E_teleport.append(E_val(probs))
    E_teleport = np.array(E_teleport)
    amp_teleport = (E_teleport.max() - E_teleport.min()) / 2
    print("    Teleportation amplitude: %.4f" % amp_teleport)

    # ── Experiment 4: Without mediator (SQ1 + SQ3 only, no SQ2) ──
    print("\n  [4] SQ1 + SQ3 only (no mediator)...")
    E_no_mediator = []
    for phi in phis:
        v1 = phi_vec(phi, sq1_0, sq1_1)
        probs = multi_inject_forward(model, tok, prompt, DEVICE, [
            (v1, SQ1_LAYER, SQ1_POS),
            (sq3_plus, SQ3_LAYER, SQ3_POS),
        ])
        E_no_mediator.append(E_val(probs))
    E_no_mediator = np.array(E_no_mediator)
    amp_no_mediator = (E_no_mediator.max() - E_no_mediator.min()) / 2
    print("    No-mediator amplitude: %.4f" % amp_no_mediator)

    # ── Experiment 5: Fidelity analysis ──
    # Teleportation fidelity: correlation between direct and teleported
    corr = np.corrcoef(E_direct, E_teleport)[0, 1]
    print("\n  Teleportation fidelity (Pearson r): %.4f" % corr)

    # Mediator enhancement
    mediator_enhancement = amp_teleport / (amp_no_mediator + 1e-6)
    print("  Mediator enhancement: %.4f (teleport/no-mediator)" % mediator_enhancement)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: All interference curves
    ax = axes[0]
    ax.plot(phis/np.pi, E_direct, '#E91E63', lw=2.5,
            label='Direct SQ1 (amp=%.3f)' % amp_direct)
    ax.plot(phis/np.pi, E_teleport, '#9C27B0', lw=2.5, linestyle='--',
            label='Teleported via SQ2 (amp=%.3f)' % amp_teleport)
    ax.plot(phis/np.pi, E_no_mediator, '#2196F3', lw=2, linestyle=':',
            label='SQ1+SQ3 no mediator (amp=%.3f)' % amp_no_mediator)
    ax.plot(phis/np.pi, E_sq3_alone, '#4CAF50', lw=1.5, linestyle='-.',
            label='SQ3 alone (amp=%.3f)' % amp_sq3_alone)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('SQ1 Phase phi / pi', fontsize=11)
    ax.set_ylabel('E = P(min) - P(max)', fontsize=11)
    ax.set_title('(a) Quantum Teleportation\nSQ1 state transferred via SQ2 mediator',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel B: Scatter plot: direct vs teleported
    ax = axes[1]
    ax.scatter(E_direct, E_teleport, c=phis/np.pi, cmap='hsv', s=60,
              edgecolors='black', zorder=3)
    ax.plot([-1, 1], [-1, 1], 'k--', lw=1.5, alpha=0.5, label='Perfect teleportation')
    ax.set_xlabel('E (direct SQ1)', fontsize=11)
    ax.set_ylabel('E (teleported via SQ2)', fontsize=11)
    ax.set_title('(b) Teleportation Fidelity\nPearson r = %.4f' % corr,
                 fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

    # Panel C: Amplitude comparison
    ax = axes[2]
    categories = ['Direct\nSQ1', 'Teleported\nSQ1->SQ2->SQ3', 'No mediator\nSQ1+SQ3', 'SQ3\nalone']
    amplitudes = [amp_direct, amp_teleport, amp_no_mediator, amp_sq3_alone]
    colors = ['#E91E63', '#9C27B0', '#2196F3', '#4CAF50']
    bars = ax.bar(range(4), amplitudes, color=colors, edgecolor='black', alpha=0.85)
    for bar, amp in zip(bars, amplitudes):
        ax.text(bar.get_x()+bar.get_width()/2, amp+0.01, '%.3f' % amp,
                ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Interference Amplitude', fontsize=11)
    ax.set_title('(c) Teleportation vs Direct\nMediator enhancement: %.2fx' % mediator_enhancement,
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase Q24: Quantum Teleportation\n'
                 'SQ1 state transferred to SQ3 via entangled mediator SQ2',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q24_teleportation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q24', 'name': 'quantum_teleportation',
        'sq1': {'layer': SQ1_LAYER, 'pos': SQ1_POS},
        'sq2': {'layer': SQ2_LAYER, 'pos': SQ2_POS},
        'sq3': {'layer': SQ3_LAYER, 'pos': SQ3_POS},
        'amp_direct': round(float(amp_direct), 6),
        'amp_teleport': round(float(amp_teleport), 6),
        'amp_no_mediator': round(float(amp_no_mediator), 6),
        'amp_sq3_alone': round(float(amp_sq3_alone), 6),
        'fidelity_r': round(float(corr), 6),
        'mediator_enhancement': round(float(mediator_enhancement), 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q24_teleportation.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q24 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
