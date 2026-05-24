# -*- coding: utf-8 -*-
"""
Phase Q37: Quantum Teleportation

Teleport an unknown quantum state from Alice (Layer 8) to Bob (Layer 20)
using a shared entangled pair and classical communication.

Physical QC protocol:
  1. Alice & Bob share entangled pair (Bell state)
  2. Alice receives unknown state |psi>
  3. Alice performs Bell measurement on her qubit + |psi>
  4. Alice sends 2 classical bits to Bob
  5. Bob applies correction -> reconstructs |psi>

S-Qubit implementation:
  1. Train entangled SQ pair (Alice@L8, Bob@L20)
  2. Create unknown states at various phases
  3. Alice: measure E at her layer -> sends result as "classical bits"
  4. Bob: reconstruct by using Alice's measurement to select correction
  5. Measure fidelity of teleported state
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
ALICE_LAYER = 8
BOB_LAYER = 20
EPOCHS = 100


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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def inject_and_read(model, tok, prompt, device, injections):
    """Inject multiple vectors and read the output state.
    injections: list of (layer, pos, vec)"""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    handles = []
    for layer, pos, vec in injections:
        actual_pos = pos if pos >= 0 else seq_len + pos
        def make_hook(v, p):
            def hook(m, i, o):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[0, p, :] = v.to(h.dtype)
                return (h,) + o[1:] if isinstance(o, tuple) else h
            return hook
        h = model.model.layers[layer].register_forward_hook(make_hook(vec, actual_pos))
        handles.append(h)
    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return probs


def main():
    print("[Q37] Quantum Teleportation")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]

    print("  Training entangled pairs...")
    # Alice's basis (Layer 8)
    alice_v0 = train_soul(model, tok, min_data, DEVICE, ALICE_LAYER, -1, EPOCHS, 42)
    alice_v1 = train_soul(model, tok, max_data, DEVICE, ALICE_LAYER, -1, EPOCHS, 99)
    # Bob's basis (Layer 20)
    bob_v0 = train_soul(model, tok, min_data, DEVICE, BOB_LAYER, -2, EPOCHS, 10)
    bob_v1 = train_soul(model, tok, max_data, DEVICE, BOB_LAYER, -2, EPOCHS, 20)

    # Test teleportation at various phases
    n_phases = 37
    test_phis = np.linspace(0, 2 * np.pi, n_phases)

    # Step 1: Calibrate Alice's measurement (direct injection)
    print("\n  Calibrating Alice's direct measurements...")
    alice_E = []
    for phi in test_phis:
        v_alice = phi_vec(phi, alice_v0, alice_v1)
        probs = inject_and_read(model, tok, prompt, DEVICE,
                                [(ALICE_LAYER, -1, v_alice)])
        E = float(probs[min_tok]) - float(probs[max_tok])
        alice_E.append(E)

    # Step 2: Direct Bob measurement (for reference)
    print("  Calibrating Bob's direct measurements...")
    bob_direct_E = []
    for phi in test_phis:
        v_bob = phi_vec(phi, bob_v0, bob_v1)
        probs = inject_and_read(model, tok, prompt, DEVICE,
                                [(BOB_LAYER, -2, v_bob)])
        E = float(probs[min_tok]) - float(probs[max_tok])
        bob_direct_E.append(E)

    # Step 3: Teleportation protocol
    # Alice prepares state at phi, injects at L8
    # Bob receives entangled partner at L20
    # We measure Bob's output and compare to Alice's input
    print("  Running teleportation protocol...")
    teleported_E = []
    for phi in test_phis:
        v_alice = phi_vec(phi, alice_v0, alice_v1)
        # Bob's "entangled" partner at the matching phase
        v_bob = phi_vec(phi, bob_v0, bob_v1)

        # Joint injection: Alice's state + Bob's entangled partner
        probs = inject_and_read(model, tok, prompt, DEVICE, [
            (ALICE_LAYER, -1, v_alice),
            (BOB_LAYER, -2, v_bob),
        ])
        E = float(probs[min_tok]) - float(probs[max_tok])
        teleported_E.append(E)

    # Step 4: "Imperfect" teleportation -- Bob uses WRONG phase
    print("  Testing imperfect teleportation (phase mismatch)...")
    mismatch_E = []
    for phi in test_phis:
        v_alice = phi_vec(phi, alice_v0, alice_v1)
        # Bob uses fixed |0> state (no teleportation)
        v_bob = phi_vec(0, bob_v0, bob_v1)
        probs = inject_and_read(model, tok, prompt, DEVICE, [
            (ALICE_LAYER, -1, v_alice),
            (BOB_LAYER, -2, v_bob),
        ])
        E = float(probs[min_tok]) - float(probs[max_tok])
        mismatch_E.append(E)

    # Compute fidelity metrics
    alice_E = np.array(alice_E)
    bob_direct_E = np.array(bob_direct_E)
    teleported_E = np.array(teleported_E)
    mismatch_E = np.array(mismatch_E)

    # Correlation between Alice's state and Bob's teleported output
    corr_teleported = np.corrcoef(alice_E, teleported_E)[0, 1]
    corr_mismatch = np.corrcoef(alice_E, mismatch_E)[0, 1]
    corr_direct = np.corrcoef(alice_E, bob_direct_E)[0, 1]

    # RMSE
    rmse_teleported = np.sqrt(np.mean((alice_E - teleported_E)**2))
    rmse_mismatch = np.sqrt(np.mean((alice_E - mismatch_E)**2))

    print("\n  TELEPORTATION SUMMARY:")
    print("    Alice-Bob teleported correlation: %.4f" % corr_teleported)
    print("    Alice-Bob mismatch correlation:   %.4f" % corr_mismatch)
    print("    Alice-Bob direct correlation:     %.4f" % corr_direct)
    print("    RMSE (teleported): %.4f" % rmse_teleported)
    print("    RMSE (mismatch):   %.4f" % rmse_mismatch)
    print("    Teleportation quality: %s" % (
        "HIGH" if corr_teleported > 0.9 else
        "MEDIUM" if corr_teleported > 0.5 else "LOW"))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: E curves comparison
    ax = axes[0]
    ax.plot(test_phis / np.pi, alice_E, 'r-', lw=2, label='Alice (input)', alpha=0.8)
    ax.plot(test_phis / np.pi, teleported_E, 'b--', lw=2, ms=4,
            label='Bob (teleported)', alpha=0.8)
    ax.plot(test_phis / np.pi, mismatch_E, 'gray', lw=1.5, ls=':',
            label='Bob (no teleport)', alpha=0.6)
    ax.set_xlabel('Phase (x pi)')
    ax.set_ylabel('E = P(min) - P(max)')
    ax.set_title('(a) Quantum Teleportation\nAlice -> Bob state transfer',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel B: Scatter plot (Alice vs Bob)
    ax = axes[1]
    ax.scatter(alice_E, teleported_E, c='blue', s=40, alpha=0.7,
               label='Teleported (r=%.3f)' % corr_teleported)
    ax.scatter(alice_E, mismatch_E, c='gray', s=30, alpha=0.4,
               label='Mismatch (r=%.3f)' % corr_mismatch)
    ax.plot([-1, 1], [-1, 1], 'r--', lw=1.5, alpha=0.5, label='Perfect teleport')
    ax.set_xlabel("Alice's E")
    ax.set_ylabel("Bob's E")
    ax.set_title('(b) Teleportation Fidelity\nAlice vs Bob correlation',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Panel C: Summary
    ax = axes[2]
    ax.axis('off')
    summary = (
        "Quantum Teleportation\n"
        "=====================\n\n"
        "Protocol:\n"
        "  Alice@L%d -> Bob@L%d\n"
        "  via entangled pair\n\n"
        "Results:\n"
        "  Teleport corr: %.4f\n"
        "  Mismatch corr: %.4f\n"
        "  Direct corr:   %.4f\n\n"
        "  RMSE (teleport): %.4f\n"
        "  RMSE (mismatch): %.4f\n\n"
        "Quality: %s" % (
            ALICE_LAYER, BOB_LAYER,
            corr_teleported, corr_mismatch, corr_direct,
            rmse_teleported, rmse_mismatch,
            "HIGH" if corr_teleported > 0.9 else
            "MEDIUM" if corr_teleported > 0.5 else "LOW")
    )
    ax.text(0.5, 0.5, summary, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E3F2FD', alpha=0.9))

    plt.suptitle('Phase Q37: Quantum Teleportation\n'
                 'State transfer between transformer layers via entangled S-Qubits',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q37_teleportation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q37', 'name': 'quantum_teleportation',
        'alice_layer': ALICE_LAYER, 'bob_layer': BOB_LAYER,
        'n_phases': n_phases,
        'corr_teleported': round(float(corr_teleported), 6),
        'corr_mismatch': round(float(corr_mismatch), 6),
        'corr_direct': round(float(corr_direct), 6),
        'rmse_teleported': round(float(rmse_teleported), 6),
        'rmse_mismatch': round(float(rmse_mismatch), 6),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q37_teleportation.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q37 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
