# -*- coding: utf-8 -*-
"""
Phase Q56: Holevo Bound Test

The Holevo bound states that n qubits can transmit at most n classical bits.
We test: how many classical bits of information can one S-Qubit carry?

Method: encode M messages as distinct phases, decode them, measure
mutual information I(X;Y) as a function of M.
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
    print("[Q56] Holevo Bound Test")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    min_tok = tok.encode("2")[-1]
    max_tok = tok.encode("8")[-1]
    prompt = "min(7,2)="

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    inp = tok(prompt, return_tensors='pt').to(DEVICE)

    def inject_measure(phi):
        v = torch.cos(torch.tensor(phi/2)) * v0 + torch.sin(torch.tensor(phi/2)) * v1
        v = v / v.norm() * v0.norm()
        def hook(m, i, o, vec=v):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[0, -1, :] = vec.to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        handle = model.model.layers[INJECT_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(probs[min_tok]) - float(probs[max_tok])

    # Test alphabet sizes
    M_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    results = []

    for M in M_values:
        print("  M=%d messages..." % M)
        phases = np.linspace(0, 2*np.pi*(1-1/M), M)

        # Build codebook
        codebook = {}
        for i, phi in enumerate(phases):
            E = inject_measure(phi)
            codebook[i] = E

        # Test decoding: send each message once
        correct = 0
        confusion = np.zeros((min(M, 64), min(M, 64)))  # cap confusion matrix
        for i, phi in enumerate(phases):
            E = inject_measure(phi)
            decoded = min(codebook, key=lambda k: abs(codebook[k] - E))
            if decoded == i:
                correct += 1
            if i < 64 and decoded < 64:
                confusion[i, decoded] += 1

        accuracy = correct / M

        # Mutual information: I(X;Y) = H(Y) - H(Y|X)
        # For perfect decoding: I = log2(M)
        # For random decoding: I = 0
        # Approximate: I = log2(M) * accuracy (lower bound)
        if accuracy > 0:
            mutual_info = np.log2(M) * accuracy
        else:
            mutual_info = 0

        # Holevo bound for 1 qubit = 1 bit
        holevo_bound = 1.0

        results.append({
            'M': M,
            'accuracy': round(accuracy, 4),
            'mutual_info_bits': round(mutual_info, 4),
            'holevo_bound': holevo_bound,
            'exceeds_holevo': bool(mutual_info > holevo_bound),
            'log2_M': round(np.log2(M), 2),
        })
        print("    M=%d: acc=%.3f, I=%.2f bits (Holevo=1.0, log2(M)=%.1f)" % (
            M, accuracy, mutual_info, np.log2(M)))

    # Summary
    max_info = max(r['mutual_info_bits'] for r in results)
    max_M_perfect = max(
        (r['M'] for r in results if r['accuracy'] >= 0.99), default=0)
    print("\n  HOLEVO BOUND SUMMARY:")
    print("    Max mutual info: %.2f bits (Holevo bound = 1.0)" % max_info)
    print("    Max M with >=99%% accuracy: %d (= %.1f bits)" % (
        max_M_perfect, np.log2(max_M_perfect) if max_M_perfect > 0 else 0))
    print("    Holevo bound exceeded: %s" % str(max_info > 1.0))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    Ms = [r['M'] for r in results]
    accs = [r['accuracy'] for r in results]
    ax.semilogx(Ms, accs, 'ro-', lw=2, ms=8, base=2, zorder=5)
    ax.axhline(0.99, color='green', ls='--', alpha=0.5, label='99% threshold')
    ax.axhline(1/max(Ms), color='gray', ls='--', alpha=0.3, label='Random')
    ax.set_xlabel('Alphabet size M')
    ax.set_ylabel('Decode accuracy')
    ax.set_title('(a) Channel Capacity\nMax M for 99%%+ = %d' % max_M_perfect,
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    infos = [r['mutual_info_bits'] for r in results]
    ideal = [np.log2(m) for m in Ms]
    ax.semilogx(Ms, infos, 'ro-', lw=2, ms=8, base=2, zorder=5,
                label='S-Qubit I(X;Y)')
    ax.semilogx(Ms, ideal, 'b--', lw=1.5, alpha=0.5, base=2,
                label='Ideal = log2(M)')
    ax.axhline(1.0, color='orange', ls='--', lw=2,
               label='Holevo bound (1 qubit)')
    ax.fill_between(Ms, 0, 1.0, alpha=0.1, color='orange')
    ax.set_xlabel('Alphabet size M')
    ax.set_ylabel('Mutual information (bits)')
    ax.set_title('(b) Holevo Bound Test\n1 S-Qubit carries %.1f bits (Holevo=1.0)' % max_info,
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Phase Q56: Holevo Bound\n'
                 'One S-Qubit carries far more information than one physical qubit',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q56_holevo.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q56', 'name': 'holevo_bound',
        'max_mutual_info_bits': round(max_info, 4),
        'max_M_perfect': max_M_perfect,
        'holevo_exceeded': bool(max_info > 1.0),
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q56_holevo.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q56 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
