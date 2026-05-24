# -*- coding: utf-8 -*-
"""
Phase Q48: Quantum Reservoir Computing

Use the LLM as a "quantum reservoir" for time series prediction.
The idea: inject input signals as phase-encoded soul vectors,
let the LLM's attention layers process them, and extract predictions
from the output probability distribution.

This bridges quantum computing and machine learning:
the LLM's attention is a high-dimensional dynamical system
that can be used as a reservoir computer.

Test tasks:
  1. Sine wave prediction
  2. XOR sequence classification
  3. Nonlinear function approximation
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


def phi_vec(phi, v0, v1):
    v = np.cos(phi / 2) * v0 + np.sin(phi / 2) * v1
    n = v.norm()
    if n > 0: v = v / n * v0.norm()
    return v


def inject_get_features(model, tok, prompt, device, vec, layer, n_features=20):
    """Inject and extract top-k logit values as reservoir features."""
    inp = tok(prompt, return_tensors='pt').to(device)
    def hook(m, i, o, v=vec):
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[0, -1, :] = v.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    handle.remove()
    logits = out.logits[0, -1, :].float()
    # Extract top-k logit values as features
    topk = torch.topk(logits, n_features)
    features = topk.values.cpu().numpy()
    return features


def main():
    print("[Q48] Quantum Reservoir Computing")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("min(7,2)=", "2"), ("min(9,1)=", "1"), ("min(5,3)=", "3")]
    max_data = [("max(1,8)=", "8"), ("max(2,9)=", "9"), ("max(3,5)=", "5")]
    prompt = "min(7,2)="

    print("  Training basis vectors...")
    v0 = train_soul(model, tok, min_data, DEVICE, INJECT_LAYER, EPOCHS, 42)
    v1 = train_soul(model, tok, max_data, DEVICE, INJECT_LAYER, EPOCHS, 99)

    N_FEATURES = 20
    N_TRAIN = 60
    N_TEST = 20

    # ─── Task 1: Sine wave prediction ───
    print("\n  Task 1: Sine wave prediction...")
    total = N_TRAIN + N_TEST
    t_vals = np.linspace(0, 4 * np.pi, total)
    y_sine = np.sin(t_vals)

    # Encode each time step as phase, extract features
    features_sine = []
    for i in range(total):
        phi = (y_sine[i] + 1) * np.pi  # map [-1,1] -> [0, 2*pi]
        v = phi_vec(phi, v0, v1)
        feat = inject_get_features(model, tok, prompt, DEVICE, v,
                                    INJECT_LAYER, N_FEATURES)
        features_sine.append(feat)
    X_sine = np.array(features_sine)

    # Train linear readout: predict next value from current features
    X_train = X_sine[:N_TRAIN]
    y_train = y_sine[1:N_TRAIN+1]  # next step target
    n_test_actual = min(N_TEST, total - N_TRAIN - 1)
    X_test = X_sine[N_TRAIN:N_TRAIN+n_test_actual]
    y_test = y_sine[N_TRAIN+1:N_TRAIN+1+n_test_actual]

    # Ridge regression
    from numpy.linalg import lstsq
    reg = 0.01
    W = lstsq(X_train.T @ X_train + reg * np.eye(N_FEATURES),
              X_train.T @ y_train, rcond=None)[0]
    y_pred_sine = X_test @ W

    rmse_sine = np.sqrt(np.mean((y_pred_sine - y_test)**2))
    corr_sine = np.corrcoef(y_pred_sine, y_test)[0, 1]
    print("    RMSE: %.4f" % rmse_sine)
    print("    Correlation: %.4f" % corr_sine)

    # ─── Task 2: XOR classification ───
    print("\n  Task 2: XOR sequence classification...")
    np.random.seed(42)
    N_xor = N_TRAIN + N_TEST
    bits_a = np.random.randint(0, 2, N_xor)
    bits_b = np.random.randint(0, 2, N_xor)
    y_xor = (bits_a ^ bits_b).astype(float)

    features_xor = []
    for i in range(N_xor):
        # Encode two bits as phase: phi = pi/4 * (2*a + b)
        phi = np.pi / 4 * (2 * bits_a[i] + bits_b[i])
        v = phi_vec(phi, v0, v1)
        feat = inject_get_features(model, tok, prompt, DEVICE, v,
                                    INJECT_LAYER, N_FEATURES)
        features_xor.append(feat)
    X_xor = np.array(features_xor)

    X_train_xor = X_xor[:N_TRAIN]
    y_train_xor = y_xor[:N_TRAIN]
    X_test_xor = X_xor[N_TRAIN:]
    y_test_xor = y_xor[N_TRAIN:]

    W_xor = lstsq(X_train_xor.T @ X_train_xor + reg * np.eye(N_FEATURES),
                   X_train_xor.T @ y_train_xor, rcond=None)[0]
    y_pred_xor = X_test_xor @ W_xor
    y_pred_xor_binary = (y_pred_xor > 0.5).astype(float)
    accuracy_xor = np.mean(y_pred_xor_binary == y_test_xor)
    print("    Accuracy: %.1f%%" % (100 * accuracy_xor))

    # ─── Task 3: Nonlinear function: y = sin(x) * cos(2x) ───
    print("\n  Task 3: Nonlinear function approximation...")
    x_vals = np.linspace(0, 2 * np.pi, total)
    y_nonlin = np.sin(x_vals) * np.cos(2 * x_vals)

    features_nonlin = []
    for i in range(total):
        phi = (x_vals[i] / (2 * np.pi)) * 2 * np.pi  # direct mapping
        v = phi_vec(phi, v0, v1)
        feat = inject_get_features(model, tok, prompt, DEVICE, v,
                                    INJECT_LAYER, N_FEATURES)
        features_nonlin.append(feat)
    X_nonlin = np.array(features_nonlin)

    X_train_nl = X_nonlin[:N_TRAIN]
    y_train_nl = y_nonlin[:N_TRAIN]
    X_test_nl = X_nonlin[N_TRAIN:]
    y_test_nl = y_nonlin[N_TRAIN:]

    W_nl = lstsq(X_train_nl.T @ X_train_nl + reg * np.eye(N_FEATURES),
                  X_train_nl.T @ y_train_nl, rcond=None)[0]
    y_pred_nl = X_test_nl @ W_nl
    rmse_nl = np.sqrt(np.mean((y_pred_nl - y_test_nl)**2))
    corr_nl = np.corrcoef(y_pred_nl, y_test_nl)[0, 1]
    print("    RMSE: %.4f" % rmse_nl)
    print("    Correlation: %.4f" % corr_nl)

    print("\n  QUANTUM RESERVOIR SUMMARY:")
    print("    Sine prediction:   r=%.3f, RMSE=%.3f" % (corr_sine, rmse_sine))
    print("    XOR classification: %.1f%%" % (100 * accuracy_xor))
    print("    Nonlinear approx:  r=%.3f, RMSE=%.3f" % (corr_nl, rmse_nl))

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Sine prediction
    ax = axes[0]
    test_t = np.arange(len(y_test))
    ax.plot(test_t, y_test, 'b-', lw=2, label='True')
    ax.plot(test_t, y_pred_sine, 'r--', lw=2, label='Predicted (r=%.3f)' % corr_sine)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.set_title('(a) Sine Wave Prediction\nRMSE=%.3f' % rmse_sine, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: XOR classification
    ax = axes[1]
    ax.scatter(range(N_TEST), y_test_xor, c='blue', s=60, label='True XOR', alpha=0.7)
    ax.scatter(range(N_TEST), y_pred_xor, c='red', s=30, marker='x',
               label='Predicted', alpha=0.7)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('XOR value')
    ax.set_title('(b) XOR Classification\nAccuracy=%.1f%%' % (100*accuracy_xor),
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel C: Nonlinear function
    ax = axes[2]
    ax.plot(range(N_TEST), y_test_nl, 'b-', lw=2, label='True')
    ax.plot(range(N_TEST), y_pred_nl, 'r--', lw=2,
            label='Predicted (r=%.3f)' % corr_nl)
    ax.set_xlabel('Sample')
    ax.set_ylabel('sin(x)*cos(2x)')
    ax.set_title('(c) Nonlinear Approximation\nRMSE=%.3f' % rmse_nl,
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Phase Q48: Quantum Reservoir Computing\n'
                 'LLM attention as a high-dimensional dynamical reservoir',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q48_reservoir.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q48', 'name': 'quantum_reservoir',
        'inject_layer': INJECT_LAYER,
        'n_features': N_FEATURES,
        'sine': {'rmse': round(rmse_sine, 4), 'corr': round(float(corr_sine), 4)},
        'xor': {'accuracy': round(float(accuracy_xor), 4)},
        'nonlinear': {'rmse': round(rmse_nl, 4), 'corr': round(float(corr_nl), 4)},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q48_reservoir.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Q48 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
