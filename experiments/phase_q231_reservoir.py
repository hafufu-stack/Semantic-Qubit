# -*- coding: utf-8 -*-
"""
Phase Q231: Quantum Reservoir Computing
==========================================
Instead of VQE (optimized), use the LLM as a FIXED quantum reservoir.
Feed input, let it evolve through layers, read out.
If reservoir outperforms linear regression -> quantum advantage
in machine learning without any optimization.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_nonlinear_task(n_samples, dim, seed=42):
    """Generate XOR-like nonlinear classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, dim).astype(np.float32)
    # Nonlinear target: product of pairs
    y = np.zeros(n_samples)
    for i in range(0, dim - 1, 2):
        y += X[:, i] * X[:, i+1]
    y = (y > 0).astype(np.float32)
    return X, y


def generate_time_series(n_samples, seed=42):
    """Generate chaotic time series (Mackey-Glass-like)."""
    rng = np.random.RandomState(seed)
    x = np.zeros(n_samples + 100)
    x[0] = 0.9
    for t in range(1, len(x)):
        x[t] = x[t-1] + 0.02 * (0.2 * x[max(0,t-17)] / (1 + x[max(0,t-17)]**10) - 0.1 * x[t-1])
    x = x[100:]  # remove transient
    return x.astype(np.float32)


def llm_reservoir(model, tok, device, X_input, layer_idx=-1, readout_dim=64):
    """Use LLM as fixed reservoir: inject input, read hidden state."""
    embed_layer = model.model.embed_tokens
    prompt = "reservoir input:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    base_embeds = embed_layer(inp).detach()

    features = []
    for x in X_input:
        embeds = base_embeds.clone()
        dim_inject = min(len(x), embeds.shape[-1])
        with torch.no_grad():
            embeds[0, -1, :dim_inject] += torch.tensor(x[:dim_inject], device=device)
            out = model(inputs_embeds=embeds.float(), output_hidden_states=True)
            h = out.hidden_states[layer_idx][0, -1, :readout_dim].float().cpu().numpy()
        features.append(h)

    return np.array(features)


def main():
    print("=" * 60)
    print("Phase Q231: Quantum Reservoir Computing")
    print("  (LLM as fixed quantum reservoir)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    n_layers = len(model.model.layers)
    readout_dim = 64
    all_results = []

    # Task 1: Nonlinear classification
    print("\n--- Task 1: Nonlinear Classification ---")
    n_train, n_test = 40, 20
    input_dim = 8
    X, y = generate_nonlinear_task(n_train + n_test, input_dim)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Baseline: linear on raw input
    ridge_raw = Ridge(alpha=1.0)
    ridge_raw.fit(X_train, y_train)
    pred_raw = ridge_raw.predict(X_test)
    acc_raw = np.mean((pred_raw > 0.5) == y_test)

    # Reservoir: linear on LLM features
    for layer_idx in [0, n_layers // 2, n_layers]:
        features_train = llm_reservoir(model, tok, device, X_train, layer_idx, readout_dim)
        features_test = llm_reservoir(model, tok, device, X_test, layer_idx, readout_dim)

        ridge_res = Ridge(alpha=1.0)
        ridge_res.fit(features_train, y_train)
        pred_res = ridge_res.predict(features_test)
        acc_res = np.mean((pred_res > 0.5) == y_test)

        improvement = (acc_res - acc_raw) / max(acc_raw, 0.01)
        print("  L%d: reservoir=%.1f%%, raw=%.1f%%, improvement=%.1f%%" %
              (layer_idx, acc_res * 100, acc_raw * 100, improvement * 100))

        all_results.append({
            'task': 'nonlinear_classification',
            'layer': layer_idx,
            'reservoir_accuracy': round(float(acc_res), 4),
            'baseline_accuracy': round(float(acc_raw), 4),
            'improvement': round(float(improvement), 4),
        })

    # Task 2: Time series prediction
    print("\n--- Task 2: Time Series Prediction ---")
    ts = generate_time_series(100)
    lookback = 5
    X_ts = np.array([ts[i:i+lookback] for i in range(len(ts) - lookback - 1)])
    y_ts = ts[lookback+1:]
    split = int(len(X_ts) * 0.7)
    X_ts_train, X_ts_test = X_ts[:split], X_ts[split:]
    y_ts_train, y_ts_test = y_ts[:split], y_ts[split:]

    # Baseline
    ridge_ts_raw = Ridge(alpha=1.0)
    ridge_ts_raw.fit(X_ts_train, y_ts_train)
    mse_raw = mean_squared_error(y_ts_test, ridge_ts_raw.predict(X_ts_test))

    for layer_idx in [0, n_layers // 2, n_layers]:
        feat_train = llm_reservoir(model, tok, device, X_ts_train, layer_idx, readout_dim)
        feat_test = llm_reservoir(model, tok, device, X_ts_test, layer_idx, readout_dim)

        ridge_ts = Ridge(alpha=1.0)
        ridge_ts.fit(feat_train, y_ts_train)
        mse_res = mean_squared_error(y_ts_test, ridge_ts.predict(feat_test))

        improvement = (mse_raw - mse_res) / max(mse_raw, 1e-10)
        print("  L%d: reservoir_MSE=%.6f, raw_MSE=%.6f, improvement=%.1f%%" %
              (layer_idx, mse_res, mse_raw, improvement * 100))

        all_results.append({
            'task': 'time_series',
            'layer': layer_idx,
            'reservoir_mse': round(float(mse_res), 6),
            'baseline_mse': round(float(mse_raw), 6),
            'improvement': round(float(improvement), 4),
        })

    # Summary
    n_improved = sum(1 for r in all_results if r['improvement'] > 0)
    avg_improvement = np.mean([r['improvement'] for r in all_results])

    if avg_improvement > 0.1:
        verdict = "QUANTUM RESERVOIR: avg %.0f%% improvement (%d/%d tasks)" % (
            avg_improvement * 100, n_improved, len(all_results))
    elif n_improved > 0:
        verdict = "PARTIAL RESERVOIR: %d/%d improved (avg %.1f%%)" % (
            n_improved, len(all_results), avg_improvement * 100)
    else:
        verdict = "NO RESERVOIR ADVANTAGE"

    print("\n--- Summary ---")
    print("  %s" % verdict)

    results = {
        'phase': 'Q231',
        'name': 'Quantum Reservoir Computing',
        'tests': all_results,
        'summary': {
            'n_improved': n_improved,
            'avg_improvement': round(float(avg_improvement), 4),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q231_reservoir.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Classification
    ax = axes[0]
    cls_data = [r for r in all_results if r['task'] == 'nonlinear_classification']
    layers = [r['layer'] for r in cls_data]
    res_acc = [r['reservoir_accuracy'] for r in cls_data]
    ax.bar(range(len(layers)), res_acc, color='#E91E63', label='Reservoir')
    ax.axhline(cls_data[0]['baseline_accuracy'], color='#607D8B', ls='--', label='Baseline')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(['L%d' % l for l in layers])
    ax.set_ylabel('Accuracy'); ax.set_title('(a) Nonlinear Classification')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Time series
    ax = axes[1]
    ts_data = [r for r in all_results if r['task'] == 'time_series']
    layers = [r['layer'] for r in ts_data]
    res_mse = [r['reservoir_mse'] for r in ts_data]
    ax.bar(range(len(layers)), res_mse, color='#2196F3', label='Reservoir')
    ax.axhline(ts_data[0]['baseline_mse'], color='#607D8B', ls='--', label='Baseline')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(['L%d' % l for l in layers])
    ax.set_ylabel('MSE'); ax.set_title('(b) Time Series Prediction')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q231: Quantum Reservoir Computing\n%s' % verdict[:60],
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q231_reservoir.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ231 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results

if __name__ == '__main__':
    main()
