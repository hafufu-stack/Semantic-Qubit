# -*- coding: utf-8 -*-
"""
Phase Q159: Model Scaling Law
===============================
Does quantum advantage scale with model size?
Compare Qwen2.5-0.5B vs Qwen2.5-1.5B on the same quantum tasks.

If advantage scales -> larger models = better quantum simulators
If no scaling -> it's just random structure, not learned
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_syk(n_qubits, seed=42):
    np.random.seed(seed)
    dim = 2 ** n_qubits
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I2 = np.eye(2)

    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    H = np.zeros((dim, dim))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            J = np.random.randn() / np.sqrt(n_qubits)
            ops = [I2] * n_qubits; ops[i] = Z; ops[j] = Z
            H += -J * kron_chain(ops)
            ops2 = [I2] * n_qubits; ops2[i] = X; ops2[j] = X
            H += -J * 0.5 * kron_chain(ops2)

    for i in range(n_qubits):
        ops = [I2] * n_qubits; ops[i] = X
        H += -0.3 * kron_chain(ops)

    return H


def rayleigh_gd(H, psi_init, max_steps=2000):
    psi = psi_init.copy() / np.linalg.norm(psi_init)
    lr = 0.01
    for step in range(max_steps):
        E = float(np.real(psi @ H @ psi))
        grad = 2 * (H @ psi - E * psi)
        psi_t = psi - lr * grad
        psi_t /= np.linalg.norm(psi_t)
        Et = float(np.real(psi_t @ H @ psi_t))
        if not np.isnan(Et) and Et < E:
            psi = psi_t
        else:
            lr *= 0.999
    return psi


def test_model(model_size, device='cpu'):
    """Test one model size and return results."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    if model_size == '0.5B':
        path = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                            "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")
    else:
        path = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                            "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")

    if not os.path.exists(path):
        print("  Model %s not found, skipping" % model_size)
        return None

    dtype = torch.float16 if device == 'cuda' else torch.float32
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map=device, local_files_only=True)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print("  %s: %d layers, hidden=%d" % (model_size, n_layers, hidden_size))

    prompt = "Quantum system ground state energy:"
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    results_per_size = []

    for n_q in [6, 8]:
        dim = 2 ** n_q
        if dim > hidden_size:
            continue

        H = build_syk(n_q)
        E_exact = float(np.linalg.eigvalsh(H)[0])

        # LLM basis
        llm_basis = []
        for li in range(0, n_layers, max(1, n_layers // 7)):
            h = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
            for offset in range(0, min(hidden_size, dim * 4), dim):
                if offset + dim <= hidden_size:
                    psi = h[offset:offset + dim].copy()
                    norm = np.linalg.norm(psi)
                    if norm > 1e-8:
                        llm_basis.append(psi / norm)

        if not llm_basis:
            continue

        scored = [(float(np.real(p @ H @ p)), p) for p in llm_basis]
        scored.sort(key=lambda x: x[0])
        best = scored[0][1].copy()

        top_k = min(10, len(scored))
        best_E = scored[0][0]
        for i in range(top_k):
            for j in range(i+1, top_k):
                mix = 0.5 * scored[i][1] + 0.5 * scored[j][1]
                n = np.linalg.norm(mix)
                if n > 1e-8:
                    mix /= n
                    Em = float(np.real(mix @ H @ mix))
                    if Em < best_E:
                        best_E = Em; best = mix.copy()

        psi_llm = rayleigh_gd(H, best)
        llm_err = abs(float(np.real(psi_llm @ H @ psi_llm)) - E_exact) * 1000

        # Random
        rand_errors = []
        for _ in range(5):
            psi_r = np.random.randn(dim); psi_r /= np.linalg.norm(psi_r)
            psi_f = rayleigh_gd(H, psi_r)
            rand_errors.append(abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000)

        rand_mean = float(np.mean(rand_errors))
        advantage = rand_mean / max(llm_err, 0.001)

        results_per_size.append({
            'n_qubits': int(n_q),
            'random_error': round(rand_mean, 4),
            'llm_error': round(llm_err, 4),
            'advantage': round(advantage, 2),
            'n_basis': len(llm_basis),
        })
        print("    N=%d: LLM=%.3f, Rand=%.3f -> %.1fx (basis=%d)" %
              (n_q, llm_err, rand_mean, advantage, len(llm_basis)))

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        'model_size': model_size,
        'n_layers': int(n_layers),
        'hidden_size': int(hidden_size),
        'results': results_per_size,
    }


def main():
    print("=" * 60)
    print("Phase Q159: Model Scaling Law")
    print("  (0.5B vs 1.5B)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = []

    for size in ['0.5B', '1.5B']:
        print("\n--- Testing %s ---" % size)
        result = test_model(size, device)
        if result:
            all_results.append(result)

    # Compare
    if len(all_results) == 2:
        print("\n--- Scaling Comparison ---")
        for nq in [6, 8]:
            r05 = next((r for r in all_results[0]['results'] if r['n_qubits'] == nq), None)
            r15 = next((r for r in all_results[1]['results'] if r['n_qubits'] == nq), None)
            if r05 and r15:
                print("  N=%d: 0.5B advantage=%.1fx, 1.5B advantage=%.1fx (%.1fx improvement)" %
                      (nq, r05['advantage'], r15['advantage'],
                       r15['advantage'] / max(r05['advantage'], 0.01)))

    # Save
    results = {
        'phase': 'Q159',
        'name': 'Model Scaling Law',
        'models': all_results,
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q159_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    if len(all_results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for qi, nq in enumerate([6, 8]):
            ax = axes[qi]
            sizes = []
            advs = []
            for mr in all_results:
                r = next((x for x in mr['results'] if x['n_qubits'] == nq), None)
                if r:
                    sizes.append(mr['model_size'])
                    advs.append(r['advantage'])
            colors = ['#FF9800', '#4CAF50']
            ax.bar(range(len(sizes)), advs, color=colors[:len(sizes)],
                   edgecolor='black', alpha=0.85)
            ax.axhline(1.0, color='red', ls='--')
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes)
            ax.set_ylabel('LLM Advantage (x)')
            ax.set_title('SYK N=%d qubits' % nq)
            ax.grid(alpha=0.3, axis='y')
            for i, v in enumerate(advs):
                ax.text(i, v + 0.05, '%.1fx' % v, ha='center', fontweight='bold')

        plt.suptitle('Q159: Model Scaling Law (0.5B vs 1.5B)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'phase_q159_scaling.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("\nQ159 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
