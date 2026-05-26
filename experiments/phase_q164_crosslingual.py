# -*- coding: utf-8 -*-
"""
Phase Q164: Cross-Lingual Quantum States
==========================================
Do Japanese prompts give different quantum advantages than English?
Tests UNIVERSALITY of semantic quantum computing.

If language-independent -> it's the MODEL's structure, not the words
If language-dependent -> it's the SEMANTICS that matters
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

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
        r = ops[0]
        for o in ops[1:]: r = np.kron(r, o)
        return r
    H = np.zeros((dim, dim))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            J = np.random.randn() / np.sqrt(n_qubits)
            ops = [I2]*n_qubits; ops[i] = Z; ops[j] = Z
            H += -J * kron_chain(ops)
    for i in range(n_qubits):
        ops = [I2]*n_qubits; ops[i] = X
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


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 60)
    print("Phase Q164: Cross-Lingual Quantum States")
    print("  (English vs Japanese vs Chinese)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Parallel prompts in 3 languages
    prompt_groups = [
        {
            'concept': 'quantum_ground_state',
            'en': "The ground state energy of the quantum system:",
            'ja': "量子系の基底状態エネルギー：",
            'zh': "量子系统的基态能量：",
        },
        {
            'concept': 'hydrogen_atom',
            'en': "The hydrogen atom has one electron orbiting:",
            'ja': "水素原子には一つの電子が軌道を回る：",
            'zh': "氢原子有一个电子在轨道上：",
        },
        {
            'concept': 'chemical_bond',
            'en': "Chemical bond formation releases energy:",
            'ja': "化学結合の形成はエネルギーを放出する：",
            'zh': "化学键的形成释放能量：",
        },
        {
            'concept': 'everyday',
            'en': "The weather today is very nice:",
            'ja': "今日の天気はとても良いです：",
            'zh': "今天的天气非常好：",
        },
    ]

    n_qubits = 8
    dim = 2 ** n_qubits
    H = build_syk(n_qubits, seed=42)
    E_exact = float(np.linalg.eigvalsh(H)[0])

    all_results = []

    for group in prompt_groups:
        concept = group['concept']
        print("\n--- Concept: %s ---" % concept)

        lang_hidden = {}
        lang_errors = {}

        for lang in ['en', 'ja', 'zh']:
            prompt = group[lang]
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            # Collect basis
            llm_basis = []
            for li in range(0, n_layers, 4):
                h = out.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
                for offset in range(0, min(hidden_size, dim * 4), dim):
                    if offset + dim <= hidden_size:
                        psi = h[offset:offset + dim].copy()
                        norm = np.linalg.norm(psi)
                        if norm > 1e-8:
                            llm_basis.append(psi / norm)

            if not llm_basis:
                lang_errors[lang] = 999.0
                continue

            scored = [(float(np.real(p @ H @ p)), p) for p in llm_basis]
            scored.sort(key=lambda x: x[0])
            best = scored[0][1].copy()

            psi_f = rayleigh_gd(H, best)
            err = abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000
            lang_errors[lang] = err

            # Save final hidden for cross-lingual similarity
            lang_hidden[lang] = out.hidden_states[-1][0, -1, :].float().cpu().numpy()

            print("  %s: %.3f mHa (prompt len=%d)" % (lang, err, len(prompt)))

        # Cross-lingual hidden state similarity
        sims = {}
        for l1 in ['en', 'ja', 'zh']:
            for l2 in ['en', 'ja', 'zh']:
                if l1 < l2 and l1 in lang_hidden and l2 in lang_hidden:
                    sim = cosine_sim(lang_hidden[l1], lang_hidden[l2])
                    sims['%s-%s' % (l1, l2)] = round(sim, 4)
                    print("    %s vs %s: cos=%.4f" % (l1, l2, sim))

        result = {
            'concept': concept,
            'errors': {k: round(v, 4) for k, v in lang_errors.items()},
            'cross_sims': sims,
        }
        all_results.append(result)

    # Random baseline
    rand_errors = []
    for _ in range(10):
        psi_r = np.random.randn(dim); psi_r /= np.linalg.norm(psi_r)
        psi_f = rayleigh_gd(H, psi_r)
        rand_errors.append(abs(float(np.real(psi_f @ H @ psi_f)) - E_exact) * 1000)
    rand_mean = float(np.mean(rand_errors))

    # Summary
    print("\n--- Cross-Lingual Summary ---")
    for lang in ['en', 'ja', 'zh']:
        errs = [r['errors'].get(lang, 999) for r in all_results
                if r['concept'] != 'everyday']
        avg = float(np.mean(errs))
        print("  %s avg (science): %.2f mHa" % (lang, avg))
    print("  Random mean: %.2f mHa" % rand_mean)

    # Save
    results = {
        'phase': 'Q164',
        'name': 'Cross-Lingual Quantum States',
        'results': all_results,
        'random_mean': round(rand_mean, 4),
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q164_crosslingual.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    concepts = [r['concept'] for r in all_results]
    x = np.arange(len(concepts))
    w = 0.25
    for i, (lang, color) in enumerate([('en', '#2196F3'), ('ja', '#E91E63'), ('zh', '#4CAF50')]):
        vals = [r['errors'].get(lang, 0) for r in all_results]
        ax.bar(x + i * w, vals, w, color=color, label=lang.upper(), alpha=0.85)
    ax.axhline(rand_mean, color='gray', ls='--', label='Random')
    ax.set_xticks(x + w)
    ax.set_xticklabels(concepts, fontsize=7, rotation=15)
    ax.set_ylabel('Error (mHa)')
    ax.set_title('(a) Quantum Advantage by Language')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    # Cross-lingual similarity heatmap
    science_sims = {'en-ja': [], 'en-zh': [], 'ja-zh': []}
    for r in all_results:
        for pair, val in r['cross_sims'].items():
            science_sims[pair].append(val)
    pairs = list(science_sims.keys())
    avg_sims = [float(np.mean(v)) for v in science_sims.values()]
    ax.bar(range(len(pairs)), avg_sims, color=['#9C27B0', '#FF9800', '#00BCD4'],
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs)
    ax.set_ylabel('Avg Cosine Similarity')
    ax.set_title('(b) Cross-Lingual Hidden State Similarity')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.suptitle('Q164: Cross-Lingual Quantum States (EN vs JA vs ZH)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q164_crosslingual.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ164 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
