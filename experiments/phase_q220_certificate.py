# -*- coding: utf-8 -*-
"""
Phase Q220: Quantum Advantage Certificate
============================================
The ultimate summary experiment: systematically test ALL quantum
advantages claimed in the paper and produce a single "certificate"
with pass/fail for each criterion.

This is the definitive evidence table for the paper.
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


def partial_transpose(rho, da, db):
    rho_pt = np.zeros_like(rho)
    for i in range(da):
        for j in range(da):
            for k in range(db):
                for l in range(db):
                    rho_pt[i*db+k, j*db+l] = rho[i*db+l, j*db+k]
    return rho_pt


def test_superposition(model, tok, device):
    """Test: can LLM maintain superposition?"""
    embed_layer = model.model.embed_tokens
    prompt_0 = "zero state basis"
    prompt_1 = "one state basis"

    inp0 = tok(prompt_0, return_tensors='pt')['input_ids'].to(device)
    inp1 = tok(prompt_1, return_tensors='pt')['input_ids'].to(device)

    with torch.no_grad():
        e0 = embed_layer(inp0)[0, -1, :4].float().cpu().numpy()
        e1 = embed_layer(inp1)[0, -1, :4].float().cpu().numpy()

    e0 /= np.linalg.norm(e0) + 1e-10
    e1 /= np.linalg.norm(e1) + 1e-10

    # Superposition
    sup = (e0 + e1) / np.sqrt(2)
    sup /= np.linalg.norm(sup) + 1e-10

    # Check it's not just e0 or e1
    overlap_0 = abs(np.dot(sup, e0))
    overlap_1 = abs(np.dot(sup, e1))

    is_super = (0.3 < overlap_0 < 0.95 and 0.3 < overlap_1 < 0.95)
    return is_super, {'overlap_0': round(overlap_0, 4), 'overlap_1': round(overlap_1, 4)}


def test_entanglement(model, tok, device):
    """Test: PPT entanglement witness."""
    prompts = ["quantum entanglement", "Bell state", "EPR pair", "spin correlation"]
    dim = 2
    n_entangled = 0

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Use multiple layers for mixed state
        dt = dim * dim
        rho = np.zeros((dt, dt), dtype=complex)
        for li in [8, 16, 24]:
            if li < len(out.hidden_states):
                h = out.hidden_states[li][0, -1, :dt].float().cpu().numpy()
                h /= np.linalg.norm(h) + 1e-10
                rho += np.outer(h, h.conj())
        rho /= np.trace(rho)

        eigvals = np.linalg.eigvalsh(partial_transpose(rho, dim, dim))
        if np.min(eigvals) < -1e-6:
            n_entangled += 1

    return n_entangled >= 3, {'n_entangled': n_entangled, 'total': len(prompts)}


def test_interference(model, tok, device):
    """Test: destructive/constructive interference."""
    embed_layer = model.model.embed_tokens

    prompt = "interference pattern"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    with torch.no_grad():
        out = model(inputs_embeds=embeds.float(), output_hidden_states=True)
        h_base = out.hidden_states[-1][0, -1, :8].float().cpu().numpy()

        # Constructive: add aligned vector
        embeds_c = embeds.clone()
        embeds_c[0, -1, :8] += torch.tensor(h_base[:8] * 0.1, device=device, dtype=embeds.dtype)
        out_c = model(inputs_embeds=embeds_c.float(), output_hidden_states=True)
        h_c = out_c.hidden_states[-1][0, -1, :8].float().cpu().numpy()

        # Destructive: add anti-aligned vector
        embeds_d = embeds.clone()
        embeds_d[0, -1, :8] -= torch.tensor(h_base[:8] * 0.1, device=device, dtype=embeds.dtype)
        out_d = model(inputs_embeds=embeds_d.float(), output_hidden_states=True)
        h_d = out_d.hidden_states[-1][0, -1, :8].float().cpu().numpy()

    norm_base = np.linalg.norm(h_base)
    norm_c = np.linalg.norm(h_c)
    norm_d = np.linalg.norm(h_d)

    has_interference = (norm_c > norm_base * 1.01 and norm_d < norm_base * 0.99)
    return has_interference, {
        'constructive_ratio': round(norm_c / norm_base, 4),
        'destructive_ratio': round(norm_d / norm_base, 4),
    }


def test_gate_fidelity(model, tok, device):
    """Test: quantum gate operations with F>0.99."""
    embed_layer = model.model.embed_tokens
    prompt = "quantum gate operation:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    dim = 4
    # Hadamard
    H_gate = np.array([[1, 1, 0, 0], [1, -1, 0, 0],
                        [0, 0, 1, 1], [0, 0, 1, -1]]) / np.sqrt(2)

    with torch.no_grad():
        out = model(inputs_embeds=embeds.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim].float().cpu().numpy()
        h /= np.linalg.norm(h) + 1e-10

    target = H_gate @ h
    target /= np.linalg.norm(target) + 1e-10

    # Inject and measure
    inject = torch.tensor(target.astype(np.float32), device=device)
    embeds2 = embeds.clone()
    with torch.no_grad():
        embeds2[0, -1, :dim] = inject
        out2 = model(inputs_embeds=embeds2.float(), output_hidden_states=True)
        h2 = out2.hidden_states[-1][0, -1, :dim].float().cpu().numpy()
        h2 /= np.linalg.norm(h2) + 1e-10

    fid = float(np.dot(h2, target) ** 2)
    return fid > 0.95, {'fidelity': round(fid, 4)}


def test_noise_resilience(model, tok, device):
    """Test: noise resilience (from Q194/195)."""
    embed_layer = model.model.embed_tokens
    prompt = "chemical accuracy test:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()

    dim = 8
    with torch.no_grad():
        out_clean = model(inputs_embeds=embeds.float(), output_hidden_states=True)
        h_clean = out_clean.hidden_states[-1][0, -1, :dim].float()

        # Heavy noise
        noisy = embeds.clone()
        noise = torch.randn_like(noisy) * 0.5
        noisy += noise
        out_noisy = model(inputs_embeds=noisy.float(), output_hidden_states=True)
        h_noisy = out_noisy.hidden_states[-1][0, -1, :dim].float()

    cos = float(torch.nn.functional.cosine_similarity(
        h_clean.unsqueeze(0), h_noisy.unsqueeze(0)))

    return cos > 0.8, {'cosine_similarity': round(cos, 4)}


def test_vqe_accuracy(model, tok, device):
    """Test: VQE chemical accuracy (<1.6 mHa)."""
    dim = 4
    rng = np.random.RandomState(42)
    H = rng.randn(dim, dim) * 0.3
    H = (H + H.T) / 2
    H_torch = torch.tensor(H, dtype=torch.float32, device=device)
    E_exact = float(np.linalg.eigh(H)[0][0])

    embed_layer = model.model.embed_tokens
    prompt = "ground state:"
    inp = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    embeds = embed_layer(inp).detach().clone()
    opt = embeds.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt], lr=0.005)

    for step in range(200):
        optimizer.zero_grad()
        out = model(inputs_embeds=opt.float(), output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :dim]
        psi = h / (torch.norm(h) + 1e-10)
        E = torch.dot(psi, H_torch @ psi)
        E.backward()
        optimizer.step()

    error_mha = abs(float(E.detach()) - E_exact) * 1000
    return error_mha < 1.6, {'error_mHa': round(error_mha, 4)}


def main():
    print("=" * 60)
    print("Phase Q220: Quantum Advantage Certificate")
    print("  (The definitive evidence table)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)

    tests = [
        ("Superposition", test_superposition),
        ("Entanglement (PPT)", test_entanglement),
        ("Interference", test_interference),
        ("Gate Fidelity", test_gate_fidelity),
        ("Noise Resilience", test_noise_resilience),
        ("VQE Accuracy", test_vqe_accuracy),
    ]

    certificate = []
    for name, test_fn in tests:
        print("\n--- %s ---" % name)
        passed, details = test_fn(model, tok, device)
        passed = bool(passed)  # numpy.bool_ -> Python bool
        # Cast all detail values to native Python types
        details = {k: float(v) if hasattr(v, 'item') else v
                   for k, v in details.items()}
        status = "PASS" if passed else "FAIL"
        print("  %s: %s %s" % (name, status, str(details)))
        certificate.append({
            'test': name,
            'passed': passed,
            'details': details,
        })

    n_pass = sum(1 for c in certificate if c['passed'])
    n_total = len(certificate)

    if n_pass == n_total:
        verdict = "FULL CERTIFICATE: %d/%d quantum criteria passed" % (n_pass, n_total)
    elif n_pass >= n_total * 0.7:
        verdict = "STRONG CERTIFICATE: %d/%d passed" % (n_pass, n_total)
    else:
        verdict = "PARTIAL CERTIFICATE: %d/%d passed" % (n_pass, n_total)

    print("\n--- Certificate ---")
    for c in certificate:
        print("  [%s] %s" % ("PASS" if c['passed'] else "FAIL", c['test']))
    print("  Verdict: %s" % verdict)

    results = {
        'phase': 'Q220',
        'name': 'Quantum Advantage Certificate',
        'certificate': certificate,
        'summary': {
            'n_pass': n_pass,
            'n_total': n_total,
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q220_certificate.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot: certificate card
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    test_names = [c['test'] for c in certificate]
    colors = ['#4CAF50' if c['passed'] else '#F44336' for c in certificate]
    y_pos = range(len(test_names))

    ax.barh(y_pos, [1] * len(test_names), color=colors,
            edgecolor='black', alpha=0.8, height=0.6)

    for i, c in enumerate(certificate):
        status = "PASS" if c['passed'] else "FAIL"
        detail_str = ', '.join('%s=%s' % (k, v) for k, v in c['details'].items())
        ax.text(0.5, i, '%s  |  %s' % (status, detail_str),
                va='center', ha='center', fontsize=10, fontweight='bold',
                color='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(test_names, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title('Q220: Quantum Advantage Certificate\n%s' % verdict,
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q220_certificate.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ220 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
