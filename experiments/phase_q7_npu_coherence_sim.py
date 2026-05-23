# -*- coding: utf-8 -*-
"""
Phase Q7: NPU Coherence Simulator
On AMD Ryzen AI (XDNA NPU), we can't do arbitrary DMA to GPU tensors directly.
However, we can SIMULATE the NPU's role as a "coherence engine" by:

1. Running a continuous oscillator (CfC-like) on CPU (simulating NPU's role)
2. The oscillator generates phase-modulated perturbation signals over time
3. These perturbations are injected into the LLM's forward pass to
   simulate "keeping the superposition from collapsing prematurely"

The key insight: real NPU would generate these signals at near-zero power
continuously. Here we simulate with CPU threads.

Experiment:
- Without coherence engine: superposition collapses quickly (classic result)
- With coherence engine (phase-modulated noise): does superposition last longer?
- Measure: entropy trajectory and collapse layer with/without engine
"""
import torch, json, os, gc, numpy as np, time, sys, threading
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


class CoherenceEngine:
    """
    Simulates NPU's continuous phase oscillator.
    Generates a time-varying perturbation vector that maintains
    the "phase" of a superposition state.
    """
    def __init__(self, hidden_size, freq=2.0, amplitude=0.05, seed=42):
        np.random.seed(seed)
        self.hidden_size = hidden_size
        self.freq = freq  # oscillation frequency (Hz analog)
        self.amplitude = amplitude
        # Generate a "coherence basis" - the direction to oscillate in
        self.basis_vec = torch.randn(hidden_size) * 0.1
        self.basis_vec = self.basis_vec / self.basis_vec.norm()
        self.t = 0.0
        self.running = False
        self._lock = threading.Lock()
        self._current_vec = torch.zeros(hidden_size)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        dt = 0.001  # 1ms steps
        while self.running:
            phase = self.amplitude * np.sin(2 * np.pi * self.freq * self.t)
            with self._lock:
                self._current_vec = self.basis_vec * phase
            self.t += dt
            time.sleep(dt)

    def get_perturbation(self):
        with self._lock:
            return self._current_vec.clone()


def train_soul(model, tok, data, device, layer=8, epochs=150, seed=42):
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
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[layer].register_forward_hook(hook)
            out = model(**inp); handle.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def run_with_engine(model, tok, prompt, device, base_vec, engine, layer,
                     inject_layers_range):
    """
    Forward pass with coherence engine perturbation added at each layer
    in inject_layers_range.
    """
    results_per_layer = {}

    def make_hook(li, bvec, eng):
        def hook(m, i, o):
            if isinstance(o, tuple):
                h = o[0].clone()
            else:
                h = o.clone()
            perturb = eng.get_perturbation().to(h.device).to(h.dtype)
            if li in inject_layers_range:
                inj = (bvec + perturb).to(h.dtype)
                if h.dim() == 3:
                    h[0, -1, :] = inj
                else:
                    h[-1, :] = inj
            if isinstance(o, tuple):
                return (h,) + o[1:]
            return h
        return hook

    handles = []
    for li in range(len(model.model.layers)):
        h = model.model.layers[li].register_forward_hook(make_hook(li, base_vec, engine))
        handles.append(h)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)

    for h in handles:
        h.remove()

    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    entropy = float(-(probs * (probs + 1e-12).log()).sum())
    top_tok = tok.decode(probs.argmax().item()).strip()
    return entropy, float(probs.max()), top_tok


def main():
    print("[Q7] NPU Coherence Simulator")
    start = time.time()
    model, tok = load_model(device=DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    hs = model.config.hidden_size

    min_data = [
        ("min(3,7)=", "3"), ("min(5,2)=", "2"), ("min(8,1)=", "1"),
        ("min(4,6)=", "4"), ("min(9,3)=", "3"),
    ]
    max_data = [
        ("min(3,7)=", "7"), ("min(5,2)=", "5"), ("min(8,1)=", "8"),
        ("min(4,6)=", "6"), ("min(9,3)=", "9"),
    ]

    print("  Training basis vectors...")
    min_vec = train_soul(model, tok, min_data, DEVICE, layer=8, seed=42)
    max_vec = train_soul(model, tok, max_data, DEVICE, layer=8, seed=99)
    super_vec = (min_vec + max_vec) / np.sqrt(2)

    test_prompts = ["min(7,2)=", "min(6,3)=", "min(2,9)=", "min(1,5)=", "min(8,4)="]
    tok_min_ids = [tok.encode('2')[-1], tok.encode('3')[-1], tok.encode('2')[-1],
                   tok.encode('1')[-1], tok.encode('4')[-1]]
    tok_max_ids = [tok.encode('7')[-1], tok.encode('6')[-1], tok.encode('9')[-1],
                   tok.encode('5')[-1], tok.encode('8')[-1]]

    # Experiment 1: Coherence engine at different frequencies
    frequencies = [0.5, 1.0, 2.0, 4.0, 8.0]
    amplitudes = [0.01, 0.05, 0.1, 0.2]
    inject_range = {4, 5, 6, 7, 8}  # inject around layer 8

    print("  Baseline: superposition injection, no coherence engine...")
    baseline_entropies = []
    baseline_p_correct = []
    for prompt, tid in zip(test_prompts, tok_min_ids):
        def hook(m, i, o, v=super_vec):
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
            h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
        handle = model.model.layers[8].register_forward_hook(hook)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        handle.remove()
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        baseline_entropies.append(float(-(probs * (probs + 1e-12).log()).sum()))
        baseline_p_correct.append(float(probs[tid]))
    baseline_entropy_avg = np.mean(baseline_entropies)
    baseline_p_correct_avg = np.mean(baseline_p_correct)
    print("  Baseline entropy: %.4f, P(correct): %.4f" % (baseline_entropy_avg, baseline_p_correct_avg))

    print("  Testing coherence engine (freq sweep)...")
    freq_results = {}
    for freq in frequencies:
        engine = CoherenceEngine(hs, freq=freq, amplitude=0.05, seed=42)
        engine.start()
        time.sleep(0.1)  # let engine spin up

        entropies, p_corrects = [], []
        for prompt, tid in zip(test_prompts, tok_min_ids):
            ent, pmax, top = run_with_engine(
                model, tok, prompt, DEVICE, super_vec, engine, 8, inject_range)
            entropies.append(ent)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            perturb = engine.get_perturbation().to(DEVICE)
            inj_vec = (super_vec + perturb.to(super_vec.dtype))
            def hook(m, i, o, v=inj_vec):
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[8].register_forward_hook(hook)
            with torch.no_grad():
                out_simple = model(**inp)
            handle.remove()
            probs_simple = torch.softmax(out_simple.logits[0, -1, :].float(), dim=-1)
            p_corrects.append(float(probs_simple[tid]))

        engine.stop()
        avg_ent = np.mean(entropies)
        avg_pc = np.mean(p_corrects)
        freq_results[freq] = {
            'avg_entropy': round(float(avg_ent), 4),
            'avg_p_correct': round(float(avg_pc), 4),
            'entropy_delta': round(float(avg_ent - baseline_entropy_avg), 4),
        }
        print("    freq=%.1f: entropy=%.4f (delta=%+.4f), P(correct)=%.4f" % (
            freq, avg_ent, avg_ent - baseline_entropy_avg, avg_pc))

    print("  Testing coherence engine (amplitude sweep)...")
    amp_results = {}
    for amp in amplitudes:
        engine = CoherenceEngine(hs, freq=2.0, amplitude=amp, seed=42)
        engine.start()
        time.sleep(0.1)

        entropies, p_corrects = [], []
        for prompt, tid in zip(test_prompts, tok_min_ids):
            ent, _, _ = run_with_engine(
                model, tok, prompt, DEVICE, super_vec, engine, 8, inject_range)
            entropies.append(ent)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            perturb = engine.get_perturbation().to(DEVICE)
            inj_vec = (super_vec + perturb.to(super_vec.dtype))
            def hook(m, i, o, v=inj_vec):
                if isinstance(o, tuple):
                    h = o[0].clone(); h[0, -1, :] = v.to(h.dtype); return (h,) + o[1:]
                h = o.clone(); h[0, -1, :] = v.to(h.dtype); return h
            handle = model.model.layers[8].register_forward_hook(hook)
            with torch.no_grad():
                out_simple = model(**inp)
            handle.remove()
            probs_simple = torch.softmax(out_simple.logits[0, -1, :].float(), dim=-1)
            p_corrects.append(float(probs_simple[tid]))
        engine.stop()

        avg_ent = np.mean(entropies)
        avg_pc = np.mean(p_corrects)
        amp_results[amp] = {
            'avg_entropy': round(float(avg_ent), 4),
            'avg_p_correct': round(float(avg_pc), 4),
            'entropy_delta': round(float(avg_ent - baseline_entropy_avg), 4),
        }
        print("    amp=%.2f: entropy=%.4f (delta=%+.4f), P(correct)=%.4f" % (
            amp, avg_ent, avg_ent - baseline_entropy_avg, avg_pc))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Frequency sweep
    ax = axes[0]
    freqs = list(freq_results.keys())
    ent_deltas = [freq_results[f]['entropy_delta'] for f in freqs]
    p_corrects_freq = [freq_results[f]['avg_p_correct'] for f in freqs]
    ax2 = ax.twinx()
    ax.bar(range(len(freqs)), ent_deltas, color='#9C27B0', alpha=0.6, edgecolor='black',
           label='Entropy delta')
    ax2.plot(range(len(freqs)), p_corrects_freq, 'ro-', lw=2, label='P(correct)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels(['%.1fHz' % f for f in freqs])
    ax.set_xlabel('Coherence Engine Frequency')
    ax.set_ylabel('Entropy delta vs baseline', color='#9C27B0')
    ax2.set_ylabel('P(correct)', color='red')
    ax.set_title('Frequency Sweep\n(NPU Coherence Engine sim)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Amplitude sweep
    ax = axes[1]
    amps = list(amp_results.keys())
    ent_deltas_amp = [amp_results[a]['entropy_delta'] for a in amps]
    p_corrects_amp = [amp_results[a]['avg_p_correct'] for a in amps]
    ax3 = ax.twinx()
    ax.bar(range(len(amps)), ent_deltas_amp, color='#FF5722', alpha=0.6, edgecolor='black')
    ax3.plot(range(len(amps)), p_corrects_amp, 'bs-', lw=2, label='P(correct)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(amps)))
    ax.set_xticklabels(['amp=%.2f' % a for a in amps])
    ax.set_xlabel('Coherence Engine Amplitude')
    ax.set_ylabel('Entropy delta vs baseline', color='#FF5722')
    ax3.set_ylabel('P(correct)', color='blue')
    ax.set_title('Amplitude Sweep\n(NPU Coherence Engine sim)', fontweight='bold')
    ax3.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Summary
    ax = axes[2]
    ax.axis('off')
    best_freq = max(freq_results, key=lambda f: freq_results[f]['entropy_delta'])
    best_amp = max(amp_results, key=lambda a: amp_results[a]['entropy_delta'])
    summary = (
        "NPU Coherence Engine Simulation\n\n"
        "Baseline (no engine):\n"
        "  Entropy: %.4f\n"
        "  P(correct): %.4f\n\n"
        "Best freq: %.1f Hz\n"
        "  Entropy delta: %+.4f\n"
        "  P(correct): %.4f\n\n"
        "Best amplitude: %.2f\n"
        "  Entropy delta: %+.4f\n"
        "  P(correct): %.4f\n\n"
        "Positive entropy delta =\n"
        "  coherence maintained!\n"
        "(Engine keeps superposition\n"
        " alive longer)"
    ) % (
        baseline_entropy_avg, baseline_p_correct_avg,
        best_freq, freq_results[best_freq]['entropy_delta'],
        freq_results[best_freq]['avg_p_correct'],
        best_amp, amp_results[best_amp]['entropy_delta'],
        amp_results[best_amp]['avg_p_correct'],
    )
    ax.text(0.05, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#E8EAF6', alpha=0.9))

    plt.suptitle(
        'Phase Q7: NPU Coherence Simulator\nPhase-Modulated Perturbation Engine',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q7_npu_coherence_sim.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'Q7', 'name': 'npu_coherence_sim',
        'baseline': {'entropy': float(baseline_entropy_avg), 'p_correct': float(baseline_p_correct_avg)},
        'freq_sweep': {str(k): v for k, v in freq_results.items()},
        'amp_sweep': {str(k): v for k, v in amp_results.items()},
        'best_freq': float(best_freq),
        'best_amp': float(best_amp),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q7_npu_coherence_sim.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Q7 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
