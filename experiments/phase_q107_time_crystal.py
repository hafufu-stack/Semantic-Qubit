# -*- coding: utf-8 -*-
"""Phase Q107: Discrete Time Crystal in Transformers
Test if cyclically repeating the same input through the model
produces output patterns with a DIFFERENT periodicity than the input,
analogous to a discrete time crystal (DTC).
A DTC breaks discrete time-translation symmetry:
  - Input has period T
  - Output has period 2T (or nT)
GPU experiment.
"""
import json, os, time, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model


def measure_time_crystal(model, tokenizer, num_layers):
    """Apply periodic perturbations and check if output oscillates
    at a different frequency (period doubling = DTC)."""

    d_model = model.config.hidden_size
    prompt = "The nature of time and space in quantum mechanics is"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Reference
    with torch.no_grad():
        ref_out = model(**inputs)
        ref_logits = ref_out.logits[0, -1, :500].cpu().float().numpy()

    # Create periodic perturbation: +sv, -sv, +sv, -sv, ... (period=2)
    np.random.seed(107)
    sv = np.random.randn(d_model).astype(np.float32)
    sv /= np.linalg.norm(sv)

    # Apply at each layer and record output difference
    num_periods = 6  # Test 6 "cycles"
    input_signal = []  # +1 or -1 (the drive)
    output_signal = []  # cosine similarity to perturbation direction

    sv_tensor = torch.tensor(sv, device=model.device)

    for cycle in range(num_periods * 2):  # 2 steps per period
        sign = 1 if cycle % 2 == 0 else -1  # Period-2 drive
        input_signal.append(sign)

        # Apply perturbation at middle layer
        mid = num_layers // 2
        applied = [False]
        def hook(module, args, output, s=sign, app=applied):
            if not app[0]:
                app[0] = True
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += s * sv_tensor.to(hs.dtype)
                    else:
                        hs[-1, :] += s * sv_tensor.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += s * sv_tensor.to(hs.dtype)
                    else:
                        hs[-1, :] += s * sv_tensor.to(hs.dtype)
                    return hs
            return output

        handle = model.model.layers[mid].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :500].cpu().float().numpy()
        handle.remove()

        # Output signal: how does output respond?
        diff = logits - ref_logits
        output_response = np.dot(diff, ref_logits) / (np.linalg.norm(diff) * np.linalg.norm(ref_logits) + 1e-10)
        output_signal.append(float(output_response))

    # Fourier analysis of output signal
    input_arr = np.array(input_signal)
    output_arr = np.array(output_signal)

    # FFT
    fft_input = np.abs(np.fft.rfft(input_arr))
    fft_output = np.abs(np.fft.rfft(output_arr))
    freqs = np.fft.rfftfreq(len(input_arr))

    # Detect period doubling: dominant output frequency != input frequency
    input_peak = np.argmax(fft_input[1:]) + 1  # skip DC
    output_peak = np.argmax(fft_output[1:]) + 1

    is_dtc = output_peak != input_peak
    period_ratio = freqs[input_peak] / (freqs[output_peak] + 1e-10) if output_peak > 0 else 0

    # Also test with period-3 drive
    input_signal_3 = []
    output_signal_3 = []
    for cycle in range(num_periods * 3):
        pattern = [1, 1, -1]  # Period-3 drive
        sign = pattern[cycle % 3]
        input_signal_3.append(sign)

        applied = [False]
        def hook3(module, args, output, s=sign, app=applied):
            if not app[0]:
                app[0] = True
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += s * sv_tensor.to(hs.dtype)
                    else:
                        hs[-1, :] += s * sv_tensor.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    if hs.dim() == 3:
                        hs[0, -1, :] += s * sv_tensor.to(hs.dtype)
                    else:
                        hs[-1, :] += s * sv_tensor.to(hs.dtype)
                    return hs
            return output

        handle = model.model.layers[mid].register_forward_hook(hook3)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :500].cpu().float().numpy()
        handle.remove()

        diff = logits - ref_logits
        response = np.dot(diff, ref_logits) / (np.linalg.norm(diff) * np.linalg.norm(ref_logits) + 1e-10)
        output_signal_3.append(float(response))

    fft_in3 = np.abs(np.fft.rfft(np.array(input_signal_3)))
    fft_out3 = np.abs(np.fft.rfft(np.array(output_signal_3)))

    return {
        'input_signal': input_signal,
        'output_signal': output_signal,
        'fft_input': fft_input.tolist(),
        'fft_output': fft_output.tolist(),
        'input_peak_freq': int(input_peak),
        'output_peak_freq': int(output_peak),
        'is_dtc': bool(is_dtc),
        'period_ratio': float(period_ratio),
        'input_signal_3': input_signal_3,
        'output_signal_3': output_signal_3,
        'fft_in3': fft_in3.tolist(),
        'fft_out3': fft_out3.tolist(),
        'freqs': freqs.tolist(),
    }


def main():
    print("=" * 60)
    print("Phase Q107: Discrete Time Crystal in Transformers")
    print("  Does periodic drive -> subharmonic response?")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model()
    num_layers = model.config.num_hidden_layers

    results = measure_time_crystal(model, tokenizer, num_layers)

    print("\n  === Time Crystal Results ===")
    print("  Input drive peak frequency idx: %d" % results['input_peak_freq'])
    print("  Output response peak frequency idx: %d" % results['output_peak_freq'])
    print("  Period ratio: %.3f" % results['period_ratio'])
    print("  Discrete Time Crystal: %s" %
          ('DETECTED!' if results['is_dtc'] else 'Not detected'))

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Time domain - period 2
    ax = axes[0][0]
    t = np.arange(len(results['input_signal']))
    ax.step(t, results['input_signal'], 'b-', linewidth=2, label='Input drive',
            where='mid', alpha=0.7)
    ax.plot(t, results['output_signal'], 'ro-', linewidth=2,
            markersize=5, label='Output response')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Signal', fontsize=11)
    ax.set_title('(a) Period-2 Drive\nInput vs Output',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (b) Frequency domain - period 2
    ax = axes[0][1]
    freqs = results['freqs']
    ax.bar(np.arange(len(results['fft_input'])) - 0.15, results['fft_input'],
           width=0.3, color='blue', alpha=0.7, label='Input FFT')
    ax.bar(np.arange(len(results['fft_output'])) + 0.15, results['fft_output'],
           width=0.3, color='red', alpha=0.7, label='Output FFT')
    ax.set_xlabel('Frequency bin', fontsize=11)
    ax.set_ylabel('|FFT|', fontsize=11)
    ax.set_title('(b) Frequency Spectrum\nDTC = peaks at different frequencies',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # (c) Time domain - period 3
    ax = axes[1][0]
    t3 = np.arange(len(results['input_signal_3']))
    ax.step(t3, results['input_signal_3'], 'b-', linewidth=2,
            label='Input drive (period 3)', where='mid', alpha=0.7)
    ax.plot(t3, results['output_signal_3'], 'ro-', linewidth=2,
            markersize=4, label='Output response')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Signal', fontsize=11)
    ax.set_title('(c) Period-3 Drive\nInput vs Output',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (d) Summary
    ax = axes[1][1]
    dtc_color = '#4CAF50' if results['is_dtc'] else '#FF9800'
    ax.text(0.5, 0.65,
            'TIME\nCRYSTAL',
            ha='center', va='center', fontsize=24, fontweight='bold',
            color=dtc_color, transform=ax.transAxes)
    ax.text(0.5, 0.35,
            'Period ratio: %.2f\n'
            'DTC detected: %s\n\n'
            'Transformers %s\n'
            'discrete time-translation symmetry' % (
                results['period_ratio'],
                'YES' if results['is_dtc'] else 'NO',
                'BREAK' if results['is_dtc'] else 'preserve'),
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('(d) Verdict', fontsize=12, fontweight='bold')

    plt.suptitle('Q107: Discrete Time Crystal in Transformer Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'phase_q107_time_crystal.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results['phase'] = 'Q107'
    results['name'] = 'Discrete Time Crystal'
    results['elapsed'] = elapsed
    res_path = os.path.join(RESULTS_DIR, 'phase_q107_time_crystal.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved. Elapsed: %.1fs" % elapsed)
    return results


if __name__ == '__main__':
    main()
