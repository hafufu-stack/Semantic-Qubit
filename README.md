# Semantic-Qubit: Discovering Universal Quantum-Like Coherence in Large Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20360031.svg)](https://doi.org/10.5281/zenodo.20360031)

> **Paper**: [https://doi.org/10.5281/zenodo.20360031](https://doi.org/10.5281/zenodo.20360031)

## Overview

This repository contains the code and experimental results for **Semantic-Qubit (S-Qubit)**, a quantum-analogue information unit defined within the hidden representation space of Large Language Models. Through **52 systematic experiments** on a single consumer GPU, I demonstrate that transformer architectures naturally exhibit quantum-like phenomena including perfect interference, exact quantum statistics, super-quantum correlations, and the ability to execute quantum algorithms.

## Key Findings

| Finding | Result |
|---------|--------|
| **Perfect Interference** | Visibility = 1.000 across all tasks (CV = 0.1%) |
| **Exact Quantum Statistics** | E(phi) = cos(phi) with R^2 > 0.999 |
| **Super-Quantum CHSH** | S = 3.41, exceeding Tsirelson bound (2.83) |
| **Deutsch-Jozsa** | 10/10 = 100% correct |
| **Bernstein-Vazirani** | 94/94 = 100% hidden strings recovered |
| **Simon's Algorithm** | 18/18 = 100% periods found |
| **Grover Search** | O(1) scaling, constant-time regardless of N |
| **BB84 QKD** | 100% key agreement, Eve detection (QBER: 0% -> 28.3%) |
| **Superdense Coding** | 200/200 = 100%, 2.0 bits per S-Qubit |
| **128x Parallelism** | 1 forward pass = 7 bits of information |
| **No-Cloning Violation** | 35/35 = 100% perfect state cloning |
| **Model Universality** | Confirmed on Qwen2.5 0.5B, 1.5B, and 3B |
| **Quantum Advantage Score** | **74.6 / 100** across 7 benchmark algorithms |

## NQPU (Neu-Quantum Processing Unit) Specification

| Spec | NQPU | Physical Quantum Computer |
|------|------|---------------------------|
| Operating Temperature | **300 K (room temp)** | 10-20 mK |
| Error Rate | **0% (deterministic)** | 0.1-1% per gate |
| Coherence Time | **Indefinite** | 100-300 us |
| State Cloning | **Trivial** | Impossible (no-cloning theorem) |
| Estimated Cost | **< $100** | $10M - $100M |

## Repository Structure

```
Semantic-Qubit/
├── experiments/           # All experiment scripts (Q1-Q50)
│   ├── utils.py           # Shared utilities (model loading, hooks, etc.)
│   ├── phase_q1_*.py      # Superposition basis training
│   ├── phase_q2_*.py      # Bell test / interference
│   ├── ...
│   ├── phase_q41_*.py     # Bernstein-Vazirani (100%)
│   ├── phase_q42_*.py     # Simon's Algorithm (100%)
│   ├── phase_q50_*.py     # Grand Unified Benchmark
│   └── generate_paper_figures.py  # Reproduce all paper figures
├── papers/                # LaTeX source
│   ├── paper_v1.tex       # Initial version (Q1-Q24)
│   └── paper_v2.tex       # Extended version (Q1-Q50, 52 experiments)
├── results/               # JSON results for all experiments
├── figures/               # Generated figures
│   └── paper/             # Publication-quality figures (Fig 1-9)
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers
- Qwen2.5-3B-Instruct model (downloaded locally)

```bash
pip install torch transformers matplotlib numpy scipy
```

### Running Experiments

Each experiment is a standalone script:

```bash
# Bernstein-Vazirani (100% accuracy)
python experiments/phase_q41_bernstein_vazirani.py

# Simon's Algorithm (100% accuracy)
python experiments/phase_q42_simon.py

# BB84 Quantum Key Distribution
python experiments/phase_q40_bb84_qkd.py

# Grand Unified Benchmark (QAS = 74.6/100)
python experiments/phase_q50_grand_benchmark.py

# CHSH Bell inequality test
python experiments/phase_q15_optimal_two_qubit.py

# Grover's algorithm
python experiments/phase_q18_grover_oneshot.py
```

### Reproducing Paper Figures

```bash
# V1 figures (Fig 1-6)
python experiments/generate_paper_figures.py

# V2 figures (Fig 7-9)
python experiments/generate_paper_figures_v2.py
```

## Model

Primary model: **Qwen2.5-3B-Instruct** (hidden_size=2048, 36 layers). Universality validated on **Qwen2.5-0.5B** (S=2.95) and **Qwen2.5-1.5B** (S=3.41).

## Experiment Phases

### Foundations (Q1-Q13)

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q1-Q2 | Basis states + interference | amp=0.498, cos_sim=-0.08 |
| Q3-Q4 | Quantum gates + hallucination suppression | H^2=I fidelity=1.0, suppress=97% |
| Q6v2 | Wavefunction anatomy | entropy peak@L10, collapse@L22-26 |
| Q9 | Layer universality | All layers amp~0.50 |
| Q10 | Task universality | visibility=1.000, CV=0.1% |
| Q11 | Quantum statistics | E(phi)=cos(phi), R^2>0.999 |
| Q13 | Decoherence | sigma_c~0.05, T_c~2.5 |

### Two-Qubit & Multi-Qubit (Q14-Q24)

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q14 | Optimal coupling | L20 peak, amp=0.82 |
| Q15 | Super-quantum CHSH | **S=3.41** (PR-box 85%) |
| Q16 | Statistical validation | 5/5 S>2.0, 3/5 S>2.83 |
| Q17 | 3-Qubit GHZ + Toffoli | 3 pairs S>2, CCNOT=2.69x |
| Q18 | Virtual Grover | **10/10, 4631x amplification** |
| Q19 | No-cloning violation | **35/35 = 100% clone** |
| Q20 | Model universality | 0.5B: S=2.95 > 2.83 |
| Q21 | Dimensionality as cryogenics | d_c ~ 1024-1536 |
| Q22 | NQPU specification | 256d, 12L, <$100, 300K |
| Q23 | Deutsch-Jozsa | **10/10 = 100% correct** |
| Q24 | Teleportation | amp=0.42 (partial) |

### Extended Experiments (Q25-Q50) — New in V2

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q25 | Period Finding | 4/5 periods correct |
| Q26 | Checkpointing | Fidelity = 1.0000 |
| Q30 | Entanglement Swap | **S = 2.995** |
| Q31 | Superdense Coding | **200/200 = 100%, 2.0 bits** |
| Q34 | LLM-QRNG | **3/3 NIST, entropy=0.998** |
| Q35 | Grover Scaling | **N^1.0 constant-time** |
| Q36 | Dimension Law | vis=1.0 @d>=32, CHSH needs training |
| Q40 | BB84 QKD | **100% key, QBER=28.3% (Eve)** |
| Q41 | Bernstein-Vazirani | **94/94 = 100%** |
| Q42 | Simon's Algorithm | **18/18 = 100%** |
| Q43 | SWAP Test | r=0.83 overlap |
| Q44 | Quantum Counting | 1.4-3.0x advantage |
| Q46 | Parallelism | **128x, 7 bits/query** |
| Q48 | Reservoir Computing | XOR 100%, sine r=0.71 |
| Q49 | State Tomography | Fidelity=0.925, 5.8 bits |
| Q50 | Grand Benchmark | **QAS = 74.6/100** |

## Citation

If you use this work, please cite:

```bibtex
@misc{funasaki2026semanticqubit,
  title={Semantic-Qubit: Discovering Universal Quantum-Like Coherence in Large Language Models},
  author={Hiroto Funasaki},
  year={2026},
  doi={10.5281/zenodo.20360031},
  url={https://doi.org/10.5281/zenodo.20360031}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## AI Collaboration Statement

This research was conducted as a collaborative effort between the human author and AI research assistants. All experimental decisions, research direction, and final interpretation were made by the human author.

## Acknowledgments

This research was conducted entirely independently, without institutional affiliation or corporate funding. The author currently faces financial constraints that make it increasingly difficult to maintain subscriptions to AI services essential for this line of research. To sustain and improve the quality of future work, the author is actively seeking community sponsorship at [https://github.com/sponsors/hafufu-stack](https://github.com/sponsors/hafufu-stack).
