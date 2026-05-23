# Semantic-Qubit: Discovering Universal Quantum-Like Coherence in Large Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20360031.svg)](https://doi.org/10.5281/zenodo.20360031)

> **Paper**: [https://doi.org/10.5281/zenodo.20360031](https://doi.org/10.5281/zenodo.20360031)

## Overview

This repository contains the code and experimental results for **Semantic-Qubit (S-Qubit)**, a quantum-analogue information unit defined within the hidden representation space of Large Language Models. By training orthogonal "soul vectors" as computational basis states and injecting their superposition into intermediate layers, we demonstrate quantum-like phenomena emerging naturally in transformer architectures.

## Key Findings

| Finding | Result |
|---------|--------|
| **Perfect Interference** | Visibility = 1.000 across all tasks (CV = 0.1%) |
| **Exact Quantum Statistics** | E(phi) = cos(phi) with R^2 > 0.999 |
| **Super-Quantum CHSH** | S = 3.41, exceeding Tsirelson bound (2.83) |
| **Virtual Grover Search** | 10/10, 4631x amplification in single forward pass |
| **Deutsch-Jozsa Algorithm** | 6/6 = 100% correct, single-query classification |
| **No-Cloning Violation** | 35/35 = 100% perfect state cloning |
| **Model Universality** | Confirmed on Qwen2.5-0.5B (S=2.95) and 1.5B (S=3.41) |
| **Dimensionality as Cryogenics** | Coherence emerges at d_c ~ 1024-1536 |

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
├── experiments/           # All experiment scripts (Q1-Q24)
│   ├── utils.py           # Shared utilities (model loading, hooks, etc.)
│   ├── phase_q1_*.py      # Superposition basis training
│   ├── phase_q2_*.py      # Bell test / interference
│   ├── ...
│   ├── phase_q23_*.py     # Deutsch-Jozsa algorithm
│   ├── phase_q24_*.py     # Quantum teleportation
│   └── generate_paper_figures.py  # Reproduce all paper figures
├── results/               # JSON results for all experiments
├── figures/               # Generated figures
│   └── paper/             # Publication-quality figures (Fig 1-6)
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers
- Qwen2.5-1.5B model (downloaded locally)

```bash
pip install torch transformers matplotlib numpy scipy
```

### Running Experiments

Each experiment is a standalone script:

```bash
# Single-qubit interference
python experiments/phase_q1_superposition_basis.py

# CHSH Bell inequality test
python experiments/phase_q15_optimal_two_qubit.py

# Grover's algorithm
python experiments/phase_q18_grover_oneshot.py

# Deutsch-Jozsa algorithm
python experiments/phase_q23_deutsch_jozsa.py
```

### Reproducing Paper Figures

```bash
python experiments/generate_paper_figures.py
```

This generates all 6 figures in `figures/paper/`.

## Model

All experiments use **Qwen2.5-1.5B** (hidden_size=1536, 28 layers) as the primary model, with **Qwen2.5-0.5B** (hidden_size=896, 24 layers) for universality validation (Q20).

## Experiment Phases

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q1-Q2 | Basis states + interference | amp=0.498, cos_sim=-0.08 |
| Q3-Q4 | Quantum gates + hallucination suppression | H^2=I fidelity=1.0, suppress=97% |
| Q6v2 | Wavefunction anatomy | entropy peak@L10, collapse@L22-26 |
| Q9 | Layer universality | All layers amp~0.50 |
| Q10 | Task universality | visibility=1.000, CV=0.1% |
| Q11 | Quantum statistics | E(phi)=cos(phi), R^2>0.999 |
| Q13 | Decoherence | sigma_c~0.05, T_c~2.5 |
| Q14 | Optimal coupling | L20 peak, amp=0.82 |
| Q15 | Super-quantum CHSH | **S=3.41** (PR-box 85%) |
| Q16 | Statistical validation | 5/5 S>2.0, 3/5 S>2.83 |
| Q17 | 3-Qubit GHZ + Toffoli | 3 pairs S>2, CCNOT=2.69x |
| Q18 | Virtual Grover | **10/10, 4631x amplification** |
| Q19 | No-cloning violation | **35/35 = 100% clone** |
| Q20 | Model universality | 0.5B: S=2.95 > 2.83 |
| Q21 | Dimensionality as cryogenics | d_c ~ 1024-1536 |
| Q22 | NQPU specification | 256d, 12L, <$100, 300K |
| Q23 | Deutsch-Jozsa | **6/6 = 100% correct** |
| Q24 | Teleportation | amp=0.42 (partial) |

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

This research was conducted as a collaborative effort between the human author and AI research assistants (Claude, Gemini). All experimental decisions, research direction, and final interpretation were made by the human author.

## Acknowledgments

This research was conducted entirely independently, without institutional affiliation or corporate funding. The author is actively seeking community sponsorship at [https://github.com/sponsors/hafufu-stack](https://github.com/sponsors/hafufu-stack).
