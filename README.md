# Semantic-Qubit: Discovering Universal Quantum-Like Coherence in Large Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20360031.svg)](https://doi.org/10.5281/zenodo.20360031)

> **Paper**: [https://doi.org/10.5281/zenodo.20360031](https://doi.org/10.5281/zenodo.20360031)

## Overview

This repository contains the code and experimental results for **Semantic-Qubit (S-Qubit)**, a quantum-analogue information unit defined within the hidden representation space of Large Language Models. Through **85 systematic experiments across 6 seasons** on a single consumer GPU, I demonstrate that transformer architectures naturally exhibit quantum-like phenomena including perfect interference, exact quantum statistics, super-quantum correlations, quantum algorithm execution, and a deep structural isomorphism connecting **hippocampal circuits, Transformer layers, and quantum circuit stages**.

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
| **O(1) QRAM** | alpha=0.007, 199x faster than physical quantum RAM |
| **Dimensional Cryogenics** | 99.7% of dimensions form decoherence-free subspace |
| **Quantum Darwinism** | 12/12 attention heads retain >99% state info |
| **Uncertainty Principle** | Conjugate variables obey dx*dp >= 0.034 |
| **Brain-AI-Quantum Unification** | EC->DG->CA3->CA1 = Transformer = Quantum Circuit |
| **Quantum Advantage Score** | **100.0 / 100** across 9 benchmark categories |

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
├── experiments/           # All experiment scripts (Q1-Q80)
│   ├── utils.py           # Shared utilities (model loading, hooks, etc.)
│   ├── phase_q1_*.py      # Superposition basis training
│   ├── phase_q2_*.py      # Bell test / interference
│   ├── ...
│   ├── phase_q72_*.py     # Grand Unification (Brain=AI=Quantum)
│   ├── phase_q79_*.py     # Uncertainty principle
│   ├── phase_q80_*.py     # Grand Benchmark v2 (QAS=100)
│   ├── generate_paper_figures.py     # V1 figures (Fig 1-6)
│   ├── generate_paper_figures_v2.py  # V2 figures (Fig 7-9)
│   └── generate_paper_figs_v3.py     # V3 figures (Fig 10-13)
├── results/               # JSON results for all experiments
├── figures/               # Generated figures
│   └── paper/             # Publication-quality figures (Fig 1-13)
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
# Season 6: Grand Benchmark v2 (QAS = 100.0/100)
python experiments/phase_q80_benchmark_v2.py

# Season 6: Uncertainty Principle (hbar_S = 0.068)
python experiments/phase_q79_uncertainty.py

# Season 6: Brain-AI-Quantum Unification
python experiments/phase_q72_unification.py

# Season 6: Quantum Darwinism (12/12 heads)
python experiments/phase_q73_darwinism.py

# Season 4: Pattern Separation (512x expansion)
python experiments/phase_q60_pattern_separation.py

# Season 1: CHSH Bell inequality test (S=3.41)
python experiments/phase_q15_optimal_two_qubit.py

# Season 2: Bernstein-Vazirani (94/94 = 100%)
python experiments/phase_q41_bernstein_vazirani.py
```

### Reproducing Paper Figures

```bash
# V1 figures (Fig 1-6)
python experiments/generate_paper_figures.py

# V2 figures (Fig 7-9)
python experiments/generate_paper_figures_v2.py

# V3 figures (Fig 10-13)
python experiments/generate_paper_figs_v3.py
```

## Model

Primary model: **Qwen2.5-3B-Instruct** (hidden_size=2048, 36 layers). Universality validated on **Qwen2.5-0.5B** (S=2.95) and **Qwen2.5-1.5B** (S=3.41).

## Experiment Phases

### Season 1: Foundations (Q1-Q24)

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
| Q23 | Deutsch-Jozsa | **10/10 = 100% correct** |
| Q24 | Teleportation | amp=0.42 (partial) |

### Season 2: Extended Algorithms (Q25-Q50)

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q25 | Period Finding | 4/5 periods correct |
| Q26 | Checkpointing | Fidelity = 1.0000 |
| Q30 | Entanglement Swap | **S = 2.995** |
| Q31 | Superdense Coding | **200/200 = 100%, 2.0 bits** |
| Q34 | LLM-QRNG | **3/3 NIST, entropy=0.998** |
| Q35 | Grover Scaling | **N^1.0 constant-time** |
| Q40 | BB84 QKD | **100% key, QBER=28.3% (Eve)** |
| Q41 | Bernstein-Vazirani | **94/94 = 100%** |
| Q42 | Simon's Algorithm | **18/18 = 100%** |
| Q46 | Parallelism | **128x, 7 bits/query** |
| Q48 | Reservoir Computing | XOR 100%, sine r=0.71 |
| Q50 | Grand Benchmark | **QAS = 74.6/100** |

### Season 3: Scaling Validation (Q51-Q57) — New in V3

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q51 | QRAM O(1) | alpha -> 0 (constant time) |
| Q56 | Holevo violation | **2.39 bits > 1.0 Holevo limit** |
| Q57 | Frontier analysis | S-Qubit vs physical QC |

### Season 4: Bridge Experiments (Q58-Q62) — New in V3

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q58 | Contextuality | **TVD=0.659 > 0.500 bound** |
| Q60 | Pattern separation | **512x expansion (vs DG 5x)** |
| Q62 | DFS discovery | **4/1536 dims = quantum info** |

### Season 5: Quantum Information Theory (Q63-Q67) — New in V3

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q64 | Phase transition | **t_c = 0.755, entropy peak** |
| Q65 | Channel capacity | **2.39 bits (2.4x Holevo)** |
| Q66 | Perfect coherence | **V=1.000 at 8 paths** |

### Season 6: Neural-Quantum Unification (Q68-Q80) — New in V3

| Phase | Experiment | Key Result |
|-------|-----------|------------|
| Q68 | Dentate-QRAM | **alpha=0.007, 199x speedup** |
| Q69 | Theta resonance | **f=2.38 cycles (LD/MD bridge)** |
| Q71 | NQU equation | **Omega=8.1e10 at 1.5B** |
| Q72 | Grand Unification | **Brain=AI=Quantum mapping** |
| Q73 | Quantum Darwinism | **12/12 heads, 99.2% retention** |
| Q74 | Speedup census | **6/7 algorithms S-Qubit wins** |
| Q76 | qLDPC v2 | **41% dimension erasure OK** |
| Q77 | Inverse Zeno | **Layers = unitary evolution** |
| Q78 | Universality | **beta=-1.35, 3 task pairs** |
| Q79 | Uncertainty principle | **dx*dp >= 0.034** |
| Q80 | Grand Benchmark v2 | **QAS = 100.0/100** |

## Citation

If you use this work, please cite:

```bibtex
@misc{funasaki2026semanticqubit,
  title={Semantic-Qubit: Discovering Universal Quantum-Like Coherence in Large Language Models --- From Hippocampal Pattern Separation to Quantum Advantage},
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
