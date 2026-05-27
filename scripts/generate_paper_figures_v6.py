# -*- coding: utf-8 -*-
"""Generate composite paper figures for v6 (fig24--fig28)."""
import os
import numpy as np
from PIL import Image

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
PAPER_DIR = os.path.join(FIGURES_DIR, 'paper')
os.makedirs(PAPER_DIR, exist_ok=True)


def combine_vertical(paths, output, gap=20, bg=(255, 255, 255)):
    """Stack images vertically with gap."""
    imgs = [Image.open(p) for p in paths if os.path.exists(p)]
    if not imgs:
        print("  SKIP: no images found for", output)
        return
    max_w = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new('RGB', (max_w, total_h), bg)
    y = 0
    for im in imgs:
        # Center horizontally
        x = (max_w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + gap
    canvas.save(output, dpi=(150, 150))
    print("  Saved:", output, f"({canvas.width}x{canvas.height})")


def combine_horizontal(paths, output, gap=20, bg=(255, 255, 255)):
    """Place images side by side horizontally."""
    imgs = [Image.open(p) for p in paths if os.path.exists(p)]
    if not imgs:
        print("  SKIP: no images found for", output)
        return
    max_h = max(im.height for im in imgs)
    total_w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new('RGB', (total_w, max_h), bg)
    x = 0
    for im in imgs:
        y = (max_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + gap
    canvas.save(output, dpi=(150, 150))
    print("  Saved:", output, f"({canvas.width}x{canvas.height})")


def combine_grid(paths, output, cols=2, gap=20, bg=(255, 255, 255)):
    """Arrange images in a grid."""
    imgs = [Image.open(p) for p in paths if os.path.exists(p)]
    if not imgs:
        print("  SKIP: no images found for", output)
        return
    rows = (len(imgs) + cols - 1) // cols
    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)
    total_w = cell_w * cols + gap * (cols - 1)
    total_h = cell_h * rows + gap * (rows - 1)
    canvas = Image.new('RGB', (total_w, total_h), bg)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        x = c * (cell_w + gap) + (cell_w - im.width) // 2
        y = r * (cell_h + gap) + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    canvas.save(output, dpi=(150, 150))
    print("  Saved:", output, f"({canvas.width}x{canvas.height})")


def main():
    print("Generating paper figures for v6...")
    print()

    # Fig 24: Season 27 Robust Foundations
    # Combine Q177 (barren plateau), Q179 (reproducibility), Q181 (blind test)
    print("--- Fig 24: Robust Foundations ---")
    combine_vertical([
        os.path.join(FIGURES_DIR, 'phase_q177_barren_plateau.png'),
        os.path.join(FIGURES_DIR, 'phase_q179_reproducibility.png'),
        os.path.join(FIGURES_DIR, 'phase_q181_cross_arch_blind.png'),
    ], os.path.join(PAPER_DIR, 'fig24_robust_foundations.png'))

    # Fig 25: Season 28 Quantum Chemistry
    # Combine Q182 (excited VQE), Q183 (multi-molecular), Q184 (Head 11)
    print("--- Fig 25: Quantum Chemistry ---")
    combine_vertical([
        os.path.join(FIGURES_DIR, 'phase_q183_quantum_advantage.png'),
        os.path.join(FIGURES_DIR, 'phase_q184_attn_entanglement.png'),
    ], os.path.join(PAPER_DIR, 'fig25_quantum_chemistry.png'))

    # Fig 26: Seasons 29-30 Scaling & Invincibility
    # Combine Q192 (multi-molecule), Q193 (phase transition), Q194 (extreme noise)
    print("--- Fig 26: Scaling & Invincibility ---")
    combine_vertical([
        os.path.join(FIGURES_DIR, 'phase_q192_multimol.png'),
        os.path.join(FIGURES_DIR, 'phase_q194_extreme_noise.png'),
    ], os.path.join(PAPER_DIR, 'fig26_scaling_invincibility.png'))

    # Fig 27: Season 31 RMSNorm Proof + Protein Folding
    # Combine Q195 (RMSNorm), Q196 (protein folding)
    print("--- Fig 27: RMSNorm & Protein ---")
    combine_vertical([
        os.path.join(FIGURES_DIR, 'phase_q195_rmsnorm_proof.png'),
        os.path.join(FIGURES_DIR, 'phase_q196_protein.png'),
    ], os.path.join(PAPER_DIR, 'fig27_rmsnorm_protein.png'))

    # Fig 28: Universal Gates + Bell + Distillation
    # Combine Q198 (gates), Q199 (distillation), Q202 (bell)
    print("--- Fig 28: Gates & Bell ---")
    combine_vertical([
        os.path.join(FIGURES_DIR, 'phase_q198_gates.png'),
        os.path.join(FIGURES_DIR, 'phase_q199_distillation.png'),
        os.path.join(FIGURES_DIR, 'phase_q202_bell.png'),
    ], os.path.join(PAPER_DIR, 'fig28_gates_bell.png'))

    print("\nAll paper figures generated!")


if __name__ == '__main__':
    main()
