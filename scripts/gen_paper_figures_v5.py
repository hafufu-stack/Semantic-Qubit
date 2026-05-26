# -*- coding: utf-8 -*-
"""Generate composite paper figures for v5.
Rule: Max 3 subplots horizontally. Use multiple rows for more."""
import os
from PIL import Image

SRC = r"c:\Users\kyjan\研究\Semantic-Qubit\figures"
DST = os.path.join(SRC, "paper")

def stack_vertical(paths, output_path):
    """Stack images vertically (one per row). Each image keeps its own subplots."""
    imgs = []
    for p in paths:
        fp = os.path.join(SRC, p)
        if os.path.exists(fp):
            imgs.append(Image.open(fp))
        else:
            print("  MISSING: %s" % fp)
    if not imgs:
        return
    # Resize all to same width
    target_w = max(img.width for img in imgs)
    resized = []
    for img in imgs:
        ratio = target_w / img.width
        new_h = int(img.height * ratio)
        resized.append(img.resize((target_w, new_h), Image.LANCZOS))
    total_h = sum(img.height for img in resized)
    canvas = Image.new('RGB', (target_w, total_h), (255, 255, 255))
    y = 0
    for img in resized:
        canvas.paste(img, (0, y))
        y += img.height
    canvas.save(output_path, dpi=(150, 150))
    print("  Saved: %s (%dx%d, %d rows)" % (
        os.path.basename(output_path), canvas.width, canvas.height, len(resized)))

import shutil

# Fig 18: Q144 only (3 subplots - OK as is)
print("Fig 18: Honest Benchmark (Q144 - 3 subplots)")
shutil.copy2(os.path.join(SRC, "phase_q144_honest.png"),
             os.path.join(DST, "fig18_honest_benchmark.png"))
print("  Copied (3 subplots, 1 row)")

# Fig 19: Q161 (3 subplots) + Q165 (3 subplots) -> 2 rows x 3 cols
print("Fig 19: Embedding VQE (Q161 + Q165 stacked vertically)")
stack_vertical([
    "phase_q161_embedding_vqe.png",
    "phase_q165_molecules.png",
], os.path.join(DST, "fig19_embedding_vqe.png"))

# Fig 20: Q163 (3 subplots) + Q167 (3 subplots) -> 2 rows x 3 cols
print("Fig 20: Temperature + QKD (Q163 + Q167 stacked vertically)")
stack_vertical([
    "phase_q163_temperature.png",
    "phase_q167_temp_qkd.png",
], os.path.join(DST, "fig20_temperature_qkd.png"))

# Fig 21: Q168 (3 subplots) + Q175 (3 subplots) -> 2 rows x 3 cols
print("Fig 21: Compression + Phase Transition (Q168 + Q175 stacked)")
stack_vertical([
    "phase_q168_compression.png",
    "phase_q175_phase_transition.png",
], os.path.join(DST, "fig21_compression_transition.png"))

# Fig 22: Q171 (3 subplots) + Q172 (3 subplots) -> 2 rows x 3 cols
print("Fig 22: Tomography + Teleportation (Q171 + Q172 stacked)")
stack_vertical([
    "phase_q171_tomography.png",
    "phase_q172_teleportation.png",
], os.path.join(DST, "fig22_tomography_teleportation.png"))

# Fig 23: Q173 (3 subplots) + Q176 (3 subplots) -> 2 rows x 3 cols
print("Fig 23: Holographic + No-Cloning (Q173 + Q176 stacked)")
stack_vertical([
    "phase_q173_holographic.png",
    "phase_q176_nocloning.png",
], os.path.join(DST, "fig23_holographic_nocloning.png"))

print("\nAll composite figures regenerated (max 3 per row, stacked vertically)!")
