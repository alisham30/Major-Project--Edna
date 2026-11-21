# Draws the data pre-processing pipeline as a wide (16:9 friendly) PNG for the slides.
# Run from the project root:  python docs/figures/make_pipeline_figure.py
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "preprocessing_pipeline.png"

INK, MUTED = "#1d2733", "#5b6675"
C_DATA, C_CLEAN, C_DNA, C_ENV, C_SPLIT, C_OUT = "#5b7083", "#2a6f97", "#2a6f97", "#4c956c", "#c8553d", "#6d597a"

fig, ax = plt.subplots(figsize=(18, 6.2), dpi=200)
ax.set_xlim(0, 180); ax.set_ylim(0, 64); ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, color, step, title, lines, count=None, count_y=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="white", ec=color, lw=2.2, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 7.2), w, 7.2, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=color, ec=color, lw=2.2, zorder=3))
    ax.add_patch(plt.Rectangle((x, y + h - 7.2), w, 3.0, fc=color, ec=color, zorder=3))  # square off the header bottom
    ax.text(x + w / 2, y + h - 3.6, title, color="white", fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=4)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 11.5 - i * 4.3, t, color=INK, fontsize=10.2, ha="center", va="center", zorder=4)
    if count:
        ax.text(x + w / 2, count_y if count_y is not None else y - 4.2, count, color=color, fontsize=13,
                fontweight="bold", ha="center", va="center")


def arrow(x0, y0, x1, y1, color="#8a97a6"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=color, zorder=1,
                                 shrinkA=0, shrinkB=0))


W, H, Y = 21, 30, 17           # standard box
mid = Y + H / 2
xs = [2, 26.5, 51, 75.5]       # raw, clean, filter, dedup

box(xs[0], Y, W, H, C_DATA, "01", "Raw data", ["asv_table.csv", "taxa_table.csv", "metadata_table.csv", "MBARI 18S V9"], "280,440 rows")
box(xs[1], Y, W, H, C_CLEAN, "02", "Cleaning", ["Missing-value", "normalisation", "'unassigned', 'no_hit',", "'g_' -> missing"], "clean labels")
box(xs[2], Y, W, H, C_CLEAN, "03", "Filtering", ["Zero-abundance", "removal", "keep only real", "detections"], "64,903 rows")
box(xs[3], Y, W, H, C_CLEAN, "04", "Deduplication", ["SHA-1 sequence", "hashing", "37 duplicate", "sequences merged"], "4,674 ASVs")
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + W, mid, b, mid)

# fork into two encoding lanes
ex, ew, eh = 103.5, 29, 24
y_dna, y_env = 35.5, 4.5
box(ex, y_dna, ew, eh, C_DNA, "05a", "DNA encoding", ["One-hot (A, C, G, T, N)", "pad / truncate to 150"], "150 x 5 matrix", count_y=y_dna + 3.4)
box(ex, y_env, ew, eh, C_ENV, "05b", "Environment", ["month (12) + depth (2)", "+ z-scored temp, sal, chl (3)"], "17-d vector", count_y=y_env + 3.4)
fx = xs[3] + W
arrow(fx, mid, ex, y_dna + eh / 2, C_DNA)
arrow(fx, mid, ex, y_env + eh / 2, C_ENV)

# open-set split
sx, sw = 137.5, 23
box(sx, Y, sw, H, C_SPLIT, "06", "Open-set split", ["Hide whole genera", "known / hidden / dark", "Group split by ASV", "(no DNA leakage)"], "5 groups")
arrow(ex + ew, y_dna + eh / 2, sx, mid, C_DNA)
arrow(ex + ew, y_env + eh / 2, sx, mid, C_ENV)

# output
ox, ow = 164, 14.5
box(ox, Y, ow, H, C_OUT, "07", "Tensors", ["X: DNA", "E: context", "y: genus"], "model input")
arrow(sx + sw, mid, ox, mid)

ax.text(90, 62.5, "Data pre-processing pipeline", fontsize=17, fontweight="bold", color=INK, ha="center", va="center")
ax.text(90, 0.6, "Depth is 0 m in every sample, so temperature, salinity and chlorophyll from the CTD replace it as context.",
        fontsize=10, color=MUTED, ha="center", va="center", style="italic")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print("saved", OUT)
