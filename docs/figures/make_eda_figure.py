# Two clean EDA charts for the slides: taxonomy coverage by rank, and sequence-length distribution.
# Run from the project root after the notebook has written data/processed/:  python docs/figures/make_eda_figure.py
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "eda_slide.png"
taxa = pd.read_csv(ROOT / "data" / "processed" / "taxa_table.csv")
asv = pd.read_csv(ROOT / "data" / "processed" / "asv_table.csv")

INK, MUTED, BLUE, RED, GRID = "#1d2733", "#5b6675", "#2a6f97", "#c8553d", "#e3e8ee"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#b9c2cc", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.6), dpi=200, gridspec_kw={"wspace": 0.22})

# ---- (a) taxonomy coverage ----
ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
pct = [taxa[r].notna().mean() * 100 for r in ranks]
colors = ["#9fb6c8"] * 5 + [BLUE, BLUE]
bars = a.bar([r.capitalize() for r in ranks], pct, color=colors, width=0.68, zorder=2)
for bar, v in zip(bars, pct):
    a.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.0f}%", ha="center", fontsize=12.5,
           fontweight="bold" if v < 30 and bar.get_x() > 4 else "normal", color=INK)
a.set_ylim(0, 100); a.set_ylabel("ASVs with a name (%)")
a.set_title("Taxonomy coverage by rank", fontsize=15, fontweight="bold", color=INK, pad=12, loc="left")
a.grid(axis="y", color=GRID, zorder=0); a.spines[["top", "right"]].set_visible(False)
a.annotate("75% of ASVs have no genus\n85% have no species", xy=(5.0, 33), xytext=(3.4, 80),
           fontsize=12, color=BLUE, fontweight="bold", ha="center",
           arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8))

# ---- (b) sequence length ----
L = asv["seq_len"]
b.hist(L[L <= 150], bins=range(112, 152, 2), color="#4c956c", edgecolor="white", zorder=2)
n_long = int((L > 150).sum())
b.bar(153, n_long, width=2, color="#9fb6c8", edgecolor="white", zorder=2)
b.text(153, n_long + 45, f"{n_long} ASVs\nover 150", ha="center", fontsize=10.5, color=MUTED)
b.axvline(150, color=RED, ls="--", lw=2.2, zorder=3)
b.text(149.4, 1650, "Model input\nL = 150", color=RED, fontsize=12, fontweight="bold", va="top", ha="right")
b.set_xlabel("Sequence length (bp)"); b.set_ylabel("Number of ASVs")
b.set_title("Sequence length distribution", fontsize=15, fontweight="bold", color=INK, pad=12, loc="left")
b.grid(axis="y", color=GRID, zorder=0); b.spines[["top", "right"]].set_visible(False)
b.text(0.5, 0.56, f"Median {L.median():.0f} bp\nRange {L.min()} to {L.max()} bp\n"
                   f"Only {(L > 150).mean() * 100:.1f}% exceed 150\n(cut to 150)",
       transform=b.transAxes, fontsize=12, color=INK, va="top", linespacing=1.4)
b.set_xlim(112, 156)
b.set_xticks([115, 120, 125, 130, 135, 140, 145, 150])

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.2, facecolor="white")
print("saved", OUT)
