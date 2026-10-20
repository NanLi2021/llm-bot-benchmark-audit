"""Figure 1 of the paper: where the accounts came from predicts the label (one column, two panels).

(a) Fox8-23 takes its bots from a phrase search and its humans from four earlier studies. The route alone gives
    the label (route-only accuracy), and a bag of words and the best single profile attribute separate bots from
    humans and, with no bot involved, humans of one study from humans of another (median over six pairs).
(b) Four pre-LLM Twitter IO campaigns: the median message-level bag-of-words AUC with each class from its own
    route, and with one route (the same hashtag and day, equal posts per class in every cell; paired protocol).
Inputs (a results directory, default results/expected/): route_provenance_table.csv, baseline_f1.csv,
route_vocabulary.csv, profile_cells_cv2.csv, route_attributes.csv, io_topic_matched.csv.
The humans-only values use 20 messages per account and the bots-vs-humans bag of words 50.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import argparse
from pathlib import Path

from plot_style import GRID, INK, SOFT, publication_style, save_figure

ap = argparse.ArgumentParser(description="Figure 1 of the paper, from a results directory.")
ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results/expected")
ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "build/figures")
cli = ap.parse_args()

BOT, HUMAN = "#D55E00", "#0072B2"          # Okabe-Ito: the bot class, the human class
BOT_FILL, HUMAN_FILL = "#FBEDE4", "#E7F1F8"
VAL, CAP, HEAD, BODY, TITLE = 10.5, 7.3, 8.0, 7.6, 8.4   # nothing below 7 pt

P = cli.results
_b = pd.read_csv(P / "baseline_f1.csv")
bow = float(_b[(_b.corpus == "Fox8-23") & (_b.condition == "all 50 messages")].auc.iloc[0])
prof = float(pd.read_csv(P / "profile_cells_cv2.csv").query("corpus == 'Fox8-23'").cv_auc.iloc[0])
_v = pd.read_csv(P / "route_vocabulary.csv")
voc = float(np.median(_v[(_v.input == "full text, 20 messages") & _v.auc.notna()].auc))
att = float(np.median(pd.read_csv(P / "route_attributes.csv").cv_auc))
route_acc = float(pd.read_csv(P / "route_provenance_table.csv").set_index("corpus")
                  .loc["Fox8-23"].route_only_accuracy)
_io = pd.read_csv(P / "io_topic_matched.csv")   # paired protocol: every hashtag-day cell balanced
m0 = _io[_io.rung == "M0"].sort_values("campaign")
m2 = _io[_io.rung == "M2"].sort_values("campaign")
assert list(m0.campaign) == list(m2.campaign) and len(m0) == 4
io_hi, io_lo = float(np.median(m0.auc)), float(np.median(m2.auc))

publication_style()
fig = plt.figure(figsize=(3.33, 4.55))
a = fig.add_axes([0.02, 0.505, 0.96, 0.445])
b = fig.add_axes([0.02, 0.015, 0.96, 0.405])
for ax in (a, b):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, w, h, text, edge, fc="white", fs=BODY, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.02",
                                linewidth=0.9, edgecolor=edge, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=3, fontweight=weight, linespacing=1.25)


def arrow(ax, p, q, colour=SOFT, lw=0.9, scale=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
                                 color=colour, shrinkA=1, shrinkB=1, zorder=1))


def value(ax, x, y, val, cap):
    ax.text(x, y, f"{val:.3f}", ha="left", va="center", fontsize=VAL, fontweight="bold", color=INK)
    if cap:
        ax.text(x, y - 0.085, cap, ha="left", va="center", fontsize=CAP, color=SOFT)


# ---------------------------------------------------------------- (a) a route for each class
box(a, 0.01, 0.770, 0.47, 0.185,
    "1,140 bots\nfrom a search for\n“as an AI language model”", BOT, fc=BOT_FILL)
box(a, 0.52, 0.770, 0.47, 0.185,
    "1,140 humans\nfrom four earlier\nbot-detection studies", HUMAN, fc=HUMAN_FILL)
box(a, 0.25, 0.615, 0.50, 0.085, "Fox8-23", INK, fs=8.4, weight="bold")
arrow(a, (0.245, 0.770), (0.37, 0.702))
arrow(a, (0.755, 0.770), (0.63, 0.702))
a.text(0.5, 0.560, "the route then carries", ha="center", va="center", fontsize=BODY,
       color=SOFT, zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

for x, head, model, rows in [
        (0.00, "the label", "route alone, accuracy", [(route_acc, "")]),
        (0.355, "the words", "bag of words, AUC",
         [(bow, "bots vs humans"), (voc, "humans only")]),
        (0.69, "the profile", "one attribute, AUC",
         [(prof, "bots vs humans"), (att, "humans only")])]:
    arrow(a, (0.5, 0.615), (x + 0.15, 0.520), colour=GRID)
    a.text(x, 0.475, head, ha="left", va="center", fontsize=HEAD, color=INK, fontweight="bold")
    a.text(x, 0.420, model, ha="left", va="center", fontsize=CAP, color=SOFT)
    a.plot([x, x + 0.29], [0.375, 0.375], color=GRID, lw=0.7)
    for i, (val, cap) in enumerate(rows):
        value(a, x, 0.300 - 0.195 * i, val, cap)

# ---------------------------------------------------------------- (b) two routes, then one
AUC_X = 0.87
b.text(AUC_X, 0.900, "median bag of\nwords AUC", ha="center", va="center", fontsize=CAP, color=SOFT,
       linespacing=1.1)
b.text(0.0, 0.930, "before: two routes", ha="left", va="center", fontsize=HEAD,
       fontweight="bold", color=INK)
for y, name, route, edge, fc in [(0.785, "bots", "takedown archive", BOT, BOT_FILL),
                                 (0.595, "humans", "topic search", HUMAN, HUMAN_FILL)]:
    b.text(0.0, y, name, ha="left", va="center", fontsize=BODY, color=edge)
    arrow(b, (0.150, y), (0.235, y), colour=edge)
    box(b, 0.24, y - 0.050, 0.36, 0.100, route, edge, fc=fc)
    arrow(b, (0.605, y), (AUC_X - 0.08, 0.690), colour=edge)
b.text(AUC_X, 0.690, f"{io_hi:.3f}", ha="center", va="center", fontsize=VAL,
       fontweight="bold", color=INK)

b.text(0.0, 0.435, "after: one route", ha="left", va="center", fontsize=HEAD,
       fontweight="bold", color=INK)
for y, name, edge in [(0.290, "bots", BOT), (0.100, "humans", HUMAN)]:
    b.text(0.0, y, name, ha="left", va="center", fontsize=BODY, color=edge)
    arrow(b, (0.150, y), (0.235, y), colour=edge)
box(b, 0.24, 0.050, 0.36, 0.290, "same hashtag\nand day, equal\nposts per class", INK)
arrow(b, (0.605, 0.195), (AUC_X - 0.08, 0.195), colour=INK)
b.text(AUC_X, 0.195, f"{io_lo:.3f}", ha="center", va="center", fontsize=VAL,
       fontweight="bold", color=INK)

arrow(b, (AUC_X, 0.625), (AUC_X, 0.260), colour=BOT, lw=1.5, scale=9)
b.text(AUC_X - 0.025, 0.442, f"$-${io_hi - io_lo:.3f}", ha="right", va="center", fontsize=CAP,
       color=BOT, fontweight="bold")

for ax, letter, title in [(a, "a", "Fox8-23: a different route for each class"),
                          (b, "b", f"Twitter IO, {len(m0)} pre-LLM campaigns")]:
    ax.text(0.0, 1.035, f"({letter})  {title}", ha="left", va="bottom", fontsize=TITLE,
            fontweight="bold", color=INK, transform=ax.transAxes)
save_figure(fig, "fig_route_column", cli.output_dir)
print(f"  label {route_acc:.3f}; words {bow:.3f} / {voc:.3f}; profile {prof:.3f} / {att:.3f}; "
      f"IO {io_hi:.3f} -> {io_lo:.3f}")
