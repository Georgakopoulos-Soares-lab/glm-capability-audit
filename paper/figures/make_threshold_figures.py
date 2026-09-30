#!/usr/bin/env python
"""Figures A1 to A3 for the identity sweep.

Note on direction, since it is easy to read backwards: raising the minimum identity
makes the criterion for declaring a hit stricter, so fewer hits are declared and MORE
sequences are retained as having no detectable nucleotide relative. Retention rises
with the threshold; 0.30 is the most conservative point.
"""
import os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(REPO, "reports", "threshold_sensitivity")
MM = 1 / 25.4
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "figure.dpi": 200, "savefig.dpi": 400,
    "savefig.bbox": "tight", "pdf.fonttype": 42})
ORDER = ["GUE phage", "HVUE host tropism hard_70", "VHDB within family",
         "HVUE pathogenicity hard", "HVUE pathogenicity std",
         "HVUE transmissibility std", "GUE virus_covid",
         "HVUE transmissibility hard", "HVUE host tropism std_95"]
SHORT = {"HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
         "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
         "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
         "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid"}
CMAP = plt.get_cmap("tab10")


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def save(fig, n):
    fig.savefig(f"{HERE}/{n}.pdf"); fig.savefig(f"{HERE}/{n}.png"); plt.close(fig)
    print(f"  wrote {n}")


def main():
    d = pd.read_csv(f"{D}/identity_sweep.csv")
    benches = [b for b in ORDER if b in set(d.benchmark)]
    col = {b: CMAP(i % 10) for i, b in enumerate(benches)}

    # A1 retained fraction
    fig, ax = plt.subplots(figsize=(120 * MM, 76 * MM)); tidy(ax)
    for b in benches:
        g = d[d.benchmark == b].sort_values("min_seq_id")
        ax.plot(g.min_seq_id, g.pct_retained, "-o", ms=3, lw=1.4, color=col[b], label=SHORT[b])
    ax.axvline(0.30, color=RED, lw=1.0, ls="--", zorder=1)
    ax.text(0.305, ax.get_ylim()[1] * .97, "primary", fontsize=6, color=RED, va="top")
    ax.set_xlabel("minimum sequence identity for declaring a hit")
    ax.set_ylabel("% of test set retained as no detectable relative")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout(); save(fig, "figA1_retention")

    # A2 excess with uncertainty
    sc = d[d.verdict != "NOT SCORED"]
    bs = [b for b in benches if b in set(sc.benchmark)]
    nc = min(3, len(bs)); nr = int(np.ceil(len(bs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(183 * MM, 52 * nr * MM), squeeze=False)
    for ax, b in zip(axes.ravel(), bs):
        g = sc[sc.benchmark == b].sort_values("min_seq_id"); tidy(ax)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.fill_between(g.min_seq_id, g.ci_low, g.ci_high, color=col[b], alpha=.20, lw=0)
        ax.plot(g.min_seq_id, g.excess, "-o", ms=3.5, lw=1.5, color=col[b])
        ax.axvline(0.30, color=RED, lw=.9, ls="--")
        ax.set_title(SHORT[b], loc="left"); ax.set_xlabel("min identity")
        ax.set_ylabel("model minus best baseline")
    for ax in axes.ravel()[len(bs):]:
        ax.axis("off")
    fig.tight_layout(); save(fig, "figA2_excess")

    # A3 independent units retained
    fig, ax = plt.subplots(figsize=(120 * MM, 76 * MM)); tidy(ax)
    for b in benches:
        g = d[d.benchmark == b].sort_values("min_seq_id")
        ax.plot(g.min_seq_id, g.n_units, "-o", ms=3, lw=1.4, color=col[b],
                label=f"{SHORT[b]} ({g.bootstrap_unit.iloc[0]})")
    ax.axvline(0.30, color=RED, lw=1.0, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("minimum sequence identity for declaring a hit")
    ax.set_ylabel("independent units retained (log)")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout(); save(fig, "figA3_units")


if __name__ == "__main__":
    main()
