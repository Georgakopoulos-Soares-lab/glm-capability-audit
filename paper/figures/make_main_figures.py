#!/usr/bin/env python
"""Main figures. Every number is read from final_source_of_truth.csv.

Palette and rcParams reuse the validated instance in make_figures.py. Verdicts are
never carried by colour alone: marker shape and a printed interval carry them too.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
T = os.path.join(REPO, "reports", "tables")
MM = 1 / 25.4
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "figure.dpi": 200, "savefig.dpi": 400,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})
ARMLABEL = {"kmer": "$k$-mer", "kmer_gbt": "$k$-mer GBT", "aa_gbt": "AA composition",
            "cnn": "CNN", "transformer": "transformer", "nn_lookup": "label lookup",
            "tax_prior": "taxonomy prior", "tax_prior_recovered": "taxonomy prior"}
ORDER = ["GUE phage", "HT hard_70", "VHDB", "Patho hard", "Patho std",
         "Transm std", "GUE covid", "Transm hard", "HT std_95"]


def require(p, by, what):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise SystemExit(f"missing input: {p}\n  holds: {what}\n  produce it with: {by}")
    return p


def load():
    from sot_path import sot_csv
    return pd.read_csv(require(sot_csv(),
        "paper/figures/make_source_of_truth.py", "every manuscript number"))


def tidy(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"{HERE}/{name}.pdf"); fig.savefig(f"{HERE}/{name}.png")
    plt.close(fig); print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------- Figure 1 ---
def fig1(df):
    """Fragment-level incremental performance, one panel per regime."""
    # equal halves across the full width, with the row labels living in the left
    # margin rather than stealing width from panel A
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 78 * MM), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.07))
    ypos = {b: i for i, b in enumerate(ORDER)}
    for ax, reg in zip(axes, ("L0", "L1")):
        d = df[(df.regime == reg) & (~df.skipped)].set_index("benchmark")
        tidy(ax)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for b in ORDER:
            if b not in d.index:
                continue
            r, i = d.loc[b], ypos[b]
            ok = bool(r.adequate_population)
            c = BLUE if r.excess > 0 else ORANGE
            ax.plot([r.frag_lo, r.frag_hi], [i, i], color=c, lw=1.7 if ok else 1.0,
                    alpha=1.0 if ok else 0.40, zorder=3, solid_capstyle="round")
            ax.plot(r.excess, i, "o" if ok else "s", ms=5 if ok else 3.4,
                    color=c, mec="white", mew=0.8, alpha=1.0 if ok else 0.45, zorder=4)
            if not ok:
                ax.text(r.frag_hi + 0.035, i, f"n={int(r.n_rows)}", va="center",
                        fontsize=5.6, color=MUTED)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER)
        ax.set_ylim(-0.7, len(ORDER) - 0.3)
        ax.invert_yaxis()
        panel, n = {"L0": ("A", "published test set"),
                    "L1": ("B", "no nucleotide relative in training")}[reg]
        ax.text(0.0, 1.035, panel, transform=ax.transAxes, fontsize=9,
                fontweight="bold", color=INK, va="bottom")
        ax.text(0.055, 1.035, f"{reg}   {n}", transform=ax.transAxes, fontsize=8,
                color=INK, va="bottom")
        ax.set_xlabel("model minus best baseline (MCC)")
    axes[0].set_xlim(-0.30, 0.62)
    h = [plt.Line2D([], [], marker="o", ls="-", color=BLUE, ms=5, lw=1.7,
                    label="model ahead"),
         plt.Line2D([], [], marker="o", ls="-", color=ORANGE, ms=5, lw=1.7,
                    label="baseline ahead"),
         plt.Line2D([], [], marker="s", ls="-", color=MUTED, ms=3.4, lw=1.0, alpha=0.45,
                    label="fewer than 100 sequences")]
    fig.legend(handles=h, loc="lower center", frameon=False, ncol=3,
               bbox_to_anchor=(0.5, -0.005), columnspacing=2.6, handlelength=1.8)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.90, bottom=0.235)
    save(fig, "fig1_performance")


# ---------------------------------------------------------------- Figure 2 ---
def fig2(df):
    """Why the picture changes: both sides fall, and the baseline loses its best arm."""
    d = df[(~df.skipped) & (df.regime.isin(["L0", "L1"]))].copy()
    keep = [b for b in ORDER
            if {"L0", "L1"} <= set(d[d.benchmark == b].regime)
            and d[(d.benchmark == b) & (d.regime == "L1")].n_rows.iloc[0] >= 100]
    fig, axes = plt.subplots(1, len(keep), figsize=(183 * MM, 66 * MM), sharey=True)
    for ax, b in zip(axes, keep):
        sub = d[d.benchmark == b].set_index("regime")
        tidy(ax, grid="y")
        mv = [sub.loc["L0", "model_mcc"], sub.loc["L1", "model_mcc"]]
        bv = [sub.loc["L0", "envelope_mcc"], sub.loc["L1", "envelope_mcc"]]
        for v, c, lab in ((mv, BLUE, "model"), (bv, ORANGE, "best baseline")):
            ax.plot([0, 1], v, "-o", color=c, lw=1.8, ms=5, mec="white", mew=0.8,
                    zorder=3, label=lab)
        # place each label on the side away from the other series, and push it clear
        # of the axis when a value sits near zero
        for i in (0, 1):
            above = mv[i] >= bv[i]
            gap = abs(mv[i] - bv[i])
            pad = 8 if gap > 0.09 else 13
            for v, c, up in ((mv[i], BLUE, above), (bv[i], ORANGE, not above)):
                dy = pad if up else -(pad + 5)
                if v < 0.12 and not up:
                    dy = pad
                ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=6.2, color=c,
                            zorder=5)
        # which conventional predictor is the best one at each regime: this is the
        # point of the figure, not decoration
        for x, reg in ((0, "L0"), (1, "L1")):
            ax.text(x, -0.105, ARMLABEL.get(sub.loc[reg, "envelope_arm"], "--"),
                    ha="center", va="top", fontsize=6, color=ORANGE,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks([0, 1]); ax.set_xticklabels(["published", "divergent"])
        ax.set_xlim(-0.45, 1.45); ax.set_ylim(-0.02, 1.06)
        ax.set_title(b, loc="left", color=INK)
    axes[0].set_ylabel("MCC")
    axes[0].legend(frameon=False, loc="lower left", ncol=1)
    fig.tight_layout()
    save(fig, "fig2_baseline_collapse")


# ---------------------------------------------------------------- Figure 3 ---
def fig3(df):
    """What each filter actually removes."""
    piv = df.pivot_table(index="benchmark", columns="regime", values="n_rows",
                         aggfunc="first")
    for c in ("L0", "L1", "L2"):
        if c not in piv:
            piv[c] = 0
    piv = piv.reindex([b for b in ORDER if b in piv.index]).fillna(0)
    # Taller than wide-screen default so nine benchmarks x three bars can breathe, and
    # the legend sits below the axes rather than over the longest bars.
    fig, ax = plt.subplots(figsize=(183 * MM, 88 * MM))
    tidy(ax, grid="x")
    y = np.arange(len(piv))
    FLOOR = 0.55
    for off, reg, c, lab in ((-0.26, "L0", BLUE, "published test set"),
                             (0.0, "L1", ORANGE, "no nucleotide relative"),
                             (0.26, "L2", AQUA, "no translated relative")):
        v = piv[reg].values.astype(float)
        ax.barh(y + off, np.maximum(v, FLOOR), height=0.24, color=c, zorder=3, label=lab)
        for i, x in enumerate(v):
            n0 = piv["L0"].values[i]
            if reg == "L0" or not n0:
                txt = f"{int(x):,}"
            else:
                txt = f"{int(x):,}  ({100 * x / n0:.0f}%)"
            # A zero population has no bar to sit beside, so pin its label just inside
            # the axis instead of letting it collide with the next row's label.
            xpos = FLOOR * 1.25 if x <= 0 else max(x, FLOOR) * 1.28
            ax.text(xpos, y[i] + off, txt, va="center", ha="left",
                    fontsize=5.9, color=INK2, zorder=4)
    ax.axvspan(0.5, 100, color=GRID, alpha=0.5, zorder=0)
    # sits above the first row; the y axis is inverted so -0.55 is the top margin
    ax.text(1.6, -0.55, "fewer than 100 sequences",
            fontsize=6, color=MUTED, va="center", ha="left")
    ax.set_xscale("log")
    # headroom on the right so the longest count label clears the axis
    ax.set_xlim(FLOOR, 6.0e5)
    ax.set_yticks(y); ax.set_yticklabels(piv.index)
    ax.set_ylim(-0.75, len(piv) - 0.25); ax.invert_yaxis()
    ax.set_xlabel("test sequences retained (log scale)")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), handlelength=1.4, columnspacing=1.8)
    fig.tight_layout()
    save(fig, "fig3_what_filters_remove")


def main():
    df = load()
    print("main figures, all values from final_source_of_truth.csv")
    fig1(df); fig2(df); fig3(df)


if __name__ == "__main__":
    main()
