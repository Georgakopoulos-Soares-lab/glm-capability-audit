"""Can ViroBench's two well-constructed strata resolve an effect at all?

CALIBRATION MATTERS HERE. A synthetic half width on a single score is NOT the
quantity this project calls the MDE. The project's MDE is the half width of the
interval on the excess over the floor, with the floor re-maximised inside every
bootstrap replicate, which is wider. Measured against the four strata where both
numbers exist, the two agree on large strata and the real interval is about 2.1
to 2.3 times wider on small ones, where the floor re-maximisation dominates:

  HT hard_70 L2 n=3,737  real 0.131  synthetic 0.134   ratio 0.98
  HT std_95 L2  n=3,882  real 0.149  synthetic 0.194   ratio 0.77
  VHDB L2     n=201    real 0.497  synthetic 0.217   ratio 2.29
  GUE phage L2  n=86     real 0.139  synthetic 0.065   ratio 2.13

ViroBench's strata are 286 and 125 rows, squarely in the small regime, so the
synthetic numbers below are multiplied by 2.3 to give a calibrated MDE. Quote
the calibrated column, never the raw one.

The ALL and RNA temporal splits are the only strata anywhere in this study that
are simultaneously homology free, genus disjoint and genus diverse. That makes
them the only candidates for an interpretable measurement. Whether they can
carry one is a question about their size and class balance, and it is answerable
before any model is trained.

The minimum detectable effect is taken the way the rest of the project takes it:
the half width of a 95 percent bootstrap interval on the metric. No model is run.
A synthetic predictor is driven to a target score and the interval around it is
measured, across a grid of target scores, because the interval is widest near
zero and the answer should not depend on guessing the effect in advance.

Both bootstrap units are reported. Rows is anti conservative here, since rows
from one genus are not independent; genus is the defensible unit and is the one
to quote.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from inject import inject, data_path, require, require_rows
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

VIRO = "/data/nvidia/genomic-capability-unlearning/scratchpad/multimodel/virobench"
MASKS = "/home/nvidia/genomic-capability-unlearning/reports/virobench_audit"
VERT = {"D1", "D2", "D3"}


def drive_to(y, target, rng, tol=0.01, iters=60):
    """A predictor scoring about `target`, made by corrupting truth."""
    if target <= 0:
        return rng.permutation(y)
    lo, hi = 0.0, 1.0
    best = y.copy()
    for _ in range(iters):
        frac = (lo + hi) / 2
        p = y.copy()
        k = int(round(frac * len(y)))
        if k:
            idx = rng.choice(len(y), k, replace=False)
            p[idx] = rng.permutation(y)[idx]
        s = matthews_corrcoef(y, p)
        best = p
        if abs(s - target) < tol:
            break
        if s > target:
            lo = frac
        else:
            hi = frac
    return best


def halfwidth(y, p, groups, rng, B=2000):
    """Half width of the 95 percent bootstrap interval, by row and by group."""
    out = {}
    n = len(y)
    s = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        s.append(matthews_corrcoef(y[i], p[i]))
    out["row"] = float((np.percentile(s, 97.5) - np.percentile(s, 2.5)) / 2) if s else float("nan")
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    s = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        i = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[i])) < 2:
            continue
        s.append(matthews_corrcoef(y[i], p[i]))
    out["genus"] = float((np.percentile(s, 97.5) - np.percentile(s, 2.5)) / 2) if s else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", default=["ALL_taxon_times", "RNA_taxon_times"])
    ap.add_argument("--targets", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-positives", type=int, default=20,
                    help="below this the binary is degenerate and no MDE is defined")
    ap.add_argument("--inflation", type=float, default=2.3,
                    help="calibration from small audited strata; see module docstring")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res, lines = {}, []

    for stem in a.stems:
        z = np.load(require(f"{MASKS}/{stem}__masks.npz", "virobench_homology_audit.py",
                            f"search masks for {stem}"), allow_pickle=True)
        m = (~z["hasN"]) & (~z["hasT"]) & z["novel_genus"].astype(bool)
        g = z["genus"][m]
        lab = np.array([str(x) for x in z["label"][m]])
        n = int(m.sum())
        lines.append(f"\n{stem}   terminal stratum n={n:,}   {len(set(g)):,} genera")
        counts = pd.Series(lab).value_counts()
        lines.append("  class sizes  " + "  ".join(f"{k} {v}" for k, v in counts.items()))
        res[stem] = dict(n=n, n_genera=len(set(g)),
                         class_sizes={str(k): int(v) for k, v in counts.items()},
                         framings={})

        framings = [("8 class host, as ViroBench defines it", lab),
                    ("vertebrate host vs not", np.where(np.isin(lab, list(VERT)), 1, 0)),
                    ("human host vs not", np.where(lab == "D1", 1, 0))]
        for fname, y in framings:
            y = np.asarray(y)
            if len(np.unique(y)) < 2:
                lines.append(f"  {fname:40s} only one class present, not measurable")
                continue
            pos = int((y == 1).sum()) if set(np.unique(y)) <= {0, 1} else None
            # With a handful of positive rows the metric is driven by single rows.
            # A number computed there is an artifact of the degenerate binary, not a
            # power estimate, so none is reported and the count is shown instead.
            if pos is not None and pos < a.min_positives:
                res[stem]["framings"][fname] = dict(positives=pos, halfwidths=None,
                                                    mde="undefined",
                                                    reason=f"only {pos} positive rows")
                lines.append(f"  {fname} ({pos} positive rows)")
                lines.append(f"    MDE undefined, the positive class is too small for "
                             f"the metric to be driven by anything but single rows")
                continue
            hw = {}
            for t in a.targets:
                p = drive_to(y, t, rng)
                ach = matthews_corrcoef(y, p)
                h = halfwidth(y, p, g, rng, a.boot)
                hw[f"{t:.1f}"] = dict(achieved=round(float(ach), 3),
                                      halfwidth_row=round(h["row"], 3),
                                      halfwidth_genus=round(h["genus"], 3))
            res[stem]["framings"][fname] = dict(positives=pos, halfwidths=hw)
            tag = f" ({pos} positive rows)" if pos is not None else ""
            lines.append(f"  {fname}{tag}")
            lines.append(f"    {'target':>8s} {'achieved':>9s} {'raw genus':>10s} "
                         f"{'calibrated MDE':>15s}")
            for t, v in hw.items():
                cal = v["halfwidth_genus"] * a.inflation
                v["mde_calibrated"] = round(float(cal), 3)
                lines.append(f"    {t:>8s} {v['achieved']:>9.3f} "
                             f"{v['halfwidth_genus']:>10.3f} {cal:>15.3f}")

    txt = "\n".join(lines)
    open(data_path("virobench_power.txt"), "w").write(txt + "\n")
    json.dump(res, open(data_path("virobench_power.json"), "w"), indent=1)
    print(txt)
    print("\n  wrote reports/tables/virobench_power.txt and .json")


if __name__ == "__main__":
    main()
