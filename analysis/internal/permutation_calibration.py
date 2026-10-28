#!/usr/bin/env python
"""PRIORITY 3 — permutation calibration of the cross-fitted selection statistic.

CPU only. Writes only into reports/finalization/.

Purpose
-------
Establish empirically, for the populations the manuscript actually discusses,
whether the cross-fitted selection statistic is distinguishable from what the
same procedure produces under label permutation. This is NOT an equivalence
test and NOT a universal sample-size cutoff; it is a per-cell calibration.

Statistic
---------
The same seed-averaged cross-fitted statistic as Priority 2: seeds aggregated
within each predictor first, selection among predictors on training folds,
evaluation on the held-out fold, fold-size weighted. Reported both as

    pos_cf     cross-fitted ceiling over arms and foundation models
    drf_cf3    (pos_cf - env_cf) / (1 - env_cf), recomputed under each permutation

Permutation unit
----------------
Where recovered source groups exist, labels are permuted ACROSS SOURCE UNITS
with all child fragments carried together, so the biological dependence between
fragments of one genome is preserved under the null. The multiset of unit labels
is held fixed; because units differ in size, row-level class counts are
preserved in expectation rather than exactly, and this is recorded per cell.
Where no provenance exists, labels are permuted across rows, which preserves row
class counts exactly, and the cell is labelled `row`.

Folds are regenerated per permutation with the same grouping rule as the
observed statistic.

Usage
-----
    export VB_RESULTS=/path/to/results_viral_bench
    python permutation_calibration.py [--perms 1000]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root
sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
import final_regime_table as F  # noqa: E402
sys.path.insert(0, HERE)
from crossfit_headroom import mcc_counts, score, make_folds, crossfit, NAME, PHAGE_UNIT_REFUSED  # noqa: E402

# every population the manuscript discusses that is small, plus the anchor cell
KEY = {("HT hard_70", "L1"), ("HT hard_70", "L2"), ("HT std_95", "L2"),
       ("VHDB", "L1"), ("VHDB", "L2"), ("GUE phage", "L2"),
       ("Transm std", "L1"), ("Patho std", "L1"), ("Transm hard", "L1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sot = {}
    with open(os.path.join(REPO, "reports", "tables", "final_source_of_truth.csv")) as fh:
        for r in csv.DictReader(fh):
            sot[(r["benchmark"], r["regime"])] = r

    rows = []
    for name, mask, esc, atags, (ucsv, prov) in F.CELLS:
        short = NAME[name]
        if not any(k[0] == short for k in KEY):
            continue
        dev_model = sot.get((short, "L0"), {}).get("model")
        z = np.load(f"{F.MASKS}/{mask}.npz")
        hasN, hasT = z["hasN"].astype(bool), z["hasT"].astype(bool)
        n = len(hasN)
        got = F.model_preds(esc, dev_model, n)
        if got is None or got[0] is None:
            continue
        y_full = got[0]
        C = int(y_full.max()) + 1

        groups_full, gl = None, "row"
        if ucsv and os.path.exists(ucsv) and os.path.basename(ucsv) != PHAGE_UNIT_REFUSED:
            c = pd.read_csv(ucsv)
            col = [x for x in c.columns if x in ("cluster", "unit", "genome")]
            if col and len(c) == n:
                groups_full, gl = c[col[0]].astype(str).values, "recovered source genome"

        mods = {}
        for m in F.MODELS:
            ss = [np.load(f)["y_pred"] for f in
                  sorted(glob.glob(f"{F.FT}/{esc}__{m}__lr*__s*__preds.npz"))
                  if len(np.load(f)["y_true"]) == n]
            if ss:
                mods[f"model:{m}"] = ss
        arms = {f"arm:{k_}": list(v) for k_, v in F.arm_preds(atags or [esc], n).items()}

        for lab, selmask in (("L0", np.ones(n, bool)), ("L1", ~hasN), ("L2", ~hasN & ~hasT)):
            if (short, lab) not in KEY:
                continue
            idx = np.where(selmask)[0]
            if len(idx) < 30 or len(np.unique(y_full[idx])) < 2:
                continue
            y = y_full[idx]
            g = groups_full[idx] if groups_full is not None else None
            A_p = {k_: [p[idx] for p in v] for k_, v in arms.items()
                   if not (lab != "L0" and k_ == "arm:nn_lookup")}
            M_p = {k_: [p[idx] for p in v] for k_, v in mods.items()}
            full_p = {**A_p, **M_p}

            def stat(yy, rng):
                fold = make_folds(len(yy), g, a.folds, rng)
                A = {k_: [(C * yy + p).astype(np.int64) for p in v] for k_, v in A_p.items()}
                FU = {k_: [(C * yy + p).astype(np.int64) for p in v] for k_, v in full_p.items()}
                e = crossfit(yy, A, fold, a.folds, C)
                po = crossfit(yy, FU, fold, a.folds, C)
                d = (po - e) / (1 - e) if np.isfinite(e) and e < 1 else np.nan
                return po, d

            obs_pos, obs_drf = stat(y, np.random.default_rng(a.seed))

            if g is not None:
                uu, inv = np.unique(g, return_inverse=True)
                unit_lab = np.array([np.bincount(y[inv == i], minlength=C).argmax()
                                     for i in range(len(uu))])
                punit = "recovered source genome (unit-label multiset preserved)"
            else:
                punit = "row (class counts preserved exactly)"

            nulls_pos, nulls_drf = [], []
            for j in range(a.perms):
                rp = np.random.default_rng(10_000 + j)
                if g is not None:
                    yy = unit_lab[rp.permutation(len(unit_lab))][inv]
                else:
                    yy = y[rp.permutation(len(y))]
                if len(np.unique(yy)) < 2:
                    continue
                p_, d_ = stat(yy, np.random.default_rng(a.seed))
                if np.isfinite(p_):
                    nulls_pos.append(p_)
                if np.isfinite(d_):
                    nulls_drf.append(d_)
            np_arr = np.array(nulls_pos)
            nd_arr = np.array(nulls_drf)
            if len(np_arr) < 50:
                continue
            p_emp = float((np.sum(np_arr >= obs_pos) + 1) / (len(np_arr) + 1))
            rows.append(dict(
                benchmark=short, regime=lab, n_rows=len(y),
                n_groups=int(len(np.unique(g))) if g is not None else len(y),
                permutation_unit=punit, n_permutations=len(np_arr), n_folds=a.folds,
                observed_pos_cf=round(obs_pos, 4), observed_drf_cf3=round(obs_drf, 4),
                null_mean=round(float(np_arr.mean()), 4),
                null_sd=round(float(np_arr.std(ddof=1)), 4),
                null_p95=round(float(np.percentile(np_arr, 95)), 4),
                null_p99=round(float(np.percentile(np_arr, 99)), 4),
                null_max=round(float(np_arr.max()), 4),
                null_drf_p95=round(float(np.percentile(nd_arr, 95)), 4) if len(nd_arr) else "",
                empirical_p=round(p_emp, 5),
                exceeds_null_p95=bool(obs_pos > np.percentile(np_arr, 95)),
                exceeds_null_p99=bool(obs_pos > np.percentile(np_arr, 99))))
            print(f"  {short:13s} {lab}  n={len(y):6d} unit={punit.split(' (')[0]:26s} "
                  f"obs={obs_pos:7.4f} null mean={np_arr.mean():+.4f} sd={np_arr.std(ddof=1):.4f} "
                  f"p95={np.percentile(np_arr,95):.4f} p99={np.percentile(np_arr,99):.4f} "
                  f"p={p_emp:.4f} >p99={obs_pos>np.percentile(np_arr,99)}", flush=True)

    cols = ["benchmark", "regime", "n_rows", "n_groups", "permutation_unit",
            "n_permutations", "n_folds", "observed_pos_cf", "observed_drf_cf3",
            "null_mean", "null_sd", "null_p95", "null_p99", "null_max",
            "null_drf_p95", "empirical_p", "exceeds_null_p95", "exceeds_null_p99"]
    p = os.path.join(HERE, "permutation_calibration.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    json.dump({"perms": a.perms, "folds": a.folds, "seed": a.seed, "vb_results": F.RVB},
              open(os.path.join(HERE, "permutation_calibration_config.json"), "w"), indent=2)
    print(f"\nwrote {p}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
