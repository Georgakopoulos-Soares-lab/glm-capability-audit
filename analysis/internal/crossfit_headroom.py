#!/usr/bin/env python
"""PRIORITY 2 — cross-fitted headroom above the conventional comparator.

CPU only. Writes only into reports/finalization/.

Statistic
---------
Within every candidate predictor, seeds are aggregated FIRST, exactly as the
manuscript's point estimator does:

    score(p, R) = mean over p's training seeds of MCC(y_R, pred_{p,s}(R))

Cross-fitting then selects among candidate predictors only. Individual seeds
never compete as separate candidates, no decision threshold is tuned, and no
hyperparameter is tuned. The foundation model removed for the leave-one-model-out
variant is the DEVELOPMENT-selected model recorded in
reports/tables/final_source_of_truth.csv, never whichever model looks best on
test.

Folds
-----
Where biological source groups are available the folds are GROUPED by recovered
source genome, so fragments from one source cannot appear in both the
selector-training folds and the selector-evaluation fold. Row-level folds are
used only where no grouping is possible and are labelled as such.

    VHDB            native unit file absent on this cluster -> ROW folds
    HVUE tasks      recovered source genomes -> GROUPED, coverage reported
    GUE phage       unit file exists at 12.5% assignment, refused by the frozen
                    pipeline as "a row bootstrap wearing a unit column" -> ROW
    GUE covid       no provenance -> ROW

Quantities
----------
    env_cf      cross-fitted comparator: conventional arms only
    pos_cf      cross-fitted ceiling over arms AND foundation models
    lomo_cf     cross-fitted ceiling with the dev-selected model removed

    drf_cf3   = (pos_cf  - env_cf) / (1 - env_cf)
    drf_lomo3 = (lomo_cf - env_cf) / (1 - env_cf)

"No measurable cross-fitted headroom" is defined operationally as
drf_cf3 <= DRF_EPS (default 0.02); the word "saturated" is not used.

Usage
-----
    export VB_RESULTS=/path/to/results_viral_bench
    python crossfit_headroom.py [--folds 5]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root
sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
import final_regime_table as F  # noqa: E402

DRF_EPS = 0.02
NAME = {"HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
        "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
        "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
        "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid"}
PHAGE_UNIT_REFUSED = "gue__GUE_phage_fragments__genomeunits.csv"


def mcc_counts(cnt, C):
    M = cnt.reshape(C, C)
    s = M.sum()
    if s <= 0:
        return np.nan
    c = np.trace(M)
    t, p = M.sum(axis=1), M.sum(axis=0)
    num = c * s - float(p @ t)
    den = np.sqrt(max(s * s - float(p @ p), 0.)) * np.sqrt(max(s * s - float(t @ t), 0.))
    return float(num / den) if den > 0 else 0.0


def score(codes, w, C):
    """Seed-averaged MCC under row indicator weights w."""
    vals = [mcc_counts(np.bincount(c, weights=w, minlength=C * C), C) for c in codes]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def make_folds(n, groups, k, rng):
    """Fold id per row. Groups are kept intact when provided."""
    if groups is None:
        f = np.full(n, -1)
        f[rng.permutation(n)] = np.repeat(np.arange(k), int(np.ceil(n / k)))[:n]
        return f
    uu, inv = np.unique(groups, return_inverse=True)
    gf = np.empty(len(uu), dtype=int)
    gf[rng.permutation(len(uu))] = np.repeat(np.arange(k), int(np.ceil(len(uu) / k)))[:len(uu)]
    return gf[inv]


def crossfit(y, cand, fold, k, C):
    """Select on training folds by seed-averaged MCC, score on the held-out fold."""
    tot, tw = 0.0, 0.0
    for j in range(k):
        te = (fold == j)
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        wtr, wte = tr.astype(np.float64), te.astype(np.float64)
        best, bs = None, -np.inf
        for nm, codes in cand.items():
            s = score(codes, wtr, C)
            if np.isfinite(s) and s > bs:
                bs, best = s, nm
        if best is None:
            continue
        h = score(cand[best], wte, C)
        if np.isfinite(h):
            tot += h * te.sum()
            tw += te.sum()
    return float(tot / tw) if tw else np.nan


def main():
    ap = argparse.ArgumentParser()
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
        dev_model = sot.get((short, "L0"), {}).get("model")
        if not dev_model:
            continue
        z = np.load(f"{F.MASKS}/{mask}.npz")
        hasN, hasT = z["hasN"].astype(bool), z["hasT"].astype(bool)
        n = len(hasN)
        got = F.model_preds(esc, dev_model, n)
        if got is None or got[0] is None:
            continue
        y_full = got[0]
        C = int(y_full.max()) + 1

        groups_full, grp_label, cover = None, "row", ""
        if ucsv and os.path.exists(ucsv) and os.path.basename(ucsv) != PHAGE_UNIT_REFUSED:
            c = pd.read_csv(ucsv)
            col = [x for x in c.columns if x in ("cluster", "unit", "genome")]
            if col and len(c) == n:
                groups_full = c[col[0]].astype(str).values
                grp_label = "recovered source genome"
        elif ucsv and os.path.basename(ucsv) == PHAGE_UNIT_REFUSED:
            grp_label = "row (unit file refused, 12.5% assignment)"
        elif ucsv:
            grp_label = "row (native unit file absent)"

        # candidate pools, seed lists per predictor
        import glob
        mods = {}
        for m in F.MODELS:
            ss = []
            for f in sorted(glob.glob(f"{F.FT}/{esc}__{m}__lr*__s*__preds.npz")):
                d = np.load(f)
                if len(d["y_true"]) == n:
                    ss.append(d["y_pred"])
            if ss:
                mods[f"model:{m}"] = ss
        arms = {f"arm:{k_}": list(v) for k_, v in F.arm_preds(atags or [esc], n).items()}

        for lab, selmask in (("L0", np.ones(n, bool)), ("L1", ~hasN), ("L2", ~hasN & ~hasT)):
            idx = np.where(selmask)[0]
            if len(idx) < 30 or len(np.unique(y_full[idx])) < 2:
                continue
            y = y_full[idx]
            g = groups_full[idx] if groups_full is not None else None
            A = {k_: [(C * y + p[idx]).astype(np.int64) for p in v] for k_, v in arms.items()
                 if not (lab != "L0" and k_ == "arm:nn_lookup")}
            M = {k_: [(C * y + p[idx]).astype(np.int64) for p in v] for k_, v in mods.items()}
            if not A:
                continue
            full = {**A, **M}
            lomo = {k_: v for k_, v in full.items() if k_ != f"model:{dev_model}"}

            for mode, gg in (("grouped", g), ("row", None)):
                if mode == "grouped" and gg is None:
                    continue
                fold = make_folds(len(y), gg, a.folds, np.random.default_rng(a.seed))
                env = crossfit(y, A, fold, a.folds, C)
                pos = crossfit(y, full, fold, a.folds, C)
                lom = crossfit(y, lomo, fold, a.folds, C)
                d_cf = (pos - env) / (1 - env) if np.isfinite(env) and env < 1 else np.nan
                d_lo = (lom - env) / (1 - env) if np.isfinite(env) and env < 1 else np.nan
                ngroups = int(len(np.unique(gg))) if gg is not None else len(y)
                rows.append(dict(
                    benchmark=short, regime=lab, fold_mode=mode,
                    grouping_unit=(grp_label if mode == "grouped" else "row"),
                    n_rows=len(y), n_groups=ngroups, n_folds=a.folds,
                    group_coverage=(round(float(np.mean([not str(u).startswith("row")
                                                         for u in gg])), 4)
                                    if gg is not None else ""),
                    dev_selected_model=dev_model,
                    manuscript_comparator=sot.get((short, lab), {}).get("envelope_arm", ""),
                    env_cf=round(env, 4), pos_cf=round(pos, 4), lomo_cf=round(lom, 4),
                    drf_cf3=round(d_cf, 4), drf_lomo3=round(d_lo, 4),
                    no_measurable_headroom=bool(np.isfinite(d_cf) and d_cf <= DRF_EPS),
                    headroom_survives_lomo=bool(np.isfinite(d_lo) and d_lo > DRF_EPS),
                    test_derived_selection="no"))
                print(f"  {short:13s} {lab} {mode:8s} grp={ngroups:6d} env={env:7.4f} "
                      f"pos={pos:7.4f} lomo={lom:7.4f} drf={d_cf:7.4f} lomo_drf={d_lo:7.4f}",
                      flush=True)

    cols = ["benchmark", "regime", "fold_mode", "grouping_unit", "n_rows", "n_groups",
            "n_folds", "group_coverage", "dev_selected_model", "manuscript_comparator",
            "env_cf", "pos_cf", "lomo_cf", "drf_cf3", "drf_lomo3",
            "no_measurable_headroom", "headroom_survives_lomo", "test_derived_selection"]
    p = os.path.join(HERE, "crossfit_headroom.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    json.dump({"folds": a.folds, "seed": a.seed, "drf_eps": DRF_EPS,
               "vb_results": F.RVB, "n_rows": len(rows)},
              open(os.path.join(HERE, "crossfit_headroom_config.json"), "w"), indent=2)
    print(f"\nwrote {p}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
