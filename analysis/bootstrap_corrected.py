#!/usr/bin/env python
"""PRIORITY 1 — re-estimate every manuscript interval with the reported estimator.

CPU only. Writes only into reports/finalization/.

The estimator the manuscript reports
------------------------------------
For a population R (a set of test rows), with the foundation model m fixed by
development selection and every baseline arm's hyperparameters fixed by
development selection:

    MCC_p(R)      = mean over predictor p's training seeds of MCC(y_R, pred_{p,s}(R))
    Envelope(R)   = max over APPLICABLE arms a of MCC_a(R)
    Delta(R)      = MCC_m(R) - Envelope(R)

"Applicable" excludes nearest-neighbour lookup at L1 and L2, where it is
unavailable by construction; it is never encoded as zero, it is removed from the
set the maximum is taken over. The point estimate is Delta(R_full).

The defect being fixed
----------------------
`final_regime_table.py:161-162` draws ONE RANDOM SEED per arm inside each
bootstrap replicate and then takes the max, whereas the point estimate averages
seeds before the max. Maximising over noisier single-seed quantities inflates
the envelope by Jensen, so the interval is an interval for a different
population quantity. Measured on GUE phage L0 the offset is +0.0187 and does not
shrink with n.

Here the replicate statistic is Delta applied to the resampled rows, with seed
aggregation identical to the point estimate. The envelope is still re-maximised
inside each replicate, because "max over applicable arms" is part of the
estimator, but individual seeds never compete as separate predictors and the
model is never reselected.

Bootstrap schemes
-----------------
fragment   resample rows with replacement.
source     resample recovered source units with replacement, carrying all rows
           of each sampled unit. Assigned fragments move together by recovered
           parent genome; unassigned rows keep the existing singleton
           convention. This is a recovered-source / partial-source bootstrap,
           NOT a fully genome-corrected one, and coverage is reported per cell.

Implementation note
-------------------
MCC depends only on the joint counts of (true, predicted), so a resample is
applied as integer row weights and the confusion counts come from a weighted
bincount over the precomputed code C*y+p. This is exact, not an approximation,
and avoids gathering large arrays 5,000 times.

Usage
-----
    export VB_RESULTS=/path/to/results_viral_bench
    python bootstrap_corrected.py [--reps 5000] [--workers 16]
"""
import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "reports", "finalization")
sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
import final_regime_table as F  # noqa: E402

EQ_MARGIN = 0.02
NAME = {"HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
        "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
        "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
        "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid"}
# the frozen pipeline refuses this unit file: 12.5% assignment is a row bootstrap
# wearing a unit column. Respected here, and the coverage is reported.
PHAGE_UNIT_REFUSED = "gue__GUE_phage_fragments__genomeunits.csv"


def mcc_from_counts(cnt, C):
    """Multiclass MCC from a flattened CxC confusion matrix of weighted counts."""
    M = cnt.reshape(C, C)                 # rows = true, cols = predicted
    s = M.sum()
    if s <= 0:
        return np.nan
    c = np.trace(M)
    t = M.sum(axis=1)                     # true totals
    p = M.sum(axis=0)                     # predicted totals
    num = c * s - float(p @ t)
    den = np.sqrt(max(s * s - float(p @ p), 0.0)) * np.sqrt(max(s * s - float(t @ t), 0.0))
    return float(num / den) if den > 0 else 0.0


def codes_for(y, preds, C):
    """C*y + pred, one vector per seed, so a weighted bincount gives the counts."""
    return [(C * y + p).astype(np.int64) for p in preds]


def seed_avg(codes, w, C, rng=None):
    """Mean over seeds of MCC under row weights w. Identical rule to the point estimate.

    With `rng`, the seed set is itself resampled with replacement before averaging,
    which propagates seed uncertainty into the interval for a k-seed mean (the
    two-level scheme). Without it, the trained predictors are treated as fixed and
    the interval reflects sampling over rows or units only.
    """
    idx = range(len(codes)) if rng is None else rng.integers(len(codes), size=len(codes))
    vals = [mcc_from_counts(np.bincount(codes[i], weights=w, minlength=C * C), C) for i in idx]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def verdict(lo, hi):
    if not np.isfinite(lo):
        return ""
    if lo > 0:
        return "SUPERIOR"
    if hi < 0:
        return "INFERIOR"
    # EQUIVALENT was removed at finalization. Declaring equivalence needs a margin,
    # and the 0.02 MCC margin had no external justification; it was also smaller than
    # the median seed-to-seed spread of 0.027 MCC. An interval that spans zero is now
    # INCONCLUSIVE, and its width is reported so precision is visible.
    return "INCONCLUSIVE"


def run_cell(job):
    (short, lab, y, model_codes, arm_codes, C, units, reps, seed) = job
    n = len(y)
    full_w = np.ones(n, dtype=np.float64)
    m_pt = seed_avg(model_codes, full_w, C)
    arm_pt = {a: seed_avg(cs, full_w, C) for a, cs in arm_codes.items()}
    earm = max(arm_pt, key=arm_pt.get)
    env_pt = arm_pt[earm]
    excess = m_pt - env_pt

    out = dict(benchmark=short, regime=lab, n_rows=n, model_mcc=round(m_pt, 4),
               baseline_arm=earm, baseline_mcc=round(env_pt, 4), excess=round(excess, 4),
               n_arms=len(arm_codes))

    def boot(unit_vec, seed_resample=False):
        rng = np.random.default_rng(seed)
        d, skipped = [], 0
        inv = k = None
        if unit_vec is not None:
            uu, inv = np.unique(unit_vec, return_inverse=True)
            k = len(uu)
        for _ in range(reps):
            if unit_vec is None:
                w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
            else:
                w = rng.multinomial(k, np.full(k, 1.0 / k)).astype(np.float64)[inv]
            if len(np.unique(y[w > 0])) < 2:
                skipped += 1
                continue
            sr = rng if seed_resample else None
            e = max(seed_avg(cs, w, C, sr) for cs in arm_codes.values())
            d.append(seed_avg(model_codes, w, C, sr) - e)
        d = np.array([x for x in d if np.isfinite(x)])
        if len(d) < 50:
            return np.nan, np.nan, skipped, len(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(lo), float(hi), skipped, len(d)

    flo, fhi, fskip, fkept = boot(None)
    out.update(frag_lo=round(flo, 4) if np.isfinite(flo) else "",
               frag_hi=round(fhi, 4) if np.isfinite(fhi) else "",
               frag_verdict=verdict(flo, fhi), frag_reps_used=fkept, frag_reps_skipped=fskip)
    # two-level: rows/units AND seeds resampled, propagating seed uncertainty
    f2lo, f2hi, _s2, _k2 = boot(None, seed_resample=True)
    out.update(frag2_lo=round(f2lo, 4) if np.isfinite(f2lo) else "",
               frag2_hi=round(f2hi, 4) if np.isfinite(f2hi) else "",
               frag2_verdict=verdict(f2lo, f2hi))

    if units is not None:
        slo, shi, sskip, skept = boot(units)
        s2lo, s2hi, _a, _b = boot(units, seed_resample=True)
        out.update(src2_lo=round(s2lo, 4) if np.isfinite(s2lo) else "",
                   src2_hi=round(s2hi, 4) if np.isfinite(s2hi) else "",
                   src2_verdict=verdict(s2lo, s2hi))
        nu = int(len(np.unique(units)))
        real = int(sum(1 for u in np.unique(units) if not str(u).startswith("row")))
        out.update(n_units=nu, n_real_units=real,
                   assign_cover=round(float(np.mean([not str(u).startswith("row")
                                                     for u in units])), 4),
                   src_lo=round(slo, 4) if np.isfinite(slo) else "",
                   src_hi=round(shi, 4) if np.isfinite(shi) else "",
                   src_verdict=verdict(slo, shi), src_reps_used=skept, src_reps_skipped=sskip)
    else:
        out.update(n_units="", n_real_units="", assign_cover="", src_lo="", src_hi="",
                   src_verdict="not estimable", src_reps_used="", src_reps_skipped="",
                   src2_lo="", src2_hi="", src2_verdict="not estimable")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sot = {}
    with open(os.path.join(REPO, "reports", "tables", "final_source_of_truth.csv")) as fh:
        for r in csv.DictReader(fh):
            sot[(r["benchmark"], r["regime"])] = r

    jobs, notes = [], []
    for name, mask, esc, atags, (ucsv, prov) in F.CELLS:
        short = NAME[name]
        sel = sot.get((short, "L0"), {}).get("model")
        if not sel:
            continue
        z = np.load(f"{F.MASKS}/{mask}.npz")
        hasN, hasT = z["hasN"].astype(bool), z["hasT"].astype(bool)
        n = len(hasN)
        got = F.model_preds(esc, sel, n)
        if got is None or got[0] is None:
            continue
        y_full, mpreds, _dev = got
        C = int(y_full.max()) + 1
        arms_full = {k: v for k, v in F.arm_preds(atags or [esc], n).items()}

        unit_full = None
        if ucsv and os.path.exists(ucsv) and os.path.basename(ucsv) != PHAGE_UNIT_REFUSED:
            c = pd.read_csv(ucsv)
            col = [x for x in c.columns if x in ("cluster", "unit", "genome")]
            if col and len(c) == n:
                unit_full = c[col[0]].astype(str).values
        elif ucsv and not os.path.exists(ucsv):
            notes.append(f"{short}: unit file ABSENT ({ucsv}); source interval NOT AUDITABLE")
        elif ucsv and os.path.basename(ucsv) == PHAGE_UNIT_REFUSED:
            notes.append(f"{short}: unit file present but refused by the frozen pipeline "
                         f"(12.5% assignment); source interval reported as not estimable")

        for lab, selmask in (("L0", np.ones(n, bool)), ("L1", ~hasN), ("L2", ~hasN & ~hasT)):
            idx = np.where(selmask)[0]
            if len(idx) < 30 or len(np.unique(y_full[idx])) < 2:
                continue
            y = y_full[idx]
            keep = {k: v for k, v in arms_full.items()
                    if not (lab != "L0" and k in F.REGIME_UNAVAILABLE)}
            if len(keep) < 2:
                continue
            jobs.append((short, lab, y,
                         codes_for(y, [p[idx] for p in mpreds], C),
                         {k: codes_for(y, [p[idx] for p in v], C) for k, v in keep.items()},
                         C, (unit_full[idx] if unit_full is not None else None),
                         a.reps, a.seed))

    print(f"{len(jobs)} cells, {a.reps} replicates, {a.workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_cell, jobs):
            rows.append(r)
            print(f"  {r['benchmark']:13s} {r['regime']}  n={r['n_rows']:6d} "
                  f"arm={r['baseline_arm']:10s} excess={r['excess']:+.4f} "
                  f"frag=[{r['frag_lo']},{r['frag_hi']}] {r['frag_verdict']:12s} "
                  f"src=[{r['src_lo']},{r['src_hi']}] {r['src_verdict']}", flush=True)

    order = {(r["benchmark"], r["regime"]): r for r in rows}
    for r in rows:
        old = sot.get((r["benchmark"], r["regime"]), {})
        for k_new, k_old in (("excess", "excess"), ("frag_verdict", "frag_verdict"),
                             ("src_verdict", "src_verdict"), ("baseline_arm", "envelope_arm")):
            r[f"manuscript_{k_old}"] = old.get(k_old, "")
        changed = []
        if old.get("excess") and abs(float(old["excess"]) - r["excess"]) > 0.0005:
            changed.append("excess")
        if old.get("frag_verdict") and old["frag_verdict"] != r["frag_verdict"]:
            changed.append("frag_verdict")
        if old.get("src_verdict") and old["src_verdict"] not in ("", "unavailable") \
                and r["src_verdict"] not in ("", "not estimable") \
                and old["src_verdict"] != r["src_verdict"]:
            changed.append("src_verdict")
        r["changed_from_manuscript"] = ",".join(changed) if changed else "no"

    cols = ["benchmark", "regime", "n_rows", "n_units", "n_real_units", "assign_cover",
            "baseline_arm", "manuscript_envelope_arm", "model_mcc", "baseline_mcc",
            "excess", "manuscript_excess", "frag_lo", "frag_hi", "frag_verdict",
            "manuscript_frag_verdict", "frag2_lo", "frag2_hi", "frag2_verdict",
            "src_lo", "src_hi", "src_verdict", "manuscript_src_verdict",
            "src2_lo", "src2_hi", "src2_verdict", "n_arms", "frag_reps_used", "frag_reps_skipped",
            "src_reps_used", "src_reps_skipped", "changed_from_manuscript"]
    rows.sort(key=lambda r: (r["benchmark"], r["regime"]))
    p = os.path.join(OUT, "bootstrap_corrected.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    json.dump({"reps": a.reps, "seed": a.seed, "eq_margin": EQ_MARGIN,
               "vb_results": F.RVB, "n_cells": len(rows), "notes": notes},
              open(os.path.join(OUT, "bootstrap_corrected_config.json"), "w"), indent=2)
    print("\nnotes:")
    for x in notes:
        print("  " + x)
    print(f"\nwrote {p}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
