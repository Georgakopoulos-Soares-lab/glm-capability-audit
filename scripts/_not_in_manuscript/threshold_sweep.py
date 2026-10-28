#!/usr/bin/env python
"""A. Sensitivity of the L1 conclusions to the minimum nucleotide identity.

No model is retrained. The frozen predictions are re-scored on the population each
threshold defines. The primary setting stays 0.30; this only asks how the answer moves.

Direction of effect, because it reads backwards easily: raising --min-seq-id makes the
criterion for DECLARING A HIT stricter, so fewer hits are declared and MORE sequences are
retained as having no detectable nucleotide relative. The retained population GROWS with
the threshold. The primary 0.30 is the most conservative point on this axis.

Masks are derived by filtering one permissive search on percent identity, using the frozen
pipeline's --max-seqs 20. That derivation is exact: on VHDB and GUE phage, at 0.70 and 0.90,
the retained ROW SETS from a true re-run and from filtering the 0.30 hits are identical.
The base masks at 0.30 also reproduce the frozen manuscript masks with zero disagreeing
rows on HT hard_70, VHDB and GUE phage.
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_regime_table as F          # frozen loaders, models, arms, verdict rule

REPO = os.path.dirname(os.path.dirname(HERE))
HITS = os.path.join(REPO, "reports", "threshold_sensitivity", "hits")
OUT = os.path.join(REPO, "reports", "threshold_sensitivity")
IDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
COVS = [0.10, 0.25, 0.50, 0.80]
# benchmark -> the hits file produced by threshold_sweep_search.sh
# keyed on F.CELLS' own benchmark names, which are the long form
HITFILE = {"HVUE host tropism hard_70": "HVUE_HT_hard_70",
           "HVUE host tropism std_95": "HVUE_HT_std_95",
           "HVUE pathogenicity hard": "HVUE_Patho_hard",
           "HVUE pathogenicity std": "HVUE_Patho_std",
           "HVUE transmissibility hard": "HVUE_Transm_hard",
           "HVUE transmissibility std": "HVUE_Transm_std",
           "VHDB within family": "VHDB_v2", "GUE phage": "GUE_phage_v2",
           "GUE virus_covid": "GUE_covid"}


def mask_for(hits, n, min_id, min_cov):
    """True where the test row has NO detectable nucleotide relative at this setting."""
    h = hits[(hits.pident >= min_id * 100) & (hits.qcov >= min_cov)]
    keep = np.ones(n, bool)
    keep[h.q.values] = False
    return keep


def score(y, mp, arms, rows, units, reps, seed=0):
    rng = np.random.default_rng(seed)
    y2, mp2 = y[rows], [p[rows] for p in mp]
    a2 = {k: [p[rows] for p in v] for k, v in arms.items()}
    if units is None:
        u, groups = np.arange(len(rows)), None
    else:
        uu = units[rows]; u = pd.unique(uu); groups = {c: np.where(uu == c)[0] for c in u}
    pt = {k: float(np.mean([F.mcc(y2, p) for p in v])) for k, v in a2.items()}
    earm = max(pt, key=pt.get)
    pm = float(np.mean([F.mcc(y2, p) for p in mp2]))
    d = []
    for _ in range(reps):
        pick = rng.choice(len(u), len(u), replace=True)
        r = pick if groups is None else np.concatenate([groups[u[i]] for i in pick])
        if len(np.unique(y2[r])) < 2:
            continue
        e = max(F.mcc(y2[r], v[rng.integers(len(v))][r]) for v in a2.values())
        x = F.mcc(y2[r], mp2[rng.integers(len(mp2))][r]) - e
        if np.isfinite(x):
            d.append(x)
    d = np.array(d)
    lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) > 20 else (np.nan, np.nan))
    return dict(model_mcc=round(pm, 4), baseline_mcc=round(pt[earm], 4), baseline_arm=earm,
                excess=round(pm - pt[earm], 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                verdict=(F.verdict(lo, hi) if len(d) > 20 else "NOT SCORED"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--mode", choices=["identity", "coverage"], default="identity")
    a = ap.parse_args()
    grid = [(i, 0.10) for i in IDS] if a.mode == "identity" else [(0.30, c) for c in COVS]
    rows = []
    for name, mask, esc, atags, (ucsv, prov) in F.CELLS:
        hf = os.path.join(HITS, f"{HITFILE[name]}.tsv")
        if not os.path.exists(hf):
            print(f"  {name}: no hits file, skipped"); continue
        hits = pd.read_csv(hf, sep="\t", header=None,
                           names=["q", "t", "pident", "qcov", "alnlen", "e", "bits"])
        z = np.load(f"{F.MASKS}/{mask}.npz"); n = len(z["hasN"])
        arms = F.arm_preds(atags or [esc], n)
        cand = {}
        for m in F.MODELS:
            y_, mp_, dev_ = F.model_preds(esc, m, n)
            if mp_: cand[m] = (y_, mp_, dev_)
        if not cand or len(arms) < 2:
            print(f"  {name}: inputs unavailable, skipped"); continue
        lead = max(cand, key=lambda m: cand[m][2])          # dev-selected, frozen
        y, mp, _ = cand[lead]
        units = None
        if ucsv and os.path.exists(ucsv):
            c = pd.read_csv(ucsv); col = [x for x in c.columns if x in ("cluster", "unit", "genome")]
            if col and len(c) == n: units = c[col[0]].astype(str).values
        for mid, mcov in grid:
            keep = mask_for(hits, n, mid, mcov)
            idx = np.where(keep)[0]
            base = dict(benchmark=name, min_seq_id=mid, coverage=mcov,
                        n_total=n, n_rows=len(idx), pct_retained=round(100 * len(idx) / n, 2),
                        n_positive=int(y[idx].sum()) if len(idx) else 0,
                        n_negative=int((1 - y[idx]).sum()) if len(idx) else 0,
                        glm=lead, bootstrap_unit=("genome" if units is not None else "row"),
                        unit_provenance=prov)
            if units is not None and len(idx):
                u = pd.Series(units[idx])
                base["n_units"] = int(u.nunique())
                base["assign_cover"] = round(float((~u.str.startswith("row")).mean()), 3)
            else:
                base["n_units"] = len(idx); base["assign_cover"] = None
            if len(idx) < 30 or len(np.unique(y[idx])) < 2:
                rows.append(dict(base, verdict="NOT SCORED")); continue
            keep_arms = {k: v for k, v in arms.items() if k not in F.REGIME_UNAVAILABLE}
            rows.append(dict(base, **score(y, mp, keep_arms, idx, units, a.reps)))
            print(f"  {name:12s} id={mid:.2f} cov={mcov:.2f} n={len(idx):6,} "
                  f"excess={rows[-1]['excess']:+.3f} {rows[-1]['verdict']}", flush=True)
    df = pd.DataFrame(rows)
    f = "identity_sweep.csv" if a.mode == "identity" else "coverage_sweep.csv"
    df.to_csv(os.path.join(OUT, f), index=False)
    print(f"\nwrote reports/threshold_sensitivity/{f}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
