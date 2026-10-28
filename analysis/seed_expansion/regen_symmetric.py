"""Recompute the symmetric check with the manuscript estimator.

final_regime_table.score_sym draws a SINGLE random fit per method inside each
replicate, on both sides. Methods defines a method's score as the mean MCC across
its fits, and the symmetric statistic maximises over five models and five to seven
arms, so a single-draw max is biased upward on both sides and does not cancel.

Rather than re-running final_regime_table.py, which hard-fails on the absent VHDB
unit file, this recomputes sym_* for every cell directly from the scored tree using
the same helpers bootstrap.py uses, so the two cannot drift.

Source units are used where a unit file exists. Where it does not, the cell is
emitted with provenance NO_UNITS and the frozen value is left in place by the merge.
"""
import csv, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import final_regime_table as F
from bootstrap import codes_for, seed_avg, verdict

REPS, SEED = 5000, 0
OUT = os.path.join(REPO, "reports", "finalization", "symmetric_regenerated.csv")
NAME = {"HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
        "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
        "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
        "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid"}


def sym_cell(y, fms, arms, rows, units):
    y2 = y[rows]
    allp = list(fms.values()) + list(arms.values())
    C = int(max(y2.max(), max(p[rows].max() for v in allp for p in v)) + 1)
    fc = {k: codes_for(y2, [p[rows] for p in v], C) for k, v in fms.items()}
    ac = {k: codes_for(y2, [p[rows] for p in v], C) for k, v in arms.items()}
    full = np.ones(len(y2))
    pt_f = max(seed_avg(c, full, C) for c in fc.values())
    pt_b = max(seed_avg(c, full, C) for c in ac.values())
    if units is None:
        inv, k = np.arange(len(rows)), len(rows)
    else:
        _u, inv = np.unique(units[rows], return_inverse=True); k = len(_u)
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(REPS):
        w = rng.multinomial(k, np.full(k, 1.0 / k)).astype(np.float64)[inv]
        if len(np.unique(y2[w > 0])) < 2:
            continue
        f = max(seed_avg(c, w, C, rng) for c in fc.values())
        b = max(seed_avg(c, w, C, rng) for c in ac.values())
        if np.isfinite(f - b):
            d.append(f - b)
    lo, hi = np.percentile(np.asarray(d), [2.5, 97.5])
    return pt_f - pt_b, lo, hi, verdict(lo, hi), k, len(d)


def main():
    rows_out = []
    for name, mask, esc, atags, (ucsv, prov) in F.CELLS:
        z = np.load(f"{F.MASKS}/{mask}.npz"); hasN, hasT = z["hasN"], z["hasT"]; n = len(hasN)
        arms = F.arm_preds(atags or [esc], n)
        fms = {}
        y = None
        for m in F.MODELS:
            yy, ps, _ = F.model_preds(esc, m, n)
            if ps:
                fms[m] = ps; y = yy
        if y is None or not arms:
            continue
        units = None
        if ucsv and os.path.exists(ucsv):
            units = pd.read_csv(ucsv)["cluster"].astype(str).values
        for lab, idx in (("L0", np.arange(n)), ("L1", np.where(~hasN)[0]),
                         ("L2", np.where(~hasN & ~hasT)[0])):
            if len(idx) < 20:
                continue
            keep = {k: v for k, v in arms.items()
                    if not (lab != "L0" and k in F.REGIME_UNAVAILABLE)}
            if len(np.unique(y[idx])) < 2:
                continue
            e, lo, hi, vd, k, nd = sym_cell(y, fms, keep, idx, units)
            rows_out.append(dict(benchmark=NAME.get(name, name), regime=lab,
                                 sym_excess=round(float(e), 4), sym_lo=round(float(lo), 4),
                                 sym_hi=round(float(hi), 4), sym_verdict=vd,
                                 sym_units=k, sym_draws=nd, n_fms=len(fms),
                                 sym_provenance="REGENERATED" if units is not None else "NO_UNITS"))
            print(f"  {NAME.get(name,name):12s} {lab}  sym {e:+.4f} [{lo:+.4f},{hi:+.4f}] {vd:<12s} "
                  f"units={k:,} fms={len(fms)}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print(f"\n  wrote {OUT}  ({len(rows_out)} cells)")


if __name__ == "__main__":
    main()
