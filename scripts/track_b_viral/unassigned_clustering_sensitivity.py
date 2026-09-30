#!/usr/bin/env python
"""Does a resolved recovered-unit verdict survive plausible hidden clustering?

Reference-based source recovery places only some fragments on a genome. Every
unplaced fragment is currently its own singleton unit, which assumes it is
independent of everything else. That assumption is false in the direction that
narrows the interval, so a SUPERIOR verdict resting on partially recovered
units has an unquantified anti-conservative component.

This calibrates it. Among fragments that WERE placed, we observe the real
distribution of fragments per genome. We then pseudo-cluster the unplaced
fragments by drawing group sizes from that observed distribution, under several
scenarios for how concentrated the hidden clustering is, and repeat the whole
bootstrap. If the verdict holds across scenarios it is not an artefact of
treating unplaced rows as independent.

Scenarios, in increasing severity.
  observed    draw hidden group sizes from the empirical assigned-size distribution
  mean        every hidden group is the mean assigned size
  pooled      hidden groups are twice the mean assigned size
  worst       all unplaced fragments collapse to a single group

"worst" is deliberately absurd. It is the bound, not a belief.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_regime_table as F


def pseudo_units(units, rng, scenario):
    """Replace singleton labels with drawn group labels; assigned rows untouched."""
    u = pd.Series(units).astype(str)
    is_single = u.str.startswith("row").values
    if not is_single.any() or scenario == "none":
        return units.copy()
    sizes = u[~is_single].value_counts().values
    if len(sizes) == 0:
        return units.copy()
    idx = np.where(is_single)[0]
    rng.shuffle(idx)
    out = units.copy()
    if scenario == "worst":
        out[idx] = "hidden0"
        return out
    if scenario == "mean":
        draw = lambda: max(1, int(round(sizes.mean())))
    elif scenario == "pooled":
        draw = lambda: max(1, int(round(2 * sizes.mean())))
    else:  # observed
        draw = lambda: int(rng.choice(sizes))
    i = g = 0
    while i < len(idx):
        k = draw()
        out[idx[i:i + k]] = f"hidden{g}"
        i += k; g += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/layered/unassigned_sensitivity.json")
    a = ap.parse_args()

    # every cell whose verdict is resolved AND whose units are only partly recovered
    res = json.load(open("reports/layered/final_regime_table.json"))
    targets = [r for r in res if not r.get("skipped")
               and r.get("src_verdict") in ("SUPERIOR", "INFERIOR", "EQUIVALENT")
               and r.get("assign_cover") is not None and r["assign_cover"] < 0.999]
    if not targets:
        print("no resolved verdict rests on partially recovered units; nothing to test")
        json.dump([], open(a.out, "w")); return
    spec = {c[0]: c for c in F.CELLS}
    out = []
    for t in targets:
        name, mask, esc, atags, (ucsv, prov) = spec[t["benchmark"]]
        z = np.load(f"{F.MASKS}/{mask}.npz"); hasN, hasT = z["hasN"], z["hasT"]; n = len(hasN)
        arms = F.arm_preds(atags or [esc], n)
        y, mp, _ = F.model_preds(esc, t["model"], n)
        units = pd.read_csv(ucsv)["cluster"].astype(str).values
        rows = {"L0": np.arange(n), "L1": np.where(~hasN)[0],
                "L2": np.where(~hasN & ~hasT)[0]}[t["regime"]]
        keep = {k: v for k, v in arms.items()
                if not (t["regime"] != "L0" and k in F.REGIME_UNAVAILABLE)}
        print(f"\n=== {name} {t['regime']} / {t['model']}   "
              f"source-aware verdict {t['src_verdict']} at {t['excess']:+.3f} "
              f"[{t['src_lo']:+.3f},{t['src_hi']:+.3f}], assignment "
              f"{100*t['assign_cover']:.0f}%")
        print(f"    {'scenario':10s} {'units':>8s} {'excess':>8s} {'95% CI':>18s}  verdict")
        for sc in ("none", "observed", "mean", "pooled", "worst"):
            rng = np.random.default_rng(a.seed)
            uu = pseudo_units(units, rng, sc)
            r = F.score(y, mp, keep, rows, uu, a.reps, a.seed)
            print(f"    {sc:10s} {r['n_units']:>8,} {r['excess']:>+8.3f} "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}]  {r['verdict']}")
            out.append(dict(benchmark=name, regime=t["regime"], model=t["model"],
                            scenario=sc, **r))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n  'worst' collapses every unplaced fragment into one group and is a bound,")
    print(f"  not a plausible scenario. 'observed' is the one to read.")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
