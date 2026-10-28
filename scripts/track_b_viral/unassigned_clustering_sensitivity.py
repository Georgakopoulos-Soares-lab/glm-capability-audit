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


def score_draws(y, mp, arms, rows, units, reps, boot_seed):
    """Replicate differences under the MANUSCRIPT estimator.

    This previously mirrored final_regime_table.score, which draws a SINGLE random fit
    per replicate. Methods defines a method's score as the mean MCC across its fits, and
    bootstrap_corrected.py implements that, so the two code paths estimated different
    quantities. The gap was negligible at three fits and material at ten: on host tropism
    hard_70 L1 the single draw version gave a 0.065 MCC wider interval and, because the
    comparator is a max over arms, a single lucky arm draw also inflated the comparator.
    The helpers are imported from bootstrap_corrected rather than reimplemented so the
    two cannot drift again. Changed 2026-10-28.
    """
    import os as _os, sys as _sys
    _an = _os.path.join(_os.path.dirname(_os.path.dirname(HERE)), "analysis")
    if _an not in _sys.path:
        _sys.path.insert(0, _an)
    from bootstrap_corrected import codes_for, seed_avg

    y2 = y[rows]
    C = int(max(y2.max(), max(p[rows].max() for p in mp)) + 1)
    m_codes = codes_for(y2, [p[rows] for p in mp], C)
    a_codes = {k: codes_for(y2, [p[rows] for p in v], C) for k, v in arms.items()}

    uu = units[rows]
    uniq, inv = np.unique(uu, return_inverse=True)
    k = len(uniq)
    rng = np.random.default_rng(boot_seed)
    d = []
    for _ in range(reps):
        w = rng.multinomial(k, np.full(k, 1.0 / k)).astype(np.float64)[inv]
        if len(np.unique(y2[w > 0])) < 2:
            continue
        e = max(seed_avg(cs, w, C, rng) for cs in a_codes.values())
        x = seed_avg(m_codes, w, C, rng) - e
        if np.isfinite(x):
            d.append(x)
    return np.asarray(d), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partitions", type=int, default=20,
                    help="independent pseudo-cluster draws to pool over")
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
            # "none" and "worst" are deterministic given the unit file, so one draw is
            # the whole story. The other three depend on an arbitrary shuffle, so the
            # partition is redrawn and the replicate statistics are POOLED across draws:
            # the interval then marginalises over the partition instead of conditioning
            # on one. Partition seed varies, bootstrap seed is held fixed, so the extra
            # width is attributable to the grouping and not to resampling.
            ndraw = 1 if sc in ("none", "worst") else a.partitions
            pooled, nunits, per_draw = [], [], []
            for p_i in range(ndraw):
                rng = np.random.default_rng(a.seed + 1000 * p_i)
                uu = pseudo_units(units, rng, sc)
                dd, nu = score_draws(y, mp, keep, rows, uu, a.reps, a.seed)
                pooled.append(dd); nunits.append(nu)
                if len(dd):
                    per_draw.append(np.percentile(dd, [2.5, 97.5]))
            d = np.concatenate(pooled)
            lo, hi = np.percentile(d, [2.5, 97.5])
            # width attributable to the partition: spread of the per-draw endpoints
            if len(per_draw) > 1:
                pw = np.array(per_draw)
                part_sd = float(np.mean([pw[:, 0].std(ddof=1), pw[:, 1].std(ddof=1)]))
                within = float(np.mean(pw[:, 1] - pw[:, 0]))
            else:
                part_sd, within = 0.0, float(hi - lo)
            # computed from the tree being scored, exactly as F.score does it, rather
            # than carried from the frozen final_regime_table.json, which is still at
            # three fits per method.
            _pm = float(np.mean([matthews_corrcoef(y[rows], p[rows]) for p in mp]))
            _pt = {k: float(np.mean([matthews_corrcoef(y[rows], p[rows]) for p in v]))
                   for k, v in keep.items()}
            _earm = max(_pt, key=_pt.get)
            r = dict(model_mcc=round(_pm, 4), envelope=round(_pt[_earm], 4),
                     envelope_arm=_earm,
                     excess=round(_pm - _pt[_earm], 4),
                     lo=round(float(lo), 4), hi=round(float(hi), 4),
                     half_width=round(float(hi - lo) / 2, 4),
                     verdict=F.verdict(lo, hi),
                     n_rows=len(rows), n_units=int(round(float(np.mean(nunits)))),
                     n_partitions=ndraw, reps_per_partition=a.reps,
                     pooled_draws=int(len(d)),
                     within_partition_width=round(within, 4),
                     partition_endpoint_sd=round(part_sd, 4))
            print(f"    {sc:10s} {r['n_units']:>8,} {r['excess']:>+8.3f} "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}]  {r['verdict']:<12s} "
                  f"draws={ndraw:>2d} partSD={part_sd:.4f}")
            out.append(dict(benchmark=name, regime=t["regime"], model=t["model"],
                            scenario=sc, **r))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n  'worst' collapses every unplaced fragment into one group and is a bound,")
    print(f"  not a plausible scenario. 'observed' is the one to read.")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
