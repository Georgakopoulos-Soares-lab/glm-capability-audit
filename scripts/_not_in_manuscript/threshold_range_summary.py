#!/usr/bin/env python
"""Compact per-benchmark view of how much the answer moves across the identity sweep.

Direction of effect, stated once because it is easy to get backwards: raising
--min-seq-id makes the criterion for DECLARING A HIT stricter, so fewer hits are
declared and MORE test sequences fall into "no detectable nucleotide relative".
The L1 population therefore GROWS as the threshold rises. The primary setting of
0.30 is the most conservative point on this axis, admitting the fewest sequences.
"""
import os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "reports", "threshold_sensitivity")


def main():
    f = os.path.join(D, "identity_sweep.csv")
    if not os.path.exists(f):
        sys.exit("identity_sweep.csv not present yet")
    d = pd.read_csv(f)
    sc = d[d.verdict != "NOT SCORED"].copy()
    out = []
    for b, g in sc.groupby("benchmark"):
        g = g.sort_values("min_seq_id")
        at30 = g[g.min_seq_id == 0.30]
        e = g.excess
        res = g[g.verdict.isin(["SUPERIOR", "INFERIOR"])]
        out.append(dict(
            benchmark=b, thresholds_scored=len(g),
            n_at_0p30=int(at30.n_rows.iloc[0]) if len(at30) else None,
            n_min=int(g.n_rows.min()), n_max=int(g.n_rows.max()),
            n_fold_change=round(g.n_rows.max() / max(g.n_rows.min(), 1), 2),
            units_at_0p30=int(at30.n_units.iloc[0]) if len(at30) else None,
            units_min=int(g.n_units.min()), units_max=int(g.n_units.max()),
            excess_min=round(e.min(), 4),
            excess_at_0p30=round(at30.excess.iloc[0], 4) if len(at30) else None,
            excess_max=round(e.max(), 4), excess_range=round(e.max() - e.min(), 4),
            # direction change and resolution change are different things: an excess
            # falling from +0.19 to +0.10 with the interval crossing zero is
            # inference-sensitive but not direction-sensitive
            sign_changes=bool((e > 0).any() and (e < 0).any()),
            verdict_at_0p30=at30.verdict.iloc[0] if len(at30) else None,
            verdicts_seen="|".join(sorted(g.verdict.unique())),
            n_resolved=len(res),
            verdict_changes=bool(g.verdict.nunique() > 1),
            verdict_stable=bool(g.verdict.nunique() == 1)))
    r = pd.DataFrame(out).sort_values("excess_range", ascending=False)
    r.to_csv(os.path.join(D, "threshold_range_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(r[["benchmark", "n_at_0p30", "n_min", "n_max", "units_min", "units_max",
             "excess_min", "excess_at_0p30", "excess_max",
             "sign_changes", "verdict_changes", "verdict_at_0p30"]].to_string(index=False))
    print(f"\nbenchmarks whose excess changes sign anywhere in the sweep: "
          f"{r[r.sign_changes].benchmark.tolist() or 'NONE'}")
    print(f"benchmarks whose verdict is identical at every threshold: "
          f"{r[r.verdict_stable].benchmark.tolist() or 'NONE'}")
    print(f"\nwrote {D}/threshold_range_summary.csv")


if __name__ == "__main__":
    main()
