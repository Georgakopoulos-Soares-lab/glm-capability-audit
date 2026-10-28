#!/usr/bin/env python
"""Record how many fits each arm actually has, per benchmark and population.

Methods states the fit counts in prose. Prose is not checkable, and the claim it
replaced ("every stochastic arm was refit ten times, on every benchmark and every
population") was false for weeks without anything noticing. This writes the counts
to a small committed table so audit_consistency.py can hold the prose to them
without needing the prediction archive.

The counts are not reported in the manuscript. An earlier draft carried them as a
supplementary table; that table was cut, and this file is what remains of it.

    VB_RESULTS=/path/to/results_viral_bench python analysis/make_fit_counts.py
        -> reports/tables/fit_counts.csv
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
import final_regime_table as F  # noqa: E402

OUT = os.path.join(REPO, "reports", "tables", "fit_counts.csv")
SHORT = {"HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
         "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
         "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
         "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid"}
# deterministic given the split, one fit by construction, never a shortfall
DETERMINISTIC = {"kmer", "nn_lookup", "tax_prior", "tax_prior_recovered"}
TARGET = 10


def selected():
    import pandas as pd
    sys.path.insert(0, os.path.join(REPO, "paper", "figures"))
    from sot_path import sot_csv
    d = pd.read_csv(sot_csv())
    d = d[d.regime == "L0"]
    return dict(zip(d.benchmark, d.model))


def main():
    sel = selected()
    rows = []
    for name, mask, esc, atags, _ in F.CELLS:
        b = SHORT[name]
        z = np.load(f"{F.MASKS}/{mask}.npz")
        n = len(z["hasN"])
        lv = {"L0": np.ones(n, bool), "L1": ~z["hasN"],
              "L2": (~z["hasN"]) & (~z["hasT"])}
        arms = F.arm_preds(atags or [esc], n)
        y, mps, _ = F.model_preds(esc, sel[b], n)
        if y is None:
            continue
        for lab, k in lv.items():
            if k.sum() < 20 or len(np.unique(y[k])) < 2:
                continue
            series = {sel[b]: mps}
            series.update(arms)
            for arm, ps in series.items():
                if arm in F.REGIME_UNAVAILABLE and lab != "L0":
                    continue
                rows.append(dict(benchmark=b, regime=lab, arm=arm, n_fits=len(ps),
                                 deterministic=arm in DETERMINISTIC,
                                 is_model=arm == sel[b]))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    short = sorted({r["benchmark"] for r in rows
                    if not r["deterministic"] and r["n_fits"] < TARGET})
    print(f"  wrote reports/tables/fit_counts.csv  ({len(rows)} arm-population rows)")
    print(f"  benchmarks with a stochastic arm below {TARGET} fits: "
          + (", ".join(short) if short else "none"))


if __name__ == "__main__":
    main()
