#!/usr/bin/env python
"""Two audit tables the manuscript needs before any interval can be quoted.

A. Positive-class taxonomy assignment coverage.
   Every genus-concentration statistic is computed over positives that CARRY a
   genus label. Rows without one are excluded and never imputed, which is the
   right call, but it means a concentration share is conditional on coverage.
   A 76.7 percent top-genus share over 7 assigned positives is a different
   statement from the same share over 80. This table prints the denominator.

B. Resampling unit provenance, per benchmark and regime.
   Rows are not the unit of statistical dependence when a benchmark chunks
   genomes into windows. This table states, for every inference, what the unit
   was, how many there were, whether it came from provenance the benchmark
   ships or from metadata we recovered by alignment, and what fraction of rows
   the recovery failed on. An unassigned row becomes a singleton unit, which
   biases the interval NARROW, so a low coverage figure is a residual
   anti-conservative bias that survives the grouping fix.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root: paper/figures -> paper -> repo
RVB = os.environ.get("VB_RESULTS",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
MASKS, CLUST = f"{RVB}/hidden_masks", f"{RVB}/clusters"
GENUS = os.path.join(REPO, "reports", "tables", "genus_concentration.json")


def require(path, produced_by, what):
    if not os.path.exists(path):
        raise SystemExit(f"missing input: {path}\n  holds: {what}\n  produce it with: {produced_by}")
    if os.path.getsize(path) == 0:
        raise SystemExit(f"empty input: {path}\n  produce it again with: {produced_by}")
    return path


# benchmark -> (mask, unit csv or None, unit column, provenance)
UNITS = {
  "GUE_covid":       (None, None, "", "none shipped"),
  "GUE_phage":       ("GUE_phage_v2", None, "", "none shipped"),
  "HVUE_HT_hard_70": ("HVUE_HT_hard_70",
                      f"{CLUST}/hvue_v2__Host_Tropism__hard_70_1000bp__genomeunits.csv",
                      "cluster", "recovered by alignment"),
  "HVUE_HT_std_95":  ("HVUE_HT_std_95",
                      f"{CLUST}/hvue_v2__Host_Tropism__standard_95_1000bp__genomeunits.csv",
                      "cluster", "recovered by alignment"),
  "HVUE_Patho_hard": ("HVUE_Patho_hard", None, "", "none shipped"),
  "HVUE_Patho_std":  ("HVUE_Patho_std", None, "", "none shipped"),
  "HVUE_Transm_hard":("HVUE_Transm_hard", None, "", "none shipped"),
  "HVUE_Transm_std": ("HVUE_Transm_std", None, "", "none shipped"),
  "VHDB_within_family": ("VHDB_v2",
                      "/data/nvidia/vhdb_tropism/within_family_1000bp/test_tax.csv",
                      "cluster", "shipped with the benchmark"),
}
# a recovered unit label that means "no genome hit, singleton fallback"
SINGLETON_PREFIX = "row"


def strata(mask):
    z = np.load(f"{MASKS}/{mask}.npz")
    hasN, hasT = z["hasN"], z["hasT"]
    n = len(hasN)
    return n, {"L0": np.ones(n, bool), "L1": ~hasN, "L2": ~hasN & ~hasT}


def table_a():
    g = json.load(open(require(GENUS, "paper/figures/make_genus_concentration.py",
                               "per-stratum genus composition")))
    rows = []
    for bench, d in g.items():
        for lab, s in d["strata"].items():
            pos, cov = s.get("positives"), s.get("positives_with_genus")
            if not pos:
                continue
            rows.append(dict(benchmark=bench, stratum=lab, labels=d["labels"],
                             rows=s["n"], positives=pos, assigned=cov,
                             coverage=round(cov / pos, 4),
                             genera=s.get("n_positive_genera"),
                             top_share=s.get("top_positive_share"),
                             underpowered=bool(s.get("share_underpowered"))))
    return rows


def table_b():
    rows = []
    for bench, (mask, csv, col, prov) in UNITS.items():
        if mask is None or not os.path.exists(f"{MASKS}/{mask}.npz"):
            continue
        n, st = strata(mask)
        units = None
        if csv:
            require(csv, "scripts/track_b_viral/assign_source_genome.py",
                    "test-row-order resampling unit per sequence")
            c = pd.read_csv(csv)
            if len(c) != n or col not in c.columns:
                raise SystemExit(f"{bench}: {csv} has {len(c)} rows / {list(c.columns)}, "
                                 f"mask has {n}")
            units = c[col].astype(str).values
        for lab, m in st.items():
            nr = int(m.sum())
            if nr == 0:
                continue
            if units is None:
                rows.append(dict(benchmark=bench, stratum=lab, rows=nr, unit="row",
                                 n_units=nr, provenance=prov, assigned_frac=None,
                                 row_bootstrap_only=True))
            else:
                u = units[m]
                real = ~pd.Series(u).str.startswith(SINGLETON_PREFIX).values
                rows.append(dict(benchmark=bench, stratum=lab, rows=nr, unit="genome",
                                 n_units=int(len(pd.unique(u))), provenance=prov,
                                 assigned_frac=round(float(real.mean()), 4),
                                 n_real_units=int(len(pd.unique(u[real]))) if real.any() else 0,
                                 row_bootstrap_only=False))
    return rows


def main():
    A, B = table_a(), table_b()
    out = os.path.join(REPO, "reports", "tables")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(coverage=A, units=B), open(f"{out}/inference_provenance.json", "w"), indent=2)

    print("A. POSITIVE-CLASS TAXONOMY ASSIGNMENT COVERAGE")
    print("   every genus statistic below is conditional on the assigned column\n")
    h = (f"   {'benchmark':20s} {'str':4s} {'labels':10s} {'rows':>7s} {'pos':>6s} "
         f"{'assigned':>9s} {'cover':>7s} {'genera':>7s} {'top share':>10s}")
    print(h); print("   " + "-" * (len(h) - 3))
    for r in A:
        ts = "--" if r["underpowered"] or r["top_share"] is None else f"{100*r['top_share']:.1f}%"
        print(f"   {r['benchmark']:20s} {r['stratum']:4s} {r['labels']:10s} {r['rows']:>7,} "
              f"{r['positives']:>6,} {r['assigned']:>9,} {100*r['coverage']:>6.1f}% "
              f"{r['genera']:>7} {ts:>10s}")
    print("\n   a top share marked -- is computed on too few assigned positives to read")

    print("\n\nB. RESAMPLING UNIT PROVENANCE, EVERY INFERENCE")
    h = (f"   {'benchmark':20s} {'str':4s} {'rows':>7s} {'unit':>7s} {'n units':>8s} "
         f"{'real units':>10s} {'assigned':>9s} {'row only':>9s}  provenance")
    print(h); print("   " + "-" * (len(h) - 3))
    for r in B:
        af = "--" if r["assigned_frac"] is None else f"{100*r['assigned_frac']:.1f}%"
        ru = "--" if r.get("n_real_units") is None else f"{r.get('n_real_units','--'):,}" \
             if isinstance(r.get("n_real_units"), int) else "--"
        print(f"   {r['benchmark']:20s} {r['stratum']:4s} {r['rows']:>7,} {r['unit']:>7s} "
              f"{r['n_units']:>8,} {ru:>10s} {af:>9s} "
              f"{'YES' if r['row_bootstrap_only'] else 'no':>9s}  {r['provenance']}")
    print("\n   real units counts only rows assigned to a genome. An unassigned row becomes")
    print("   its own singleton unit, which narrows the interval, so a low assigned figure")
    print("   is residual anti-conservative bias that the grouping fix does not remove.")
    print(f"\n   wrote {out}/inference_provenance.json")


if __name__ == "__main__":
    main()
