"""Verdict diff: 3-seed vs 10-seed, both bootstrap schemes.

Deliverable (c) of the seed-expansion work order, plus the comparator-maximum
check (e). Reads two bootstrap.csv outputs; computes nothing itself.

In bootstrap.csv the column families are:
  frag_*  / src_*   conditional scheme (seeds held fixed)
  frag2_* / src2_*  two-level scheme (seeds resampled) -- the manuscript's primary

Usage: python verdict_diff.py <boot_3seed.csv> <boot_10seed.csv> <out_prefix>
"""
import sys

import pandas as pd

b3 = pd.read_csv(sys.argv[1])
b10 = pd.read_csv(sys.argv[2])
PRE = sys.argv[3]
TARGETS = ["HT hard_70", "HT std_95", "Transm hard"]
DISPUTED = {("HT hard_70", "L2"), ("HT std_95", "L2"),
            ("Transm hard", "L0"), ("Transm hard", "L1")}

keep = ["benchmark", "regime", "n_rows", "baseline_arm", "excess",
        "frag_lo", "frag_hi", "frag_verdict", "frag2_lo", "frag2_hi", "frag2_verdict",
        "src2_lo", "src2_hi", "src2_verdict"]
a = b3[b3.benchmark.isin(TARGETS)][keep].copy()
b = b10[b10.benchmark.isin(TARGETS)][keep].copy()
m = a.merge(b, on=["benchmark", "regime"], suffixes=("_3", "_10"), how="outer")
m["disputed"] = [(r.benchmark, r.regime) in DISPUTED for r in m.itertuples()]
m["primary_changed"] = m.frag2_verdict_3 != m.frag2_verdict_10
m["schemes_agree_10"] = m.frag_verdict_10 == m.frag2_verdict_10
m = m.sort_values(["disputed", "benchmark", "regime"], ascending=[False, True, True])
m.to_csv(f"{PRE}_verdict_diff.csv", index=False)

cols = ["benchmark", "regime", "disputed", "excess_3", "excess_10",
        "frag2_verdict_3", "frag2_verdict_10", "frag_verdict_10", "schemes_agree_10"]
print("\n=== primary (two-level) verdict, 3-seed vs 10-seed ===")
print(m[cols].to_string(index=False))
print(f"\nwrote {PRE}_verdict_diff.csv")

print("\n=== (e) comparator identity and value ===")
e = m[["benchmark", "regime", "baseline_arm_3", "baseline_arm_10"]].copy()
e["arm_changed"] = e.baseline_arm_3 != e.baseline_arm_10
print(e.to_string(index=False))
e.to_csv(f"{PRE}_comparator_shift.csv", index=False)
