#!/usr/bin/env python
"""Freeze the three-seed development scores that made the model selection.

Why this file exists
--------------------
Selection was pre-committed on the mean development MCC over three seeds. The
selected arms were later refit ten times, which overwrote their finetune JSONs,
so for six of the nine benchmarks the three-seed development mean can no longer
be recomputed from the tree. Recomputing at the current fit count would print a
ten-fit number as the quantity that made a three-seed decision.

The pre-finalization source of truth, `reports/tables/final_source_of_truth.csv`,
was written before the refit and is never modified. It carries the original
three-seed `dev_mcc` for the selected model on every benchmark. That file is the
surviving record of the pre-commitment, and this script lifts it into a standalone
artifact that the supplement reads.

Rule, per row
-------------
selected model      -> `dev_mcc` from the pre-finalization source of truth
                       (provenance FROZEN_PRECOMMIT)
non-selected model  -> recomputed from the finetune JSONs, which still hold
                       exactly the three original seeds for these arms
                       (provenance RECOMPUTED_3SEED, asserted to be 3 fits)

Any row whose live JSON carries more than three runs and is not the selected
model aborts the script rather than silently printing a ten-fit mean.

    python analysis/make_selection_artifact.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
import final_regime_table as F  # noqa: E402

PREFINAL = os.path.join(REPO, "reports", "tables", "final_source_of_truth.csv")
OUT = os.path.join(REPO, "reports", "tables", "selection_3seed.csv")

SHORT = {"HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
         "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
         "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
         "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid"}


def live(esc, mdl):
    """(dev mean, lr, n_runs) for the learning rate this model would select on, live."""
    c = []
    for j in sorted(glob.glob(f"{F.FT}/{esc}__{mdl}__lr*__fullft.json")):
        if any(x in j for x in ("_meanpool", "_ep4", "_truncated")):
            continue
        d = json.load(open(j))
        runs = d.get("runs", [])
        if len(runs) >= 3:
            c.append((float(np.mean([r["dev_mcc"] for r in runs])), d["lr"], len(runs)))
    return max(c) if c else (None, None, 0)


def main():
    frozen = {}
    for r in csv.DictReader(open(PREFINAL)):
        key = r["benchmark"]
        if key not in frozen and r.get("dev_mcc"):
            frozen[key] = (r["model"], float(r["dev_mcc"]))

    rows, problems = [], []
    for name, _mask, esc, _atags, _ in F.CELLS:
        b = SHORT[name]
        sel_model, sel_dev = frozen[b]
        for m in F.MODELS:
            dev, lr, n = live(esc, m)
            if dev is None:
                continue
            if m == sel_model:
                rows.append(dict(benchmark=b, model=m, lr=f"{lr:g}", dev_mcc_3seed=f"{sel_dev:.4f}",
                                 n_fits_now=n, provenance="FROZEN_PRECOMMIT"))
            else:
                if n != 3:
                    problems.append(f"{b}/{m}: {n} runs, cannot be a three-seed mean")
                rows.append(dict(benchmark=b, model=m, lr=f"{lr:g}", dev_mcc_3seed=f"{dev:.4f}",
                                 n_fits_now=n, provenance="RECOMPUTED_3SEED"))
    if problems:
        raise SystemExit("refusing to write a mixed-fit-count selection table:\n  "
                         + "\n  ".join(problems))

    # the frozen value must still select the same model it selected originally
    for b, (sel_model, _) in frozen.items():
        sub = [r for r in rows if r["benchmark"] == b]
        top = max(sub, key=lambda r: float(r["dev_mcc_3seed"]))
        if top["model"] != sel_model:
            raise SystemExit(f"{b}: frozen dev ranks {top['model']} above the "
                             f"pre-committed {sel_model}; selection record is inconsistent")

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_frozen = sum(r["provenance"] == "FROZEN_PRECOMMIT" for r in rows)
    print(f"  wrote reports/tables/selection_3seed.csv  ({len(rows)} rows, "
          f"{n_frozen} carried from the pre-finalization table)")


if __name__ == "__main__":
    main()
