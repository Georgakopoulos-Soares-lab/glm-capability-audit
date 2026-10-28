"""Per-seed MCC by method, benchmark and level, for a given results tree.

Reads the frozen prediction arrays directly -- no bootstrap, no selection -- so the
3-seed and 10-seed trees can be tabulated with identical code. Deliverables (a) and (e)
of the seed-expansion work order.

Usage:  python seed_tables.py <results_tree> <out.csv>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

RVB = sys.argv[1]
OUT = sys.argv[2]
MASKS, PRED, FT = f"{RVB}/hidden_masks", f"{RVB}/preds_rescore", f"{RVB}/geneb_finetune"

# benchmark label, mask stem, finetune prefix, dev-selected model, dev-selected lr
CELLS = [
    ("HT hard_70",  "HVUE_HT_hard_70", "Host_Tropism__hard_70_1000bp",          "hvilm", "3e-05"),
    ("HT std_95",   "HVUE_HT_std_95",  "Host_Tropism__standard_95_1000bp",      "hvilm", "3e-05"),
    ("Transm hard", "HVUE_Transm_hard","Transmissibility__hard_capped_1000bp",  "hvilm", "0.0001"),
]
ARM_TAGS = {"kmer": "kmer", "kmer_gbt": "kmergbt", "aa_gbt": "aagbt",
            "cnn": "cnn_ceiling", "transformer": "transformer__cnn_ceiling",
            "nn_lookup": "nnlookup", "tax_prior_recovered": "taxpriorrec"}
# nearest-neighbour lookup requires a nucleotide match in training, so it does not apply
# off the published test set. Same rule the manuscript uses.
REGIME_UNAVAILABLE = {"nn_lookup"}


def seed_of(path):
    base = os.path.basename(path)
    tok = [t for t in base.split("__") if t.startswith("s") and t[1:].isdigit()]
    return int(tok[-1][1:]) if tok else -1


def rows_for(label, mask_stem, esc, model, lr):
    z = np.load(f"{MASKS}/{mask_stem}.npz")
    hasN, hasT = z["hasN"], z["hasT"]
    levels = {"L0": np.ones(len(hasN), bool), "L1": ~hasN, "L2": (~hasN) & (~hasT)}
    n = len(hasN)

    series = {}
    for f in sorted(glob.glob(f"{FT}/{esc}__{model}__lr{lr}__s*__preds.npz")):
        d = np.load(f)
        if len(d["y_pred"]) == n:
            series[(model, seed_of(f))] = (d["y_true"], d["y_pred"])
    for arm, tag in ARM_TAGS.items():
        for f in sorted(glob.glob(f"{PRED}/*{esc}*__{tag}__s*__preds.npz")):
            if arm == "cnn" and "transformer" in os.path.basename(f):
                continue
            d = np.load(f)
            if "y_pred" in d and len(d["y_pred"]) == n:
                series[(arm, seed_of(f))] = (d["y_true"], d["y_pred"])

    out = []
    for (meth, sd), (yt, yp) in sorted(series.items()):
        for lv, keep in levels.items():
            if meth in REGIME_UNAVAILABLE and lv != "L0":
                continue
            k = keep & np.isfinite(yt.astype(float))
            if k.sum() < 2 or len(np.unique(yt[k])) < 2:
                continue
            out.append(dict(benchmark=label, level=lv, method=meth, seed=sd,
                            n=int(k.sum()), mcc=round(float(matthews_corrcoef(yt[k], yp[k])), 6)))
    return out


rows = []
for c in CELLS:
    rows += rows_for(*c)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

agg = (df.groupby(["benchmark", "level", "method"])["mcc"]
         .agg(n_seeds="count", mean="mean", sd=lambda s: s.std(ddof=1), min="min", max="max")
         .reset_index())
agg.to_csv(OUT.replace(".csv", "_summary.csv"), index=False)
print(f"wrote {OUT}  ({len(df)} per-seed rows)")
print(f"wrote {OUT.replace('.csv', '_summary.csv')}  ({len(agg)} method x level rows)")
