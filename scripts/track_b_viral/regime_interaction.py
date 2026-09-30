#!/usr/bin/env python
"""Does a model's incremental value change between the published population and
the nucleotide divergent one?

The two populations are nested, L1 inside L0, which is the dependence structure
the test should preserve rather than a reason to avoid testing. We resample the
L0 units, carry each sequence's fixed L1 membership with it, and inside every
replicate recompute all four quantities:

    D = (model_L1 - envelope_L1) - (model_L0 - envelope_L0)

The envelope is re-maximised over its arms inside each replicate, exactly as the
threshold is everywhere else in the audit, so the selection effect is inside the
interval rather than outside it.

Two envelopes are reported and they answer different questions.

  operational    every arm applicable in that regime. Nearest neighbour lookup
                 counts at L0 and collapses at L1. This is the strongest cheap
                 alternative actually available for those sequences.
  common arm     nearest neighbour lookup excluded from both levels. L1 is
                 defined as having no nucleotide match and NN lookup is a
                 nucleotide match predictor, so its collapse there is partly
                 definitional. This asks whether ordinary learned and
                 compositional baselines lose more than the model regardless.

Five cells are tested, so p values are reported with a Benjamini-Hochberg
adjustment. VHDB resamples genome clusters; the rest resample rows and are
anti-conservative by an unknown factor, which is flagged per row.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

RVB = os.environ.get("VB_RESULTS",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
MASKS, PRED, FT = f"{RVB}/hidden_masks", f"{RVB}/preds_rescore", f"{RVB}/geneb_finetune"
CLUST = f"{RVB}/clusters"

# benchmark -> (mask, finetune prefix, focus model, grouping csv or None, arm tags)
# the arm tags are a list because a benchmark's baseline arms are not all stored
# under one spelling of its name
VHDB_TAX = "/data/nvidia/vhdb_tropism/within_family_1000bp/test_tax.csv"

CELLS = [
    ("HT hard_70",  "HVUE_HT_hard_70",   "Host_Tropism__hard_70_1000bp",       "hvilm",
     f"{CLUST}/hvue_v2__Host_Tropism__hard_70_1000bp__genomeunits.csv",   None),
    ("HT std_95",   "HVUE_HT_std_95",    "Host_Tropism__standard_95_1000bp",   "hvilm",
     f"{CLUST}/hvue_v2__Host_Tropism__standard_95_1000bp__genomeunits.csv", None),
    ("GUE phage",   "GUE_phage_v2",      "leannmlindsey_GUE_phage_fragments",  "nt_v2_500m", None,
     ["leannmlindsey_GUE_phage_fragments", "gue_phage_fragments"]),
    ("VHDB",        "VHDB_v2",           "vhdb_tropism_wf1000",                "hvilm", VHDB_TAX, None),
    ("Transm hard", "HVUE_Transm_hard",  "Transmissibility__hard_capped_1000bp", "hvilm",  None,     None),
]
ARM_TAGS = {"kmer": "kmer", "kmer_gbt": "kmergbt", "aa_gbt": "aagbt", "cnn": "cnn_ceiling",
            "nn_lookup": "nnlookup", "tax_prior": "taxprior", "tax_prior_recovered": "taxpriorrec"}


def arm_preds(tags, n):
    """Every baseline arm's predictions for one benchmark, aligned to the test order."""
    out = {}
    for arm, t in ARM_TAGS.items():
        fs = [f for tag in tags
              for f in sorted(glob.glob(f"{PRED}/*{tag}*__{t}__s*__preds.npz"))]
        for f in fs:
            d = np.load(f)
            if "y_pred" in d and len(d["y_pred"]) == n:
                out[arm] = d["y_pred"]; break
    return out


def model_preds(esc, mdl, n):
    """The dev selected run, matching how the paper scores each model."""
    best = None
    for f in glob.glob(f"{FT}/{esc}__*__preds.npz"):
        m = re.match(r".*__(\w+)__lr[\d.e-]+__s\d+__preds\.npz", f)
        if not m or m.group(1) != mdl:
            continue
        d = np.load(f)
        if len(d["y_true"]) != n:
            continue
        s = matthews_corrcoef(d["y_true"], d["y_pred"])
        if best is None or s > best[0]:
            best = (s, d["y_true"], d["y_pred"])
    return best


def mcc(y, p):
    return matthews_corrcoef(y, p) if len(np.unique(y)) > 1 else np.nan


def run_cell(y, mp, arms, inL1, units, drop_nn, B, rng):
    keep = {k: v for k, v in arms.items() if not (drop_nn and k == "nn_lookup")}
    if units is None:
        groups = [np.array([i]) for i in range(len(y))]
    else:
        groups = [np.where(units == u)[0] for u in np.unique(units)]
    gidx = np.arange(len(groups))
    ds = []
    for _ in range(B):
        pick = rng.choice(gidx, len(gidx), replace=True)
        idx = np.concatenate([groups[g] for g in pick])
        sub = idx[inL1[idx]]
        if len(np.unique(y[idx])) < 2 or len(sub) < 20 or len(np.unique(y[sub])) < 2:
            continue
        e0 = np.nanmax([mcc(y[idx], a[idx]) for a in keep.values()])
        e1 = np.nanmax([mcc(y[sub], a[sub]) for a in keep.values()])
        d = (mcc(y[sub], mp[sub]) - e1) - (mcc(y[idx], mp[idx]) - e0)
        if np.isfinite(d):
            ds.append(d)
    ds = np.array(ds)
    e0 = np.nanmax([mcc(y, a) for a in keep.values()])
    e1 = np.nanmax([mcc(y[inL1], a[inL1]) for a in keep.values()])
    D = (mcc(y[inL1], mp[inL1]) - e1) - (mcc(y, mp) - e0)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return D, lo, hi, max(p, 1.0 / len(ds)), mcc(y, mp) - e0, mcc(y[inL1], mp[inL1]) - e1


def bh(ps):
    o = np.argsort(ps); n = len(ps); adj = np.empty(n); run = 1.0
    for r, i in enumerate(o[::-1]):
        run = min(run, ps[i] * n / (n - r)); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = {}
    for drop_nn in (False, True):
        lab = "common arm envelope, NN lookup excluded from both" if drop_nn else \
              "operational envelope, every applicable arm"
        out[lab] = []
        print(f"\n=== {lab} ===")
        print(f"{'cell':22s} {'d L0':>8s} {'d L1':>8s} {'D':>8s} {'95% CI':>18s} {'p':>7s}  unit")
        rows = []
        for name, mask, esc, mdl, cl, atags in CELLS:
            z = np.load(f"{MASKS}/{mask}.npz"); inL1 = ~z["hasN"]; n = len(inL1)
            mr = model_preds(esc, mdl, n)
            arms = arm_preds(atags or [esc], n)
            if mr is None or len(arms) < 2:
                raise SystemExit(f"{name}: need the {mdl} run and at least two baseline "
                                 f"arms; found model={mr is not None} arms={sorted(arms)}")
            _, y, mp = mr
            units = None
            if cl:
                if not os.path.exists(cl):
                    raise SystemExit(f"missing grouping file: {cl}")
                c = pd.read_csv(cl)
                col = [x for x in c.columns if "unit" in x or "cluster" in x]
                if not col:
                    raise SystemExit(f"no unit or cluster column in {cl}: {list(c.columns)}")
                if len(c) != n:
                    raise SystemExit(f"{cl} has {len(c)} rows, mask {mask} has {n}")
                units = c[col[0]].values
            rng = np.random.default_rng(a.seed)
            r = run_cell(y, mp, arms, inL1, units, drop_nn, a.boot, rng)
            rows.append((name, r, units is not None))
        ps = [r[1][3] for r in rows]
        adj = bh(np.array(ps)) if rows else []
        for (name, r, grouped), q in zip(rows, adj):
            D, lo, hi, p, d0, d1 = r
            # resolution is judged on the adjusted value. Five cells are tested
            # together, so an unadjusted interval that excludes zero is not by
            # itself a finding.
            star = "resolved" if q < 0.05 else ""
            raw = "raw CI excludes zero" if (lo > 0 or hi < 0) and q >= 0.05 else ""
            print(f"{name:22s} {d0:>+8.3f} {d1:>+8.3f} {D:>+8.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] {p:>7.3f}  "
                  f"{'genome' if grouped else 'row*'}  BH q={q:.3f} {star}{raw}")
            out[lab].append(dict(cell=name, d_L0=d0, d_L1=d1, D=D, lo=lo, hi=hi,
                                 p=p, q=q, unit="genome" if grouped else "row",
                                 resolved=bool(q < 0.05)))
        print("  * row bootstrap, anti conservative by an unknown factor")


    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "reports", "layered", "regime_interaction.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
