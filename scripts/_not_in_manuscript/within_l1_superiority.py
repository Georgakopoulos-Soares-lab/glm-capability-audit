#!/usr/bin/env python
"""Model versus baseline envelope INSIDE a regime, with the correct resampling unit.

This is a separate question from the L0 to L1 interaction. The interaction asks
whether a model's incremental value changes between regimes. This asks whether
the model has resolved incremental value within one regime, which is what the
paper's verdicts are about.

Two corrections relative to the earlier layered run.

  unit        Host tropism was resampled by row. Its sequences are 1 kb windows
              cut from parent genomes, so rows are not independent. Genome unit
              files were produced but never passed. They are passed here and the
              run hard-errors if an expected one is missing.

  NN lookup   L1 is defined as having no detectable nucleotide training match.
              Nearest neighbour lookup predicts from exactly that match, so at L1
              it is unavailable, not merely bad. Scoring it as 0.000 and feeding
              that into a pointwise maximum is harmless for the maximum but
              dishonest in the arm table, so it is excluded from L1 outright and
              reported as NA. It remains a valid L0 arm.

Seed variance is propagated the same way the layered figure does it, by drawing a
random training seed for every arm and model inside each replicate.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

RVB = os.environ.get("VB_RESULTS",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
MASKS, PRED, FT = f"{RVB}/hidden_masks", f"{RVB}/preds_rescore", f"{RVB}/geneb_finetune"
CLUST = f"{RVB}/clusters"
EQ_MARGIN = 0.02

ARM_TAGS = {"kmer": "kmer", "kmer_gbt": "kmergbt", "aa_gbt": "aagbt", "cnn": "cnn_ceiling",
            "transformer": "transformer__cnn_ceiling", "nn_lookup": "nnlookup",
            "tax_prior": "taxprior", "tax_prior_recovered": "taxpriorrec"}
# an arm whose prediction IS the thing the regime removes
REGIME_UNAVAILABLE = {"nn_lookup"}

CELLS = [
    ("HT hard_70",  "HVUE_HT_hard_70",  "Host_Tropism__hard_70_1000bp",     "hvilm",
     f"{CLUST}/hvue_v2__Host_Tropism__hard_70_1000bp__genomeunits.csv",   None),
    ("HT std_95",   "HVUE_HT_std_95",   "Host_Tropism__standard_95_1000bp", "hvilm",
     f"{CLUST}/hvue_v2__Host_Tropism__standard_95_1000bp__genomeunits.csv", None),
    ("GUE phage",   "GUE_phage_v2",     "leannmlindsey_GUE_phage_fragments", "nt_v2_500m",
     None, ["leannmlindsey_GUE_phage_fragments", "gue_phage_fragments"]),
    ("VHDB",        "VHDB_v2",          "vhdb_tropism_wf1000",              "hvilm",
     "/data/nvidia/vhdb_tropism/within_family_1000bp/test_tax.csv",       None),
    ("Transm hard", "HVUE_Transm_hard", "Transmissibility__hard_capped_1000bp", "hvilm",
     None, None),
]


def arm_preds(tags, n):
    """All seeds of every baseline arm, aligned to test order."""
    out = {}
    for arm, t in ARM_TAGS.items():
        seen = []
        for tag in tags:
            for f in sorted(glob.glob(f"{PRED}/*{tag}*__{t}__s*__preds.npz")):
                # "cnn_ceiling" is a substring of "transformer__cnn_ceiling", so the
                # CNN glob also matches the from-scratch transformer runs. Filtering
                # on the filename is the only reliable separation; deduplicating the
                # loaded arrays does not work because they are distinct objects.
                if arm == "cnn" and "transformer" in os.path.basename(f):
                    continue
                d = np.load(f)
                if "y_pred" in d and len(d["y_pred"]) == n:
                    seen.append(d["y_pred"])
        if seen:
            out[arm] = seen
    return out


def model_preds(esc, mdl, n):
    """All seeds of the dev-selected learning rate, matching the paper's selection."""
    by_lr = {}
    for f in glob.glob(f"{FT}/{esc}__*__preds.npz"):
        m = re.match(r".*__(\w+)__lr([\d.e-]+)__s\d+__preds\.npz", f)
        if not m or m.group(1) != mdl:
            continue
        d = np.load(f)
        if len(d["y_true"]) != n:
            continue
        by_lr.setdefault(m.group(2), []).append((d["y_true"], d["y_pred"]))
    if not by_lr:
        return None, None
    best = max(by_lr.values(),
               key=lambda v: np.mean([matthews_corrcoef(a, b) for a, b in v]))
    return best[0][0], [b for _, b in best]


def mcc(y, p):
    return matthews_corrcoef(y, p) if len(np.unique(y)) > 1 else np.nan


def verdict(lo, hi, margin=EQ_MARGIN):
    if lo > 0:
        return "SUPERIOR"
    if hi < 0:
        return "INFERIOR"
    # EQUIVALENT was removed at finalization. Declaring equivalence needs a margin,
    # and the 0.02 MCC margin had no external justification; it was also smaller than
    # the median seed-to-seed spread of 0.027 MCC. An interval that spans zero is now
    # INCONCLUSIVE, and its width is reported so precision is visible.
    return "INCONCLUSIVE"


def score(y, mp, arms, rows, units, reps, seed):
    """Excess of model over the re-maximised envelope, resampling units."""
    rng = np.random.default_rng(seed)
    y, mp = y[rows], [p[rows] for p in mp]
    arms = {k: [p[rows] for p in v] for k, v in arms.items()}
    if units is None:
        groups, u = None, np.arange(len(rows))
    else:
        uu = units[rows]
        u = pd.unique(uu)
        groups = {c: np.where(uu == c)[0] for c in u}
    # seeds are AVERAGED, matching how the layered figure scores both arms and
    # models. Taking the max over seeds would silently give the envelope a
    # best-of-three advantage the model does not get.
    pt_arms = {k: float(np.mean([mcc(y, p) for p in v])) for k, v in arms.items()}
    env_arm = max(pt_arms, key=pt_arms.get)
    pt_model = float(np.mean([mcc(y, p) for p in mp]))
    draws = []
    for _ in range(reps):
        pick = rng.choice(len(u), len(u), replace=True)
        r = np.arange(len(y))[pick] if groups is None else \
            np.concatenate([groups[u[i]] for i in pick])
        if len(np.unique(y[r])) < 2:
            continue
        e = max(mcc(y[r], v[rng.integers(len(v))][r]) for v in arms.values())
        d = mcc(y[r], mp[rng.integers(len(mp))][r]) - e
        if np.isfinite(d):
            draws.append(d)
    d = np.array(draws)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(model=round(pt_model, 4), envelope=round(pt_arms[env_arm], 4),
                envelope_arm=env_arm, excess=round(pt_model - pt_arms[env_arm], 4),
                lo=round(lo, 4), hi=round(hi, 4), half_width=round((hi - lo) / 2, 4),
                verdict=verdict(lo, hi), n_rows=int(len(rows)), n_units=int(len(u)),
                grouped=units is not None,
                arms={k: round(v, 4) for k, v in sorted(pt_arms.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/layered/within_l1_superiority.json")
    a = ap.parse_args()
    res = {}
    for name, mask, esc, mdl, cl, atags in CELLS:
        z = np.load(f"{MASKS}/{mask}.npz")
        hasN, hasT = z["hasN"], z["hasT"]
        n = len(hasN)
        y, mp = model_preds(esc, mdl, n)
        arms = arm_preds(atags or [esc], n)
        if mp is None or len(arms) < 2:
            raise SystemExit(f"{name}: model={mp is not None} arms={sorted(arms)}")
        units = None
        if cl:
            if not os.path.exists(cl):
                raise SystemExit(f"{name}: expected grouping file missing: {cl}")
            c = pd.read_csv(cl)
            col = [x for x in c.columns if x in ("cluster", "unit", "genome")]
            if not col or len(c) != n:
                raise SystemExit(f"{name}: {cl} has {len(c)} rows / cols {list(c.columns)}, "
                                 f"mask has {n}")
            units = c[col[0]].values
        res[name] = dict(model_name=mdl, unit=("genome" if units is not None else "row"))
        for lab, rows, drop in (("L0", np.arange(n), set()),
                                ("L1", np.where(~hasN)[0], REGIME_UNAVAILABLE),
                                ("L2", np.where(~hasN & ~hasT)[0], REGIME_UNAVAILABLE)):
            if len(rows) < 30 or len(np.unique(y[rows])) < 2:
                res[name][lab] = dict(skipped=f"{len(rows)} rows"); continue
            keep = {k: v for k, v in arms.items() if k not in drop}
            r = score(y, mp, keep, rows, units, a.reps, a.seed)
            r["arms_unavailable"] = sorted(drop & set(arms))
            res[name][lab] = r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    hdr = (f"{'cell / regime':22s} {'model':>7s} {'envlp':>7s} {'excess':>8s} "
           f"{'95% CI':>18s} {'hw':>6s} {'rows':>7s} {'units':>7s} {'env arm':>12s}  verdict")
    print(hdr); print("-" * len(hdr))
    for name, d in res.items():
        for lab in ("L0", "L1", "L2"):
            r = d.get(lab)
            if not r or "skipped" in r:
                print(f"{name+' '+lab:22s} {(r or {}).get('skipped','no data'):>7s}"); continue
            g = "" if r["grouped"] else "*"
            print(f"{name+' '+lab:22s} {r['model']:>7.3f} {r['envelope']:>7.3f} "
                  f"{r['excess']:>+8.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] {r['half_width']:>6.3f} "
                  f"{r['n_rows']:>7,} {str(r['n_units'])+g:>7s} {r['envelope_arm']:>12s}  "
                  f"{r['verdict']}")
    print("\n  * row bootstrap, conditional on treating rows as independent")
    print(f"  NN lookup excluded at L1 and L2 as regime-unavailable, kept at L0")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
