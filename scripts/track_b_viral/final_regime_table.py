#!/usr/bin/env python
"""The nine-benchmark source of truth: two estimands, three axes of evidence.

Fragment resampling and source resampling are NOT competing definitions of the
truth. They answer different questions and both are reported here.

  fragment   resamples the 1 kb test fragments. Asks whether the difference is
             stable across fragments of the kind this benchmark ships. This is
             benchmark performance, and it is the question a metagenomic
             pipeline that literally receives contigs actually faces.
  source     resamples the parent genomes, holding the same fragment-level
             scores. Asks whether the difference would survive a different draw
             of source viruses. This is generalization across biological
             sources.

A difference can be resolved under fragment resampling and unresolved under
source resampling. That is not a contradiction and the second does not
invalidate the first. The point estimate is the observed benchmark performance
either way, and it is reported first.

One row per benchmark and regime. The leading model is chosen by MCC on the
published population and held fixed down the regimes, so a benchmark is not
credited with a different model at each level.

Three things this reports that the earlier analysis did not.

  unit provenance   whether the resampling unit is shipped with the benchmark,
                    recovered by aligning each fragment to a viral genome
                    reference, or unavailable so that rows are the only option.
  assignment cover  the fraction of rows the recovery actually placed on a
                    genome. An unplaced row becomes a singleton, which narrows
                    the interval, so low coverage is residual anti-conservative
                    bias that survives the grouping fix.
  NN availability   nearest neighbour lookup is a valid arm on the published
                    population and unavailable at L1 and L2, which are defined
                    by the absence of the nucleotide hit it transfers from.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

RVB = os.environ.get("VB_RESULTS",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
MASKS, PRED, FT = f"{RVB}/hidden_masks", f"{RVB}/preds_rescore", f"{RVB}/geneb_finetune"
CL = f"{RVB}/clusters"
EQ_MARGIN = 0.02
MODELS = ["hvilm", "nt_v2_500m", "dnabert2", "gena_lm", "hyenadna"]

ARM_TAGS = {"kmer": "kmer", "kmer_gbt": "kmergbt", "aa_gbt": "aagbt", "cnn": "cnn_ceiling",
            "transformer": "transformer__cnn_ceiling", "nn_lookup": "nnlookup",
            "tax_prior": "taxprior", "tax_prior_recovered": "taxpriorrec"}
REGIME_UNAVAILABLE = {"nn_lookup"}
SHIPPED, RECOVERED, NONE = "shipped", "recovered", "row only"
# Reference-based recovery placed 12.5% of GUE phage fragments, giving 4,542
# units for 4,630 rows. That is not a grouping, it is a row bootstrap wearing a
# unit column, so the unit file is deliberately NOT used for inference here.
# ViroBench is a human-infecting virus collection and phage are bacterial
# viruses, so no reference we hold can do better.
PHAGE_RECOVERY_NOTE = "recovery inadequate, 12.5% assignment"

# label, mask, finetune prefix, arm tags, (unit csv, provenance)
CELLS = [
 ("HVUE host tropism hard_70", "HVUE_HT_hard_70", "Host_Tropism__hard_70_1000bp", None,
  (f"{CL}/hvue_v2__Host_Tropism__hard_70_1000bp__genomeunits.csv", RECOVERED)),
 ("HVUE host tropism std_95", "HVUE_HT_std_95", "Host_Tropism__standard_95_1000bp", None,
  (f"{CL}/hvue_v2__Host_Tropism__standard_95_1000bp__genomeunits.csv", RECOVERED)),
 ("HVUE pathogenicity hard", "HVUE_Patho_hard", "Pathogenicity__hard_capped_1000bp", None,
  (f"{CL}/hvue_v2__Pathogenicity__hard_capped_1000bp__genomeunits.csv", RECOVERED)),
 ("HVUE pathogenicity std", "HVUE_Patho_std", "Pathogenicity__standard_capped_1000bp", None,
  (f"{CL}/hvue_v2__Pathogenicity__standard_capped_1000bp__genomeunits.csv", RECOVERED)),
 ("HVUE transmissibility hard", "HVUE_Transm_hard", "Transmissibility__hard_capped_1000bp", None,
  (f"{CL}/hvue_v2__Transmissibility__hard_capped_1000bp__genomeunits.csv", RECOVERED)),
 ("HVUE transmissibility std", "HVUE_Transm_std", "Transmissibility__standard_capped_1000bp", None,
  (f"{CL}/hvue_v2__Transmissibility__standard_capped_1000bp__genomeunits.csv", RECOVERED)),
 ("VHDB within family", "VHDB_v2", "vhdb_tropism_wf1000", None,
  ("/data/nvidia/vhdb_tropism/within_family_1000bp/test_tax.csv", SHIPPED)),
 ("GUE phage", "GUE_phage_v2", "leannmlindsey_GUE_phage_fragments",
  ["leannmlindsey_GUE_phage_fragments", "gue_phage_fragments"],
  (None, NONE)),  # see PHAGE_RECOVERY_NOTE
 ("GUE virus_covid", "GUE_covid", "leannmlindsey_GUE_virus_covid",
  ["leannmlindsey_GUE_virus_covid", "gue_virus_covid"], (None, NONE)),
]


def arm_preds(tags, n):
    out = {}
    for arm, t in ARM_TAGS.items():
        seen = []
        for tag in tags:
            for f in sorted(glob.glob(f"{PRED}/*{tag}*__{t}__s*__preds.npz")):
                if arm == "cnn" and "transformer" in os.path.basename(f):
                    continue
                d = np.load(f)
                if "y_pred" in d and len(d["y_pred"]) == n:
                    seen.append(d["y_pred"])
        if seen:
            out[arm] = seen
    return out


def model_preds(esc, mdl, n):
    """Learning rate chosen on DEV, never on test.

    Selecting the run that scores best on the test set and then testing that
    run's superiority on the same test set biases the estimate upward by
    construction. The dev split is what the selection is for.
    """
    c = []
    for j in sorted(glob.glob(f"{FT}/{esc}__{mdl}__lr*__fullft.json")):
        if any(x in j for x in ("_meanpool", "_ep4", "_truncated")):
            continue
        d = json.load(open(j))
        if len(d.get("runs", [])) >= 3:
            sfx = "".join(x for x in ("__probs", "__ep16", "__ep32", "__ep64") if x in j)
            c.append((float(np.mean([r["dev_mcc"] for r in d["runs"]])), d["lr"], sfx))
    if not c:
        return None, None, None
    dev, lr, sfx = max(c)
    y, ps = None, []
    for f in sorted(glob.glob(f"{FT}/{esc}__{mdl}__lr{lr:g}{sfx}__s*__preds.npz")):
        d = np.load(f)
        if len(d["y_true"]) != n:
            continue
        y = d["y_true"]; ps.append(d["y_pred"])
    ps = [p for p in ps if len(np.unique(p)) > 1]
    return (y, ps, dev) if ps else (None, None, None)


def mcc(y, p):
    return matthews_corrcoef(y, p) if len(np.unique(y)) > 1 else np.nan


def verdict(lo, hi, m=EQ_MARGIN):
    if lo > 0:
        return "SUPERIOR"
    if hi < 0:
        return "INFERIOR"
    if lo > -m and hi < m:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def score(y, mp, arms, rows, units, reps, seed):
    rng = np.random.default_rng(seed)
    y, mp = y[rows], [p[rows] for p in mp]
    arms = {k: [p[rows] for p in v] for k, v in arms.items()}
    if units is None:
        u, groups = np.arange(len(rows)), None
    else:
        uu = units[rows]; u = pd.unique(uu)
        groups = {c: np.where(uu == c)[0] for c in u}
    pt = {k: float(np.mean([mcc(y, p) for p in v])) for k, v in arms.items()}
    earm = max(pt, key=pt.get)
    pm = float(np.mean([mcc(y, p) for p in mp]))
    d = []
    for _ in range(reps):
        pick = rng.choice(len(u), len(u), replace=True)
        r = pick if groups is None else np.concatenate([groups[u[i]] for i in pick])
        if len(np.unique(y[r])) < 2:
            continue
        e = max(mcc(y[r], v[rng.integers(len(v))][r]) for v in arms.values())
        x = mcc(y[r], mp[rng.integers(len(mp))][r]) - e
        if np.isfinite(x):
            d.append(x)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(model_mcc=round(pm, 4), envelope=round(pt[earm], 4), envelope_arm=earm,
                excess=round(pm - pt[earm], 4), lo=round(lo, 4), hi=round(hi, 4),
                half_width=round((hi - lo) / 2, 4), verdict=verdict(lo, hi),
                n_rows=len(rows), n_units=len(u))


def score_sym(y, fms, arms, rows, units, reps, seed):
    """max over FMs minus max over baseline arms, both re-maximised per replicate."""
    rng = np.random.default_rng(seed)
    y = y[rows]
    fms = {k: [p[rows] for p in v] for k, v in fms.items()}
    arms = {k: [p[rows] for p in v] for k, v in arms.items()}
    if units is None:
        u, groups = np.arange(len(rows)), None
    else:
        uu = units[rows]; u = pd.unique(uu)
        groups = {c: np.where(uu == c)[0] for c in u}
    pt_f = max(float(np.mean([mcc(y, p) for p in v])) for v in fms.values())
    pt_b = max(float(np.mean([mcc(y, p) for p in v])) for v in arms.values())
    d = []
    for _ in range(reps):
        pick = rng.choice(len(u), len(u), replace=True)
        r = pick if groups is None else np.concatenate([groups[u[i]] for i in pick])
        if len(np.unique(y[r])) < 2:
            continue
        f = max(mcc(y[r], v[rng.integers(len(v))][r]) for v in fms.values())
        b = max(mcc(y[r], v[rng.integers(len(v))][r]) for v in arms.values())
        if np.isfinite(f - b):
            d.append(f - b)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(excess=round(pt_f - pt_b, 4), lo=round(lo, 4), hi=round(hi, 4),
                verdict=verdict(lo, hi), n_fms=len(fms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/layered/final_regime_table.json")
    a = ap.parse_args()
    res = []
    for name, mask, esc, atags, (ucsv, prov) in CELLS:
        z = np.load(f"{MASKS}/{mask}.npz")
        hasN, hasT = z["hasN"], z["hasT"]; n = len(hasN)
        arms = arm_preds(atags or [esc], n)
        cand, devs = {}, {}
        for m in MODELS:
            y_, mp_, dev_ = model_preds(esc, m, n)
            if mp_:
                cand[m] = (y_, mp_); devs[m] = dev_
        if not cand or len(arms) < 2:
            raise SystemExit(f"{name}: models={sorted(cand)} arms={sorted(arms)}")
        # PREDECLARED on dev, then held fixed across every regime
        lead = max(devs, key=devs.get)
        y, mp = cand[lead]
        fm_all = {m: v[1] for m, v in cand.items()}
        units, cover = None, None
        if ucsv:
            if not os.path.exists(ucsv):
                raise SystemExit(f"{name}: expected unit file missing: {ucsv}")
            c = pd.read_csv(ucsv)
            col = [x for x in c.columns if x in ("cluster", "unit", "genome")]
            if not col or len(c) != n:
                raise SystemExit(f"{name}: {ucsv} has {len(c)} rows cols {list(c.columns)}, "
                                 f"mask {n}")
            units = c[col[0]].astype(str).values
        for lab, rows in (("L0", np.arange(n)), ("L1", np.where(~hasN)[0]),
                          ("L2", np.where(~hasN & ~hasT)[0])):
            base = dict(benchmark=name, regime=lab, model=lead, provenance=prov)
            if len(rows) < 30 or len(np.unique(y[rows])) < 2:
                res.append(dict(base, n_rows=int(len(rows)), skipped=True)); continue
            keep = {k: v for k, v in arms.items() if not (lab != "L0" and k in REGIME_UNAVAILABLE)}
            # ESTIMAND 1, always available: benchmark performance across fragments
            frag = score(y, mp, keep, rows, None, a.reps, a.seed)
            r = dict(model_mcc=frag["model_mcc"], envelope=frag["envelope"],
                     envelope_arm=frag["envelope_arm"], excess=frag["excess"],
                     n_rows=frag["n_rows"], dev_mcc=round(devs[lead], 4),
                     frag_lo=frag["lo"], frag_hi=frag["hi"],
                     frag_half_width=frag["half_width"], frag_verdict=frag["verdict"])
            # ESTIMAND 2, only where provenance permits: generalization across sources
            if units is not None:
                src = score(y, mp, keep, rows, units, a.reps, a.seed)
                u = pd.Series(units[rows]).astype(str)
                r.update(n_units=src["n_units"], src_lo=src["lo"], src_hi=src["hi"],
                         src_half_width=src["half_width"], src_verdict=src["verdict"],
                         assign_cover=round(float((~u.str.startswith("row")).mean()), 4),
                         n_real_units=int(u[~u.str.startswith("row")].nunique()))
            else:
                r.update(n_units=None, src_lo=None, src_hi=None, src_half_width=None,
                         src_verdict="unavailable", assign_cover=None, n_real_units=None)
            # symmetric selection check, on whichever unit is the stricter available one
            r["sym"] = score_sym(y, fm_all, keep, rows, units, a.reps, a.seed)
            r["nn_available"] = (lab == "L0" and "nn_lookup" in arms)
            res.append(dict(base, **r, skipped=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    h = (f"{'benchmark':27s} {'reg':4s} {'model':11s} {'FM':>6s} {'base':>6s} {'excess':>8s} "
         f"{'fragment CI':>18s} {'':12s} {'source-aware CI':>18s} {'':12s} "
         f"{'cov':>5s} {'rows':>7s} {'units':>7s}")
    print(h); print("-" * len(h))
    for r in res:
        if r.get("skipped"):
            print(f"{r['benchmark']:27s} {r['regime']:4s} {r['model']:11s} "
                  f"{r['n_rows']:>6,} rows, too few to score"); continue
        sc = "--" if r["src_lo"] is None else f"[{r['src_lo']:+.3f},{r['src_hi']:+.3f}]"
        sv = r["src_verdict"] if r["src_verdict"] == "unavailable" else r["src_verdict"]
        cov = "--" if r["assign_cover"] is None else f"{100*r['assign_cover']:.0f}%"
        nu = "--" if r["n_units"] is None else f"{r['n_units']:,}"
        print(f"{r['benchmark']:27s} {r['regime']:4s} {r['model']:11s} {r['model_mcc']:>6.3f} "
              f"{r['envelope']:>6.3f} {r['excess']:>+8.3f} "
              f"[{r['frag_lo']:+.3f},{r['frag_hi']:+.3f}] {r['frag_verdict']:<12.12s} "
              f"{sc:>18s} {sv:<12.12s} {cov:>5s} {r['n_rows']:>7,} {nu:>7s}")

    print("""
  excess            observed benchmark performance, FM minus baseline envelope. This is
                    the result. Neither interval changes it.
  fragment CI       stability across 1 kb test fragments of the kind the benchmark ships.
                    Answers whether the benchmark difference is stable on this task.
  source-aware CI   stability across parent genomes, same fragment scores. Answers whether
                    the difference would survive a different draw of source viruses.
                    Resolved under fragments and unresolved under sources is not a
                    contradiction; it bounds what the difference generalises over.
  cov               fraction of fragments placed on a source genome. Unplaced fragments are
                    singletons, so a recovered-unit interval with low cov stays somewhat
                    anti-conservative.""")
    print(f"  GUE phage and GUE virus_covid ship no parent provenance and reference-based")
    print(f"        {PHAGE_RECOVERY_NOTE} for phage, so source uncertainty is not estimable")
    print("  NN lookup scored at L0 only; L1 and L2 are defined by absence of its nucleotide hit")
    print("  model and learning rate are both selected on DEV, never on test")
    print("\n  symmetric-selection check, best FM minus best baseline re-maximised per replicate:")
    for r in res:
        if r.get("skipped"):
            continue
        y = r["sym"]
        if y["verdict"] != (r["src_verdict"] if r["n_units"] else r["frag_verdict"]):
            print(f"    {r['benchmark']:27s} {r['regime']:3s} dev-selected "
                  f"{r['src_verdict'] if r['n_units'] else r['frag_verdict']:14s} vs symmetric "
                  f"{y['excess']:+.3f} [{y['lo']:+.3f},{y['hi']:+.3f}] {y['verdict']}")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
