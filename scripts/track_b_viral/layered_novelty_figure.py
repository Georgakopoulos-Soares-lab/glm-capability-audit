"""Deliverables 2-4: graded translated-homology curve, layered-novelty ladder, group bootstrap.

Reads the masks the audit already cached, so it never re-runs MMseqs2 and never
redefines a stratum. Three outputs:

  (2) GRADED CURVE. Within the nucleotide-no-hit population, model excess over
      the corrected floor against CONTINUOUS translated relatedness -- bitscore,
      length-normalised bitscore, query coverage, aligned aa length, -log10(E),
      and bitscore x coverage -- in FIXED pre-declared quartiles, with strict
      no-hit as a separate final category.  Descriptive only.  No threshold is
      searched; no cutoff is chosen.  The question is whether excess TRACKS
      relatedness or is flat, and flat means any detectable relationship suffices.

  (3) LAYERED LADDER. Surviving fraction and per-model excess at each level, with
      n_rows AND n_units (source genome / cluster / genus) so small strata
      declare their own weakness.

  (4) GROUP BOOTSTRAP + EQUIVALENCE. Resampling unit is the biological group, not
      the row; viral sequences are not iid and a row bootstrap overstates
      precision exactly where the strata are smallest.  On the phage N-/T-
      stratum an explicit two-one-sided-test verdict against a PRE-DECLARED MCC
      margin, because "not significant" is not "equivalent" and the paper must
      say which it has.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _arm_guard import sweep_arm_ready                                   # noqa: E402

R = os.environ.get("VB_VIRAL_OUT",
                   "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
PRED, GLM, SWEEP = f"{R}/preds_rescore", f"{R}/geneb_finetune", f"{R}/capacity_sweep"
MODELS = ["hvilm", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna", "lucavirus"]

# Pre-declared before any curve was plotted. Quartiles of the hit population, so
# the bin edges are a property of the data's spread, not of where a p-value moves.
QUANTILES = [0.0, 0.25, 0.50, 0.75, 1.0]
# Pre-declared equivalence margin, in MCC. 0.02 is the smallest excess this study
# has ever called meaningful; anything inside +-0.02 we are willing to call "no
# practically relevant difference from the floor".
EQ_MARGIN = 0.02


def load(pat):
    ps = sorted(glob.glob(pat))
    if not ps:
        return None, []
    yt, pr = None, []
    for p in ps:
        z = np.load(p)
        if yt is None:
            yt = z["y_true"]
        elif not np.array_equal(yt, z["y_true"]):
            return None, []
        pr.append(z["y_pred"])
    return yt, pr


def collect_arms(esc, ke, n, sweep_tag, require):
    arms, excluded = {}, {}
    specs = (("kmer", f"{PRED}/hvue_v2__{ke}__kmer__s*__preds.npz", None),
             ("kmer_gbt", f"{PRED}/hvue_v2__{ke}__kmergbt__s*__preds.npz", None),
             ("aa_gbt", f"{PRED}/hvue_v2__{ke}__aagbt__s*__preds.npz", None),
             ("tax_prior", f"{PRED}/hvue_v2__{ke}__taxprior__s*__preds.npz", None),
             ("nn_lookup", f"{PRED}/hvue_v2__*{ke}*__nnlookup__s*__preds.npz", None),
             ("cnn", f"{PRED}/*{esc}*cw_balanced__cnn_ceiling__s*__preds.npz",
              f"{SWEEP}/{sweep_tag}.json" if sweep_tag else None),
             ("transformer", f"{PRED}/*{esc}*transformer__cnn_ceiling__s*__preds.npz",
              f"{SWEEP}/{sweep_tag}__transformer.json" if sweep_tag else None))
    for tag, pat, sj in specs:
        if sj is not None:
            ok, why = sweep_arm_ready(sj, pat)
            if not ok:
                excluded[tag] = why
                print(f"  ! {tag} EXCLUDED: {why}")
                if require:
                    sys.exit(f"--require-arms: refusing to score without a complete {tag}")
                continue
        yy, pp = load(pat)
        if pp and yy is not None and len(yy) == n:
            arms[tag] = pp
    return arms, excluded


def collect_models(esc, n):
    out = {}
    for m in MODELS:
        c = []
        for j in sorted(glob.glob(f"{GLM}/{esc}__{m}__lr*__fullft.json")):
            if any(x in j for x in ("_meanpool", "_ep4", "_truncated")):
                continue
            d = json.load(open(j))
            if len(d.get("runs", [])) >= 3:
                sfx = "".join(s for s in ("__probs", "__ep16", "__ep32", "__ep64") if s in j)
                c.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"], sfx))
        if not c:
            continue
        _, lr, sfx = max(c)
        ym, pr = load(f"{GLM}/{esc}__{m}__lr{lr:g}{sfx}__s*__preds.npz")
        if pr and ym is not None and len(ym) == n:
            ok = [p for p in pr if len(np.unique(p)) > 1]
            if ok:
                out[m] = ok
    return out


def group_boot(y, arms, models, mask, units, reps, seed=0):
    """Resample GROUPS. Returns per-model CI plus the draw array for equivalence."""
    rng = np.random.default_rng(seed)
    rows = np.where(mask)[0]
    u = pd.unique(units[rows])
    idx = {c: rows[units[rows] == c] for c in u}
    draws = {m: [] for m in models}
    for _ in range(reps):
        r = np.concatenate([idx[c] for c in rng.choice(u, size=len(u), replace=True)])
        yy = y[r]
        if len(np.unique(yy)) < 2:
            continue
        b = max(matthews_corrcoef(yy, v[rng.integers(len(v))][r]) for v in arms.values())
        for m, pr in models.items():
            draws[m].append(matthews_corrcoef(yy, pr[rng.integers(len(pr))][r]) - b)
    return {m: np.asarray(d) for m, d in draws.items() if d}, len(u)


def verdict(d, margin=EQ_MARGIN):
    """Two one-sided tests. A wide interval is INCONCLUSIVE, never equivalence."""
    lo, hi = np.percentile(d, [2.5, 97.5])
    if lo > 0:
        return "SUPERIOR", lo, hi
    if hi < 0:
        return "INFERIOR", lo, hi
    if lo > -margin and hi < margin:
        return f"EQUIVALENT (whole CI inside +-{margin})", lo, hi
    return (f"INCONCLUSIVE (CI spans 0 AND exceeds the +-{margin} margin; "
            f"not shown to differ is NOT shown to be equivalent)"), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--esc", required=True); ap.add_argument("--kmer-esc", default="")
    ap.add_argument("--sweep-tag", default="")
    ap.add_argument("--masks", required=True, help="npz from hidden_homology_audit --cache")
    ap.add_argument("--hits-npz", default="", help="optional npz of per-row translated stats")
    ap.add_argument("--units-csv", default="", help="test-row-order csv of grouping columns")
    ap.add_argument("--unit-col", default="cluster")
    ap.add_argument("--tax-test", default=""); ap.add_argument("--tax-train", default="")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--focus", default="", help="model for the equivalence verdict")
    ap.add_argument("--require-arms", action="store_true")
    ap.add_argument("--out", default=os.environ.get(
        "VB_REPORTS", "/home/nvidia/genomic-capability-unlearning/reports") + "/layered")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    te = pd.read_csv(a.test_csv).rename(columns={"text": "sequence"})
    y = te.label.values.astype(int); n = len(te)
    z = np.load(a.masks)
    hasN, hasT, hasP, tbits = z["hasN"], z["hasT"], z["hasP"], z["tbits"]
    ke = a.kmer_esc or a.esc

    arms, excluded = collect_arms(a.esc, ke, n, a.sweep_tag, a.require_arms)
    models = collect_models(a.esc, n)
    if not arms or not models:
        sys.exit(f"arms={list(arms)} models={list(models)} -- nothing to score")

    # grouping units, most biological available
    units, usrc = np.arange(n), "ROW (anti-conservative; no grouping metadata shipped)"
    if a.units_csv and os.path.exists(a.units_csv):
        u = pd.read_csv(a.units_csv)
        if len(u) == n and a.unit_col in u.columns:
            units, usrc = u[a.unit_col].values, a.unit_col

    unseen = {}
    if a.tax_train and a.tax_test:
        tt, tx = pd.read_csv(a.tax_train), pd.read_csv(a.tax_test)
        if len(tx) == n:
            for col in ("species", "genus", "family"):
                if col in tt.columns and col in tx.columns:
                    unseen[col] = ~tx[col].astype(str).isin(
                        set(tt[col].dropna().astype(str))).values

    levels = [("all", np.ones(n, bool)),
              ("nucleotide_no_hit", ~hasN),
              ("translated_no_hit", ~hasN & ~hasT)]
    if hasP.any():
        levels.append(("protein_no_hit", ~hasN & ~hasT & ~hasP))
    for col in ("species", "genus", "family"):
        if col in unseen:
            levels.append((f"unseen_{col}", ~hasN & ~hasT & unseen[col]))

    out = dict(name=a.name, n_test=int(n), unit=usrc, arms=list(arms),
               excluded_arms=excluded, equivalence_margin=EQ_MARGIN,
               bin_quantiles=QUANTILES, panelA=[], panelB={}, graded={})

    print(f"\n=== {a.name}   unit={usrc}   arms={', '.join(arms)}")
    if excluded:
        print(f"    EXCLUDED: {', '.join(excluded)}")
    print(f"\n  PANEL A -- apparent novelty collapses with search sensitivity")
    print(f"    {'level':22s} {'n_rows':>8s} {'frac':>7s} {'n_units':>8s} {'n_genus':>8s}")
    tx = pd.read_csv(a.tax_test) if (a.tax_test and os.path.exists(a.tax_test)) else None
    for lab, m in levels:
        nu = len(pd.unique(units[m])) if m.sum() else 0
        ng = (tx.loc[m, "genus"].nunique() if tx is not None and "genus" in tx.columns and m.sum()
              else None)
        print(f"    {lab:22s} {int(m.sum()):>8,} {m.mean():>7.1%} {nu:>8,} "
              f"{(ng if ng is not None else '-'):>8}")
        out["panelA"].append(dict(level=lab, n_rows=int(m.sum()), frac=round(float(m.mean()), 4),
                                  n_units=int(nu), n_genus=(int(ng) if ng is not None else None)))

    print(f"\n  PANEL B -- excess over the corrected floor, group bootstrap ({a.reps:,} reps)")
    for lab, m in levels:
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            print(f"    {lab:22s} n={int(m.sum()):>6,}  skipped (too few rows / single class)")
            continue
        fl = {k: float(np.mean([matthews_corrcoef(y[m], p[m]) for p in v])) for k, v in arms.items()}
        floor = max(fl.values()); farm = max(fl, key=fl.get)
        draws, nu = group_boot(y, arms, models, m, units, a.reps)
        print(f"    {lab:22s} n={int(m.sum()):>6,} units={nu:>5,} floor={floor:.4f} ({farm})")
        rr = {}
        for mo in sorted(models, key=lambda k: -np.mean([matthews_corrcoef(y[m], p[m])
                                                         for p in models[k]])):
            sc = float(np.mean([matthews_corrcoef(y[m], p[m]) for p in models[mo]]))
            e = sc - floor
            d = draws.get(mo)
            if d is None:
                continue
            v, lo, hi = verdict(d)
            print(f"        {mo:11s} {sc:.4f}  excess {e:+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]  {v}")
            rr[mo] = dict(mcc=round(sc, 4), excess=round(e, 4),
                          ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                          p_le_0=round(float((d <= 0).mean()), 4), verdict=v)
        out["panelB"][lab] = dict(n_rows=int(m.sum()), n_units=int(nu), floor=round(floor, 4),
                                  floor_arm=farm,
                                  baselines={k: round(v, 4) for k, v in fl.items()}, models=rr)

    # ---- graded curve -------------------------------------------------------
    base = ~hasN
    stats = {"bitscore": tbits}
    if a.hits_npz and os.path.exists(a.hits_npz):
        h = np.load(a.hits_npz)
        for k in ("bits_per_aa", "qcov", "alnlen", "neglog10E", "bits_x_cov"):
            if k in h.files:
                stats[k] = h[k]
    print(f"\n  GRADED CURVE -- excess vs CONTINUOUS translated relatedness")
    print(f"    (within the {int(base.sum()):,} nucleotide-no-hit rows; fixed quartiles, "
          f"declared before plotting; strict no-hit is its own final category)")
    for sname, sv in stats.items():
        hit = base & hasT
        if hit.sum() < 100:
            continue
        qs = np.quantile(sv[hit], QUANTILES)
        bins = [("strict no-hit", base & ~hasT)]
        for i in range(4):
            lo, hi = qs[i], qs[i + 1]
            sel = base & hasT & (sv >= lo) & ((sv <= hi) if i == 3 else (sv < hi))
            bins.append((f"Q{i+1} [{lo:.3g},{hi:.3g}]", sel))
        print(f"\n    -- {sname}")
        rows = []
        for lab, sel in bins:
            if sel.sum() < 40 or len(np.unique(y[sel])) < 2:
                print(f"       {lab:24s} n={int(sel.sum()):>6,}  too small to score")
                rows.append(dict(bin=lab, n=int(sel.sum()), skipped=True)); continue
            f2 = max(float(np.mean([matthews_corrcoef(y[sel], p[sel]) for p in v]))
                     for v in arms.values())
            ex = {mo: round(float(np.mean([matthews_corrcoef(y[sel], p[sel]) for p in pr])) - f2, 4)
                  for mo, pr in models.items()}
            best = max(ex, key=ex.get)
            print(f"       {lab:24s} n={int(sel.sum()):>6,} floor={f2:.4f}  "
                  f"{best}={ex[best]:+.4f}")
            rows.append(dict(bin=lab, n=int(sel.sum()), floor=round(f2, 4), excess=ex))
        out["graded"][sname] = rows

    # ---- equivalence on the focus stratum -----------------------------------
    if a.focus and a.focus in models:
        m = ~hasN & ~hasT
        if m.sum() >= 30 and len(np.unique(y[m])) > 1:
            draws, nu = group_boot(y, arms, models, m, units, a.reps)
            d = draws.get(a.focus)
            if d is not None:
                v, lo, hi = verdict(d)
                print(f"\n  EQUIVALENCE TEST -- {a.focus} on the strict no-detectable-homology "
                      f"stratum\n    n_rows={int(m.sum()):,} n_units={nu:,} "
                      f"margin=+-{EQ_MARGIN}\n    excess CI [{lo:+.4f}, {hi:+.4f}]  ->  {v}")
                out["equivalence"] = dict(model=a.focus, n_rows=int(m.sum()), n_units=int(nu),
                                          margin=EQ_MARGIN, ci_lo=round(float(lo), 4),
                                          ci_hi=round(float(hi), 4), verdict=v)

    p = f"{a.out}/{a.name}__layered.json"
    json.dump(out, open(p, "w"), indent=2, default=str)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
