"""P1a: re-score VHDB on the HVUE-DISJOINT remainder.

Why
---
VHDB was built to be an independent replication of the HVUE host-tropism finding. It is not:
50.3% of its test rows have a >=95% identity / >=80% coverage match in HVUE v2 training data
(76.4% at id>=0.7/cov>=0.3). Shared provenance -- HVUE draws on NCBI Virus + BV-BRC +
Virus-Host DB, VHDB on Virus-Host DB labels + ViroBench genomes.

The VHDB result is not contaminated (trained on VHDB train, tested on VHDB test) but it is not
an independent test of HVUE either. This script asks the sharper question:

    does HViLM's excess survive on the VHDB test rows that have NO match in HVUE training?

If yes, the claim becomes "survives removal of all HVUE-overlapping sequence" -- stronger and
more precise than "independent replication" ever was. If no, the replication pillar retracts
and the positive rests on the single-benchmark homology-filtering result.

No retraining is needed: deleting test rows does not change training, so this re-scores the
already-saved predictions on the surviving subset. Union of the two HVUE configs' overlap sets
is removed (a row matching EITHER hard_70 or standard_95 training is dropped) -- the strict
reading.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score

R = "/data/nvidia/genomic-capability-unlearning/results_viral_bench"
PRED, GLM = f"{R}/preds_rescore", f"{R}/geneb_finetune"
XROWS = f"{R}/leaked_rows_crossdataset"
VHDB_INTERNAL = f"{R}/leaked_rows"
MODELS = ["hvilm", "lucavirus", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna"]


def load(pat):
    ps = sorted(glob.glob(pat))
    if not ps:
        return None, [], []
    yt, pred, prob = None, [], []
    for p in ps:
        z = np.load(p)
        if yt is None: yt = z["y_true"]
        elif not np.array_equal(yt, z["y_true"]): raise SystemExit(f"{p}: y_true mismatch")
        pred.append(z["y_pred"])
        pr = z["y_prob"] if "y_prob" in z else None
        prob.append(pr[:, 1] if (pr is not None and pr.ndim > 1) else pr)
    return yt, pred, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ident", default="95", choices=["70", "80", "90", "95"])
    ap.add_argument("--cov", default="80", choices=["30", "50", "70", "80"])
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--out", default="/home/nvidia/genomic-capability-unlearning/reports/"
                                     "P1a_vhdb_hvue_disjoint.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    esc = "vhdb_tropism_wf1000"
    yk, kmer, _ = load(f"{PRED}/hvue_v2__{esc}__kmer__s*__preds.npz")
    yc, cnn, _ = load(f"{PRED}/*{esc}*cnn_ceiling__s*__preds.npz")
    if not kmer or not cnn: raise SystemExit("baseline predictions missing")
    y = yk; n = len(y)
    assert np.array_equal(yk, yc)

    # union of rows overlapping EITHER HVUE config's training set
    drop = set()
    for cfg in ("hard_70_1000bp", "standard_95_1000bp"):
        p = (f"{XROWS}/VHDB_test__vs__HVUE_Host_Tropism_{cfg}_train"
             f"__id{a.ident}_cov{a.cov}.rows")
        if os.path.exists(p) and os.path.getsize(p):
            drop |= {int(x) for x in open(p).read().split()}
    keep = np.array([i for i in range(n) if i not in drop])
    units = pd.read_csv("/data/nvidia/vhdb_tropism/within_family_1000bp/test_full.csv")["cluster"].values

    print(f"VHDB test n={n:,}  removing {len(drop):,} rows overlapping HVUE train "
          f"(id>={a.ident}%, cov>={a.cov}%)  -> {len(keep):,} HVUE-disjoint rows "
          f"({100*len(keep)/n:.1f}%)")
    print(f"  pos rate: full {y.mean():.4f} -> disjoint {y[keep].mean():.4f}")
    uk = pd.unique(units[keep])
    print(f"  bootstrap units surviving: {len(uk):,} genome clusters "
          f"({len(keep)/max(len(uk),1):.1f} rows/unit)\n")

    models = {}
    for m in MODELS:
        c = []
        for j in sorted(glob.glob(f"{GLM}/{esc}__{m}__lr*__fullft.json")):
            if "__probs" in j or "_meanpool" in j or "_ep4" in j: continue
            d = json.load(open(j))
            if len(d.get("runs", [])) >= 3:
                c.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"]))
        if not c: print(f"  ! {m}: no 3-seed record, skipped"); continue
        dev, lr = max(c)
        ym, pr, pb = load(f"{GLM}/{esc}__{m}__lr{lr:g}__s*__preds.npz")
        if pr and np.array_equal(ym, y):
            ok = [i for i, p in enumerate(pr) if len(np.unique(p)) > 1]
            if not ok: print(f"  ! {m}: all seeds degenerate, excluded"); continue
            models[m] = (dev, lr, [pr[i] for i in ok], [pb[i] for i in ok], len(pr) - len(ok))

    by_u = {c: np.where(units == c)[0] for c in pd.unique(units)}
    keepmask = np.zeros(n, dtype=bool); keepmask[keep] = True

    def base_at(rows, pick=None):
        out = []
        for arm in (kmer, cnn):
            if pick is None:
                out.append(float(np.mean([matthews_corrcoef(y[rows], p[rows]) for p in arm])))
            else:
                out.append(matthews_corrcoef(y[rows], arm[pick % len(arm)][rows]))
        return max(out)

    res = {}
    for label, rows in (("full_test", np.arange(n)), ("hvue_disjoint", keep)):
        b0 = base_at(rows)
        km = float(np.mean([matthews_corrcoef(y[rows], p[rows]) for p in kmer]))
        cn = float(np.mean([matthews_corrcoef(y[rows], p[rows]) for p in cnn]))
        print(f"[{label}] n={len(rows):,}  k-mer={km:.4f}  CNN={cn:.4f}  floor={b0:.4f}")
        for m, (dev, lr, pr, pb, ncol) in sorted(
                models.items(), key=lambda kv: -np.mean([matthews_corrcoef(y[rows], p[rows])
                                                         for p in kv[1][2]])):
            sc = [matthews_corrcoef(y[rows], p[rows]) for p in pr]
            ap_ = [average_precision_score(y[rows], p[rows]) for p in pb if p is not None]
            exc = float(np.mean(sc) - b0)
            d = []
            for _ in range(a.reps):
                cs = rng.choice(pd.unique(units), size=len(pd.unique(units)), replace=True)
                rr = np.concatenate([by_u[c] for c in cs]); rr = rr[keepmask[rr]] if label != "full_test" else rr
                if len(rr) < 50 or len(np.unique(y[rr])) < 2: continue
                d.append(matthews_corrcoef(y[rr], pr[rng.integers(len(pr))][rr])
                         - base_at(rr, int(rng.integers(6))))
            d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) > 100 else (np.nan, np.nan)
            v = "SURVIVES" if lo > 0 else ("spans 0" if hi > 0 else "NEGATIVE")
            print(f"    {m:11s} mcc={np.mean(sc):.4f}+-{np.std(sc,ddof=1):.4f} "
                  f"{'auprc='+format(np.mean(ap_),'.4f') if ap_ else 'auprc=n/a':<14s} "
                  f"excess={exc:+.4f} CI[{lo:+.4f},{hi:+.4f}] {v}"
                  + (f"  ({ncol} seed(s) collapsed, excluded)" if ncol else ""))
            res.setdefault(label, {})[m] = dict(
                mcc=round(float(np.mean(sc)), 4), sd=round(float(np.std(sc, ddof=1)), 4),
                auprc=(round(float(np.mean(ap_)), 4) if ap_ else None), n_seeds=len(sc),
                n_collapsed=ncol, lr=lr, floor=round(b0, 4), kmer=round(km, 4), cnn=round(cn, 4),
                excess=round(exc, 4), ci=[round(float(lo), 4), round(float(hi), 4)], verdict=v)
        res.setdefault(label, {})["_meta"] = dict(n=int(len(rows)), floor=round(b0, 4),
                                                  pos_rate=round(float(y[rows].mean()), 4))
        print()

    out = dict(question="does HViLM's VHDB excess survive on rows with NO HVUE-train match?",
               overlap_threshold=f"id>={a.ident}% cov>={a.cov}%",
               n_test=int(n), n_removed=int(len(drop)), n_disjoint=int(len(keep)),
               n_units_disjoint=int(len(uk)), reps=a.reps, results=res)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
