"""One command that produces the paper's viral table: excess, filtered excess, bootstrap CI.

Combines the three things that decide the paper:
  H1  excess          = best gLM - max(k-mer, CNN)          [all arms 3 seeds, dev-selected]
  H3  filtered excess = the same after deleting test rows homologous to train
  CI  cluster bootstrap at the leakage unit, resampling seeds as well as units

Reads only saved predictions, so it never retrains and never re-selects on test.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score

R    = "/data/nvidia/genomic-capability-unlearning/results_viral_bench"
PRED = f"{R}/preds_rescore"
GLM  = f"{R}/geneb_finetune"
LEAK = f"{R}/leaked_rows"
CLUS = f"{R}/clusters"

TASKS = {
  "hvue_v2__Host_Tropism__hard_70_1000bp":
      dict(esc="Host_Tropism__hard_70_1000bp",
           units=f"{CLUS}/hvue_v2__Host_Tropism__hard_70_1000bp__genomeunits.csv",
           unit_note="INFERRED (2.3 rows/unit, 41% singletons) -> CI is a LOWER BOUND on width"),
  "hvue_v2__Host_Tropism__standard_95_1000bp":
      dict(esc="Host_Tropism__standard_95_1000bp",
           units=f"{CLUS}/hvue_v2__Host_Tropism__standard_95_1000bp__genomeunits.csv",
           unit_note="INFERRED (2.3 rows/unit, 41% singletons) -> CI is a LOWER BOUND on width"),
  "vhdb_tropism_wf1000":
      dict(esc="vhdb_tropism_wf1000",
           units="/data/nvidia/vhdb_tropism/within_family_1000bp/test_full.csv",
           unit_note="SHIPPED genome clusters (10.8 rows/unit) -> CI is exact"),
}
MODELS = ["hvilm", "lucavirus", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna"]


def load(pattern):
    ps = sorted(glob.glob(pattern))
    if not ps:
        return None, [], []
    yt, pred, prob = None, [], []
    for p in ps:
        z = np.load(p)
        if yt is None:
            yt = z["y_true"]
        elif not np.array_equal(yt, z["y_true"]):
            raise SystemExit(f"{p}: y_true mismatch")
        pred.append(z["y_pred"])
        prob.append(z["y_prob"] if "y_prob" in z else None)
    return yt, pred, prob


def baselines(esc):
    """max(k-mer, CNN); k-mer taken as the BEST over training regimes (capped / full), which
    is the honest ceiling -- capping helped k-mer on both HVUE cells."""
    arms = {}
    for tag, pat in (("kmer_cap",  f"{PRED}/hvue_v2__{esc}__kmer__cap*__s*__preds.npz"),
                     ("kmer_full", f"{PRED}/hvue_v2__{esc}__kmer__s*__preds.npz"),
                     ("cnn",       f"{PRED}/*{esc}*cnn_ceiling__s*__preds.npz")):
        y, pr, _ = load(pat)
        if pr:
            arms[tag] = (y, pr)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--filters", nargs="+", default=["id70_cov30", "id90_cov50"])
    ap.add_argument("--out", default="/home/nvidia/genomic-capability-unlearning/reports/"
                                     "FINAL_VIRAL_TABLE.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    allout = {}

    for task, cfg in TASKS.items():
        esc = cfg["esc"]
        arms = baselines(esc)
        if "cnn" not in arms or not any(k.startswith("kmer") for k in arms):
            print(f"\n### {task}: SKIP (baselines not ready: have {sorted(arms)})"); continue
        y = arms["cnn"][0]
        n = len(y)

        models = {}
        for m in MODELS:
            cands = []
            for j in sorted(glob.glob(f"{GLM}/{esc}__{m}__lr*__fullft.json")):
                # EXCLUDE the __probs re-runs from LR selection. They re-run configs that
                # already have a record, so letting them compete on dev means selecting the
                # luckier of two runs of the SAME config -- winner's curse on duplicates.
                # The originals define MCC and the LR; __probs exists only to supply
                # probabilities for AUPRC at that already-chosen LR.
                if "__probs" in j:
                    continue
                d = json.load(open(j))
                if len(d.get("runs", [])) >= 3:
                    cands.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"], j))
            if not cands:
                continue
            dev, lr, j = max(cands)
            sfx = ""
            ym, pr, _ = load(f"{GLM}/{esc}__{m}__lr{lr:g}{sfx}__s*__preds.npz")
            if pr and np.array_equal(ym, y):
                # A seed that collapsed to one class is NOT a capability estimate -- averaging
                # a 0.0 into the mean conflates "could not be trained reliably" with "has no
                # capability". Those are different claims and both deserve reporting, so
                # degenerate seeds are excluded from the mean and counted separately.
                # (LucaVirus escapes the majority-class basin stochastically: the SAME lr and
                # seed escaped in one run and collapsed in another.)
                ok = [p for p in pr if len(np.unique(p)) > 1]
                n_collapsed = len(pr) - len(ok)
                if not ok:
                    print(f"      {m:11s} ALL {len(pr)} seeds collapsed to one class -> "
                          f"NOT a capability estimate, excluded")
                    continue
                if n_collapsed:
                    print(f"      note: {m} had {n_collapsed}/{len(pr)} seeds collapse; mean "
                          f"is over the {len(ok)} non-degenerate seeds")
                models[m] = (dev, lr, ok, n_collapsed)

        units = pd.read_csv(cfg["units"])["cluster"].values
        uniq = pd.unique(units)
        by_u = {c: np.where(units == c)[0] for c in uniq}
        print(f"\n### {task}   n_test={n:,}  units={len(uniq):,}  ({cfg['unit_note']})")

        def base_at(rows, pick):
            """Baseline on `rows`. With `pick` given (a bootstrap replicate) use that seed;
            with pick empty use the MEAN over seeds -- the point estimate must describe the
            same baseline the CI resamples, or the two disagree (seed 42 alone read 0.5640
            against a bootstrap mean of 0.5386 on VHDB)."""
            out = []
            for k in arms:
                pr = arms[k][1]
                if k in pick:
                    out.append(matthews_corrcoef(y[rows], pr[pick[k] % len(pr)][rows]))
                else:
                    out.append(float(np.mean([matthews_corrcoef(y[rows], p[rows]) for p in pr])))
            return max(out)

        res = {}
        for fname in ["none"] + a.filters:
            if fname == "none":
                keep = np.arange(n)
            else:
                p = f"{LEAK}/{task}__{fname}.rows"
                if not os.path.exists(p):
                    continue
                drop = {int(x) for x in open(p).read().split()} if os.path.getsize(p) else set()
                keep = np.array([i for i in range(n) if i not in drop])
            if len(keep) < 50 or len(np.unique(y[keep])) < 2:
                continue
            keepmask = np.zeros(n, dtype=bool); keepmask[keep] = True
            b0 = base_at(keep, {})
            print(f"  [{fname:10s}] n={len(keep):6,} ({100*len(keep)/n:5.1f}%)  floor={b0:.4f}")
            for m, (dev, lr, pr, ncol) in sorted(models.items(),
                                           key=lambda kv: -np.mean([matthews_corrcoef(y[keep], p[keep]) for p in kv[1][2]])):
                sc = [matthews_corrcoef(y[keep], p[keep]) for p in pr]
                exc = float(np.mean(sc) - b0)
                d = []
                for _ in range(a.reps):
                    cs = rng.choice(uniq, size=len(uniq), replace=True)
                    rows = np.concatenate([by_u[c] for c in cs])
                    rows = rows[keepmask[rows]]   # boolean mask, not np.isin: the latter is
                                                  # O(n log n) per replicate and dominates
                                                  # runtime on the 51k-row HVUE cells
                    if len(rows) < 50 or len(np.unique(y[rows])) < 2:
                        continue
                    pk = {k: rng.integers(len(arms[k][1])) for k in arms}
                    d.append(matthews_corrcoef(y[rows], pr[rng.integers(len(pr))][rows])
                             - base_at(rows, pk))
                d = np.array(d)
                lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) > 100 else (np.nan, np.nan))
                verdict = "SURVIVES" if lo > 0 else ("spans 0" if hi > 0 else "NEGATIVE")
                print(f"      {m:11s} lr={lr:<8g} mcc={np.mean(sc):.4f}+-{np.std(sc, ddof=1):.4f} "
                      f"excess={exc:+.4f} CI[{lo:+.4f},{hi:+.4f}] {verdict}")
                res.setdefault(fname, {})[m] = dict(
                    mcc=round(float(np.mean(sc)), 4), sd=round(float(np.std(sc, ddof=1)), 4),
                    n_seeds=len(sc), n_collapsed_seeds=ncol, lr=lr, floor=round(b0, 4),
                    excess=round(exc, 4),
                    ci=[round(float(lo), 4), round(float(hi), 4)], verdict=verdict)
            res.setdefault(fname, {})["_meta"] = dict(n_kept=int(len(keep)), floor=round(b0, 4))
        allout[task] = dict(n_test=int(n), n_units=int(len(uniq)),
                            unit_note=cfg["unit_note"], filters=res)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(allout, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
