"""THE TREND ARGUMENT: how lookup accuracy, surviving sample, and model excess move together
as the homology filter tightens.

Interpretation rule (stated before looking at the numbers)
----------------------------------------------------------
Tightening the filter removes test rows that are homologous to training data. Therefore:

  * An excess that GROWS or HOLDS under tightening is evidence the advantage is REAL.
    Leakage-driven performance depends on the removed rows, so it must shrink.
  * An excess that SHRINKS toward zero is evidence the advantage is LEAKAGE.
  * The NN-lookup curve is the leakage yardstick: if lookup never reaches chance (MCC 0),
    exploitable homology was never removed, and no cell on that curve is "clean".

Two confounds this rule must survive, both reported per cell:
  * Surviving sample size. A shrinking excess with a collapsing n is ambiguous (loss of power
    vs loss of signal), so n_kept and n_clusters are printed beside every excess.
  * A collapsing floor. If the baselines fall faster than the model, excess can grow for
    uninteresting reasons; the per-arm floor is printed so that is visible.

Floor is max over ALL available baseline arms at that cell -- k-mer (logistic), k-mer GBT,
searched CNN, NN-lookup, from-scratch transformer -- each re-scored on the surviving rows.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

R = "/data/nvidia/genomic-capability-unlearning/results_viral_bench"
PRED, GLM, LEAK = f"{R}/preds_rescore", f"{R}/geneb_finetune", f"{R}/leaked_rows"
MODELS = ["hvilm", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna", "lucavirus"]
IDENTS, COVS = [0.70, 0.80, 0.90, 0.95], [0.30, 0.50, 0.70, 0.80]


def load(pat):
    ps = sorted(glob.glob(pat))
    if not ps:
        return None, []
    yt, pr = None, []
    for p in ps:
        z = np.load(p)
        if yt is None: yt = z["y_true"]
        elif not np.array_equal(yt, z["y_true"]): return None, []
        pr.append(z["y_pred"])
    return yt, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="leaked_rows prefix")
    ap.add_argument("--esc", required=True, help="prediction-file task string")
    ap.add_argument("--cluster-csv", default="")
    ap.add_argument("--out", default="/home/nvidia/genomic-capability-unlearning/reports/trend")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    arms = {}
    for tag, pat in (("kmer",  f"{PRED}/hvue_v2__{a.esc}__kmer__s*__preds.npz"),
                     ("kmer_cap", f"{PRED}/hvue_v2__{a.esc}__kmer__cap*__s*__preds.npz"),
                     ("kmer_gbt", f"{PRED}/hvue_v2__{a.esc}__kmergbt__s*__preds.npz"),
                     ("cnn", f"{PRED}/*{a.esc}*cnn_ceiling__s*__preds.npz"),
                     ("transformer", f"{PRED}/*{a.esc}*transformer*__preds.npz")):
        y, p = load(pat)
        if p: arms[tag] = (y, p)
    if not arms: raise SystemExit("no baseline predictions found")
    y = next(iter(arms.values()))[0]; n = len(y)

    models = {}
    for m in MODELS:
        c = []
        for j in sorted(glob.glob(f"{GLM}/{a.esc}__{m}__lr*__fullft.json")):
            if any(x in j for x in ("__probs", "_meanpool", "_ep4")): continue
            d = json.load(open(j))
            if len(d.get("runs", [])) >= 3:
                c.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"]))
        if not c: continue
        ym, pr = load(f"{GLM}/{a.esc}__{m}__lr{max(c)[1]:g}__s*__preds.npz")
        if pr and np.array_equal(ym, y):
            ok = [p for p in pr if len(np.unique(p)) > 1]
            if ok: models[m] = ok

    # NN-lookup predictions, if the baseline arm saved them; else read its scalar from json
    nnfile = glob.glob(f"{PRED}/*{a.esc}*nnlookup*preds.npz")
    nn_pred = np.load(nnfile[0])["y_pred"] if nnfile else None

    clusters = (pd.read_csv(a.cluster_csv)["cluster"].values
                if a.cluster_csv and os.path.exists(a.cluster_csv) else None)

    rows = []
    cells = [("none", None)] + [(f"id{int(i*100)}_cov{int(c*100)}", (i, c))
                                for i in IDENTS for c in COVS]
    for label, _ in cells:
        if label == "none":
            keep = np.arange(n)
        else:
            p = f"{LEAK}/{a.task}__{label}.rows"
            if not os.path.exists(p): continue
            drop = {int(x) for x in open(p).read().split()} if os.path.getsize(p) else set()
            keep = np.array([i for i in range(n) if i not in drop])
        if len(keep) < 50 or len(np.unique(y[keep])) < 2: continue
        fl = {k: float(np.mean([matthews_corrcoef(y[keep], q[keep]) for q in v[1]]))
              for k, v in arms.items()}
        if nn_pred is not None:
            fl["nn_lookup"] = float(matthews_corrcoef(y[keep], nn_pred[keep]))
        floor = max(fl.values())
        rec = dict(cell=label, n_kept=int(len(keep)), frac=round(len(keep)/n, 4),
                   n_clusters=(int(pd.unique(clusters[keep]).size) if clusters is not None else None),
                   pos_rate=round(float(y[keep].mean()), 4),
                   floor=round(floor, 4), floor_arm=max(fl, key=fl.get),
                   **{f"base_{k}": round(v, 4) for k, v in fl.items()})
        for m, pr in models.items():
            sc = float(np.mean([matthews_corrcoef(y[keep], q[keep]) for q in pr]))
            rec[f"{m}_mcc"] = round(sc, 4); rec[f"{m}_excess"] = round(sc - floor, 4)
        rows.append(rec)

    df = pd.DataFrame(rows)
    p = f"{a.out}/{a.task}__trend.csv"; df.to_csv(p, index=False)
    cols = ["cell", "n_kept", "n_clusters", "floor", "floor_arm"] + \
           [c for c in df.columns if c.endswith("_excess")]
    print(f"\n=== {a.task} ===")
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    # trend verdict per model: does the excess grow/hold/shrink as the filter tightens?
    print("\n  TREND (excess at harshest id70_cov30 vs unfiltered):")
    base = df[df.cell == "none"]
    harsh = df[df.cell == "id70_cov30"]
    if len(base) and len(harsh):
        for m in models:
            b = base[f"{m}_excess"].iloc[0]; h = harsh[f"{m}_excess"].iloc[0]
            verdict = ("GROWS -> advantage is real" if h > b + 0.01 else
                       "HOLDS -> advantage is real" if h > b - 0.01 else
                       "SHRINKS -> consistent with leakage")
            print(f"    {m:11s} {b:+.4f} -> {h:+.4f}  ({h-b:+.4f})  {verdict}")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
