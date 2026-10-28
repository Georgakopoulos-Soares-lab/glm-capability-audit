"""Gradient-boosted-tree baseline on k-mer features — a stronger composition baseline than
the logistic-regression k-mer arm.

Why
---
Our k-mer baseline is LOGISTIC REGRESSION on normalised k-mer frequencies, i.e. a LINEAR
model. Boosted trees on the same features capture non-linear interactions between k-mers
("high CpG AND low GC") that a linear model cannot. On tabular count features GBTs routinely
beat logistic regression, so the linear k-mer arm may be UNDERSTATING what composition alone
achieves -- the same class of error as the searched CNN turning out weaker than a trivial
nearest-neighbour lookup.

This runs XGBoost and sklearn's HistGradientBoosting on the identical feature matrix the
logistic arm uses (reusing `gue_baselines.kmer_feats`), selects on the SAME dev split, and
reports MCC/AUPRC so the numbers drop straight into `max(k-mer, CNN, NN-lookup, GBT)`.

Class weighting is applied (scale_pos_weight / balanced sample weights) because host tropism
is 3.5-19% positive and an unweighted tree ensemble collapses to the majority class.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "track_a_benchmarks")); sys.path.insert(0, HERE)
from gue_baselines import kmer_feats
import paths as P


def evaluate(y, pred, prob):
    out = {"mcc": round(float(matthews_corrcoef(y, pred)), 4),
           "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4)}
    if prob is not None and len(np.unique(y)) == 2:
        out["auprc"] = round(float(average_precision_score(y, prob)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--csvdir", default="")
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--train-cap", type=int, default=0)
    ap.add_argument("--predictions-dir", default="")
    # Seed for the estimator only. The train subsample below stays at random_state=0 so
    # every seed sees identical training rows, exactly as the fine-tuning arm holds its
    # --cap_train subsample fixed; this isolates optimiser/validation-split noise.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    OUT = P.sub("kmer_gbt"); os.makedirs(OUT, exist_ok=True)
    V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")
    root = a.csvdir or f"{V2}/{a.task}"
    rd = lambda p: pd.read_csv(f"{root}/{p}.csv").rename(columns={"text": "sequence"})
    tr, dv, te = rd("train"), rd("dev"), rd("test")
    if a.train_cap and len(tr) > a.train_cap:
        tr = tr.groupby("label", group_keys=False).sample(
            frac=min(1.0, a.train_cap / len(tr)), random_state=0).reset_index(drop=True)
    print(f"{a.task}: train={len(tr):,} dev={len(dv):,} test={len(te):,} "
          f"pos={100*tr.label.mean():.2f}%", flush=True)
    X = {k: kmer_feats(d.sequence.astype(str).str.upper().tolist(), a.ks)
         for k, d in (("tr", tr), ("dv", dv), ("te", te))}
    y = {k: d.label.values.astype(int) for k, d in (("tr", tr), ("dv", dv), ("te", te))}
    ncls = len(np.unique(y["tr"]))
    cnt = np.bincount(y["tr"], minlength=ncls)
    sw = (len(y["tr"]) / (ncls * np.maximum(cnt, 1)))[y["tr"]]     # sklearn 'balanced'

    res, best = {}, (-np.inf, None, None)
    try:
        import xgboost as xgb
        for nest, depth, lr in ((400, 6, 0.1), (800, 8, 0.05)):
            m = xgb.XGBClassifier(n_estimators=nest, max_depth=depth, learning_rate=lr,
                                  subsample=0.8, colsample_bytree=0.5, n_jobs=-1,
                                  tree_method="hist", eval_metric="logloss", random_state=a.seed,
                                  objective="multi:softprob" if ncls > 2 else "binary:logistic",
                                  num_class=ncls if ncls > 2 else None)
            m.fit(X["tr"], y["tr"], sample_weight=sw)
            pv = m.predict(X["dv"]); mv = matthews_corrcoef(y["dv"], pv)
            print(f"  xgb n={nest} d={depth} lr={lr}: dev mcc={mv:.4f}", flush=True)
            if mv > best[0]: best = (mv, f"xgb_n{nest}_d{depth}", m)
    except ImportError:
        print("  xgboost unavailable, using HistGradientBoosting only", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    for it, lf in ((400, 31), (800, 63)):
        m = HistGradientBoostingClassifier(max_iter=it, max_leaf_nodes=lf, learning_rate=0.1,
                                           early_stopping=True, random_state=a.seed)
        m.fit(X["tr"], y["tr"], sample_weight=sw)
        pv = m.predict(X["dv"]); mv = matthews_corrcoef(y["dv"], pv)
        print(f"  hgb it={it} leaves={lf}: dev mcc={mv:.4f}", flush=True)
        if mv > best[0]: best = (mv, f"hgb_it{it}_lf{lf}", m)

    devm, name, m = best
    pt = m.predict(X["te"])
    pp = m.predict_proba(X["te"])[:, 1] if ncls == 2 else None
    test = evaluate(y["te"], pt, pp)
    print(f"  SELECTED {name} (dev mcc={devm:.4f}) -> TEST {test}", flush=True)
    if a.predictions_dir:
        os.makedirs(a.predictions_dir, exist_ok=True)
        p = (f"{a.predictions_dir}/hvue_v2__{a.task.replace('/','__')}__kmergbt__s{a.seed}__preds.npz")
        np.savez_compressed(p, y_true=y["te"], y_pred=pt,
                            **({"y_prob": pp} if pp is not None else {}))
        print(f"  wrote {p}", flush=True)
    out = dict(task=a.task, arm="kmer_gbt", ks=a.ks, selected=name, dev_mcc=round(devm, 4),
               n_train=len(tr), n_test=len(te), test=test,
               note="GBT on the SAME k-mer features as the logistic arm; class-weighted")
    p = a.out or f"{OUT}/{a.task.replace('/','__')}__kmer_gbt.json"
    json.dump(out, open(p, "w"), indent=2); print(f"  wrote {p}")


if __name__ == "__main__":
    main()
