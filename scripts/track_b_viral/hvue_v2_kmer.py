"""k-mer composition baseline on HVUE v2 — the other half of `max(k-mer, CNN)`.

No k-mer baseline had been run on HVUE v2, so `max(k-mer, fair CNN)` was not computable
and the gLM excess could not be stated honestly. This fills that gap.

Reuses `gue_baselines.kmer_feats` (normalised k-mer frequency vectors, multiple k
concatenated) rather than reimplementing it. Discipline matched to the CNN arm:

* k = 3..6, `C` swept on the authors' OWN dev split (never test, never train-internal).
* `class_weight="balanced"` — mandatory here. Host_Tropism is 3.5-5.5% positive; an
  unweighted linear model collapses to the majority class and would understate the
  baseline on exactly the task under test.
* **MCC and AUPRC reported, never accuracy.** Predicting all-negative on Host_Tropism
  scores 96.5% accuracy and MCC 0.
* Test predictions saved so bootstrap CIs and homology-filtered re-scoring are possible
  without refitting.

k-mer is deterministic given the split, so it has no seed spread; bootstrap CIs over test
rows are the only interval that means anything for this arm, and the CI is what carries
the minority-class sample size into the comparison.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (matthews_corrcoef, average_precision_score, roc_auc_score,
                             f1_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "track_a_benchmarks"))
sys.path.insert(0, HERE)
from gue_baselines import kmer_feats          # reused, not reimplemented
import paths as P

V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")
OUT = P.sub("hvue_v2_kmer")
C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0]


def evaluate(y, pred, prob):
    return {
        "mcc": round(float(matthews_corrcoef(y, pred)), 4),
        # AUPRC/AUROC are binary-only here; virus_covid is 9-class and sklearn raises on a
        # 1-d prob vector. Emit None rather than crashing the whole run.
        "auprc": (round(float(average_precision_score(y, prob)), 4)
                  if prob is not None and len(np.unique(y)) == 2 else None),
        "auroc": (round(float(roc_auc_score(y, prob)), 4)
                  if prob is not None and len(np.unique(y)) == 2 else None),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "pos_rate_pred": round(float((pred == 1).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="<Task>/<config>, e.g. Host_Tropism/hard_70_1000bp")
    ap.add_argument("--csvdir", default="",
                    help="read <csvdir>/{train,dev,test}.csv instead of HVUE_V2_DIR/<task>; "
                         "for externally-built tasks such as the Virus-Host DB tropism set")
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--train_cap", type=int, default=0,
                    help="class-stratified cap; pass the SAME value used for the CNN/gLM arms")
    ap.add_argument("--select_on", default="mcc", choices=["mcc", "auprc"])
    ap.add_argument("--predictions_dir", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    # train_cap MUST be in the filename. Without it a capped run and a full-train run write
    # to the same path, and --force silently replaces one measurement with a different one
    # -- which is exactly how the original capped HVUE hard_70 k-mer result was lost.
    _cap = f"__cap{a.train_cap}" if a.train_cap else ""
    tag = f"{OUT}/{a.task.replace('/', '__')}__kmer{_cap}.json"
    if os.path.exists(tag) and not a.force:
        print(f"SKIP {tag} exists (--force to redo)"); return

    root = a.csvdir or f"{V2}/{a.task}"
    rd = lambda part: pd.read_csv(f"{root}/{part}.csv")
    tr, dv, te = rd("train"), rd("dev"), rd("test")
    if a.train_cap and len(tr) > a.train_cap:
        tr = tr.groupby("label", group_keys=False)[list(tr.columns)].apply(
            lambda g: g.sample(n=max(1, int(round(a.train_cap * len(g) / len(tr)))), random_state=0)
        ).reset_index(drop=True)
    print(f"{a.task}: train={len(tr)} dev={len(dv)} test={len(te)}  "
          f"train pos={100*tr.label.mean():.2f}%  test pos={100*te.label.mean():.2f}%  "
          f"k={a.ks}", flush=True)

    Xtr = kmer_feats(tr.sequence.str.upper().tolist(), a.ks)
    Xdv = kmer_feats(dv.sequence.str.upper().tolist(), a.ks)
    Xte = kmer_feats(te.sequence.str.upper().tolist(), a.ks)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xdv_s, Xte_s = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
    ytr, ydv, yte = (d.label.values.astype(int) for d in (tr, dv, te))

    search = []
    best = (-np.inf, None)
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                 n_jobs=-1).fit(Xtr_s, ytr)
        pd_ = clf.predict(Xdv_s)
        _d = clf.predict_proba(Xdv_s); pp_ = _d[:, 1] if _d.shape[1] == 2 else None
        m = evaluate(ydv, pd_, pp_)
        search.append(dict(C=C, dev=m))
        _a = m['auprc']
        print(f"  C={C:<7g} dev mcc={m['mcc']:.4f} "
              f"auprc={_a:.4f}" if _a is not None else
              f"  C={C:<7g} dev mcc={m['mcc']:.4f} auprc=n/a(multiclass)", flush=True)
        if m[a.select_on] > best[0]:
            best = (m[a.select_on], C)
    C = best[1]

    # Refit on train+dev at the dev-selected C, then score test ONCE.
    Xfull = np.vstack([Xtr, Xdv]); yfull = np.concatenate([ytr, ydv])
    scf = StandardScaler().fit(Xfull)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                             n_jobs=-1).fit(scf.transform(Xfull), yfull)
    pred = clf.predict(scf.transform(Xte))
    _pp = clf.predict_proba(scf.transform(Xte))
    prob = _pp[:, 1] if _pp.shape[1] == 2 else None      # multiclass: no single-column prob
    test = evaluate(yte, pred, prob)
    _fmt = lambda v: f"{v:.4f}" if v is not None else "n/a(multiclass)"
    print(f"  SELECTED C={C:g} (on dev {a.select_on}) -> TEST mcc={test['mcc']:.4f} "
          f"auprc={_fmt(test['auprc'])} auroc={_fmt(test['auroc'])}", flush=True)

    if a.predictions_dir:
        os.makedirs(a.predictions_dir, exist_ok=True)
        pp = (f"{a.predictions_dir}/hvue_v2__{a.task.replace('/','__')}__kmer{_cap}"
              f"__s0__preds.npz")
        np.savez_compressed(pp, y_true=yte, y_pred=pred, y_prob=prob)
        print(f"  wrote {pp}", flush=True)

    json.dump(dict(task=a.task, arm="kmer", ks=a.ks, class_weight="balanced",
                   selected_C=C, select_on=a.select_on, train_cap=a.train_cap or None,
                   n_train=len(tr), n_dev=len(dv), n_test=len(te),
                   train_pos_rate=round(float(tr.label.mean()), 4),
                   test_pos_rate=round(float(te.label.mean()), 4),
                   refit_on="train+dev", search=search, test=test),
              open(tag, "w"), indent=2)
    print(f"  wrote {tag}")


if __name__ == "__main__":
    main()
