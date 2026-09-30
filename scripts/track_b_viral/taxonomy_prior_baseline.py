"""Taxonomy-prior baseline: how much of the label is just the lineage?

If a virus has no detectable sequence match in training but its FAMILY is well
represented and nearly every member carries the same label, then a model can
score by doing

    sequence -> family -> label

That is real statistical generalisation, but it is much weaker than learning a
transferable host-tropism mechanism, and it is not distinguished from one by any
homology audit. This baseline makes the distinction measurable: predict the
majority training label of the test row's genus, backing off to family, then to
the global majority.

It is cheap and it can be brutal. If it matches the model, the model's "capability"
on that task is taxonomy recognition.
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tax", required=True, help="train-row-order csv: label + genus/family")
    ap.add_argument("--test-tax", required=True, help="test-row-order csv: label + genus/family")
    ap.add_argument("--esc", required=True); ap.add_argument("--name", default="")
    ap.add_argument("--levels", nargs="+", default=["genus", "family"],
                    help="back-off order, most specific first")
    ap.add_argument("--predictions-dir", default=os.environ.get(
        "VB_PREDS", os.environ.get("VB_VIRAL_OUT",
        "/data/nvidia/genomic-capability-unlearning/results_viral_bench") + "/preds_rescore"))
    a = ap.parse_args(); os.makedirs(a.predictions_dir, exist_ok=True)

    tr, te = pd.read_csv(a.train_tax), pd.read_csv(a.test_tax)
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    glob_major = int(pd.Series(ytr).mode()[0])

    maps, cov = {}, {}
    for lv in a.levels:
        if lv not in tr.columns or lv not in te.columns:
            print(f"  ! {lv} absent, skipped"); continue
        maps[lv] = (tr.assign(_y=ytr).groupby(lv)["_y"]
                    .agg(lambda s: int(s.mode()[0])).to_dict())
        cov[lv] = float(te[lv].astype(str).isin({str(k) for k in maps[lv]}).mean())

    pred = np.full(len(te), glob_major, int)
    src = np.array(["global"] * len(te), object)
    for lv in reversed(a.levels):            # apply general first, specific overrides
        if lv not in maps:
            continue
        v = te[lv].astype(str).map({str(k): x for k, x in maps[lv].items()})
        hit = v.notna().values
        pred[hit] = v[hit].astype(int).values
        src[hit] = lv

    print(f"{a.name or a.esc}: n_test={len(te):,}")
    for lv in a.levels:
        if lv in cov:
            print(f"  {lv:8s} known for {cov[lv]:.1%} of test rows "
                  f"({len(maps[lv]):,} distinct in train)")
    for lv in set(src):
        print(f"  resolved by {lv:8s}: {(src == lv).sum():,}")
    print(f"  TAXONOMY-PRIOR  MCC={matthews_corrcoef(yte, pred):.4f}")
    if len(np.unique(yte)) == 2:
        print(f"                  AUPRC={average_precision_score(yte, pred):.4f} "
              f"(hard labels, so this is a lower bound)")

    p = f"{a.predictions_dir}/hvue_v2__{a.esc}__taxprior__s0__preds.npz"
    np.savez_compressed(p, y_true=yte, y_pred=pred, y_prob=pred.astype(float))
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
