"""Make NN-lookup a first-class BASELINE ARM, comparable to k-mer/CNN/gLMs.

The probe in nn_lookup_probe.py scores only the test rows that HAVE a training neighbour
(68-99% depending on dataset). Comparing that number against a model scored on ALL rows is
not like-for-like and would overstate the lookup.

Two defensible full-test-set versions, both reported:
  (a) fallback=majority -- rows with no neighbour get the training-majority label. This is
      "NN-lookup as a baseline you could actually deploy", scored on every row, directly
      comparable to k-mer/CNN/gLM.
  (b) matched-subset    -- every arm re-scored on ONLY the rows that have a neighbour, so
      model and lookup answer the same questions.

(a) is the one that belongs in the floor: max(k-mer, CNN, NN-lookup). (b) is the diagnostic
that says whether a model adds anything where homology is available.
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nn_lookup_probe import search_hits
R = "/data/nvidia/genomic-capability-unlearning/results_viral_bench"
PRED, GLM = f"{R}/preds_rescore", f"{R}/geneb_finetune"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True); ap.add_argument("--test-csv", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--esc", required=True,
                    help="prediction-filename task string, for the model arms")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--predictions-dir",
                    default="/data/nvidia/genomic-capability-unlearning/results_viral_bench/preds_rescore")
    ap.add_argument("--out", default="/home/nvidia/genomic-capability-unlearning/reports/nn_baseline")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    # GUE ships the sequence column as `text`; HVUE/VHDB use `sequence`.
    _rd = lambda p: (pd.read_csv(p).rename(columns={"text": "sequence"})
                       .reset_index(drop=True))
    tr = _rd(a.train_csv)
    te = _rd(a.test_csv)
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    maj = int(np.bincount(ytr).argmax())

    h = search_hits(tr, te, threads=a.threads)
    best = h.sort_values("bits", ascending=False).drop_duplicates("q")
    hit_rows = best.q.values.astype(int)
    nn_full = np.full(len(te), maj, dtype=int)          # (a) fallback = training majority
    nn_full[hit_rows] = ytr[best.t.values.astype(int)]
    has_nn = np.zeros(len(te), dtype=bool); has_nn[hit_rows] = True

    print(f"\n=== {a.name}: n_test={len(te):,}  with neighbour {has_nn.sum():,} "
          f"({100*has_nn.mean():.1f}%)  majority-fallback label={maj}")
    res = {"name": a.name, "n_test": int(len(te)), "frac_with_nn": round(float(has_nn.mean()), 4)}
    res["nn_lookup_fullset_majfallback"] = round(float(matthews_corrcoef(yte, nn_full)), 4)
    res["nn_lookup_hitrows_only"] = round(float(matthews_corrcoef(yte[has_nn], nn_full[has_nn])), 4)
    print(f"  NN-lookup, FULL test set (majority fallback): MCC={res['nn_lookup_fullset_majfallback']:+.4f}")
    print(f"  NN-lookup, hit rows only                    : MCC={res['nn_lookup_hitrows_only']:+.4f}")

    def arm(pat):
        ps = sorted(glob.glob(pat))
        if not ps: return None
        out = []
        for p in ps:
            z = np.load(p)
            if len(z["y_true"]) != len(te): return None
            out.append(z["y_pred"])
        return out

    arms = {}
    for tag, pat in (("kmer", f"{PRED}/hvue_v2__{a.esc}__kmer__s*__preds.npz"),
                     ("kmer_cap", f"{PRED}/hvue_v2__{a.esc}__kmer__cap*__s*__preds.npz"),
                     ("cnn", f"{PRED}/*{a.esc}*cnn_ceiling__s*__preds.npz")):
        v = arm(pat)
        if v: arms[tag] = v
    for m in ("hvilm", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna", "lucavirus"):
        c = []
        for j in sorted(glob.glob(f"{GLM}/{a.esc}__{m}__lr*__fullft.json")):
            if any(x in j for x in ("__probs", "_meanpool", "_ep4")): continue
            d = json.load(open(j))
            if len(d.get("runs", [])) >= 3:
                c.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"]))
        if not c: continue
        v = arm(f"{GLM}/{a.esc}__{m}__lr{max(c)[1]:g}__s*__preds.npz")
        if v: arms[m] = v

    print(f"  {'arm':12s} {'FULL set':>10s} {'hit rows':>10s}")
    print(f"  {'NN-lookup':12s} {res['nn_lookup_fullset_majfallback']:10.4f} "
          f"{res['nn_lookup_hitrows_only']:10.4f}")
    for k, v in arms.items():
        full = float(np.mean([matthews_corrcoef(yte, p) for p in v]))
        sub = float(np.mean([matthews_corrcoef(yte[has_nn], p[has_nn]) for p in v]))
        res[f"{k}_fullset"] = round(full, 4); res[f"{k}_hitrows"] = round(sub, 4)
        print(f"  {k:12s} {full:10.4f} {sub:10.4f}")

    # Save the full-test-set NN predictions so the threshold-trend analysis can include
    # NN-lookup in the floor AT EVERY CELL, re-scored on the surviving rows like any other arm.
    if a.predictions_dir:
        os.makedirs(a.predictions_dir, exist_ok=True)
        pp = f"{a.predictions_dir}/hvue_v2__{a.esc}__nnlookup__s0__preds.npz"
        np.savez_compressed(pp, y_true=yte, y_pred=nn_full)
        print(f"  wrote {pp}")
    p = f"{a.out}/{a.name}__nn_baseline.json"
    json.dump(res, open(p, "w"), indent=2); print(f"  wrote {p}")


if __name__ == "__main__":
    main()
