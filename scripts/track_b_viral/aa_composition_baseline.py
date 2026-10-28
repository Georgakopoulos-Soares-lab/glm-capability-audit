"""Amino-acid composition baseline: is the uplift just crude protein composition?

The nucleotide k-mer GBT answers "is this nucleotide composition?". It cannot
answer "is this PROTEIN composition?", and for viral tasks those come apart:
synonymous divergence destroys nucleotide k-mer similarity while leaving amino
acid usage nearly intact. A model that looks like it has learned protein-level
biology may only be reading amino-acid frequency.

Six-frame translation, no ORF calling. Prodigal is not installed here, and the
annotation-free choice is the conservative one: it cannot be wrong about where
the genes are, and a real coding signal survives the extra frames as noise.

Sits between the nucleotide composition floor and the homology floors in the
hierarchy:  nt composition < AA composition < nt homology < translated homology < FM
"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score

CODON = {}
_B = "TCAG"
_AA = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
for i, (a, b, c) in enumerate(itertools.product(_B, _B, _B)):
    CODON[a + b + c] = _AA[i]
AAS = "ACDEFGHIKLMNPQRSTVWY"


def revcomp(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def six_frame(seq):
    seq = seq.upper()
    out = []
    for s in (seq, revcomp(seq)):
        for off in (0, 1, 2):
            cs = [s[i:i + 3] for i in range(off, len(s) - 2, 3)]
            out.append("".join(CODON.get(c, "X") for c in cs))
    return out


def featurise(seqs, ks=(1, 2), cap=100000):
    """AA k-mer frequencies pooled over all six frames, stop codons dropped."""
    idx = {}
    for k in ks:
        for t in itertools.product(AAS, repeat=k):
            idx["".join(t)] = len(idx)
    X = np.zeros((len(seqs), len(idx)), np.float32)
    for r, s in enumerate(seqs):
        for prot in six_frame(s[:cap]):
            for seg in prot.replace("*", "X").split("X"):
                for k in ks:
                    for i in range(len(seg) - k + 1):
                        j = idx.get(seg[i:i + k])
                        if j is not None:
                            X[r, j] += 1
        t = X[r].sum()
        if t:
            X[r] /= t
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True); ap.add_argument("--test-csv", required=True)
    ap.add_argument("--esc", required=True); ap.add_argument("--name", default="")
    ap.add_argument("--train-cap", type=int, default=40000)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2])
    # Seed for the estimator only; the train subsample stays at random_state=0 (see
    # kmer_gbt_baseline.py for the rationale).
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predictions-dir", default=os.environ.get(
        "VB_PREDS", os.environ.get("VB_VIRAL_OUT",
        "/data/nvidia/genomic-capability-unlearning/results_viral_bench") + "/preds_rescore"))
    a = ap.parse_args(); os.makedirs(a.predictions_dir, exist_ok=True)

    rd = lambda p: pd.read_csv(p).rename(columns={"text": "sequence"}).reset_index(drop=True)
    tr, te = rd(a.train_csv), rd(a.test_csv)
    if a.train_cap and len(tr) > a.train_cap:
        # See geneb_finetune.py: pandas 3.0 drops the grouping column from apply().
        tr = tr.groupby("label", group_keys=False)[list(tr.columns)].apply(
            lambda g: g.sample(n=max(1, int(round(a.train_cap * len(g) / len(tr)))),
                               random_state=0)).reset_index(drop=True)
    print(f"{a.name or a.esc}: train={len(tr):,} test={len(te):,} ks={a.ks}", flush=True)

    Xtr, Xte = featurise(tr.sequence.astype(str), a.ks), featurise(te.sequence.astype(str), a.ks)
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    print(f"  featurised: {Xtr.shape[1]} AA k-mer features", flush=True)

    try:
        from xgboost import XGBClassifier
        ncls = len(np.unique(ytr))
        m = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=16,
                          tree_method="hist", eval_metric="mlogloss", random_state=a.seed,
                          **({"objective": "multi:softprob", "num_class": ncls}
                             if ncls > 2 else {}))
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        m = H(max_iter=400, learning_rate=0.08, random_state=a.seed)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    prob = m.predict_proba(Xte)
    mcc = matthews_corrcoef(yte, pred)
    ap_ = (average_precision_score(yte, prob[:, 1]) if prob.shape[1] == 2 else float("nan"))
    print(f"  AA-composition GBT  MCC={mcc:.4f}  AUPRC={ap_:.4f}")

    p = f"{a.predictions_dir}/hvue_v2__{a.esc}__aagbt__s{a.seed}__preds.npz"
    np.savez_compressed(p, y_true=yte, y_pred=pred,
                        y_prob=(prob[:, 1] if prob.shape[1] == 2 else prob.max(1)))
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
