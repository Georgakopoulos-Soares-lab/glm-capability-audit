"""Part 1: nearest-neighbour label-lookup exploitability probe, and the consequence-driven
cleaning threshold.

What this is FOR
----------------
easy-cluster and easy-search are homology DETECTORS: they say a relative exists. They do not
say the relationship is USABLE. This probe says how usable it is, with no model at all:

    for each test sequence, take its single nearest TRAINING neighbour by alignment and copy
    that neighbour's label.

If that lookup scores well, residual homology is *exploitable* -- any model can reach that
score by memorising neighbours, so a "model advantage" measured on such a split is not
evidence of capability. This is the novel piece. That clustering under-removes nucleotide
homology is already established (GraphPart, NAR Genomics 2023, which also shows DNA has no
principled identity threshold); that performance inflates with similarity is already shown by
PathoLM's threshold sweep. Neither demonstrates EXPLOITABILITY, and neither uses it to choose
a threshold.

Two modes
---------
1. **Probe (threshold-free).** Take the nearest neighbour regardless of similarity. No
   threshold is chosen, so nothing is assumed about what counts as "too similar". This is the
   headline number per dataset.
2. **Consequence-driven cleaning curve.** Sweep a cleaning threshold T over
   identity x coverage; at each T drop the test rows whose nearest training match exceeds T,
   and re-measure lookup accuracy on what remains. The right T is where the lookup falls to
   chance -- i.e. where exploitable homology is gone. Threshold set by consequence, not by
   convention (which per GraphPart does not exist for DNA).

Cost
----
MMseqs2 `easy-search` gives the nearest hit directly; all-vs-all alignment is never performed.
One search per dataset serves both modes: the probe uses the best hit per query, and the
sweep post-filters the same hit table. Reverse complements are covered -- verified that
`--search-type 3` returns a revcomp match at 100% identity / full coverage.

Chance level is computed from the test label distribution (majority-class MCC is 0 by
construction; AUROC chance is 0.5; accuracy chance is the majority rate, which is why accuracy
is NOT the criterion on these imbalanced tasks).
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score

MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
SCRATCH = os.environ.get("VB_SCRATCH", "/data/nvidia/tmp_mmseqs")
IDENTS = [0.70, 0.80, 0.90, 0.95]
COVFRACS = [0.30, 0.50, 0.70, 0.80]


def search_hits(train, test, threads=24, min_id=0.30, min_cov=0.10, max_seqs=20):
    """One easy-search; returns a hit table with query row index, target row index,
    identity and query-coverage. Nucleotide mode, so reverse complements are included."""
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        with open(f"{d}/tr.fa", "w") as f:
            for i, s in enumerate(train.sequence.astype(str)):
                f.write(f">{i}\n{s}\n")
        with open(f"{d}/te.fa", "w") as f:
            for i, s in enumerate(test.sequence.astype(str)):
                f.write(f">{i}\n{s}\n")
        cmd = [MMSEQS, "easy-search", f"{d}/te.fa", f"{d}/tr.fa", f"{d}/h.tsv", f"{d}/tmp",
               "--search-type", "3", "--min-seq-id", str(min_id), "-c", str(min_cov),
               "--cov-mode", "2", "-e", "1e-3", "--max-seqs", str(max_seqs),
               "--threads", str(threads), "-v", "1",
               "--format-output", "query,target,pident,alnlen,qlen,qcov,bits"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"easy-search failed: {r.stderr[-400:]}")
        if not os.path.getsize(f"{d}/h.tsv"):
            return pd.DataFrame(columns=["q", "t", "pident", "alnlen", "qlen", "qcov", "bits"])
        h = pd.read_csv(f"{d}/h.tsv", sep="\t", header=None,
                        names=["q", "t", "pident", "alnlen", "qlen", "qcov", "bits"])
        h["pident"] = h["pident"] / 100.0 if h["pident"].max() > 1.5 else h["pident"]
        h["qcov"] = h["qcov"] / 100.0 if h["qcov"].max() > 1.5 else h["qcov"]
        h["covfrac"] = np.maximum(h["qcov"], h["alnlen"] / h["qlen"].clip(lower=1))
        return h
    finally:
        shutil.rmtree(d, ignore_errors=True)


def lookup_scores(h, ytr, yte, keep_mask=None):
    """Nearest-neighbour (highest bitscore) label transfer. Returns metrics over the test rows
    that HAVE a neighbour, plus coverage. Rows with no hit are reported, never silently
    imputed -- imputing the majority class would flatter the probe."""
    if keep_mask is not None:
        h = h[keep_mask.reindex(h.index, fill_value=False)] if isinstance(keep_mask, pd.Series) else h
    if len(h) == 0:
        return None
    best = h.sort_values("bits", ascending=False).drop_duplicates("q")
    q = best.q.values.astype(int)
    pred = ytr[best.t.values.astype(int)]
    true = yte[q]
    if len(np.unique(true)) < 2:
        return None
    out = {
        "n_with_neighbour": int(len(q)),
        "frac_with_neighbour": round(float(len(q) / len(yte)), 4),
        "mcc": round(float(matthews_corrcoef(true, pred)), 4),
        "macro_f1": round(float(f1_score(true, pred, average="macro", zero_division=0)), 4),
    }
    try:
        out["auroc"] = round(float(roc_auc_score(true, pred)), 4)
    except ValueError:
        out["auroc"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="/home/nvidia/genomic-capability-unlearning/reports/nn_lookup")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--test-cap", type=int, default=0,
                    help="cap QUERY rows only; the target (train) side is never capped, "
                         "since capping it would bias homology downward")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # GUE ships the sequence column as `text`; HVUE/VHDB use `sequence`.
    _rd = lambda p: (pd.read_csv(p).rename(columns={"text": "sequence"})
                       .reset_index(drop=True))
    tr = _rd(a.train_csv)
    te = _rd(a.test_csv)
    n_full = len(te)
    if a.test_cap and len(te) > a.test_cap:
        te = te.sample(n=a.test_cap, random_state=a.seed).reset_index(drop=True)
    ytr = tr.label.values.astype(int); yte = te.label.values.astype(int)
    pos = float(yte.mean())
    print(f"\n=== {a.name}: train={len(tr):,} test={n_full:,}"
          f"{f' (query capped to {len(te):,})' if len(te)!=n_full else ''}  "
          f"test pos_rate={pos:.4f}", flush=True)

    # --- revcomp-exact duplicates: the string-equality check misses these ---
    rc = str.maketrans("ACGT", "TGCA")
    trs = set(tr.sequence.astype(str).str.upper())
    te_u = te.sequence.astype(str).str.upper()
    exact = int(te_u.isin(trs).sum())
    rcx = int(sum(1 for s in te_u if s.translate(rc)[::-1] in trs))
    print(f"  exact duplicates: {exact} ({100*exact/len(te):.3f}%)   "
          f"REVCOMP-exact duplicates: {rcx} ({100*rcx/len(te):.3f}%)", flush=True)

    h = search_hits(tr, te, threads=a.threads)
    print(f"  easy-search returned {len(h):,} hits for {h.q.nunique() if len(h) else 0:,} "
          f"of {len(te):,} test rows", flush=True)

    # --- mode 1: threshold-free probe ---
    base = lookup_scores(h, ytr, yte)
    print(f"\n  [THRESHOLD-FREE NN LOOKUP]  {base}", flush=True)

    # --- mode 2: consequence-driven cleaning curve ---
    curve = []
    for i in IDENTS:
        for c in COVFRACS:
            drop_q = set(h[(h.pident >= i) & (h.covfrac >= c)].q.astype(int))
            keep = np.array([k for k in range(len(te)) if k not in drop_q])
            if len(keep) < 50 or len(np.unique(yte[keep])) < 2:
                curve.append(dict(ident=i, cov=c, n_kept=int(len(keep)), scores=None,
                                  note="too few / single-class after cleaning")); continue
            hk = h[h.q.astype(int).isin(set(keep.tolist()))]
            sc = lookup_scores(hk, ytr, yte)
            curve.append(dict(ident=i, cov=c, n_kept=int(len(keep)),
                              frac_kept=round(len(keep)/len(te), 4), scores=sc))
            m = sc["mcc"] if sc else float("nan")
            print(f"    clean id>={i:.2f} cov>={c:.2f}: kept {len(keep):6,} "
                  f"({100*len(keep)/len(te):5.1f}%)  NN-lookup MCC={m:+.4f}", flush=True)

    res = dict(name=a.name, n_train=len(tr), n_test_full=n_full, n_test_queried=len(te),
               test_pos_rate=round(pos, 4),
               chance=dict(mcc=0.0, auroc=0.5, majority_accuracy=round(max(pos, 1-pos), 4)),
               exact_duplicates=exact, revcomp_exact_duplicates=rcx,
               threshold_free_lookup=base, cleaning_curve=curve,
               note="NN lookup = copy the label of the single best-bitscore training hit. "
                    "Reverse complements included (easy-search --search-type 3, verified).")
    p = f"{a.out}/{a.name}__nn_lookup.json"
    json.dump(res, open(p, "w"), indent=2)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
