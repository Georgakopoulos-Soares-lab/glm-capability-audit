"""Cluster-level bootstrap CI on the excess = best gLM - max(k-mer, CNN).

Why the resampling unit is the cluster, not the row
---------------------------------------------------
Test rows are 1000bp chunks of genomes. VHDB's 3,754 test rows come from 349 homology
clusters -- ~10.8 chunks per cluster, all sharing one label and near-identical sequence. A
row-level bootstrap treats those as 3,754 independent draws and reports an interval ~sqrt(10.8)
times too narrow. Resampling whole clusters is the honest unit.

Excess is recomputed inside every replicate: the baseline is re-evaluated as
max(k-mer, CNN) on the SAME resampled rows, so the comparison stays paired and the "max"
is not frozen at its full-test-set winner.

Seed variance is folded in by drawing one seed per arm per replicate, so the interval carries
both sampling noise and training noise rather than only the first.
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile, shutil
from collections import defaultdict
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
SCRATCH = os.environ.get("VB_SCRATCH", "/data/nvidia/tmp_mmseqs")
PRED = "/data/nvidia/genomic-capability-unlearning/results_viral_bench/preds_rescore"
GLM  = "/data/nvidia/genomic-capability-unlearning/results_viral_bench/geneb_finetune"


def cluster_test(seqs, ident, cov=0.8, threads=24):
    """mmseqs easy-cluster over the test sequences; returns an array of cluster ids."""
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        with open(f"{d}/t.fa", "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">{i}\n{s}\n")
        r = subprocess.run([MMSEQS, "easy-cluster", f"{d}/t.fa", f"{d}/c", f"{d}/tmp",
                            "--min-seq-id", str(ident), "-c", str(cov),
                            "--threads", str(threads), "-v", "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"easy-cluster failed: {r.stderr.strip().splitlines()[-1]}")
        rep = {}
        for line in open(f"{d}/c_cluster.tsv"):
            a, b = line.rstrip("\n").split("\t")
            rep[int(b)] = int(a)
        return np.array([rep.get(i, i) for i in range(len(seqs))])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def load_arm(pattern):
    """-> (y_true, [per-seed y_pred])"""
    ps = sorted(glob.glob(pattern))
    if not ps:
        return None, []
    yt, preds = None, []
    for p in ps:
        z = np.load(p)
        if yt is None:
            yt = z["y_true"]
        elif not np.array_equal(yt, z["y_true"]):
            raise SystemExit(f"{p}: y_true differs from a sibling -- predictions misaligned")
        preds.append(z["y_pred"])
    return yt, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--esc", required=True, help="prediction-filename task string")
    ap.add_argument("--cluster-csv", default="", help="csv with a `cluster` column, test order")
    ap.add_argument("--cluster-fasta-from", default="", help="csv with `sequence`, cluster here")
    ap.add_argument("--cluster-id", type=float, default=0.70)
    ap.add_argument("--models", nargs="+",
                    default=["hvilm", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna", "lucavirus"])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    yk, kmer = load_arm(f"{PRED}/hvue_v2__{a.esc}__kmer__s*__preds.npz")
    yc, cnn  = load_arm(f"{PRED}/*{a.esc}*cnn_ceiling__s*__preds.npz")
    if not kmer or not cnn:
        raise SystemExit(f"missing baseline predictions (kmer={len(kmer)} cnn={len(cnn)})")
    y = yk
    assert np.array_equal(yk, yc), "k-mer and CNN predictions are not in the same row order"
    n = len(y)

    models = {}
    for m in a.models:
        cands = []
        for j in sorted(glob.glob(f"{GLM}/{a.esc}__{m}__lr*__fullft.json")):
            d = json.load(open(j))
            if len(d.get("runs", [])) >= 3:
                cands.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"]))
        if not cands:
            print(f"  ! {m}: no 3-seed record, skipped"); continue
        dev, lr = max(cands)
        ym, pr = load_arm(f"{GLM}/{a.esc}__{m}__lr{lr:g}__s*__preds.npz")
        if not pr:
            print(f"  ! {m}: no prediction files, skipped"); continue
        if not np.array_equal(ym, y):
            print(f"  ! {m}: y_true mismatch vs baselines, skipped"); continue
        models[m] = pr
        print(f"  {m}: lr={lr:g} dev={dev:.4f}, {len(pr)} seeds")

    # --- cluster labels ---
    if a.cluster_csv:
        cl = pd.read_csv(a.cluster_csv)["cluster"].values
        src = f"column in {os.path.basename(a.cluster_csv)}"
    else:
        seqs = pd.read_csv(a.cluster_fasta_from)["sequence"].astype(str).tolist()
        cl = cluster_test(seqs, a.cluster_id)
        src = f"mmseqs easy-cluster at {a.cluster_id:.0%}"
    assert len(cl) == n, f"cluster labels {len(cl)} != n_test {n}"
    uniq = pd.unique(cl)
    idx_by_cluster = {c: np.where(cl == c)[0] for c in uniq}
    print(f"\n{a.task}: n_test={n:,}  clusters={len(uniq):,} ({src})  "
          f"mean {n/len(uniq):.1f} rows/cluster  pos_rate={y.mean():.4f}")

    def excess(rows, pick):
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            return None
        base = max(matthews_corrcoef(yy, kmer[pick["kmer"]][rows]),
                   matthews_corrcoef(yy, cnn[pick["cnn"]][rows]))
        return {m: matthews_corrcoef(yy, models[m][pick[m]][rows]) - base for m in models}, base

    allrows = np.arange(n)
    # The POINT ESTIMATE must use the same quantity the bootstrap resamples: the mean over
    # seeds, not seed index 0. Using seed 42's CNN alone gave a baseline of 0.5640 against a
    # bootstrap mean of 0.5386 -- the reported excess and its CI were then describing
    # different baselines.
    def mean_mcc(preds):
        return float(np.mean([matthews_corrcoef(y, p) for p in preds]))
    base0 = max(mean_mcc(kmer), mean_mcc(cnn))
    obs = {m: mean_mcc(models[m]) - base0 for m in models}
    print(f"  baseline arms: k-mer {mean_mcc(kmer):.4f} ({len(kmer)} seeds) | "
          f"CNN {mean_mcc(cnn):.4f} ({len(cnn)} seeds) -> floor {base0:.4f}")

    draws = defaultdict(list); bases = []
    skipped = 0
    for _ in range(a.reps):
        cs = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_cluster[c] for c in cs])
        pick = {m: rng.integers(len(models[m])) for m in models}
        pick["kmer"] = rng.integers(len(kmer)); pick["cnn"] = rng.integers(len(cnn))
        r = excess(rows, pick)
        if r is None:
            skipped += 1; continue
        e, b = r
        for m, v in e.items():
            draws[m].append(v)
        bases.append(b)

    print(f"\n  full-test max(k-mer, CNN) = {base0:.4f}   "
          f"bootstrap mean baseline = {np.mean(bases):.4f}")
    print(f"  {a.reps:,} cluster replicates ({skipped} skipped for single-class draws)\n")
    print(f"  {'model':12s} {'excess':>8s} {'95% CI':>22s} {'P(<=0)':>8s}")
    res = {}
    for m in sorted(draws, key=lambda k: -obs[k]):
        d = np.array(draws[m]); lo, hi = np.percentile(d, [2.5, 97.5])
        p = float((d <= 0).mean())
        res[m] = dict(excess=round(float(obs[m]), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                      p_le_0=round(p, 4), boot_mean=round(float(d.mean()), 4))
        star = "  SURVIVES" if lo > 0 else ("  CI spans 0" if hi > 0 else "  NEGATIVE")
        print(f"  {m:12s} {obs[m]:+8.4f} [{lo:+8.4f}, {hi:+8.4f}] {p:8.4f}{star}")

    out = dict(task=a.task, n_test=int(n), n_clusters=int(len(uniq)), cluster_source=src,
               rows_per_cluster=round(n / len(uniq), 2), reps=a.reps,
               baseline_full_test=round(float(base0), 4), unit="homology cluster",
               skipped_replicates=skipped, results=res)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
