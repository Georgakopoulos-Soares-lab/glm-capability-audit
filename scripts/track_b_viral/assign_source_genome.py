"""Recover a source-genome label for each HVUE test chunk, for use as the bootstrap unit.

Why this is needed
------------------
The bootstrap resampling unit must be the unit of statistical dependence. For VHDB that is
shipped (`test_full.csv` has taxid/family/cluster). HVUE v2 ships only `sequence,label`.

Homology clustering does NOT recover it. HVUE chunks each genome into NON-OVERLAPPING windows,
so two chunks of one genome are different parts of that genome and are not homologous to each
other -- mmseqs easy-cluster puts HVUE Host_Tropism test into 40,369 clusters for 51,187 rows
(1.3 rows/cluster), which looks like near-independence and is not. Those chunks still share one
genome and one label, so a row-level or homology-cluster-level bootstrap overstates precision.

What this does instead
----------------------
easy-search every test chunk against a viral genome collection (ViroBench, which carries
taxids) and assign each chunk to its best-hit genome. Chunks landing on the same genome are one
resampling unit. Chunks with no hit keep a singleton unit of their own -- that is conservative
in the right direction (it can only make the interval narrower than truth, never wider), and
the unassigned fraction is reported so the reader can judge.
"""
import argparse, os, subprocess, sys, tempfile, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
SCRATCH = os.environ.get("VB_SCRATCH", "/data/nvidia/tmp_mmseqs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-id", type=float, default=0.80)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    import virobench_baselines as VBB

    te = pd.read_csv(a.test_csv)
    vb = pd.concat([VBB.load("ALL", "times", s) for s in ("train", "val", "test")],
                   ignore_index=True).drop_duplicates("taxid").reset_index(drop=True)
    print(f"queries: {len(te):,} test chunks | targets: {len(vb):,} ViroBench genomes",
          flush=True)

    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        with open(f"{d}/q.fa", "w") as f:
            for i, s in enumerate(te.sequence.astype(str)):
                f.write(f">{i}\n{s}\n")
        with open(f"{d}/t.fa", "w") as f:
            for i, r in vb.iterrows():
                f.write(f">{r.taxid}\n{str(r.sequence)}\n")
        cmd = [MMSEQS, "easy-search", f"{d}/q.fa", f"{d}/t.fa", f"{d}/hits.tsv", f"{d}/tmp",
               # cov-mode 2 = coverage of the QUERY. The query is a 1000bp chunk and the
               # target is a whole 40kb+ genome, so cov-mode 1 (coverage of target) demands
               # the chunk cover half the genome -- impossible, and it matched 0.3% of rows.
               "--min-seq-id", str(a.min_id), "-c", "0.5", "--cov-mode", "2", "-e", "1e-5",
               "--search-type", "3", "--max-seqs", "5", "--threads", str(a.threads), "-v", "1",
               "--format-output", "query,target,pident,bits"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"easy-search failed: {r.stderr[-500:]}")
        if not os.path.getsize(f"{d}/hits.tsv"):
            raise SystemExit("no hits at all -- check --min-id")
        h = pd.read_csv(f"{d}/hits.tsv", sep="\t", header=None,
                        names=["q", "t", "pident", "bits"])
        best = h.sort_values("bits", ascending=False).drop_duplicates("q")
        unit = np.array([f"row{i}" for i in range(len(te))], dtype=object)
        unit[best.q.values.astype(int)] = ["tax" + str(t) for t in best.t.values]
    finally:
        shutil.rmtree(d, ignore_errors=True)

    n_assigned = int((pd.Series(unit).str.startswith("tax")).sum())
    ngen = pd.Series(unit[pd.Series(unit).str.startswith("tax").values]).nunique()
    nunits = pd.Series(unit).nunique()
    print(f"  assigned {n_assigned:,}/{len(te):,} ({100*n_assigned/len(te):.1f}%) to "
          f"{ngen:,} genomes; {len(te)-n_assigned:,} singletons")
    print(f"  bootstrap units = {nunits:,}  ({len(te)/nunits:.1f} rows/unit)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame({"cluster": unit}).to_csv(a.out, index=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
