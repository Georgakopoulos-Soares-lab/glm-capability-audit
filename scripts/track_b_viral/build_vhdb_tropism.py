"""Build an INDEPENDENT host-tropism task: Virus-Host DB labels x ViroBench genomes.

Why this exists
---------------
HViLM shows a large host-tropism advantage on HVUE v2 (+0.184 / +0.239 MCC over a
class-weighted, fully-searched CNN, ratio 2.33 / 2.45) while showing essentially nothing on
Pathogenicity and Transmissibility (+0.0009 to +0.0095). That is the authors' own stated
pattern, so it replicates -- but HVUE v2 was built by the HViLM authors and HViLM was
continued-pretrained on viral genomes, so the advantage could be exposure to these specific
genomes rather than a transferable capability. This task was BUILT to be independent on both
axes -- but see the correction at the end of this docstring: it is independent on labels,
split and design, NOT on sequence.

  labels   <- Virus-Host DB (genome.jp), an external curated virus->host mapping
  genomes  <- ViroBench (RefSeq/GenBank viral), an external genome collection
  split    <- our own cluster-aware pipeline
  audit    <- our own easy-search grid

Design decision that matters
----------------------------
Raw "human vs non-human" is NOT the right task. In VHDB the non-human side is dominated by
bacteriophage families (Straboviridae, Herelleviridae, Drexlerviridae, ...), so the task
degenerates into "eukaryotic virus vs phage" -- solvable from composition alone, and not the
dual-use question. We therefore keep only families containing **both** classes (>=3 each),
which forces WITHIN-FAMILY discrimination: "does this virus, from a family that contains
human pathogens, infect humans?" That is the hazard-relevant question and a harder task than
HVUE's.

Split construction mirrors HVUE v2 so the comparison is like-for-like:
  1. cluster GENOMES at --cluster-id identity (mmseqs easy-cluster)
  2. assign whole clusters to train/dev/test 70/15/15, stratified on label
  3. chunk to --chunk bp AFTER splitting (never before -- that was the v1 defect)
  4. drop exact duplicate chunks within and across splits (SHA-256)
Residual leakage is then MEASURED with easy-search, never assumed.

CORRECTION 2026-09-19 -- NOT SEQUENCE-INDEPENDENT OF HVUE v2
------------------------------------------------------------
Measured after the fact with `cross_dataset_homology.py`: **50.3% of this task's TEST rows have
a >=95% identity / >=80% coverage match inside HVUE v2 Host_Tropism TRAIN** (76.4% at
id>=0.7/cov>=0.3). HVUE v2 draws from NCBI Virus, BV-BRC and Virus-Host DB; this task draws
labels from Virus-Host DB and genomes from ViroBench, so the two overlap heavily by provenance.

The task is still internally sound (cluster-aware split, 13.5% internal leakage at
id>=0.9/cov>=0.5) and results on it are valid. What it is NOT is an INDEPENDENT replication of
an HVUE finding. Do not describe it as one.
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, shutil
from collections import defaultdict
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
SCRATCH = os.environ.get("VB_SCRATCH", "/data/nvidia/tmp_mmseqs")
VB = ("/data/nvidia/hf_cache/hub/datasets--YDXX--ViroBench/snapshots/"
      "77461edda1bc4f3c0ea96008384a678fefe04a14/Classification/ALL/taxon/times")
VHDB = "/data/nvidia/virushostdb/virushostdb.tsv"


def load_labels(min_per_class):
    vh = pd.read_csv(VHDB, sep="\t", dtype=str, on_bad_lines="skip")
    vh.columns = [c.strip() for c in vh.columns]
    vh["virus tax id"] = pd.to_numeric(vh["virus tax id"], errors="coerce")
    vh = vh.dropna(subset=["virus tax id"])
    vh["virus tax id"] = vh["virus tax id"].astype(int)
    human = set(vh[vh["host tax id"].astype(str).str.strip() == "9606"]["virus tax id"])
    known = set(vh["virus tax id"])

    meta = pd.concat([pd.read_csv(f"{VB}/{s}.csv") for s in ("train", "val", "test")],
                     ignore_index=True).drop_duplicates("taxid")
    meta = meta[meta.taxid.isin(known)].copy()
    meta["label"] = meta.taxid.isin(human).astype(int)
    # phage families appear under host=human only because they were isolated from human
    # samples; Anelloviridae is commensal. Both are excluded per the Virus-Host DB notes.
    meta = meta[~meta.family.astype(str).str.contains(
        "Microviridae|Inoviridae|Anelloviridae", na=False)]
    # keep only families with both classes represented -> within-family discrimination
    t = meta.groupby("family").label.agg(["sum", "count"])
    t["neg"] = t["count"] - t["sum"]
    keep_fams = t[(t["sum"] >= min_per_class) & (t["neg"] >= min_per_class)].index
    meta = meta[meta.family.isin(keep_fams)].reset_index(drop=True)
    return meta, sorted(keep_fams)


def load_sequences(taxids):
    want = set(taxids)
    seqs = {}
    for split in ("train", "val", "test"):
        p = f"{VB}/{split}_sequences.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            if d["taxid"] in want:
                ss = [s for s in d.get("sequences", []) if s]
                if ss:
                    seqs[d["taxid"]] = max(ss, key=len).upper()
    return seqs


def cluster_genomes(seqs, cluster_id, cov, threads):
    """Cluster whole genomes; returns {taxid: cluster_rep}. Clusters are the split unit."""
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        ids = list(seqs)
        with open(f"{d}/g.fa", "w") as f:
            for i, t in enumerate(ids):
                f.write(f">{i}\n{seqs[t]}\n")
        r = subprocess.run([MMSEQS, "easy-cluster", f"{d}/g.fa", f"{d}/clu", f"{d}/tmp",
                            "--min-seq-id", str(cluster_id), "-c", str(cov),
                            "--threads", str(threads), "-v", "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"genome clustering failed: {r.stderr.strip().splitlines()[-1]}")
        rep = {}
        for line in open(f"{d}/clu_cluster.tsv"):
            a, b = line.rstrip("\n").split("\t")
            rep[ids[int(b)]] = ids[int(a)]
        return rep
    finally:
        shutil.rmtree(d, ignore_errors=True)


def assign_clusters(meta, rep, seed):
    """Whole clusters to train/dev/test 70/15/15, label-stratified by cluster majority."""
    lab = dict(zip(meta.taxid, meta.label))
    members = defaultdict(list)
    for t, r in rep.items():
        members[r].append(t)
    rows = [(r, int(round(np.mean([lab[t] for t in ms]))), len(ms)) for r, ms in members.items()]
    rng = np.random.default_rng(seed)
    part = {}
    for cls in (0, 1):
        cl = [r for r, c, _ in rows if c == cls]
        rng.shuffle(cl)
        n = len(cl); n_tr = int(0.70 * n); n_dv = int(0.15 * n)
        for i, r in enumerate(cl):
            part[r] = "train" if i < n_tr else ("dev" if i < n_tr + n_dv else "test")
    return {t: part[rep[t]] for t in rep}, members


def chunk(seq, size, stride=None):
    stride = stride or size
    return [seq[i:i + size] for i in range(0, max(1, len(seq) - size + 1), stride)
            if len(seq[i:i + size]) == size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/nvidia/vhdb_tropism/within_family_1000bp")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--cluster-id", type=float, default=0.95,
                    help="genome clustering identity; 0.95 matches HVUE v2 'standard'")
    ap.add_argument("--cluster-cov", type=float, default=0.8)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--max-chunks-per-genome", type=int, default=30,
                    help="cap so a few long genomes cannot dominate; recorded in meta.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)

    meta, fams = load_labels(a.min_per_class)
    print(f"labels: {len(meta)} taxa across {len(fams)} families, "
          f"human={int(meta.label.sum())} non-human={int((1-meta.label).sum())} "
          f"({100*meta.label.mean():.1f}% positive)", flush=True)
    seqs = load_sequences(meta.taxid.tolist())
    meta = meta[meta.taxid.isin(seqs)].reset_index(drop=True)
    print(f"with sequence: {len(meta)} taxa  human={int(meta.label.sum())}", flush=True)

    rep = cluster_genomes({t: seqs[t] for t in meta.taxid}, a.cluster_id, a.cluster_cov, a.threads)
    print(f"genome clusters at {a.cluster_id:.0%} identity: {len(set(rep.values()))} "
          f"from {len(rep)} genomes", flush=True)
    part, members = assign_clusters(meta, rep, a.seed)

    lab = dict(zip(meta.taxid, meta.label)); fam = dict(zip(meta.taxid, meta.family))
    rows = defaultdict(list)
    rng = np.random.default_rng(a.seed)
    for t in meta.taxid:
        cs = chunk(seqs[t], a.chunk)
        if not cs:
            continue
        if len(cs) > a.max_chunks_per_genome:
            cs = [cs[i] for i in rng.choice(len(cs), a.max_chunks_per_genome, replace=False)]
        for c in cs:
            rows[part[t]].append((c, lab[t], t, fam[t], rep[t]))

    # SHA-256 exact-duplicate removal, within and across splits (train wins ties)
    seen = {}
    kept = defaultdict(list)
    for sp in ("train", "dev", "test"):
        for c, l, t, f, r in rows[sp]:
            h = hashlib.sha256(c.encode()).hexdigest()
            if h in seen:
                continue
            seen[h] = sp
            kept[sp].append((c, l, t, f, r))

    summary = {}
    for sp in ("train", "dev", "test"):
        df = pd.DataFrame(kept[sp], columns=["sequence", "label", "taxid", "family", "cluster"])
        df[["sequence", "label"]].to_csv(f"{a.out}/{sp}.csv", index=False)
        df.to_csv(f"{a.out}/{sp}_full.csv", index=False)   # taxid/family/cluster for cluster bootstrap
        summary[sp] = dict(n=len(df), pos=int(df.label.sum()),
                           pos_rate=round(float(df.label.mean()), 4),
                           n_taxa=int(df.taxid.nunique()), n_clusters=int(df.cluster.nunique()),
                           n_families=int(df.family.nunique()))
        print(f"  {sp}: {len(df):6d} chunks  {int(df.label.sum()):5d} pos "
              f"({100*df.label.mean():.1f}%)  taxa={df.taxid.nunique()} "
              f"clusters={df.cluster.nunique()} families={df.family.nunique()}", flush=True)

    dup_dropped = sum(len(rows[s]) for s in rows) - sum(len(kept[s]) for s in kept)
    json.dump(dict(source_labels="Virus-Host DB (genome.jp) virushostdb.tsv",
                   source_genomes="ViroBench ALL/taxon/times, rev 77461edd",
                   task="host tropism: does this virus infect humans (host tax id 9606)",
                   design="within-family only: families with >=%d of each class" % a.min_per_class,
                   families=fams, chunk_bp=a.chunk, cluster_id=a.cluster_id,
                   cluster_cov=a.cluster_cov, split="cluster-level 70/15/15, label-stratified",
                   order="cluster -> split -> chunk (matches HVUE v2; v1 chunked first)",
                   max_chunks_per_genome=a.max_chunks_per_genome, seed=a.seed,
                   exact_duplicate_chunks_dropped=dup_dropped, splits=summary),
              open(f"{a.out}/meta.json", "w"), indent=2)
    print(f"\nexact duplicate chunks dropped: {dup_dropped}")
    print(f"wrote {a.out}/{{train,dev,test}}.csv + meta.json")


if __name__ == "__main__":
    main()
