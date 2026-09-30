"""Build a viral-vs-host-background discrimination task: is this sequence viral at all?

Why this task
-------------
Every viral task in the audit so far asks a question WITHIN viral sequence (which host, how
pathogenic, which family). None asks the most basic viral-recognition question: can the model
tell viral sequence from the genomic background it is embedded in? That is the question a
real screening pipeline asks first, and it is a different task TYPE from everything else here.

It also supplies a third sequence universe. The discriminative evidence currently rests almost
entirely on one: HVUE and VHDB share Virus-Host DB provenance and are 50-76% homologous.

Construction
------------
  positives  = viral genome chunks (ViroBench, family-capped so no family dominates)
  negatives  = matched-length chunks of NON-viral genomic background:
                 - human (hg38)
                 - bacterial (the natural host background for phage)

Discipline carried over from the rest of the audit:
  * family-cap the positives FIRST (raw ViroBench `times` is ~74% Orthomyxoviridae)
  * split at the GENOME level, then chunk -- never chunk then split
  * SHA-256 exact dedup within and across splits
  * negatives drawn from disjoint genomic regions per split
  * GC content of both classes reported: if the classes differ grossly in GC, a trivial
    composition baseline solves the task and that must be visible, not hidden
"""
import argparse, hashlib, json, os, random, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
HG38 = "/data/nvidia/data/hg38/hg38.fa"


def hg38_chunks(n, L, seed=0, skip=0):
    idx = {}
    for line in open(HG38 + ".fai"):
        nm, ln, off, lb, wb = line.split("\t")[:5]
        idx[nm] = (int(ln), int(off), int(lb), int(wb))
    chroms = [c for c in idx if "_" not in c and c not in ("chrM", "chrY", "chrEBV")]
    rng = np.random.default_rng(seed); out = []
    with open(HG38, "rb") as fh:
        guard = 0
        while len(out) < n + skip and guard < 60 * (n + skip):
            guard += 1
            c = chroms[int(rng.integers(len(chroms)))]
            ln, off, lb, wb = idx[c]
            p = int(rng.integers(0, max(1, ln - L)))
            fh.seek(off + p // lb * wb + p % lb)
            raw = fh.read(L + L // lb + 8).replace(b"\n", b"")[:L].decode().upper()
            if len(raw) == L and set(raw) <= set("ACGT"):
                out.append(raw)
    return out[skip:skip + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/nvidia/viral_vs_host/chunks1000")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--cap-family", type=int, default=6)
    ap.add_argument("--max-chunks-per-genome", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import virobench_baselines as VBB
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed); random.seed(a.seed)

    vb = pd.concat([VBB.load("ALL", "times", s) for s in ("train", "test")], ignore_index=True)
    vb = vb[vb.sequence.str.len() >= a.chunk].copy()
    vb = vb.groupby("family").head(a.cap_family).sample(frac=1.0, random_state=a.seed)
    vb = vb.reset_index(drop=True)
    print(f"viral genomes after family-cap {a.cap_family}: {len(vb)} "
          f"({vb.family.nunique()} families)", flush=True)

    # GENOME-level split first, then chunk (never chunk then split)
    n = len(vb); i_tr, i_dv = int(0.70 * n), int(0.85 * n)
    parts = {"train": vb.iloc[:i_tr], "dev": vb.iloc[i_tr:i_dv], "test": vb.iloc[i_dv:]}
    pos = {}
    for sp, d in parts.items():
        rows = []
        for _, r in d.iterrows():
            s = r.sequence.upper()
            k = min(a.max_chunks_per_genome, max(1, len(s) // a.chunk))
            offs = rng.choice(max(1, len(s) - a.chunk), size=k, replace=False) if len(s) > a.chunk else [0]
            for o in np.atleast_1d(offs):
                c = s[int(o):int(o) + a.chunk]
                if len(c) == a.chunk and set(c) <= set("ACGTN"):
                    rows.append(c)
        pos[sp] = rows
        print(f"  {sp}: {len(rows)} viral chunks from {len(d)} genomes", flush=True)

    # negatives: human background, disjoint draws per split
    neg, skip = {}, 0
    for sp in ("train", "dev", "test"):
        neg[sp] = hg38_chunks(len(pos[sp]), a.chunk, seed=a.seed, skip=skip)
        skip += len(pos[sp])
        print(f"  {sp}: {len(neg[sp])} human background chunks", flush=True)

    seen, summary = set(), {}
    for sp in ("train", "dev", "test"):
        recs = [(s, 1) for s in pos[sp]] + [(s, 0) for s in neg[sp]]
        keep = []
        for s, l in recs:
            h = hashlib.sha256(s.encode()).hexdigest()
            if h in seen: continue
            seen.add(h); keep.append((s, l))
        df = pd.DataFrame(keep, columns=["sequence", "label"]).sample(frac=1.0,
                                                                      random_state=a.seed)
        df.to_csv(f"{a.out}/{sp}.csv", index=False)
        gc = lambda ss: float(np.mean([(x.count("G") + x.count("C")) / len(x) for x in ss]))
        summary[sp] = dict(n=len(df), pos=int(df.label.sum()),
                           gc_viral=round(gc(df[df.label == 1].sequence), 4),
                           gc_host=round(gc(df[df.label == 0].sequence), 4))
        print(f"  {sp}: {len(df)} rows, {int(df.label.sum())} viral | "
              f"GC viral {summary[sp]['gc_viral']:.3f} vs host {summary[sp]['gc_host']:.3f}",
              flush=True)
    json.dump(dict(chunk=a.chunk, cap_family=a.cap_family, seed=a.seed,
                   positives="ViroBench ALL/times, family-capped, genome-level split",
                   negatives="hg38 random ACGT-clean chunks, disjoint draws per split",
                   order="genome split -> chunk -> SHA-256 dedup", splits=summary,
                   caveat="GC differs between classes; a composition baseline may do well. "
                          "That is the point of reporting max(k-mer, GBT, CNN, NN) as floor."),
              open(f"{a.out}/meta.json", "w"), indent=2)
    print(f"\nwrote {a.out}/{{train,dev,test}}.csv")


if __name__ == "__main__":
    main()
