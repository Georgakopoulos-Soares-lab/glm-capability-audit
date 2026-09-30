"""P1b: build a viral pool with NO easy-search hit to HVUE v2 training, for the pseudo-PPL DiD.

Why
---
The pseudo-PPL memorisation test compared HViLM vs DNABERT-2 on pool A (HVUE Host_Tropism test)
against pool B (ViroBench). DiD was -0.017, read as "no HVUE-specific memorisation".

That reading is invalid: ViroBench and HVUE share a sequence universe. The VHDB task, built
from ViroBench genomes, is 50-76% homologous to HVUE training. If both pools contain the same
genomes, a null DiD is what you observe whether or not HViLM memorised them.

This builds pool B' = ViroBench chunks with NO easy-search hit to HVUE Host_Tropism training at
a permissive threshold (so the filter is aggressive: anything even loosely related to HVUE
training is excluded). Re-running the DiD on pool B' either restores the memorisation
conclusion or refutes it.

Aggressive by design: we filter at id>=0.70 / cov>=0.30, the LOOSEST identity (which detects
the MOST homology and therefore removes the MOST sequence). Reverse complements are included
(easy-search --search-type 3, verified).
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
SCRATCH = os.environ.get("VB_SCRATCH", "/data/nvidia/tmp_mmseqs")
V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--cap-family", type=int, default=3)
    ap.add_argument("--n-out", type=int, default=4000)
    ap.add_argument("--min-id", type=float, default=0.70)
    ap.add_argument("--min-cov", type=float, default=0.30)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--out", default="/data/nvidia/hvue_disjoint_pool/poolB_prime.csv")
    a = ap.parse_args()
    import virobench_baselines as VBB

    rng = np.random.default_rng(0)
    vb = pd.concat([VBB.load("ALL", "times", s) for s in ("train", "test")], ignore_index=True)
    vb = vb[vb.sequence.str.len() >= a.chunk + 1000].reset_index(drop=True)
    vb = vb.sample(frac=1.0, random_state=0).reset_index(drop=True)
    # family-cap FIRST: raw ViroBench times is ~74% Orthomyxoviridae
    vb = (vb.groupby("family", group_keys=False).apply(lambda g: g.head(a.cap_family))
            .sample(frac=1.0, random_state=0).reset_index(drop=True))
    cand, fams = [], []
    for _, r in vb.iterrows():
        s = r.sequence.upper()
        o = int(rng.integers(0, max(1, len(s) - a.chunk)))
        cand.append(s[o:o + a.chunk]); fams.append(r.family)
        if len(cand) >= a.n_out * 3:
            break
    print(f"candidates: {len(cand):,} chunks from {len(set(fams)):,} families", flush=True)

    tr = pd.concat([pd.read_csv(f"{V2}/Host_Tropism/{c}/train.csv")
                    for c in ("hard_70_1000bp", "standard_95_1000bp")], ignore_index=True)
    print(f"HVUE Host_Tropism train (both configs): {len(tr):,} rows", flush=True)

    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        with open(f"{d}/q.fa", "w") as f:
            for i, s in enumerate(cand): f.write(f">{i}\n{s}\n")
        with open(f"{d}/t.fa", "w") as f:
            for i, s in enumerate(tr.sequence.astype(str)): f.write(f">{i}\n{s}\n")
        r = subprocess.run([MMSEQS, "easy-search", f"{d}/q.fa", f"{d}/t.fa", f"{d}/h.tsv",
                            f"{d}/tmp", "--search-type", "3", "--min-seq-id", str(a.min_id),
                            "-c", str(a.min_cov), "--cov-mode", "2", "-e", "1e-3",
                            "--max-seqs", "1", "--threads", str(a.threads), "-v", "1",
                            "--format-output", "query,target,pident,qcov"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"easy-search failed: {r.stderr[-400:]}")
        hit = set()
        if os.path.getsize(f"{d}/h.tsv"):
            h = pd.read_csv(f"{d}/h.tsv", sep="\t", header=None,
                            names=["q", "t", "pident", "qcov"])
            hit = set(h.q.astype(int))
    finally:
        shutil.rmtree(d, ignore_errors=True)

    keep = [i for i in range(len(cand)) if i not in hit]
    print(f"  {len(hit):,}/{len(cand):,} candidates hit HVUE train "
          f"({100*len(hit)/len(cand):.1f}%) -> {len(keep):,} DISJOINT", flush=True)
    if len(keep) < 300:
        print("  WARNING: fewer than 300 disjoint chunks -- pool B' may be underpowered")
    keep = keep[:a.n_out]
    out = pd.DataFrame({"sequence": [cand[i] for i in keep],
                        "family": [fams[i] for i in keep]})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"  wrote {a.out}  ({len(out):,} chunks, {out.family.nunique()} families)")
    json.dump(dict(n_candidates=len(cand), n_hit_hvue=len(hit), n_disjoint=len(keep),
                   filter=f"id>={a.min_id} cov>={a.min_cov} (loosest = most aggressive)",
                   families=int(out.family.nunique())),
              open(a.out.replace(".csv", "_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
