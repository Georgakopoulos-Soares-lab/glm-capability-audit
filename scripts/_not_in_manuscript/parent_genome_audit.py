"""Parent-genome leakage audit -- an axis orthogonal to homology search.

Benchmarks built by chopping genomes into windows can place two chunks of the
SAME accession on opposite sides of the split. Those chunks need not align to
each other (HVUE's windows are non-overlapping), so an MMseqs2 audit can call
them clean while the split is still leaking at the genome level. ViTrace groups
fragments by source sequence before splitting for exactly this reason.

This emits, per benchmark, one row of the table a reviewer will ask for:

  exact duplicates | revcomp duplicates | shared parent genome | species | genus | family

Cheap, and it is the audit that MMseqs2 structurally cannot perform.
"""
import argparse, hashlib, os, sys
import numpy as np, pandas as pd


def _h(s):
    return hashlib.md5(s.strip().upper().encode()).hexdigest()


def _rc(s):
    return s.strip().upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True); ap.add_argument("--test-csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-meta", default="", help="train-row-order csv with taxid/family/...")
    ap.add_argument("--test-meta", default="", help="test-row-order csv, same columns")
    ap.add_argument("--keys", nargs="+", default=["cluster", "taxid", "species", "genus", "family"])
    ap.add_argument("--out", default=os.environ.get(
        "VB_REPORTS", "/home/nvidia/genomic-capability-unlearning/reports") + "/leakage")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    rd = lambda p: pd.read_csv(p).rename(columns={"text": "sequence"}).reset_index(drop=True)
    tr, te = rd(a.train_csv), rd(a.test_csv)
    trs = tr.sequence.astype(str); tes = te.sequence.astype(str)

    htr = set(trs.map(_h))
    dup = np.array([_h(s) in htr for s in tes])
    rcd = np.array([_h(_rc(s)) in htr for s in tes])
    rec = {"name": a.name, "n_train": len(tr), "n_test": len(te),
           "exact_duplicate_frac": round(float(dup.mean()), 4),
           "revcomp_duplicate_frac": round(float(rcd.mean()), 4)}

    # duplicates carrying a DIFFERENT label are worse than duplicates: they cap
    # the achievable score and they mean the labels disagree with themselves.
    if dup.any():
        lab = {}
        for s, l in zip(trs, tr.label.values):
            lab.setdefault(_h(s), set()).add(int(l))
        conf = sum(1 for s, l in zip(tes[dup], te.label.values[dup])
                   if int(l) not in lab.get(_h(s), set()))
        rec["exact_duplicates_with_conflicting_label"] = int(conf)

    if a.train_meta and a.test_meta:
        mtr, mte = pd.read_csv(a.train_meta), pd.read_csv(a.test_meta)
        if len(mtr) == len(tr) and len(mte) == len(te):
            for k in a.keys:
                if k in mtr.columns and k in mte.columns:
                    seen = set(mtr[k].dropna().astype(str))
                    v = mte[k].astype(str)
                    rec[f"shared_{k}_frac"] = round(float(v.isin(seen).mean()), 4)
                    rec[f"n_{k}_train"] = int(mtr[k].nunique())
                    rec[f"n_{k}_test"] = int(v.nunique())
        else:
            rec["meta_note"] = (f"metadata row counts ({len(mtr)}/{len(mte)}) do not match "
                                f"the splits ({len(tr)}/{len(te)}); shared-key audit skipped")
    else:
        rec["meta_note"] = ("no taxonomy/accession metadata shipped with this benchmark, so "
                            "parent-genome leakage CANNOT be measured -- absence of evidence only")

    w = max(len(k) for k in rec)
    print(f"\n=== parent-genome leakage: {a.name} ===")
    for k, v in rec.items():
        print(f"  {k:{w}s}  {v}")
    import json
    p = f"{a.out}/{a.name}__parent_genome.json"
    json.dump(rec, open(p, "w"), indent=2)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
