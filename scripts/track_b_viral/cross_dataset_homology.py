"""BLOCKING CHECK: is the VHDB replication actually independent of HVUE v2?

HVUE v2 draws from NCBI Virus, BV-BRC and Virus-Host DB. Our VHDB task draws labels from
Virus-Host DB and genomes from ViroBench (RefSeq/GenBank). The Virus-Host DB overlap alone
makes shared sequence likely, and if VHDB test rows are homologous to HVUE TRAIN rows then
"independent replication" is confounded -- HViLM fine-tuned on HVUE-like data would be seeing
relatives of its VHDB test set.

Measures, in both directions, with easy-search (which detects reverse complements -- verified):
  * VHDB test   vs HVUE v2 train   <- the one that matters for the replication claim
  * VHDB train  vs HVUE v2 train   <- shared training material (less critical, still reported)

Reuses `partial_overlap_audit.search()`; emits per-row leaked indices so the VHDB result can
be re-scored with HVUE-overlapping rows removed.
"""
import argparse, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "common"))
from partial_overlap_audit import search

V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")
VHDB = "/data/nvidia/vhdb_tropism/within_family_1000bp"
IDENTS = [0.70, 0.80, 0.90, 0.95]
COVFRACS = [0.30, 0.50, 0.70, 0.80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hvue-cfgs", nargs="+",
                    default=["Host_Tropism/hard_70_1000bp", "Host_Tropism/standard_95_1000bp"])
    ap.add_argument("--rows-dir", default="/data/nvidia/genomic-capability-unlearning/"
                                          "results_viral_bench/leaked_rows_crossdataset")
    ap.add_argument("--out", default=f"{REPO}/../reports/vhdb_hvue_independence.json")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()

    res = []
    for part in ("test", "train"):
        q = pd.read_csv(f"{VHDB}/{part}.csv").reset_index(drop=True)
        for cfg in a.hvue_cfgs:
            tgt = pd.read_csv(f"{V2}/{cfg}/train.csv").reset_index(drop=True)
            tag = f"VHDB_{part}__vs__HVUE_{cfg.replace('/','_')}_train"
            print(f"\n=== {tag}: query={len(q):,}  target={len(tgt):,}", flush=True)
            r = search(tgt, q, tag, nt=True, idents=IDENTS, covfracs=COVFRACS,
                       threads=a.threads,
                       leaked_ids_dir=(a.rows_dir if part == "test" else None))
            if r is None:
                print(f"  {tag}: FAILED", flush=True); continue
            for k, v in sorted(r.items()):
                print(f"    {k:>14s}: {v:6.2f}%  ({int(round(v/100*len(q))):,}/{len(q):,} rows)",
                      flush=True)
            res.append(dict(query=f"VHDB/{part}", target=f"HVUE_v2/{cfg}/train",
                            n_query=len(q), n_target=len(tgt), pct_with_hit=r))
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("\nVERDICT GUIDE: if VHDB *test* shows substantial overlap with HVUE *train* at "
          "id>=0.8, the 'independent replication' claim is confounded and VHDB must be "
          "re-scored with those rows removed before the claim is made.")


if __name__ == "__main__":
    main()
