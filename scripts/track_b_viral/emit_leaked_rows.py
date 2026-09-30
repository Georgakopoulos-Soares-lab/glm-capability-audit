"""Emit the leaked TEST ROW INDICES for the three cells where a gLM beats max(k-mer, CNN).

H3 asks whether an excess survives deleting the test rows that are homologous to train. That
needs a per-row mask, not the aggregate percentage the audit already reports, so this calls
`partial_overlap_audit.search(..., leaked_ids_dir=)` and writes one `.rows` file per
(identity, coverage) cell for `rescore_filtered.py` to consume.

Index discipline: predictions were saved in test-CSV row order, so `test_df` is passed with a
reset RangeIndex and the fasta headers ARE the 0-based row numbers. Do not pass a filtered or
re-sorted frame here.

Grid note: a LOOSER identity threshold detects more homology and therefore removes MORE rows.
id>=0.7/cov>=0.3 is the most aggressive filter and is the headline; id>=0.9 is the weakest.
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "common"))
from partial_overlap_audit import search

V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")
TASKS = {
    "hvue_v2__Host_Tropism__hard_70_1000bp":     f"{V2}/Host_Tropism/hard_70_1000bp",
    "hvue_v2__Host_Tropism__standard_95_1000bp": f"{V2}/Host_Tropism/standard_95_1000bp",
    "hvue_v2__Pathogenicity__hard_capped_1000bp": f"{V2}/Pathogenicity/hard_capped_1000bp",
    "hvue_v2__Transmissibility__hard_capped_1000bp": f"{V2}/Transmissibility/hard_capped_1000bp",
    "vhdb_tropism_wf1000": "/data/nvidia/vhdb_tropism/within_family_1000bp",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/nvidia/genomic-capability-unlearning/"
                                     "results_viral_bench/leaked_rows")
    ap.add_argument("--idents", nargs="+", type=float, default=[0.70, 0.80, 0.90, 0.95])
    ap.add_argument("--covfracs", nargs="+", type=float, default=[0.30, 0.50, 0.70, 0.80])
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for tag, root in TASKS.items():
        if a.tasks and tag not in a.tasks:
            continue
        tr = pd.read_csv(f"{root}/train.csv").reset_index(drop=True)
        te = pd.read_csv(f"{root}/test.csv").reset_index(drop=True)
        print(f"\n=== {tag}: train={len(tr):,} test={len(te):,}", flush=True)
        r = search(tr, te, tag, nt=True, idents=a.idents, covfracs=a.covfracs,
                   threads=a.threads, leaked_ids_dir=a.out)
        if r is None:
            print(f"  {tag}: FAILED", flush=True); continue
        for k, v in sorted(r.items()):
            p = f"{a.out}/{tag}__{k}.rows"
            n = sum(1 for _ in open(p)) if os.path.exists(p) and os.path.getsize(p) else 0
            print(f"    {k:>14s}: {v:6.2f}%  -> {n:,} rows dropped, {len(te)-n:,} clean",
                  flush=True)


if __name__ == "__main__":
    main()
