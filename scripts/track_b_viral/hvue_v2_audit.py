"""HVUE **v2** leakage audit — easy-search grid + easy-cluster comparison.

Context
-------
Our v1 critique (98% of Pathogenicity/Transmissibility test rows homologous to train;
96 and 60 clean rows respectively) is **SUPERSEDED**. The HViLM authors released HVUE v2
(HF `duttaprat/HVUE-v2`, rev 9d7bfa32, 2026-09-02), which acknowledges the v1 defect and
rebuilds with a six-stage pipeline: genome-level dedup -> MMseqs2 clustering at 95%
(standard) or 70% (hard) -> cluster-aware 70/15/15 split -> chunking into 500/1000/2000bp
windows -> SHA-256 exact dedup -> automated leakage audit. The key fix is ordering: v1
chunked then split randomly; v2 splits at cluster level then chunks. Do not resurrect the
v1 claim.

The question this script answers
--------------------------------
The v2 README states "Zero exact-match or near-match overlap confirmed across all
configurations." We test that with a stricter instrument than clustering.

**Mechanistic prediction being tested.** The authors cluster *genomes* at 95% identity,
then chunk. Two genomes that are, say, 88% identical fall into *different* clusters at a
95% threshold, so they are assigned to different splits -- but their chunks remain ~88%
identical. Cluster-aware splitting at 95% therefore cannot, by construction, remove
homology below its own threshold. So `standard_95` configs should show residual leakage
concentrated in the 70-90% identity band, while `hard_70` configs should be materially
cleaner. If that holds, it is a stronger finding than the v1 critique: careful,
well-intentioned cluster-aware splitting still leaves partial homology that local
alignment finds.

Method
------
* **easy-search** (local alignment) via the existing `partial_overlap_audit.search()`:
  one permissive search per config, then post-filtered over the full threshold grid
  identity {70,80,90}% x coverage {30,50,70}% of the TEST sequence. 70% identity is
  lenient for a 4-letter alphabet (unrelated DNA is ~25% identical by chance), so the
  primary claim is read at the field-standard 80-90% band; the full grid is reported so
  the verdict is threshold-robust.
* **easy-cluster -c 0.9** via the existing `mmseqs_leakage_check.cluster_overlap()`, on
  the same data, to quantify the tool gap. easy-cluster needs 90% *bidirectional*
  coverage and in a direct test caught 0 of 150 half-overlap pairs.

Query-side capping
------------------
Host_Tropism/standard_95_500bp has 100,403 test rows against 454,018 train. Test rows are
sampled down to `--test-cap` where needed; the TARGET (train) side is never subsampled,
since that would bias leakage downward. A leakage rate is a proportion, so a capped query
set gives an unbiased estimate with a binomial CI, reported alongside.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "common"))
sys.path.insert(0, os.path.join(REPO, "positive_control"))

from partial_overlap_audit import search          # easy-search + grid post-filter
from mmseqs_leakage_check import cluster_overlap  # easy-cluster -c 0.9

V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")
V2_REVISION = "9d7bfa32fe0ba7f697fec551260de16d1a51ec6c"
IDENTS = [0.70, 0.80, 0.90]
COVFRACS = [0.30, 0.50, 0.70]


def configs(root):
    out = []
    for task in sorted(os.listdir(root)):
        tdir = os.path.join(root, task)
        if not os.path.isdir(tdir):
            continue
        for cfg in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, cfg)
            if os.path.exists(os.path.join(cdir, "train.csv")):
                out.append((task, cfg, cdir))
    return out


def wilson(k, n, z=1.96):
    """95% CI for a proportion. The query side is capped, so the interval is the honest
    statement of what a capped audit can resolve."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, c - h), 2), round(100 * min(1.0, c + h), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=V2)
    ap.add_argument("--out", default=f"{REPO}/../reports/hvue_v2_leakage_audit.json")
    ap.add_argument("--test-cap", type=int, default=20000,
                    help="cap on TEST (query) rows; train side never capped")
    ap.add_argument("--cluster-cap", type=int, default=30000,
                    help="easy-cluster pools train+test, so it needs its own cap")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--skip-cluster", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for task, cfg, cdir in configs(a.root):
        if a.tasks and task not in a.tasks:
            continue
        tr = pd.read_csv(f"{cdir}/train.csv")
        te = pd.read_csv(f"{cdir}/test.csv")
        n_test_full = len(te)
        if a.test_cap and len(te) > a.test_cap:
            te = te.sample(n=a.test_cap, random_state=a.seed).reset_index(drop=True)
        tag = f"{task}/{cfg}"
        print(f"\n=== {tag}: train={len(tr):,} test={n_test_full:,} "
              f"(query capped to {len(te):,})", flush=True)

        # --- exact duplicates: the authors' own SHA-256 stage should make this zero ---
        train_seqs = set(tr.sequence.str.upper())
        exact = int(te.sequence.str.upper().isin(train_seqs).sum())
        print(f"  exact duplicates: {exact}/{len(te)} ({100*exact/len(te):.3f}%)", flush=True)

        # --- easy-search over the full grid ---
        r = search(tr, te, tag, nt=True, idents=IDENTS, covfracs=COVFRACS, threads=a.threads)
        if r is None:
            print(f"  {tag}: easy-search FAILED", flush=True)
            continue
        for key, pct in sorted(r.items()):
            k = int(round(pct / 100.0 * len(te)))
            lo, hi = wilson(k, len(te))
            print(f"    easy-search {key:>16s}: {pct:6.2f}%  [{lo}, {hi}]  "
                  f"clean={len(te)-k:,}/{len(te):,}", flush=True)

        # --- easy-cluster -c 0.9, same data, for the tool-gap comparison ---
        clu = None
        if not a.skip_cluster:
            ctr = tr if len(tr) <= a.cluster_cap else tr.sample(n=a.cluster_cap, random_state=a.seed)
            cte = te if len(te) <= a.cluster_cap else te.sample(n=a.cluster_cap, random_state=a.seed)
            try:
                clu = cluster_overlap(ctr.sequence.tolist(), cte.sequence.tolist(),
                                      f"{tag} easy-cluster", min_id=0.90, cov=0.9)
            except Exception as exc:
                print(f"  easy-cluster failed: {type(exc).__name__}: {exc}", flush=True)

        rows.append(dict(
            task=task, config=cfg, revision=V2_REVISION,
            n_train=len(tr), n_test_full=n_test_full, n_test_queried=len(te),
            exact_duplicates=exact,
            exact_duplicate_pct=round(100 * exact / len(te), 4),
            easy_search={k: v for k, v in sorted(r.items())},
            easy_search_ci={k: wilson(int(round(v / 100.0 * len(te))), len(te))
                            for k, v in r.items()},
            easy_cluster_pct=(round(100 * clu["frac_leaked"], 2) if clu else None),
            easy_cluster_n_train=(clu["n_train"] if clu else None),
            easy_cluster_n_test=(clu["n_test"] if clu else None),
        ))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=2)

    print(f"\nwrote {a.out}  ({len(rows)} configs)")
    if rows:
        tbl = pd.DataFrame([
            dict(task=r["task"], config=r["config"], n_test=r["n_test_full"],
                 exact=r["exact_duplicate_pct"],
                 **{k.replace("id", "s"): v for k, v in r["easy_search"].items()},
                 cluster90=r["easy_cluster_pct"])
            for r in rows])
        pd.set_option("display.width", 250)
        print("\n%% of TEST rows with >=1 train hit (easy-search), and easy-cluster -c 0.9:")
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
