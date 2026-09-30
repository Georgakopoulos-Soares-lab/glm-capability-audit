#!/usr/bin/env python
"""ViroBench release characterization. Descriptive only, no model is trained.

Reports properties of the released files we audited. It does not evaluate the
benchmark's published results, which cannot be reproduced from the release
because no preprocessing or evaluation code is distributed with it.
"""
import hashlib, json, os, sys, glob
import pandas as pd

SNAP = glob.glob("/data/nvidia/hf_cache/hub/datasets--YDXX--ViroBench/snapshots/*")[0]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "reports", "virobench")
os.makedirs(OUT, exist_ok=True)
C = f"{SNAP}/Classification"


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


man = dict(
    source="HuggingFace dataset YDXX/ViroBench",
    snapshot=os.path.basename(SNAP),
    license="cc-by-4.0", arxiv_on_card="2605.25388",
    date_accessed="2026-09-29",
    release_contents=sorted(os.path.relpath(p, SNAP) for p in glob.glob(f"{C}/*/*/*")),
    official_preprocessing_or_eval_code_present=False,
    note=("The release ships data only. No README, preprocessing script or evaluation "
          "code is present in the snapshot, and the arXiv identifier on the dataset card "
          "returns no record from the arXiv API. The model context-length and truncation "
          "policy used for the published results therefore cannot be reconstructed from "
          "the available release."),
    files={})
rows_ov, lens, comp = [], [], []
for split in ("times", "genus"):
    for part in ("train", "val", "test"):
        p = f"{C}/ALL/taxon/{split}/{part}.csv"
        man["files"][f"ALL/taxon/{split}/{part}.csv"] = dict(
            sha256_16=sha(p), rows=sum(1 for _ in open(p)) - 1)
    tr = pd.read_csv(f"{C}/ALL/taxon/{split}/train.csv")
    te = pd.read_csv(f"{C}/ALL/taxon/{split}/test.csv")
    for lvl in ("taxid", "species", "genus", "family"):
        TR, TE = set(tr[lvl].dropna()), set(te[lvl].dropna())
        rows_ov.append(dict(released_split=split, level=lvl,
                            test_unique=len(TE), shared_with_train=len(TE & TR),
                            pct_test_levels_seen_in_train=round(100 * len(TE & TR) / max(len(TE), 1), 1),
                            pct_test_rows_with_train_seen_level=round(100 * te[lvl].isin(TR).mean(), 1)))
    man[f"dates_{split}"] = dict(
        train=[str(tr.first_release_date.min())[:10], str(tr.first_release_date.max())[:10]],
        test=[str(te.first_release_date.min())[:10], str(te.first_release_date.max())[:10]])
    fam_tr, fam_te = set(tr.family.dropna()), set(te.family.dropna())
    man[f"family_label_space_{split}"] = dict(
        official_families_train_union_test=len(fam_tr | fam_te),
        test_families=len(fam_te), test_families_absent_from_train=len(fam_te - fam_tr),
        test_rows_in_absent_families=int((~te.family.isin(fam_tr)).sum()),
        pct_test_rows_in_absent_families=round(100 * (~te.family.isin(fam_tr)).mean(), 2))
    for nm, d in (("train", tr), ("test", te)):
        L = d.seq_len.dropna().astype(float)
        lens.append(dict(released_split=split, part=nm, n=len(L),
                         **{k: int(v) for k, v in zip(
                             ("min", "q25", "median", "q75", "p90", "p99", "max"),
                             (L.min(), L.quantile(.25), L.median(), L.quantile(.75),
                              L.quantile(.9), L.quantile(.99), L.max()))}))

te = pd.read_csv(f"{C}/ALL/taxon/times/test.csv")
pd.DataFrame(rows_ov).to_csv(f"{OUT}/split_overlap.csv", index=False)
pd.DataFrame(lens).to_csv(f"{OUT}/sequence_lengths.csv", index=False)
pd.crosstab(te.realm, te.na_type).to_csv(f"{OUT}/na_type_by_realm.csv")
pd.crosstab(te.host_label, te.na_type).to_csv(f"{OUT}/na_type_by_host_label.csv")
hb = pd.read_csv(f"{C}/ALL/taxon/times/train.csv").host_label.value_counts()
pd.DataFrame({"train": hb, "test": te.host_label.value_counts()}).fillna(0).astype(int)\
  .to_csv(f"{OUT}/host_label_composition.csv")
te.groupby("na_type").agg(rows=("taxid", "size"), unique_taxid=("taxid", "nunique"),
                          median_seq_len=("seq_len", "median")).to_csv(f"{OUT}/na_type_counts.csv")
json.dump(man, open(f"{OUT}/dataset_manifest.json", "w"), indent=2)
print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=2)[:1200])
print(f"\nwrote {OUT}/ : dataset_manifest.json split_overlap.csv sequence_lengths.csv "
      f"na_type_by_realm.csv na_type_by_host_label.csv host_label_composition.csv na_type_counts.csv")
