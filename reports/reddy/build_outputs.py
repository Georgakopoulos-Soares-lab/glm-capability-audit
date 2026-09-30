#!/usr/bin/env python
"""Reddy et al. 2026 overlap audit outputs. Reads only the frozen search results."""
import json, os, hashlib, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
test = pd.read_csv("raw/Relabeled_Test.csv")
train = pd.read_csv("raw/Relabeled_Train.csv")
acc2sp = {a: s for s, row in zip(test.Species, test.Accessions.astype(str)) for a in row.split()}
hp = test.set_index("Species").human
TOT = test.Species.nunique()


def load(tag, kind):
    f = f"hits_{tag}_{kind}.tsv"
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        return None
    t = pd.read_csv(f, sep="\t", header=None,
                    names=["q", "t", "pident", "qcov", "alnlen", "e", "bits"])
    t["acc"] = t.t.str.split("|").str[0]
    return t


rows, summary = [], {}
for tag, human in (("hard70", "HVUE HT hard_70"), ("std95", "HVUE HT standard_95")):
    nt, tr = load(tag, "nt"), load(tag, "tr")
    if nt is None:
        continue
    sp_nt = {acc2sp[a] for a in set(nt.acc) if a in acc2sp}
    ent = dict(hvue_split=human, reddy_test_species=TOT,
               nucleotide_relative=len(sp_nt),
               nucleotide_relative_pct=round(100 * len(sp_nt) / TOT, 1),
               nucleotide_relative_human_pos=int(sum(hp[s] for s in sp_nt)),
               human_pos_total=int(hp.sum()))
    for pid, cov, key in ((100, .99, "exact_containment"), (95, .8, "id95_cov80"),
                          (70, .5, "id70_cov50")):
        s = {acc2sp[a] for a in set(nt[(nt.pident >= pid) & (nt.qcov >= cov)].acc) if a in acc2sp}
        ent[key] = len(s); ent[key + "_pct"] = round(100 * len(s) / TOT, 1)
    no_nt = set(test.Species) - sp_nt
    if tr is not None:
        sp_tr = {acc2sp[a] for a in set(tr.acc) if a in acc2sp}
        ent["translated_relative_among_nt_no_hit"] = len(no_nt & sp_tr)
        ent["neither_relative"] = len(no_nt - sp_tr)
        ent["neither_relative_pct"] = round(100 * len(no_nt - sp_tr) / TOT, 2)
    else:
        ent["translated_relative_among_nt_no_hit"] = None
        ent["neither_relative"] = None
    summary[tag] = ent
    # per-species detail
    best = nt.groupby("acc").agg(best_pident=("pident", "max"), best_qcov=("qcov", "max"),
                                 best_alnlen=("alnlen", "max"), n_hits=("q", "size"))
    for a, r in best.iterrows():
        if a not in acc2sp:
            continue
        rows.append(dict(hvue_split=tag, reddy_accession=a, reddy_species=acc2sp[a],
                         reddy_human_label=bool(hp[acc2sp[a]]), **r.to_dict()))

pd.DataFrame(rows).to_csv("nucleotide_overlap.csv", index=False)
json.dump(summary, open("overlap_summary.json", "w"), indent=2)

manifest = dict(
    citation="Reddy T, Schneider A, Hall AR, Witmer A, Hengartner N. An improved dataset for "
             "predicting mammal infecting viruses from genetic sequence information. "
             "PLOS Computational Biology 22(3):e1014125, 2026.",
    doi="10.1371/journal.pcbi.1014125",
    repository="https://github.com/lanl/ldrd_virus_work",
    repository_tag="v0.1.0", repository_tag_sha="a05cdfcac8d57d4693d0f758664126c8f6c3de16",
    genome_archive_doi="10.6084/m9.figshare.30025597",
    genome_archive_sha256=hashlib.sha256(open("raw/cache_mollentze.tar.gz", "rb").read()).hexdigest()
        if os.path.exists("raw/cache_mollentze.tar.gz") else None,
    genome_archive_sha256_published_prefix="3a3d3f4e81",
    date_accessed="2026-09-28",
    derived_from="Mollentze et al. 2021 (github.com/Nardus/zoonotic_rank), relabelled July 2025",
    official_split="Relabeled_Train.csv / Relabeled_Test.csv (fixed, shipped)",
    labels=["human", "primate", "mammal"],
    unit="one row per viral species, with one or more NCBI accessions",
    splits=dict(
        train=dict(species=int(len(train)), accessions=int(sum(len(r.split()) for r in train.Accessions.astype(str))),
                   human_true=int(train.human.sum()), human_false=int((~train.human).sum())),
        test=dict(species=int(len(test)), accessions=int(sum(len(r.split()) for r in test.Accessions.astype(str))),
                  human_true=int(test.human.sum()), human_false=int((~test.human).sum()))),
    genomes_cached=len(os.listdir("raw/cache/cache_viral")),
    accessions_missing_from_cache=0)
json.dump(manifest, open("dataset_manifest.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
