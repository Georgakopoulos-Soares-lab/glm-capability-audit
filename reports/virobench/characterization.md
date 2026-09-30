# ViroBench release characterization

Descriptive audit of the released files. No model was trained. This does not evaluate
ViroBench's published results and makes no claim about their validity; the release ships data
only, so the protocol behind those results cannot be reconstructed from it.

## 1. Release audited

HuggingFace dataset `YDXX/ViroBench`, licence CC-BY-4.0, accessed 2026-09-29. Snapshot and
per-file SHA256 prefixes and row counts are in `dataset_manifest.json`. The release contains
`Classification/{ALL,DNA,RNA}/taxon/{genus,times}/{train,val,test}.csv` with accompanying
`*_sequences.jsonl` for ALL and DNA. ALL temporal split: 46,651 train, 5,831 val, 5,832 test.

## 2. Train and test overlap in the released splits

Measured directly from the released CSVs (`split_overlap.csv`). The column headings below use
the split names as released.

| released split | level | % of test levels seen in train | % of test rows whose level was seen in train |
|---|---|---|---|
| `times` | taxid | 0.0 | 0.0 |
| `times` | species | 2.1 | 4.4 |
| `times` | genus | 50.3 | 80.0 |
| `times` | family | 72.7 | 94.4 |
| `genus` | taxid | 0.0 | 0.0 |
| `genus` | species | 17.7 | 78.8 |
| `genus` | genus | **82.1** | **97.9** |
| `genus` | family | 97.8 | 99.9 |

In the split named `genus`, 82.1 percent of test genera and 97.9 percent of test rows belonged
to genera represented in training. We report this as an observed property of the released files.
We do not know how the split was constructed and therefore do not characterise its intended
design.

## 3. Date distributions

`times`: train 1982-06-09 to 2017-10-21, test 2020-02-03 to 2025-07-12, with no overlap and a
gap of more than two years.

`genus`: train 1982-06-09 to 2025-07-12, test 1983-07-13 to 2024-10-21, fully overlapping.

## 4. Family label space and unseen test families

Official family label space, train union test, with no filtering applied: **308 families** in the
temporal split and 310 in the `genus` split. The temporal test split contains 238 families, of
which **65 do not appear in training**, covering **327 test rows, 5.6 percent** of the split.

For reference, restricting to families with at least 10 training examples reduces the space to
104, which is the derivation used by earlier internal runs and is not the official label space.

## 5. Sequence lengths

Full distributions in `sequence_lengths.csv`. Temporal split, base pairs:

| | median | q75 | p90 | p99 | max |
|---|---|---|---|---|---|
| train | 13,699 | 13,783 | 14,746 | 159,562 | 1,573,834 |
| test | 41,239 | 85,472 | 164,871 | 270,865 | 659,950 |

Test genomes are about three times longer than training genomes at the median.

## 6. Nucleic-acid type, realm and host category

`na_type` in the temporal test split: DNA 4,178 rows (4,178 unique taxid, median 51,584 bp),
RNA 1,625 (1,625 taxid, median 9,245 bp), UNKNOWN 29.

`na_type` is collinear with realm by construction (`na_type_by_realm.csv`). Duplodnaviria,
Monodnaviria, Varidnaviria and Adnaviria are entirely DNA; Riboviria and Ribozyviria are entirely
RNA. The two strata share no realm.

Host category is also strongly associated with `na_type` (`na_type_by_host_label.csv`). The
`host_label` field carries 8 categories; A is 3,687 DNA against 15 RNA, B is 1 against 326, C is
86 against 520, E is 31 against 441. Example hosts per category place A at bacteria, B at fungi,
C at plants, D1 at primates, D2 at livestock and birds, D3 at other vertebrates, E at
invertebrates.

Host composition also shifts strongly between the temporal train and test splits
(`host_label_composition.csv`): category A rises from 3.1 to 63.6 percent of rows while D1 falls
from 38.4 to 1.0 percent.

Note that `host_processed` carries an accompanying `probability` field at approximately 0.9999,
which indicates the host strings were normalised by an automated procedure rather than curated
directly.

## 7. Absence of an official preprocessing protocol

The snapshot contains no README, preprocessing script or evaluation code, and the arXiv
identifier on the dataset card returns no record from the arXiv API. The model context length,
cropping and truncation policy behind the published results is therefore not recoverable from
the available release. Any evaluation we ran would be our protocol applied to ViroBench data,
not a reproduction of the published ViroBench evaluation.

## 8. Analyses considered and not pursued

**Five-model fine-tuning on family classification.** Not run. The only official task is 308-class
taxonomy classification, which differs substantially from the phenotype prediction tasks in this
manuscript, and no official protocol exists to reproduce.

**DNA versus RNA predictive comparison.** Considered and not pursued. Nucleic-acid type in this
dataset is confounded with viral realm, which it determines exactly, with host category, with
genome length (51.6 kb against 9.2 kb median) and consequently with the fraction of each genome
a fixed-length model input can observe. Matching on genome length does not address the realm and
host confounding, because the two strata share no realms. A measured difference would not be
attributable to nucleic-acid type.

## 9. Excluded prior runs

Six earlier internal ViroBench fine-tuning runs (DNABERT-2 and HViLM, ALL/times/family) are
**invalid and excluded**. They were configured with a maximum length of 5 tokens, a 6,000-row
training cap against an official training population of 46,651, and a non-official label space
derived by a minimum-count filter. They are not reported anywhere in this work.
