# Reddy et al. 2026 as an external validation set: overlap audit

Status: **complete.** All four searches finished: nucleotide and six-frame translated, against
both HVUE host tropism training splits.

## What was acquired

Official release, verified. Repository `github.com/lanl/ldrd_virus_work` at tag `v0.1.0`
(`a05cdfca`), genomes from figshare DOI `10.6084/m9.figshare.30025597`. The downloaded archive's
SHA256 begins `3a3d3f4e81`, matching the hash stated in the paper's Data Availability statement.
All 2,709 accessions referenced by the official split are present in the cache; none missing.

Official fixed split used as shipped (`Relabeled_Train.csv`, `Relabeled_Test.csv`). No re-splitting.

## Method

The frozen L1 and L2 protocols from the manuscript, unchanged: MMseqs2 `easy-search`,
`--search-type 3` nucleotide and `--search-type 2 -s 7.5` six-frame translated, with
`--min-seq-id 0.30 -c 0.10 --cov-mode 2 -e 1e-3`, both strands.

One adaptation, stated because it matters. In the manuscript both query and target are 1 kb
fragments, so `--cov-mode 2` (coverage of the query) refers to a 1 kb sequence. Reddy records are
whole genomes up to 175 kb. Making a genome the query would silently change what the 10 percent
coverage threshold means. The search is therefore run with **HVUE train fragments as the query and
Reddy genomes as the target**, which keeps the coverage reference at 1 kb. No threshold was altered.

## Result: Reddy test (736 species) against HVUE host tropism train

| measure | `hard_70` | `standard_95` |
|---|---|---|
| contains an HVUE train fragment verbatim (100% identity, >=99% of the 1 kb fragment) | 325 (**44.2%**) | 331 (**45.0%**) |
| >=95% identity over >=80% of a train fragment | 342 (46.5%) | 354 (48.1%) |
| >=70% identity over >=50% of a train fragment | 614 (83.4%) | 626 (85.1%) |
| detectable nucleotide relative at the frozen threshold | 723 (**98.2%**) | 724 (**98.4%**) |
| ... among the 140 human-positive species | **140 / 140** | 139 / 140 |
| nucleotide no-hit species that have a translated relative | 13 / 13 | 12 / 12 |
| **neither nucleotide nor translated relative** | **0 (0.0%)** | **0 (0.0%)** |

Median best identity among species with a nucleotide hit: 87.5%, ninetieth percentile 100%.

## Every Reddy test species is related to HVUE training data

The nucleotide no-hit set is 13 species for `hard_70` and 12 for `standard_95`, and the six-frame
translated search finds a relative for **every one of them**, at 35.8 to 75.0 percent identity.
The species involved are mostly non-mammalian or non-human, namely duck and heron hepatitis B
viruses, avian and rabbit gemyviruses, several torque teno viruses and an equine picobirnavirus.
The single human-positive species in either no-hit set, *Human associated gemyvongvirus 1*, has a
translated relative at 64.0 percent identity.

There is therefore **no subset of the Reddy test split that is evolutionarily independent of HVUE
host tropism training data** at the manuscript's detection thresholds. The L2 population is empty.

## Classification

**C. SUBSTANTIALLY OVERLAPPING BENCHMARK.**

Not B (externally curated but evolutionarily related), because the overlap is not merely taxonomic:
44.2 percent of Reddy test species contain an HVUE training fragment as an exact substring. Not A,
and not D — the labels are compatible enough (see `label_semantics.md`); the obstacle is sequence
overlap.

## What is nonetheless valuable

Three things, none of which require independence.

1. **Clean source provenance.** One row per species with accessions, which HVUE does not provide.
   This is the sampling unit our manuscript says is missing.
2. **Independently curated labels**, reviewed to July 2025, including explicit review of negatives
   that Mollentze's holdout left as absence-of-evidence.
3. **An independent replication of this manuscript's central claim.** Reddy report human-infection
   ROC AUC of 0.663 +/- 0.070 under Mollentze's original split (relative entropy 3.00), 0.784 +/-
   0.013 under a random split (relative entropy 0.08), and **0.50 +/- 0.08 when train and test share
   no viral family** (relative entropy > 24). That is a dose-response between train/test phylogenetic
   distance and measured performance, obtained with conventional ML on the same phenotype, by
   authors with no connection to this work.
