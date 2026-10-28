# Viral capability in genomic language models: complete results

> **Status: working record, superseded by the manuscript.**
> This document predates the final analysis and keeps vocabulary the manuscript has since
> dropped ("floor", "novelty ladder", L3-L5 strata). It also covers work that is not in the
> paper, among it the Evo 2 generative evaluation and the splice and GENEB tracks. Where it
> and `paper/main.tex` differ, the manuscript is correct. The current, generated record of
> what each claim rests on is [`reports/CLAIM_LEDGER.md`](./CLAIM_LEDGER.md), and the numbers
> themselves are in `reports/tables/final_source_of_truth_finalized.csv`.


Every number here is reproduced by `python pipelines/run.py all`. Per-task JSON
sits in `reports/{layered,hidden_homology,frozen_probe,leakage,author_checkpoints}/`
and `results/viral_bench/`.

**Vocabulary.** *Lookup-explainable* and *homology-exploitable*, never
"memorisation". *No-detectable-homology under \<protocol\>* and
*lookup-unavailable*, never "memorisation impossible". Every novelty claim names
the search that produced it.

---

## 1. What we set out to test, and what we found

Genomic language models are credited with viral capability on the strength of
benchmark tables. We tested whether those tables support the claim, using a floor
of `max(k-mer logistic, k-mer GBT, AA-composition GBT, searched CNN ladder,
from-scratch transformer ladder, nearest-neighbour lookup, taxonomy prior)` and a
nested novelty audit that strips increasingly remote forms of evolutionary
familiarity.

Three findings, in order of how much they should change what the field does.

**1. Nucleotide-only homology screens overstate evolutionary novelty by 65-97%.**
Translated (six-frame) search reclassifies most of what a nucleotide search calls
novel as homologous. Every "homology-clean" viral split in this study is far less
clean than its construction implies.

**2. The confound that most threatens these benchmarks cannot be audited on
eight of nine of them.** A taxonomy-prior baseline reaches 0.6809 MCC against
HViLM's 0.7036 on the one dataset that ships lineage labels. Eight of nine ship
`sequence,label` and nothing else, so the control cannot be run, parent-genome
leakage cannot be measured, and the bootstrap unit cannot be recovered.

**3. Every discriminative advantage that survives a complete floor dissolves
once translated homology is removed.** Not one model is SUPERIOR to the floor in
any no-detectable-homology stratum on any of nine tasks.

---

## 2. Apparent novelty collapses with search sensitivity

Test rows with **no detectable training neighbour**, by search:

| task | n | nucleotide hit | translated hit | rows reclassified | share of the "homolog-free" set that was hidden homology |
|---|---:|---:|---:|---:|---:|
| GUE phage | 4,630 | 37.9% | **98.1%** | 2,788 | **97.0%** |
| HVUE Transm hard | 2,497 | 97.8% | 99.9% | 52 | **94.5%** |
| HVUE Patho std | 6,651 | 99.4% | 100.0% | 38 | **92.7%** |
| HVUE Transm std | 12,728 | 99.7% | 100.0% | 38 | **92.7%** |
| HVUE Patho hard | 1,792 | 99.1% | 99.9% | 14 | **87.5%** |
| VHDB within-family | 3,754 | 68.0% | 94.6% | 999 | **83.2%** |
| HVUE HT hard_70 | 51,187 | 76.6% | 92.6% | 8,257 | **68.8%** |
| HVUE HT std_95 | 50,817 | 78.1% | 92.3% | 7,242 | **65.1%** |
| GUE covid | 9,168 | 100.0% | 100.0% | 0 | n/a (no novel rows at all) |

Search protocol, identical everywhere and fixed before any result was seen:
MMseqs2 `easy-search`, nucleotide `--search-type 3` and translated
`--search-type 2 -s 7.5`, `--min-seq-id 0.30 -c 0.10 --cov-mode 2 -e 1e-3`, both
strands. These are a *detection floor*, not a cleaning threshold: nothing
downstream is tuned on them, and **no threshold was relaxed at any point to
enlarge a stratum**.

Four of nine benchmarks (both Pathogenicity, both Transmissibility) retain
**fewer than 60 rows** with no nucleotide hit, and **three or fewer** with no
translated hit. They cannot support a novelty-controlled evaluation at all.

---

## 3. The layered ladder: excess over the complete floor

Group bootstrap, 5,000-10,000 replicates, floor re-maximised inside every
replicate. Verdicts are two-one-sided tests against a **pre-declared ±0.02 MCC
margin**; `INCONCLUSIVE` means the CI spans zero *and* exceeds the margin — not
shown to differ is not shown to be equivalent.

| task | level | n | units | floor (arm) | best model | excess | 95% CI | verdict |
|---|---|---:|---:|---|---|---:|---|---|
| **GUE phage** | all | 4,630 | 4,630 | 0.8130 (cnn) | NT-v2 | +0.1067 | [+0.047, +0.121] | **SUPERIOR** |
| | nucleotide no-hit | 2,874 | 2,874 | 0.7705 (gbt) | NT-v2 | +0.1202 | [+0.049, +0.144] | **SUPERIOR** |
| | translated no-hit | **86** | 86 | 0.8044 (gbt) | NT-v2 | +0.0162 | [−0.126, +0.152] | INCONCLUSIVE |
| | protein no-hit | 79 | 79 | 0.7930 (gbt) | NT-v2 | +0.0308 | [−0.111, +0.159] | INCONCLUSIVE |
| **HT hard_70** | all | 51,187 | — | 0.6233 (nn) | HViLM | +0.0997 | [+0.055, +0.141] | **SUPERIOR** |
| | nucleotide no-hit | 11,994 | — | 0.0919 (gbt) | HViLM | +0.1935 | [+0.112, +0.265] | **SUPERIOR** |
| | translated no-hit | 3,737 | — | 0.1169 (aa) | HViLM | +0.1083 | [−0.031, +0.235] | INCONCLUSIVE |
| | protein no-hit | 3,634 | — | 0.0962 (aa) | HViLM | +0.1216 | [−0.018, +0.246] | INCONCLUSIVE |
| **HT std_95** | all | 50,817 | — | 0.6185 (nn) | HViLM | −0.0589 | [−0.080, −0.038] | **INFERIOR** |
| | nucleotide no-hit | 11,124 | — | 0.1508 (cnn) | HViLM | +0.1826 | [+0.081, +0.262] | **SUPERIOR** |
| | translated no-hit | 3,882 | — | 0.1913 (cnn) | HViLM | +0.1074 | [−0.048, +0.248] | INCONCLUSIVE |
| | protein no-hit | 3,810 | — | 0.1924 (cnn) | HViLM | +0.1061 | [−0.053, +0.239] | INCONCLUSIVE |
| **VHDB** | all | 3,754 | **349** | 0.6809 (**tax**) | HViLM | +0.0227 | [−0.083, +0.131] | INCONCLUSIVE |
| | nucleotide no-hit | 1,200 | 194 | 0.1754 (tax) | HViLM | +0.3199 | [−0.052, +0.603] | INCONCLUSIVE |
| | translated no-hit | 201 | 49 | 0.3354 (tax) | HViLM | +0.3482 | [−0.237, +0.757] | INCONCLUSIVE |
| | unseen species | 199 | 48 | 0.0267 (kmer) | HViLM | +0.5669 | [−0.023, +0.815] | INCONCLUSIVE |
| | unseen genus | 74 | **14** | 0.0000 | GENA-LM | −0.0130 | [−0.053, 0.000] | INCONCLUSIVE |
| | unseen family | **0** | — | — | — | — | — | untestable |
| **Patho hard** | all | 1,792 | — | 0.7995 (cnn) | HViLM | +0.0047 | [−0.032, +0.027] | INCONCLUSIVE |
| **Patho std** | all | 6,651 | — | 0.8095 (gbt) | DNABERT-2 | +0.0006 | [−0.017, +0.012] | **EQUIVALENT** |
| **Transm hard** | all | 2,497 | — | 0.6886 (gbt) | HViLM | −0.0241 | [−0.065, +0.006] | INCONCLUSIVE |
| **Transm std** | all | 12,728 | — | 0.6934 (nn) | HViLM | −0.0006 | [−0.018, +0.011] | **EQUIVALENT** |
| **GUE covid** | all | 9,168 | — | 0.7193 (gbt) | HViLM | −0.0227 | [−0.035, −0.011] | **INFERIOR** |

**The pattern.** Four SUPERIOR results exist, all at the nucleotide level. Every
one becomes INCONCLUSIVE at the translated level. **No model is SUPERIOR to the
floor in any no-detectable-homology stratum on any task.** Two tasks return
EQUIVALENT (statistically indistinguishable from composition), two return
INFERIOR.

**Units collapse faster than rows.** VHDB goes 349 → 194 → 49 → 14 independent
genome clusters. The strata that matter most are the ones least able to resolve
anything, and that is a property of the benchmarks, not of the analysis.

---

## 4. Taxonomy recognition accounts for most of the host-tropism signal

On VHDB — the only benchmark shipping lineage labels — predicting the majority
training label of a test sequence's **genus**, backing off to species then
family, gives:

| arm | MCC |
|---|---|
| **taxonomy prior** | **0.6809** |
| NN-lookup | 0.6330 |
| CNN ladder (13 cells, 3 families) | 0.5383 |
| k-mer GBT | 0.5284 |
| AA-composition GBT | 0.4871 |
| k-mer logistic | 0.3121 |
| transformer ladder (6 cells) | 0.0959 |
| **HViLM (full fine-tuning, 3 seeds)** | **0.7036** |

HViLM's excess over the complete floor is **+0.0227, CI [−0.083, +0.131]**. On
the 74 rows whose genus is absent from training it scores **−0.0195**.

The same control cannot be run on the other eight benchmarks.

---

## 5. The authors' own released checkpoints

`duttaprat/HViLM-{Tropism,Patho,R0}`, run as shipped over our test splits. This
removes the objection that our uniform protocol under-trained HViLM.

| task | checkpoint | MCC | floor (arm) | excess |
|---|---|---:|---:|---:|
| Patho hard | HViLM-Patho | 0.9088 | 0.7962 (nn) | +0.1127 |
| HT hard_70 | HViLM-Tropism | 0.7129 | 0.6233 (nn) | +0.0896 |
| Transm hard | HViLM-R0 | 0.7613 | 0.6886 (gbt) | +0.0727 |
| Transm std | HViLM-R0 | 0.7243 | 0.6934 (nn) | +0.0309 |
| Patho std | HViLM-Patho | 0.8317 | 0.8095 (gbt) | +0.0222 |
| **HT std_95** | HViLM-Tropism | 0.4901 | 0.6185 (nn) | **−0.1284** |
| **VHDB** | HViLM-Tropism | 0.4765 | 0.6809 (tax) | **−0.2044** |

All label orientations as-shipped. The authors' own host-tropism model scores
**0.2044 below a taxonomy prior** on VHDB and below nearest-neighbour lookup on
HT std_95. Where it clears the floor, margins are +0.02 to +0.11 — and those are
**upper bounds**, because the checkpoints' training split is unauditable and may
contain our test rows.

---

## 6. Controls that could have rescued the models, and did not

**Class weighting.** The baselines are class-weighted; the fine-tuning arm used
the HuggingFace forward's unweighted cross-entropy, which handicaps models on a
3.5%-positive task and biases toward our own thesis. Re-run with matched
weighting on both host-tropism configs, 3 seeds each:

| model | hard_70 Δ | std_95 Δ |
|---|---:|---:|
| HViLM | −0.0328 | −0.0342 |
| DNABERT-2 | −0.0298 | −0.0382 |
| NT-v2 | −0.0108 | −0.0007 |
| GENA-LM | +0.0123 | −0.0171 |
| HyenaDNA | +0.0273 | −0.0208 |

Class weighting rescues nothing; HViLM drops on both. (One of three std_95 HViLM
seeds peaked at the epoch boundary, so that cell is a lower bound; it would need
to gain 0.034 to close the gap.)

**Training budget.** Best epochs are interior on every HVUE and VHDB cell. GUE
needed 16 epochs; HyenaDNA on virus_covid needed 64 (dev 0.417 at 16 → 0.599 at
64), and is reported at 64.

**Adaptation regime.** Frozen linear probes, matched at 1,500 training rows:

| task | ranking |
|---|---|
| GUE phage | **Evo 2 0.924** · NT-v2 0.792 · HViLM 0.610 · GENA-LM 0.591 · DNABERT-2 0.576 · HyenaDNA 0.573 |
| Patho hard | HViLM 0.653 · DNABERT-2 0.556 · Evo 2 0.500 · HyenaDNA 0.483 · GENA-LM 0.483 · NT-v2 0.477 |
| HT hard_70 | HViLM 0.547 · DNABERT-2 0.279 · NT-v2 0.217 · Evo 2 0.214 · GENA-LM 0.205 · HyenaDNA 0.149 |
| VHDB | HViLM 0.551 · Evo 2 0.275 · NT-v2 0.227 · DNABERT-2 0.221 · HyenaDNA 0.211 · GENA-LM 0.179 |

The fine-tuning ranking reproduces under a frozen backbone, so the discriminative
results are not fine-tuning artefacts.

---

## 7. Generative evaluation: Evo 2's edge tracks training exposure

Bits/base at 8 kb, family-capped at 3 (uncapped, ViroBench `ALL/times` is ~74%
Orthomyxoviridae, which made an earlier Markov floor read 0.80 against a true
1.95).

| stratum | interpolated Markov | Evo 2 7B base | advantage |
|---|---:|---:|---:|
| **prokaryotic** (phage — *in* Evo 2's corpus) | 1.9725 | 1.5230 | **−0.4495** |
| **eukaryotic** (deliberately *excluded* from its corpus) | 1.9745 | 1.7725 | −0.2020 |

Compression floors on 2-bit-packed DNA: xz 1.9786, zlib 1.9786, bzip2 2.0072.
Interpolated backoff Markov (EM-fitted, K=8) is the strongest floor at 1.9682.

Evo 2's advantage is **2.2× larger inside its training exposure**, and the same
split appears in the frozen probe (phage 0.924, host tropism 0.214). Masked-LM
pseudo-perplexity is reported separately and never in the same column: a
bidirectional PLL is not an autoregressive bits/base.

---

## 8. Provenance and duplication

| dataset | exact dup | revcomp dup | conflicting labels | parent-genome auditable |
|---|---:|---:|---:|---|
| **GUE covid** | **10.66%** | 0 | **248** | no |
| VHDB | 0 | 0 | — | **yes** — 0% shared cluster, 0% shared taxid |
| GUE phage | 0 | 0 | — | no |
| HVUE ×6 | 0 | 0 | — | no |

GUE virus_covid carries 10.66% exact train/test duplicates with 248 conflicting
labels — a ceiling on any achievable score and an internal inconsistency in the
labels themselves.

Eight of nine benchmarks ship no accession, so parent-genome leakage is an
**unaudited channel**. Absence of evidence, not evidence of absence.

---

## 9. What would make these benchmarks auditable

Three columns, at negligible cost to the publisher:

```
sequence, label, source_accession, taxid, split_group
```

`source_accession` makes parent-genome leakage measurable and supplies the
correct bootstrap unit. `taxid` makes the taxonomy-prior control and
genus/family-disjoint evaluation possible. `split_group` lets a reader verify
the grouping the authors used was respected. ViroBench already ships
genus-disjoint and temporal splits; ViTrace groups fragments by source sequence.
The practice exists; it is not general.

---

## 10. Limitations, stated plainly

- **"No detectable homology" is protocol-relative.** A homolog below MMseqs2's
  sensitivity sits in the no-hit stratum. That biases *toward* finding
  capability, so it does not rescue any negative here — but it caps how strongly
  a positive could ever be stated.
- **Only VHDB has a defensible bootstrap unit** (349 genome clusters, 10.8
  rows each). Every other interval resamples rows and is anti-conservative by an
  unknown factor. Flagged in every output as `ROW (anti-conservative)`.
- **The phage case is unresolved, not negative.** n=86 in the strict stratum;
  the CI contains the nucleotide-level effect. We did not relax any threshold to
  enlarge it, and we do not claim equivalence.
- **VHDB cannot test family novelty** — it is within-family by construction, and
  only 9.2% of its test genera are unseen.
- **LucaVirus is restricted** to host tropism and VHDB by design, and collapsed
  to all-negative on HVUE host tropism at every learning rate (~19 GPU-h per lr).
  Reported as a missing cell, not imputed.
- **ViroBench is excluded**: its CNN ladder was cut at ~42 h remaining and it has
  no k-mer, GBT or NN-lookup arm, so it has no floor.
- **Evo 2 enters as bits/base and a frozen probe only.** Full fine-tuning is
  protocol-incompatible; `evo2_7b` needs FP8 kernels unusable on SM 8.0.

---

## 11. Errors found and corrected during this work

Each produced a plausible number rather than an error, which is why they are
listed: the same class of failure is what the audit is about.

| defect | effect | fix |
|---|---|---|
| Transformer sweep overwrote the CNN ladder (two tag strings had drifted apart) | every arm labelled `cnn` on VHDB and HT hard_70 was the transformer; VHDB read 0.135 instead of 0.538 | one tag; `arch_family` in every output name; 12 files quarantined |
| GENA-LM loaded as stock post-LN BERT | 48 LayerNorms randomly initialised; four probes read 0.01–0.05 | load the remote pre-LN class; `_no_fresh_weights` now hard-fails |
| `--train_cap` ignored for `geneb` | covid CNN trained on 62,334 rows vs the models' 40,000 | cap applied with matched ordering and seed |
| Arms globbed while a sweep was live | mixture of one fresh and five stale seeds silently set a floor | `_arm_guard`: internal mtime consistency, not recency |
| GUE k-mer arm filed under a short tag | arm silently absent, floor dropped | required arms hard-fail; tag override in config |
| HyenaDNA truncated at 8 and 16 epochs | virus_covid dev 0.417 vs 0.599 converged | 64 epochs |
| Blended NN-lookup floor with majority fallback | forced label transfer where it is impossible; floor 0.890 → 0.649 | regime stratification; NN-lookup excluded where disabled |
| Row bootstrap on clustered data | intervals ~3.3× too narrow on VHDB | genome-cluster bootstrap where the unit exists; labelled anti-conservative where it does not |

---

## 12. The claim, stated precisely

Nucleotide-only homology audits overstate evolutionary novelty in viral
benchmarks by 65-97%. Once translated homology, lineage structure and a complete
baseline floor are controlled, no genomic language model in this study is
statistically superior to the floor in any no-detectable-homology stratum of any
of nine tasks — and on the single benchmark where a taxonomy control is
runnable, it removes most of the leading virus-specialised model's advantage,
including for the authors' own released checkpoint. The one unresolved case
(NT-v2 on phage) is inconclusive because the genuinely novel stratum contains 86
rows.

We do not claim the other eight benchmarks are taxonomy-confounded. We claim
they **cannot be checked**, and that this is the more actionable finding.
