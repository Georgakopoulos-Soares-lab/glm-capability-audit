# Manuscript briefing

For an agent optimizing `paper/main.tex`. Everything here is measured and
reproduced by `python pipelines/run.py all`. Per-task JSON is in `reports/`.

Read this before changing any claim. Several obvious-sounding edits would make
the paper wrong in ways a reviewer will catch.

---

## 1. What the paper is

An audit of nine published viral classification benchmarks, asking whether the
capability claims made on them survive a complete baseline floor and a nested
removal of evolutionary familiarity.

The contribution is **the audit and what it reveals about the benchmarks**, not a
verdict on any model. Every component has precedent (homology-aware
partitioning, overlap spectra, memorization diagnostics, tuned supervised
baselines, sequence-neighbour transfer). What is new is assembling them into one
nested audit, adding the **translated-homology layer**, and turning it on
published viral capability claims.

**Do not upgrade this into "genomic language models cannot do viral tasks."**
The data does not support that and the paper does not claim it.

---

## 2. What was tested

**Nine benchmarks.** HVUE host tropism (`hard_70`, `std_95`), pathogenicity
(hard, std), transmissibility (hard, std); VHDB within-family host tropism
(built here, from Virus-Host DB); GUE phage fragments and GUE virus_covid.

**Six models.** HViLM (virus-specialized), DNABERT-2, NT-v2 500M, GENA-LM,
HyenaDNA, LucaVirus (restricted). Evo 2 by frozen probe and bits/base only.

**Seven floor arms.** k-mer logistic, k-mer GBT, AA-composition GBT, CNN ladder
(13 cells, 3 families, 0.04M to 9.4M params), from-scratch transformer ladder
(6 cells), nearest-neighbour lookup, taxonomy prior.

**Three regimes.** Full fine-tuning (primary, 3 seeds, lr grid, dev-selected),
frozen linear probe (secondary), generative bits/base.

**Five strata.** all rows, no nucleotide hit, no translated hit, no protein hit,
unseen species, unseen genus, unseen family.

---

## 3. The results, and exactly how far each one reaches

### 3.1 Translated homology (the strongest result, all 9 benchmarks)

Six-frame translated search finds homology in **65 to 97 percent** of the test
sequences a nucleotide search calls novel. This needs no model and no floor. It
holds on every benchmark.

### 3.2 Testability (all 9 benchmarks)

After that correction, sequences with no detectable homology to their own
training split:

| benchmark | rows | usable |
|---|---:|---|
| HVUE HT std_95 | 3,882 | yes |
| HVUE HT hard_70 | 3,737 | yes |
| VHDB | 201 (49 genome units) | yes, thin |
| GUE phage | 86 | no |
| HVUE Transm hard | 3 | no |
| HVUE Patho std | 3 | no |
| HVUE Transm std | 3 | no |
| HVUE Patho hard | 2 | no |
| GUE covid | 0 | no |

**Six of nine cannot test generalization against any baseline, however strong.**

### 3.3 Models vs the floor (only 3 benchmarks can carry this)

On HT hard_70, HT std_95 and VHDB, no model beats the floor once translated
homology is removed. Whole-benchmark verdicts across all nine:

| benchmark | floor (arm) | best model | excess | verdict |
|---|---:|---|---:|---|
| GUE phage | 0.813 (cnn) | NT-v2 | +0.107 | superior |
| HVUE HT hard_70 | 0.623 (nn) | HViLM | +0.100 | superior |
| VHDB | 0.681 (**tax**) | HViLM | +0.023 | inconclusive |
| HVUE Patho hard | 0.799 (cnn) | HViLM | +0.005 | inconclusive |
| HVUE Patho std | 0.809 (gbt) | DNABERT-2 | +0.001 | equivalent |
| HVUE Transm std | 0.693 (nn) | HViLM | −0.001 | equivalent |
| HVUE Transm hard | 0.689 (gbt) | HViLM | −0.024 | inconclusive |
| GUE covid | 0.719 (gbt) | HViLM | −0.023 | inferior |
| HVUE HT std_95 | 0.619 (nn) | HViLM | −0.059 | inferior |

Across 126 model-by-stratum tests: **5 superior, 2 equivalent, 49 inferior,
70 inconclusive**. All 5 superior verdicts sit in strata that still contain
translated homology. **Zero superior verdicts in any no-translated-hit or
deeper stratum.** That last sentence is the paper's central negative and it is
exact.

### 3.4 Lineage (1 benchmark only)

On VHDB, a genus-majority prior reaches **0.681 MCC** against HViLM's **0.704**.
It beats NN-lookup (0.633), the CNN (0.538) and every composition arm. On the 74
sequences whose genus is unseen, HViLM falls to **−0.020**.

**This is one dataset.** The paper must not imply the other eight are lineage
confounded. It claims they *cannot be checked*, which is different and stronger.

### 3.5 Author checkpoints (control)

`duttaprat/HViLM-{Tropism,Patho,R0}` run as shipped. On VHDB the host-tropism
checkpoint scores **0.204 below the taxonomy prior**; on HT std_95 it is below
NN-lookup. Where it clears the floor, margins are +0.02 to +0.11 and are **upper
bounds**, because that training split is unauditable and may contain our test
rows. This closes the "you under-trained HViLM" objection.

### 3.6 Generative (Evo 2)

Bits/base at 8 kb vs an interpolated backoff Markov floor (1.968 in-dist). Evo 2
beats it by **0.450 on bacteriophage (inside its corpus)** and **0.202 on
eukaryotic viruses (excluded by design)**, a 2.2x difference. The same split
appears in the frozen probe.

### 3.7 Frozen probes at matched cap (1,500 rows)

| task | ranking |
|---|---|
| GUE phage | **Evo 2 0.924**, NT-v2 0.792, HViLM 0.610, GENA-LM 0.591, DNABERT-2 0.576, HyenaDNA 0.573 |
| Patho hard | HViLM 0.653, DNABERT-2 0.556, Evo 2 0.500, HyenaDNA 0.483, GENA-LM 0.483, NT-v2 0.477 |
| VHDB | HViLM 0.551, Evo 2 0.275, NT-v2 0.227, DNABERT-2 0.221, HyenaDNA 0.211, GENA-LM 0.179 |
| HT hard_70 | HViLM 0.547, DNABERT-2 0.279, NT-v2 0.217, Evo 2 0.214, GENA-LM 0.205, HyenaDNA 0.149 |

The fine-tuning ranking reproduces frozen, so the discriminative results are not
a fine-tuning artifact.

### 3.8 Provenance

GUE virus_covid: **10.66 percent** exact train/test duplicates, **248 with
conflicting labels**. VHDB: clean by construction (0 shared clusters or taxa).
The other seven ship no accession, so parent-genome leakage is unmeasurable.

---

## 4. Negative and null results (all load-bearing, do not cut)

1. **AA-composition GBT rarely matters.** It beats the nucleotide GBT on 1 of 8
   benchmarks (HT std_95, +0.034) and is otherwise slightly worse. Kept because
   it is the only protein-composition control and its irrelevance is informative.
2. **Class weighting does not rescue any model.** Matching the fine-tuning loss
   to the class-weighted baselines lowers HViLM by 0.033 and 0.034 on the two
   host-tropism configs.
3. **The transformer ladder is weak everywhere** (0.025 to 0.51). It is in the
   floor to attribute advantage to pretraining rather than architecture, and it
   essentially never sets the floor.
4. **LucaVirus collapsed** to all-negative on HVUE host tropism at every learning
   rate. Reported as a missing cell, never imputed.
5. **ViroBench excluded.** Its CNN ladder was stopped at ~42 h remaining and it
   has no k-mer, GBT or NN-lookup arm, so it has no floor.
6. **GUE phage unresolved, not negative.** 86 rows in the strict stratum; the
   interval contains the nucleotide-level effect. Equivalence is *not* claimed.
7. **VHDB cannot test family novelty** (within-family by construction; 0 percent
   of test families unseen, 9.2 percent of genera).
8. **Pseudo-perplexity is reported separately** and never mixed with
   autoregressive bits/base.

---

## 5. Claims that would be wrong if strengthened

| tempting phrasing | why it is wrong |
|---|---|
| "no model beats the floor on any of nine tasks" | six have no testable stratum; it is three |
| "genomic LMs lack viral capability" | not measured; the benchmarks cannot resolve it |
| "the benchmarks are lineage confounded" | measured on one; the other eight are unauditable |
| "NT-v2's phage advantage is leakage" | inconclusive at n=86, not negative |
| "translated search proves the sequences are homologous" | it proves homology is *detectable*; absence is protocol-relative |
| "the taxonomy prior explains 97 percent of HViLM" | a ratio of two MCCs is not a share; say "0.681 against 0.704" |
| "human tropism cannot be benchmarked" | not shown; none of the ten examined has a stratum meeting the three conditions, and whether that is fundamental is open |
| "ViroBench is a good benchmark" | the finding is about the split rule, not the benchmark; its genus splits carry the most concentrated test sets measured here |
| "ViroBench's clean strata show models do or do not generalise" | no model was run on them, deliberately; they are well constructed and underpowered |
| "the strata are confounded by genus" | measured as composition of the positive class; it limits what a stratum can ask, it is not a claim about any model |

---

## 6. Style constraints already applied, keep them

- **No em dashes, en dashes or colons in prose.** Ranges read "65 to 97 percent".
  Hyphens survive only inside technical terms (k-mer) and literal flags.
- **Section titles are plain**: "How much of each benchmark is novel", "Models
  against the baseline floor", "The effect of viral lineage".
- **Figure captions describe, they do not conclude.** A caption says what the
  reader is looking at; the text says what it means.
- **Confidence intervals live in the tables**, not in sentences. Prose gives the
  number and the verdict word and points at Supplementary Table S3.
- **Vocabulary**: *lookup explainable*, not "memorization". *Lookup unavailable
  under a stated protocol*, not "memorization impossible". Every novelty claim
  names the search that produced it.

---

## 7. Known weaknesses a reviewer will find

1. **The lineage result is one dataset**, and one we built. Stated in the text.
2. **Two of the three testable benchmarks have no bootstrap unit**, so their
   intervals are anti-conservative by an unknown factor. Flagged in every output.
3. **The 200-row testability threshold was chosen after seeing the data.** The
   natural gap is between VHDB's 201 and phage's 86, so it is not arbitrary, but
   it is not pre-registered either. The counts are also reported raw.
4. **"No detectable homology" is protocol-relative.** Biases toward finding
   capability, so it cannot rescue a negative, but it caps any positive.
5. **VHDB's clean stratum is 49 independent units.** Wide intervals follow.

---

## 8. Display items

**Status: the rewrite is done.** `paper/main.tex` carries the amended Results
sections 3, 5 and 6, the specification ending, and the updated abstract,
introduction, limitations and methods. Regenerate every table with
`make_arm_decay_table.py`, `make_virobench_table.py`, `make_mde_calibration.py`,
`make_genus_concentration.py`, `explain_genus_floor.py`, `virobench_power.py`,
then `make_supplementary.py`. No LaTeX on this host, so the document has not been
compiled; it was checked by inspection for balanced braces, label collisions,
reference resolution and table column counts.

Final numbering. Main text has four figures and three tables, Table 1 arm decay,
Table 2 ViroBench splits, Table 3 MDE calibration. Supplement runs S1 to S10,
where S7 is genus concentration, S8 the floor explanation, S9 the ViroBench
detectable effect and S10 the full arm decay grid.

Four figures, one main-text table, and the supplementary tables. Figures 2 to 4
replaced dense tables; the full numbers are in the supplement, generated from
JSON by `paper/figures/make_supplementary.py`.

- Fig 1 `fig1_pipeline.pdf`, schematic, no data
- Fig 2 `fig2_novelty.pdf`, survival per stratum, 9 benchmarks, log scale
- Fig 3 `fig3_ladder.pdf`, forest plot, excess with CI and verdict
- Fig 4 `fig4_lineage.pdf`, VHDB floor decomposition; Evo 2 exposure split
- **Table 1** `tables/arm_decay_maintext.tex`, every baseline arm at all rows and
  in the strictest stratum each benchmark retains. Promoted from the supplement
  on 22 September 2026. It is the cleanest evidence that homology filtering is
  not selectively removing the rows a lookup could solve: the CNN and
  transformer arms are supervised on the same task as the models and fall with
  the composition arms rather than holding. Put it in the section that argues
  the floor collapse is general, `\input{tables/arm_decay_maintext}`, label
  `tab:arm-decay`. The full per-stratum grid stays in the supplement as
  `tables/arm_decay_full.tex`.

The ViroBench audit is a display item in its own right:

- `tables/virobench_audit.{txt,json,tex}` from `make_virobench_table.py`, reading
  masks persisted by `virobench_homology_audit.py`. Six splits of one contemporary
  benchmark, three taxonomic and three temporal, built from the same taxon pool by
  the same group. It carries the constructive ending: the two large temporal splits
  yield strata that are homology free, genus disjoint and genus diverse at once,
  which no stratum in the nine audited benchmarks achieves. The three taxonomic
  splits do not, and two of them have the most concentrated full test sets measured
  anywhere here. Do not state this as ViroBench being a good or bad benchmark; state
  it as the split rule being the thing that differs.

Two diagnostics feed the supplement and are generated separately:

- `tables/genus_concentration.{txt,json}` from `make_genus_concentration.py`,
  the number of genera contributing positive rows and the largest genus's share
  of the positive class, for every stratum in the ladder, plus GC of positives
  against negatives where concentration is at least 50 percent. Native labels
  where the benchmark ships them, recovered otherwise, never pooled. This goes
  in the supplement regardless of what it shows.
- `tables/genus_floor_explanation.{txt,json}` from `explain_genus_floor.py`,
  why the unseen-genus verdict differs between the two host tropism
  configurations.

Colour was validated with `paper/figures/validate_palette.py` (a port of the
dataviz validator; no node on the host). Green/red for verdicts fails CVD
separation at dE=4.1, so verdicts use the diverging blue/red pair at dE=23.8
plus marker shape plus a text column.

---

## 9. Pipeline defects found during the work

Each produced a plausible number rather than an error. Worth a supplementary
note; they are direct evidence the audit method catches what it claims to.

| defect | effect |
|---|---|
| Transformer sweep overwrote the CNN ladder (two tag strings drifted apart) | every arm labelled `cnn` on two benchmarks was the transformer; VHDB read 0.135 instead of 0.538 |
| GENA-LM loaded as stock post-LN BERT | 48 LayerNorms randomly initialised; four probes read 0.01 to 0.05 instead of 0.23 to 0.62 |
| `--train_cap` ignored for one loader | covid CNN trained on 62,334 rows vs the models' 40,000 |
| Arms globbed while a sweep was live | one fresh seed mixed with five stale ones silently set a floor |
| NN-lookup filed under a differently cased tag | arm silently absent from both GUE floors |
| HyenaDNA truncated at 8 and 16 epochs | virus_covid 0.417 vs 0.599 converged |
| Blended NN-lookup floor with majority fallback | forced label transfer where impossible; floor 0.890 to 0.649 |
| Row bootstrap on clustered data | intervals ~3.3x too narrow on VHDB |
| Blanket rename of stratum names to L0..L5 caught ViroBench's own split keys | its generator raised KeyError on `L0`; ViroBench's splits are its own, not this ladder |
| The same rename rewrote a mask **filename** and an array key inside an `.npz` | the host tropism L5 population silently vanished from the genus table, leaving em dashes in the capability matrix that the Molluscipoxvirus argument rests on |
| A generator kept reading `paper/tables/` after that folder was removed | genus columns emptied silently rather than failing; it now raises if the input is absent |
| Syncing layered JSON from the repo to the working tree | overwrote the only copies holding the host tropism L5 populations. `/home/nvidia/genomic-capability-unlearning` is a symlink to `/data`, so there was no second copy; recovered by re-running the bootstrap |

---

## 10. Where things are

```
paper/main.tex                  manuscript
paper/supplementary.tex         S1 to S6, generated
paper/figures/make_figures.py   figures, regenerate with one command
paper/figures/make_supplementary.py
reports/RESULTS.md              full analytical write-up, all numbers
reports/layered/*.json          per-task ladder, CIs, verdicts
reports/hidden_homology/*.json  per-task search results
reports/frozen_probe/*.json     probe results
reports/author_checkpoints/*.json
reports/leakage/*.json          duplicates and provenance
pipelines/run.py                one entry point, 12 stages
```
