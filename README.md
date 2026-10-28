# How much viral capability do genomic language models add over conventional baselines?

Code, frozen predictions and the reproduction path for the manuscript in [`paper/`](paper/).

Five genomic language models against up to seven conventional predictors on nine viral
classification tasks, each scored at three levels of train-to-test relatedness, with
independent training runs repeated per method so that optimisation variability enters the
intervals.

---

## What the study finds

**Neither side wins consistently on the published benchmarks.** A foundation model is ahead
on two of the nine, a conventional predictor on three, and four do not resolve. Three
different conventional methods take the lead across the nine, so no single baseline is
sufficient — though nearest-neighbour label lookup is the most frequent winner, taking five.

**Measured over viruses rather than sequence fragments, only two comparisons resolve at all,
and both are conventional-predictor wins** (host tropism `std_95` and transmissibility
`hard`). Treating each 1,000 bp window as independent overstates the evidence because many
windows come from the same virus.

**Withdrawing nucleotide-level transfer lowers every method substantially.** On the stricter
host tropism split the best conventional predictor falls 0.62 → 0.10 MCC and the selected
model 0.73 → 0.28. Much of the published performance on both sides rested on a close
labelled relative being available.

**On the filtered populations the selected model beats every conventional method that exists
there, but it does not pull further ahead.** Holding the comparator arm set fixed leaves the
margin flat or declining on every benchmark we could rescore except VHDB. The reported gap
widens because the comparator loses its homology-transfer arms, not because the model gains.

**The one result that survives everything** is HViLM on nucleotide-divergent host tropism
`hard_70`: **+0.174 MCC**, fragment `[+0.12, +0.23]`, source `[+0.07, +0.30]`, and SUPERIOR
under every hidden-dependence scenario tested.

**Nucleotide divergence is not evolutionary novelty.** A six-frame translated search finds a
relative for most of what the nucleotide filter keeps, so these results describe viruses
distant at the nucleotide level whose proteins remain recognisable.

### A methodological finding

Running three fits per method is not enough to estimate optimisation variability. Across six
benchmarks the ratio of the ten-fit to the three-fit standard deviation spans **0.51× to
5.13×** with no consistent direction. At three fits two bootstrap schemes disagreed on 4 of 20
populations; at ten they agree everywhere.

---

## Layout

```
analysis/          intervals, the finalized table, the selection record    CPU, minutes
  seed_expansion/    fit expansion, symmetric check, per-seed tables
  internal/          robustness analyses not cited in the paper
paper/             manuscript, generators, the three figures it uses
  figures/           every generator; nothing writes LaTeX except these
scripts/           upstream pipeline that produced the predictions         GPU + corpora
  track_b_viral/     the viral audit proper
  track_a_benchmarks/, common/
  _not_in_manuscript/  written, run, not cited; kept for the record
reports/           derived outputs, provenance record, claim ledger
  tables/            the source of truth every number passes through
  finalization/      how the final numbers were arrived at, and from what
results/           committed per-run JSON and prediction artifacts
tests/             82 tests, no GPU, no corpora
```

Two stages, deliberately separated. **Stage 1** (`scripts/`) fine-tunes the models and runs
the homology searches; it needs the benchmark corpora, MMseqs2 and a GPU. **Stage 2**
(`analysis/` and `paper/`) turns the frozen predictions into every number in the manuscript
and needs neither.

Stage 2 is what this repository is for. Stage 1 is included as the record of how the
predictions were made, not as a one-command rebuild.

---

## Reproducing the manuscript

### 1. Environment

Python 3.10+ with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`. Nothing else is
needed for stage 2.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`pdflatex` is needed only to compile the PDF. The manuscript includes figures without a file
extension, so LaTeX resolves the committed `.pdf`; the `.png` proofs beside them are
gitignored.

### 2. Point at the predictions

The prediction archive is **not** in this repository: it is roughly 20 GB of per-example
arrays. `VB_RESULTS` must point at a directory holding `hidden_masks/`, `preds_rescore/`,
`geneb_finetune/` and `clusters/`.

```bash
export VB_RESULTS=/path/to/results_viral_bench
```

Two further roots are read only by `make_genus_concentration.py`, which regenerates an
artifact that is already committed, so most readers can leave them unset:

| variable | holds | default |
|---|---|---|
| `VB_RESULTS` | the prediction archive | the authoring cluster's path |
| `VB_REPORTS` | upstream `taxonomy_recovery/` | the authoring cluster's path |
| `VB_CORPORA` | `hvue_v2/`, `vhdb_tropism/`, `gue_devcarve/` | the authoring cluster's path |

> **Archive availability.** The archive is not yet deposited. Until it is, the committed
> tables under `reports/tables/` and `reports/finalization/` are the reproducible record:
> every number in the manuscript is derivable from them, and step 4 below verifies that it
> still is, without the archive.

### 3. Run it

```bash
# intervals: 5,000-replicate two-level bootstrap over 20 scored populations
python analysis/bootstrap_corrected.py --reps 5000 --workers 16

# symmetric check: both sides reselected inside each replicate
python analysis/seed_expansion/regen_symmetric.py

# hidden-dependence sensitivity: 20 pseudo-cluster partitions, pooled
python scripts/track_b_viral/unassigned_clustering_sensitivity.py --partitions 20 --reps 5000

# source of truth -> finalized table -> figures -> tables -> supplement -> ledger -> LaTeX
python paper/figures/build_paper.py          # --no-pdf to stop before LaTeX
```

`build_paper.py` runs the whole chain, including `analysis/build_finalized_sot.py`. It injects
the generated table into `main.tex` between its marker comments and rewrites
`paper/supplementary.tex` in full.

Two scripts are deliberately **not** in that chain, because both record something the chain
would otherwise overwrite. `analysis/make_selection_artifact.py` froze the three-seed
development scores that made the model selection; re-running it is only meaningful on a tree
that still holds the original three-seed finetune JSONs, and it refuses to write a
mixed-fit-count table. `analysis/make_fit_counts.py` records how many fits each arm actually
has. Both outputs are committed under `reports/tables/`.

### 4. Check it

```bash
python -m pytest tests/                   # 82 tests
python paper/figures/audit_consistency.py # prose and tables against the data
```

`audit_consistency.py` re-checks 41 asserted claims against the source of truth, holds the
Methods fit counts to `reports/tables/fit_counts.csv`, verifies every table row against its
column spec, checks that the stated supplementary range covers the tables that exist, and
enforces the vocabulary and punctuation rules. Run it after editing prose. Neither command
needs `VB_RESULTS`.

---

## How numbers reach the manuscript

```
frozen predictions
  -> analysis/bootstrap_corrected.py                 two-level and conditional intervals
  -> analysis/seed_expansion/regen_symmetric.py      symmetric check
  -> analysis/build_finalized_sot.py                 merge + per-cell provenance
  -> reports/tables/final_source_of_truth_finalized.csv
  -> paper/figures/make_*                            tables, figures, supplement
```

Everything downstream reads the finalized table through `paper/figures/sot_path.py`. A number
cannot enter the manuscript without passing through it, and every generator fails on a
missing or empty input rather than emitting an empty table.

Generated LaTeX is injected between marker comments, never kept as a separate file:

```latex
% <<< GENERATED table_headtohead (do not edit by hand)
...
% >>> GENERATED table_headtohead
```

`paper/supplementary.tex` is written whole by `make_supplement_v2.py`. Edit the generator, not
the file; a test enforces the banner that says so.

**Three rules the pipeline learned the hard way.**

*Point estimates and intervals must come from the same run.* An earlier version took point
estimates from a frozen table and intervals from the bootstrap, producing rows whose printed
model minus baseline did not equal the printed difference. `build_finalized_sot.py` now
sources all of them from `bootstrap_corrected.csv`.

*A quantity that predates a rerun must not be recomputed after it.* Model selection was
pre-committed on three seeds; the selected arms were later refit ten times, overwriting their
finetune JSONs. Recomputing development MCC now would print a ten-fit number as the basis of a
three-seed decision. `reports/tables/selection_3seed.csv` freezes the original values, lifted
from the pre-finalization table, and the supplement reads it rather than the tree.

*The same number must be rounded once.* Table 1 and Supplementary S2 once printed 0.282 and
0.283 for one value, because one path parsed it from CSV at four decimals and the other
computed it live at full precision. The supplement now takes the selected model's cell from
the source of truth.

---

## Estimator

A method's score on a population is the **mean MCC across its fits**; the comparator is the
maximum of that quantity over the applicable arms. Intervals are 5,000-replicate percentile
intervals from a **two-level bootstrap** that resamples test units and, within each method,
its observed fits, with the comparator re-maximised inside every replicate.

This matters more than it looks. A variant that draws a single random fit per replicate
estimates a different quantity, and because the comparator is a maximum over arms, a single
lucky draw inflates it. Two code paths implementing what we took to be the same statistic
differed by **0.065 MCC** on one interval — invisible at three fits, material at ten. The
helpers are now imported from one module so the paths cannot drift again.

`SUPERIOR` and `INFERIOR` mean the interval excludes zero. Everything else is `INCONCLUSIVE`.
There is no equivalence verdict, because declaring equivalence needs a practical-irrelevance
threshold we have no external basis to set.

---

## Fit counts and known gaps

Seven of the nine benchmarks carry ten fits on the selected model and on every stochastic
conventional arm. Two do not, and the manuscript states both exceptions in Methods. The counts
are not a supplementary table; they are recorded per arm and population in
[`reports/tables/fit_counts.csv`](reports/tables/fit_counts.csv), written by
`analysis/make_fit_counts.py`, and `audit_consistency.py` holds the Methods prose to that file
so the two cannot drift.

| gap | status |
|---|---|
| **VHDB** | corpus no longer available, so the benchmark keeps its original counts throughout: three fits on the model arm, six on each neural ladder, one on each gradient-boosted arm. It also carries the one gain that survives a fixed arm set, so the weakest-evidenced positive in the study is the one measured over the fewest fits. Its source intervals are carried forward rather than recomputed. |
| **GUE covid, model arm** | the frozen run's epoch count is not recorded, so a refit would not reproduce the selected configuration and the arm retains three fits. Its ladder and GBT arms are at ten. |
| **non-selected foundation models** | not refit at all. They keep the three fits used for selection, enter no reported interval, and appear only in Supplementary Table S2. |
| **`--max-seqs 20`** | the homology search caps the prefilter below the MMseqs2 default of 300, so a more sensitive search would find relatives this protocol does not. On HT `hard_70` the cap changes 2 rows of 11,994. See `reports/finalization/maxseqs_sensitivity_status.md`. |

`reports/finalization/` carries the provenance record: the estimator written out and
re-derived, the before/after diff of every generated row, the three-fit intervals the ten-fit
run is compared against, the claim ledger, and `FINAL_DECISION.md`. Each cell of the finalized
table carries `provenance_frag` and `provenance_src`:

| value | meaning |
|---|---|
| `REGENERATED` | recomputed here from the frozen predictions |
| `FROZEN_VERIFIED_INPUT` | carried from the original run, inputs verified present |
| `PRESENT_BUT_NOT_REPRODUCED` | carried forward, could not be recomputed |

---

## Where to look for what

| question | file |
|---|---|
| every main-text claim and the file licensing it | [`reports/CLAIM_LEDGER.md`](reports/CLAIM_LEDGER.md) |
| why the final numbers are what they are | [`reports/finalization/FINAL_DECISION.md`](reports/finalization/FINAL_DECISION.md) |
| the estimator, written out | [`reports/finalization/bootstrap_estimand.md`](reports/finalization/bootstrap_estimand.md) |
| what changed when the intervals were finalized | [`reports/finalization/sot_before_after_diff.csv`](reports/finalization/sot_before_after_diff.csv) |
| per-arm fit counts | [`reports/tables/fit_counts.csv`](reports/tables/fit_counts.csv) |

Two longer documents, [`reports/RESULTS.md`](reports/RESULTS.md) and
[`results/README.md`](results/README.md), are the working record from while the analysis was
running. They cover work that did not reach the paper and use vocabulary it has since
dropped. Each carries a banner saying so. Read the manuscript and the claim ledger first;
where they and the working record disagree, the manuscript is correct.

---

## License

See [LICENSE](LICENSE).
