# glm-capability-audit

Code and frozen predictions for *How much viral capability do genomic language models add
over conventional baselines?* ([`paper/main.tex`](paper/main.tex)).

Five genomic language models against up to seven conventional predictors, nine viral
classification tasks, three levels of train-to-test relatedness.

The work splits into two stages that share nothing but a directory of predictions.

| stage | does | needs | cost |
|---|---|---|---|
| **1 — pipeline** (`scripts/`, `pipelines/`) | fine-tunes the models, fits the baselines, runs the homology searches | GPU, benchmark corpora, MMseqs2 | days |
| **2 — analysis** (`analysis/`, `paper/`) | turns the frozen predictions into every number, table and figure in the manuscript | CPU only | minutes |

Stage 2 is what most readers want. Stage 1 is the record of how the predictions were made.

---

## 1. Environment

### Stage 2 only (reproduce the manuscript)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pytest`. Python 3.10 and 3.12
verified. `pdflatex` is needed only to compile the PDF.

### Stage 1 as well (rerun the pipeline)

```bash
conda env create -f environment.yml && conda activate glm-audit
# or, with mmseqs2 already on PATH:
pip install -r requirements.txt -r requirements-pipeline.txt
```

Adds `torch>=2.1`, `transformers>=4.40,<4.48` (DNABERT-2 breaks above 4.48), `peft`,
`huggingface_hub`, `pyarrow`. **MMseqs2 is not installable from PyPI** — use conda-forge /
bioconda or a release binary.

### Environment variables

Stage 2 needs one. Stage 1 needs the rest; copy
[`pipelines/env.example.sh`](pipelines/env.example.sh) to `pipelines/env.sh`, edit, and
`source` it.

| variable | holds | used by |
|---|---|---|
| `VB_RESULTS` | prediction archive: `hidden_masks/`, `preds_rescore/`, `geneb_finetune/`, `clusters/` | stage 2 |
| `VB_REPORTS` | upstream `taxonomy_recovery/` | `make_genus_concentration.py` |
| `VB_CORPORA` | `hvue_v2/`, `vhdb_tropism/`, `gue_devcarve/` | `make_genus_concentration.py` |
| `VB_DATA` | parent of the benchmark corpora | stage 1 |
| `VB_VIRAL_OUT`, `VB_PREDS` | where the pipeline writes predictions and masks | stage 1 |
| `VB_SCRATCH` | MMseqs2 temp, needs tens of GB | stage 1 |
| `VB_MMSEQS` | path to the `mmseqs` binary | stage 1 |
| `VB_ENV_*` | per-model interpreters; the models need mutually incompatible deps | stage 1 |

> The prediction archive is **not** in this repository and is not yet deposited. Until it is,
> the committed tables under `reports/tables/` and `reports/finalization/` are the
> reproducible record, and the verification in section 5 runs without the archive.

---

## 2. The paper

`paper/main.tex` includes `paper/supplementary.tex` and is built entirely by generators — one
table is injected between marker comments, the supplement is rewritten whole.

| in the paper | label | written by |
|---|---|---|
| Table 1, model vs best baseline | `tab:headtohead` | `make_main_tables.py`, injected into `main.tex` |
| Figure 1, MCC difference at L0 and L1 | `fig:performance` | `make_main_figures.py:fig1` |
| Figure 2, before and after filtering | `fig:collapse` | `make_main_figures.py:fig2` |
| Figure 3, sequences retained per level | `fig:filters` | `make_main_figures.py:fig3` |
| S1 model selection | `tab:selection` | `make_supplement_v2.py:s2` |
| S2 all models and all arms | `tab:allarms` | `make_supplement_v2.py:s1` |
| S3 bootstrap schemes at 3 and 10 fits | `tab:bootstrap` | `make_supplement_v2.py:s6` |
| S4 sequence vs source resampling | `tab:source` | `make_supplement_v2.py:s3` |
| S5 hidden-dependence sensitivity | `tab:hidden` | `make_supplement_v2.py:s4` |

Function names do not track the printed numbers; labels do, and the text cites labels only.

**Never hand-edit `paper/supplementary.tex` or the injected block** — both are overwritten on
the next build. Change the generator. A test enforces the banner that says so.

```latex
% <<< GENERATED table_headtohead (do not edit by hand)
...
% >>> GENERATED table_headtohead
```

Figures are included without a file extension so LaTeX resolves the committed `.pdf`; the
`.png` proofs beside them are gitignored.

---

## 3. Running stage 2, rebuild the manuscript

```bash
export VB_RESULTS=/path/to/results_viral_bench
```

| step | command | writes |
|---|---|---|
| intervals | `python analysis/bootstrap_corrected.py --reps 5000 --workers 16` | `reports/finalization/bootstrap_corrected.csv` |
| symmetric check | `python analysis/seed_expansion/regen_symmetric.py` | `reports/finalization/symmetric_regenerated.csv` |
| hidden dependence | `python scripts/track_b_viral/unassigned_clustering_sensitivity.py --partitions 20 --reps 5000` | `reports/layered/unassigned_sensitivity.json` |
| everything downstream | `python paper/figures/build_paper.py` | tex, figures, PDF |

`build_paper.py` runs the chain in dependency order and is the only command most people need:

```
make_source_of_truth.py              consolidate the frozen analyses
analysis/build_finalized_sot.py      merge intervals + per-cell provenance
make_main_figures.py                 fig1-3, .pdf and .png
make_main_tables.py                  Table 1
make_supplement_v2.py                supplementary.tex, all five tables
make_claim_ledger.py                 reports/CLAIM_LEDGER.md
-> inject -> pdflatex x3 + bibtex
```

`--no-pdf` stops before LaTeX. Two consecutive runs produce byte-identical `.tex`.

Everything downstream of the finalized table reads it through `paper/figures/sot_path.py`; a
number cannot enter the manuscript without passing through
`reports/tables/final_source_of_truth_finalized.csv`.

### Run-once recorders, deliberately outside the chain

Both record something a rerun would destroy. Outputs are committed.

| command | writes | why it is not in the chain |
|---|---|---|
| `python analysis/make_selection_artifact.py` | `reports/tables/selection_3seed.csv` | selection was pre-committed on three seeds; the selected arms were later refit ten times and their JSONs overwritten, so recomputing prints a ten-fit mean as the basis of a three-seed decision |
| `python analysis/make_fit_counts.py` | `reports/tables/fit_counts.csv` | the per-arm fit counts Methods states in prose; `audit_consistency.py` holds the prose to this file |

---

## 4. Running stage 1, regenerate the predictions

One entry point. Stages run in dependency order and each is a legal standalone target.

```bash
cp pipelines/env.example.sh pipelines/env.sh && $EDITOR pipelines/env.sh
source pipelines/env.sh                                 # env.sh is gitignored; env.example.sh is the template

python pipelines/run.py --list                          # tasks, models, arms configured
python pipelines/run.py baselines --dry-run             # print commands, run nothing
python pipelines/run.py finetune --tasks gue_phage --models nt_v2_500m
python pipelines/run.py homology regime --tasks paper
python pipelines/run.py all --gpus 0,1,2,3
```

| stage | does |
|---|---|
| `homology` | MMseqs2 train-vs-test search, giving the NN-lookup arm and the cached relatedness masks |
| `baselines` | k-mer logistic, k-mer GBT, AA-composition GBT, CNN ladder, from-scratch transformer ladder |
| `finetune` | full fine-tuning, the primary regime, identical budget per model |
| `probe` | frozen linear probe, secondary regime |
| `generative` | bits per base and pseudo-perplexity, with their floors |
| `hidden` | nested homology audit, nucleotide then six-frame translated |
| `layered` | graded relatedness curve and group bootstrap |
| `authorckpt` | the authors' released task checkpoints as a labelled contrast |
| `provenance` | duplicate, reverse-complement, conflicting-label and parent-genome audit |
| `regime` | relatedness stratification and bootstrap intervals |
| `grid` | identity by coverage threshold sensitivity |
| `table` | assemble the tables from everything above |

**Tasks** (`pipelines/config/tasks.yaml`): `vhdb`, `ht_hard_70`, `ht_std_95`, `patho_hard`,
`patho_std`, `transm_hard`, `transm_std`, `gue_phage`, `gue_covid`, `virobench`,
`viral_vs_host`. `--tasks paper` selects the ones in the manuscript.

**Models** (`pipelines/config/models.yaml`): `hvilm`, `dnabert2`, `nt_v2_500m`, `gena_lm`,
`hyenadna`, `lucavirus`, `evo2`.

Nothing is overwritten: a stage whose output exists is skipped unless `--force`, and `--force`
writes under a distinct tag.

### Individual methods

The pipeline shells out to these; they also run standalone.

| arm or step | script |
|---|---|
| k-mer logistic, k-mer GBT | `scripts/track_b_viral/hvue_v2_kmer.py`, `kmer_gbt_baseline.py` |
| amino-acid composition GBT | `scripts/track_b_viral/aa_composition_baseline.py` |
| CNN and transformer ladders | `scripts/common/capacity_sweep.py` |
| nearest-neighbour lookup | `scripts/track_b_viral/nn_lookup_as_baseline.py` |
| taxonomy prior | `scripts/track_b_viral/taxonomy_prior_baseline.py` |
| gLM fine-tuning | `scripts/track_a_benchmarks/geneb_finetune.py` |
| relatedness masks, L1 and L2 | `scripts/track_b_viral/hidden_homology_audit.py` |
| source-genome recovery | `scripts/track_b_viral/assign_source_genome.py` |
| VHDB task construction | `scripts/track_b_viral/build_vhdb_tropism.py` |

`scripts/_not_in_manuscript/` holds code that was written and run but is not cited.

---

## 5. Verification

Neither command needs `VB_RESULTS`.

```bash
python -m pytest tests/                   # 87 tests
python paper/figures/audit_consistency.py # 41 claims, plus structural checks
```

`audit_consistency.py` checks every number the prose stakes a claim on against the source of
truth, holds the Methods fit counts to `reports/tables/fit_counts.csv`, verifies every table
row against its column spec, checks that the stated supplementary range covers the tables
that exist, and enforces the vocabulary and punctuation rules. Run it after editing prose.

`tests/test_reproduction_path.py` checks the reproduction path against the git **index**, not
the working tree, so a file that exists only on your disk fails the suite.

---

## 6. Layout

```
analysis/            intervals, finalized table, run-once recorders
  seed_expansion/      fit expansion, symmetric check, per-seed tables
  internal/            robustness analyses not cited
paper/
  main.tex             the manuscript, includes supplementary.tex
  supplementary.tex    GENERATED, do not edit
  figures/             every generator, plus fig1-3 as .pdf
pipelines/           stage-1 entry point, scheduler, task/model/protocol config
scripts/
  track_b_viral/       the viral audit proper
  track_a_benchmarks/  fine-tuning and aggregation
  common/              shared paths, capacity sweep, split audits
  _not_in_manuscript/  run, not cited
reports/
  tables/              source of truth, selection record, fit counts
  finalization/        provenance, how the final numbers were arrived at
  layered/, regime/    per-analysis outputs
results/             committed per-run JSON and prediction artifacts
tests/
```

---

## 7. Known gaps

| gap | status |
|---|---|
| **VHDB fit counts** | corpus no longer available, so the benchmark keeps its original counts: three fits on the model arm, six on each neural ladder, one on each gradient-boosted arm. Its source intervals are carried forward, not recomputed. |
| **GUE covid, model arm** | the frozen run's epoch count is unrecorded, so a refit would not reproduce the selected configuration; stays at three fits. Its ladder and GBT arms are at ten. |
| **non-selected gLMs** | not refit. They keep the three fits used for selection and enter no reported interval. |
| **`--max-seqs 20`** | the homology search caps the prefilter below the MMseqs2 default of 300. On HT `hard_70` the cap changes 2 rows of 11,994. See `reports/finalization/maxseqs_sensitivity_status.md`. |
| **prediction archive** | about 20 GB, not deposited yet. |

Per-cell provenance is carried in the finalized table as `provenance_frag` and
`provenance_src`: `REGENERATED` (recomputed here), `FROZEN_VERIFIED_INPUT` (carried, inputs
verified present), `PRESENT_BUT_NOT_REPRODUCED` (carried, could not be recomputed).

---

## 8. Where to look

| question | file |
|---|---|
| every main-text claim and the file licensing it | [`reports/CLAIM_LEDGER.md`](reports/CLAIM_LEDGER.md) |
| why the final numbers are what they are | [`reports/finalization/FINAL_DECISION.md`](reports/finalization/FINAL_DECISION.md) |
| the estimator, written out | [`reports/finalization/bootstrap_estimand.md`](reports/finalization/bootstrap_estimand.md) |
| what changed when the intervals were finalized | [`reports/finalization/sot_before_after_diff.csv`](reports/finalization/sot_before_after_diff.csv) |
| per-arm fit counts | [`reports/tables/fit_counts.csv`](reports/tables/fit_counts.csv) |

[`reports/RESULTS.md`](reports/RESULTS.md) and [`results/README.md`](results/README.md) are the
working record from while the analysis was running. They cover work that did not reach the
paper and use vocabulary it has since dropped; each carries a banner saying so.

---

## License

See [LICENSE](LICENSE).
