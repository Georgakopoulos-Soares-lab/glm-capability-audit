# Viral genomic language model capability audit

Code for measuring how much a genomic language model adds over conventional
sequence predictors on viral phenotype tasks, at three levels of test-to-train
relatedness.

Every benchmark is scored twice. Once for the foundation model, once for an
envelope of conventional arms (k-mer logistic regression, k-mer gradient
boosting, amino-acid composition, a CNN capacity ladder, a from-scratch
transformer ladder, nearest-neighbour label transfer, a taxonomy prior), and the
reported quantity is the difference. The same scoring is then repeated on
progressively less related subsets of the test set.

| Level | Test rows kept | What it asks |
|-------|----------------|--------------|
| **L0** | all | performance on the benchmark as published |
| **L1** | no nucleotide match to any training row | performance where nucleotide lookup cannot work |
| **L2** | no nucleotide and no six-frame translated match | performance where protein-level lookup cannot work either |

L1 and L2 are defined by the absence of an MMseqs2 hit against the supervised
training split, so they are properties of the split, not of model pretraining.

---

## Contents

| Path | What it holds |
|------|---------------|
| `pipelines/` | the single entry point, `run.py`, plus the YAML that defines tasks, models, arms and protocol |
| `scripts/common/` | shared path resolution, data download, the capacity ladder, the paired bootstrap |
| `scripts/track_a_benchmarks/` | fine-tuning and the GUE / GeneB benchmark loaders |
| `scripts/track_b_viral/` | the viral benchmarks: baseline arms, homology searches, regime splits, the level table |
| `scripts/_not_in_manuscript/` | run, but feeding no reported number; kept so the exclusion is explicit |
| `analysis/` | everything computed from the predictions: consolidation, bootstrap intervals, fit counts |
| `tests/` | configuration and estimator tests, runnable without any data |

`results/`, `reports/` and `paper/` are not tracked. They are outputs: `results/`
is written by the pipeline, `reports/` by the analysis layer, and both are
rebuilt by the commands below.

---

## 1. Environment

One environment runs everything, the pipeline and the analysis layer alike.

```bash
conda env create -f environment.yml
conda activate glm-audit

cp pipelines/env.example.sh pipelines/env.sh     # edit it, then
source pipelines/env.sh

python scripts/common/download_data.py           # the benchmark corpora
```

`pipelines/env.example.sh` is commented and names every path the pipeline needs.
Set them there; `env.sh` is gitignored.

The conda environment includes MMseqs2, which every homology search needs and pip
cannot provide. Without conda, `pip install -r requirements.txt` covers the Python
side, and MMseqs2 is installed separately.

Only fine-tuning and the capacity ladders need a GPU. The k-mer arms, the homology
searches and all of `analysis/` run on CPU.

---

## 2. How a number is produced

```
corpora
   |
   |  pipelines/run.py finetune            models, one prediction array per seed
   |  pipelines/run.py baselines homology  conventional arms, same test rows
   v
predictions                                one .npz per arm, per seed
   |
   |  pipelines/run.py hidden              MMseqs2 train-vs-test, nucleotide then six-frame
   v
relatedness masks                          per-row hasN / hasT, which define L1 and L2
   |
   |  final_regime_table.py                MCC per arm, per benchmark, per level
   |  analysis/make_source_of_truth.py     joins the levels into one table
   |  analysis/bootstrap_corrected.py      intervals on the difference
   v
final_source_of_truth_finalized.csv        one row per benchmark and level
```

Predictions are written once and scored many times. No stage re-runs a model to
produce a level: L1 and L2 are the same prediction arrays restricted to the rows
the homology search did not match.

---

## 3. Reproducing each quantity

### 3.1 L0 for the models

Full fine-tuning with an identical budget per model: three seeds across three
learning rates, eight epochs (sixteen on GUE). Selection is on development MCC;
the test set is never used for it.

```bash
python pipelines/run.py finetune                                  # every model, every task
python pipelines/run.py finetune --tasks gue_phage --models nt_v2_500m
```

Writes `geneb_finetune/<task>__<model>__lr<lr>__fullft.json` and the prediction
arrays beside it. Protocol knobs (seeds, learning rates, epochs, selection
metric) live in `pipelines/config/protocol.yaml`.

### 3.2 L0 for the baselines

```bash
python pipelines/run.py baselines     # k-mer, k-mer GBT, aa composition, taxonomy prior,
                                      # CNN ladder, from-scratch transformer ladder
python pipelines/run.py homology      # nearest-neighbour label transfer
```

`homology` is a separate stage because nearest-neighbour label transfer needs an
MMseqs2 search rather than a fit. It writes one more prediction array, on the same
test rows as every other arm, so it enters the comparator on equal terms. It is
scored at L0 only; §3.3 says why.

To run one arm on its own, bypassing the scheduler:

| Arm | Script |
|-----|--------|
| k-mer logistic regression | `scripts/track_b_viral/hvue_v2_kmer.py` |
| k-mer gradient boosting | `scripts/track_b_viral/kmer_gbt_baseline.py` |
| amino-acid composition | `scripts/track_b_viral/aa_composition_baseline.py` |
| taxonomy prior | `scripts/track_b_viral/taxonomy_prior_baseline.py` |
| nearest-neighbour lookup | `scripts/track_b_viral/nn_lookup_as_baseline.py` |
| CNN ladder | `scripts/common/capacity_sweep.py --arch_family cnn` |
| transformer ladder | `scripts/common/capacity_sweep.py --arch_family transformer` |

`python pipelines/run.py baselines --dry-run` prints the exact command for each,
with the flags the protocol fixes.

### 3.3 L1 and L2

```bash
python pipelines/run.py hidden        # nucleotide -> six-frame translated search
```

Writes `hasN` and `hasT` per test row. A nucleotide-only screen calls most of
these test sets novel when translated search does not, so every L1 and L2 number
depends on this stage having run.

Scoring every arm at all three levels:

```bash
python scripts/track_b_viral/final_regime_table.py \
    --out reports/layered/final_regime_table.json
```

One row per benchmark and level, carrying the model's MCC, each arm's MCC, the
difference and the resampling-unit provenance. Nearest-neighbour lookup is scored
at L0 only: L1 and L2 are defined by the absence of the hit it transfers from, so
it is removed from the comparator set rather than scored as zero.

### 3.4 Consolidation

Four scripts, in this order. The last one joins the other three into a single
table, and everything after this point reads only that table.

```bash
python scripts/track_b_viral/translated_relatedness_effect.py
python analysis/make_genus_concentration.py
python analysis/make_inference_provenance.py
python analysis/make_source_of_truth.py
```

Writes `reports/tables/final_source_of_truth.csv`, one row per benchmark and
level. If an input is missing, the script stops and names the command that
produces it rather than writing an empty table.

### 3.5 Bootstrap intervals

Run after §3.4, which fixes the model in each cell.

```bash
python analysis/bootstrap_corrected.py --reps 5000 --workers 16
python analysis/build_finalized_sot.py
```

Writes `reports/tables/final_source_of_truth_finalized.csv`. **This is the table
to read.**

Each cell gets two intervals. **Fragment** resamples the 1 kb test fragments and
asks whether the difference is stable across fragments of this kind. **Source**
resamples the parent genomes and asks whether it would survive a different draw
of source viruses. Both are reported, and one resolving while the other does not
is informative rather than contradictory.

Test units and training seeds are resampled together, so the interval covers the
same quantity as the point estimate. CPU only.

---

## 4. Pipeline stages

`pipelines/run.py` is the only entry point. Stages run in dependency order and
each is a legal standalone target. A stage whose output already exists is skipped
unless `--force`, and `--force` writes under a distinct tag, so nothing is
overwritten.

```bash
python pipelines/run.py --list                   # what is configured
python pipelines/run.py <stage>... [--tasks ...] [--models ...] [--dry-run]
python pipelines/run.py all --gpus 0,1,2,3
```

| Stage | What it does |
|-------|--------------|
| `homology` | MMseqs2 train-vs-test nucleotide search; produces the nearest-neighbour lookup arm |
| `baselines` | k-mer, k-mer GBT, amino-acid composition, taxonomy prior, CNN ladder, from-scratch transformer ladder |
| `finetune` | full fine-tuning, identical budget per model |
| `hidden` | nested homology audit: nucleotide, then six-frame translated; writes `hasN` and `hasT` |
| `layered` | graded relatedness curve and group bootstrap |

Every stage feeds a reported number. Analyses that were run and not reported
live under `scripts/_not_in_manuscript/`, are not wired to a stage, and are
invoked directly if you want them.

### Tasks

`vhdb`, `ht_hard_70`, `ht_std_95`, `patho_hard`, `patho_std`, `transm_hard`,
`transm_std`, `gue_phage`, `gue_covid`, `virobench`, `viral_vs_host`.

`--tasks paper` is the default and selects the ten marked `in_paper`.
`viral_vs_host` is excluded from the headline table and kept so the exclusion is
explicit. Definitions are in `pipelines/config/tasks.yaml`; every field there is
part of the cell definition and is written into output filenames, so changing one
measures something else.

### Models

`hvilm`, `dnabert2`, `nt_v2_500m`, `gena_lm`, `hyenadna`.

Declared in `pipelines/config/models.yaml`, which also records how each
checkpoint must be instantiated. That field is not cosmetic: resolving GENA-LM
with `trust_remote_code` alone silently loaded a stock post-LN encoder, half of
it randomly initialised, and the result was a plausible number rather than a
crash.

---

## 5. Tests

```bash
pytest
```

Configuration integrity, the capacity ladder's receptive-field arithmetic, the
paired bootstrap, path resolution and the reproduction entry points. No benchmark
data required.

---

## License

MIT. See `LICENSE`.
