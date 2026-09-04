# Genomic foundation-model capability: a benchmark audit and a scoped viral negative result

Do published genomic language model (gLM) benchmark tables support the capability
claims made from them? We find that on viral tasks they do not — and we show that
with the same harness, models, and protocols that *do* detect capability on
non-viral tasks.

The negative result is **scoped and positively controlled**. This is not "gLMs
don't work". It is: *gLMs work, and here is a domain where the evidence that they
work does not survive contact with the baselines and the splits.*

Working title, thesis, and section plan: **[reports/PAPER_OUTLINE.md](reports/PAPER_OUTLINE.md)**.

---

## The result in one table

Evaluation-method choices move results far more than the margin by which the best
model beats the best baseline.

| lever | measured spread | where |
|:--|--:|:--|
| split / subset choice | **0.37** macro-F1 | k-mer3-6 on ViroBench family: 0.9252 → 0.5566 |
| baseline architecture | **0.69** MCC | CNN ladder on splice_all: 0.2771 → 0.9695 |
| adaptation regime | **0.60** MCC | NT-v2 splice: frozen probe 0.3636 → full FT 0.9674 |
| homology-audit tool | **0.80 → 0.02** | HVUE Path. "leak-free" by `easy-cluster`, 80.5% leaked by `easy-search` |
| comparator class | **0.17** macro-F1 | ViroBench family: best gLM 0.6148 vs mmseqs nearest-hit 0.7823 |
| model, everything else fixed | **0.32** macro-F1 | ViroBench family: NT-v2 0.611 → LucaVirus 0.295 |
| **best model vs best baseline** | **+0.01 to +0.015** MCC | splice: NT-v2 0.9674 vs CNN ladder 0.9528 |

The last two rows are the point, and they are two different quantities — keep them
apart:

- **Model excess over the best baseline is +0.01–0.015.** Every method lever above
  is **25–45× larger** than that. This is the strong claim.
- **Model spread, everything else fixed, is 0.32.** Changing the split (0.37) moves
  results only ~1.2× more than changing the model. So "method dominates model" is
  true of the *excess*, not of the *spread*.

Conflating the two — quoting a 5–15× multiplier against a 0.32 spread — is the
first thing a referee will check. See
[reports/FINDINGS.md](reports/FINDINGS.md) F1.

### Three independent failure modes, each with a measured instance

1. **Homology saturation.** HVUE Pathogenecity test rows are **91–99.8% leaked at
   90% identity / 50% coverage** — the strictest threshold anyone calls standard,
   so this is not threshold-shopping.
   Strict refiltering leaves Pathogenecity at 96 of 5,194 test rows and
   Transmissibility at 60 of 4,956. Two of three HVUE tasks cannot support a
   homology-clean evaluation at all. ViroBench, by contrast, is 2.1% overlapping
   and temporally disjoint — and there **no model beats a k-mer baseline
   significantly at any of five taxonomic levels**.
2. **Under-powered baselines.** On 400 bp splice, a **0.149M** U-Net that spans the
   input scores **0.9685** while a **9.438M** ResNet with an 89 bp receptive field
   scores **0.3073** — a 0.661 MCC gap at 63× fewer parameters. Published
   FM-vs-CNN gaps of +0.31–0.60 shrink to +0.02–0.03 once the baseline can see the
   sequence. The receptive-field arithmetic is unit-tested
   ([tests/test_capacity_sweep.py](tests/test_capacity_sweep.py)).
3. **Wrong comparator class.** On ViroBench family assignment, mmseqs nearest-hit
   scores 0.7823 macro-F1 against a best gLM of 0.6148 — and **0.94–0.96 on the
   queries it can align** (both mmseqs and BLASTn). Viral family assignment is
   solved by database lookup wherever a homolog exists. Comparing a gLM only to a
   k-mer model on a taxonomic task omits the method that actually solves it.

### The specialised viral model shows the defect on itself

HViLM (`duttaprat/HViLM-Patho`) scores **0.9879 AUROC** on the HVUE Pathogenecity
split the benchmark calls identity-disjoint, and **0.5582 — chance, MCC 0.017** —
on Host_Tropism once homology is actually removed. The benchmark-validity defect,
on the field's own specialised model, with the benchmark's own labels.

And it is not a sample-size artifact: a matched-size random subsample of the same
test set scores 0.7923 [0.7682, 0.8168], with the strict value below that range in
**100% of draws** (`results/matched_subsample_control.json`). Full tables:
[results/README.md](results/README.md).

### The positive controls that make the null informative

Same harness, same protocols, non-viral tasks: GENA-LM wins 11 of 13 GENEB
categories and NT-v2 10 of 13; splice fine-tuning reproduces published numbers at
0.968 MCC. A harness that detects nothing anywhere would prove nothing.

On the 22-task GENEB comparison with three seeds per arm, a properly tuned CNN —
not a k-mer — is the binding baseline on 21 of 22 tasks, and only 5 tasks show a
significant model excess ([reports/ERASABLE_EXCESS.md](reports/ERASABLE_EXCESS.md)).

---

## Read in this order

| # | document | what it gives you |
|--:|:--|:--|
| 0 | [reports/FINDINGS.md](reports/FINDINGS.md) | **the claim ledger.** Eight findings, each marked SOLID or CONTINGENT, with its evidence path and the framing risk a referee will attack. Start here |
| 0 | [reports/OPEN_RUNS.md](reports/OPEN_RUNS.md) | **the work queue.** What is closed, what remains, each item naming the findings it gates |
| 0 | [results/README.md](results/README.md) | **the evidence map.** Every artifact → what it shows → which finding → which script. Cite from here |
| 1 | [reports/PAPER_OUTLINE.md](reports/PAPER_OUTLINE.md) | the thesis, why this framing survives review, section-by-section plan |
| 2 | [reports/STATE_OF_EVIDENCE.md](reports/STATE_OF_EVIDENCE.md) | **every number, with provenance.** Own runs vs `C2` (Vista cluster) vs published. Start here for any table |
| 3 | [reports/PAPER_DESIGN.md](reports/PAPER_DESIGN.md) | the validity grid the design rests on, and what populates each quadrant |
| 4 | [reports/PROTOCOL.md](reports/PROTOCOL.md) | 27 binding rules; each exists because violating it changed a conclusion here |
| 5 | [reports/TESTED_MATRIX.md](reports/TESTED_MATRIX.md) | benchmark × method × regime, marked done / partial / **not run** / invalid |
| 6 | [reports/ERASABLE_EXCESS.md](reports/ERASABLE_EXCESS.md) | the finalised 22-task excess table, 3 seeds per arm |
| 7 | [reports/BASELINE_CAPACITY_CEILING.md](reports/BASELINE_CAPACITY_CEILING.md) | the CNN ladder result in full |
| 8 | [reports/SPLIT_DESIGN_EXPLAINED.md](reports/SPLIT_DESIGN_EXPLAINED.md) | composition-disjoint vs homology holdout, and which is correct for what |
| 9 | [reports/VALIDITY_GATES.md](reports/VALIDITY_GATES.md) | the gate order a capability claim must pass |

Supporting: [MEASUREMENT_PROTOCOL.md](reports/MEASUREMENT_PROTOCOL.md),
[FULL_FT_DESIGN.md](reports/FULL_FT_DESIGN.md),
[CROSS_CLUSTER_SYNTHESIS.md](reports/CROSS_CLUSTER_SYNTHESIS.md),
[RESEARCH_PLAN.md](reports/RESEARCH_PLAN.md),
[PRIOR_WORK_DASHA_COMPARISON.md](reports/PRIOR_WORK_DASHA_COMPARISON.md),
[OUTCOME_FOR_UNLEARNING.md](reports/OUTCOME_FOR_UNLEARNING.md) — what this means
for the sister unlearning project.

`reports/archive/` holds superseded working documents. **Do not cite them**; cite
`STATE_OF_EVIDENCE.md` and the artifacts under `results/`.

---

## The two tracks

Both share one harness, because a viral null is only informative if the harness is
shown to detect capability elsewhere on comparably honest ground.

| | **Track A — benchmark & method audit** | **Track B — viral capability** |
|:--|:--|:--|
| question | across existing benchmarks, what does each *method class* achieve? | is there model-specific viral capability over the strongest comparator on a defensible split? |
| benchmarks | GENEB, GUE, NT, EPI, DART-Eval | HVUE, ViroBench, GUE viral, ProteinGym viral, escape |
| comparators | k-mer, CNN ladder | + alignment nearest-hit, Kraken2, MSA |
| contribution | published gaps are largely baseline artifacts | scoped negative result, positively controlled by Track A |
| code | `scripts/track_a_benchmarks/`, `scripts/geneb/` | `scripts/track_b_viral/` |

Track A's headline was discovered *because* Track B forced a proper comparator.

---

## Quickstart

```bash
conda env create -f environment.yml && conda activate glm-audit   # includes mmseqs2
# or: pip install -r requirements.txt  +  conda install -c bioconda mmseqs2

pytest                        # 41 passed, ~0.2s, CPU only
python scripts/common/paths.py   # print resolved config; "(missing)" = corpus not fetched yet
```

**Configuration.** Every input path resolves through
[scripts/common/paths.py](scripts/common/paths.py) with an environment override
and a repository-relative default — nothing is hardcoded to a particular machine.
On a new host set at minimum:

```bash
export VB_DATA=/scratch/$USER/glm-audit/data    # where benchmark corpora live
export VB_MMSEQS=$(which mmseqs)                # required by every homology audit
export HF_HOME=/scratch/$USER/glm-audit/hf_cache
```

Then `python scripts/common/download_data.py`, or point the per-corpus variables
(`VB_HVUE_DIR`, `VB_GENEB_DIR`, `VB_NT_DIR`, `VB_VIRO_DIR`, `VB_GUE_DIR`,
`VB_PROTEINGYM_DMS`, `VB_ESCAPE_CSV`, …) at copies you already have.
`python scripts/common/paths.py` prints what resolved and what is missing.

**mmseqs2 is not optional.** Every homology claim in the paper uses
`easy-search` (local alignment) rather than `easy-cluster`, and that distinction
is itself a finding — see failure mode 1.

**GPU.** Only the gLM runs need one. The k-mer baselines, the CNN ladder, and the
homology audits are CPU work.

---

## Layout

```text
scripts/
  common/            paths resolver, capacity sweep (CNN ladder), paired bootstrap,
                     homology audits (partial_overlap_audit, build_strict_splits),
                     split auditing, corpus download
  track_a_benchmarks/  GENEB and splice: k-mer + CNN + frozen probe + full FT, aggregation
  track_b_viral/       HVUE, ViroBench, escape, ProteinGym: baselines, alignment
                     comparator, gLM probes, LoRA, viral likelihood
  geneb/             GENEB fair-k-mer sentinel and feature extractors
  positive_control/  GUE / NT / EPI split preparation and leakage checks
reports/             the paper-facing documents, plus small result tables
  archive/           superseded working documents — not for citation
results/             482 committed run artifacts (~2.3 MB): JSON summaries and
                     per-example .npz predictions
tests/               41 CPU-only tests
```

Every script resolves inputs through `paths.py`; the per-directory `paths.py`
files are shims re-exporting `scripts/common/paths.py`.

---

## Reproducing a number

Result artifacts are committed, so a table can be rebuilt without a GPU:

```bash
python scripts/track_a_benchmarks/aggregate_geneb.py    # four-way GENEB comparison
python scripts/track_a_benchmarks/aggregate_splice.py   # splice ladder + FT
python scripts/common/paired_bootstrap.py family        # ViroBench paired CI + equivalence
```

`results/` holds `capacity_sweep/` (21 CNN ladders, per-arch params and receptive
field), `geneb_finetune/` (454 files: per model × LR × seed, with `.npz`
predictions so the paired bootstrap can be re-run), and `geneb_evo/` (7 Evo-1 LoRA
runs).

**Equivalence, not absence of evidence.** `paired_bootstrap.py` declares its
margins (`DELTAS = [0.01, 0.02, 0.03, 0.05]`) *before* computing any CI, and
`equivalence_verdict()` requires the whole interval inside ±margin. A wide
interval returns `False` — "we failed to show a difference" is not "we showed
equivalence", and the tests enforce that
([tests/test_paired_bootstrap.py](tests/test_paired_bootstrap.py)).

---

## Known gaps

Taken from [reports/TESTED_MATRIX.md](reports/TESTED_MATRIX.md); these are what a
referee will ask about.

- **GENEB CNN ladder** is marked a critical gap on the 13 sentinel tasks: the
  frozen-probe wins there are k-mer-anchored, and a tuned CNN is the binding
  baseline everywhere it has been run.
- **GENEB remainder**: 87 of 100 tasks untouched.
- **Never run**: ESM-2 / ESM-1v (pLMs), DNABERT-2 (blocked by a `config_class`
  conflict with transformers ≥4.48), Caduceus, Evo-2, GenoJEPA; protein one-hot +
  biochemical baselines. HViLM **has** been run — see above.
- **The Evo likelihood run is broken** (0 bases scored, all sequences skipped), so
  there is currently no Evo generative result. `reports/OPEN_RUNS.md` R6.
- **Frozen-probe layer sweep** was done only on ViroBench family; elsewhere the
  read-out layer is `L−1`, worth 0.02–0.04 macro-F1.
- **CNN ladder coverage** is 9 task-cells, not all of them.
- **Evo-1** has LoRA-only coverage on HVUE and no ViroBench runs.
- **Test coverage** is 41 tests over the pure logic — path resolution, the
  receptive-field arithmetic, the equivalence verdict. The training and
  extraction paths are exercised by running them, not by unit tests.


---

MIT — see [LICENSE](LICENSE).
