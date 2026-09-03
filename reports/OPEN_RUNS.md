# Open runs — the work queue

Ordered by what most changes the paper. Each entry names the findings it gates.

**Two of the original blockers turned out to be already done** — the artifacts
existed but were gitignored and undocumented. Check `results/README.md` before
starting anything here; this project's recurring failure mode is re-running work
that was already complete.

Set up first (see [../README.md](../README.md)):

```bash
export VB_DATA=/scratch/$USER/glm-audit/data
export VB_MMSEQS=$(which mmseqs)
export HF_HOME=/scratch/$USER/glm-audit/hf_cache
python scripts/common/paths.py     # "(missing)" tells you what to fetch
```

Commit every run's output under `results/` and add a row to
[../results/README.md](../results/README.md). The evidence base was gitignored for
most of this project's life; do not reintroduce that.

---

## CLOSED — verify, do not re-run

| was | status | evidence |
|:--|:--|:--|
| **R2** aligner reconciliation (mmseqs 0.738 vs BLASTn 0.424) | **DONE** | `results/aligner_reconcile.json` — the gap was hit rate (0.767 vs 0.899), not accuracy; hit-only agreement within 0.03. F4 is now SOLID |
| **R4** homology threshold sensitivity | **DONE** | `results/hvue_official_leakage.json` — Pathogenecity 91–99.8% leaked at id90/cov50. F2's gate is closed and the answer is the strong one |
| **R5** HViLM on strict splits | **PARTLY DONE** | `results/hvilm/` has 6 runs; only strict-Pathogenecity is missing → **R3** below |
| matched-size control for the strict drop | **DONE** | `results/matched_subsample_control.json` — strict is below the matched-random range in 100% of draws |
| GENEB homology-vs-size decomposition | **DONE** | `results/homology_threshold_replication.json` — per task × model, with matched-size control bands |

---

## R1 — 3-seed gLM refits · gates **F1, F5** · highest priority

The small model margins (median +0.02, and every "significant" row in
`ERASABLE_EXCESS.md`) rest on partial seed coverage. Without 3 seeds on both arms,
F5's significance calls and F1's "best model vs best baseline" row stay
provisional.

```bash
python scripts/track_a_benchmarks/geneb_finetune.py --seeds 42,43,44 --lr-sweep
python scripts/track_a_benchmarks/aggregate_geneb.py       # rebuilds geneb_four_way.csv
```

Afterwards: does every **YES** row in `ERASABLE_EXCESS.md` still clear 2·SD? Any
that flips must be restated in `FINDINGS.md`.

Resolve while here: **GENA-LM wins 0 of 22 in `ERASABLE_EXCESS.md` but "11 of 13
categories" elsewhere.** Different regimes and comparators — write the
reconciliation down explicitly, or a referee reads it as an error.

---

## R2 — Number reconciliation against the artifacts · gates **F1, F3** · cheap, do it first

Several quoted numbers do not match the committed files. No GPU needed; this is an
afternoon with a text editor and `results/`.

Known mismatches, all detailed in [../results/README.md](../results/README.md):

- **F3 splice**: reports say "RF-89 ResNet 0.336 vs 0.26M U-Net 0.951 on 600 bp".
  Artifacts say `splice_sites_all` is **400 bp**, RF-89 ResNet is **0.3073**, the
  0.26M U-Net is **0.9655**; the 600 bp task is `revised_splice_sites_acceptors`
  where RF-89 scores **0.499**. Ladder endpoints are **0.3095 → 0.963**, not
  0.2909 → 0.9538.
- **F4 comparator class**: the 0.7383 figure is mmseqs *permissive*. Default is
  0.7823, BLASTn 0.592. The 0.115 gap should be restated.
- Sweep the rest of `STATE_OF_EVIDENCE.md` and `PAPER_OUTLINE.md` the same way.

Rule to adopt: **every number in a report carries the artifact path it came from.**
That is what made these mismatches findable at all.

---

## R3 — HViLM on strict-split Pathogenecity · gates **F7** · highest value per GPU-hour

`results/hvilm/` has HViLM on identity-disjoint Pathogenecity (**0.9879 AUROC**)
and on strict Host_Tropism (**0.5582**, chance). The missing cell is **strict
Pathogenecity** — the direct 0.99 → ? comparison on the same task.

```bash
python scripts/track_b_viral/hvilm_eval.py \
  --split Pathogenecity__strict_id70_cov30.parquet --which patho
```

State the caveat up front: the strict Pathogenecity split has ≈96 rows and may be
too small to be conclusive. **Report the CI either way** — "too few clean rows to
measure" is itself F7's claim, so a wide interval is a result, not a failure. Run
the matched-size control alongside it, as was done for Host_Tropism.

---

## R4 — Viral CNN capacity search · gates **F6**

The CNN ladder covers only 9 task-cells. The viral rows in F6 lean on a
single-architecture CNN — exactly the mistake F3 criticises. Applying our own
standard to our own negative result is not optional; a referee will ask.

```bash
python scripts/common/capacity_sweep.py --dataset hvue     --task Pathogenecity
python scripts/common/capacity_sweep.py --dataset hvue     --task Host_Tropism
python scripts/common/capacity_sweep.py --dataset gue      --task virus_covid
python scripts/common/capacity_sweep.py --dataset virobench --level family
```

If a searched CNN raises the viral baselines, F6 strengthens. If it lowers them,
the excesses in F6 must be revised upward and reported. Either outcome is
publishable; not running it is not.

---

## R5 — Fix or drop the Evo likelihood run · gates **F8**

Evo scored 0 bases and skipped all 100 sequences
(`results/viral_likelihood/fullctx8000_cap200__evo_hyenadna.json`:
`evo_bases_scored: [0, 0]`, `evo_skipped: [100, 100]`). The reported
`evo_vs_markov = −1.969` is an artifact of the zero.

```bash
python scripts/track_b_viral/viral_likelihood_fullctx.py --models evo,hyenadna --window 8000
```

Likely causes: tokenizer/vocabulary mismatch dropping every sequence; a length cap
above Evo's 8k context; or a scoring path that silently skips on OOM. **Confirm
`evo_bases_scored > 0` before trusting any number.**

If it does not fix cheaply, **drop Evo from F8** and report the honest version: no
evaluated model, including a high-order Markov chain, gets meaningfully below the
2.0 bits/base uniform floor; only `generator12` does (1.891 in-distribution). F8
is corroborating support either way — do not spend much on it.

---

## R6 — Resolve the baseline-selection audit · gates **F1** · cheap

In `results/baseline_selection_audit.csv`, `bb_val` equals `bb_test` on every row,
so `d_val` equals `d_test`. If the baseline cell is selected on dev and scored on
test, those must differ. Either the file reports the same quantity twice, or the
selection is not happening on dev — and selecting on test is precisely the error
this paper accuses others of.

Read `scripts/track_a_benchmarks/aggregate_geneb.py`, confirm which column is
which, and either fix the audit or fix the selection. Do not cite this file until
resolved.

---

## R7 — Optional: extend model coverage

Only under reviewer pressure. Never-run per [TESTED_MATRIX.md](TESTED_MATRIX.md):
ESM-2 / ESM-1v, Caduceus, Evo-2, GenoJEPA, protein one-hot + biochemical
baselines. DNABERT-2 needs `transformers<4.48` (already pinned in
`requirements.txt`).

The paper does not need these. F7's HViLM result already answers the coverage
question that matters: the specialised viral model was tested, and it shows the
defect on itself.

---

## Standing rules

From [PROTOCOL.md](PROTOCOL.md) — the ones most often violated here:

- **Never mix adaptation regimes** in a comparison. The frozen→full-FT gap on
  splice alone is 0.60 MCC, larger than any gap between models.
- **Select on dev, report on test.** See R6.
- **Declare equivalence margins before computing the CI.**
  `scripts/common/paired_bootstrap.py` fixes
  `DELTAS = [0.01, 0.02, 0.03, 0.05]` as a literal for this reason, and
  `tests/test_paired_bootstrap.py` enforces that a wide interval is **not**
  equivalence.
- **A negative claim needs an equivalence test**, not a failed significance test.
- **Run the matched-size control** whenever a filtered split is smaller than the
  original. `results/matched_subsample_control.json` is the template; without it,
  "the score dropped after filtering" is confounded with "the test set shrank".
- **Commit the artifact with the claim, and cite its path in the report.**
