# Open runs — the work queue

Ordered by what most changes the paper. Each entry names the findings it gates, so
nothing here is optional busywork: **R1–R4 are what convert the central claims
from provisional to publishable.**

Set up first (see [../README.md](../README.md)):

```bash
export VB_DATA=/scratch/$USER/glm-audit/data
export VB_MMSEQS=$(which mmseqs)
export HF_HOME=/scratch/$USER/glm-audit/hf_cache
python scripts/common/paths.py     # "(missing)" tells you what to fetch
```

Write every run's output under `results/` and commit it. The evidence base was
gitignored for most of this project's life; do not reintroduce that.

---

## R1 — 3-seed gLM refits · gates **F1, F5** · highest priority

The small model margins (median +0.02, and every "significant" row in
`ERASABLE_EXCESS.md`) currently rest on partial seed coverage. Without 3 seeds on
both arms, F5's significance calls and F1's "best model vs best baseline" row are
provisional.

```bash
python scripts/track_a_benchmarks/geneb_finetune.py --seeds 42,43,44 --lr-sweep
python scripts/track_a_benchmarks/aggregate_geneb.py       # rebuilds the 4-way table
```

Check afterwards: does every row in `ERASABLE_EXCESS.md` marked **YES** still
clear 2·SD? Any that flips must be restated in `FINDINGS.md`.

Also resolve while here: **GENA-LM wins 0 of 22 in `ERASABLE_EXCESS.md` but "11 of
13 categories" elsewhere.** Different regimes and comparators — write the
reconciliation down.

---

## R2 — Aligner reconciliation · gates **F4** · blocking

mmseqs nearest-hit reports **0.7383** macro-F1 on ViroBench family; BLASTn reports
**0.424**. A 0.31 gap between two implementations of the same comparator. F4
cannot be published until this is explained.

```bash
python scripts/track_b_viral/virobench_alignment_baseline.py --level family
```

Diagnose in this order — the likely causes, cheapest first:

1. **Tie-breaking and no-hit policy.** How is a query with no hit scored? Assigned
   to majority class, or dropped? Dropping inflates; assigning deflates. Make both
   implementations use the same rule.
2. **E-value / coverage thresholds.** mmseqs and BLASTn defaults are not
   comparable. Fix both explicitly.
3. **Database construction.** Is the search database train-only? A test sequence
   present in the database gives a trivially perfect hit.
4. **Sensitivity.** `mmseqs search -s` default is far less sensitive than BLASTn;
   a *less* sensitive tool scoring *higher* is itself the anomaly to explain.

Report the reconciled number with the settings that produced it, and state which
tool the paper uses and why.

---

## R3 — Viral CNN capacity search · gates **F6**

The CNN ladder covers only 9 task-cells. The viral rows in F6 lean on a
single-architecture CNN, and F3 shows that is exactly the mistake this paper
criticises. Applying our own standard to our own negative result is not optional.

```bash
python scripts/common/capacity_sweep.py --dataset hvue   --task Pathogenecity
python scripts/common/capacity_sweep.py --dataset hvue   --task Host_Tropism
python scripts/common/capacity_sweep.py --dataset gue     --task virus_covid
python scripts/common/capacity_sweep.py --dataset virobench --level family
```

If a searched CNN raises the viral baselines further, F6 strengthens. If it
lowers them, the excesses in F6 must be revised upward and reported.

---

## R4 — Homology threshold sensitivity · gates **F2, F7**

The 70% identity / 30% coverage threshold is ours. Re-run at field-standard
80–90% identity and report the range, for both the leak percentages (F2) and the
surviving-row counts (F7: 96 of 5,194 and 60 of 4,956).

```bash
for ID in 0.70 0.80 0.90; do
  python scripts/common/partial_overlap_audit.py --ident $ID --covfrac 0.30
done
python scripts/common/build_strict_splits.py --ident 0.80 --covfrac 0.30
```

`partial_overlap_audit.py` already sweeps `IDENTS = [0.90, 0.70, 0.50]` and
`COVFRACS = [0.50, 0.30]` — the sweep may already be in `results/splits_strict/`.
**Check before re-running.** What is missing is the *reporting* of the range, not
necessarily the computation.

---

## R5 — HViLM on strict-split Pathogenecity · gates **F7** · highest value per GPU-hour

`results/hvilm/` has HViLM on identity-disjoint Pathogenecity (**0.9879 AUROC**)
and on strict Host_Tropism (**0.5582**, chance). The missing cell is **strict
Pathogenecity** — the direct 0.99 → ? comparison on the same task.

```bash
python scripts/track_b_viral/hvilm_eval.py \
  --split Pathogenecity__strict_id70_cov30.parquet --which patho
```

Caveat to state up front: the strict Pathogenecity split has ≈96 rows. It may be
too small to be conclusive. **Report the confidence interval either way** — "too
few clean rows to measure" is itself F7's claim, so a wide CI is a result, not a
failure.

---

## R6 — Fix or drop the Evo likelihood run · gates **F8**

Evo scored 0 bases and skipped all 100 sequences
(`results/viral_likelihood/fullctx8000_cap200__evo_hyenadna.json`:
`evo_bases_scored: [0, 0]`, `evo_skipped: [100, 100]`). The reported
`evo_vs_markov = −1.969` is an artifact of the zero.

```bash
python scripts/track_b_viral/viral_likelihood_fullctx.py --models evo,hyenadna --window 8000
```

Likely causes: tokenizer/vocabulary mismatch dropping every sequence; a length
cap above Evo's 8k context; or a scoring path that silently skips on OOM. Confirm
`evo_bases_scored > 0` before trusting any number.

If it cannot be fixed cheaply, **drop Evo from F8** and report the honest version:
no evaluated model, including a high-order Markov chain, gets meaningfully below
the 2.0 bits/base uniform floor; only `generator12` does (1.891 in-distribution).
F8 is corroborating support either way — do not spend much on it.

---

## R7 — Optional: extend model coverage

Only if reviewer pressure demands it. Never-run models per
[TESTED_MATRIX.md](TESTED_MATRIX.md): ESM-2 / ESM-1v (protein), Caduceus, Evo-2,
GenoJEPA, and protein one-hot + biochemical baselines. DNABERT-2 is blocked by a
`config_class` conflict — pin `transformers<4.48` (already in
`requirements.txt`).

The paper does not need these. F7's HViLM result already covers "we tested the
specialised model", which is the coverage question that actually matters.

---

## Standing rules

From [PROTOCOL.md](PROTOCOL.md) — the ones most often violated:

- **Never mix adaptation regimes** in a comparison. The frozen→full-FT gap on
  splice alone is 0.60 MCC, larger than any gap between models.
- **Select on dev, report on test.** Selecting the CNN cell on test is the same
  error this paper accuses others of.
- **Declare equivalence margins before computing the CI.** `paired_bootstrap.py`
  fixes `DELTAS = [0.01, 0.02, 0.03, 0.05]` as a literal for exactly this reason,
  and `tests/test_paired_bootstrap.py` enforces that a wide interval is **not**
  equivalence.
- **A negative claim needs an equivalence test**, not a failed significance test.
- **Commit the artifact with the claim.**
