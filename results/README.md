# Evidence map

Every committed artifact, what it shows, which finding it backs, and which script
regenerates it. **Cite from here.** If a number in `reports/` is not traceable to a
row in this table, treat it as unverified until it is.

Findings are defined in [../reports/FINDINGS.md](../reports/FINDINGS.md).

---

## Top-level summary artifacts

These are the small, directly citable files. Read these before descending into the
per-run directories.

| artifact | shows | finding | produced by |
|:--|:--|:--|:--|
| `hvue_official_leakage.json` | HVUE train/test homology at 6 identity×coverage thresholds | **F2, F7** | `scripts/common/partial_overlap_audit.py` |
| `partial_overlap_audit.json` | leakage remaining *inside* the official "identity_disjoint" splits | **F2** | same |
| `viro_overlap.json` | ViroBench ALL_times overlap (the clean-split reference) | F2, F6 | same |
| `geneb_overlap.json` | GENEB per-task overlap (why GENEB is the positive control) | F5 | same |
| `aligner_reconcile.json` | **BLASTn vs mmseqs reconciled** — 3 settings, hit-rate vs hit-only accuracy | **F4** | `scripts/track_b_viral/virobench_alignment_baseline.py` |
| `virobench_alignment_family.json` | alignment nearest-hit on ViroBench family | **F4** | same |
| `matched_subsample_control.json` | **the strict-split drop is not a sample-size artifact** | **F7** | `scripts/common/build_strict_splits.py` |
| `homology_threshold_replication.json` | per-task × per-model score after dropping ≥X% identity, vs a matched-size control | **F2, the reversal** | `scripts/common/partial_overlap_audit.py` |
| `paired_bootstrap_viro_{kingdom,phylum,class,order,family}.json` | gLM vs k-mer paired CI + equivalence at **all 5 taxonomic levels** | **F6** | `scripts/common/paired_bootstrap.py` |
| `splice_final_table.csv` | incumbent vs dev-selected baseline vs frozen vs full FT | **F1, F3** | `scripts/track_a_benchmarks/aggregate_splice.py` |
| `geneb_four_way.csv` | fair k-mer \| CNN ladder \| frozen probe \| full FT, 40 rows | **F1, F5** | `scripts/track_a_benchmarks/aggregate_geneb.py` |
| `baseline_selection_audit.csv` | baseline dev-vs-test selection check — **see caveat below** | F1 | `scripts/track_a_benchmarks/aggregate_geneb.py` |
| `homcache_*.npz` (13) | cached pairwise homology per GENEB task | F2 | `scripts/common/partial_overlap_audit.py` |

## Per-run directories

| directory | files | contents | finding | produced by |
|:--|--:|:--|:--|:--|
| `capacity_sweep/` | 46 | CNN ladders: per cell `arch, params_M, rf_bp, dev, test, runs[]` | **F3** | `scripts/common/capacity_sweep.py` |
| `cnn_results/` | 63 | CNN baseline runs | F3, F5, F6 | same |
| `geneb_finetune/` | 864 | GENEB full FT: model × lr × seed, plus `__preds.npz` | **F5** | `scripts/track_a_benchmarks/geneb_finetune.py` |
| `geneb_evo/` | 7 | Evo-1 LoRA on GENEB | F5 | `scripts/track_b_viral/geneb_evo_lora.py` |
| `glm_results/` | 99 | frozen-probe gLM runs | F1, F5 | `scripts/track_a_benchmarks/gue_glm.py` |
| `learning_curve/` | 13 | GENEB sample-size curves | F5 | `scripts/track_a_benchmarks/geneb_learning_curve.py` |
| `splice_finetune/` | 17 | splice full FT, LR-swept | **F1, F3** | `scripts/track_a_benchmarks/splice_finetune.py` |
| `splice_positive_control/` | 3 | splice reproduction of published numbers | positive control | `scripts/track_a_benchmarks/splice_positive_control.py` |
| `virobench/` | 10 | ViroBench k-mer baselines + per-example predictions | **F6** | `scripts/track_b_viral/virobench_baselines.py`, `virobench_kmer_predictions.py` |
| `virobench_frozen/` | 70 | ViroBench frozen probes, layer-swept | **F1, F4, F6** | `scripts/track_b_viral/virobench_frozen_probe.py` |
| `virobench_glm/` | 25 | ViroBench gLM runs | F1, F6 | `scripts/track_b_viral/virobench_glm.py` |
| `hvue_glm/` | 72 | HVUE gLM probes and LoRA | F6, F7 | `scripts/track_b_viral/hvue_glm.py`, `hvue_evo_lora.py` |
| `hvue_cnn/` | 6 | HVUE CNN baselines | F6 | `scripts/track_b_viral/hvue_cnn.py` |
| `hvilm/` | 12 | **HViLM, the specialised viral model, on leaky vs strict splits** | **F7** | `scripts/track_b_viral/hvilm_eval.py` |
| `escape_conservation/` | 4 | escape prediction vs MSA conservation | F6 | `scripts/track_b_viral/escape_conservation_{baseline,matched}.py` |
| `proteingym_supervised/` | 30 | ProteinGym supervised comparison | F6 | `scripts/track_b_viral/proteingym_supervised.py` |
| `viral_likelihood/` | 5 | bits/base, Markov vs gLM — **Evo entry is broken, see F8** | F8 | `scripts/track_b_viral/viral_likelihood*.py` |
| `gue_baselines/`, `gue_glm/`, `gue_glm_results/`, `gue_results/`, `gue_viral/` | 42 | GUE non-viral and viral | F5, F6 | `scripts/track_a_benchmarks/gue_{baselines,glm}.py` |
| `splits_strict/`, `splits_ungated/`, `splits_idsweep/` | 5 | split construction audits | F2, F7 | `scripts/common/build_strict_splits.py`, `scripts/track_b_viral/build_identity_splits.py` |
| `evo_results/`, `virobench_results/` | 4 | aggregation outputs | — | aggregators |

---

## Numbers verified directly from these artifacts

Re-read from the files on 2026-09-03. Where a report disagrees, the artifact wins
until someone shows otherwise.

### F7 — HViLM, and the sample-size control

`hvilm/`, model `duttaprat/HViLM-Patho`:

| head | split | n | AUROC | MCC |
|:--|:--|--:|--:|--:|
| patho | Pathogenecity, identity-disjoint | 5,194 | **0.9879** | 0.8739 |
| patho | Host_Tropism, identity-disjoint | 8,390 | 0.6909 | 0.2578 |
| patho | Host_Tropism, **strict id70/cov30** | 3,391 | **0.5582** | **0.0173** |
| tropism | Host_Tropism, identity-disjoint | 8,390 | 0.7925 | 0.5758 |
| tropism | Host_Tropism, **strict id70/cov30** | 3,391 | 0.6467 | 0.3621 |
| tropism | Pathogenecity, identity-disjoint | 5,194 | 0.7531 | 0.5725 |

`matched_subsample_control.json` rules out the obvious objection that the strict
drop is just a smaller test set:

```
auc_full 0.7925  →  auc_strict 0.6467   (n_strict 3391)
matched-size random subsample: 0.7923, CI [0.7682, 0.8168]
frac_below = 0.0
verdict: "HOMOLOGY MATTERS — strict is BELOW the matched-random range by 0.1215"
```

The strict value sits below the matched-random range in **100%** of draws. This is
the rigorous core of F7 and it should be in the paper next to the HViLM table.

### F2/F7 — leakage survives the field-standard threshold

`hvue_official_leakage.json`, % of test rows with a training hit:

| task | exact dup | id90/cov50 | id90/cov30 | id70/cov30 |
|:--|--:|--:|--:|--:|
| Host_Tropism | 3.04 | 32.62 | 35.58 | 60.14 |
| Pathogenecity / BVBRC_Calci | 0.00 | **99.76** | 99.79 | 99.99 |
| Pathogenecity / BVBRC_CoV | 0.00 | **99.55** | 99.62 | 99.95 |
| Pathogenecity / CINI | 0.00 | **91.01** | 91.19 | 96.51 |

**The 70/30 threshold was never load-bearing.** Pathogenecity is 91–99.8% leaked
at 90% identity / 50% coverage, the strictest setting anyone would call standard.

`partial_overlap_audit.json` — the official *identity_disjoint* splits are still
42.2% (Host_Tropism) and higher leaked at id90/cov50. `viro_overlap.json` —
ViroBench ALL_times is **2.07%** at id90/cov50, which is why it is the clean
reference.

### F4 — the aligner discrepancy, reconciled

`aligner_reconcile.json`:

| setting | hit rate | macro-F1 (all queries) | macro-F1 (hit-only) |
|:--|--:|--:|--:|
| BLASTn | 0.7671 | 0.5920 | **0.9364** |
| mmseqs default | 0.8992 | **0.7823** | 0.9467 |
| mmseqs sensitive | 0.8992 | 0.7823 | 0.9467 |
| mmseqs permissive | 0.8501 | 0.7384 | **0.9637** |

**The 0.738-vs-0.424 discrepancy was a hit-rate difference, not a disagreement
about accuracy.** Where each tool finds a hit the two agree to within 0.03
macro-F1 (0.936 vs 0.947). The previously reported 0.7383 is the *permissive*
mmseqs setting; reconciled BLASTn is 0.592, not 0.424.

This is a better result than the original claim: alignment solves viral family
assignment at **0.94–0.96 macro-F1 on the queries it can align**, against a best
gLM of 0.6148. The honest framing is coverage-limited alignment, not "alignment
wins by 0.115".

### F6 — the ViroBench null at all five levels

`paired_bootstrap_viro_*.json`, k-mer test macro-F1 vs frozen-probe gLMs, 2000
bootstrap replicates, margins pre-declared as `[0.01, 0.02, 0.03, 0.05]`:

| level | classes | k-mer test | example gLM | gLM − k-mer |
|:--|--:|--:|:--|--:|
| kingdom | 18 | 0.5612 | nt_v2_500m 0.5778 | +0.017 |
| phylum | 28 | 0.5446 | nt_v2_500m 0.5293 | −0.015 |
| class | 45 | 0.5293 | nt_v2_500m 0.5246 | −0.005 |
| order | 67 | 0.5987 | nt_v2_500m 0.5584 | −0.040 |
| family | 173 | 0.5738 | gena_lm 0.5035 | −0.070 |

Full per-run comparisons with CIs and equivalence verdicts are in each file.

### F3 — the receptive-field result, recomputed

From `capacity_sweep/geneb__..._splice_sites_all.json` (**seqlen 400**, MCC, n_test 3000):

| arch | params | RF (bp) | test MCC |
|:--|--:|--:|--:|
| unet | **0.149M** | global | **0.9685** |
| unet | 0.257M | global | 0.9655 |
| dilated | 9.335M | 505 | 0.9695 |
| resnet | 6.076M | 89 | 0.3163 |
| resnet | **9.438M** | **89** | **0.3073** |
| dilated | 0.406M | 21 | 0.2771 |

A **0.149M** U-Net beats a **9.438M** ResNet by **0.661 MCC** at 63× fewer
parameters, purely because it can span the input. Ladder spread (best test to
worst test) is **0.692**.

> **Three number corrections the reports need.** `PAPER_OUTLINE.md` and
> `STATE_OF_EVIDENCE.md` currently say "ResNet 9.44M (RF 89bp) = 0.336 vs U-Net
> 0.26M (global) = 0.951 on **600bp** splice", and "0.2909 → 0.9538 (U-Net 5.3M)".
> Against the artifacts: (1) the task is `splice_sites_all` at **400 bp** — the
> 600 bp task is `revised_splice_sites_acceptors`, where RF-89 ResNet scores
> **0.499**, not 0.336; (2) RF-89 ResNet on splice_all is **0.3073**, and the
> 0.26M U-Net is **0.9655**; (3) the quoted 0.2909/0.9538 endpoints are
> **0.3095** (dilated 0.041M) and **0.963** (unet 5.274M). The `test` field is
> already seed-aggregated with `test_sd`, so this is not a seed-averaging
> difference. **Reconcile before submission** — the finding is unaffected and in
> fact stronger, but the specific numbers must match a committed file.

---

## Caveats on specific artifacts

**`baseline_selection_audit.csv` — `bb_val` equals `bb_test` on every row**, so
`d_val` equals `d_test`. If the baseline cell is meant to be chosen on dev and
scored on test, those columns should differ. Either the audit is reporting the
same quantity twice, or the selection is not actually happening on dev. Resolve
before citing anything from this file (see `OPEN_RUNS.md` R7).

**`viral_likelihood/fullctx8000_cap200__evo_hyenadna.json`** reports `"evo": 0.0`
with `evo_bases_scored: [0, 0]` and `evo_skipped: [100, 100]`. Evo scored nothing;
`evo_vs_markov_indist = -1.969` is an artifact of the zero. Do not cite any Evo
likelihood number.

**`geneb_finetune/` contains multiple LRs per model**; the LR must be selected on
dev, not test. `geneb_four_way.csv` records the selected `FT_lr` per row — use
that, not the per-file maximum.

**Raw corpora are deliberately absent** (parquet, and train/test sequence CSVs up
to 70 MB). Only run summaries, per-example predictions, and summary tables are
committed. Fetch corpora per [../README.md](../README.md).
