# Findings register

One row per publishable claim. For each: the claim, the artifact that backs it,
whether it is **SOLID** or **CONTINGENT**, and the framing risk a reviewer will
attack. This is the ledger; [PAPER_OUTLINE.md](PAPER_OUTLINE.md) is the prose
structure and [STATE_OF_EVIDENCE.md](STATE_OF_EVIDENCE.md) is the number-by-number
record.

Nothing here may be written up as SOLID until its evidence path resolves to a file
under `../results/` or a table in `STATE_OF_EVIDENCE.md`. Contingent findings are
gated by [OPEN_RUNS.md](OPEN_RUNS.md).

---

## The thesis

Genomic foundation models are evaluated almost entirely against each other, on
benchmarks whose splits leak homology and whose baselines are under-powered. Fix
both — search the baseline properly, clean the splits honestly — and the
model-specific advantage shrinks to a small margin confined to the largest models
on a subset of tasks, and vanishes on the viral tasks that motivate biosecurity
concern, where the capability is recoverable by composition or alignment. **The
dominant driver of reported performance is the evaluation method, not the model.**
Safety consequence: capability-removal safeguards applied to viral capability
target something that is not uniquely in the weights.

---

## F1 — The evaluation-lever hierarchy · **SOLID (with a framing split)**

Method-of-evaluation choices move results far more than model choice.

| lever | spread | evidence |
|:--|--:|:--|
| baseline architecture | 0.66 MCC | `results/capacity_sweep/`, 13-cell ladder on splice |
| adaptation regime | 0.60 MCC | `results/splice_finetune/`, frozen 0.3636 → full FT 0.9674 |
| split / subset choice | 0.37 macro-F1 | `results/virobench/`, k-mer3-6 0.9252 → 0.5566 |
| homology-audit tool | 0.80 → 0.02 | `reports/mmseqs_leakage_check.csv` + partial-overlap audit |
| comparator class | 0.115 macro-F1 | `results/virobench_frozen/`, alignment 0.7383 vs gLM 0.6148 |
| model, everything else fixed | **0.32** macro-F1 | `results/virobench_glm/`, NT-v2 0.611 → LucaVirus 0.295 |
| best model vs best baseline | **+0.01–0.015** MCC | splice: NT-v2 0.9674 vs CNN 0.9528 |

> **Framing risk — the one most likely to break this finding.** The last two rows
> are different quantities and the multiplier depends on which you use as the
> denominator:
>
> - against **model *excess* over the best baseline** (0.015): method levers are
>   **25–45×** larger. This is the strong, defensible claim.
> - against **model *spread*** (0.32): the split lever is only **1.2×** larger.
>
> State them as two separate sentences. A summary that says "method choices move
> results 5–15× more than model choice" against a model spread of 0.32 is
> arithmetically wrong and a referee will find it. `STATE_OF_EVIDENCE.md` already
> separates the two rows correctly — keep it that way.

Also unresolved for write-up: the levers are measured on **different benchmarks
and metrics** (MCC on splice, macro-F1 on ViroBench). Either normalise, or state
plainly that the comparison is across task families and is indicative of
magnitude, not a within-task decomposition.

---

## F2 — The homology-audit tool defect · **SOLID pending threshold sensitivity**

MMseqs2 `easy-cluster` at `-c 0.9` requires 90% bidirectional coverage and is
therefore blind to partial homology. Splits it certifies as identity-disjoint
remain substantially leaked; a **search**-based audit (`easy-search`, local
alignment) is required.

- HVUE Pathogenecity: "leak-free" under `easy-cluster`; **80.5% leaked** under
  `easy-search`.
- Evidence: `scripts/common/partial_overlap_audit.py`,
  `reports/mmseqs_leakage_check.csv`, `results/splits_strict/`.
- Propagates beyond this paper: PathoLM used the same tool. This is the cleanest
  methods contribution here and the field can adopt it immediately.

**Gate:** the 70% identity / 30% coverage threshold is ours, not a field standard.
Re-run the audit at 80–90% identity and report the range. If the leak figure
survives at field-standard thresholds the finding is much harder to dismiss; if it
does not, the claim must be scoped to the threshold. See OPEN_RUNS R4.

---

## F3 — Receptive field, not capacity, binds CNN baselines · **SOLID**

A 0.26M-parameter U-Net with global receptive field beats a 9.44M-parameter ResNet
with an 89 bp receptive field by **0.62 MCC** on a 600 bp splice task (0.951 vs
0.336). Published FM-over-CNN gaps of +0.31–0.60 shrink to +0.02–0.03 once the
baseline can see the input.

- Evidence: `results/capacity_sweep/` (per-cell params and `rf_bp`),
  `reports/BASELINE_CAPACITY_CEILING.md`.
- The arithmetic is unit-tested: `tests/test_capacity_sweep.py` pins the 89 bp and
  asserts the incumbent baseline (dilated/128/5, **249 bp**) cannot span 600 bp.
- Mechanistic, generalisable, and it explains prior weak-baseline results rather
  than merely reporting them. Strongest self-contained contribution.

---

## F4 — Alignment is the missing comparator on taxonomy · **CONTINGENT — BLOCKED**

A parameter-free nearest-hit aligner beats every model on ViroBench taxonomy
(0.7383 vs best gLM 0.6148), winning on ~10–13% coverage of a conserved gene.
Viral family taxonomy is an alignment problem, not a capability problem.

- Evidence: `results/virobench_frozen/`,
  `reports/virobench_full_spec_baselines.md`.
- **Blocker: mmseqs nearest-hit 0.738 vs BLASTn 0.424 — a 0.31 discrepancy
  between two implementations of the same comparator.** Until reconciled, the
  number cannot be published: a referee will simply ask which is right. See
  OPEN_RUNS R2.

---

## F5 — Model-specific advantage is small and confined to the largest models · **CONTINGENT**

Against a fully-searched CNN on 22 GENEB tasks with 3 seeds per arm, only **5
tasks** show a significant excess, and a **tuned CNN — not a k-mer — is the
binding baseline on 21 of 22**. Median excess ≈ +0.02.

- Evidence: `reports/ERASABLE_EXCESS.md`, `results/geneb_finetune/` (864 files,
  per model × LR × seed, with predictions), `results/cnn_results/`.
- Both "gLMs beat CNNs" and "CNNs beat gLMs" are wrong; the truth is
  task-and-scale dependent.

> **Consistency problem to resolve before writing.** In `ERASABLE_EXCESS.md` the
> best gLM is NT-v2 on 11 tasks and DNABERT-2 on 10 — **GENA-LM wins 0 of 22**.
> Elsewhere the positive-control claim is "GENA-LM wins 11 of 13 GENEB
> categories". Those are different comparisons (regime, and vs published baseline
> rather than vs tuned CNN), but as written they read as a contradiction.
> Reconcile explicitly in the text, or a reviewer will treat it as an error.

**Gate:** 3-seed gLM refits (OPEN_RUNS R1).

---

## F6 — Viral capability is not model-specific · **SOLID in direction, CONTINGENT in number**

Every dual-use viral task has erasable excess ≤ 0 or fails the significance gate:

| task | best baseline | best gLM | excess |
|:--|:--|:--|--:|
| GUE virus_covid | k-mer 0.7282 | NT-v2 0.679 | **−0.049** |
| HVUE Pathogenecity (comp-disjoint) | CNN 0.8436 | all 3 lose | −0.038…−0.095 |
| GUE phage_fragments | CNN 0.8185 ± 0.053 | NT-v2 0.9196 ± 0.005 | +0.101, **fails gate** (≈1.9× combined SD) |
| HVUE Host_Tropism (strict, **n=96**) | CNN 0.8588 | NT-v2 FT 0.9028 | +0.058, **candidate not result** |

This is the payload. Present the single exception honestly as a candidate resting
on 96 rows.

**Gate:** viral CNN search (OPEN_RUNS R3).

---

## F7 — Some viral benchmarks structurally cannot measure generalization · **SOLID — and now demonstrated on the field's own model**

HVUE Pathogenecity retains **96 of 5,194** homology-clean test rows;
Transmissibility **60 of 4,956**. No method's score on them — ours, the original
paper's, or a viral-specialised model's — distinguishes memorisation from
generalisation.

**This is now shown directly on HViLM, the specialised viral model** (`results/hvilm/`,
`duttaprat/HViLM-Patho`, run 2026-08-30):

| head | split | n | AUROC | MCC |
|:--|:--|--:|--:|--:|
| patho | Pathogenecity, identity-disjoint | 5,194 | **0.9879** | 0.8739 |
| patho | Host_Tropism, identity-disjoint | 8,390 | 0.6909 | 0.2578 |
| patho | Host_Tropism, **strict id70/cov30** | 3,391 | **0.5582** | **0.0173** |
| tropism | Host_Tropism, identity-disjoint | 8,390 | 0.7925 | 0.5758 |
| tropism | Host_Tropism, **strict id70/cov30** | 3,391 | 0.6467 | 0.3621 |
| tropism | Pathogenecity, identity-disjoint | 5,194 | 0.7531 | 0.5725 |

The specialised model scores **0.988 AUROC on the split the benchmark calls
disjoint and 0.558 — chance — once homology is actually removed** (MCC 0.017).
That single comparison does more work than any of our own model runs: it is the
defect, on the field's own SOTA model, with the benchmark's own labels.

**Consequences for the paper:**

1. **Lead the viral section with this**, not with our model runs.
2. **The "we did not audit the SOTA viral model" caveat is obsolete** — delete it.
   The claim is now "including the specialised viral model", not "on these
   benchmarks".
3. Missing cell: there is **no strict-split Pathogenecity HViLM run**. It is the
   most valuable single remaining run, though n≈96 may be too small to be
   conclusive — report the CI either way. See OPEN_RUNS R5.

---

## F8 — Generative axis · **NOT PUBLISHABLE AS WRITTEN — the Evo run is broken**

Intended claim: a 16k-count Markov model beats Evo-1-8k in in-distribution
bits/base, with Evo above the random-DNA floor, and all models tying that floor
out of distribution.

**The Evo number does not exist.** From
`results/viral_likelihood/fullctx8000_cap200__evo_hyenadna.json`:

```json
"indist": { "markov_order7": 1.969, "evo": 0.0, "hyenadna": 1.9716 },
"evo_bases_scored": [0, 0],
"evo_skipped":      [100, 100]
```

Evo scored **zero bases and skipped all 100 sequences**. The reported
`evo_vs_markov_indist = -1.969` is an artifact of the zero, not a measurement.
Any statement that Markov beats Evo is currently unsupported.

Second problem: every Markov order sits at **1.956–1.99 bits/base** and HyenaDNA
at 1.97–2.00, against a uniform-DNA floor of 2.0. So on this corpus *nothing* —
including the Markov models — gets meaningfully below the floor. The only entry
that does is `generator12` at **1.891/1.887**. The honest version of the finding
is therefore *"no evaluated model, including a high-order Markov chain, achieves a
meaningful likelihood advantage over uniform composition on viral sequence"* —
which is a different sentence from the one drafted, and does not rank Evo at all.

**Do one of two things:** fix the Evo scoring path and re-run (OPEN_RUNS R6), or
drop Evo from the claim and report the floor result with `generator12` as the only
model below it. Either way this is **corroborating support**, not a headline —
three 2026 papers reach a compatible conclusion.

---

## The complication to report, not hide

**Homology filtering hurts baselines more than gLMs.** Filtering degraded
baselines 4–5× more than gLMs, and the gLM advantage was *equal* on clean and
leaked tasks. This contradicts the naive "leaky benchmarks inflate foundation
models" framing.

Report it prominently. It makes the paper more honest and the viral null
*stronger*: it is the baselines, not the model, that exploit leakage — so the
absence of a viral advantage holds even after accounting for that. A harness that
produced only findings flattering to its own thesis would be the less credible
one.

---

## Venue and scope — an honest assessment

This is a rigorous **benchmark-validity and capability-audit paper** for a
computational-biology or ML-for-bio venue (Bioinformatics, Genome Biology, NAR
Genomics, NeurIPS Datasets & Benchmarks). It is **not** a top-tier general-ML
main-track paper: the bare "CNNs match FMs" outcome is already published at least
three times (2411.02796, 2606.30140, DART-Eval), and no amount of rigour changes
that a general-ML referee will call it known.

What makes it publishable despite the overlap, in descending order of strength:

1. **F7 + HViLM** — the specialised viral model at 0.99 → chance across a
   homology fix. Nobody else has this.
2. **F2** — the homology-tool defect, immediately reusable by the field.
3. **F3** — receptive field as the mechanism behind prior weak-baseline results.
4. **F1** — the lever hierarchy, quantified and ranked.
5. **F4** — alignment as the missing comparator (once unblocked).
6. The viral/biosecurity framing and the unlearning consequence.

A strong domain paper with one genuinely striking result, not a landmark. Worth
being at peace with.

**Title-level framing:**

> Evaluation method dominates model choice in genomic foundation-model
> benchmarking: a quantified lever hierarchy, a homology-audit defect, and the
> absence of model-specific viral capability.
