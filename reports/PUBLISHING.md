# How to publish this — direction, strengths, and the honest limits

Written for whoever writes the manuscript. Read [FINDINGS.md](FINDINGS.md) first
for the claim ledger; this document is about *strategy*: which venue, which
findings carry the paper, what order to write them in, and which attacks to
pre-empt.

---

## 1. What this paper is

A **benchmark-validity and capability-audit paper.** The contribution is not a new
model or a new benchmark — it is a demonstration that the evidence behind an
existing capability claim does not survive fixing the baselines and the splits,
shown with a harness that provably detects capability elsewhere.

**One-sentence version:**

> gLMs work, and here is a domain where the evidence that they work does not
> survive contact with the baselines and the splits.

That framing is deliberately narrow, and it is what makes the paper defensible.
The version to avoid is "gLMs don't work" — it is false, our own positive controls
refute it, and it invites the one attack that would sink the paper ("your harness
is broken").

**Title-level framing:**

> Evaluation method dominates model choice in genomic foundation-model
> benchmarking: a quantified lever hierarchy, a homology-audit defect, and the
> absence of model-specific viral capability.

---

## 2. Strengths, ranked by how much they carry the paper

### #1 — HViLM shows the defect on itself (F7)

The specialised viral model, `duttaprat/HViLM-Patho`, scores **0.9879 AUROC** on
the HVUE Pathogenecity split the benchmark itself labels *identity-disjoint*, and
**0.5582 — chance, MCC 0.017** — on Host_Tropism once homology is actually
removed. And it is **not a sample-size artifact**: a matched-size random subsample
of the same test set scores 0.7923 [0.7682, 0.8168], with the strict value below
that band in **100% of draws**.

This is the paper's best asset and nobody else has it. It is the benchmark-validity
defect, on the field's own SOTA model, with the benchmark's own labels, with the
obvious confound already controlled. **Lead the viral section with this**, not with
our own model runs. It also makes the "did you test the specialised model?" review
question a strength rather than a gap.

### #2 — The homology-audit tool defect (F2)

`mmseqs easy-cluster -c 0.9` requires 90% *bidirectional* coverage and is
therefore blind to partial homology. Splits it certifies as identity-disjoint are
42%+ leaked; HVUE Pathogenecity is **91–99.8% leaked at 90% identity / 50%
coverage** — the strictest setting anyone calls standard, so this cannot be
dismissed as threshold-shopping.

This is the cleanest *methods* contribution: it is a concrete, reusable correction
the field can adopt immediately, and it propagates (PathoLM used the same tool).
Reviewers reward a paper that hands them a tool.

### #3 — Receptive field, not capacity, binds baselines (F3)

On 400 bp splice a **0.149M** U-Net reaches **0.9685** while a **9.438M** ResNet
with an 89 bp receptive field reaches **0.3073** — a 0.661 MCC gap at 63× fewer
parameters. Published FM-over-CNN gaps of +0.31–0.60 collapse to +0.02–0.03 once
the baseline can see the input.

Mechanistic and generalisable: it *explains* prior weak-baseline results rather
than merely reporting another instance of them. This is the finding most likely to
be cited outside the viral context.

### #4 — The lever hierarchy (F1)

Quantified and ranked: baseline architecture 0.69, adaptation regime 0.60, split
0.37, comparator class 0.17 — against a best-model-over-best-baseline excess of
**0.01–0.015**. Method levers are 25–45× the model excess. No prior paper ranks
these.

**Handle the denominator carefully** — see the framing risk in F1. Method levers
dominate the *excess*, not the *spread* (0.32).

### #5 — Alignment as the missing comparator (F4)

Viral family assignment is solved by database lookup: alignment reaches
**0.94–0.96 macro-F1 on the queries it can align**, against a best gLM of 0.6148.
Frame it as coverage-limited alignment, which is both more accurate and more
interesting than "alignment wins".

### #6 — The scoped viral null, positively controlled (F6 + controls)

Every dual-use viral task has erasable excess ≤ 0 or fails the significance gate,
across all five ViroBench taxonomic levels with pre-declared equivalence margins.
The null is credible *because* the same harness finds GENA-LM winning 11 of 13
GENEB categories, splice FT reproducing published numbers at 0.968 MCC, and our
DNABERT-2 full-FT beating the *published* DNABERT-2 on the identical checkpoint by
**+0.0482**.

That last number is worth stating explicitly in the methods: it is a direct check
that the harness is not merely competent but slightly better than the reference
implementation.

---

## 3. Venue — an honest assessment

**Target:** Bioinformatics, Genome Biology, NAR Genomics and Bioinformatics, or
NeurIPS Datasets & Benchmarks.

**Not** a top-tier general-ML main track. The bare "CNNs match foundation models"
outcome is already published at least three times (2411.02796, 2606.30140,
DART-Eval). No amount of rigour changes that a general-ML referee will call the
headline known.

What makes it publishable despite that overlap: findings #1, #2 and #3 above are
each things those three papers do not have, and the viral/biosecurity framing plus
the unlearning consequence give it a reason to exist beyond benchmark hygiene.

**This is a strong domain paper with one genuinely striking result. It is not a
landmark. Be at peace with that** — it is honest, useful, citable work, and
over-claiming is the fastest way to lose it.

---

## 4. Writing order

Do not write in the order the work happened. Write in the order that makes the
argument land.

1. **Introduction** — the claim under test. Why viral tasks specifically
   (biosecurity, unlearning motivation). State up front that capability *is* found
   on non-viral tasks, so the reader knows this is scoped, not nihilistic.
2. **What a valid capability claim requires** — the four conditions from
   [PAPER_OUTLINE.md](PAPER_OUTLINE.md) §2. This section doubles as a reviewer
   checklist and is the methodological spine.
3. **Benchmark validity: homology saturation** (F2) — lead with the tool defect,
   then the leak table, then the strict-refiltering row counts.
4. **The specialised model shows it** (F7) — HViLM 0.99 → chance, with the
   matched-size control. This is the emotional centre of the paper; give it a
   figure.
5. **Baseline validity: three failure modes** (F3, F4, and the scaling one) —
   receptive field first because it is mechanistic.
6. **The lever hierarchy** (F1) — now the reader has seen each lever measured, so
   the ranking is a synthesis rather than an assertion.
7. **The viral null** (F6) — all five taxonomic levels, equivalence margins
   pre-declared.
8. **Positive controls** (GENEB, splice, DNABERT-2 vs published) — placed here so
   the null is already on the table when you prove the harness works.
9. **The homology-filtering reversal** — see §5. Give it its own subsection.
10. **Consequences for unlearning** — [unlearning/OUTCOME_FOR_UNLEARNING.md](unlearning/OUTCOME_FOR_UNLEARNING.md).
11. **Limitations** — §6 below, stated before a referee states them for you.

Generative results (F8) are **corroborating support at most**, and currently not
publishable at all (the Evo run is broken). Do not build a section on them.

---

## 5. The complication to report, not hide

**Homology filtering hurts baselines more than gLMs.** Filtering degraded
baselines 4–5× more than gLMs, and the gLM advantage was *equal* on clean and
leaked tasks. On GENEB, most apparent homology effects sit *inside* a matched-size
control band (`results/homology_threshold_replication.json`) — meaning the drop is
largely a test-set-size effect, not homology.

This **contradicts** the naive "leaky benchmarks inflate foundation models"
framing, and you must report it.

It also makes the paper better. It means the viral null is *stronger*, not weaker:
it is the baselines, not the model, that exploit leakage — so the absence of a
viral model advantage holds even after accounting for that. And a harness that
produced only findings flattering to its own thesis would be the less credible
one. Referees notice when a paper reports the result that cuts against it.

---

## 6. Limitations to state before a referee does

- **The lever comparison spans task families and metrics** (MCC on splice,
  macro-F1 on ViroBench). Either normalise or say plainly it is indicative of
  magnitude, not a within-task decomposition.
- **HVUE Pathogenecity/Transmissibility retain 96 and 60 clean test rows.** That is
  the finding, but it also means *our* numbers there are equally unmeasurable. Say
  so.
- **The one candidate positive** (HVUE Host_Tropism, +0.058) rests on 96 rows.
  Present it as a candidate, never a result.
- **Frozen-probe layer sweep** was done only on ViroBench family; elsewhere the
  read-out is `L−1`, worth 0.02–0.04 macro-F1.
- **Never run:** ESM-2/ESM-1v, Caduceus, Evo-2, GenoJEPA, protein one-hot +
  biochemical baselines. HViLM and DNABERT-2 *were* run.
- **GENEB remainder:** roughly 66 of 100 tasks untouched.
- **No Evo generative result exists** — the run scored zero bases.

---

## 7. Before submission

Blocking, in order — see [OPEN_RUNS.md](OPEN_RUNS.md):

1. **Number reconciliation** (no GPU). Several quoted numbers do not match the
   committed artifacts, including F3's headline splice figures. Adopt the rule:
   *every number in the manuscript carries the artifact path it came from.*
2. **NT:H4 / DNABERT-2 seeds 43, 44** — one single-seed cell in a table headed
   "3 seeds both arms" is exactly what a referee circles.
3. **Reconcile GENA-LM** — 0 of 22 wins in `ERASABLE_EXCESS.md` versus "11 of 13
   categories" elsewhere. Different regimes; say so explicitly.
4. **Strict-split Pathogenecity HViLM** — completes the 0.99 → ? comparison on one
   task. Report the CI even if n≈96 is too small to conclude.
5. **`results/baseline_selection_audit.csv` has `bb_val == bb_test`.** If the
   baseline is selected on dev and scored on test those must differ. Selecting on
   test is precisely the error this paper accuses others of — resolve it or do not
   cite the file.
6. **Evo likelihood: fix or drop.**

Also decide repository visibility. The repo currently contains the full findings
register, including claims that do not yet match the artifacts. That is correct for
a handoff and arguably good practice for an audit trail — but consider keeping it
private until items 1–3 land, then flipping it public at submission.
