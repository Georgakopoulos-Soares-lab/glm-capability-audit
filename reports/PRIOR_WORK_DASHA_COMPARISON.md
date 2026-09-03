# Head-to-head with arXiv:2411.02796 (DASHA, ICLR 2025)

**Xu, Gupta, Cheng, Shen, Shen, Talwalkar, Khodak — "Specialized Foundation Models Struggle to
Beat Supervised Baselines."** All DASHA/FM numbers below are **EXTERNAL/PUBLISHED**, transcribed
from their Tables 8-11. All CNN/gLM numbers labelled *ours* are **OUR RUN**.

Verification artefacts: `scratchpad/dasha_per_task.csv`, `scratchpad/nt_bench_leakage.csv`,
`scratchpad/cnn_enhancers_leaksplit.json`.

---

## 0. Bottom line — the "we contradict them" claim does not survive

I previously recorded this paper as the one place our GENEB results **contradict** prior work.
That was wrong, and the reason is task selection, not numbers.

- Our 13-task sentinel set overlaps their 18-task benchmark on **4 tasks**; only **3** are
  metric- and dataset-comparable.
- All 3 sit in **regulatory elements / RNA production** — the category where **DASHA itself
  reports losing to the FMs on 0/8 tasks won**.
- DASHA's headline rests on **histone modification** (7/10 won, mean +0.053). We ran **1 of
  those 10** (H3), and H3 is one of the **3 histone tasks DASHA also loses** (−0.025).

**We do not contradict them. On the overlap we agree with them. We have not tested their claim.**

---

## 1. Are the splits the same? Yes — byte-identical

GENEB ships the official InstaDeepAI release verbatim; we carve a stratified 15% dev from train
(seed 42), they use a 0.1 validation split. Test sets are untouched by both.

| task | HF test n | GENEB test n | test-seq overlap | HF train n | GENEB train n |
|:--|--:|--:|--:|--:|--:|
| H3 | 1497 | 1497 | **1497 (100%)** | 13468 | 13468 |
| enhancers | 400 | 400 | **400 (100%)** | 14968 | 14968 |
| promoter_all | 5920 | 5920 | **5920 (100%)** | 53276 | 53276 |

## 2. Which of the 4 overlapping tasks are actually comparable

| our sentinel task | dataset version | their version | their metric | our metric | comparable? |
|:--|:--|:--|:--|:--|:--|
| `..._H3` | v1, 13468/1497 | same | MCC | MCC | **yes** |
| `..._enhancers` | v1, 14968/400 | same | MCC | MCC | **yes** (but see §5) |
| `..._promoter_all` | v1, 53276/5920 | same | F1 in Tbl 8; **MCC 0.92 in Tbl 11** | MCC | **yes, via Tbl 11** |
| `..._revised_splice_sites_acceptors` | **revised**, 30000/3000 | **v1**, 19961/2218 | F1 | MCC | **no — different dataset** |

## 3. Our CNN reproduces their tuned CNN — the baseline is not a strawman

This is the check that closes the "your CNN was under-searched" attack.

| task (MCC) | their WideResNet | their UNet | **their DASHA** | **our CNN** (3 seeds) | our CNN − DASHA |
|:--|--:|--:|--:|--:|--:|
| H3 | 0.798 | 0.797 | 0.790 | **0.7833** ±0.019 | −0.007 |
| enhancers | 0.525 | 0.490 | 0.527 | **0.5539** ±0.048 | **+0.027** |
| promoter_all | 0.952 | — | 0.92 *(Tbl 11)* | **0.9245** ±0.001 | +0.005 |

Our three-stage search (arch × capacity × kernel{5,9,15,21} × dilation-base{1,2,4} × LR, ~49-56
cells/task) lands within **−0.007 to +0.027** of their DASH+ASHA workflow. Different search
machinery, same ceiling.

## 4. Where DASHA's claim actually lives — and it is not where we looked

DASHA vs the **best FM they cite**, recomputed per task from their Tables 8-9:

| category | DASHA wins | mean Δ vs best cited FM |
|:--|:--|--:|
| regulatory elements / RNA production (8 tasks) | **0/8** | **−0.013** |
| histone modification (10 tasks) | **7/10** | **+0.053** |
| all 18 | 7/18 | +0.023 |

Biggest DASHA wins: H3K4me3 **+0.164**, H3K4me2 **+0.153**, H4ac **+0.123**, H3K9ac +0.087.
Their own text agrees: *"our strong performance is driven in large part by outstanding
performance on the histone modification tasks"*; *"in the promoters task it generally performs
worse than all reported FMs."*

**Two further protocol asymmetries in their comparison:**

1. **They never ran the FMs.** Their FM column is copied from Dalla-Torre et al. 2023
   Supplementary Table 6 (InstaDeep's own fine-tuning, 10-fold CV). Their CNN is their own run
   on a 0.1 val split. It is a cross-protocol comparison — the thing `PROTOCOL.md` forbids us.
2. **No seeds reported** for the genomics tables; per-task numbers are single point estimates.
   Our enhancers CNN has a **seed SD of 0.048** on the same task — larger than the entire
   0.032 gap they report there.

**Our matched-protocol result (both sides OUR RUN, 3 seeds, one harness) is therefore
complementary to theirs, not contradictory — and it is the comparison their design cannot make.**

## 5. New finding — the original NT benchmark has class-asymmetric train/test leakage

Exact-duplicate audit over all 36 NT task variants on disk (`nt_bench_leakage.csv`):

| task | test rows verbatim in train | % of test | label agreement | class of leaked rows |
|:--|--:|--:|--:|:--|
| **enhancers** | **100 / 400** | **25.0%** | 100% | **all class 1** (test is 200/200) |
| **enhancers_types** | 100 / 400 | 25.0% | 100% | — |
| **splice_sites_all** | 221 / 3000 | 7.4% | 100% | 110/99/12 |
| promoter_tata | 6 / 621 | 1.0% | 100% | all class 1 |
| promoter_all | 21 / 5920 | 0.4% | 100% | all class 1 |
| promoter_no_tata | 15 / 5299 | 0.3% | 100% | all class 1 |
| all 10 histone tasks | 0 | 0% | — | — |
| **all 20 `revised_*` tasks** | **0** | **0%** | — | — |

On `enhancers`, **half of the positive class and none of the negative class is verbatim in
train.** That is a directional shortcut, not noise. The `revised_*` variants are clean, which is
presumably why they exist — but DASHA, the NT paper, Caduceus and DNABERT-2 all report on the
**original** set (their Table 7 lists 14968/400 and 19961/2218, the v1 sizes).

### Re-scoring `enhancers` with the 100 leaked rows removed (all OUR RUN)

| model | MCC full (n=400) | MCC clean (n=300) | Δ | acc on 100 leaked | acc on clean positives |
|:--|--:|--:|--:|--:|--:|
| NT-v2-500M | 0.6100 | **0.4358** | −0.174 | 0.98 | 0.64 |
| GENA-LM | 0.5659 | 0.3752 | −0.191 | 0.99 | 0.63 |
| **CNN** (dilated 0.41M, 3 seeds) | 0.5539 | 0.3562 | −0.198 | 0.91 | 0.48 |
| DNABERT-2 | 0.5457 | 0.3523 | −0.193 | 0.97 | 0.60 |
| HyenaDNA | 0.4408 | 0.2267 | −0.214 | 0.97 | 0.57 |

Every model scores **0.91-0.99 on the memorizable rows and 0.39-0.64 on clean positives**.
Removing 25% of the test set costs **0.17-0.21 MCC — roughly 35% of the reported score — for
every architecture, pretrained or not.**

Two honest readings, both worth stating:
- **Against the benchmark:** a third of the headline number on this task is recall of training
  rows. Any paper reporting NT-`enhancers` inherits this.
- **Against ourselves, and it cuts the other way:** the ranking is preserved, and NT-v2's margin
  over the CNN **grows** from +0.056 to +0.080 once the leak is removed. The leakage was
  *compressing* the gLM advantage here, not manufacturing it.

## 6. What this changes

**Retract:** "we contradict 2411.02796." Replace with: *on the 3 comparable tasks we reproduce
their tuned-CNN level (−0.007/+0.027/+0.005) and agree with their finding that CNNs trail FMs in
regulatory/RNA tasks; their positive claim rests on 10 histone tasks we have not run.*

**Keep and strengthen:** ours is the only **matched-protocol** FM-vs-CNN comparison — both arms
run in one harness, 3 seeds, dev-only selection. Theirs is OUR-RUN-CNN vs PUBLISHED-FM, single
run. That is a methodological edge, not a contradiction.

**New, unclaimed:** the class-asymmetric leakage in NT v1 `enhancers`/`splice_sites_all` and the
leak-free re-scoring.

**The one experiment that would let us speak to their claim:** the **9 remaining histone tasks**
(H3K14ac, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K79me3, H3K9ac, H4, H4ac). All are already on
disk under `/data/nvidia/geneb_data/tasks/`, all leak-free, all 13k-33k train rows at 500bp.
This is exactly where the only published "CNN beats FMs" claim lives, it has never been tested
under a matched protocol, and DASHA's margins there (+0.05 to +0.16) are an order of magnitude
larger than anything else in this project.

---

# 7. RESULT (2026-09-01): the 9 histone tasks ran. DASHA's claim replicates.

9 CNN sweeps (56 cells each, 3-seed refit) + 4 gLMs × 3 LRs × 9 tasks, then 3-seed refit at the
dev-selected LR. 153 model-trainings, one harness, dev-only selection. Full table:
`scratchpad/histone/RESULTS.md`.

## 7.1 The CNN wins 8 of 9

| gLM (full FT, 3 seeds) | beats our swept CNN | mean Δ | median Δ |
|:--|:--|--:|--:|
| NT-v2-500M | **0/9** | −0.0732 | −0.0642 |
| DNABERT-2 | **1/9** (H3K36me3, +0.029) | −0.0494 | −0.0471 |
| GENA-LM | **0/9** | −0.0877 | −0.1038 |
| HyenaDNA-medium | **0/9** | −0.1620 | −0.1505 |

Compare the 13-task sentinel set, same harness: NT-v2 won **11/13** there (median +0.024).
**The sign of the FM-vs-CNN comparison flips with task category, not with method.**

## 7.2 Reproduction check — the gLM arm is validated, the CNN arm is conservative

Ours vs published, **same checkpoint** where possible:

| arm | comparison | mean Δ | reading |
|:--|:--|--:|:--|
| NT-v2-500M | vs published NT-MS-500M | **+0.0005** | near-exact reproduction |
| DNABERT-2 | vs published DNABERT-2 (identical ckpt) | **+0.0482** | ours is *better* |
| GENA-LM | vs published GENA-LM (identical ckpt) | **−0.0650** | ours **under-performs** — flagged |
| our CNN | vs published DASHA (DASH+ASHA) | −0.0601 | below their NAS'd CNN |
| our CNN | vs published plain WideResNet | **−0.0249** (median −0.004) | ≈ their un-NAS'd baseline |

Two things follow. First, **the harness is not handicapping the gLMs** — NT-v2 lands on the
published number and DNABERT-2 above it. Second, **the CNN result is conservative**: our search
reaches their plain-WRN level but sits 0.06 below their NAS'd DASHA, so a stronger CNN would
widen the CNN's margin, not close it. The 8/9 is a lower bound.

The CNN search itself is healthy, not pathological: 56 cells each, dev-selected configs spanning
0.41M–21.8M params and RF 221–1365bp (the input is 500bp), dev-selected ≈ oracle-best-over-ladder,
seed SD 0.002–0.030.

**GENA-LM is the one arm to caveat.** −0.065 against its own published number, consistently
negative across 8/9 tasks. Its LR grid {1e-5, 3e-5, 1e-4} or the 8-epoch budget is likely the
cause. Any GENA-LM-specific claim here should be marked provisional; it does not affect the
headline, since GENA-LM is not the best gLM on any histone task.

## 7.3 Consequences

1. **Retracted claim stays retracted, and now with evidence.** We do not contradict 2411.02796.
   Run under a matched protocol, on the tasks where their claim lives, **their claim holds** —
   and it holds against four gLMs they never ran themselves.
2. **The real finding is the category split, and it is ours.** 22 tasks, one harness, 3 seeds:
   gLMs win 11/13 on regulatory/RNA/species tasks, lose 8/9 on histone modification. Neither
   DASHA (CNN-only runs) nor DART-Eval (probes) could see this, because neither ran both arms
   across both categories.
3. **Erasable excess is ≤ 0 on 8 of 9 histone tasks** — one more category with nothing for
   weight-space unlearning to remove. Only H3K36me3 (DNABERT-2 +0.029, ~1 SD) is even a
   candidate, and k-mer was not measured there, so that +0.029 is an upper bound.
