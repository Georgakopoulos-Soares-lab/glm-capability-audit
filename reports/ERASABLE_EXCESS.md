# Finalised erasable excess — 22 GENEB tasks, 3 seeds both arms

| task | cat | k-mer | CNN | binds | best gLM | model | n | **excess** | 2·SD | sig |
|:--|:-:|--:|--:|:-:|--:|:--|--:|--:|--:|:-:|
| Plant:lncrna.g_max | sent | 0.1431 | 0.3750 | CNN | 0.5116 | NT-v2 | 3 | **+0.1366** | 0.0949 | **YES** |
| GUE:phage_fragments | sent | 0.6290 | 0.8185 | CNN | 0.9196 | NT-v2 | 3 | **+0.1011** | 0.1071 | no |
| iDHS-EL_DNase_I | sent | 0.6411 | 0.5900 | k-mer | 0.7054 | DNABERT-2 | 3 | **+0.0643** | 0.0894 | no |
| GUE:mouse_0 | sent | 0.4617 | 0.5935 | CNN | 0.6359 | NT-v2 | 3 | **+0.0424** | 0.0519 | no |
| NT:promoter_all | sent | 0.8227 | 0.9245 | CNN | 0.9557 | NT-v2 | 3 | **+0.0312** | 0.0051 | **YES** |
| NT:H3K36me3 | hist | 0.4055 | 0.6290 | CNN | 0.6582 | DNABERT-2 | 3 | **+0.0292** | 0.0146 | **YES** |
| GB:coding_vs_interg | sent | 0.7620 | 0.8869 | CNN | 0.9113 | NT-v2 | 3 | **+0.0244** | 0.0045 | **YES** |
| GB:human_or_worm | sent | 0.8819 | 0.9342 | CNN | 0.9550 | DNABERT-2 | 3 | **+0.0208** | 0.0041 | **YES** |
| NT:splice_acceptors | sent | 0.3977 | 0.9514 | CNN | 0.9656 | NT-v2 | 3 | **+0.0142** | 0.0134 | **YES** |
| NT:H3 | sent | 0.6564 | 0.7833 | CNN | 0.7937 | NT-v2 | 3 | **+0.0104** | 0.0488 | no |
| GUE:human_tf_0 | sent | 0.6085 | 0.6727 | CNN | 0.6776 | DNABERT-2 | 3 | **+0.0049** | 0.0531 | no |
| NT:enhancers | sent | 0.4214 | 0.5539 | CNN | 0.5550 | NT-v2 | 3 | **+0.0011** | 0.1005 | no |
| GB:human_ensembl_regulator | sent | 0.3891 | 0.9097 | CNN | 0.9082 | NT-v2 | 3 | **-0.0015** | 0.0064 | no |
| NT:H3K4me1 | hist | 0.3058 | 0.5497 | CNN | 0.5467 | NT-v2 | 3 | **-0.0030** | 0.0092 | no |
| NT:H4 | hist | 0.6873 | 0.7974 | CNN | 0.7879 | DNABERT-2 | 1 | **-0.0095** | 0.0154 | no |
| deep4mc_A.thaliana_4mC | sent | 0.2061 | 0.6607 | CNN | 0.6477 | Hyena | 3 | **-0.0130** | 0.0072 | no |
| NT:H3K79me3 | hist | 0.5338 | 0.7060 | CNN | 0.6662 | DNABERT-2 | 3 | **-0.0398** | 0.0431 | no |
| NT:H3K9ac | hist | 0.3909 | 0.6303 | CNN | 0.5832 | DNABERT-2 | 3 | **-0.0471** | 0.0422 | no |
| NT:H4ac | hist | 0.2905 | 0.5938 | CNN | 0.5413 | DNABERT-2 | 3 | **-0.0525** | 0.0609 | no |
| NT:H3K14ac | hist | 0.3007 | 0.5957 | CNN | 0.5416 | NT-v2 | 3 | **-0.0541** | 0.0322 | no |
| NT:H3K4me3 | hist | 0.2054 | 0.5617 | CNN | 0.4830 | DNABERT-2 | 3 | **-0.0787** | 0.0559 | no |
| NT:H3K4me2 | hist | 0.2708 | 0.5409 | CNN | 0.4172 | DNABERT-2 | 3 | **-0.1237** | 0.0994 | no |

**sentinel** (n=13): positive 11/13, significant **5/13**, median +0.0208, mean +0.0336

**histone** (n=9): positive 1/9, significant **1/9**, median -0.0471, mean -0.0421

**ALL 22** (n=22): positive 12/22, significant **6/22**, median +0.0030, mean +0.0026

k-mer binds (beats the CNN) on 1/22 tasks.

---

## Method

`excess = best full-FT gLM (dev-selected over {NT-v2-500M, DNABERT-2, GENA-LM, HyenaDNA} x
{1e-5, 3e-5, 1e-4}) - max(k-mer, swept CNN)`

Both arms OUR RUN, one harness, stratified 15% dev carve (seed 42), selection on dev only,
3 seeds each side. CNN = three-stage search over arch x capacity x kernel{5,9,15,21} x
dilation-base{1,2,4} x LR, 42-56 cells/task. k-mer = 4-mer frequencies + StandardScaler +
C swept on dev + class_weight=balanced, refit on full train.

Significance gate (pre-declared): `excess > 2*sqrt(sd_cnn^2 + sd_fm^2)` over seeds.

**Why this quantity.** Only the excess is a candidate for weight-space unlearning. Anything a
k-mer logistic regression or a from-scratch CNN recovers from the same labelled data is not held
uniquely in the pretrained weights — deleting it from the weights leaves the capability fully
reconstructable by a baseline, so the removal is cosmetic.

## Corrections this run forced on earlier numbers

| task | earlier | finalised | cause |
|:--|--:|--:|:--|
| NT:enhancers | +0.0561 | **+0.0011** | earlier gLM was 1 seed (0.6100); 3-seed mean is 0.5550 |
| iDHS-EL_DNase_I | +0.1154 | **+0.0643**, not sig | k-mer re-measured at **0.6411 > CNN 0.5900** — k-mer binds |
| GUE:phage_fragments | +0.1025 "survives" | **+0.1011, fails the gate** | 3-seed CNN SD is 0.053; gap is ~1.9x combined SD |
| GUE:mouse_0 | +0.0429 | +0.0424, not sig | 3-seed |

The k-mer re-measurement matters beyond `iDHS-EL`: the previously reported k-mer column was
understated by up to +0.072 (H3 0.5899 -> 0.6564, human_or_worm 0.8122 -> 0.8819,
human_tf_0 0.5368 -> 0.6085). Baselines were re-measured with one extractor across all 22 tasks
so the `max(k-mer, CNN)` floor is consistent.

## What survives

**6 of 22 tasks** clear the gate: Plant lncRNA (+0.137), promoter_all (+0.031), H3K36me3 (+0.029),
coding-vs-intergenomic (+0.024), human-or-worm (+0.021), splice acceptors (+0.014).

All six are ordinary regulatory / species-discrimination genomics. **No dual-use task is on the
list.** The category split is stark: sentinel 5/13 significant at median +0.021; histone 1/9 at
median **-0.047**.

## The viral picture is unchanged and negative

| viral / dual-use task | best baseline | best gLM | excess |
|:--|:--|--:|--:|
| ViroBench taxonomy (whole genome) | k-mer3-6 **0.9475** | NT-v2 0.8867 | **-0.061** |
| ViroBench @ 3.1kb matched context | k-mer 0.8469 | GENA-LM 0.6378 | **-0.209** |
| ViroBench @ 20kb matched context | k-mer 0.9297 | HyenaDNA 0.8784 | **-0.051** |
| GUE virus_covid | k-mer 0.7282 | NT-v2 0.679 | **-0.049** |
| HVUE Pathogenecity (comp-disjoint) | CNN 0.8436 | all 3 lose | -0.038…-0.095 |
| HVUE Host_Tropism (strict, **n=96**) | CNN 0.8588 | NT-v2 FT 0.9028 | +0.058 |
| GUE:phage_fragments | CNN 0.8185 +- 0.053 | NT-v2 0.9196 +- 0.005 | +0.101, **borderline** |

`phage_fragments` is the only viral-domain task with a large positive point estimate, and it fails
the gate **because of CNN seed variance, not gLM variance** (gLM SD 0.005 vs CNN SD 0.053). More
CNN seeds would resolve it; at 3 seeds it is under-powered, not refuted.

## Implication for the unlearning programme

For the capabilities that motivate genomic-model biosecurity concern, there is **no measurable
model-specific excess over composition baselines to remove.** This is a stronger and more useful
statement than "unlearning is hard": it says the localise-and-excise framing is mis-specified for
this domain, because the capability is recoverable from data statistics rather than held uniquely
in pretrained weights.

The six significant tasks are where a weight-space unlearning method could be *validated* against
a real, non-zero target. `phage_fragments` is the one candidate that is both viral-domain and
plausibly non-zero, and it needs more CNN seeds before it can carry weight.

---

## Why `phage_fragments` fails the gate — an early-stopping artifact, not task noise

Per-seed trace of the dev-selected CNN cell (dilated 0.68M, k=9, d=4, RF 2729):

| seed | dev | test | best_epoch |
|--:|--:|--:|--:|
| 42 | 0.8604 | **0.8679** | 13 |
| 43 | 0.8147 | 0.8256 | 7 |
| 44 | 0.7888 | **0.7621** | 6 |

Seeds 43/44 tripped patience-6 at epochs 6-7 while seed 42 kept improving to 13. They are
**under-trained, not differently converged.** The surrounding cell landscape contradicts the
0.8185 mean outright — the six best cells by dev all test **0.840-0.868** at one seed each:

| arch | params | k | d | RF | dev | test |
|:--|--:|--:|--:|--:|--:|--:|
| dilated | 0.68M | 9 | 4 | 2729 | 0.8604 | 0.8679 |
| dilated | 0.41M | 5 | 4 | 1365 | 0.8561 | 0.8403 |
| dilated | 1.47M | 5 | 4 | 1365 | 0.8557 | 0.8528 |
| dilated | 2.53M | 9 | 4 | 2729 | 0.8547 | 0.8498 |
| dilated | 9.33M | 9 | 4 | 10921 | 0.8529 | 0.8507 |
| dilated | 1.49M | 21 | 4 | 6821 | 0.8503 | 0.8540 |

So the CNN's converged level on this task is ~0.85, the +0.101 excess is probably an
**over**-estimate (true value ~+0.07 vs NT-v2's 0.9196 +- 0.005), and the 0.053 SD that sinks it
is manufactured by the stopping rule. Fix = raise `min_epochs`/patience and add seeds; do not
report +0.101 either way.

## What predicts gLM excess — one suggestive pattern, not an established finding

Spearman against excess over all 22 tasks:

| predictor | rho | p |
|:--|--:|--:|
| **CNN-over-k-mer** (`cnn - kmer`) | **-0.511** | 0.015 |
| **k-mer level** | **+0.483** | 0.023 |
| baseline level `max(kmer,CNN)` | +0.282 | 0.20 |
| n_train | -0.098 | 0.67 |
| seqlen | -0.027 | 0.90 |
| n_classes | -0.052 | 0.82 |

**Reading.** gLM excess is largest where the task is already solvable from *composition*
(high k-mer) and smallest where a from-scratch CNN adds a lot over composition — i.e. where the
signal is **positional / motif-structured**, exactly what a convolution is built for. Splice
acceptors (+0.554 CNN-over-k-mer), ensembl regulatory (+0.521), 4mC (+0.455) and the histone
marks (+0.22 to +0.36) are the low/negative-excess tasks; promoter_all (k-mer 0.823),
human_or_worm (0.882) and coding-vs-intergenomic (0.762) are the reliable positive ones.

**Three caveats, all load-bearing:**

1. **Neither correlation survives multiple-comparison correction.** 8 predictors tested,
   Bonferroni alpha = 0.00625; p = 0.015 and 0.023 both fail. This is a hypothesis generated
   from 22 points, not a result.
2. **The two predictors are algebraically linked** (`cnn_over_kmer = cnn - kmer`). One pattern
   reported twice, not independent evidence.
3. **Scope: it holds only where the model can see the whole input** (these 22 tasks are
   41-4000bp). ViroBench breaks it — k-mer 0.9475 (maximally compositional) yet excess **-0.061**
   — because there the binding constraint is the 43.5kb input exceeding every context window, not
   the feature type. **These are two distinct gLM failure regimes and should not be merged.**

## Effect size vs statistical power — read the 6 "significant" tasks carefully

| task | excess | 2·SD | ratio | k-mer | CNN-over-k-mer |
|:--|--:|--:|--:|--:|--:|
| NT:promoter_all | +0.0312 | 0.0051 | 6.10 | 0.823 | +0.102 |
| GB:coding_vs_interg | +0.0244 | 0.0045 | 5.42 | 0.762 | +0.125 |
| GB:human_or_worm | +0.0208 | 0.0041 | 5.03 | 0.882 | +0.052 |
| NT:H3K36me3 | +0.0292 | 0.0146 | 2.00 | 0.406 | +0.223 |
| Plant:lncrna.g_max | +0.1366 | 0.0949 | 1.44 | 0.143 | +0.232 |
| NT:splice_acceptors | +0.0142 | 0.0134 | 1.06 | 0.398 | +0.554 |

Four of the six have excess of only **+0.014 to +0.031**. They clear the gate because their test
sets are large and seed SDs tiny (ratio 5-6), **not** because the effects are large. Only Plant
lncRNA is a large effect (+0.137), and it barely clears (ratio 1.44). "6/22 significant" must not
be read as "6 solid capability wins."

## How this translates for unlearning — the anti-correlation is the whole result

The excess is concentrated **precisely where the capability is most redundantly recoverable from
composition.** Delete NT-v2's +0.031 on promoter_all and a 4-mer logistic regression still scores
**0.823**. Delete its +0.021 on human_or_worm and k-mer still scores **0.882**.

So *erasable excess* and *residual reconstructability* are anti-correlated across these 22 tasks:
the tasks with something to remove are the tasks where removal changes least, and the tasks where
removal would matter (positional/motif signal) have nothing model-specific to remove.

That is the sharpest form of the negative result, and it is a claim about the target rather than
about any unlearning method — so it cannot be answered by trying a better method.

---

## Target quality — the two properties an unlearning target needs, and how they trade off

A usable target needs **both** (a) a measurable excess and (b) low *residual reconstructability* —
if a baseline still recovers the capability after you erase the excess, the removal is cosmetic.

`recoverability = max(k-mer, CNN) / gLM` = the fraction of gLM performance a baseline still
reaches once the model-specific part is gone.

| task | excess | ratio (excess/2SD) | k-mer | baseline | gLM | **recoverability** | n_test | n_train | sig |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| **Plant:lncrna.g_max** | **+0.1366** | 1.44 | 0.143 | 0.375 | 0.512 | **73.3%** | 490 | 9,715 | **YES** |
| GUE:phage_fragments | +0.1011 | 0.94 | 0.629 | 0.819 | 0.920 | 89.0% | 4,630 | 31,487 | no |
| iDHS-EL_DNase_I | +0.0643 | 0.72 | 0.641 | 0.641 | 0.705 | 90.9% | 306 | 604 | no |
| GUE:mouse_0 | +0.0424 | 0.82 | 0.462 | 0.594 | 0.636 | 93.3% | 810 | 5,506 | no |
| NT:H3K36me3 | +0.0292 | 2.00 | 0.406 | 0.629 | 0.658 | 95.6% | 3,488 | 26,683 | **YES** |
| NT:promoter_all | +0.0312 | **6.10** | 0.823 | 0.924 | 0.956 | 96.7% | 5,920 | 45,284 | **YES** |
| GB:coding_vs_interg | +0.0244 | **5.42** | 0.762 | 0.887 | 0.911 | 97.3% | 25,000 | 63,750 | **YES** |
| GB:human_or_worm | +0.0208 | **5.03** | 0.882 | 0.934 | 0.955 | 97.8% | 25,000 | 63,750 | **YES** |
| NT:splice_acceptors | +0.0142 | 1.06 | 0.398 | 0.951 | 0.966 | 98.5% | 3,000 | 25,500 | **YES** |
| NT:H3 | +0.0104 | 0.21 | 0.656 | 0.783 | 0.794 | 98.7% | 11,447 | 11,447 | no |

**The trade-off is the finding.** The three most statistically robust excesses (ratio 5-6) are all
**>=96.7% recoverable** — erase them and a baseline loses ~3% of the capability. The one task where
removal would genuinely cost capability (**73.3%**) is Plant lncRNA, and it is under-powered
(ratio 1.44, n_test 490).

So the profile of a *good* target is: **large excess, low k-mer, weak baselines overall, large
enough test set to resolve it.** Plant lncRNA hits the first three and fails the fourth.

### The obvious next move: the 5 sibling lncRNA tasks

GENEB ships **six** plant lncRNA tasks — `g_max` (ours) plus `m_esculenta`, `s_bicolor`,
`s_lycopersicum`, `t_aestivum`, `z_mays`. If +0.137 at 73% recoverability replicates across six
species, that is a replicated capability finding with a real removal target, and it converts the
single under-powered result into the paper's one positive claim. They are in the prospective
screen.

---

## Missing-comparator check on the lncRNA task (the one positive result)

`lncrna.g_max` is coding-vs-non-coding, so a reading-frame/ORF feature is an obvious comparator
that neither k-mer nor a CNN represents directly. If it beat the CNN, the +0.137 would be a
missing-comparator artifact -- the same error already caught with alignment on ViroBench.

Class means (train) confirm the label semantics — **label 0 = coding (mRNA), label 1 = lncRNA**:

| label | longest ORF | ORF/length | seq length | starts with ATG |
|:--|--:|--:|--:|--:|
| 0 (coding) | 305.3 bp | 0.577 | 643.6 | 0.459 |
| 1 (lncRNA) | 258.8 bp | 0.520 | 641.4 | 0.256 |

Baseline built from 4 ORF features (longest ORF, ORF/length, length, starts-with-ATG), C swept
on dev:

| comparator | test MCC |
|:--|--:|
| k-mer 4-mer | 0.1431 |
| **ORF features** | **0.2696** |
| swept CNN | **0.3750** |
| NT-v2-500M full FT | 0.5116 |

**The check passes.** ORF beats k-mer by +0.127 but does not displace the CNN, so
`max(k-mer, CNN, ORF)` is unchanged at 0.375 and the excess stays **+0.1366**.

Note the class lengths are near-identical (643.6 vs 641.4), so sequence length is not a shortcut
here — and the weak ORF signal (0.27) is itself informative: these plant transcripts are not
cleanly separable by reading-frame structure, which is consistent with this being a genuinely
hard task where pretraining has room to contribute.
