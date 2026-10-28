# Priority 1 — the bootstrap estimand, formally stated and re-estimated

Run 2026-10-06. CPU only, 5,000 replicates, 20 workers.
Script `bootstrap_corrected.py`, table `bootstrap_corrected.csv`.
**Nothing in the manuscript or in any frozen output was modified.**

---

## 1. The estimator the manuscript reports

For a population $R$ (a set of test rows), with the foundation model $m$ fixed by
development selection and every baseline arm's hyperparameters fixed by
development selection:

$$\mathrm{MCC}_p(R) \;=\; \frac{1}{|S_p|}\sum_{s \in S_p} \mathrm{MCC}\!\left(y_R,\; \hat y_{p,s}(R)\right)$$

$$\mathrm{Env}(R) \;=\; \max_{a \in \mathcal{A}(R)} \mathrm{MCC}_a(R), \qquad
\Delta(R) \;=\; \mathrm{MCC}_m(R) - \mathrm{Env}(R)$$

where $S_p$ is predictor $p$'s set of training seeds and $\mathcal{A}(R)$ is the
set of *applicable* arms — nearest-neighbour lookup is **removed from the set the
maximum is taken over** at L1 and L2, never encoded as zero. The reported point
estimate is $\Delta(R_{\text{full}})$.

**Validation.** Implementing this independently reproduces, on all 20 scored
cells, the manuscript's `model_mcc`, `envelope_mcc`, `envelope_arm` and `excess`
**exactly**. The estimator above is therefore the one the paper reports.

## 2. The defect

`final_regime_table.py:161-162` draws **one random seed per arm** inside each
replicate and then maximises, while the point estimate averages seeds before
maximising. Maximising over noisier single-seed quantities inflates the envelope
by Jensen's inequality, so the published interval is an interval for a different
population quantity than the number printed beside it. Measured earlier on GUE
phage L0 the offset is **+0.0187** and does not shrink with $n$.

## 3. Two defensible repairs, and they disagree

| scheme | replicate statistic | estimand |
|---|---|---|
| **conditional** | $\Delta$ applied to the resampled rows, seeds averaged exactly as in the point estimate | the difference *for these specific trained predictors* on the population |
| **two-level** | additionally resamples each predictor's seed set with replacement before averaging | the difference for a $k$-seed-trained predictor *of this type* |

The work order specifies seed aggregation matching the point estimator, which is
the conditional scheme. But the manuscript's Methods (`main.tex:104`) state the
intent explicitly — *"a training seed drawn at random for every method so that
seed variance is propagated"* — and the conditional scheme propagates **no** seed
variance at all. The two-level scheme is the correct implementation of that
stated intent; the shipped code attempted it and got it wrong in both centring
and width.

## 4. Results

**Point estimates and selected arms: unchanged on every cell.** No headline
number moves.

### Fragment verdicts

The conditional scheme changes five verdicts relative to the manuscript.
**Every one of them reverts under the two-level scheme.**

| cell | n | excess | manuscript | conditional | two-level |
|---|---:|---:|---|---|---|
| HT hard_70 L2 | 3,737 | +0.108 | INCONCLUSIVE | **SUPERIOR** | INCONCLUSIVE |
| HT std_95 L2 | 3,882 | +0.107 | INCONCLUSIVE | **SUPERIOR** | INCONCLUSIVE |
| Patho hard L0 | 1,792 | +0.005 | INCONCLUSIVE | **EQUIVALENT** | INCONCLUSIVE |
| Transm hard L0 | 2,497 | −0.024 | INCONCLUSIVE | **INFERIOR** | INCONCLUSIVE |
| Transm hard L1 | 55 | +0.240 | INCONCLUSIVE | **SUPERIOR** | INCONCLUSIVE |

Every verdict the manuscript already reports as resolved survives under **both**
schemes: GUE phage L0/L1 SUPERIOR, GUE covid L0 INFERIOR, HT hard_70 L0/L1
SUPERIOR, HT std_95 L0 INFERIOR and L1 SUPERIOR, VHDB L1 SUPERIOR,
Patho std L0 and Transm std L0 EQUIVALENT.

**The two-level scheme agrees with the manuscript on all 20 fragment verdicts.**

### Source / recovered-unit verdicts

One change under two-level: **Patho std L0, INCONCLUSIVE → EQUIVALENT**
($[-0.0185, +0.0088]$, 1,849 units, 85 % coverage). The conditional scheme
additionally flips Transm hard L0 to INFERIOR and Transm hard L1 to SUPERIOR;
both revert under two-level.

### The headline cell

HT hard_70 L1 is **SUPERIOR under all four schemes**:

| | interval | verdict |
|---|---|---|
| fragment, conditional | $[+0.140, +0.244]$ | SUPERIOR |
| fragment, two-level | $[+0.131, +0.255]$ | SUPERIOR |
| source, conditional | $[+0.076, +0.331]$ | SUPERIOR |
| source, two-level | $[+0.073, +0.336]$ | SUPERIOR |

7,223 units, 43.2 % recovered coverage. Recovered-source / partial-source
bootstrap, not fully genome-corrected.

## 5. Why I recommend the two-level scheme

1. It matches the Methods' stated intent; the conditional scheme silently drops
   seed uncertainty.
2. It is the conservative choice, and it leaves every published verdict intact,
   so no manuscript claim requires revision.
3. **The conditional scheme's extra resolutions land exactly where the data are
   weakest.** It calls Transm hard L1 SUPERIOR — a population of 55 rows with
   14 positives, all in a single genus, and `assign_cover = 0.000`, meaning its
   "source" interval is literally the row bootstrap. Permutation calibration
   already flagged that cell as not separable from label-permuted noise. A
   scheme that resolves it is producing a false positive, which is direct
   evidence against using it on small populations.

The old interval was wrong, but wrong in the **conservative** direction: it never
manufactured a claim, it only widened and mis-centred. Replacing it with the
two-level scheme corrects the construction while changing no conclusion.

## 6. What could not be computed, and why

- **VHDB source intervals: NOT AUDITABLE.** `final_regime_table.py:76` hard-codes
  `/data/nvidia/vhdb_tropism/within_family_1000bp/test_tax.csv`, which is absent.
  VHDB is the only benchmark with *native* provenance, and the manuscript reports
  three VHDB source intervals in Table S4 that cannot be verified on this
  cluster. Per the work order, no approximate replacement was constructed.
- **GUE phage and GUE covid source intervals: not estimable.** Phage has a unit
  file but at 12.5 % assignment; the frozen pipeline refuses it as "a row
  bootstrap wearing a unit column" and that refusal is respected here.
- **Transm hard L1 `assign_cover = 0.000`.** Reported, but its source interval is
  the row bootstrap and should not be presented as source-resolved.

## 7. Replicate accounting

Replicates are skipped when a resample is single-class. Counts are recorded per
cell in `bootstrap_corrected.csv` (`frag_reps_skipped`, `src_reps_skipped`),
which the shipped pipeline never reported.

---

## Decision required before any manuscript edit

Which scheme becomes the manuscript's interval?

- **two-level** — recommended. Nothing in the paper changes except one source
  verdict (Patho std L0 → EQUIVALENT), and the Methods sentence about seed
  variance becomes true.
- **conditional** — permitted by the work order's literal wording, but resolves
  five additional cells including one that is demonstrably noise.

The excesses, arms and point estimates are identical either way.
