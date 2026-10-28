# Priority 3 — permutation calibration of the cross-fitted selection statistic

Run 2026-10-06. **1,000 permutations per cell**, 5 folds. CPU only.
Script `permutation_calibration.py`, table `permutation_calibration.csv`.

## What this tests, and what it does not

It asks, per population: is the seed-averaged cross-fitted selection statistic
distinguishable from what the *same procedure* produces when labels are
permuted? It is a calibration of the selection machinery, not an equivalence
test, and it establishes no universal sample-size cutoff.

Labels are permuted **across recovered source units**, carrying all child
fragments together, wherever provenance exists; the multiset of unit labels is
held fixed, so row-level class counts are preserved in expectation rather than
exactly, because units differ in size. Where no provenance exists labels are
permuted across rows, which preserves class counts exactly, and the cell is
labelled `row`. Folds are regenerated under the same grouping rule each time.

## Results

| cell | n | permutation unit | observed | null mean | null SD | null p95 | null p99 | p | > p99 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| HT hard_70 L1 | 11,994 | source genome | **0.266** | +0.003 | 0.018 | 0.027 | 0.060 | 0.0010 | **yes** |
| HT hard_70 L2 | 3,737 | source genome | **0.244** | −0.000 | 0.019 | 0.033 | 0.067 | 0.0010 | **yes** |
| HT std_95 L2 | 3,882 | source genome | **0.080** | +0.001 | 0.018 | 0.035 | 0.065 | 0.0040 | **yes** |
| VHDB L1 | 1,200 | row | **0.501** | +0.002 | 0.032 | 0.063 | 0.084 | 0.0010 | **yes** |
| VHDB L2 | 201 | row | **0.655** | −0.000 | 0.071 | 0.137 | 0.249 | 0.0010 | **yes** |
| GUE phage L2 | 86 | row | **0.843** | +0.003 | 0.121 | 0.211 | 0.276 | 0.0010 | **yes** |
| Transm hard L1 | 55 | source genome | **0.613** | −0.002 | 0.159 | 0.262 | 0.346 | 0.0010 | **yes** |
| Transm std L1 | 41 | source genome | **0.574** | +0.004 | 0.188 | 0.320 | 0.462 | 0.0020 | **yes** |
| Patho std L1 | 41 | source genome | **−0.113** | +0.005 | 0.171 | 0.295 | 0.478 | 0.7443 | no |

## This retracts an earlier conclusion in this audit

Earlier passes in `reports/cluster_handoff/positive_control/` applied a crude
"null gate" that compared the *null* statistic against a fixed absolute
threshold (null max < 0.10 from 5 draws) and declared several small cells
unusable on that basis. **That criterion was wrong**, and the work order was
right to forbid it.

A large null spread does not make an observed value indistinguishable from
noise; it only raises the threshold the observed value must clear. On
Transm hard L1 the null SD is 0.159, but the observed statistic is 0.613 against
a null 99th percentile of 0.346. With a proper 1,000-permutation null, **eight
of the nine key cells exceed their own null 99th percentile**, including every
cell the crude gate had rejected: Transm hard L1, Transm std L1, GUE phage L2
and VHDB L2.

Only **Patho std L1** fails, and it fails emphatically: its observed statistic is
**negative** (−0.113) and sits below the null mean, with p = 0.744.

## The distinction that matters for the manuscript

Passing permutation calibration and being inferentially resolved are **different
things**, and several cells pass the first while failing the second.

- Permutation asks: *is there any real signal here, or is the selection
  procedure just overfitting?*
- The corrected bootstrap asks: *is the model-minus-comparator difference
  resolved?*

Transm hard L1 is the clearest case. It passes permutation decisively
(p = 0.001), so the population does contain signal that an honest selection
procedure can find. But its two-level bootstrap interval is
$[-0.050, +0.430]$, INCONCLUSIVE — the *difference between the model and the
comparator* is not resolved, on 55 rows with 14 positives in a single genus and
zero recovered provenance. Both statements are true simultaneously.

So permutation calibration must **not** be used to promote a cell to a finding.
Its correct use is the negative direction only: a cell that fails it cannot
support a selection-based claim.

## Preferred wording

For Patho std L1, the one failing cell:

> Selection-based differences in this population are not reliably distinguishable
> from values produced under label permutation.

This is not a claim of equivalence.
