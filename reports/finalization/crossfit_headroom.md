# Priority 2 — cross-fitted headroom above the conventional comparator

Run 2026-10-06. CPU only. Script `crossfit_headroom.py`, table
`crossfit_headroom.csv` (33 rows: every cell under grouped folds where possible,
and under row folds for comparison).

## Construction

Seeds are aggregated **within** each candidate predictor first, exactly as the
manuscript's point estimator does; cross-fitting then selects among predictors
only. No individual seed competes as a separate candidate, no decision threshold
is tuned, no hyperparameter is tuned. The model removed for the
leave-one-model-out variant is the **development-selected** model recorded in
`final_source_of_truth.csv`, never the model that looks best on test.
`test_derived_selection = no` on every row.

Folds are **grouped by recovered source genome** wherever provenance exists, so
fragments of one source cannot appear in both the selector-training folds and
the selector-evaluation fold.

| benchmark | fold unit | why |
|---|---|---|
| HVUE host tropism, pathogenicity, transmissibility | recovered source genome | cluster files present, coverage 37–85 % |
| VHDB | **row** | native unit file absent on this cluster |
| GUE phage | **row** | unit file exists at 12.5 % assignment; frozen pipeline refuses it |
| GUE covid | **row** | no provenance |

`drf_cf3 = (pos_cf − env_cf)/(1 − env_cf)`, and `drf_lomo3` the same with the
dev-selected model removed. "No measurable cross-fitted headroom" is defined
operationally as `drf_cf3 ≤ 0.02`. The word *saturated* is not used.

## Grouping changes the answer, so the grouped result is used

| L0 cell | drf_cf3 grouped | drf_cf3 row | fold unit used |
|---|---:|---:|---|
| HT hard_70 | **−0.091** | +0.087 | grouped |
| HT std_95 | 0.000 | 0.000 | grouped |
| Patho hard | 0.000 | 0.000 | grouped |
| Patho std | 0.000 | 0.000 | grouped |
| Transm hard | 0.000 | 0.000 | grouped |
| Transm std | −0.006 | 0.000 | grouped |
| VHDB | — | +0.083 | row (no units) |
| GUE phage | — | +0.566 | row (units refused) |
| GUE covid | — | 0.000 | row (no provenance) |

Under grouped folds HT hard_70 L0 **loses its headroom entirely** — the pool
selected on training groups transfers worse to held-out groups than the
conventional comparator does (`pos_cf` 0.589 against `env_cf` 0.623). That is
coherent with the manuscript's own finding that HT hard_70 L0 moves from
SUPERIOR to INCONCLUSIVE under genome resampling.

## The five provisional claims, tested

**A. "Six of nine L0 benchmarks have drf_cf3 ≈ 0" — REVISED to seven of nine.**
Using grouped folds where available: HT hard_70 (−0.091), HT std_95 (0.000),
Patho hard (0.000), Patho std (0.000), Transm hard (0.000), Transm std (−0.006)
and GUE covid (0.000) all show no measurable cross-fitted headroom. The earlier
"six of nine" was a row-fold result.

**B. "Headroom at L0 occurs only for GUE phage, HT hard_70 and VHDB" — FALSE as
stated.** HT hard_70 drops out under grouped folds. Headroom at L0 remains only
for **GUE phage (+0.566) and VHDB (+0.083)**.

**Important confound, which must be stated wherever this is reported:** those are
exactly the two benchmarks where grouped folds are impossible. Both figures are
row-fold values. GUE phage's margin is large enough that grouping is unlikely to
erase it, but VHDB's +0.083 is of the same order as the change grouping produced
on HT hard_70, and it should not be treated as established.

**C. "Removing the dev-selected model eliminates essentially all headroom" —
CONFIRMED, strongly.** `drf_lomo3` is 0.0000 in almost every cell under both fold
schemes. The exceptions are GUE phage L0 (+0.092) and L1 (+0.040), HT std_95 L2
grouped (+0.026), and HT hard_70 L1 grouped (+0.005, negligible).

**D. "HT hard_70 L1 remains a robust positive" — CONFIRMED, and stronger under
grouping.** Grouped `drf_cf3 = 0.186` against row 0.147, with `env_cf` 0.099 and
`pos_cf` 0.266 over 7,223 source groups. `drf_lomo3 = 0.005`, so the headroom is
supplied by the dev-selected model.

**E. "GUE phage is the clearest case where more than one model contributes" —
CONFIRMED.** It is the only benchmark whose headroom survives removing the
dev-selected model, at L0 (+0.092) and L1 (+0.040).

## Reading

On seven of the nine published test sets there is **no measurable cross-fitted
headroom above the conventional comparator within the evaluated model pool**.
Where headroom does exist it is almost entirely attributable to one model:
removing the dev-selected foundation model collapses it to zero everywhere
except GUE phage.

This is a statement about the evaluated pool on these populations. It does not
say these benchmarks are incapable of discriminating any possible model, and it
must not be written that way.
