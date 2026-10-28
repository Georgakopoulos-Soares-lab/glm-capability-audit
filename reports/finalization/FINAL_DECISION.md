# FINAL_DECISION

Finalization pass completed 2026-10-06. Primary interval scheme: **two-level
bootstrap** (test units and training seeds resampled). Conditional bootstrap
retained as supplementary sensitivity. No manuscript file has been edited.

---

### 1. Did any headline point estimate change?

**No.** The estimator was written down formally, implemented independently, and
reproduces the manuscript's `model_mcc`, `envelope_mcc`, `envelope_arm` and
`excess` **exactly on all 20 scored cells**. No excess, no comparator identity
and no selected model moves.

### 2. Did any headline CI/verdict change after fixing the bootstrap estimand?

**One, and it is a correction to accept.**

- **Patho std L0, source verdict: INCONCLUSIVE → EQUIVALENT**
  (`[-0.0185, +0.0088]`, 1,849 units, 84.8 % coverage).

All 20 **fragment** verdicts are unchanged from the manuscript under the adopted
two-level scheme. Every verdict the paper reports as resolved survives.

The conditional sensitivity scheme additionally resolves five cells —
HT hard_70 L2 and HT std_95 L2 to SUPERIOR, Patho hard L0 to EQUIVALENT,
Transm hard L0 to INFERIOR, Transm hard L1 to SUPERIOR. These are reported as
sensitivity only, and the supplement should state that all manuscript-resolved
findings hold under both schemes while those five resolve only when conditioning
on the observed fitted seeds.

### 3. Which L0 cells show measurable cross-fitted headroom?

Using grouped folds wherever provenance allows, and `drf_cf3 > 0.02`:

- **GUE phage L0, +0.566** (row folds; unit file refused at 12.5 % assignment)
- **VHDB L0, +0.083** (row folds; native unit file absent)

Seven of nine show **no measurable cross-fitted headroom above the conventional
comparator within the evaluated model pool**: HT hard_70 (−0.091 grouped),
HT std_95 (0.000), Patho hard (0.000), Patho std (0.000), Transm hard (0.000),
Transm std (−0.006), GUE covid (0.000).

**HT hard_70 L0 loses its headroom under grouped folds** (+0.087 row → −0.091
grouped). The provisional "six of nine" was a row-fold result; grouped is used
and the discrepancy is reported.

**Confound that must travel with this claim:** the only two L0 cells retaining
headroom are exactly the two where grouped folds are impossible. VHDB's +0.083 is
of the same order as the shift grouping produced on HT hard_70 and should not be
treated as established.

### 4. Which headroom results survive LOMO?

Removing the **development-selected** model (never a test-selected one):

- **GUE phage L0, +0.092** and **L1, +0.040** — the only survivors.
- `drf_lomo3 = 0.0000` in **15 of 20** cells, including every other L0 cell, all
  three VHDB cells, and all HT std_95 cells.

Headroom, where it exists, is supplied almost entirely by a single model.

### 5. Which small/L2 cells fail permutation calibration?

With **1,000 permutations**, grouped by recovered source genome where provenance
exists: **only Patho std L1** (empirical p = 0.744; observed statistic −0.113,
below the null mean).

**This retracts an earlier conclusion in this audit.** A crude 5-permutation
"null gate" comparing the null against a fixed absolute threshold had rejected
Transm hard L1, Transm std L1, GUE phage L2 and VHDB L2. That criterion was
wrong. Against a proper null distribution, **eight of nine key cells exceed their
own null 99th percentile**, including all four previously rejected.

Passing permutation does **not** promote a cell to a finding. Transm hard L1
passes decisively (p = 0.001) yet its two-level interval is `[-0.050, +0.430]`,
INCONCLUSIVE: the population contains signal, but the model-minus-comparator
difference is not resolved.

### 6. Is HT hard_70 L1 still robust under every relevant check?

**Yes, under every check run.**

| check | result |
|---|---|
| point estimate | +0.1935, unchanged |
| fragment, two-level | `[+0.131, +0.255]` SUPERIOR |
| recovered-source, two-level | `[+0.073, +0.336]` SUPERIOR |
| fragment, conditional | `[+0.140, +0.244]` SUPERIOR |
| recovered-source, conditional | `[+0.076, +0.331]` SUPERIOR |
| hidden-dependence scenarios | SUPERIOR in all 5, tightest bound +0.038 |
| grouped cross-fitted headroom | `drf_cf3 = 0.186` (higher than row 0.147) |
| permutation, 1,000 draws, source-grouped | observed 0.266 vs null p99 0.060, p = 0.001 |
| LOMO | `drf_lomo3 = 0.005` — the headroom is the dev-selected model |

Qualification: 7,223 units at 43.2 % recovered coverage. Recovered-source /
partial-source bootstrap, not fully genome-corrected.

### 7. Is Analysis A fully reproducible?

**No.** The shortcut validation cannot be regenerated: the benchmark corpora are
absent from this cluster, and the base MMseqs2 hit tables were discarded.
MMseqs2 2:18.8cc5c *is* available, so the binary is not the blocker — the inputs
are. `validate_threshold_derivation.py`, named in `threshold_sweep_search.sh`,
exists in no commit. The 11,994 vs 11,992 discrepancy cannot be diffed for the
same reason, and the frozen 11,994 was not altered.

**Analysis A stays out of the manuscript.** It is retained internally.

### 8. Which claims must be removed or softened?

**Removed**

- **Supplementary Table S7 and the interaction claim at `main.tex:178`.** Its
  generator selects the run by test MCC, violating `CLAIM_LEDGER` C10, and
  `test_tax.csv` is absent. No approximate replacement constructed.
- **The covid de-duplication sentence and the 10.66 % figure at `main.tex:70`.**
  De-duplication is claimed and was never applied; the corpus is unavailable.
- **The 87.6 % genus figure** — already removed 2026-10-01, no artifact exists.

**Softened**

- `main.tex:211` must no longer offer the null interaction as positive support.
  Safe form: *"The fragment-level pattern differs between L0 and L1, but we do
  not interpret filtering as causing an increase in incremental value."*
- `main.tex:196`, the L2 sentence, contradicts `CLAIM_LEDGER` C9 and the L2
  adjudication.
- "The models" in the abstract and Discussion must become model-specific.
- VHDB source intervals are **PRESENT BUT NOT REPRODUCED**, not verified.
- "Best of seven" is 5 or 6 arms outside L0.
- GUE covid is 9-class; its S1 "positive class" row is undefined.

### 9. Which analyses are descriptive only?

- **All four scorable L2 cells** — HT hard_70, HT std_95, VHDB, GUE phage. Every
  one is INCONCLUSIVE under the primary two-level fragment bootstrap. The other
  five L2 cells (n ≤ 3) are NOT RELIABLY ESTIMABLE.
- **The unseen-genus subsets.** Row-bootstrapped only, no source-aware interval,
  absent from the capability matrix. Not a genus-disjoint retraining experiment
  and must be called an *unseen-genus test subset*.
- **The small L1 cells** — Transm hard L1, Transm std L1 descriptive;
  Patho std L1 NOT AUDITABLE on permutation.
- **The cross-fitted headroom analysis itself** is a point estimate with no
  interval; it should be reported as such.

### 10. Is the manuscript statistically ready to rewrite?

**Yes** — the blocking estimand issue is resolved, no point estimate moved, one
source verdict is corrected, and every removal is determined by artifact
availability rather than judgement.

**One mechanical obstacle stands in the way of regenerating the tables, and it
needs a decision before I touch `paper/`.** The manuscript's Table 1 and the
supplement are generated from `final_regime_table.json`, and that generator
hard-fails at the VHDB cell because `test_tax.csv` is absent — it writes nothing
at all. So the generated tables cannot simply be rebuilt end to end on this
cluster. The options are:

1. **Merge** — patch the generator's bootstrap to the two-level scheme, write the
   corrected intervals for the 8 benchmarks that can be computed, and carry
   VHDB's three frozen source intervals through unchanged as historical values,
   flagged in the ledger as PRESENT BUT NOT REPRODUCED. Nothing is recomputed
   that cannot be recomputed, and nothing frozen is overwritten.
2. **Restage `test_tax.csv`** from a verified source with a hash, then regenerate
   cleanly end to end.

Option 1 changes one printed number (Patho std L0 source verdict) and leaves
every other printed interval as the manuscript already has it, because the
two-level scheme agrees with it everywhere. Option 2 is cleaner but blocked on a
file.

I have not edited the manuscript, patched the generator, or written to any
frozen output.
