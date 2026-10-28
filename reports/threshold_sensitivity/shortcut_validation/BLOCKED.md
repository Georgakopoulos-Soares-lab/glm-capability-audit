# Shortcut validation — NOT RUN, blocked on missing inputs

Attempted 2026-10-01 on TACC. **No validation was performed and no validation
result exists.** Nothing in `reports/threshold_sensitivity/` was modified. This
file records why, and the protocol to run once the blocker clears.

Analysis A stays out of the manuscript until this directory holds a passing
result. As of now it is cited nowhere in `main.tex` or `supplementary.tex`, and
`figA1`–`figA3` are generated but never included, so nothing needs retracting.

---

## What the validation has to establish

The sweep ran **one** MMseqs2 search per benchmark at `--min-seq-id 0.30 -c 0.10
--cov-mode 2 -e 1e-3 --max-seqs 20`, captured `pident`/`qcov`, and derived every
stricter identity and coverage point by filtering that one hit table
(`threshold_sweep.py:43-48`). MMseqs2 is never re-run at the stricter settings.

The specific risk is `--max-seqs 20`: the base search keeps at most 20 targets
per query, ranked at the 0.30 setting. A true search at identity 0.90 could
surface a ≥90 %-identity target that did not make the top 20 at 0.30. The
derivation is therefore **not exact a priori**, which is why an empirical check
is required rather than optional.

`threshold_sweep_search.sh:6-7` names `validate_threshold_derivation.py` as that
check. **That script exists in no commit and nowhere in the archive.** The claim
currently rests on two prose sentences (`threshold_sweep.py:13-14`,
`threshold_summary.md:28-31`), which say the reruns were at identity **0.70 and
0.90** on VHDB and GUE phage. The task brief says **0.50 and 0.90**. The two
accounts disagree and neither is backed by data.

## Why it could not run

| requirement | status |
|---|---|
| MMseqs2 binary | **AVAILABLE** — `/scratch/10906/arisk/biojepa-env/.pixi/envs/default/bin/mmseqs`, version **18.8cc5c** |
| Python with pandas | **AVAILABLE** — same pixi env |
| Benchmark corpora (`hvue_v2/Host_Tropism/hard_70_1000bp/{train,test}.csv`, `gue_devcarve/.../{train,test}.csv`) | **MISSING** — required for the true-search side |
| Base 0.30 hit table (`reports/threshold_sensitivity/hits/*.tsv`, 433 MB) | **MISSING** — required for the derived-mask side |
| MMseqs2 version used for the frozen run | **UNRECORDED** anywhere in repo or archive |

Both sides of the comparison are unavailable, and the corpora are the single
upstream blocker: with them, the base search regenerates and the derived side
follows. The only HVUE copy on this cluster is
`/work/10906/arisk/ls6/evo-locking/data/hvue`, which is HVUE **v1**, carries
train/validation only with **no test split**, and is not the suite the audit used.

Do not substitute, reconstruct or approximate any of these inputs.

## The version confound, which changes the protocol

The frozen run's MMseqs2 version is not recorded. Regenerating the base search
with 18.8cc5c may not reproduce the frozen hit table. If it does not, a mismatch
at 0.50 or 0.90 cannot be attributed to the derivation shortcut rather than to
the version, and the test is uninterpretable.

The validation therefore needs a **step 0** that the original design did not have:
reproduce the 0.30 anchor first, and only proceed if it matches exactly.

| anchor | frozen value | source |
|---|---|---|
| HT hard_70, retained at id 0.30 | **11,994** | `hidden_masks/HVUE_HT_hard_70.npz`, 51,187 − 39,193 |
| GUE phage, retained at id 0.30 | **2,874** | `hidden_masks/GUE_phage_v2.npz`, 4,630 − 1,756 |

This is also where the unreconciled 11,992-vs-11,994 question lives. If the
regenerated base returns 11,992, that is a finding to report, not to absorb.

## Protocol, ready to run once the corpora are restaged

Targets, read from `identity_sweep.csv` (row sets, not just counts — the
acceptance criterion is symmetric difference = 0 on retained row IDs):

| benchmark | identity | expected retained |
|---|---|---|
| HT hard_70 | 0.30 (anchor) | 11,994 |
| HT hard_70 | 0.50 | 11,994 |
| HT hard_70 | 0.90 | 25,509 |
| GUE phage | 0.30 (anchor) | 2,874 |
| GUE phage | 0.90 | 4,057 |

GUE phage is the second benchmark because it is the smallest (4,630 test rows,
5,148 base hits), so it is the cheapest true re-run.

Note that at identity 0.50 the derived mask for HT hard_70 is **identical** to
the 0.30 mask (11,994 both). Comparing counts there would be near-vacuous;
comparing row IDs is not. **0.90 is the discriminating point**, because that is
where `--max-seqs 20` truncation would bite.

Steps:

0. Regenerate the base 0.30 search for both benchmarks. Confirm retained counts
   are exactly 11,994 and 2,874. **If not, stop and report.**
1. Derive masks at 0.50 and 0.90 from that base table using the sweep's own
   filter, unmodified.
2. Run true MMseqs2 searches at `--min-seq-id 0.50` and `0.90` (HT hard_70) and
   `0.90` (GUE phage), holding `-c 0.10 --cov-mode 2 -e 1e-3 --max-seqs 20`.
3. For each, compute the retained row-ID sets both ways and their symmetric
   difference. **Require |symmetric difference| = 0.**
4. Write into this directory: retained row IDs per side, counts, the symmetric
   difference (and the offending IDs if non-empty), SHA256 of every input and
   output, the exact commands, `mmseqs version`, and the corpus file hashes.
5. If any comparison fails, **stop and report before altering the sweep.** Do
   not relax `--max-seqs`, do not re-derive, do not edit `identity_sweep.csv`.

Only after a clean pass should Analysis A be considered for the manuscript.
