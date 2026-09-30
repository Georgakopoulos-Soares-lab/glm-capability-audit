# A. Sensitivity of the L1 conclusions to the MMseqs2 identity criterion

No model was retrained. The frozen predictions are re-scored on the population each
threshold defines, with the development-selected model held fixed, the baseline set
unchanged, nearest-neighbour lookup excluded as regime-unavailable, and the same bootstrap
unit the manuscript uses for each benchmark.

## Direction of effect

Raising `--min-seq-id` makes the criterion for **declaring a hit** stricter. Fewer hits are
declared, so **more** test sequences fall into "no detectable nucleotide relative" and the
retained population **grows**. Higher thresholds are therefore *less* conservative about
calling a sequence divergent. The primary setting of 0.30 is the **most conservative** point
on this axis, admitting the fewest sequences into L1.

## Reproduction and validation

The sweep reproduces the manuscript's primary masks **exactly**: at 0.30 the retained row
sets match the frozen masks with **zero disagreeing rows** on host tropism `hard_70`
(11,994), VHDB (1,200) and GUE phage (2,874).

An earlier implementation differed by two rows on `hard_70`. The cause was `--max-seqs`,
which is 20 in the frozen pipeline and 300 by MMseqs2 default. Both disputed sequences were
marginal detections, at 70.4 percent identity over 16.2 percent coverage and 75.5 percent
over 13.8 percent, whose targets rank low enough in the prefilter to be cut at 20 candidates
and found at 300. The frozen convention was adopted.

Higher thresholds are derived by filtering one permissive search on percent identity rather
than by re-running the search. That derivation is exact: on VHDB and GUE phage, at 0.70 and
0.90, the retained **row sets** from a true re-run and from filtering are identical in all
four comparisons.

## 1. Population sensitivity: large

Retained population sizes move substantially across the sweep (`figA1_retention`,
`figA3_units`).

| benchmark | n at 0.30 | n range over sweep | fold change | source units range |
|---|---|---|---|---|
| HVUE transmissibility `std` | 41 | 41 to 4,541 | 110.8x | 32 to 1,850 |
| HVUE pathogenicity `std` | 41 | 41 to 2,569 | 62.7x | 29 to 1,062 |
| HVUE transmissibility `hard` | 55 | 55 to 1,076 | 19.6x | 55 to 652 |
| HVUE pathogenicity `hard` | 16, not scored | 222 to 709 | 3.2x | 167 to 375 |
| VHDB | 1,200 | 1,200 to 3,410 | 2.8x | 194 to 348 |
| HVUE host tropism `std_95` | 11,124 | 11,124 to 27,576 | 2.5x | 6,247 to 13,411 |
| HVUE host tropism `hard_70` | 11,994 | 11,994 to 28,686 | 2.4x | 7,223 to 12,577 |
| GUE phage | 2,874 | 2,874 to 4,155 | 1.4x | 2,874 to 4,155 |

Populations are flat from 0.30 to 0.60 on every benchmark and move only above 0.70, which is
consistent with most detected relatives sitting well above 60 percent identity.

## 2. Effect sensitivity: small on the benchmarks that carry the conclusions

For the two benchmarks whose L1 comparison resolves in the manuscript, the excess moves
little and never approaches zero.

- **host tropism `hard_70`**: 0.1746 to 0.2776 across the sweep, 0.1935 at the primary
  setting. The interval excludes zero at all eight thresholds.
- **GUE phage**: 0.1079 to 0.1219, 0.1202 at the primary setting. Range of 0.014 MCC.

Four benchmarks do change sign somewhere in the sweep: transmissibility `hard`,
transmissibility `std`, pathogenicity `std` and pathogenicity `hard`. All four hold 16 to 55
sequences at the primary setting, and none of them resolves at any threshold, so the sign
change occurs entirely within intervals that span zero.

## 3. Inference sensitivity: one cell changes verdict

Seven of eight benchmarks return the **same verdict at every threshold**.

The exception is **host tropism `std_95`**, which is INCONCLUSIVE at seven thresholds and
SUPERIOR at 0.80 alone, where the interval is $[+0.004, +0.345]$ and barely clears zero.
Its excess never changes sign, ranging from 0.1172 to 0.2166. This cell is therefore
inference-sensitive but not direction-sensitive, and the isolated resolution at a single
interior threshold on an interval whose lower bound is 0.004 is better read as marginal than
as a finding.

Pathogenicity `hard` is not scorable at the primary setting, holding 16 sequences, and
becomes scorable only at 0.80 and above. It is INCONCLUSIVE wherever it can be scored.

## Answer to the question posed

**No major manuscript conclusion depends on choosing 30 percent minimum nucleotide identity.**

The exact size of the nucleotide-divergent population depends strongly on the identity
criterion, varying by up to 111-fold across the prespecified range, but the qualitative
model-versus-baseline conclusions are stable across that range. Both resolved positives hold
at every threshold, every inconclusive cell stays inconclusive, and the single verdict change
is a marginal resolution at one interior threshold on a cell the manuscript already reports
as unresolved.

Stable cells: host tropism `hard_70` (SUPERIOR throughout), GUE phage (SUPERIOR
throughout), VHDB, transmissibility `hard`, transmissibility `std`, pathogenicity `std`,
pathogenicity `hard` (all INCONCLUSIVE throughout).

Threshold-sensitive cell: host tropism `std_95`, resolving only at 0.80.

## Files

`identity_sweep.csv`, `coverage_sweep.csv`, `threshold_range_summary.csv`,
`figA1_retention`, `figA2_excess`, `figA3_units`, and the base searches under `hits/`.
