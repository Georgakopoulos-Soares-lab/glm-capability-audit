# `--max-seqs` sensitivity

**Status: NOT CURRENTLY REPRODUCIBLE — SOURCE CORPORA UNAVAILABLE.**

Not "not auditable". The parameter that was used, and its consequence, are both
established from the code and stated in the manuscript. What cannot be done on
this cluster is the alternative run.

## What is established

Both searches in `scripts/track_b_viral/nn_lookup_probe.py` and
`hidden_homology_audit.py` cap the prefilter at `--max-seqs 20`. The MMseqs2
default is 300. Lowering the cap can reduce sensitivity, because the prefilter
and the final alignment rank candidates differently, so a true relative that does
not reach the top 20 under the prefilter is never aligned and the query is
recorded as having no detectable relative.

The manuscript states this directly in Methods rather than describing the
protocol as maximally sensitive, and the Limitations repeat that the populations
called divergent may retain relatives this protocol does not detect.

## Partial evidence that does exist (found 2026-10-14)

A direct `20` versus `300` comparison was never run as a dedicated sweep, but one
exists incidentally for **HT hard_70**, recorded in
`ARCHIVE/repo_reports/threshold_sensitivity/threshold_summary.md`:

> An earlier implementation differed by two rows on `hard_70`. The cause was
> `--max-seqs`, which is 20 in the frozen pipeline and 300 by MMseqs2 default.
> Both disputed sequences were marginal detections, at 70.4 percent identity over
> 16.2 percent coverage and 75.5 percent over 13.8 percent, whose targets rank low
> enough in the prefilter to be cut at 20 candidates and found at 300.

So on HT hard_70 the cap changes **2 rows out of 11,994**, or 0.017 percent of the
population, and both are weak, low-coverage detections. A change of that size
cannot move an excess of +0.194 with a fragment interval of [+0.131, +0.255].

This supersedes the claim below that the 11,994 versus 11,992 count cannot be
diffed. It can be, and it has been: 11,992 is the count at `--max-seqs 300`.

**Scope limit.** This is established for HT hard_70 only. No equivalent comparison
exists for VHDB, GUE phage, HT std_95 or any of the small cells, so the cap's
effect on those populations remains unmeasured and the Limitations wording stands.

## Why the full sweep cannot be rerun here

The alternative requires re-running the nucleotide search, which requires the
benchmark corpora. Those were not transferred to this cluster. MMseqs2 itself is
available (2:18.8cc5c in the pixi environment), so the binary is not the blocker.

No corpora were reconstructed and no substitute was used.

## The run to do, if the exact frozen corpora become available

Mask generation only. No model training and no model inference is required,
because every prediction array already exists and is indexed by test row.

1. Re-run the nucleotide search at `--max-seqs 300`, holding every other
   parameter at the frozen values (`--search-type 3 --min-seq-id 0.30 -c 0.10
   --cov-mode 2 -e 1e-3`).
2. Rebuild the `hasN` mask from those hits.
3. Compare **exact retained row-ID sets** against the `--max-seqs 20` masks, not
   just counts. Report the symmetric difference and the proportion of rows whose
   \textsc{l1} status changes.
4. Priority order: HT hard_70, HT std_95, VHDB, GUE phage.
5. Re-score the existing predictions on any changed population and report whether
   any primary model-versus-comparator conclusion moves, in particular
   HT hard_70 \textsc{l1}.

The 11,994 versus 11,992 count on HT hard_70 \textsc{l1} is already settled by
the threshold-sensitivity record quoted above; what the full run would add is the
same measurement on the other eight benchmarks.
