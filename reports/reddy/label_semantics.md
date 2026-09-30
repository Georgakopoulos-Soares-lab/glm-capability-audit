# Label semantics: Reddy et al. 2026 vs HVUE host tropism

## 1. Reddy definition

Source: Reddy T, Schneider A, Hall AR, Witmer A, Hengartner N. *An improved dataset for
predicting mammal infecting viruses from genetic sequence information.* PLOS Computational
Biology 22(3):e1014125, 2026. DOI 10.1371/journal.pcbi.1014125.

The dataset is a relabelled extension of Mollentze et al. 2021 (`github.com/Nardus/zoonotic_rank`).
Three binary labels per viral species: `human`, `primate`, `mammal`.

Quoting the paper: the original classes were "'Human-host' vs 'Not human-host' labels for viral
infection indicating whether a certain virus is known to infect a human or not". The authors state
that initial results showed "a large amount of false positives, which indicates that the true label
may be 'human-host' but there may not be sufficient supporting data to make that claim" — i.e. the
original negatives were absence-of-evidence negatives. They add that the Mollentze holdout "did not
include review of the human infectivity of the viruses not isolated from humans", and that they
therefore "manually relabeled the viral dataset based on current literature (July 2025)".

So the Reddy `human` label is **known ability to infect humans, curated from literature as of July
2025**, with negatives explicitly reviewed rather than inferred from isolation source. The `primate`
and `mammal` labels were added partly as a proxy for zoonotic potential and to reduce false
positives arising from absence of evidence.

Unit: one row per **viral species**, carrying one or more NCBI accessions (median 1, up to 12 for
segmented genomes). Test split is 736 species, 19.0% human-positive.

## 2. HVUE definition

HVUE Host_Tropism ships only `(sequence, label)` over 1,000 nt fragments. It carries no accession,
no taxonomy and no host field, so the label definition cannot be audited from the release. The task
is described by its authors as human host tropism, i.e. whether the virus infects humans. The v2
training split used here is 230,934 fragments at 5.4% positive.

## 3. Compatible, partially compatible, incompatible

**Compatible.** Both target the same underlying biological question, namely whether a virus is known
to infect humans. Both are binary. Reddy's `human` label is the closest available analogue to HVUE
host tropism, and is the right one to compare against (not `mammal`, despite the paper's title).

**Partially compatible.**
- *Unit.* Reddy labels a species; HVUE labels a 1 kb fragment. A species-level label is not a
  fragment-level label, and a model trained on one is not directly scored on the other without an
  explicit aggregation rule.
- *Prevalence.* Reddy test is 19.0% positive; HVUE v2 train is 5.4% positive. MCC is
  prevalence-sensitive, so values are not comparable across the two without care.
- *Curation date.* Reddy is literature-reviewed to July 2025. HVUE's curation date is not stated in
  the release.

**Incompatible for the purpose originally proposed.** Reddy cannot serve as an *independent* external
validation set for a model trained on HVUE, for reasons of sequence overlap rather than label
semantics. See `overlap_report.md`.

**Not established.** Whether either dataset treats vector-borne or incidental detections as positive
is not recoverable from the HVUE release, so this axis cannot be compared.
