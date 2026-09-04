# reports/ — index

Grouped by role. The flat files are manuscript-facing; the subdirectories hold
supporting material.

## Start here

| file | what it is |
|:--|:--|
| **[PUBLISHING.md](PUBLISHING.md)** | how to publish this: venue, strengths ranked, writing order, limitations, pre-submission checklist |
| **[FINDINGS.md](FINDINGS.md)** | the claim ledger — 8 findings, each SOLID or CONTINGENT, with evidence path and framing risk |
| **[OPEN_RUNS.md](OPEN_RUNS.md)** | the work queue — what is closed, what remains |
| **[../results/README.md](../results/README.md)** | the evidence map — artifact → finding → script. Cite from here |

## The manuscript

| file | what it is |
|:--|:--|
| [PAPER_OUTLINE.md](PAPER_OUTLINE.md) | thesis, why the framing survives review, section plan |
| [PAPER_DESIGN.md](PAPER_DESIGN.md) | the validity grid and what populates each quadrant |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | the two tracks and their status |

## The evidence record

| file | what it is |
|:--|:--|
| [STATE_OF_EVIDENCE.md](STATE_OF_EVIDENCE.md) | **every number with provenance** — own runs vs `C2` vs published |
| [ERASABLE_EXCESS.md](ERASABLE_EXCESS.md) | the finalised 22-task excess table |
| [BASELINE_CAPACITY_CEILING.md](BASELINE_CAPACITY_CEILING.md) | the CNN ladder result in full |
| [PRIOR_WORK_DASHA_COMPARISON.md](PRIOR_WORK_DASHA_COMPARISON.md) | us vs published numbers, including the DNABERT-2 harness check |
| [CROSS_CLUSTER_SYNTHESIS.md](CROSS_CLUSTER_SYNTHESIS.md) | reconciliation across the two compute clusters |
| [TESTED_MATRIX.md](TESTED_MATRIX.md) | benchmark × method × regime coverage, derived from `results/` |

## The rules

| file | what it is |
|:--|:--|
| [PROTOCOL.md](PROTOCOL.md) | 27 binding rules; each exists because violating it changed a conclusion |
| [MEASUREMENT_PROTOCOL.md](MEASUREMENT_PROTOCOL.md) | how each metric is computed |
| [VALIDITY_GATES.md](VALIDITY_GATES.md) | the gate order a capability claim must pass |
| [SPLIT_DESIGN_EXPLAINED.md](SPLIT_DESIGN_EXPLAINED.md) | composition-disjoint vs homology holdout, and which is correct for what |
| [FULL_FT_DESIGN.md](FULL_FT_DESIGN.md) | full-FT task selection and the disjoint-set framing |

## Subdirectories

| dir | contents |
|:--|:--|
| [`experiments/`](experiments/) | per-experiment notes: EPI, GENEB sentinel, HVUE data verification, positive controls, ViroBench, task inventory |
| [`tables/`](tables/) | the small CSV/JSON summary tables the reports cite |
| [`unlearning/`](unlearning/) | the sister unlearning project — what this audit implies for it, plus its eval protocol and forget/retain task recommendations. **Not part of this paper.** See also `/home/nvidia/unlearning` |
| [`archive/`](archive/) | superseded working documents. **Do not cite** |

## Conventions

- **Every number carries its artifact path.** Numbers without one are unverified
  until traced — that convention is what caught the F3 and F4 mismatches.
- Cite `STATE_OF_EVIDENCE.md` and files under `../results/`, never `archive/`.
- `C2` marks results from the Vista cluster; everything else is our own run unless
  marked `EXTERNAL/PUBLISHED`.
