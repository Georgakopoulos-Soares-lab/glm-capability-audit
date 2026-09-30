# Simple baselines match viral genomic language models

Genomic language models are credited with viral capabilities such as host
tropism, pathogenicity and transmissibility prediction, and those scores now
inform assessments of open-weight model risk. This repository contains the audit
behind the paper, which asks two questions of nine published viral benchmarks:
does the model beat what is cheaply available, and can the test set tell
generalization from familiarity?

**Paper:** [`paper/main.tex`](paper/main.tex) · rebuild with `python paper/build.py`
**Every number:** [`reports/RESULTS.md`](reports/RESULTS.md)

## What the audit found

**On the test sets as published, the nine divide three ways.** A simple baseline
matches or beats the leading virus-specialized model on four. The model is shown
to beat every baseline on two, GUE phage and HVUE host tropism `hard_70`. The
remaining three are inconclusive, and our own equivalence test forbids counting
those for either side.

Which baseline wins varies, which is why the paper speaks of a *baseline
threshold* rather than any one method. A k-mer gradient-boosted tree sets it most
often, on 10 of 26 populations; a supervised CNN sets it on 6, including all of
GUE phage; nearest-neighbour lookup and a genus prior take the rest.

**Nucleotide filtering certifies nucleotide divergence, not evolutionary
independence.** Six-frame translated search finds a relative in 65 to 97 percent
of the test sequences a nucleotide search calls distant, on every benchmark that
retains any such sequence.

**What survives is too small and too narrow.** Once translated relatives are
withheld, six of nine benchmarks retain 86 sequences or fewer. Every surviving
population has a detectable difference larger than the difference it contains.
The two largest draw 77 and 66 percent of their positive class from a single
genus, which neither sequence count nor class balance exposes.

**The threshold falls with the model, not against it.** Every arm collapses as
relatives are withheld, including the two supervised on the task, so what is
withheld was available to all predictors rather than to lookup alone. On HVUE
host tropism `hard_70` the model goes from 0.723 to 0.225 while the threshold
goes from 0.623 to 0.117. The margin survives; roughly 70 percent of the absolute
score does not.

**Lineage is a predictor, not a defect.** On the one benchmark shipping taxonomy,
a parameter-free genus prior reaches 0.681 MCC against 0.704 for the leading
model. Eight of nine ship `sequence,label` and nothing else, so that
decomposition cannot be computed on them, parent-genome leakage cannot be
measured, and no correct bootstrap unit can be built.

**A tenth benchmark shows the split rule is what matters.** ViroBench applies two
rules to one pool of taxa. The temporal cut yields roughly twenty times more
evaluable novel sequence than the genus cut beside it.

## Reproducing

```bash
cp pipelines/env.example.sh pipelines/env.sh   # point at your data + interpreters
$EDITOR pipelines/env.sh && source pipelines/env.sh

python pipelines/run.py --list                 # what is configured
python pipelines/run.py all --dry-run          # every command, nothing executed
python pipelines/run.py all --gpus 0,1,2,3     # run it
```

`mmseqs` must be on `$PATH`; every relatedness number depends on it.

Then rebuild the paper from the results:

```bash
python paper/build.py              # regenerate figures and tables, compile the PDF
python paper/build.py --no-pdf     # regenerate only
python paper/build.py --quick      # skip the slow bootstrap calibration
```

### How the paper's tables are generated

The tables are **injected into `main.tex` and `supplementary.tex` between marker
comments**, not kept as separate files, so there is nothing to `\input` and
nothing that can drift out of sync with the text:

```latex
% <<< GENERATED capability_matrix (do not edit by hand)
...
% >>> GENERATED capability_matrix
```

Re-running a generator rewrites only its own block. Numeric intermediates go to
`reports/tables/`, which is committed because those numbers are the evidence
behind the tables.

Every generator **fails on a missing or empty input rather than emitting a
table**. This is deliberate: an empty table typesets and looks finished, and a
renamed mask file once blanked the genus columns of the capability matrix without
any error. `paper/figures/inject.py` holds the guard all of them route through.

`make_supplementary.py` additionally refuses to overwrite `supplementary.tex` if
it has been edited since the script last wrote it, and needs `--force` to
regenerate. That file has been hand-edited and the guard is there because
regenerating it once destroyed that work.

## Layout

```
pipelines/run.py          single entry point, 12 stages
pipelines/registry.py     tasks, models and baseline arms
paper/build.py            rebuild every figure and table, then compile
paper/main.tex            manuscript, with generated tables inlined
paper/supplementary.tex   supplementary figures and tables
paper/figures/            figure and table generators
scripts/track_b_viral/    the audit itself: search, strata, bootstrap, recovery
reports/layered/          per-task ladder, intervals and verdicts
reports/tables/           numeric intermediates the paper's tables are built from
tests/                    54 tests, `python -m pytest`
```

## Vocabulary

The paper is careful about these and the code follows it.

| term | meaning |
|---|---|
| **baseline threshold** | the highest score any single simpler predictor reaches on the population being scored, re-maximized inside every bootstrap replicate. Not an ensemble, not an average |
| **withholding level** | `published`, then `no DNA hit`, `no 6-frame hit`, `no protein hit`, each nested inside the last; `unseen species` and `unseen genus` select on taxonomy instead and are not nested |
| **population** | the test sequences surviving into a level. Internally these are keyed `L0` to `L5` (`paper/figures/regimes.py`) |
| **detectable difference** | the smallest model-to-threshold difference a population's own bootstrap interval could have separated from zero |
| **inconclusive** | the interval spans zero *and* exceeds the ±0.02 MCC margin. The absence of a finding, not a weak *equivalent* |

## Caveats carried in the code

- Only VHDB defines a genome-level bootstrap unit. Every other interval resamples
  rows and is anti-conservative by an unknown factor; the pipeline labels these
  `unit=ROW (anti-conservative)` and the flag reaches every output.
- "No detectable relative" is relative to the MMseqs2 protocol in
  `scripts/track_b_viral/`. A more sensitive search would find more, which would
  strengthen the reclassification rather than weaken it.
- Recovered genus labels are never pooled with shipped ones, and unassigned
  sequences are excluded rather than imputed.
- GENA-LM's first probe runs used a mis-loaded checkpoint that scored near zero.
  Those files are still on disk; `make_probe_table.py` takes the corrected run and
  refuses to emit a value matching the defect.

## License

See [LICENSE](LICENSE).
