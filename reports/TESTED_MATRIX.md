# What has actually been tested — benchmark × method × regime

Legend: **D** done and trusted · **P** partial · **X** not run · **!** run but invalid/superseded

## Track A — benchmark & method audit

**Derived from `../results/` on 2026-09-03, not from memory.** Counts are committed
artifacts. The previous version of this table predated most of the runs and marked
several completed cells as gaps; see `../results/README.md` for the artifact map.

| benchmark | tasks | k-mer | CNN ladder | frozen probe | LoRA | full FT |
|:--|--:|:--|:--|:--|:--|:--|
| NT splice | 3 | **D** | **D** — 3 tasks, 56-cell ladder (`capacity_sweep/splice__*`) | **D** (3 models) | X | **D** — `splice_finetune/` (17) |
| GENEB | 27 scored / 34 laddered | **D** (fair refit) | **D** — 34 tasks × 56 cells × {dilated, unet, resnet} | **D** (3 models, layer NOT swept) | **P** — Evo-1 only (`geneb_evo/`, 7) | **D** — 27 tasks × 4 models × 3 LR × 3 seeds = 321 summaries |
| GENEB remainder | ~66 of 100 | X | X | X | X | X |
| GUE non-viral | 12 | **D** | **P** (1 arch) | **D** — `gue_glm*/` (37) | X | vs published only |
| NT benchmark | 18 | **D** | **P** (1 arch) | X | X | vs published only |
| EPI | 6 | **D** | **D** | X | X | X |
| DART-Eval task 1 | 1 | **D** | **D** | X | X | X |
| learning curves | 13 | — | — | — | — | **D** — `learning_curve/` |

Models with full-FT GENEB coverage: **NT-v2-500M, DNABERT-2, GENA-LM, HyenaDNA**
(~81 summaries each). Seeds: LR swept on seed 42, then seeds 43/44 at the
dev-selected LR — the correct pattern. `ERASABLE_EXCESS.md` finalises 22 tasks with
3 seeds on both arms; **21 of 22 rows have n=3**, the single exception being
NT:H4 / DNABERT-2 at n=1.

**DNABERT-2 is fully covered.** Earlier notes calling it "blocked by a
`config_class` conflict" or "never run" are stale: pinning `transformers<4.48`
fixed it, and it is the best gLM on 10 of the 22 `ERASABLE_EXCESS` tasks. It also
serves as a harness check — our full-FT run beats the *published* DNABERT-2 number
on the identical checkpoint by +0.0482.

## Track B — viral capability

| benchmark | tasks | k-mer | CNN ladder | alignment | frozen probe | LoRA | full FT |
|:--|--:|:--|:--|:--|:--|:--|:--|
| ViroBench ALL/times | 5 levels | **D** (matched ctx at family; **P** elsewhere) | **D** — ladder at all 5 levels | **D** family, **reconciled** (`aligner_reconcile.json`) | **D** (layer swept at family) | X | X |
| ViroBench ALL/genus | 5 levels | X | X | X | **X ← harder split, no gLM data** | X | X |
| ViroBench DNA/RNA subsets | — | **P** | X | X | **P** | X | X |
| HVUE Host_Tropism (strict) | 1 | **D** | **D** — `capacity_sweep/hvue__Host_Tropism__strict*` | X | **D** (3 models × 3 seeds) | X | X |
| HVUE Path./Trans. | 2 | ! | **D** ladder exists (`capacity_sweep/hvue__{Pathogenecity,Transmissibility}`) but ! on the split | X | ! | ! | ! |
| HVUE composition-gated splits | 3 | ! | ! | X | ! | ! | ! |
| GUE viral (deduped) | 2 | **D** | **D** | X | X | X | **D** (3 models × 3 seeds) |
| ProteinGym viral | 22 | **P** | X | X (MSA needed) | X | X | **P** — `proteingym_supervised/` (30) |
| Antibody escape | 3 | **D** (MSA) | X | X | X | **P** | **P** |

**!** on HVUE Path./Trans. = the task retains 96/5,194 and 60/4,956 homology-clean test rows;
no method's score there separates memorisation from generalisation. **!** on composition-gated
splits = built in the k-mer's feature space and gated on k-mer degradation; invalid by construction.

## Models covered

| model | params | HVUE | ViroBench | GUE viral | splice | GENEB |
|:--|--:|:--|:--|:--|:--|:--|
| NT-v2-500M | 500M | D | D | D | D | D |
| GENA-LM-bert-base-t2t | 111M | D | D | D | D | D |
| HyenaDNA-medium-160k | 7M | D | D | D | D | D |
| LucaVirus | 944M | X | D (4 layers) | X | X | X |
| Evo-1-8k | 6.5B | P (LoRA) | X | X | X | P (LoRA, `geneb_evo/`) |
| **DNABERT-2** | 117M | X | X | X | X | **D** (27 tasks × 3 LR × 3 seeds) |
| **HViLM-Patho / -Tropism** | — | **D** (leaky **and** strict splits, `hvilm/`) | X | X | X | X |
| ESM-2 / ESM-1v / pLMs | — | — | — | — | — | X |

## Splits used, and what each licenses

| split | construction | licenses | status |
|:--|:--|:--|:--|
| composition-cluster (gated) | kmer5→PCA→KMeans, accepted iff k-mer drops ≥0.03 | **nothing** | invalid |
| identity-disjoint | `easy-cluster -c 0.9` (90% bidirectional) | duplicate removal | insufficient |
| **strict** | `easy-search`, drop test rows ≥70% id over ≥30% cov | homology-clean claims | **only valid HVUE split** |
| ViroBench `times` | train ≤2017-10-21 / test ≥2020-02-03 | temporal generalisation | clean (2.2% dup) |
| ViroBench `genus` | whole genera withheld | **harder** generalisation | **unused** |
| NT splice | chromosome-disjoint, 0% exact/revcomp | FT positive control | clean, verified |
| GUE official | unverified | in-distribution only | must not be called OOD |
| GUE virus_covid deduped | 11.9% exact dups removed | usable | clean |

## Headline numbers we stand behind

| claim | evidence |
|:--|:--|
| HVUE Path./Trans. cannot support homology-clean evaluation | 96/5,194 and 60/4,956 surviving rows |
| Baseline receptive field, not capacity, binds on positional tasks | ResNet 9.44M/RF89 = 0.336 vs U-Net 0.26M/global = 0.951 |
| Harness reproduces published splice numbers when regime matches | NT-v2 FT 0.9674 ± 0.0025 vs published 0.971–0.984 |
| Viral family taxonomy is an alignment task | alignment 0.7383 vs best gLM 0.6148; 0.9915 acc on aligned subset at 10–13% coverage |
| No viral advantage at matched context | ViroBench family: k-mer3-5 0.6231 > NT-v2 0.6111 (dev-selected) |
| Deduplication does not change the GUE viral ranking | k-mer 0.7171 > NT-v2 0.6764 on n=8,050 |

## Claims withdrawn

- ~~NT-v2 significantly beats k-mer on ViroBench family~~ — context asymmetry + incomplete k-mer sweep.
- ~~Our k-mer is the ViroBench ceiling~~ — alignment is +0.16 above it.
- ~~gLMs beat the baseline on GENEB 13/13~~ — k-mer-anchored; no CNN was run.
- ~~Published splice gap is +0.31 to +0.60~~ — receptive-field artifact; real gap +0.02–0.03.
- ~~LucaVirus's deficit is a read-out artifact~~ — persists across 4 layers (−0.28).
