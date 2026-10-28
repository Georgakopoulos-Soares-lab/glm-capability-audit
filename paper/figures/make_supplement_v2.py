#!/usr/bin/env python
"""Writes paper/supplementary.tex in full. Six tables, emitted in citation order.

    S1  tab:selection  model selection, from the frozen three-seed artifact   s2()
    S2  tab:allarms    every model and every conventional arm, per population s1()
    S3  tab:bootstrap  primary vs conditional scheme, at three and ten fits   s6()
    S4  tab:source     sequence-level and source-level resampling             s3()
    S5  tab:hidden     dependence among unassigned windows                    s4()
    S6  tab:seedsd     fits per method and the spread across them             s7()

The function names are historical and do not track the printed numbers; the labels
do, and the main text cites labels only, so this order can change without touching
prose. Every table is computed from the frozen predictions under VB_RESULTS except
S1, which reads reports/tables/selection_3seed.csv because its quantity predates the
fit expansion and can no longer be recomputed. See NOTE 8 in s2().
"""
import glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef as MCC

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
T, R = os.path.join(REPO, "reports", "tables"), os.path.join(REPO, "reports", "layered")
import sys
sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
import final_regime_table as F

ORDER = ["GUE phage", "HT hard_70", "VHDB", "Patho hard", "Patho std",
         "Transm std", "GUE covid", "Transm hard", "HT std_95"]
PRETTY = {"HT hard_70": r"HT \textsc{hard\_70}", "HT std_95": r"HT \textsc{std\_95}",
          "Patho hard": r"Patho \textsc{hard}", "Patho std": r"Patho \textsc{std}",
          "Transm hard": r"Transm \textsc{hard}", "Transm std": r"Transm \textsc{std}",
          "VHDB": "VHDB", "GUE phage": "GUE phage", "GUE covid": r"GUE \textsc{covid}"}
MODEL = {"hvilm": "HViLM", "nt_v2_500m": "NT-v2", "dnabert2": "DNABERT-2",
         "gena_lm": "GENA-LM", "hyenadna": "HyenaDNA"}
ARMS = ["kmer", "kmer_gbt", "aa_gbt", "cnn", "transformer", "nn_lookup",
        "tax_prior", "tax_prior_recovered"]
ARMN = {"kmer": "$k$-mer", "kmer_gbt": "$k$-GBT", "aa_gbt": "AA", "cnn": "CNN",
        "transformer": "Tfmr", "nn_lookup": "NN", "tax_prior": "Tax",
        "tax_prior_recovered": "Tax$^{r}$"}
# short label -> the benchmark name used in the frozen outputs
NAME2SHORT = {
    "HT hard_70": "HVUE host tropism hard_70", "HT std_95": "HVUE host tropism std_95",
    "Patho hard": "HVUE pathogenicity hard", "Patho std": "HVUE pathogenicity std",
    "Transm hard": "HVUE transmissibility hard", "Transm std": "HVUE transmissibility std",
    "VHDB": "VHDB within family", "GUE phage": "GUE phage",
    "GUE covid": "GUE virus_covid"}
SHORT = {v: k for k, v in NAME2SHORT.items()}
TRNAME = {"HVUE_HT_hard_70": "HT hard_70", "HVUE_HT_std_95": "HT std_95"}


def _selected():
    """Development selected model per benchmark, read from the source of truth so it is
    the pre-committed choice and is never recomputed from the expanded fit set."""
    from sot_path import sot_csv
    _d = pd.read_csv(sot_csv())
    _d = _d[_d.regime == "L0"]
    return dict(zip(_d.benchmark, _d.model))


SELECTED = _selected()


def require(p, by, what):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise SystemExit(f"missing input: {p}\n  holds: {what}\n  produce it with: {by}")
    return p


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"


def s1():
    # Computed live from the tree being scored. This previously read a frozen
    # fragment_level_all_scores.csv that nothing regenerated, so the table stayed at
    # three fits while every other table moved to ten. Changed 2026-10-18.
    recs = []
    for _name, _mask, _esc, _atags, _ in F.CELLS:
        _z = np.load(f"{F.MASKS}/{_mask}.npz"); _n = len(_z["hasN"])
        _lv = {"L0": np.ones(_n, bool), "L1": ~_z["hasN"],
               "L2": (~_z["hasN"]) & (~_z["hasT"])}
        _arms = F.arm_preds(_atags or [_esc], _n)
        _y0 = None
        _series = {}
        for _m in F.MODELS:
            _y, _ps, _ = F.model_preds(_esc, _m, _n)
            if _ps:
                _series[_m] = _ps
                _y0 = _y
        if _y0 is None:
            continue
        for _a, _ps in _arms.items():
            _series[_a] = _ps
        for _lab, _k in _lv.items():
            if _k.sum() < 20 or len(np.unique(_y0[_k])) < 2:
                continue
            _row = {"task": _name, "regime": _lab}   # F.CELLS yields the long name df.task uses
            for _c, _ps in _series.items():
                if _c in F.REGIME_UNAVAILABLE and _lab != "L0":
                    continue
                _row[_c] = float(np.mean([MCC(_y0[_k], _p[_k]) for _p in _ps]))
            recs.append(_row)
    df = pd.DataFrame(recs)

    # The selected model's cell is printed from the source of truth rather than from the
    # live mean, so this table and Table~\ref{tab:headtohead} round the same number. They
    # are the same quantity, but the source of truth stores it at four decimals and the
    # live mean carries full precision, which put HT std_95 L1 on either side of a
    # rounding boundary (0.2825 -> 0.282 in Table 1, 0.283 here). Added 2026-10-28.
    from sot_path import sot_csv
    _sot = pd.read_csv(sot_csv())
    for _, _r in _sot.iterrows():
        _long = NAME2SHORT.get(_r.benchmark)
        _m = df.index[(df.task == _long) & (df.regime == _r.regime)]
        if len(_m) and _r.model in df.columns and pd.notna(_r.model_mcc):
            df.loc[_m[0], _r.model] = float(_r.model_mcc)

    L = [r"\begingroup\tiny\setlength{\tabcolsep}{1.9pt}",
         r"\begin{longtable}{ll rrrrr rrrrrrrr}",
         r"\caption{\textbf{All foundation models and all conventional predictors.} "
         r"MCC averaged over each method's fits on the population named. Most stochastic "
         r"arms carry ten fits; the $k$-mer logistic arm, nearest neighbour lookup and the "
         r"taxonomy prior are deterministic and carry one. VHDB and the model arm on GUE "
         r"\textsc{covid} are the exceptions, and every arm's fit count is given in "
         r"Table~\ref{tab:seedsd}. Tax and Tax$^{r}$ are the same arm, the second on genus "
         r"labels recovered by alignment. A dash means the arm is not defined for that "
         r"cell; nearest neighbour lookup is undefined at \textsc{l1} and \textsc{l2} by "
         r"construction.}"
         r"\label{tab:allarms}\\", r"\toprule",
         r"& & \multicolumn{5}{c}{foundation models} & \multicolumn{8}{c}{baseline arms} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-15}",
         r"Benchmark & Pop. & " + " & ".join(MODEL[m] for m in F.MODELS) + " & "
         + " & ".join(ARMN[a] for a in ARMS) + r" \\", r"\midrule", r"\endfirsthead",
         r"\toprule Benchmark & Pop. & " + " & ".join(MODEL[m] for m in F.MODELS) + " & "
         + " & ".join(ARMN[a] for a in ARMS) + r" \\ \midrule \endhead"]
    for b in ORDER:
        sub = df[df.task == NAME2SHORT[b]]
        for _, r in sub.iterrows():
            cells = []
            for c in list(F.MODELS) + ARMS:
                v = r.get(c)
                cells.append("--" if v is None or pd.isna(v) else f"{v:.3f}")
            L.append(f"{PRETTY[b]} & \\textsc{{{r.regime.lower()}}} & "
                     + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{longtable}\endgroup"]
    return "\n".join(L)


def s2():
    """Model selection, printed from the frozen three-seed artifact.

    NOTE 8. This must not be recomputed from the tree. Selection was pre-committed on
    three seeds and the selected arms were later refit ten times, so a live recomputation
    prints a ten-fit mean as the quantity that made a three-seed decision, and on
    pathogenicity \textsc{std} it reorders the top two rows on a 0.0006 margin. The
    three-seed values live in reports/tables/selection_3seed.csv, written by
    analysis/make_selection_artifact.py. Test MCC is the current ten-fit quantity and is
    computed live, matching Table~\ref{tab:allarms}.
    """
    sel_csv = require(os.path.join(T, "selection_3seed.csv"),
                      "analysis/make_selection_artifact.py",
                      "the frozen three-seed selection scores")
    frozen = pd.read_csv(sel_csv)
    rows = []
    for name, mask, esc, atags, _ in F.CELLS:
        z = np.load(f"{F.MASKS}/{mask}.npz"); n = len(z["hasN"])
        short = SHORT.get(name, name)
        sub = frozen[frozen.benchmark == short]
        if sub.empty:
            raise SystemExit(f"selection_3seed.csv has no rows for {short}")
        for r in sub.itertuples():
            y, ps, _ = F.model_preds(esc, r.model, n)
            te = float(np.mean([MCC(y, p) for p in ps])) if ps else np.nan
            rows.append(dict(bench=short, model=r.model, lr=r.lr,
                             dev=float(r.dev_mcc_3seed), test=te))
    df = pd.DataFrame(rows)
    L = [r"\begingroup\small\setlength{\tabcolsep}{5pt}",
         r"\begin{longtable}{ll r rr}",
         r"\caption{\textbf{Model selection.} Each model's best learning rate by mean "
         r"development MCC over three seeds, the development score used for selection, and "
         r"the corresponding test score. The development column is the pre-committed "
         r"selection quantity, computed over three seeds, and is not a ten-fit estimate; it "
         r"is reported as the number that made the selection rather than as a current "
         r"estimate of development performance. Bold marks the selected model, which is held "
         r"fixed across all populations. Test scores repeat values from "
         r"Table~\ref{tab:allarms} and enter no selection.}"
         r"\label{tab:selection}\\", r"\toprule",
         r"Benchmark & Model & lr & \makecell[r]{dev MCC\\(3 seeds, selection)} & test MCC \\",
         r"\midrule\endfirsthead",
         r"\toprule Benchmark & Model & lr & \makecell[r]{dev MCC\\(3 seeds, selection)} & "
         r"test MCC \\ \midrule \endhead"]
    for b in ORDER:
        sub = df[df.bench == b].sort_values("dev", ascending=False)
        if sub.empty:
            continue
        sel = sub.iloc[0].model
        if SELECTED.get(b) not in (None, sel):
            raise SystemExit(f"{b}: frozen dev bolds {sel}, source of truth says "
                             f"{SELECTED[b]}; selection record is inconsistent")
        first = True
        for _, r in sub.iterrows():
            bn = PRETTY[b] if first else ""
            first = False
            bold = (lambda s: r"\textbf{" + s + "}") if r.model == sel else (lambda s: s)
            L.append(f"{bn} & {bold(MODEL[r.model])} & {r.lr} & "
                     f"{bold(f'{r.dev:.4f}')} & {r.test:.3f} \\\\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(L)


def s3():
    """The table the Limitations paragraph needs. Units and coverage on their own do not
    show that a verdict changes, so the two intervals are given side by side."""
    from sot_path import sot_csv
    df = pd.read_csv(sot_csv())
    d = df[(~df.skipped) & (df.src_lo.notna()) & (df.n_rows >= 100)]
    L = [r"\begingroup\scriptsize\setlength{\tabcolsep}{2.4pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Sequence-level and source-level resampling.} Populations of at "
         r"least 100 sequences for which a source genome could be established. The "
         r"difference in MCC is identical under both schemes and is shown once. Units is "
         r"the number of source units; coverage is the share of sequences assigned to a "
         r"genome, with an unassigned sequence forming its own unit. VHDB supplies genome "
         r"identifiers directly; HVUE units are recovered by alignment. The two GUE "
         r"benchmarks supply no provenance and are absent.}\label{tab:source}",
         r"\begin{tabular}{ll r ll rr r}", r"\toprule",
         r"Benchmark & Pop. & diff. & sequence 95\% CI & source 95\% CI "
         r"& $n$ & units & cov. \\", r"\midrule"]
    fv = {"SUPERIOR": r"\textsc{sup}", "INFERIOR": r"\textsc{inf}",
          "EQUIVALENT": r"\textsc{eqv}", "INCONCLUSIVE": r"\textsc{inc}"}
    for b in ORDER:
        sub = d[d.benchmark == b]
        first = True
        for _, r in sub.iterrows():
            nm = PRETTY[b] if first else ""
            first = False
            cov = "100\\%" if r.provenance == "shipped" else f"{100*r.assign_cover:.0f}\\%"
            L.append(f"{nm} & \\textsc{{{r.regime.lower()}}} & {r.excess:+.3f} & "
                     f"$[{r.frag_lo:+.2f}, {r.frag_hi:+.2f}]$ {fv[r.frag_verdict]} & "
                     f"$[{r.src_lo:+.2f}, {r.src_hi:+.2f}]$ {fv[r.src_verdict]} & "
                     f"{int(r.n_rows):,} & {int(r.n_units):,} & {cov} \\\\")
        if not sub.empty:
            L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}",
          r"\end{table}\endgroup"]
    return "\n".join(L)


def s4():
    """Hidden-dependence sensitivity, in the form the Methods paragraph describes.

    Scenarios are values of an assumed group size s imposed on the u unassigned windows,
    labelled A to E. The numbers come from the corrected run: the manuscript estimator
    (mean over resampled fits, imported from bootstrap_corrected rather than
    reimplemented), 5,000 replicates per partition draw, 20 draws pooled for the three
    stochastic scenarios. Scenario A must reproduce the main-text source interval, and
    that is asserted below rather than left to inspection.
    """
    s = json.load(open(require(f"{R}/unassigned_sensitivity.json",
        "scripts/track_b_viral/unassigned_clustering_sensitivity.py",
        "hidden dependence scenarios")))
    if not s:
        raise SystemExit("unassigned_sensitivity.json is empty; rerun the sensitivity script")

    # m, the mean assigned-group size, is read off the same unit file the sensitivity
    # script partitions over, by the same rule (a label starting "row" is a singleton
    # the recovery failed to place). Hard-coding it would let the caption drift.
    spec = {c[0]: c for c in F.CELLS}
    ucsv = spec[s[0]["benchmark"]][4][0]
    u = pd.read_csv(ucsv)["cluster"].astype(str)
    m = float(u[~u.str.startswith("row")].value_counts().values.mean())

    # Scenario A is the partition the main analysis uses, so its interval is the
    # main-text source interval for this cell. If that stops holding, the two numbers
    # have drifted apart and the table must not print.
    from sot_path import sot_csv
    sot = pd.read_csv(sot_csv())
    anchor = sot[(sot.benchmark == SHORT.get(s[0]["benchmark"], s[0]["benchmark"]))
                 & (sot.regime == s[0]["regime"])]
    a = next(r for r in s if r["scenario"] == "none")
    if not anchor.empty:
        lo, hi = float(anchor.iloc[0].src_lo), float(anchor.iloc[0].src_hi)
        if abs(lo - a["lo"]) > 0.01 or abs(hi - a["hi"]) > 0.01:
            raise SystemExit(
                f"scenario A gives [{a['lo']:+.3f}, {a['hi']:+.3f}] but the source of truth "
                f"has [{lo:+.3f}, {hi:+.3f}] for this cell; they must be the same interval")

    SIZE = {"none": "$1$", "observed": "empirical", "mean": "$m$",
            "pooled": "$2m$", "worst": "$u$"}
    LETTER = dict(zip(["none", "observed", "mean", "pooled", "worst"], "ABCDE"))
    # the long benchmark name, since this caption is the only place it stands alone
    bench = re.sub(r"^HVUE ", "", s[0]["benchmark"])
    _suf = re.search(r"(hard_70|std_95|hard|std)$", bench)
    if _suf:
        bench = (bench[:_suf.start()]
                 + "\\textsc{" + _suf.group(1).replace("_", "\\_") + "}")
    L = [r"\begingroup\scriptsize\setlength{\tabcolsep}{5pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Sensitivity of the source-level interval to dependence among "
         r"unassigned windows.} " + bench + r" at \textsc{" + s[0]["regime"].lower() + r"}. "
         r"The $u$ unassigned windows are partitioned at random into groups of assumed "
         r"size $s$ and the source-level bootstrap repeated within each partition; "
         rf"$m={m:.2f}$ is the mean assigned-group size over the \textsc{{l0}} rows, and "
         r"realised is the resulting mean number of \textsc{" + s[0]["regime"].lower() +
         r"} rows per unit. A is the partition used in the main analysis, E a bound. The "
         rf"difference is $+${a['excess']:.3f} in every row. Partition SD is the spread of "
         rf"the interval endpoints over {max(r['n_partitions'] for r in s)} draws. The "
         r"procedure is described in Methods.}\label{tab:hidden}",
         r"\begin{tabular}{l l r r l l r}", r"\toprule",
         r" & $s$ & units & realised & 95\% CI & verdict & \makecell[r]{partition\\SD} \\",
         r"\midrule"]
    for r in s:
        psd = ("--" if r["n_partitions"] < 2 else f"{r['partition_endpoint_sd']:.4f}")
        L.append(f"{LETTER[r['scenario']]} & {SIZE[r['scenario']]} & {r['n_units']:,} & "
                 f"{r['n_rows'] / r['n_units']:.1f} & "
                 f"$[{r['lo']:+.3f}, {r['hi']:+.3f}]$ & "
                 f"\\textsc{{{r['verdict'].lower()}}} & {psd} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}\endgroup"]
    return "\n".join(L)


def s6():
    """Bootstrap sensitivity, reframed at ten fits.

    At three fits the two schemes disagreed on four populations. At ten they agree
    everywhere, so the table now shows those four and how each resolved rather than
    listing a disagreement that no longer exists.
    """
    import csv as _csv
    three = require(os.path.join(REPO, "reports", "finalization", "bootstrap_3seed.csv"),
                    "analysis/bootstrap_corrected.py --reps 5000 on the three-fit tree",
                    "the three-fit intervals this table compares against")
    old = {(r["benchmark"], r["regime"]): r for r in _csv.DictReader(open(three))}
    from sot_path import sot_csv
    new = pd.read_csv(sot_csv()).set_index(["benchmark", "regime"])
    fv = {"SUPERIOR": r"\textsc{sup}", "INFERIOR": r"\textsc{inf}",
          "INCONCLUSIVE": r"\textsc{inc}"}
    disputed = [k for k, r in old.items()
                if r.get("frag_verdict") and r.get("frag2_verdict")
                and r["frag_verdict"] != r["frag2_verdict"]]
    L = [r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Bootstrap scheme comparison at three and ten fits.} The four "
         r"populations on which the primary and conditional schemes returned different "
         r"verdicts at three fits. The primary scheme resamples test units and, within each "
         r"method, its observed fits; the conditional scheme resamples test units only. The "
         r"difference and verdict are given under each scheme at both fit counts.}",
         r"\label{tab:bootstrap}",
         r"\begin{tabular}{ll rll rl}",
         r"\toprule",
         r"& & \multicolumn{3}{c}{three fits} & \multicolumn{2}{c}{ten fits} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Benchmark & Pop. & diff. & primary & conditional & diff. & both schemes \\",
         r"\midrule"]
    for k in sorted(disputed):
        o = old[k]
        b, reg = k
        n = new.loc[(b, reg)] if (b, reg) in new.index else None
        nd = f"{float(n.excess):+.3f}" if n is not None else "--"
        nv = fv.get(n.frag_verdict, "--") if n is not None else "--"
        L.append(f"{PRETTY.get(b, b)} & \\textsc{{{reg.lower()}}} & "
                 f"{float(o['excess']):+.3f} & {fv.get(o['frag2_verdict'],'--')} & "
                 f"{fv.get(o['frag_verdict'],'--')} & {nd} & {nv} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}\endgroup"]
    return "\n".join(L)



def s7():
    """Per method fit counts and seed spread, so a deterministic arm, an arm measured
    at ten fits and an arm retaining three can be told apart. Computed live from the
    tree being scored."""
    rows = []
    for name, mask, esc, atags, _ in F.CELLS:
        z = np.load(f"{F.MASKS}/{mask}.npz"); n = len(z["hasN"])
        lv = {"L0": np.ones(n, bool), "L1": ~z["hasN"], "L2": (~z["hasN"]) & (~z["hasT"])}
        arms = F.arm_preds(atags or [esc], n)
        y0, series = None, {}
        for m in F.MODELS:
            y, ps, _ = F.model_preds(esc, m, n)
            if ps:
                series[m] = ps; y0 = y
        if y0 is None:
            continue
        sel = SELECTED.get(SHORT.get(name, name))
        for a, ps in arms.items():
            series[a] = ps
        for lab, k in lv.items():
            if k.sum() < 20 or len(np.unique(y0[k])) < 2:
                continue
            for c, ps in series.items():
                if c in F.REGIME_UNAVAILABLE and lab != "L0":
                    continue
                if c in F.MODELS and c != sel:
                    continue          # non-selected models enter no reported interval
                v = [MCC(y0[k], p[k]) for p in ps]
                rows.append(dict(bench=SHORT.get(name, name), lv=lab,
                                 meth=MODEL.get(c, ARMN.get(c, c)), nfit=len(v),
                                 mean=float(np.mean(v)),
                                 sd=(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0),
                                 lo=float(min(v)), hi=float(max(v))))
    df = pd.DataFrame(rows)

    # Same rounding rule as s1: the selected model's mean is taken from the source of
    # truth so this table, Table~\ref{tab:allarms} and Table~\ref{tab:headtohead} cannot
    # print one number three ways. The sd, min and max stay live, since the source of
    # truth does not carry them.
    from sot_path import sot_csv
    _sot = pd.read_csv(sot_csv())
    for _, _r in _sot.iterrows():
        if pd.isna(_r.model_mcc):
            continue
        _hit = df.index[(df.bench == _r.benchmark) & (df.lv == _r.regime)
                        & (df.meth == MODEL.get(_r.model, _r.model))]
        if len(_hit):
            df.loc[_hit[0], "mean"] = float(_r.model_mcc)

    # The shortfall sentence is derived, never asserted. An arm that is stochastic but
    # carries fewer than TARGET fits is named here so the caption cannot drift from the
    # table beneath it, which is how the "ten fits everywhere" claim survived as long as
    # it did. Added 2026-10-28.
    TARGET = 10
    det = {ARMN["kmer"], ARMN["nn_lookup"], ARMN["tax_prior"], ARMN["tax_prior_recovered"]}
    short = {}
    for _, r in df[(df.nfit < TARGET) & (~df.meth.isin(det))].iterrows():
        short.setdefault(r.bench, {})[r.meth] = int(r.nfit)
    if not short:
        SHORTFALL_SENTENCE = (f"Every stochastic arm carries {TARGET} fits.")
    else:
        bits = []
        for b, d in short.items():
            if len(set(d.values())) == 1:
                bits.append(f"on {PRETTY[b]} every stochastic arm listed carries "
                            f"{next(iter(d.values()))}")
            else:
                bits.append(f"on {PRETTY[b]} " + ", ".join(
                    f"{m} carries {n}" for m, n in sorted(d.items(), key=lambda kv: -kv[1])))
        SHORTFALL_SENTENCE = (
            f"Stochastic arms carry {TARGET} fits except where this table shows otherwise, "
            + "; ".join(bits) + ". These shortfalls are explained in Methods.")

    L = [r"\begingroup\tiny\setlength{\tabcolsep}{3pt}",
         r"\begin{longtable}{ll l r rrrr}",
         r"\caption{\textbf{Fits per method and the spread across them.} For the "
         r"development selected model and every conventional arm, on each population. "
         r"\emph{fits} is the number of independent training runs averaged into that "
         r"method's score, \emph{sd} their standard deviation. A deterministic arm has one "
         r"fit and zero spread by construction. The four foundation models not selected on "
         r"a benchmark enter no reported interval and are omitted. " + SHORTFALL_SENTENCE + "}"
         r"\label{tab:seedsd}\\", r"\toprule",
         r"Benchmark & Pop. & Method & fits & mean & sd & min & max \\",
         r"\midrule\endfirsthead",
         r"\toprule Benchmark & Pop. & Method & fits & mean & sd & min & max \\ \midrule \endhead"]
    for b in ORDER:
        sub = df[df.bench == b]
        if sub.empty:
            continue
        first = True
        for _, r in sub.iterrows():
            L.append(f"{PRETTY[b] if first else ''} & \\textsc{{{r.lv.lower()}}} & {r.meth} & "
                     f"{r.nfit} & {r['mean']:.3f} & {r.sd:.3f} & {r.lo:.3f} & {r.hi:.3f} \\\\")
            first = False
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{longtable}\endgroup"]
    return "\n".join(L)

def main():
    parts = ["% Generated by paper/figures/make_supplement_v2.py. Do not edit by hand.",
             "% Edits made here are lost on the next build. Change the generator instead.",
             r"\ifdefined\SUPPLEMENTINMAIN\else", r"\fi",
             r"\setcounter{table}{0}",
             r"\renewcommand{\thetable}{S\arabic{table}}",
             r"\section*{Supplementary tables}", ""]
    # Order follows first citation in the main text. The benchmark-characteristics table,
    # the translated-relative association test and the unseen-genus subset were all cut
    # from the paper and their generators deleted on 2026-10-28; the underlying artifacts
    # remain under reports/ and reports/layered/.
    for fn in (s2, s1, s6, s3, s4, s7):
        parts.append(fn()); parts.append("")
    out = os.path.join(REPO, "paper", "supplementary.tex")
    open(out, "w").write("\n".join(parts) + "\n")
    print(f"  wrote paper/supplementary.tex  ({len(parts)//2 - 3} tables, semantic labels)")


if __name__ == "__main__":
    main()
