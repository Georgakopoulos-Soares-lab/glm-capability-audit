#!/usr/bin/env python
"""Main Tables 1 and 2, injected into main.tex between GENERATED markers.

Both read final_source_of_truth.csv only. A number cannot reach the manuscript
without passing through that file.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
T = os.path.join(REPO, "reports", "tables")
ORDER = ["GUE phage", "HT hard_70", "VHDB", "Patho hard", "Patho std",
         "Transm std", "GUE covid", "Transm hard", "HT std_95"]
PRETTY = {"HT hard_70": r"HVUE host tropism \textsc{hard\_70}",
          "HT std_95": r"HVUE host tropism \textsc{std\_95}",
          "Patho hard": r"HVUE pathogenicity \textsc{hard}",
          "Patho std": r"HVUE pathogenicity \textsc{std}",
          "Transm hard": r"HVUE transmissibility \textsc{hard}",
          "Transm std": r"HVUE transmissibility \textsc{std}",
          "VHDB": "VHDB within family", "GUE phage": "GUE phage",
          "GUE covid": r"GUE \textsc{virus\_covid}"}
MODEL = {"hvilm": "HViLM", "nt_v2_500m": "NT-v2", "dnabert2": "DNABERT-2",
         "gena_lm": "GENA-LM", "hyenadna": "HyenaDNA"}
ARM = {"kmer": "$k$-mer logistic", "kmer_gbt": "$k$-mer GBT", "aa_gbt": "AA composition",
       "cnn": "CNN ladder", "transformer": "transformer ladder",
       "nn_lookup": "NN lookup", "tax_prior": "taxonomy prior",
       "tax_prior_recovered": "taxonomy prior"}
PROV = {"shipped": "ship.", "recovered": "recov.", "row only": "none"}


def require(p, by, what):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise SystemExit(f"missing input: {p}\n  holds: {what}\n  produce it with: {by}")
    return p


def load():
    from sot_path import sot_csv
    return pd.read_csv(require(sot_csv(),
        "paper/figures/make_source_of_truth.py", "every manuscript number"))


def ci(lo, hi):
    if pd.isna(lo):
        return "not estimable"
    return f"[{lo:+.3f}, {hi:+.3f}]"


def table1(df):
    """Benchmark inventory. The L2 column stays because the text uses it to say what
    the L1 filter does and does not certify; nothing else from that regime is claimed."""
    L = []
    A = L.append
    A(r"\begingroup\footnotesize\setlength{\tabcolsep}{5pt}")
    A(r"\begin{table}[htbp]\centering")
    A(r"\caption{\textbf{What each benchmark retains at each regime.} \textsc{l0} is the "
      r"published test set, \textsc{l1} the fragments with no detectable nucleotide relative "
      r"in the benchmark's own training split, and \textsc{l2} those with no detectable six "
      r"frame translated relative either. Percentages are of \textsc{l0}. \textsc{l2} is "
      r"shown to indicate how much of the nucleotide divergent population still has a "
      r"detectable relative, not as a population we evaluate on.}")
    A(r"\label{tab:benchmarks}")
    A(r"\begin{tabular}{l rrr r}")
    A(r"\toprule")
    A(r"Benchmark & \textsc{l0} & \textsc{l1} & \textsc{l2} & "
      r"\textsc{l1} with a translated relative \\")
    A(r"\midrule")
    for b in ORDER:
        d = df[df.benchmark == b].set_index("regime")
        g = lambda r, c: (d.loc[r, c] if r in d.index else np.nan)
        n0 = int(g("L0", "n_rows")); n1, n2 = g("L1", "n_rows"), g("L2", "n_rows")
        if not (n1 == n1) or not n1:
            A(f"{PRETTY[b]} & {n0:,} & 0 & 0 & -- \\\\")
            continue
        n2 = 0 if n2 != n2 else n2
        f1 = f"{int(n1):,}\\,({100*n1/n0:.0f}\\%)"
        f2 = f"{int(n2):,}\\,({100*n2/n0:.1f}\\%)"
        rel = f"{100*(1 - n2/n1):.0f}\\%"
        A(f"{PRETTY[b]} & {n0:,} & {f1} & {f2} & {rel} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\vspace{3pt}\par\parbox{\textwidth}{\footnotesize GUE \textsc{virus\_covid} "
      r"retains no nucleotide divergent fragment, so it contributes only at \textsc{l0}. "
      r"The last column is the share of the \textsc{l1} population that acquires a "
      r"detectable relative once the search is translated.}")
    A(r"\end{table}\endgroup")
    return "\n".join(L)


def table2(df):
    d = df[(~df.skipped) & (df.regime.isin(["L0", "L1"]))].copy()
    L = []
    A = L.append
    A(r"\begingroup\scriptsize\setlength{\tabcolsep}{1.9pt}")
    A(r"\begin{longtable}{ll l rr r ll r}")
    # Kept short on purpose. The protocol belongs in Methods, not in a caption: an
    # earlier version restated the source-unit scheme, the VHDB carry-forward and the
    # VHDB fit count here, and the fit count was wrong (VHDB is not uniformly three)
    # while Methods said something different again. Per-arm counts are in
    # Table~\ref{tab:seedsd}; the resampling schemes are in Methods. Changed 2026-10-28.
    A(r"\caption{\textbf{Model and best baseline MCC.} \textsc{l0} is the published test "
      r"set, \textsc{l1} the sequences with no detectable nucleotide relative in the "
      r"training data. The model and its learning rate are selected on the development "
      r"split and held fixed. Intervals are 95 percent two-level bootstrap intervals, "
      r"resampling test units and, within each method, its observed fits. Verdicts are "
      r"\textsc{sup} when the interval lies above zero, \textsc{inf} when below, and "
      r"\textsc{inc} when it spans zero.}")
    A(r"\label{tab:headtohead}\\")
    A(r"\toprule")
    A(r"Benchmark & Pop. & Model & model & base. & diff. & sequence 95\% CI "
      r"& source 95\% CI & $n$ \\")
    A(r"\midrule\endfirsthead")
    A(r"\toprule Benchmark & Pop. & Model & model & base. & diff. & sequence 95\% CI "
      r"& source 95\% CI & $n$ \\ \midrule \endhead")
    for b in ORDER:
        sub = d[d.benchmark == b]
        if sub.empty:
            continue
        first = True
        for _, r in sub.iterrows():
            nm = PRETTY[b] if first else ""
            first = False
            fv = {"SUPERIOR": r"\textsc{sup}", "INFERIOR": r"\textsc{inf}",
                  "EQUIVALENT": r"\textsc{eqv}", "INCONCLUSIVE": r"\textsc{inc}"}
            f = f"$[{r.frag_lo:+.2f}, {r.frag_hi:+.2f}]$ {fv[r.frag_verdict]}"
            g = ("not estimable" if r.src_verdict == "unavailable"
                 else f"$[{r.src_lo:+.2f}, {r.src_hi:+.2f}]$ {fv[r.src_verdict]}")
            A(f"{nm} & \\textsc{{{r.regime.lower()}}} & {MODEL[r.model]} & "
              f"{r.model_mcc:.3f} & {r.envelope_mcc:.3f} & {r.excess:+.3f} & "
              f"{f} & {g} & "
              f"{int(r.n_rows):,}{'$^{*}$' if r.n_rows < 100 else ''} \\\\")
        A(r"\addlinespace")
    L[-1] = r"\bottomrule"
    A(r"\end{longtable}")
    # "fewer than 100 sequences" named only one of the two reporting thresholds; the
    # minority-class floor is the binding one on the imbalanced benchmarks.
    A(r"\vspace{-8pt}\par\noindent\footnotesize Every model and conventional predictor "
      r"on every benchmark and population is reported in Supplementary "
      r"Table~\ref{tab:allarms}. An asterisk denotes a population falling below one of "
      r"the reporting thresholds defined in Methods, which therefore supports no summary "
      r"claim.")
    A(r"\endgroup")
    return "\n".join(L)


def main():
    df = load()
    os.makedirs(f"{T}/generated", exist_ok=True)
    # table1() is retained for reference but no longer emitted: it duplicates
    # Supplementary Table S1 and main.tex carries no marker to inject it into, so
    # generating it produced an orphan file that could drift unnoticed.
    for key, body in (("table_headtohead", table2(df)),):
        open(f"{T}/generated/{key}.tex", "w").write(body + "\n")
        print(f"  wrote reports/tables/generated/{key}.tex")


if __name__ == "__main__":
    main()
