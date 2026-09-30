#!/usr/bin/env python
"""Supplementary Tables S1 to S6, written as a complete supplementary.tex.

S1 every model and every conventional predictor, per benchmark and population
S2 model selection provenance, showing dev selection and no test selection
S3 source unit reconstruction and assignment coverage
S4 hidden dependence sensitivity for the one cell a source verdict decides
S5 translated relative direct test
S6 L0 to L1 interaction, both definitions of the comparator, with BH adjustment
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


def require(p, by, what):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise SystemExit(f"missing input: {p}\n  holds: {what}\n  produce it with: {by}")
    return p


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"


def s0():
    """Benchmark characteristics. Methods points here for sizes and class balance."""
    df = pd.read_csv(f"{T}/final_source_of_truth.csv")
    L = [r"\begingroup\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Benchmark characteristics.} Test sequences and positive class "
         r"at each level of homology filtering. \textsc{l0} is the published test set, "
         r"\textsc{l1} the sequences with no detectable nucleotide relative in training, "
         r"\textsc{l2} those with no detectable translated relative either. A dash means "
         r"the level retains too few sequences to score.}\label{tab:s1}",
         r"\begin{tabular}{l rrr rr}", r"\toprule",
         r"& \multicolumn{3}{c}{test sequences} & \multicolumn{2}{c}{positive class at "
         r"\textsc{l0}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"Benchmark & \textsc{l0} & \textsc{l1} & \textsc{l2} & count & share \\",
         r"\midrule"]
    for b in ORDER:
        d = df[df.benchmark == b].set_index("regime")
        g = lambda r, c: (d.loc[r, c] if r in d.index and not d.loc[r, "skipped"] else float("nan"))
        n0 = int(d.loc["L0", "n_rows"])
        cells = [f"{n0:,}"]
        for r in ("L1", "L2"):
            v = d.loc[r, "n_rows"] if r in d.index else float("nan")
            cells.append("--" if v != v or not v else f"{int(v):,}")
        pos = g("L0", "positives")
        cells += (["--", "--"] if pos != pos
                  else [f"{int(pos):,}", f"{100*pos/n0:.1f}\\%"])
        L.append(f"{PRETTY[b]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}\endgroup"]
    return "\n".join(L)


def s1():
    df = pd.read_csv(require(f"{T}/fragment_level_all_scores.csv",
        "the fragment-level score dump", "every model and arm score"))
    L = [r"\begingroup\tiny\setlength{\tabcolsep}{1.9pt}",
         r"\begin{longtable}{ll rrrrr rrrrrrrr}",
         r"\caption{\textbf{Every foundation model and every conventional predictor.} "
         r"MCC averaged over three training seeds on the population named. Nearest "
         r"neighbour lookup is unavailable at \textsc{l1} and \textsc{l2} by construction. "
         r"A dash means the arm does not exist for that benchmark.}"
         r"\label{tab:s2}\\", r"\toprule",
         r"& & \multicolumn{5}{c}{foundation models} & \multicolumn{8}{c}{baseline arms} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-15}",
         r"Benchmark & Reg. & " + " & ".join(MODEL[m] for m in F.MODELS) + " & "
         + " & ".join(ARMN[a] for a in ARMS) + r" \\", r"\midrule", r"\endfirsthead",
         r"\toprule Benchmark & Reg. & " + " & ".join(MODEL[m] for m in F.MODELS) + " & "
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
    """Dev and test MCC for every model, showing selection used dev only."""
    rows = []
    for name, mask, esc, atags, _ in F.CELLS:
        z = np.load(f"{F.MASKS}/{mask}.npz"); n = len(z["hasN"])
        best = None
        for m in F.MODELS:
            c = []
            for j in sorted(glob.glob(f"{F.FT}/{esc}__{m}__lr*__fullft.json")):
                if any(x in j for x in ("_meanpool", "_ep4", "_truncated")):
                    continue
                d = json.load(open(j))
                if len(d.get("runs", [])) >= 3:
                    sfx = "".join(x for x in ("__probs", "__ep16", "__ep32", "__ep64") if x in j)
                    c.append((float(np.mean([r["dev_mcc"] for r in d["runs"]])), d["lr"], sfx))
            if not c:
                continue
            dev, lr, sfx = max(c)
            y, ps, _ = F.model_preds(esc, m, n)
            te = float(np.mean([MCC(y, p) for p in ps])) if ps else np.nan
            rows.append(dict(bench=name, model=m, lr=lr, dev=dev, test=te))
    df = pd.DataFrame(rows)
    L = [r"\begingroup\small\setlength{\tabcolsep}{5pt}",
         r"\begin{longtable}{ll r rr}",
         r"\caption{\textbf{Model selection.} Each model's best learning rate by mean "
         r"development MCC over three seeds, the development score that chose it, and the "
         r"test score it goes on to obtain. The selected model is the one with the highest "
         r"development MCC, shown in bold, and it is held fixed across all populations. "
         r"The test column repeats values from Table~S2 and is placed beside the "
         r"development column so the two orderings can be compared; it enters no "
         r"selection. On pathogenicity \textsc{std} the two orderings disagree.}"
         r"\label{tab:s3}\\", r"\toprule",
         r"Benchmark & Model & lr & dev MCC & test MCC \\", r"\midrule\endfirsthead",
         r"\toprule Benchmark & Model & lr & dev MCC & test MCC \\ \midrule \endhead"]
    for name in [NAME2SHORT[b] for b in ORDER]:
        sub = df[df.bench == name].sort_values("dev", ascending=False)
        if sub.empty:
            continue
        sel = sub.iloc[0].model
        first = True
        for _, r in sub.iterrows():
            bn = PRETTY[SHORT[name]] if first else ""
            first = False
            bold = (lambda s: r"\textbf{" + s + "}") if r.model == sel else (lambda s: s)
            L.append(f"{bn} & {bold(MODEL[r.model])} & {r.lr:g} & "
                     f"{bold(f'{r.dev:.4f}')} & {r.test:.4f} \\\\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(L)


def s3():
    """The table the Limitations paragraph needs. Units and coverage on their own do not
    show that a verdict changes, so the two intervals are given side by side."""
    df = pd.read_csv(f"{T}/final_source_of_truth.csv")
    d = df[(~df.skipped) & (df.src_lo.notna()) & (df.n_rows >= 100)]
    L = [r"\begingroup\scriptsize\setlength{\tabcolsep}{2.4pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Resampling test sequences against resampling their parent "
         r"genomes.} Populations of at least 100 sequences for which a source genome could "
         r"be established. The difference in MCC is identical under both schemes and is "
         r"shown once. Source identifiers ship with VHDB and are recovered by alignment "
         r"for the HVUE tasks, where coverage is the share of sequences placed on a genome "
         r"and an unplaced sequence forms its own unit. The two GUE benchmarks ship no "
         r"provenance and are absent.}\label{tab:s4}",
         r"\begin{tabular}{ll r ll rr r}", r"\toprule",
         r"Benchmark & Reg. & diff. & sequence 95\% CI & genome 95\% CI "
         r"& $n$ & units & cov. \\", r"\midrule"]
    fv = {"SUPERIOR": r"\textsc{sup}", "INFERIOR": r"\textsc{inf}",
          "EQUIVALENT": r"\textsc{eqv}", "INCONCLUSIVE": r"\textsc{inc}"}
    for b in ORDER:
        sub = d[d.benchmark == b]
        first = True
        for _, r in sub.iterrows():
            nm = PRETTY[b] if first else ""
            first = False
            cov = "shipped" if r.provenance == "shipped" else f"{100*r.assign_cover:.0f}\\%"
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
    s = json.load(open(require(f"{R}/unassigned_sensitivity.json",
        "scripts/track_b_viral/unassigned_clustering_sensitivity.py",
        "hidden dependence scenarios")))
    SC = {"none": "as analysed, unplaced sequences are singletons",
          "observed": "hidden group sizes drawn from the observed distribution",
          "mean": "every hidden group at the mean placed group size",
          "pooled": "every hidden group at twice the mean placed size",
          "worst": "bound, all unplaced sequences in one group"}
    L = [r"\begingroup\small\setlength{\tabcolsep}{5pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Hidden dependence sensitivity.} Applied to every cell whose "
         r"genome interval excludes zero and whose source assignment is incomplete, "
         r"which is host tropism \textsc{hard\_70} at \textsc{l1} with 43 percent coverage. "
         r"Unplaced sequences are pseudo clustered under four scenarios and the genome "
         r"bootstrap repeated. The final row is a bound and not a plausible "
         r"scenario. The observed difference is unchanged throughout.}\label{tab:s5}",
         r"\begin{tabular}{p{0.42\textwidth} r r l l}", r"\toprule",
         r"Scenario & units & excess & 95\% CI & verdict \\", r"\midrule"]
    for r in s:
        L.append(f"{SC.get(r['scenario'], r['scenario'])} & {r['n_units']:,} & "
                 f"{r['excess']:+.3f} & $[{r['lo']:+.3f}, {r['hi']:+.3f}]$ & "
                 f"\\textsc{{{r['verdict'].lower()}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}\endgroup"]
    return "\n".join(L)


def s5():
    c = pd.read_csv(f"{T}/translated_effect_cells.csv")
    p = json.load(open(f"{T}/translated_effect_pooled.json"))
    L = [r"\begingroup\small\setlength{\tabcolsep}{5pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Association between translated relatedness and performance, "
         r"inside \textsc{l1}.} Each \textsc{l1} population is split into the sequences "
         r"that do and do not acquire a detectable six frame translated relative. Each "
         r"part is scored against the best conventional predictor and the difference "
         r"between the two parts bootstrapped. The $k$-mer "
         r"retaining both parts at usable size are shown.}\label{tab:s6}",
         r"\begin{tabular}{ll rrr l}", r"\toprule",
         r"Benchmark & Predictor & with relative & without & difference & 95\% CI \\",
         r"\midrule"]
    for b in c.benchmark.unique():
        sub = c[c.benchmark == b]
        first = True
        for _, r in sub.iterrows():
            bn = PRETTY.get(TRNAME.get(b, b), b) if first else ""
            first = False
            nm = MODEL.get(r.arm, ARMN.get(r.arm, r.arm))
            if r.is_baseline:
                nm += r" \textit{(control)}"
            L.append(f"{bn} & {nm} & {r.with_rel:+.3f} & {r.without_rel:+.3f} & "
                     f"{r.gap:+.3f} & $[{r.lo:+.3f}, {r.hi:+.3f}]$ \\\\")
        L.append(r"\addlinespace")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{l}{\textbf{pooled across all model by benchmark cells}} & "
             f"\\textbf{{{p['pooled_gap']:+.3f}}} & "
             f"$[{p['pooled_lo']:+.3f}, {p['pooled_hi']:+.3f}]$ \\\\")
    L.append(r"\multicolumn{4}{l}{sign of the difference, positive cells of all cells} & "
             f"{p['n_positive']} of {p['n_cells']} & $p={p['sign_p']}$ \\\\")
    L.append(r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}\endgroup"]
    return "\n".join(L)


def s6():
    j = json.load(open(require(f"{R}/regime_interaction.json",
        "scripts/track_b_viral/regime_interaction.py", "L0 to L1 interaction")))
    tex = lambda t: t.replace("_", r"\_")
    L = [r"\begingroup\small\setlength{\tabcolsep}{5pt}",
         r"\begin{table}[htbp]\centering",
         r"\caption{\textbf{Direct test of the \textsc{l0} to \textsc{l1} change in "
         r"incremental value.} $D$ is the excess at \textsc{l1} minus the excess at "
         r"\textsc{l0}, obtained by resampling \textsc{l0} units, carrying each sequence's "
         r"\textsc{l1} membership, and re selecting the best conventional predictor inside every replicate. "
         r"The first definition uses every predictor applicable in each population. The "
         r"second excludes nearest neighbour lookup from both, because "
         r"\textsc{l1} is defined by the absence of the nucleotide match it transfers from, "
         r"so its collapse there is definitional. $q$ is Benjamini-Hochberg adjusted across "
         r"the five cells of each definition. No test resolves after adjustment.}\label{tab:s7}",
         r"\begin{tabular}{l rr r l rr}", r"\toprule",
         r"Cell & $\Delta_{\textsc{l0}}$ & $\Delta_{\textsc{l1}}$ & $D$ & 95\% CI & $p$ & $q$ \\",
         r"\midrule"]
    RELABEL = {"operational envelope, every applicable arm":
               "every predictor applicable in each population",
               "common arm envelope, NN lookup excluded from both":
               "nearest neighbour lookup excluded from both populations"}
    for lab, rows in j.items():
        L.append(r"\multicolumn{7}{l}{\textit{" + RELABEL.get(lab, lab) + r"}} \\")
        for r in rows:
            L.append(f"\\quad {tex(r['cell'])} & {r['d_L0']:+.3f} & {r['d_L1']:+.3f} & "
                     f"{r['D']:+.3f} & $[{r['lo']:+.3f}, {r['hi']:+.3f}]$ & "
                     f"{r['p']:.3f} & {r['q']:.3f} \\\\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}",
          r"\vspace{2pt}\par\noindent\footnotesize VHDB and the two host tropism cells "
          r"resample source genomes; GUE phage and transmissibility \textsc{hard} resample "
          r"sequences because no source provenance is available, and transmissibility "
          r"\textsc{hard} retains only 55 sequences at \textsc{l1}.",
          r"\end{table}\endgroup"]
    return "\n".join(L)


def main():
    parts = ["% Generated by paper/figures/make_supplement_v2.py. Do not edit by hand.",
             r"\ifdefined\SUPPLEMENTINMAIN\else", r"\fi",
             r"\setcounter{table}{0}",
             r"\renewcommand{\thetable}{S\arabic{table}}",
             r"\section*{Supplementary tables}", ""]
    for fn in (s0, s1, s2, s3, s4, s5, s6):
        parts.append(fn()); parts.append("")
    out = os.path.join(REPO, "paper", "supplementary.tex")
    open(out, "w").write("\n".join(parts) + "\n")
    print(f"  wrote paper/supplementary.tex  (S1 to S7)")


if __name__ == "__main__":
    main()
