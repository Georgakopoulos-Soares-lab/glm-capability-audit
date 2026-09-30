"""ViroBench audit table: homology survival and genus concentration, six splits.

ViroBench is a contemporary benchmark built after the ones audited here, and it
ships both a taxonomic and a temporal split of the same taxon pool. That makes it
the one place where the two designs can be compared without entangling them with
the group, the sequences or the task. The question is whether either design yields
a stratum that is at once homology free, genus disjoint and taxonomically diverse.

Reads the masks persisted by virobench_homology_audit.py. No models, no floor.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from inject import inject, data_path, require, require_rows
import argparse, collections, json, os
import numpy as np

STEMS = ["ALL_taxon_genus", "ALL_taxon_times", "DNA_taxon_genus",
         "DNA_taxon_times", "RNA_taxon_genus", "RNA_taxon_times"]
STRATA = [("all test rows", "all"),
          ("genus disjoint rows", "gd"),
          ("no homology at all", "nh"),
          ("genus disjoint and no homology", "gdnh")]


def esc(s):
    return s.replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="/home/nvidia/genomic-capability-unlearning/reports/virobench_audit")
    ap.add_argument("--min-rows", type=int, default=10,
                    help="below this a share is reported but flagged as unreadable")
    a = ap.parse_args()
    res, lines = {}, []

    missing = [s_ for s_ in STEMS if not os.path.exists(f"{a.masks}/{s_}__masks.npz")]
    if missing:
        raise SystemExit("missing search masks for " + ", ".join(missing) +
                         "\n  produce them with: virobench_homology_audit.py")
    for stem in STEMS:
        z = np.load(f"{a.masks}/{stem}__masks.npz", allow_pickle=True)
        hn, ht, ng, g, y = z["hasN"], z["hasT"], z["novel_genus"], z["genus"], z["label"]
        n = len(g)
        masks = dict(all=np.ones(n, bool), gd=ng.astype(bool),
                     nh=(~hn) & (~ht), gdnh=(~hn) & (~ht) & ng.astype(bool))
        r = dict(n_test=n,
                 genus_disjoint_frac=round(float(ng.mean()), 4),
                 no_nucleotide_hit=round(float((~hn).mean()), 4),
                 translated_rescue=round(float((ht[~hn]).mean()), 4) if (~hn).any() else None)
        lines.append(f"\n{stem}   n_test={n:,}   genus disjoint {ng.mean():.1%}   "
                     f"no nucleotide hit {(~hn).mean():.1%}   "
                     f"of those translated hit {(ht[~hn]).mean():.1%}")
        lines.append(f"  {'stratum':32s} {'n':>7s} {'genera':>8s} {'largest':>9s} "
                     f"{'top 5':>8s}  {'classes usable':>14s}")
        for lab, key in STRATA:
            m = masks[key]
            if m.sum() == 0:
                continue
            c = collections.Counter(g[m])
            top = c.most_common(1)[0]
            top5 = sum(k for _, k in c.most_common(5)) / m.sum()
            cls = collections.Counter(y[m])
            usable = sum(1 for _, k in cls.items() if k >= a.min_rows)
            per = {}
            for cl, k in cls.items():
                cc = collections.Counter(g[m][y[m] == cl])
                per[str(cl)] = dict(n=int(k), n_genera=len(cc),
                                    top_share=round(cc.most_common(1)[0][1] / k, 4),
                                    underpowered=bool(k < a.min_rows))
            r[key] = dict(n=int(m.sum()), n_genera=len(c), top_genus=str(top[0]),
                          top_share=round(top[1] / m.sum(), 4),
                          top5_share=round(float(top5), 4),
                          n_classes_usable=int(usable), per_class=per)
            lines.append(f"  {lab:32s} {m.sum():>7,} {len(c):>8,} "
                         f"{top[1] / m.sum():>8.1%} {top5:>7.1%}  {usable:>13,}"
                         f"   {str(top[0])[:24]}")
        res[stem] = r

    txt = "\n".join(lines)
    open(data_path("virobench_audit.txt"), "w").write(txt + "\n")
    json.dump(res, open(data_path("virobench_audit.json"), "w"), indent=1)

    tex = [r"\begin{table}[htbp]", r"\centering", r"\small",
           r"\caption{ViroBench's own splits, searched against their own training "
           r"split. For each split the table gives how much of the test set has no "
           r"detectable relative, and how many genera the surviving sequences draw on. "
           r"The temporal splits and the taxonomic splits were built from the same "
           r"taxon pool by the same group, so the difference between them is the split "
           r"rule alone. The terminal population is the set of sequences that are both "
           r"disjoint in genus and free of any detected relationship.}",
           r"\label{tab:virobench}",
           r"\begin{tabular}{l r r r r r r}", r"\toprule",
           r"Split & $n$ & genus disjoint & no nucl.\ hit & terminal $n$ & genera & "
           r"largest genus \\", r"\midrule"]
    for stem, r in res.items():
        gd = r.get("gdnh") or r.get("nh")
        tex.append(f"{esc(stem)} & {r['n_test']:,} & {r['genus_disjoint_frac']:.1%} & "
                   f"{r['no_nucleotide_hit']:.1%} & {gd['n']:,} & {gd['n_genera']:,} & "
                   f"{gd['top_share']:.1%} \\\\".replace("%", r"\%"))
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    inject("main.tex", "virobench_audit", "\n".join(tex))
    print(txt)
    print("\n  virobench table injected into main.tex")


if __name__ == "__main__":
    main()
