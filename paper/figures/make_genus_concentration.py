"""Genus concentration of the positive class, for every stratum in the ladder.

A stratum can reduce to one compositionally distinctive genus without that
being visible in its row count or its class balance. The std_95 unseen-genus
stratum does exactly this: every positive row in it is Molluscipoxvirus. This
script measures the same two quantities everywhere, so a stratum being read as
informative can be checked against the same failure.

Native genus labels are used where the benchmark ships them, alignment-recovered
labels otherwise, and the two are never pooled. Rows whose genus could not be
assigned are counted and excluded from the concentration numbers, never imputed.

Where concentration is high, GC content of the positive class is reported
against the negative class: a stratum separable by base composition alone is
not testing what the benchmark says it tests.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from inject import inject, data_path, require, require_rows
import argparse, collections, glob, json, os
import numpy as np, pandas as pd

# Three roots, all overridable, so this runs off the cluster it was written on.
#   VB_RESULTS   the prediction archive: preds_rescore/, hidden_masks/, geneb_finetune/
#   VB_REPORTS   the upstream report tree holding taxonomy_recovery/
#   VB_CORPORA   the benchmark corpora: hvue_v2/, vhdb_tropism/, gue_devcarve/
_RVB = os.environ.get("VB_RESULTS",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
_REP = os.environ.get("VB_REPORTS", "/data/nvidia/genomic-capability-unlearning/reports")
_COR = os.environ.get("VB_CORPORA", "/data/nvidia")

PRED = f"{_RVB}/preds_rescore"
TAX = f"{_REP}/taxonomy_recovery"
MASK = f"{_RVB}/hidden_masks"
NATIVE = {"VHDB_within_family": f"{_COR}/vhdb_tropism/within_family_1000bp/test_tax.csv"}
NATIVE_TRAIN = {"VHDB_within_family": f"{_COR}/vhdb_tropism/within_family_1000bp/train_tax.csv"}
MASK_ALIAS = {"VHDB_within_family": "VHDB_v2", "GUE_phage": "GUE_phage_v2"}
TESTCSV = {
    "HVUE_HT_hard_70": f"{_COR}/hvue_v2/Host_Tropism/hard_70_1000bp/test.csv",
    "HVUE_HT_std_95": f"{_COR}/hvue_v2/Host_Tropism/standard_95_1000bp/test.csv",
    "HVUE_Transm_hard": f"{_COR}/hvue_v2/Transmissibility/hard_capped_1000bp/test.csv",
    "HVUE_Transm_std": f"{_COR}/hvue_v2/Transmissibility/standard_capped_1000bp/test.csv",
    "HVUE_Patho_hard": f"{_COR}/hvue_v2/Pathogenicity/hard_capped_1000bp/test.csv",
    "HVUE_Patho_std": f"{_COR}/hvue_v2/Pathogenicity/standard_capped_1000bp/test.csv",
    "GUE_covid": f"{_COR}/gue_devcarve/leannmlindsey_GUE_virus_covid/test.csv",
    "GUE_phage": f"{_COR}/gue_devcarve/leannmlindsey_GUE_phage_fragments/test.csv",
    "VHDB_within_family": f"{_COR}/vhdb_tropism/within_family_1000bp/test.csv",
}
ESC = {"HVUE_HT_hard_70": "Host_Tropism__hard_70_1000bp",
       "HVUE_HT_std_95": "Host_Tropism__standard_95_1000bp",
       "HVUE_Patho_hard": "Pathogenicity__hard_capped_1000bp",
       "HVUE_Patho_std": "Pathogenicity__standard_capped_1000bp",
       "HVUE_Transm_hard": "Transmissibility__hard_capped_1000bp",
       "HVUE_Transm_std": "Transmissibility__standard_capped_1000bp",
       "GUE_phage": "gue_phage_fragments",
       "GUE_covid": "gue_virus_covid",
       "VHDB_within_family": "vhdb_tropism_wf1000"}
LEVEL_LABEL = {"L0": "L0", "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4", "L5": "L5"}


def gc_of(s):
    u = str(s).upper()
    n = u.count("G") + u.count("C") + u.count("A") + u.count("T")
    return (u.count("G") + u.count("C")) / n if n else np.nan


def load_labels(name, n):
    """(genus array, source) or (None, reason). Never pools native with recovered."""
    p = NATIVE.get(name)
    if p and os.path.exists(p):
        d = pd.read_csv(p)
        if "genus" in d.columns and len(d) == n:
            return d["genus"].astype(str).fillna("unassigned").values, "native"
    for root in (TAX, "/home/nvidia/genomic-capability-unlearning/reports/taxonomy_recovery"):
        q = f"{root}/{name}__recovered.npz"
        if os.path.exists(q):
            z = np.load(q, allow_pickle=True)
            k = "test_genus" if "test_genus" in z else z.files[0]
            if len(z[k]) == n:
                return np.asarray(z[k], dtype=object), "recovered"
    return None, "no genus labels available"


def strata_masks(name, lay, n):
    """Row masks for every stratum the ladder reports, rebuilt from the caches."""
    out = [("L0", np.ones(n, bool))]
    p = f"{MASK}/{MASK_ALIAS.get(name, name)}.npz"
    if not os.path.exists(p):
        p = f"{MASK}/{name}.npz"
    if os.path.exists(p):
        z = np.load(p)
        if len(z["hasN"]) == n:
            hn, ht = z["hasN"], z["hasT"]
            out.append(("L1", ~hn))
            out.append(("L2", ~hn & ~ht))
            if "hasP" in z:
                out.append(("L3", ~hn & ~ht & ~z["hasP"]))
    # native lineage strata, rebuilt the way the ladder builds them
    tp, trp = NATIVE.get(name), NATIVE_TRAIN.get(name)
    if tp and trp and os.path.exists(tp) and os.path.exists(trp):
        tx, tr = pd.read_csv(tp), pd.read_csv(trp)
        if len(tx) == n:
            # the ladder nests these inside the no-translated-hit stratum
            nest = dict(out).get("L2", np.ones(n, bool))
            for col, lvl in (("species", "L4"), ("genus", "L5")):
                if col in tx.columns and col in tr.columns:
                    u = (~tx[col].astype(str).isin(set(tr[col].astype(str)))).values
                    out.append((lvl, u & nest))
    # mask files are named by the pipeline, not by this paper's regime names
    MASKFILE = {"L4": "unseen_species", "L5": "unseen_genus_recovered"}
    for lvl in ("L4", "L5"):
        for esc in {ESC.get(name, ""), name}:
            q = f"{PRED}/hvue_v2__{esc}__{MASKFILE[lvl]}__mask.npz"
            if esc and os.path.exists(q):
                zz = np.load(q)
                k = [x for x in zz.files if len(zz[x]) == n]
                if k:
                    out.append((lvl, zz[k[0]].astype(bool)))
                break
    keep, seen = [], set()
    for lab, m in out:
        if lab not in seen and lab in lay["panelB"]:
            keep.append((lab, m)); seen.add(lab)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layered", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                  "reports", "layered"))
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--min-labelled", type=int, default=10,
                    help="below this many labelled positive rows a share is not reported")
    ap.add_argument("--gc-threshold", type=float, default=0.50,
                    help="report GC where the largest genus holds at least this share")
    a = ap.parse_args()
    res, lines = {}, []

    for f in sorted(glob.glob(f"{a.layered}/*__layered.json")):
        lay = json.load(open(f)); name = lay["name"]
        yf = sorted(glob.glob(f"{PRED}/hvue_v2__{ESC.get(name, name)}__kmer__s*__preds.npz")) \
            or sorted(glob.glob(f"{PRED}/*{name}*__kmer__s*__preds.npz"))
        if not yf:
            raise SystemExit(f"no prediction file for {name}, so its labels cannot be read"
                             f"\n  expected {PRED}/hvue_v2__*__kmer__s*__preds.npz"
                             f"\n  produce it with: pipelines/run.py --stage baselines")
        y = np.load(yf[0])["y_true"]
        n = len(y)
        g, src = load_labels(name, n)
        seq = None
        tp = TESTCSV.get(name)
        if tp and os.path.exists(tp):
            d = pd.read_csv(tp)
            sc = [c for c in d.columns if c.lower() in ("sequence", "seq", "text")]
            if sc and len(d) == n:
                seq = d[sc[0]].astype(str).values

        lines.append(f"\n{name}   labels: {src}   n_test={n:,}")
        if g is None:
            res[name] = dict(labels=src, strata={}); continue
        res[name] = dict(labels=src, strata={})
        ncls = len(np.unique(y))
        multi = ncls > 2
        if multi:
            lines.append(f"  label has {ncls} classes, so there is no single positive "
                         f"class; concentration is reported over all rows")
        res[name]["n_classes"] = int(ncls)
        lines.append(f"  {'stratum':32s} {'n':>7s} {'pos':>6s} {'rate':>7s} "
                     f"{'genera+':>8s} {'top share':>10s}  {'GC pos/neg':>14s}")
        for lvl, m in strata_masks(name, lay, n):
            if m.sum() < a.min_rows:
                continue
            ym, gm = y[m], g[m]
            pos = np.ones(len(ym), bool) if multi else (ym == 1)
            ok = pos & (gm != "unassigned") & (gm != "nan") & (gm != "")
            npos = int(pos.sum())
            if npos == 0:
                continue
            c = collections.Counter(gm[ok])
            ngen = len(c)
            top, topn = (c.most_common(1)[0] if c else (None, 0))
            share = topn / int(ok.sum()) if ok.sum() else float("nan")
            r = dict(n=int(m.sum()), positives=npos, multiclass=bool(multi),
                     positive_rate=None if multi else round(float((ym == 1).mean()), 4),
                     positives_with_genus=int(ok.sum()),
                     n_positive_genera=ngen,
                     top_positive_genus=top,
                     top_positive_share=None if np.isnan(share) else round(float(share), 4))
            # a share computed on a handful of labelled rows is not a measurement
            thin = int(ok.sum()) < a.min_labelled
            r["share_underpowered"] = bool(thin)
            gcs = ""
            if (seq is not None and not multi and not np.isnan(share)
                    and share >= a.gc_threshold and not thin):
                v = np.array([gc_of(s) for s in seq[m]])
                gp, gn = float(np.nanmean(v[pos])), float(np.nanmean(v[~pos]))
                sd = float(np.nanstd(v[~pos]))
                r.update(gc_positive=round(gp, 4), gc_negative=round(gn, 4),
                         gc_negative_sd=round(sd, 4),
                         gc_separation_sd=round((gp - gn) / sd, 2) if sd else None)
                gcs = f"  {gp:.3f}/{gn:.3f} ({(gp - gn) / sd:+.1f} sd)" if sd else ""
            res[name]["strata"][lvl] = r
            rate = "  --  " if multi else f"{(ym == 1).mean():>6.1%}"
            sh = "--" if np.isnan(share) else (f"{share:>8.1%}*" if thin else f"{share:>9.1%}")
            lines.append(f"  {LEVEL_LABEL.get(lvl, lvl):32s} {m.sum():>7,} {npos:>6,} "
                         f"{rate} {ngen:>8,} {sh}{gcs}")
            if thin:
                lines.append(f"      * only {int(ok.sum())} positive rows carry a genus "
                             f"label, too few to read a share from")
            if ngen <= 5 and c:
                for gg, k in c.most_common():
                    lines.append(f"      {gg[:30]:30s} {k:>5,} positive rows")
            if int(ok.sum()) < npos:
                lines.append(f"      {npos - int(ok.sum()):,} positive rows have no genus "
                             f"label and are excluded, never imputed")

    require_rows([k for k, v in res.items() if v.get("strata")],
                 "genus_concentration", "pipelines/run.py --stage layered")
    txt = "\n".join(lines)
    open(data_path("genus_concentration.txt"), "w").write(txt + "\n")
    json.dump(res, open(data_path("genus_concentration.json"), "w"), indent=1)
    print(txt)
    print("\n  wrote reports/tables/genus_concentration.txt and .json")


if __name__ == "__main__":
    main()
