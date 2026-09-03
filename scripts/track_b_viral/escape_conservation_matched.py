"""Phenotype-MATCHED test: does site-independent conservation clear AUROC 0.6 on antibody escape?

The earlier version compared conservation on Starr-style ACE2-BINDING DMS (0.686) against a
reported 0.6 on ANTIBODY ESCAPE. Different phenotype, different label type, different metric --
a cross-assay comparison, which this project's own rules forbid. That comparison is withdrawn.

This one is matched:
  task    = antibody escape, Bloom-lab SARS2_RBD_Ab_escape_maps (392,456 variant x antibody rows,
            1,599 antibody conditions, RBD sites 331-531)
  labels  = binary escape, thresholded on escape magnitude aggregated over antibodies
  metric  = AUROC, same as the reported number
  scorer  = site-independent conservation from ProteinGym's OWN SPIKE_SARS2 MSA
            (36,931 sequences, reference = full spike 1-1273, so site numbering maps directly)

Sequence weighting at 80% identity, add-one pseudocounts. Score = log2(f_mut / f_wt) at the site,
the standard site-independent substitution score. No model, no training, no fitting to labels.
"""
import argparse, json, os, sys, gzip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import paths as P
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
OUT = P.sub("escape_conservation")
A2M = "/data/nvidia/escape_data/DMS_msa_files/SPIKE_SARS2_theta0.99_full_11-26-2021_b01.a2m"
ESC = "/data/nvidia/escape_data/escape_data.csv"
AAS = "ACDEFGHIKLMNPQRSTVWY"

def read_a2m(path, max_seqs=None):
    seqs=[]; cur=[]
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur)); cur=[]
                if max_seqs and len(seqs)>=max_seqs: break
            else: cur.append(line.strip())
        if cur and (not max_seqs or len(seqs)<max_seqs): seqs.append("".join(cur))
    ref=seqs[0]
    # keep only match columns (uppercase or gap in the reference) -> reference coordinates
    keep=[i for i,c in enumerate(ref) if c.isupper() or c=="-"]
    M=np.full((len(seqs), len(keep)), 20, np.int8)   # 20 = gap/other
    idx={a:i for i,a in enumerate(AAS)}
    for r,s in enumerate(seqs):
        if len(s)<len(ref): s=s.ljust(len(ref),"-")
        for j,i in enumerate(keep):
            c=s[i].upper()
            if c in idx: M[r,j]=idx[c]
    return M, ref

def seq_weights(M, thresh=0.8, sample=4000, rng=None):
    """1/n_neighbours weighting at `thresh` identity, estimated on a subsample for tractability."""
    rng = rng or np.random.default_rng(0)
    n = M.shape[0]
    sub = rng.choice(n, min(sample, n), replace=False)
    S = M[sub]
    w = np.ones(n, np.float64)
    # neighbour count of every sequence against the subsample
    for i in range(n):
        ident = (M[i][None, :] == S).mean(1)
        w[i] = 1.0 / max(1.0, (ident >= thresh).sum() * (n / len(sub)) ** 0.0)
    return w

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max_seqs", type=int, default=12000)
    ap.add_argument("--weight_thresh", type=float, default=0.8)
    ap.add_argument("--agg", default="mean", choices=["mean","max","frac"])
    ap.add_argument("--pctiles", nargs="+", type=float, default=[80,85,90,95])
    a=ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    print(f"reading MSA (max {a.max_seqs} seqs)...", flush=True)
    M, ref = read_a2m(A2M, a.max_seqs)
    print(f"  MSA {M.shape[0]} x {M.shape[1]} | ref len {len(ref)}", flush=True)
    print("computing sequence weights...", flush=True)
    w = seq_weights(M, a.weight_thresh)
    print(f"  effective n = {w.sum():.1f}", flush=True)
    # weighted per-site aa frequencies with add-one pseudocounts
    L=M.shape[1]; F=np.ones((L,20))*1.0
    for r in range(M.shape[0]):
        row=M[r]; ok=row<20
        np.add.at(F, (np.where(ok)[0], row[ok]), w[r])
    F=F/F.sum(1,keepdims=True)
    e=pd.read_csv(ESC, low_memory=False)
    e=e[(e.site>=331)&(e.site<=531)]
    print(f"  escape rows {len(e)} | antibody conditions {e.condition.nunique()}", flush=True)
    if a.agg=="mean":   agg=e.groupby(["site","wildtype","mutation"]).mut_escape.mean()
    elif a.agg=="max":  agg=e.groupby(["site","wildtype","mutation"]).mut_escape.max()
    else:               agg=e.groupby(["site","wildtype","mutation"]).apply(lambda g:(g.mut_escape>0.1).mean())
    v=agg.reset_index().rename(columns={0:"escape"})
    v.columns=["site","wildtype","mutation","escape"] if "mut_escape" not in v else v.columns
    if "mut_escape" in v: v=v.rename(columns={"mut_escape":"escape"})
    idx={c:i for i,c in enumerate(AAS)}
    keep=[]; cons=[]
    for _,r in v.iterrows():
        s=int(r.site)-1
        if s<0 or s>=L or r.wildtype not in idx or r.mutation not in idx: continue
        # sanity: MSA reference must agree with the escape-map wildtype at this site
        if ref[s].upper()!=r.wildtype: continue
        keep.append(True); cons.append(np.log2(F[s,idx[r.mutation]]/F[s,idx[r.wildtype]]))
    v=v.iloc[:len(keep)][np.array(keep)] if len(keep)!=len(v) else v
    v=v.copy(); v["cons"]=cons
    print(f"  scored {len(v)} unique (site,mut) variants after wildtype-agreement check", flush=True)
    res=dict(n_variants=int(len(v)), n_msa=int(M.shape[0]), agg=a.agg,
             spearman=round(float(spearmanr(-v.cons, v.escape).correlation),4), aurocs={})
    print(f"\n  Spearman(-conservation, escape) = {res['spearman']:+.4f}")
    print(f"  {'escape label threshold':<26}{'n_pos':>7}{'AUROC':>9}")
    for p in a.pctiles:
        thr=np.percentile(v.escape,p); y=(v.escape>=thr).astype(int)
        if y.sum()<20 or y.sum()==len(y): continue
        au=roc_auc_score(y,-v.cons); au=max(au,1-au)
        res["aurocs"][f"top{100-int(p)}pct"]=round(float(au),4)
        print(f"  top {100-int(p)}% most-escaping{'':<6}{int(y.sum()):>7}{au:>9.4f}")
    res["max_auroc"]=max(res["aurocs"].values()) if res["aurocs"] else None
    json.dump(res, open(f"{OUT}/matched_escape_conservation.json","w"), indent=2)
    v.to_csv(f"{OUT}/matched_escape_variant_scores.csv", index=False)
    print(f"\n  => conservation AUROC on ANTIBODY ESCAPE: max {res['max_auroc']}")
    print(f"  wrote {OUT}/matched_escape_conservation.json")

if __name__=="__main__": main()
