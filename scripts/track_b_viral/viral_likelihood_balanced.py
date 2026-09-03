"""Family-balanced viral likelihood: does the Markov-beats-gLM result survive de-confounding?

The first run was confounded three ways. ViroBench ALL/times train is 84% Orthomyxoviridae
(38,997/46,651 influenza segments, median 2,316bp, 92% RNA); the test split is bacteriophage
(median 41,233bp, 72% DNA). So "order-7 Markov beats Evo by 0.78 bits/base in-distribution" may
just be "an 8-mer table memorises influenza".

Controls applied here:
  1. FAMILY CAP  -- at most --cap_family genomes per family in train AND eval
  2. LENGTH MATCH -- every scored sequence truncated to the same window
  3. MOLECULE MATCH -- optional --na_type filter so RNA/DNA does not shift between splits
  4. in-distribution eval is a held-out slice of the SAME (balanced) pool as train, so the
     only difference from the OOD eval is the temporal split
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import paths as P
import virobench_baselines as VBB
from viral_likelihood import fit_markov_tables, interp_bits_per_base, markov_bits_per_base, glm_bits_per_base
import warnings; warnings.filterwarnings("ignore")
OUT = P.sub("viral_likelihood")

def balance(df, cap, rng, na=None):
    if na: df = df[df.na_type == na]
    keep = []
    for fam, g in df.groupby("family"):
        idx = g.index.values
        if len(idx) > cap: idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    return df.loc[np.concatenate(keep)].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["hyenadna"])
    ap.add_argument("--cap_family", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--maxlen", type=int, default=2000)
    ap.add_argument("--minlen", type=int, default=2000, help="require >= this so length is matched")
    ap.add_argument("--na_type", default=None, choices=[None,"RNA","DNA"])
    ap.add_argument("--orders", nargs="+", type=int, default=[3,5,7])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    tr, dv, te = (VBB.load("ALL", "times", x) for x in ("train","val","test"))
    for d in (tr, dv, te):
        d.drop(d.index[d.sequence.str.len() < a.minlen], inplace=True)
    trb = balance(tr, a.cap_family, rng, a.na_type)
    teb = balance(te, a.cap_family, rng, a.na_type)
    # in-distribution eval = held-out slice of the SAME balanced train pool
    perm = rng.permutation(len(trb)); n_hold = min(a.n_eval, len(trb)//5)
    hold, fit = trb.iloc[perm[:n_hold]], trb.iloc[perm[n_hold:]]
    ev_ood = teb.iloc[rng.permutation(len(teb))[:a.n_eval]]
    cut = lambda d: [s[:a.maxlen] for s in d.sequence.tolist()]
    print(f"BALANCED (cap {a.cap_family}/family, len>={a.minlen} truncated to {a.maxlen}, na={a.na_type})")
    print(f"  fit={len(fit)} genomes over {fit.family.nunique()} families "
          f"| in-dist held-out={len(hold)} | OOD test={len(ev_ood)}", flush=True)
    print(f"  fit top families: {fit.family.value_counts().head(3).to_dict()}")
    print(f"  fit na_type: {fit.na_type.value_counts().to_dict()} | ood na_type: {ev_ood.na_type.value_counts().to_dict()}", flush=True)
    res = dict(cap_family=a.cap_family, maxlen=a.maxlen, na_type=a.na_type,
               n_fit=len(fit), n_families_fit=int(fit.family.nunique()),
               n_indist=len(hold), n_ood=len(ev_ood), indist={}, ood={})
    fits, hs, os_ = cut(fit), cut(hold), cut(ev_ood)
    for k in a.orders:
        bi,_ = markov_bits_per_base(fits, hs, k, cap=a.maxlen)
        bo,_ = markov_bits_per_base(fits, os_, k, cap=a.maxlen)
        res["indist"][f"markov_order{k}"]=round(float(bi),4); res["ood"][f"markov_order{k}"]=round(float(bo),4)
        print(f"  markov-{k}: in-dist {bi:.4f} | OOD {bo:.4f}", flush=True)
    for key in a.models:
        try:
            bi,_ = glm_bits_per_base(key, hs, a.device, a.maxlen)
            bo,_ = glm_bits_per_base(key, os_, a.device, a.maxlen)
            res["indist"][key]=round(float(bi),4); res["ood"][key]=round(float(bo),4)
            print(f"  {key}: in-dist {bi:.4f} | OOD {bo:.4f}", flush=True)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
    bm_i=min(v for k,v in res["indist"].items() if k.startswith("markov"))
    bm_o=min(v for k,v in res["ood"].items() if k.startswith("markov"))
    for key in a.models:
        if key in res["indist"]:
            res[f"{key}_vs_markov_indist"]=round(res["indist"][key]-bm_i,4)
            res[f"{key}_vs_markov_ood"]=round(res["ood"][key]-bm_o,4)
            print(f"\n  {key} vs best Markov: in-dist {res[f'{key}_vs_markov_indist']:+.4f} | "
                  f"OOD {res[f'{key}_vs_markov_ood']:+.4f} bits/base", flush=True)
    tag=f"balanced_cap{a.cap_family}"+(f"_{a.na_type}" if a.na_type else "")+"__"+"_".join(sorted(a.models))
    json.dump(res, open(f"{OUT}/{tag}.json","w"), indent=2)
    print(f"wrote {OUT}/{tag}.json", flush=True)

if __name__=="__main__": main()
