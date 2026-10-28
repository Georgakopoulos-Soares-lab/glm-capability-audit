"""Full-context viral likelihood across tokenisation schemes.

Two corrections over the earlier runs:

1. BITS PER BASE, not per token. Evo and HyenaDNA are character-level (1 token = 1 base) so
   dividing NLL by token count is fine. GENERator is ~6 bp/token (vocab 4128) -- dividing by
   tokens would understate its bits/base by ~6x and make it look far better than it is.
   Here every model's NLL is divided by the number of BASES it actually predicted.

2. FULL CONTEXT. The balanced run truncated to 2,000 bp, which may remove the long-range
   structure Evo (8k) and GENERator (16k) are built for. This scores an 8,000 bp window --
   Evo's full context -- identically for every model and for the Markov baseline.

Corpus is the FAMILY-BALANCED pool (cap 200/family): the unbalanced ViroBench train split is
84% Orthomyxoviridae and an order-7 Markov table memorises it (1.26 vs 1.96 bits/base balanced).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import paths as P
import virobench_baselines as VBB
from viral_likelihood import markov_bits_per_base
import transformers; transformers.logging.set_verbosity_error()
import warnings; warnings.filterwarnings("ignore")
OUT = P.sub("viral_likelihood")
SPECS = {
 "evo":        ("togethercomputer/evo-1-8k-base", 8192, torch.bfloat16),
 "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf", 160000, torch.float32),
 "generator12":("GenerTeam/GENERator-eukaryote-1.2b-base", 16384, torch.bfloat16),
}

@torch.no_grad()
def bits_per_base(key, seqs, device, window):
    """total NLL in bits / number of BASES predicted -- correct for any bp-per-token ratio."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid, maxtok, dt = SPECS[key]
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, torch_dtype=dt).to(device).eval()
    tot_bits = 0.0; tot_bases = 0; t0 = time.time(); skipped = 0
    for i, s in enumerate(seqs):
        s = s[:window]
        ids = tok(s, return_tensors="pt")["input_ids"]
        if ids.shape[1] > maxtok: ids = ids[:, :maxtok]
        if ids.shape[1] < 8: skipped += 1; continue
        ids = ids.to(device)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(ids).logits.float()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); skipped += 1; continue
        lp = torch.log_softmax(lg[:, :-1], -1)
        nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum().item()
        # bases predicted = length of the span covered by tokens 1..n-1
        try:
            first = tok.decode(ids[0, :1], skip_special_tokens=True)
            n_bases = len([c for c in s if c.upper() in "ACGT"]) - len([c for c in first if c.upper() in "ACGT"])
        except Exception:
            n_bases = ids.shape[1] - 1
        if n_bases <= 0: skipped += 1; continue
        tot_bits += nll / np.log(2); tot_bases += n_bases
        if i and i % 25 == 0: print(f"    {key} {i}/{len(seqs)} ({time.time()-t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
    return tot_bits / max(tot_bases, 1), tot_bases, skipped

def balance(df, cap, rng):
    keep=[]
    for fam,g in df.groupby("family"):
        idx=g.index.values
        if len(idx)>cap: idx=rng.choice(idx,cap,replace=False)
        keep.append(idx)
    return df.loc[np.concatenate(keep)].reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",nargs="+",default=["evo"])
    ap.add_argument("--window",type=int,default=8000)
    ap.add_argument("--cap_family",type=int,default=200)
    ap.add_argument("--n_eval",type=int,default=100)
    ap.add_argument("--orders",nargs="+",type=int,default=[3,5,7])
    ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    rng=np.random.default_rng(0)
    tr,dv,te=(VBB.load("ALL","times",x) for x in ("train","val","test"))
    for d in (tr,te): d.drop(d.index[d.sequence.str.len()<a.window], inplace=True)
    trb=balance(tr,a.cap_family,rng); teb=balance(te,a.cap_family,rng)
    perm=rng.permutation(len(trb)); nh=min(a.n_eval,len(trb)//5)
    hold,fit=trb.iloc[perm[:nh]],trb.iloc[perm[nh:]]
    ood=teb.iloc[rng.permutation(len(teb))[:a.n_eval]]
    cut=lambda d:[s[:a.window] for s in d.sequence.tolist()]
    print(f"FULL-CONTEXT window={a.window}bp, balanced cap {a.cap_family}/family")
    print(f"  fit={len(fit)} ({fit.family.nunique()} families) | in-dist={len(hold)} | OOD={len(ood)}",flush=True)
    print(f"  fit na_type {fit.na_type.value_counts().to_dict()} | ood {ood.na_type.value_counts().to_dict()}",flush=True)
    res=dict(window=a.window,cap_family=a.cap_family,n_fit=len(fit),
             n_families=int(fit.family.nunique()),indist={},ood={})
    fits,hs,os_=cut(fit),cut(hold),cut(ood)
    for k in a.orders:
        bi,_=markov_bits_per_base(fits,hs,k,cap=a.window); bo,_=markov_bits_per_base(fits,os_,k,cap=a.window)
        res["indist"][f"markov_order{k}"]=round(float(bi),4); res["ood"][f"markov_order{k}"]=round(float(bo),4)
        print(f"  markov-{k}: in-dist {bi:.4f} | OOD {bo:.4f}",flush=True)
    for key in a.models:
        try:
            bi,nb,sk=bits_per_base(key,hs,a.device,a.window)
            bo,nb2,sk2=bits_per_base(key,os_,a.device,a.window)
            res["indist"][key]=round(float(bi),4); res["ood"][key]=round(float(bo),4)
            res[f"{key}_bases_scored"]=[nb,nb2]; res[f"{key}_skipped"]=[sk,sk2]
            print(f"  {key}: in-dist {bi:.4f} | OOD {bo:.4f}  (bases {nb}/{nb2}, skipped {sk}/{sk2})",flush=True)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:170]}",flush=True); res["indist"][key]=None
    bi_m=min(v for k,v in res["indist"].items() if k.startswith("markov"))
    bo_m=min(v for k,v in res["ood"].items() if k.startswith("markov"))
    for key in a.models:
        if res["indist"].get(key) is not None:
            res[f"{key}_vs_markov_indist"]=round(res["indist"][key]-bi_m,4)
            res[f"{key}_vs_markov_ood"]=round(res["ood"][key]-bo_m,4)
            print(f"\n  {key} vs best Markov: in-dist {res[f'{key}_vs_markov_indist']:+.4f} | OOD {res[f'{key}_vs_markov_ood']:+.4f}",flush=True)
    tag=f"fullctx{a.window}_cap{a.cap_family}__"+"_".join(sorted(a.models))
    json.dump(res,open(f"{OUT}/{tag}.json","w"),indent=2)
    print(f"wrote {OUT}/{tag}.json",flush=True)

if __name__=="__main__": main()
