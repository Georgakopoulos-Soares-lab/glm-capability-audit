"""P5: supervised single-variant effect on ProteinGym VIRAL assays.

The 95-scorer ProteinGym leaderboard is entirely ZERO-SHOT. Nobody has tested whether SUPERVISED
adaptation on viral DMS beats MSA methods. Given the regime effect we measured elsewhere
(frozen->FT is worth up to +0.60 MCC on splice), a zero-shot-only leaderboard is exactly where a
real capability could be hidden.

Baselines are the point: a one-hot(position x AA) ridge and a biochemical-property ridge are the
pLM equivalent of the CNN on the DNA side. Without them a pLM "win" is meaningless.
Splits are position-disjoint (contiguous / modulo), so no mutated position appears in both sides.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import paths as P
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
OUT = P.sub("proteingym_supervised")
DMS = os.environ.get("VB_PGYM_DIR", "/data/nvidia/proteingym/DMS_ProteinGym_substitutions")
AAS = "ACDEFGHIKLMNPQRSTVWY"
# Kyte-Doolittle hydropathy, volume, charge, polarity -- a cheap biochemical featurisation
KD = dict(zip(AAS, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL= dict(zip(AAS, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHG= dict(zip(AAS, [0,0,-1,-1,0,0,0.1,0,1,0,0,0,0,0,1,0,0,0,0,0]))

def parse_mut(m):
    out=[]
    for tok in str(m).split(":"):
        tok=tok.strip()
        if len(tok)>=3 and tok[0] in AAS and tok[-1] in AAS:
            try: out.append((tok[0], int(tok[1:-1]), tok[-1]))
            except ValueError: pass
    return out

def feats_onehot(muts, positions):
    pi={p:i for i,p in enumerate(sorted(positions))}
    X=np.zeros((len(muts), len(pi)*len(AAS)+len(AAS)), np.float32)
    for r,ms in enumerate(muts):
        for wt,pos,mt in ms:
            if pos in pi: X[r, pi[pos]*len(AAS)+AAS.index(mt)]=1.0
            X[r, len(pi)*len(AAS)+AAS.index(wt)]-=1.0
    return X

def feats_biochem(muts):
    X=np.zeros((len(muts),8), np.float32)
    for r,ms in enumerate(muts):
        for wt,pos,mt in ms:
            X[r,0]+=KD[mt]-KD[wt]; X[r,1]+=VOL[mt]-VOL[wt]; X[r,2]+=CHG[mt]-CHG[wt]
            X[r,3]+=abs(KD[mt]-KD[wt]); X[r,4]+=abs(VOL[mt]-VOL[wt])
            X[r,5]+=1; X[r,6]=pos; X[r,7]+=KD[wt]
    return X

def esm_embed(seqs, mid, device, bs=8, maxlen=1022):
    from transformers import AutoTokenizer, AutoModel
    tok=AutoTokenizer.from_pretrained(mid); m=AutoModel.from_pretrained(mid).to(device).eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(seqs),bs):
            e=tok(seqs[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
            e={k:v.to(device) for k,v in e.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h=m(**e).last_hidden_state.float()
            mask=e["attention_mask"].unsqueeze(-1).float()
            out.append(((h*mask).sum(1)/mask.sum(1)).cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.vstack(out)

def split_idx(positions, scheme, n):
    ps=np.array(positions)
    if scheme=="modulo": return (ps % 5) < 4, (ps % 5) >= 4     # position-disjoint
    cut=np.percentile(ps, 80); return ps <= cut, ps > cut        # contiguous

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--esm", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--scheme", default="modulo", choices=["modulo","contiguous"])
    ap.add_argument("--device", default="cuda:0")
    a=ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    d=pd.read_csv(f"{DMS}/{a.assay}.csv")
    d["ms"]=d.mutant.map(parse_mut); d=d[d.ms.map(len)>0].reset_index(drop=True)
    d["pos"]=d.ms.map(lambda m:m[0][1])
    tr_m, te_m = split_idx(d.pos.values, a.scheme, len(d))
    print(f"{a.assay}: n={len(d)} train={tr_m.sum()} test={te_m.sum()} "
          f"positions={d.pos.nunique()} scheme={a.scheme}", flush=True)
    y=d.DMS_score.values; res=dict(assay=a.assay, scheme=a.scheme, n=len(d),
        n_train=int(tr_m.sum()), n_test=int(te_m.sum()), n_positions=int(d.pos.nunique()))
    banks={"onehot": feats_onehot(d.ms.tolist(), set(d.pos)), "biochem": feats_biochem(d.ms.tolist())}
    t0=time.time()
    banks["esm2"]=esm_embed(d.mutated_sequence.tolist(), a.esm, a.device)
    print(f"  esm2 embed {time.time()-t0:.0f}s dim={banks['esm2'].shape[1]}", flush=True)
    for nm,X in banks.items():
        sc=StandardScaler().fit(X[tr_m]); Xtr, Xte = sc.transform(X[tr_m]), sc.transform(X[te_m])
        best=(-9,None)
        for al in (0.1,1.0,10.0,100.0,1000.0):
            r=Ridge(alpha=al).fit(Xtr, y[tr_m])
            # alpha selected on a held-out slice of TRAIN, never on test
            k=max(1,int(0.15*Xtr.shape[0]))
            rr=Ridge(alpha=al).fit(Xtr[:-k], y[tr_m][:-k])
            s=spearmanr(rr.predict(Xtr[-k:]), y[tr_m][-k:]).correlation
            if s is not None and s>best[0]: best=(s, r)
        rho=spearmanr(best[1].predict(Xte), y[te_m]).correlation
        res[nm]=round(float(rho),4)
        print(f"  {nm:<9} test Spearman={rho:.4f}", flush=True)
    json.dump(res, open(f"{OUT}/{a.assay}__{a.scheme}.json","w"), indent=2)
    print(f"  wrote {a.assay}__{a.scheme}.json", flush=True)

if __name__=="__main__": main()
