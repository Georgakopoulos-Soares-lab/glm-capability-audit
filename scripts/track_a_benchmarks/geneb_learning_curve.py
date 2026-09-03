"""Is the FM's excess over a searched baseline a function of TRAINING-SET SIZE?

Hypothesis: pretraining buys DATA EFFICIENCY, not a higher ceiling. If so, the
'erasable excess' = FM - max(kmer, CNN) should be large at small n and decay to ~0 at large n.

That would explain our whole result set at once, and it lands the unlearning argument: the
safety-relevant benchmarks (HVUE 47k-458k rows, ViroBench 46k) all sit in the large-n regime
where the excess is predicted to vanish -- which is what we measure.

Critically, the CNN is re-searched at EVERY n (a baseline tuned at n=100k is the wrong baseline
at n=100), and the k-mer C is re-tuned at every n. Dev is carved from the SUBSAMPLED train only.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import paths as P
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import transformers, logging
transformers.logging.set_verbosity_error(); logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
from capacity_sweep import Dilated, UNet, ResNet, build, receptive_field   # shared architectures
OUT = P.sub("learning_curve"); GENEB = os.environ.get("VB_GENEB_DIR", "/data/nvidia/geneb_data/tasks")
MAPI = np.full(256, 4, np.int64)
for i,c in enumerate("ACGT"): MAPI[ord(c)]=i; MAPI[ord(c.lower())]=i
FM = {"nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 1024, 6),
      "dnabert2":   ("zhihan1996/DNABERT-2-117M", 512, 5)}
# compact ladder: the (arch,k,dbase) families that actually won the full 3-stage search
LAD = [("dilated",128,5,9,2),("dilated",128,5,9,4),("unet",48,3,9,1),("unet",24,3,9,1),
       ("resnet",128,4,9,1),("resnet",128,4,21,2),("dilated",32,3,5,4)]
LRS = [3e-4,1e-3,3e-3]

def enc(seqs,L):
    X=np.full((len(seqs),L),4,np.int64)
    for i,s in enumerate(seqs):
        a=MAPI[np.frombuffer(s[:L].encode(),np.uint8)]; X[i,:len(a)]=a
    return X

def kmer_feats(seqs,ks=(3,4,5)):
    out=[]
    for k in ks:
        V=4**k; X=np.zeros((len(seqs),V),np.float32); pw=(4**np.arange(k-1,-1,-1)).astype(np.int64)
        for r,s in enumerate(seqs):
            c=MAPI[np.frombuffer(s.encode(),np.uint8)]; ok=c!=4
            if len(c)<k: continue
            w=np.lib.stride_tricks.sliding_window_view(c,k); v=np.lib.stride_tricks.sliding_window_view(ok,k).all(1)
            idx=(w@pw)[v]
            if idx.size: np.add.at(X[r],idx,1.0); X[r]/=idx.size
        out.append(X)
    return np.hstack(out)

def train_cnn(tr,dv,te,ncls,L,dev,epochs):
    Itr=torch.from_numpy(enc(tr.text.tolist(),L)); ytr=torch.tensor(tr.label.values,dtype=torch.long)
    Idv=torch.from_numpy(enc(dv.text.tolist(),L)); Ite=torch.from_numpy(enc(te.text.tolist(),L))
    best=(-9,None)
    for arch,ch,nb,k,db in LAD:
        for lr in LRS:
            torch.manual_seed(0)
            m=build(arch,ncls,ch,nb,k=k,dbase=db).to(dev)
            opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-2)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
            bs=min(64,max(8,len(tr)//8)); bd=(-9,None); noimp=0
            for ep in range(epochs):
                perm=torch.randperm(len(Itr))
                for i in range(0,max(1,len(perm)-bs+1),bs):
                    b=perm[i:i+bs]
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        loss=F.cross_entropy(m(Itr[b].to(dev)),ytr[b].to(dev))
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
                sch.step()
                m.eval()
                with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                    pd_=torch.cat([m(Idv[i:i+256].to(dev)).float().argmax(-1).cpu() for i in range(0,len(Idv),256)]).numpy()
                m.train()
                s=matthews_corrcoef(dv.label.values,pd_)
                if s>bd[0]+1e-4:
                    m.eval()
                    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                        pt=torch.cat([m(Ite[i:i+256].to(dev)).float().argmax(-1).cpu() for i in range(0,len(Ite),256)]).numpy()
                    m.train(); bd=(s,matthews_corrcoef(te.label.values,pt)); noimp=0
                else:
                    noimp+=1
                    if noimp>=4: break
            if bd[0]>best[0]: best=(bd[0],bd[1])
            del m; torch.cuda.empty_cache()
    return best

def probe_fm(key,tr,dv,te,dev):
    """Frozen mean-pooled embedding -> logistic regression, C on dev.

    At small n nobody full-fine-tunes a 500M model -- it diverges (we measured MCC -0.076 at
    n=85). Giving the FM only the FT arm at small n would manufacture the decay we are testing
    for. So the FM gets best(probe, full-FT) at every n, exactly as the CNN gets best-over-ladder.
    """
    from transformers import AutoModel, AutoModelForMaskedLM
    mid,mx,bpt=FM[key]; need=int(max(len(x) for x in te.text)); ml=min(mx,max(16,need//bpt+4))
    tok=AutoTokenizer.from_pretrained(mid,trust_remote_code=True)
    if key=="dnabert2":
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        Cfg=get_class_from_dynamic_module("configuration_bert.BertConfig",mid)
        Mdl=get_class_from_dynamic_module("bert_layers.BertModel",mid)
        sys.modules[Mdl.__module__].flash_attn_qkvpacked_func=None
        m=Mdl.from_pretrained(mid,config=Cfg.from_pretrained(mid)).to(dev).eval()
    else:
        # NT-v2 registers an EsmConfig that AutoModel does not accept; AutoModelForMaskedLM does
        cls_ = AutoModelForMaskedLM if key=="nt_v2_500m" else AutoModel
        m=cls_.from_pretrained(mid,trust_remote_code=True).to(dev).eval()
    def emb(seqs):
        out=[]
        with torch.no_grad():
            for i in range(0,len(seqs),16):
                e=tok(list(seqs[i:i+16]),padding="longest",truncation=True,max_length=ml,return_tensors="pt")
                e.pop("token_type_ids",None); e={k:v.to(dev) for k,v in e.items()}
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    o=m(**e, output_hidden_states=True)
                hs=getattr(o,"hidden_states",None)
                h=hs[-1] if hs is not None else (o[0] if isinstance(o,(tuple,list))
                    else (o.last_hidden_state if hasattr(o,"last_hidden_state") else o))
                am=e.get("attention_mask")
                if am is None: p_=h.float().mean(1)
                else:
                    mm=am.unsqueeze(-1).float(); p_=(h.float()*mm).sum(1)/mm.sum(1).clamp(min=1)
                out.append(p_.cpu().numpy())
        return np.vstack(out)
    Et,Ed,Ee=(emb(x.text.tolist()) for x in (tr,dv,te))
    del m; torch.cuda.empty_cache()
    sc=StandardScaler().fit(Et); Et,Ed,Ee=sc.transform(Et),sc.transform(Ed),sc.transform(Ee)
    best=(-9,None)
    for C in (0.001,0.01,0.1,1.0,10.0):
        cl=LogisticRegression(C=C,max_iter=2000,class_weight="balanced").fit(Et,tr.label)
        s_=matthews_corrcoef(dv.label,cl.predict(Ed))
        if s_>best[0]: best=(s_,matthews_corrcoef(te.label,cl.predict(Ee)))
    return best


def train_fm(key,tr,dv,te,ncls,dev,epochs):
    mid,mx,bpt=FM[key]; need=int(max(len(s) for s in te.text)); ml=min(mx,max(16,need//bpt+4))
    tok=AutoTokenizer.from_pretrained(mid,trust_remote_code=True)
    def tk(s):
        e=tok(list(s),padding="longest",truncation=True,max_length=ml,return_tensors="pt")
        return e["input_ids"],e.get("attention_mask")
    Xtr,Mtr=tk(tr.text); Xdv,Mdv=tk(dv.text); Xte,Mte=tk(te.text)
    ytr=torch.tensor(tr.label.values,dtype=torch.long); best=(-9,None)
    for lr in (1e-5,3e-5,1e-4):
        torch.manual_seed(0)
        if key=="dnabert2":
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            Cfg=get_class_from_dynamic_module("configuration_bert.BertConfig",mid)
            Seq=get_class_from_dynamic_module("bert_layers.BertForSequenceClassification",mid)
            sys.modules[Seq.__module__].flash_attn_qkvpacked_func=None
            m=Seq.from_pretrained(mid,config=Cfg.from_pretrained(mid,num_labels=ncls)).to(dev)
        else:
            m=AutoModelForSequenceClassification.from_pretrained(mid,num_labels=ncls,trust_remote_code=True).to(dev)
        bs=min(16,max(4,len(tr)//8)); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
        tot=max(1,(len(Xtr)//bs))*epochs; sch=get_linear_schedule_with_warmup(opt,int(0.1*tot),tot)
        def ev(X,M):
            m.eval(); pr=[]
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                for i in range(0,len(X),32):
                    kw={"attention_mask":M[i:i+32].to(dev)} if M is not None else {}
                    pr.append(m(X[i:i+32].to(dev),**kw).logits.float().argmax(-1).cpu().numpy())
            m.train(); return np.concatenate(pr)
        bd=(-9,None); noimp=0
        for ep in range(epochs):
            perm=torch.randperm(len(Xtr))
            for i in range(0,max(1,len(perm)-bs+1),bs):
                b=perm[i:i+bs]
                kw={"attention_mask":Mtr[b].to(dev)} if Mtr is not None else {}
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    loss=m(Xtr[b].to(dev),**kw,labels=ytr[b].to(dev)).loss
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0)
                opt.step(); sch.step()
            s=matthews_corrcoef(dv.label.values,ev(Xdv,Mdv))
            if s>bd[0]+1e-4: bd=(s,matthews_corrcoef(te.label.values,ev(Xte,Mte))); noimp=0
            else:
                noimp+=1
                if noimp>=3: break
        if bd[0]>best[0]: best=(bd[0],bd[1])
        del m; torch.cuda.empty_cache()
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",required=True)
    ap.add_argument("--sizes",nargs="+",type=int,default=[100,300,1000,3000,10000,0])
    ap.add_argument("--models",nargs="+",default=["nt_v2_500m","dnabert2"])
    ap.add_argument("--seeds",nargs="+",type=int,default=[0,1])
    ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    tr_all=pd.read_csv(f"{GENEB}/{a.task}/train.csv"); te=pd.read_csv(f"{GENEB}/{a.task}/test.csv")
    ncls=int(pd.concat([tr_all,te]).label.nunique()); L=int(pd.concat([tr_all,te]).text.str.len().max())
    res=dict(task=a.task,n_classes=ncls,seqlen=L,n_test=len(te),n_train_full=len(tr_all),curve=[])
    print(f"{a.task}: cls={ncls} L={L} train_full={len(tr_all)} test={len(te)}",flush=True)
    for n in a.sizes:
        N=len(tr_all) if n==0 else min(n,len(tr_all))
        if N<60: continue
        ep = 40 if N<=1000 else (20 if N<=10000 else 8)
        per=[]
        for sd in a.seeds:
            sub=tr_all.sample(n=N,random_state=sd).reset_index(drop=True) if N<len(tr_all) else tr_all
            i1,i2=train_test_split(np.arange(len(sub)),test_size=0.15,stratify=sub.label.values,random_state=42)
            tr,dv=sub.iloc[i1].reset_index(drop=True),sub.iloc[i2].reset_index(drop=True)
            if dv.label.nunique()<2 or tr.label.nunique()<2: continue
            Xt,Xd,Xe=(kmer_feats(x.text.tolist()) for x in (tr,dv,te))
            sc=StandardScaler().fit(Xt); Xt,Xd,Xe=sc.transform(Xt),sc.transform(Xd),sc.transform(Xe)
            bk=(-9,None)
            for C in (0.001,0.01,0.1,1.0,10.0):
                cl=LogisticRegression(C=C,max_iter=2000,class_weight="balanced").fit(Xt,tr.label)
                s=matthews_corrcoef(dv.label,cl.predict(Xd))
                if s>bk[0]: bk=(s,matthews_corrcoef(te.label,cl.predict(Xe)))
            cd,ct=train_cnn(tr,dv,te,ncls,L,a.device,ep)
            row=dict(seed=sd,n_train=int(len(tr)),kmer=round(bk[1],4),cnn=round(ct,4))
            for m in a.models:
                pdv,pt=probe_fm(m,tr,dv,te,a.device)
                fdv,ft=train_fm(m,tr,dv,te,ncls,a.device,ep)
                # dev decides which regime the FM gets -- never test
                row[m]=round(ft if fdv>=pdv else pt,4)
                row[f"{m}_probe"]=round(pt,4); row[f"{m}_ft"]=round(ft,4)
                row[f"{m}_regime"]="ft" if fdv>=pdv else "probe"
            per.append(row)
            print(f"  n={N:<7} seed={sd}  kmer={row['kmer']:.4f} cnn={row['cnn']:.4f} "
                  +" ".join(f"{m}={row[m]:.4f}({row[f'{m}_regime']})" for m in a.models),flush=True)
        if not per: continue
        agg=dict(n_requested=N,n_train=int(np.mean([p['n_train'] for p in per])),
                 kmer=round(float(np.mean([p['kmer'] for p in per])),4),
                 cnn=round(float(np.mean([p['cnn'] for p in per])),4), runs=per)
        for m in a.models:
            agg[m]=round(float(np.mean([p[m] for p in per])),4)
            agg[f"{m}_probe"]=round(float(np.mean([p[f"{m}_probe"] for p in per])),4)
            agg[f"{m}_ft"]=round(float(np.mean([p[f"{m}_ft"] for p in per])),4)
            agg[f"{m}_regime"]=[p[f"{m}_regime"] for p in per]
        base=max(agg['kmer'],agg['cnn']); bf=max(agg[m] for m in a.models)
        agg['baseline']=round(base,4); agg['best_fm']=round(bf,4); agg['excess']=round(bf-base,4)
        res['curve'].append(agg)
        print(f"  => n={N:<7} baseline={base:.4f} bestFM={bf:.4f} EXCESS={bf-base:+.4f}",flush=True)
        json.dump(res,open(f"{OUT}/{a.task}.json","w"),indent=2)
    print(f"wrote {OUT}/{a.task}.json",flush=True)

if __name__=="__main__": main()
