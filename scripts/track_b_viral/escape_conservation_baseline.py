"""Measurement, not argument: does a conservation/MSA baseline clear the escape-task bar?

The claim to test is that a reported AUROC ~0.6 on viral escape is conservation-recoverable.
ProteinGym ships per-variant scores for the published MSA/conservation family (Site_Independent,
EVmutation, DeepSequence, EVE, MSA_Transformer) alongside pLM scores, so this needs no MSA of our
own -- it uses the field's own baselines on the field's own data.

Reported per assay:
  - Spearman and AUROC (on DMS_score_bin) for each conservation/MSA scorer
  - the same for the ESM family
  - our SUPERVISED numbers from proteingym_supervised.py for context
A conservation scorer at/above ~0.6 AUROC means the escape signal is conservation-recoverable
and a positive result there is the unvalidated kind our audit predicts.
"""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import paths as P
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")
Z = os.environ.get("VB_PGYM_ZS", "/data/nvidia/proteingym/zeroshot")
OUT = P.sub("escape_conservation")
CONS = ["Site_Independent","EVmutation","DeepSequence_ensemble","EVE_ensemble","MSA_Transformer_ensemble"]
PLM  = ["ESM1b","ESM1v_ensemble","ESM2_650M","ESM2_3B","ESM2_15B"]
VIRAL = ("ZIKV","9HIV1","9INFA","SARS2","POLG","CAPSD","NRAM_I33A0","A4D664","C6KNH7","Q2N0S5")

def main():
    os.makedirs(OUT, exist_ok=True)
    files=[f for f in sorted(glob.glob(f"{Z}/*.csv")) if any(k in os.path.basename(f) for k in VIRAL)]
    rows=[]
    for f in files:
        a=os.path.basename(f)[:-4]
        d=pd.read_csv(f)
        if "DMS_score" not in d: continue
        y=d.DMS_score.values
        ybin=d.DMS_score_bin.values if "DMS_score_bin" in d else None
        r=dict(assay=a, n=len(d))
        for col in CONS+PLM:
            if col not in d: continue
            s=d[col].values
            ok=~pd.isna(s)
            if ok.sum()<50: continue
            r[f"{col}__rho"]=round(float(spearmanr(s[ok],y[ok]).correlation),4)
            if ybin is not None and len(set(ybin[ok]))>1:
                au=roc_auc_score(ybin[ok],s[ok])
                r[f"{col}__auroc"]=round(float(max(au,1-au)),4)   # sign-invariant
        rows.append(r)
    df=pd.DataFrame(rows)
    def bestof(cols, suffix):
        sub=[c+suffix for c in cols if c+suffix in df.columns]
        return df[sub].max(axis=1) if sub else pd.Series([np.nan]*len(df))
    df["best_cons_auroc"]=bestof(CONS,"__auroc"); df["best_plm_auroc"]=bestof(PLM,"__auroc")
    df["best_cons_rho"]=bestof(CONS,"__rho");    df["best_plm_rho"]=bestof(PLM,"__rho")
    df["plm_minus_cons_auroc"]=(df.best_plm_auroc-df.best_cons_auroc).round(4)
    pd.set_option("display.width",200)
    print("VIRAL ProteinGym assays — published conservation/MSA vs pLM zero-shot scores\n")
    print(df[["assay","n","best_cons_auroc","best_plm_auroc","plm_minus_cons_auroc","best_cons_rho","best_plm_rho"]]
          .to_string(index=False))
    n=df.best_cons_auroc.notna().sum()
    print(f"\n  conservation/MSA AUROC >= 0.60 on {int((df.best_cons_auroc>=0.60).sum())}/{n} viral assays")
    print(f"  conservation/MSA AUROC >= 0.70 on {int((df.best_cons_auroc>=0.70).sum())}/{n}")
    print(f"  median best-conservation AUROC {df.best_cons_auroc.median():.4f} | median best-pLM {df.best_plm_auroc.median():.4f}")
    print(f"  pLM beats conservation on {int((df.plm_minus_cons_auroc>0.01).sum())}/{n} assays "
          f"(mean margin {df.plm_minus_cons_auroc.mean():+.4f})")
    df.to_csv(f"{OUT}/viral_conservation_vs_plm.csv", index=False)
    json.dump(dict(n_assays=int(n),
                   frac_cons_ge_060=round(float((df.best_cons_auroc>=0.60).mean()),4),
                   median_cons_auroc=round(float(df.best_cons_auroc.median()),4),
                   median_plm_auroc=round(float(df.best_plm_auroc.median()),4),
                   plm_wins=int((df.plm_minus_cons_auroc>0.01).sum())),
              open(f"{OUT}/summary.json","w"), indent=2)
    print(f"\nwrote {OUT}/viral_conservation_vs_plm.csv")

if __name__=="__main__": main()
