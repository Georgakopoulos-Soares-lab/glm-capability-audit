"""P4: HViLM on the homology-strict HVUE split — a falsifiable test of OUR hypothesis.

HViLM-Tropism / -Patho are already fine-tuned on the HVUE tasks, so they are evaluated ZERO-SHOT
on our filtered test set. Our finding (homology filtering costs baselines 19-24% of above-chance
performance but gLMs only ~5%) predicts HViLM's advantage should SURVIVE OR GROW under clean
evaluation, not collapse. This tests that prediction rather than assuming the older one.

Loading mirrors DNABERT-2: HViLM ships the same four remote files, so AutoModel* raises a
config_class mismatch and the vendored triton kernel is incompatible with triton 3.x.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, transformers
import paths as P
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
transformers.logging.set_verbosity_error()
import warnings; warnings.filterwarnings("ignore")
OUT = P.sub("hvilm")
MODELS = {"tropism": "duttaprat/HViLM-Tropism", "patho": "duttaprat/HViLM-Patho"}

def load(mid, ncls):
    Cfg = get_class_from_dynamic_module("configuration_bert.BertConfig", mid)
    Seq = get_class_from_dynamic_module("bert_layers.BertForSequenceClassification", mid)
    sys.modules[Seq.__module__].flash_attn_qkvpacked_func = None
    return Seq.from_pretrained(mid, config=Cfg.from_pretrained(mid, num_labels=ncls)), \
           AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="tropism", choices=list(MODELS))
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    d = pd.read_parquet(a.split_file)
    te = d[d.partition == "val"].reset_index(drop=True)
    mid = MODELS[a.which]
    m, tok = load(mid, int(te.label.nunique()))
    m = m.to(a.device).eval()
    print(f"{a.which}: n_test={len(te)} pos_rate={te.label.mean():.3f} split={os.path.basename(a.split_file)}", flush=True)
    probs = []
    with torch.no_grad():
        for i in range(0, len(te), a.bs):
            e = tok(te.sequence.iloc[i:i+a.bs].tolist(), return_tensors="pt", padding=True,
                    truncation=True, max_length=a.maxlen)
            e.pop("token_type_ids", None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(**{k: v.to(a.device) for k, v in e.items()}).logits.float()
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    p = np.concatenate(probs); y = te.label.values
    # label polarity is not guaranteed to match ours -> report both and flag
    au = float(roc_auc_score(y, p)); au_flip = float(roc_auc_score(y, -p))
    flipped = au_flip > au
    res = dict(model=mid, which=a.which, split=os.path.basename(a.split_file), n_test=len(te),
               pos_rate=round(float(y.mean()), 4), auroc=round(max(au, au_flip), 4),
               auroc_raw=round(au, 4), label_polarity_flipped=flipped,
               mcc=round(float(matthews_corrcoef(y, (p > 0.5).astype(int))), 4),
               acc=round(float(accuracy_score(y, (p > 0.5).astype(int))), 4))
    print(f"  AUROC={res['auroc']:.4f} (raw {au:.4f}, flipped={flipped})  MCC={res['mcc']:.4f}", flush=True)
    tag = f"{a.which}__{os.path.basename(a.split_file).replace('.parquet','')}"
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
    np.savez_compressed(f"{OUT}/{tag}__preds.npz", y_true=y, p=p)
    print(f"  wrote {tag}.json", flush=True)

if __name__ == "__main__":
    main()
