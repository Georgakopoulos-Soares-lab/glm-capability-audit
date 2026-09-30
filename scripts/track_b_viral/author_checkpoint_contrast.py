"""The authors' own task-fine-tuned HViLM checkpoints, as a labelled secondary contrast.

Our primary comparison fine-tunes HViLM-BASE under the same protocol as every
other model, which is the only way the numbers are comparable. But it invites
one objection: maybe the uniform protocol under-trains HViLM, and the authors'
released checkpoints would clear the floor.

This tests that directly. `duttaprat/HViLM-{Tropism,Patho,R0}` are the authors'
released task models. We run them as-is over our test split -- no fine-tuning,
no selection -- and compare against the floor those tasks already have.

Two honest caveats, reported rather than resolved:

1. LABEL ORIENTATION. The checkpoints ship an id2label, but nothing guarantees
   their positive class matches ours. We report MCC under the shipped mapping
   AND under the flip, and mark which was used. A flip is a labelling
   convention, not a performance claim, so taking the better of the two is
   legitimate here -- but it must be visible.

2. TRAIN/TEST PROVENANCE. These were fine-tuned on the authors' own splits,
   which we cannot audit. Our test rows may well be in their training data. So
   a high score here is an UPPER bound that cannot be cleaned, and a low score
   is the informative direction.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import matthews_corrcoef, average_precision_score

R = os.environ.get("VB_VIRAL_OUT",
                   "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
PRED = f"{R}/preds_rescore"


def load_model(mid, device):
    """HViLM is the DNABERT-2 family: the remote bert_layers class, flash-attn stubbed."""
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    Cfg = get_class_from_dynamic_module("configuration_bert.BertConfig", mid)
    Cls = get_class_from_dynamic_module("bert_layers.BertForSequenceClassification", mid)
    sys.modules[Cls.__module__].flash_attn_qkvpacked_func = None
    m = Cls.from_pretrained(mid, config=Cfg.from_pretrained(mid))
    return tok, m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--esc", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.environ.get(
        "VB_REPORTS", "/home/nvidia/genomic-capability-unlearning/reports") + "/author_checkpoints")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    te = pd.read_csv(a.test_csv).rename(columns={"text": "sequence"})
    y = te.label.values.astype(int)
    tok, m = load_model(a.checkpoint, a.device)
    print(f"{a.name}: {a.checkpoint}  n_test={len(te):,}", flush=True)

    probs = []
    seqs = te.sequence.astype(str).tolist()
    with torch.no_grad():
        for i in range(0, len(seqs), a.bs):
            e = tok(seqs[i:i + a.bs], padding="longest", truncation=True,
                    max_length=a.max_len, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(e["input_ids"].to(a.device),
                       attention_mask=e["attention_mask"].to(a.device)).logits
            probs.append(torch.softmax(lg.float(), -1).cpu().numpy())
    P = np.concatenate(probs)
    pred = P.argmax(1)

    mcc_as_is = matthews_corrcoef(y, pred)
    mcc_flip = matthews_corrcoef(y, 1 - pred)
    flipped = mcc_flip > mcc_as_is
    use, prob = (1 - pred, P[:, 0]) if flipped else (pred, P[:, 1])
    mcc = max(mcc_as_is, mcc_flip)
    aup = average_precision_score(y, prob) if len(np.unique(y)) == 2 else float("nan")

    # the floor this task already has
    fl = {}
    for tag, pat in (("kmer", f"{PRED}/hvue_v2__{a.esc}__kmer__s*__preds.npz"),
                     ("kmer_gbt", f"{PRED}/hvue_v2__{a.esc}__kmergbt__s*__preds.npz"),
                     ("aa_gbt", f"{PRED}/hvue_v2__{a.esc}__aagbt__s*__preds.npz"),
                     ("tax_prior", f"{PRED}/hvue_v2__{a.esc}__taxprior__s*__preds.npz"),
                     ("nn_lookup", f"{PRED}/hvue_v2__*{a.esc}*__nnlookup__s*__preds.npz")):
        f = sorted(glob.glob(pat))
        if f:
            z = np.load(f[0])
            if len(z["y_true"]) == len(y):
                fl[tag] = float(matthews_corrcoef(z["y_true"], z["y_pred"]))
    floor = max(fl.values()) if fl else float("nan")
    farm = max(fl, key=fl.get) if fl else "-"

    print(f"  MCC as-shipped={mcc_as_is:.4f}  flipped={mcc_flip:.4f}  "
          f"-> using {'FLIPPED' if flipped else 'as-shipped'} = {mcc:.4f}")
    print(f"  AUPRC={aup:.4f}")
    print(f"  composition/lookup floor = {floor:.4f} ({farm})   excess {mcc - floor:+.4f}")
    print(f"  NOTE: the authors' training split is unauditable, so our test rows may be in "
          f"it. This number is an UPPER bound.")

    rec = dict(name=a.name, checkpoint=a.checkpoint, n_test=int(len(y)),
               mcc_as_shipped=round(mcc_as_is, 4), mcc_flipped=round(mcc_flip, 4),
               label_orientation=("flipped" if flipped else "as-shipped"),
               mcc=round(mcc, 4), auprc=round(float(aup), 4),
               floor=round(floor, 4), floor_arm=farm, excess=round(mcc - floor, 4),
               caveat="authors' training split unauditable; treat as an upper bound")
    p = f"{a.out}/{a.name}__author_ckpt.json"
    json.dump(rec, open(p, "w"), indent=2)
    np.savez_compressed(f"{PRED}/hvue_v2__{a.esc}__authorckpt__s0__preds.npz",
                        y_true=y, y_pred=use, y_prob=prob)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
