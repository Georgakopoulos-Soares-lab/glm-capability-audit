"""Frozen-embedding linear probe, applied IDENTICALLY to every model including Evo 2.

Why a probe and not full fine-tuning for Evo 2
----------------------------------------------
Evo 2 7B cannot be full-fine-tuned under our protocol on this hardware: bf16 weights + bf16
grads + fp32 Adam states is 84 GB, over an 80 GB A100 before a single activation. Making it fit
needs 8-bit Adam plus gradient checkpointing -- a DIFFERENT optimisation regime from the one
every other model received, which is exactly the asymmetry that invalidated the first LucaVirus
run. Its raw forward path also breaks against this flash-attn version
(`FlashSelfAttention.forward() got an unexpected keyword argument 'key_padding_mask'`).

A frozen probe sidesteps both and, crucially, can be applied UNIFORMLY: every model is reduced
to embeddings by the same pooling, and the same logistic head is fitted with the same C sweep.
Probe-vs-probe is internally fair. It is reported as the SECONDARY regime, never mixed with
full-FT numbers, and the probe-vs-FT gap is itself a diagnostic.

Pooling: masked mean over the final hidden states, for every model, so no model gets a learned
pooler advantage the others lack.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
# moved out of track_b_viral/ when this stopped feeding a reported
# number; the siblings it imports still live there.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "track_b_viral"))
import paths as P

HF = {  # model -> (hf id or local path, loader, max tokens)
 "hvilm":      ("duttaprat/HViLM-base",                     "dnabert2", 512),
 "dnabert2":   ("zhihan1996/DNABERT-2-117M",                "dnabert2", 512),
 # GENA-LM's auto_map points AutoModel at its own BertForMaskedLM, so the returned object is
 # a MaskedLMOutput with logits and no last_hidden_state -- request hidden states explicitly.
 # ... but `trust_remote_code=True` alone still resolved to STOCK BertForMaskedLM (post-LN).
 # GENA-LM's checkpoint is pre-LN: it ships pre_attention_ln/post_attention_ln, which stock
 # BERT discards (48 tensors) while freshly initialising 48 post-LN LayerNorms it does not
 # have. Half the normalisation was random and the probe read MCC ~0.01-0.05 on every task.
 # Load the custom class explicitly, exactly as viral_pseudo_ppl.py already does.
 "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t",     "dyn_mlm_hidden", 512,
                "modeling_bert.BertForMaskedLM"),
 # NT-v2's config is not registered for AutoModel; AutoModelForMaskedLM is the supported path.
 "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "mlm_hidden", 1024),
 "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf","automodel", 8192),
 "lucavirus":  (os.environ.get("LUCAVIRUS_DIR",
                "/data/nvidia/models/lucavirus_step38M/207da0192381a3cedcb1cf82a9850239d5fdd9d5"),
                "automodel", 4096),
}
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _no_fresh_weights(mid, fn):
    """Run `fn` and hard-fail if transformers reports freshly-initialised weights.

    Ported from viral_pseudo_ppl.py. A half-random encoder still returns finite embeddings,
    so without this the failure mode is a plausible-looking near-zero probe score rather than
    a crash -- which is exactly how GENA-LM's four probe results got recorded as real.
    """
    import io, logging, transformers
    buf = io.StringIO(); h = logging.StreamHandler(buf)
    lg = logging.getLogger("transformers.modeling_utils"); prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.WARNING)
    transformers.logging.set_verbosity_warning()
    try:
        m = fn()
    finally:
        lg.removeHandler(h); lg.setLevel(prev); transformers.logging.set_verbosity_error()
    txt = buf.getvalue()
    if "newly initialized" in txt:
        bad = [l for l in txt.splitlines() if "newly initialized" in l]
        raise RuntimeError(f"{mid}: encoder has freshly-initialised weights -> the frozen "
                           f"representation is partly random and the probe is meaningless. "
                           f"{bad[0][:300]}")
    return m


def embed_hf(name, seqs, device, bs=16):
    import torch
    from transformers import AutoTokenizer, AutoModel
    _spec = HF[name]
    mid, how, maxtok = _spec[0], _spec[1], _spec[2]
    dyn = _spec[3] if len(_spec) > 3 else None
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if how == "dnabert2":
        import sys as _s
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        Cfg = get_class_from_dynamic_module("configuration_bert.BertConfig", mid)
        Body = get_class_from_dynamic_module("bert_layers.BertModel", mid)
        _s.modules[Body.__module__].flash_attn_qkvpacked_func = None
        m = Body(Cfg.from_pretrained(mid))
        from huggingface_hub import snapshot_download
        import glob as _g
        p = snapshot_download(mid)
        w = _g.glob(f"{p}/pytorch_model.bin") or _g.glob(f"{p}/model.safetensors")
        if w:
            if w[0].endswith("safetensors"):
                from safetensors.torch import load_file as lf; st = lf(w[0])
            else:
                st = torch.load(w[0], map_location="cpu")
            m.load_state_dict({k.replace("bert.", "", 1): v for k, v in st.items()}, strict=False)
    elif how == "dyn_mlm_hidden":
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        MlmCls = get_class_from_dynamic_module(dyn, mid)
        m = _no_fresh_weights(mid, lambda: MlmCls.from_pretrained(mid,
                                                                  output_hidden_states=True))
    elif how == "mlm_hidden":
        from transformers import AutoModelForMaskedLM
        m = _no_fresh_weights(mid, lambda: AutoModelForMaskedLM.from_pretrained(
            mid, trust_remote_code=True, output_hidden_states=True))
    else:
        m = AutoModel.from_pretrained(mid, trust_remote_code=True)
    m = m.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            e = tok(list(seqs[i:i+bs]), padding="longest", truncation=True,
                    max_length=maxtok, return_tensors="pt")
            ids = e["input_ids"].to(device)
            am = e.get("attention_mask")
            am = am.to(device) if am is not None else None
            kw = dict(attention_mask=am) if am is not None else {}
            if how in ("mlm_hidden", "dyn_mlm_hidden"): kw["output_hidden_states"] = True
            o = m(ids, **kw)
            if isinstance(o, (tuple, list)):
                h = o[0]
            elif getattr(o, "hidden_states", None) is not None:
                h = o.hidden_states[-1]          # final layer, not the LM logits
            else:
                h = o.last_hidden_state
            if am is None:
                v = h.mean(1)
            else:
                mm = am.unsqueeze(-1).to(h.dtype)
                v = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            out.append(v.float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.vstack(out)


def embed_evo2(seqs, device, model_name="evo2_7b_base", tier="tier2", layer_frac=0.5):
    """Evo 2 embeddings via forward hooks -- the SAME representation the other models get.

    Native Evo 2 exposes `Evo2.forward(..., return_embeddings=True, layer_names=[...])`, which
    is implemented as forward hooks on named modules. TurboQuant's tier2 deliberately returns
    the raw `StripedHyena` rather than the `Evo2` wrapper (the wrapper's constructor downloads
    the ~80 GB bf16 checkpoint before load_checkpoint, which tier2 exists to avoid). So the
    convenience API is absent, but the mechanism is not: we register the same hook ourselves.

    This yields mean-pooled hidden states, directly comparable to every other model in the
    suite. If hooking fails the caller falls back to logprob summary statistics, which are a
    WEAKER representation and are labelled as such.
    """
    import torch
    from turboquant import load_evo2
    m, tok = load_evo2(model_name, tier=tier, device=device, verbose=False)
    blocks = getattr(m, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("no .blocks on the returned model; cannot hook hidden states")
    idx = max(0, min(len(blocks) - 1, int(len(blocks) * layer_frac)))
    cap = {}

    def hook(_, __, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    # The RAW forward `m(ids)` is broken against this flash-attn build
    # (`FlashSelfAttention.forward() got an unexpected keyword argument 'key_padding_mask'`),
    # but turboquant's score() routes around it. So drive the forward with score() and let the
    # hook capture the hidden state as a side effect. Verified: (1, 1000, 4096), finite.
    from turboquant import score as _score
    hd = blocks[idx].register_forward_hook(hook)
    out = []
    try:
        with torch.no_grad():
            for s in seqs:
                cap.clear()
                _ = _score(m, tok, s, model_name=model_name, return_per_token=True)
                h = cap.get("h")
                if h is None:
                    raise RuntimeError("hook produced no hidden state")
                out.append(h.float().mean(1).cpu().numpy())
    finally:
        hd.remove(); del m; torch.cuda.empty_cache()
    return np.vstack(out), f"mean-pooled hidden states, block {idx}/{len(blocks)} (hook)"


def embed_evo2_logprob_fallback(seqs, device, model_name="evo2_7b_base", tier="tier2"):
    """Fallback only: summary stats of the per-token logprob profile. WEAKER than hidden-state
    pooling and not comparable to the other models -- recorded as such in the output."""
    import torch
    from turboquant import load_evo2, score
    m, tok = load_evo2(model_name, tier=tier, device=device, verbose=False)
    feats = []
    for s in seqs:
        r = score(m, tok, s, model_name=model_name, return_per_token=True)
        a = np.asarray(r.get("per_token") if isinstance(r, dict) else r, dtype=float).ravel()
        q = np.percentile(a, [5, 25, 50, 75, 95]) if a.size else np.zeros(5)
        feats.append(np.concatenate([[a.mean(), a.std(), a.min(), a.max(), a.size], q]))
    del m; torch.cuda.empty_cache()
    return np.vstack(feats), "per-token logprob summary stats (10-d) -- WEAKER, not comparable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--train-csv", required=True); ap.add_argument("--test-csv", required=True)
    ap.add_argument("--dev-csv", default="")
    ap.add_argument("--cap-train", type=int, default=8000)
    ap.add_argument("--cap-test", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/home/nvidia/genomic-capability-unlearning/reports/frozen_probe")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rd = lambda p: pd.read_csv(p).rename(columns={"text": "sequence"}).reset_index(drop=True)
    tr, te = rd(a.train_csv), rd(a.test_csv)
    rng = np.random.default_rng(0)
    if a.cap_train and len(tr) > a.cap_train:
        tr = tr.groupby("label", group_keys=False).sample(
            frac=min(1.0, a.cap_train/len(tr)), random_state=0).reset_index(drop=True)
    if a.cap_test and len(te) > a.cap_test:
        te = te.sample(n=a.cap_test, random_state=0).reset_index(drop=True)
    print(f"{a.model} / {a.task}: train={len(tr):,} test={len(te):,}", flush=True)
    t0 = time.time()
    if a.model == "evo2":
        try:
            Xtr, rep = embed_evo2(tr.sequence.astype(str).tolist(), a.device)
            Xte, _ = embed_evo2(te.sequence.astype(str).tolist(), a.device)
        except Exception as e:
            print(f"  hidden-state hook failed ({type(e).__name__}: {str(e)[:120]}); "
                  f"falling back to logprob summary -- NOT comparable to other models",
                  flush=True)
            Xtr, rep = embed_evo2_logprob_fallback(tr.sequence.astype(str).tolist(), a.device)
            Xte, _ = embed_evo2_logprob_fallback(te.sequence.astype(str).tolist(), a.device)
    else:
        Xtr = embed_hf(a.model, tr.sequence.astype(str).tolist(), a.device)
        Xte = embed_hf(a.model, te.sequence.astype(str).tolist(), a.device)
        rep = "masked mean over final hidden states"
    print(f"  embedded in {time.time()-t0:.0f}s  dim={Xtr.shape[1]}", flush=True)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    best = (-np.inf, None, None)
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                 n_jobs=-1).fit(Xtr_s, ytr)
        p = clf.predict(Xte_s); m = matthews_corrcoef(yte, p)
        if m > best[0]: best = (m, C, clf)
    mcc, C, clf = best
    pr = clf.predict(Xte_s)
    res = dict(model=a.model, task=a.task, regime="frozen_linear_probe",
               representation=rep, n_train=len(tr), n_test=len(te), dim=int(Xtr.shape[1]),
               selected_C=C, mcc=round(float(mcc), 4),
               macro_f1=round(float(f1_score(yte, pr, average="macro", zero_division=0)), 4))
    if len(np.unique(yte)) == 2:
        res["auprc"] = round(float(average_precision_score(
            yte, clf.predict_proba(Xte_s)[:, 1])), 4)
    print(f"  PROBE mcc={mcc:.4f} C={C:g} " +
          (f"auprc={res['auprc']:.4f}" if "auprc" in res else ""), flush=True)
    p = f"{a.out}/{a.task}__{a.model}__probe.json"
    json.dump(res, open(p, "w"), indent=2); print(f"  wrote {p}")


if __name__ == "__main__":
    main()
