"""The modality gap: is the GENERATIVE capability also composition-recoverable?

Every result in this project is about discrimination. The biosecurity concern about genomic LMs
is generative -- assigning high likelihood to, or designing, viable pathogen sequences. If a
k-th-order Markov model trained on the same corpus matches a gLM's held-out likelihood on viral
genomes, then the generative capability is composition-recoverable too, and weight-space
unlearning has nothing model-specific to remove on that axis either.

Metric: bits per base on HELD-OUT test genomes (lower = better). Markov baseline is trained only
on the TRAIN split, so the comparison is like-for-like with the gLMs' pretraining being external.
Autoregressive models (Evo, HyenaDNA) give exact NLL; masked LMs are excluded -- pseudo-likelihood
is not comparable to a true autoregressive bits/base.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import paths as P
import virobench_baselines as VBB
import warnings; warnings.filterwarnings("ignore")
OUT = P.sub("viral_likelihood")
IDX = {c: i for i, c in enumerate("ACGT")}

def _codes(s, cap):
    a = np.frombuffer((s[:cap] if cap else s).upper().encode(), np.uint8)
    c = np.full(len(a), -1, np.int64)
    for ch, i in IDX.items(): c[a == ord(ch)] = i
    return c[c >= 0]


def fit_markov_tables(train_seqs, kmax, cap, alpha=0.1):
    """counts for every order 0..kmax, so we can INTERPOLATE rather than pick one order."""
    tabs=[]
    for k in range(kmax+1):
        V=4**k; tabs.append(np.full((V,4), alpha, np.float64))
    for s in train_seqs:
        c=_codes(s,cap)
        if len(c)<kmax+1: continue
        for k in range(kmax+1):
            if k==0: np.add.at(tabs[0], (np.zeros(len(c),np.int64), c), 1.0); continue
            if len(c) <= k: continue
            ctx=np.lib.stride_tricks.sliding_window_view(c[:-1], k) @ (4**np.arange(k-1,-1,-1))
            np.add.at(tabs[k], (ctx, c[k:]), 1.0)
    return [t/t.sum(1,keepdims=True) for t in tabs]


def interp_bits_per_base(probs, seqs, kmax, cap, weights):
    """linear interpolation across orders 0..kmax -- the standard fix for high-order sparsity.

    Fitting only add-alpha at a single order is a WEAK baseline: on this data order-7 scored
    2.42 bits/base, worse than order-3's 2.03, i.e. it was overfitting 16k contexts. Interpolating
    with dev-tuned weights is the fair composition baseline.
    """
    tot=0.0; n=0
    for s in seqs:
        c=_codes(s,cap)
        if len(c)<kmax+1: continue
        tgt=c[kmax:]
        mix=np.zeros((len(tgt),4))
        for k in range(kmax+1):
            if k==0:
                mix += weights[0]*probs[0][0][None,:]
            else:
                ctx=np.lib.stride_tricks.sliding_window_view(c[kmax-k:-1], k) @ (4**np.arange(k-1,-1,-1))
                assert len(ctx)==len(tgt), f"align k={k}: {len(ctx)} vs {len(tgt)}"
                mix += weights[k]*probs[k][ctx]
        pr=np.clip(mix[np.arange(len(tgt)), tgt], 1e-12, None)
        tot += -np.log2(pr).sum(); n += len(tgt)
    return tot/max(n,1), n


def markov_bits_per_base(train_seqs, test_seqs, k, cap=None, alpha=0.1):
    """single-order (kept for the ablation table)"""
    probs=fit_markov_tables(train_seqs, k, cap, alpha)
    w=np.zeros(k+1); w[k]=1.0
    return interp_bits_per_base(probs, test_seqs, k, cap, w)


@torch.no_grad()
def glm_bits_per_base(key, test_seqs, device, maxlen, bs=1):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = {"hyenadna": "LongSafari/hyenadna-medium-160k-seqlen-hf",
           "evo": "togethercomputer/evo-1-8k-base"}[key]
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if key == "evo" else torch.float32).to(device).eval()
    if key == "evo":
        torch.cuda.empty_cache()
    tot_bits = 0.0; tot_n = 0; t0 = time.time()
    for i, s in enumerate(test_seqs):
        ids = tok(s[:maxlen], return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 8: continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(ids).logits.float()
        lp = torch.log_softmax(lg[:, :-1], -1)
        nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum().item()
        tot_bits += nll / np.log(2); tot_n += ids.shape[1] - 1
        if i and i % 50 == 0: print(f"    {key} {i}/{len(test_seqs)} ({time.time()-t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
    return tot_bits / max(tot_n, 1), tot_n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["hyenadna"])
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--maxlen", type=int, default=8000)
    ap.add_argument("--orders", nargs="+", type=int, default=[3,4,5,6,7,8])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tr, dv, te = (VBB.load("ALL", "times", x) for x in ("train", "val", "test"))
    rng = np.random.default_rng(0)
    tes = te.sequence.iloc[rng.permutation(len(te))[:a.n_test]].tolist()
    tes = [s[:a.maxlen] for s in tes if isinstance(s, str) and len(s) > 200]
    trs = tr.sequence.tolist()
    print(f"train={len(trs)} genomes | test={len(tes)} genomes truncated to {a.maxlen}bp", flush=True)
    res = dict(n_train=len(trs), n_test=len(tes), maxlen=a.maxlen, markov={}, glm={})
    for k in a.orders:
        t0 = time.time(); bpb, n = markov_bits_per_base(trs, tes, k, cap=a.maxlen)
        res["markov"][f"order{k}"] = round(float(bpb), 4)
        print(f"  markov order-{k}: {bpb:.4f} bits/base  (n={n}, {time.time()-t0:.0f}s)", flush=True)
    # INTERPOLATED baseline: weights chosen on the official val split, never on test
    kmax = max(a.orders)
    dvs = [s[:a.maxlen] for s in dv.sequence.tolist()[:200] if isinstance(s, str) and len(s) > 200]
    probs = fit_markov_tables(trs, kmax, a.maxlen)
    best = (9e9, None)
    for tilt in (0.0, 0.5, 1.0, 2.0, 4.0):
        w = np.array([(i + 1) ** tilt for i in range(kmax + 1)], float); w /= w.sum()
        b, _ = interp_bits_per_base(probs, dvs, kmax, a.maxlen, w)
        if b < best[0]: best = (b, w)
    bpb, n = interp_bits_per_base(probs, tes, kmax, a.maxlen, best[1])
    res["markov"]["interpolated"] = round(float(bpb), 4)
    res["markov"]["interp_weights"] = [round(float(x), 4) for x in best[1]]
    print(f"  markov INTERPOLATED(0..{kmax}): {bpb:.4f} bits/base  (dev {best[0]:.4f})", flush=True)
    # in-distribution reference: the official val split resembles train, test does not
    res["dev_reference"] = {}
    kmaxd = max(a.orders)
    for k in a.orders:
        b, _ = markov_bits_per_base(trs, dvs, k, cap=a.maxlen)
        res["dev_reference"][f"markov_order{k}"] = round(float(b), 4)
    print(f"  [dev, in-distribution] markov: {res['dev_reference']}", flush=True)
    for key in a.models:
        try:
            bd, _ = glm_bits_per_base(key, dvs, a.device, a.maxlen)
            res["dev_reference"][key] = round(float(bd), 4)
            print(f"  [dev, in-distribution] {key}: {bd:.4f} bits/base", flush=True)
        except Exception as e:
            print(f"  [dev] {key} FAILED {type(e).__name__}", flush=True)
    for key in a.models:
        try:
            bpb, n = glm_bits_per_base(key, tes, a.device, a.maxlen)
            res["glm"][key] = round(float(bpb), 4)
            print(f"  {key}: {bpb:.4f} bits/base (n={n})", flush=True)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            res["glm"][key] = None
    best_m = min(v for k, v in res["markov"].items() if isinstance(v, (int, float)))
    res["best_markov"] = best_m
    for key, v in res["glm"].items():
        if v is not None:
            res[f"{key}_vs_markov"] = round(v - best_m, 4)
            print(f"\n  {key} vs best Markov: {v-best_m:+.4f} bits/base "
                  f"({'gLM better' if v < best_m else 'MARKOV BETTER'})", flush=True)
    tag = "virobench_times__" + "_".join(sorted(a.models))
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
    print(f"wrote {OUT}/{tag}.json", flush=True)

if __name__ == "__main__":
    main()
