"""Strong composition baselines for the bits/base (generative) comparison.

Why the current floor is too weak
---------------------------------
We report fixed-order bidirectional Markov at k in {3,5,7}, and k=7 comes out WORSE than k=5
(1.9829 vs 1.9782). That is a sparsity artefact, not a modelling result: order-7 needs 4^14
contexts against ~3M positions, so most contexts are unseen and add-one smoothing pushes them
toward 2 bits. A fixed-order table is therefore NOT the strongest composition baseline, and
using it flatters every neural model.

Lossless compression is the standard reference for this (compression and probabilistic
sequence modelling are dual: a model assigning probability p to a sequence encodes it in
-log2 p bits). The literature evaluates genomic LMs exactly this way -- general-purpose
compressors (gzip/xz/zstd/bzip2) as the easy floor, and specialised DNA compressors
(GeCo3, JARVIS3, XM) as the hard floor. Reported reference points: DNAGPT2 reaches 1.47 bpb on
T2T human, beating every general-purpose compressor but still trailing JARVIS3 at 1.384 bpb,
because specialised tools model long repeats with explicit copy/paste that transformers
emulate poorly.

What this adds
--------------
1. **Interpolated / backoff Markov** -- linear interpolation across orders 1..K with weights
   fitted on held-out data. Strictly stronger than any fixed order, and it cannot degrade the
   way k=7 did, because sparse high-order contexts back off to dense low-order ones.
2. **General-purpose compressors** -- xz, bzip2, zlib bits/base on the same sequences, 2-bit
   packed so the compressor sees DNA rather than ASCII (ASCII inflates everything ~4x and
   would make the baseline look absurdly weak).

Both are composition/repeat baselines with NO biology and NO training on our task. If a gLM
cannot beat them it has not demonstrated generative capability over composition.
"""
import argparse, bz2, json, lzma, os, sys, zlib
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths as P

MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def pack2bit(s):
    """2 bits/base packing so a compressor sees DNA, not 8-bit ASCII."""
    v = [MAP.get(c, 0) for c in s.upper()]
    out = bytearray()
    for i in range(0, len(v) - 3, 4):
        out.append((v[i] << 6) | (v[i+1] << 4) | (v[i+2] << 2) | v[i+3])
    return bytes(out)


def compressor_bits_per_base(seqs, name):
    """bits/base after compressing the concatenated 2-bit-packed sequences."""
    raw = b"".join(pack2bit(s) for s in seqs)
    nb = sum(len(s) for s in seqs)
    if name == "xz":
        c = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    elif name == "bzip2":
        c = bz2.compress(raw, 9)
    elif name == "zlib":
        c = zlib.compress(raw, 9)
    else:
        raise ValueError(name)
    return 8.0 * len(c) / nb


def interpolated_markov(fit, ev, K=8, held=0.1, seed=0):
    """Linear-interpolation backoff across orders 0..K, weights fitted by EM on held-out data.

    Strictly stronger than any single fixed order: sparse high-order contexts fall back to
    dense low-order ones instead of being smoothed to uniform.
    """
    rng = np.random.default_rng(seed)
    fit = list(fit); rng.shuffle(fit)
    nh = max(1, int(len(fit) * held))
    tune, train = fit[:nh], fit[nh:]

    cnt = [defaultdict(lambda: np.zeros(4)) for _ in range(K + 1)]
    for s in train:
        s = s.upper()
        idx = [MAP.get(c, -1) for c in s]
        for i, b in enumerate(idx):
            if b < 0: continue
            for k in range(K + 1):
                if i - k < 0: break
                ctx = tuple(idx[i-k:i])
                if any(x < 0 for x in ctx): break
                cnt[k][ctx][b] += 1

    def probs(idx, i):
        out = []
        for k in range(K + 1):
            if i - k < 0: out.append(np.full(4, 0.25)); continue
            ctx = tuple(idx[i-k:i])
            if any(x < 0 for x in ctx): out.append(np.full(4, 0.25)); continue
            v = cnt[k].get(ctx)
            out.append((v + 0.1) / (v.sum() + 0.4) if v is not None else np.full(4, 0.25))
        return np.array(out)                      # (K+1, 4)

    w = np.ones(K + 1) / (K + 1)
    for _ in range(12):                            # EM on the tuning split
        num = np.zeros(K + 1); den = 0.0
        for s in tune:
            idx = [MAP.get(c, -1) for c in s.upper()]
            for i, b in enumerate(idx):
                if b < 0: continue
                pk = probs(idx, i)[:, b]
                mix = float(w @ pk)
                if mix <= 0: continue
                num += w * pk / mix; den += 1
        if den: w = num / num.sum()

    tot, n = 0.0, 0
    for s in ev:
        idx = [MAP.get(c, -1) for c in s.upper()]
        for i, b in enumerate(idx):
            if b < 0: continue
            mix = float(w @ probs(idx, i)[:, b])
            tot += -np.log2(max(mix, 1e-12)); n += 1
    return tot / max(n, 1), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8000)
    ap.add_argument("--cap-family", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-fit", type=int, default=400)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import virobench_baselines as VBB
    OUT = P.sub("viral_likelihood"); os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    tr = VBB.load("ALL", "times", "train"); te = VBB.load("ALL", "times", "test")
    for d in (tr, te): d.drop(d.index[d.sequence.str.len() < a.window], inplace=True)
    bal = lambda d: d.groupby("family").head(a.cap_family).reset_index(drop=True)
    trb, teb = bal(tr), bal(te)
    perm = rng.permutation(len(trb)); nh = min(a.n_eval, len(trb) // 5)
    hold, fit = trb.iloc[perm[:nh]], trb.iloc[perm[nh:]]
    ood = teb.iloc[rng.permutation(len(teb))[:a.n_eval]]
    cut = lambda d: [s[:a.window].upper() for s in d.sequence.tolist()]
    fits, hs, os_ = cut(fit)[:a.n_fit], cut(hold), cut(ood)
    print(f"fit={len(fits)} in-dist={len(hs)} ood={len(os_)} window={a.window}", flush=True)

    res = dict(window=a.window, cap_family=a.cap_family, n_fit=len(fits),
               indist={}, ood={},
               note="compressors run on 2-bit-packed DNA so they are not penalised by ASCII")
    for nm in ("xz", "bzip2", "zlib"):
        res["indist"][nm] = round(compressor_bits_per_base(hs, nm), 4)
        res["ood"][nm] = round(compressor_bits_per_base(os_, nm), 4)
        print(f"  {nm:16s} in-dist {res['indist'][nm]:.4f} | OOD {res['ood'][nm]:.4f}", flush=True)
    bi, w = interpolated_markov(fits, hs, K=a.max_order)
    bo, _ = interpolated_markov(fits, os_, K=a.max_order)
    res["indist"][f"interp_markov_K{a.max_order}"] = round(float(bi), 4)
    res["ood"][f"interp_markov_K{a.max_order}"] = round(float(bo), 4)
    res["interp_weights"] = [round(float(x), 4) for x in w]
    print(f"  interp_markov_K{a.max_order:<2d}  in-dist {bi:.4f} | OOD {bo:.4f}")
    print(f"     order weights (0..K): {res['interp_weights']}")
    best_i = min(res["indist"].values()); best_o = min(res["ood"].values())
    res["strongest_baseline_indist"] = round(best_i, 4)
    res["strongest_baseline_ood"] = round(best_o, 4)
    print(f"\n  STRONGEST composition baseline: in-dist {best_i:.4f} | OOD {best_o:.4f}")
    print(f"  (fixed-order Markov k=5 was 1.9782 / 1.9807 -- anything below that was too weak)")
    p = a.out or f"{OUT}/generative_baselines_w{a.window}_cap{a.cap_family}.json"
    json.dump(res, open(p, "w"), indent=2); print(f"wrote {p}")


if __name__ == "__main__":
    main()
