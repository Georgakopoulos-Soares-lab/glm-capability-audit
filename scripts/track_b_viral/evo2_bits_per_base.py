"""Part 3 (Option A): per-nucleotide bits/base for Evo 2 via TurboQuant, vs Markov baselines.

Calibration verified before use: score(..., return_per_token=True) returns LOG-PROBABILITIES
(<=0), not NLL, so bits/base = -mean(logprob)/ln 2. Checked against known-entropy controls --
uniform random DNA -> 2.018 bits/base (theory 2.000), poly-A -> 0.029, ACGT-repeat -> 0.038.
Getting this sign wrong yields negative "bits", which is how it was caught.

Model choice: `evo2_7b` maps to the 1M-context config which REQUIRES FP8 input projections and
therefore Transformer Engine; our A100s are SM 8.0 with no FP8, so that variant cannot run
here. `evo2_7b_base` (8k config) runs without TE. int4 via TurboQuant puts it in 5.5 GB.

Test-set composition is recorded in the output (families, na_type, in/out-of-distribution
split) because "gLM is bad at viral" means different things depending on whether the sequences
are inside the model's training distribution -- Evo 2 excluded eukaryotic viruses by design.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths as P


def bits_per_base(model, tok, seqs, name, window):
    from turboquant import score
    tot_bits, tot_bases, sk = 0.0, 0, 0
    t0 = time.time()
    for i, s in enumerate(seqs):
        s = s[:window]
        try:
            r = score(model, tok, s, model_name=name, return_per_token=True)
        except Exception as e:
            sk += 1
            if sk <= 2: print(f"    skip {i}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        a = np.asarray(r.get("per_token") if isinstance(r, dict) else r, dtype=float).ravel()
        if a.size == 0: sk += 1; continue
        tot_bits += float(-a.sum() / np.log(2)); tot_bases += int(a.size)
        if i and i % 25 == 0:
            print(f"    {i}/{len(seqs)} running={tot_bits/max(tot_bases,1):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return tot_bits / max(tot_bases, 1), tot_bases, sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b_base")
    ap.add_argument("--tier", default="tier2")
    ap.add_argument("--window", type=int, default=8000)
    ap.add_argument("--cap-family", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-fit", type=int, default=3000)
    ap.add_argument("--orders", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import virobench_baselines as VBB
    from viral_likelihood import markov_bits_per_base
    OUT = P.sub("viral_likelihood"); os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    tr = VBB.load("ALL", "times", "train"); te = VBB.load("ALL", "times", "test")
    for d in (tr, te): d.drop(d.index[d.sequence.str.len() < a.window], inplace=True)
    # groupby().head() rather than groupby().apply(head): under pandas 3.x apply() excludes
    # the grouping column from the frame it passes, so `family` vanished downstream.
    bal = lambda d: d.groupby("family").head(a.cap_family).reset_index(drop=True)
    trb, teb = bal(tr), bal(te)
    perm = rng.permutation(len(trb)); nh = min(a.n_eval, len(trb) // 5)
    hold, fit = trb.iloc[perm[:nh]], trb.iloc[perm[nh:]]
    ood = teb.iloc[rng.permutation(len(teb))[:a.n_eval]]
    cut = lambda d: [s[:a.window] for s in d.sequence.tolist()]

    comp = dict(
        window_bp=a.window, cap_family=a.cap_family,
        indist=dict(n=len(hold), families=int(hold.family.nunique()),
                    na_type=hold.na_type.value_counts().to_dict() if "na_type" in hold else None,
                    source="ViroBench ALL/times TRAIN split, family-capped, held out from the Markov fit"),
        ood=dict(n=len(ood), families=int(ood.family.nunique()),
                 na_type=ood.na_type.value_counts().to_dict() if "na_type" in ood else None,
                 source="ViroBench ALL/times TEST split (temporal), family-capped"),
        markov_fit=dict(n=len(fit), families=int(fit.family.nunique())))
    print(json.dumps(comp, indent=1), flush=True)

    res = dict(model=a.model, tier=a.tier, test_set_composition=comp,
               convention="bits/base = -mean(logprob)/ln2; verified on random DNA -> 2.018",
               indist={}, ood={})
    fits, hs, os_ = cut(fit), cut(hold), cut(ood)
    for k in a.orders:
        bi, _ = markov_bits_per_base(fits, hs, k, cap=a.window)
        bo, _ = markov_bits_per_base(fits, os_, k, cap=a.window)
        res["indist"][f"markov_order{k}"] = round(float(bi), 4)
        res["ood"][f"markov_order{k}"] = round(float(bo), 4)
        print(f"  markov-{k}: in-dist {bi:.4f} | OOD {bo:.4f}", flush=True)

    from turboquant import load_evo2
    t0 = time.time()
    model, tok = load_evo2(a.model, tier=a.tier, device=a.device, verbose=False)
    print(f"  {a.model} loaded in {time.time()-t0:.0f}s", flush=True)
    for label, seqs in (("indist", hs), ("ood", os_)):
        b, nb, sk = bits_per_base(model, tok, seqs, a.model, a.window)
        res[label][a.model] = round(float(b), 4)
        res[f"{a.model}_{label}_bases"] = nb; res[f"{a.model}_{label}_skipped"] = sk
        print(f"  {a.model} {label}: {b:.4f} bits/base (bases {nb:,}, skipped {sk})", flush=True)
    for label in ("indist", "ood"):
        mk = min(v for k, v in res[label].items() if k.startswith("markov"))
        res[f"{a.model}_vs_markov_{label}"] = round(res[label][a.model] - mk, 4)
        print(f"  {a.model} vs best Markov ({label}): "
              f"{res[f'{a.model}_vs_markov_{label}']:+.4f}", flush=True)
    tag = a.tag or f"evo2_{a.model}_{a.tier}_w{a.window}"
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
    print(f"wrote {OUT}/{tag}.json")


if __name__ == "__main__":
    main()
