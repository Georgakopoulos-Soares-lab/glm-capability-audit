"""Is Evo 2's viral bits/base advantage real capability, or exposure to its training data?

The question
------------
Evo 2's pretraining corpus (OpenGenome2) EXCLUDED viruses of eukaryotic hosts as a deliberate
biosecurity measure, but INCLUDED prokaryotic viruses (bacteriophage, archaeal viruses). Our
bits/base pool is a family-capped mix of both. So a headline "-0.264 bits/base vs composition"
could mean either:

  (a) genuine generative capability on viral sequence, or
  (b) in-distribution recall of the phage it was trained on, diluted by the eukaryotic viruses
      it was never shown.

These are distinguishable: score the SAME model on prokaryotic-host and eukaryotic-host strata
separately. If the advantage is concentrated on prokaryotic-host viruses, it is exposure. If it
holds on eukaryotic-host viruses -- which Evo 2 was deliberately never trained on -- it is
capability, and the more interesting (and more biosecurity-relevant) result.

This is the same discipline applied to HViLM: never accept an advantage without asking what
the model has already seen.

HyenaDNA is scored on the identical strata as a control: it has no such exclusion policy, so
if the prokaryote/eukaryote gap is an artefact of the strata themselves (e.g. phage genomes
being intrinsically more repetitive) HyenaDNA and the Markov floor will show it too.
"""
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths as P

# host-name -> prokaryote. Curated rather than inferred: the ViroBench `host` field is free
# text, so a genus whitelist is more reliable than any parsing heuristic.
PROK = re.compile(
    r"escherichia|pseudomonas|salmonella|staphylococc|streptococc|streptomyc|bacillus|"
    r"mycobacteri|vibrio|lactobacill|lactococc|clostridi|enterococc|klebsiella|acinetobacter|"
    r"burkholderia|xanthomonas|erwinia|ralstonia|rhizobium|agrobacterium|caulobacter|"
    r"synechococc|prochlorococc|cyanobact|bacteroides|helicobacter|campylobacter|listeria|"
    r"yersinia|shigella|proteus|serratia|citrobacter|enterobacter|aeromonas|flavobacteri|"
    r"arthrobacter|corynebacteri|rhodococcus|gordonia|microbacteri|paenibacillus|"
    r"sulfolobus|haloarcula|halobacteri|haloferax|methano|thermococc|thermoprote|pyrococc|"
    r"archaeo|sacchar\w*bacter|nitrosopumilus|acidianus|\bphage\b|bacterium|archaea", re.I)


def host_class(h):
    if not isinstance(h, str) or not h.strip():
        return "unknown"
    return "prokaryotic" if PROK.search(h) else "eukaryotic"


def bits_per_base_evo2(model, tok, seqs, name, window):
    from turboquant import score
    per = []
    for s in seqs:
        try:
            r = score(model, tok, s[:window], model_name=name, return_per_token=True)
        except Exception:
            continue
        a = np.asarray(r.get("per_token") if isinstance(r, dict) else r, dtype=float).ravel()
        if a.size:
            per.append(float(-a.mean() / np.log(2)))
    return np.array(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8000)
    ap.add_argument("--cap-family", type=int, default=3)
    ap.add_argument("--n-per-stratum", type=int, default=60)
    ap.add_argument("--model", default="evo2_7b_base")
    ap.add_argument("--tier", default="tier2")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import virobench_baselines as VBB
    from viral_likelihood import markov_bits_per_base
    OUT = P.sub("viral_likelihood"); os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    d = pd.concat([VBB.load("ALL", "times", s) for s in ("train", "test")], ignore_index=True)
    d = d[d.sequence.str.len() >= a.window].copy()
    d["hostclass"] = d.host.map(host_class)
    d = d.groupby("family").head(a.cap_family).reset_index(drop=True)
    print("pool after family-cap:", d.hostclass.value_counts().to_dict(), flush=True)

    strata, fitpool = {}, []
    for hc in ("prokaryotic", "eukaryotic"):
        sub = d[d.hostclass == hc].sample(frac=1.0, random_state=0).reset_index(drop=True)
        n = min(a.n_per_stratum, len(sub) // 2)
        strata[hc] = [s[:a.window].upper() for s in sub.sequence[:n]]
        fitpool += [s[:a.window].upper() for s in sub.sequence[n:n + 200]]
        print(f"  {hc}: eval n={n}, families={sub.family[:n].nunique()}", flush=True)

    res = dict(window=a.window, cap_family=a.cap_family, model=a.model,
               question="is Evo 2's viral advantage exposure (prokaryotic, IN its training "
                        "corpus) or capability (eukaryotic, EXCLUDED by design)?",
               strata={k: dict(n=len(v)) for k, v in strata.items()}, bits={})
    for hc, seqs in strata.items():
        res["bits"].setdefault(hc, {})
        for k in (3, 5):
            b, _ = markov_bits_per_base(fitpool, seqs, k, cap=a.window)
            res["bits"][hc][f"markov_order{k}"] = round(float(b), 4)
        print(f"  markov {hc}: " +
              " ".join(f"k{k}={res['bits'][hc][f'markov_order{k}']:.4f}" for k in (3, 5)),
              flush=True)

    from turboquant import load_evo2
    t0 = time.time()
    m, tok = load_evo2(a.model, tier=a.tier, device=a.device, verbose=False)
    print(f"  {a.model} loaded ({time.time()-t0:.0f}s)", flush=True)
    per = {}
    for hc, seqs in strata.items():
        v = bits_per_base_evo2(m, tok, seqs, a.model, a.window)
        per[hc] = v
        res["bits"][hc][a.model] = round(float(v.mean()), 4)
        mk = min(res["bits"][hc][f"markov_order{k}"] for k in (3, 5))
        res["bits"][hc][f"{a.model}_vs_markov"] = round(float(v.mean() - mk), 4)
        print(f"  {a.model} {hc:12s}: {v.mean():.4f} bits/base  vs markov "
              f"{v.mean()-mk:+.4f}  (n={v.size})", flush=True)

    if all(k in per for k in ("prokaryotic", "eukaryotic")):
        rb = np.random.default_rng(0); B = 20000
        pk, ek = per["prokaryotic"], per["eukaryotic"]
        mk_p = min(res["bits"]["prokaryotic"][f"markov_order{k}"] for k in (3, 5))
        mk_e = min(res["bits"]["eukaryotic"][f"markov_order{k}"] for k in (3, 5))
        adv_p = mk_p - pk        # positive = Evo2 better than composition
        adv_e = mk_e - ek
        bs = np.array([adv_p[rb.integers(0, len(adv_p), len(adv_p))].mean()
                       - adv_e[rb.integers(0, len(adv_e), len(adv_e))].mean()
                       for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res["advantage_prokaryotic"] = round(float(adv_p.mean()), 4)
        res["advantage_eukaryotic"] = round(float(adv_e.mean()), 4)
        res["exposure_gap"] = dict(value=round(float(adv_p.mean() - adv_e.mean()), 4),
                                   ci=[round(float(lo), 4), round(float(hi), 4)],
                                   p_le_0=round(float((bs <= 0).mean()), 4))
        print(f"\n  advantage over composition: prokaryotic {adv_p.mean():+.4f} | "
              f"eukaryotic {adv_e.mean():+.4f}")
        print(f"  EXPOSURE GAP = {adv_p.mean()-adv_e.mean():+.4f} "
              f"CI [{lo:+.4f},{hi:+.4f}] P(<=0)={float((bs<=0).mean()):.3f}")
        print("  gap >> 0 -> advantage is concentrated on sequence Evo 2 was TRAINED on")
        print("  gap ~ 0  -> advantage holds on viruses it was never shown = real capability")
    p = f"{OUT}/evo2_stratified_by_host_w{a.window}.json"
    json.dump(res, open(p, "w"), indent=2); print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
