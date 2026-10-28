"""Masked-LM pseudo-perplexity on viral sequence — the half of the likelihood control that
`viral_likelihood*.py` cannot reach.

Why a second instrument
-----------------------
`viral_likelihood_fullctx.py` measures bits/base by autoregressive next-token prediction. That
covers Evo, HyenaDNA and GENERator and correctly EXCLUDES DNABERT-2, GENA-LM, NT-v2 and HViLM,
which are masked LMs with no next-token head. Those four are exactly the models that carry our
supervised viral results, so the positive control was silent on them. This fills that gap with
pseudo-log-likelihood (PLL): mask one token, score the true token given BOTH sides, repeat.

    PLL(x) = sum_i log P(x_i | x_\\i)      bits/base = -PLL / (bases scored) / ln2

TWO NUMBERS THAT MUST NOT BE MIXED
----------------------------------
PLL uses bidirectional context; autoregressive NLL uses left context only. PLL is a strictly
easier prediction problem and its bits/base is systematically lower. **Never put a PLL number
in the same column as an Evo/HyenaDNA number and call one better.** Two consequences:

  * the Markov floor here is BIDIRECTIONAL -- P(base | k left, k right), fit on the same
    corpus -- so the model is compared against a composition baseline with the same
    information advantage it has.
  * cross-model PLL comparison is also confounded by tokenisation: masking one DNABERT-2 BPE
    token hides ~5 bases at once, masking one NT-v2 token hides 6, GENA-LM ~9. Per-base
    difficulty therefore differs by model. Cross-model PLL is reported but is NOT the
    headline; the headline is the paired probe below.

THE PAIRED PROBE (the part that is actually clean)
--------------------------------------------------
HViLM is DNABERT-2 continued-pretrained on ~5M VIRION-linked viral fragments. Same
architecture, same tokeniser, same vocabulary. So

    delta = bits_per_base(DNABERT-2) - bits_per_base(HViLM)

isolates the effect of the viral pretraining with every tokenisation confound cancelled. Run it
on two pools and take the difference of differences:

    pool A: sequences plausibly IN HViLM's pretraining corpus (HVUE Host_Tropism genomes)
    pool B: viral sequences OUTSIDE it (ViroBench families held out of HVUE)

    DiD = delta_A - delta_B

DiD ~ 0  ->  the viral pretraining bought general viral modelling; the downstream host-tropism
             advantage is a transferable capability.
DiD >> 0 ->  HViLM is disproportionately unsurprised by the specific genomes it was trained on;
             the downstream advantage is consistent with memorisation of that corpus.

This is the measurement that discriminates the two hypotheses left open by the VHDB
replication, which is independent of HVUE but NOT of HViLM's pretraining corpus.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "track_a_benchmarks"))
import numpy as np, pandas as pd, torch
import paths as P
import transformers; transformers.logging.set_verbosity_error()
import warnings; warnings.filterwarnings("ignore")

OUT = P.sub("viral_pseudo_ppl")
HVUE_V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")

# (hf id, max tokens, dynamic-module class or None for stock AutoModelForMaskedLM).
# Each of the three load paths exists because a stock load is WRONG for that checkpoint:
#   DNABERT-2/HViLM -> config_class mismatch + triton-3 flash-attn break (see
#                      geneb_finetune.load_dnabert2 for the full reasoning).
#   GENA-LM         -> it is a PRE-LN bert whose auto_map points AutoModel (not
#                      AutoModelForMaskedLM) at its own class. Loading it with the stock
#                      BertForMaskedLM silently re-initialises all 48 LayerNorms and the
#                      model scores 7.5 bits/base -- worse than random. Caught by the guard.
#   NT-v2           -> stock path is correct; verified no freshly-initialised weights.
SPECS = {
 "dnabert2":   ("zhihan1996/DNABERT-2-117M", 512, "bert_layers.BertForMaskedLM"),
 "hvilm":      ("duttaprat/HViLM-base",      512, "bert_layers.BertForMaskedLM"),
 "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t", 512, "modeling_bert.BertForMaskedLM"),
 "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 1024, None),
}


def _no_fresh_weights(mid, fn):
    """Run `fn` and hard-fail if transformers reports freshly-initialised weights.

    A half-random MLM still returns a finite bits/base, so without this the failure mode is a
    plausible-looking number rather than a crash. GENA-LM hit exactly that.
    """
    import io, logging
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
        raise RuntimeError(f"{mid}: MLM head has freshly-initialised weights -> "
                           f"pseudo-PPL would be meaningless. {bad[0][:300]}")
    return m


def load_mlm(key):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    mid, maxtok, dyn = SPECS[key]
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if dyn is None:
        m = _no_fresh_weights(mid, lambda: AutoModelForMaskedLM.from_pretrained(
            mid, trust_remote_code=True))
    else:
        import sys as _s
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        MlmCls = get_class_from_dynamic_module(dyn, mid)
        mod = _s.modules[MlmCls.__module__]
        if hasattr(mod, "flash_attn_qkvpacked_func"):
            mod.flash_attn_qkvpacked_func = None          # triton-3 fallback, DNABERT-2 family
        if dyn.startswith("bert_layers"):
            CfgCls = get_class_from_dynamic_module("configuration_bert.BertConfig", mid)
            m = _no_fresh_weights(mid, lambda: MlmCls.from_pretrained(
                mid, config=CfgCls.from_pretrained(mid)))
        else:
            m = _no_fresh_weights(mid, lambda: MlmCls.from_pretrained(mid))
    return tok, m.eval(), maxtok


def n_bases(tok, tid):
    """Bases a single token covers -- 5 for DNABERT-2 BPE, 6 for NT-v2, ~9 for GENA-LM."""
    s = tok.decode([int(tid)], skip_special_tokens=True).strip().upper()
    return sum(c in "ACGT" for c in s)


@torch.no_grad()
def pll_bits_per_base(key, seqs, device, batch=64, stride=1, max_pos=0):
    """Exact PLL: one forward pass per masked position, batched.

    `stride` subsamples positions (still unbiased for bits/BASE, since the denominator counts
    only the bases of the positions actually scored). `max_pos` caps positions per sequence.
    """
    tok, m, maxtok = load_mlm(key)
    m = m.to(device)
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError(f"{key}: tokenizer has no mask token")
    special = set(tok.all_special_ids)
    tot_bits, tot_bases, t0, skipped = 0.0, 0, time.time(), 0
    per_seq = []
    for si, s in enumerate(seqs):
        ids = tok(s, return_tensors="pt", truncation=True, max_length=maxtok)["input_ids"][0]
        pos = [i for i, t in enumerate(ids.tolist()) if t not in special]
        if stride > 1:
            pos = pos[::stride]
        if max_pos and len(pos) > max_pos:
            pos = list(np.linspace(0, len(pos) - 1, max_pos).astype(int))
            pos = sorted(set(pos))
        if not pos:
            skipped += 1; continue
        sbits, sbases = 0.0, 0
        for b0 in range(0, len(pos), batch):
            chunk = pos[b0:b0 + batch]
            X = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
            for r, p in enumerate(chunk):
                X[r, p] = mask_id
            X = X.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(X).logits.float()
            lp = torch.log_softmax(lg, -1)
            for r, p in enumerate(chunk):
                true = int(ids[p])
                nb = n_bases(tok, true)
                if nb == 0:
                    continue
                sbits += -lp[r, p, true].item() / np.log(2)
                sbases += nb
        if sbases == 0:
            skipped += 1; continue
        tot_bits += sbits; tot_bases += sbases
        per_seq.append(sbits / sbases)
        if si and si % 20 == 0:
            print(f"    {key} {si}/{len(seqs)} running={tot_bits/tot_bases:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
    return tot_bits / max(tot_bases, 1), tot_bases, skipped, per_seq


def bidir_markov_bits_per_base(fit, evalseqs, k):
    """Bidirectional floor: P(base | k bases left, k bases right), add-one smoothed.

    The autoregressive Markov baseline is the wrong control for PLL -- it sees only the left
    side. This one has the same two-sided information the MLM has.
    """
    from collections import defaultdict
    cnt = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    idx = {c: i for i, c in enumerate("ACGT")}
    for s in fit:
        s = s.upper()
        for i in range(k, len(s) - k):
            c = s[i]
            if c not in idx:
                continue
            ctx = s[i - k:i] + s[i + 1:i + 1 + k]
            if any(ch not in idx for ch in ctx):
                continue
            cnt[ctx][idx[c]] += 1
    tot_bits, n = 0.0, 0
    for s in evalseqs:
        s = s.upper()
        for i in range(k, len(s) - k):
            c = s[i]
            if c not in idx:
                continue
            ctx = s[i - k:i] + s[i + 1:i + 1 + k]
            v = cnt.get(ctx)
            v = (v + 1.0) if v is not None else np.ones(4)
            tot_bits += -np.log2(v[idx[c]] / v.sum()); n += 1
    return tot_bits / max(n, 1), n


def load_pools(a):
    """pool A = HVUE Host_Tropism genomes (plausibly in HViLM's pretraining corpus)
       pool B = ViroBench genomes (external collection)

    Each pool gets its OWN bidirectional-Markov fit corpus, drawn from that pool's source and
    disjoint from its eval rows. A single shared fit corpus would be in-distribution for one
    pool and out for the other -- in the smoke run that made pool B's floor 0.8 bits/base
    lower than pool A's for reasons having nothing to do with the models.
    """
    import virobench_baselines as VBB
    rng = np.random.default_rng(a.seed)

    ht_te = pd.read_csv(f"{HVUE_V2}/Host_Tropism/{a.hvue_cfg}/test.csv")
    ht_tr = pd.read_csv(f"{HVUE_V2}/Host_Tropism/{a.hvue_cfg}/train.csv")
    A = [s for s in ht_te.sequence.str.upper().tolist() if len(s) >= a.minlen]
    rng.shuffle(A); A = A[:a.n_eval]
    L = len(A[0]) if A else a.window
    fitA = [s[:L] for s in ht_tr.sequence.str.upper().tolist() if len(s) >= a.minlen]
    rng.shuffle(fitA); fitA = fitA[:a.n_fit]

    vb = pd.concat([VBB.load("ALL", "times", x) for x in ("train", "test")], ignore_index=True)
    vb = vb[vb.sequence.str.len() >= L + a.window].reset_index(drop=True)
    vb = vb.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    # FAMILY-BALANCE. The raw ViroBench `times` pool is ~74% Orthomyxoviridae; unbalanced it is
    # an influenza control, not a viral one, and both a Markov table fit on it and any
    # influenza-heavy pretrained model score absurdly low (0.80 and 0.55 bits/base in the first
    # run). `viral_likelihood_fullctx.py` caps per family for the same reason.
    if a.cap_family:
        vb = (vb.groupby("family", group_keys=False)
                .apply(lambda g: g.head(a.cap_family))
                .sample(frac=1.0, random_state=a.seed).reset_index(drop=True))
    # chunk to pool A's exact length so sequence length is not a confound; take the chunk from
    # a random offset rather than always position 0 (genome 5' ends are atypical)
    B, fams = [], []
    for _, r in vb.iterrows():
        s = r.sequence.upper()
        o = int(rng.integers(0, max(1, len(s) - L)))
        B.append(s[o:o + L]); fams.append(r.family)
        if len(B) >= a.n_eval:
            break
    fitB = []
    for _, r in vb.iloc[len(B):].iterrows():
        s = r.sequence.upper()
        o = int(rng.integers(0, max(1, len(s) - L)))
        fitB.append(s[o:o + L])
        if len(fitB) >= a.n_fit:
            break
    pools = {"A_hvue_ht": A, "B_virobench_bal": B}
    fits  = {"A_hvue_ht": fitA, "B_virobench_bal": fitB}

    if a.poolB_prime_csv:
        # P1b: pool B (ViroBench) is NOT independent of pool A -- 54.3% of family-capped
        # ViroBench chunks hit HVUE Host_Tropism training under easy-search. A null DiD
        # between two overlapping pools says nothing about memorisation. B' is built from
        # ViroBench chunks with NO hit to HVUE train at the loosest (= most aggressive)
        # identity, so DiD(A - B') is a real memorisation test.
        bp = pd.read_csv(a.poolB_prime_csv)
        seqs = [s[:L] for s in bp.sequence.astype(str).str.upper() if len(s) >= L]
        rs = np.random.default_rng(a.seed + 11); rs.shuffle(seqs)
        n_ev = min(a.n_eval, max(1, len(seqs) // 2))
        pools["Bprime_hvue_disjoint"] = seqs[:n_ev]
        fits["Bprime_hvue_disjoint"]  = seqs[n_ev:n_ev + a.n_fit]
        print(f"  pool B' (HVUE-disjoint): {len(pools['Bprime_hvue_disjoint'])} eval / "
              f"{len(fits['Bprime_hvue_disjoint'])} fit, {bp.family.nunique()} families")

    if a.extra_pools:
        # C: non-viral. Fixes the hole in a DiD of ~0 between two VIRAL pools -- both could be
        # inside HViLM's pretraining corpus, in which case "no differential" says nothing. If
        # HViLM's edge over DNABERT-2 is viral-specific it must SHRINK or invert here.
        _c = human_chunks(L, a.n_eval + a.n_fit, a.seed)
        # eval and fit must be DISJOINT or the Markov floor sees its own eval rows
        pools["C_human"] = _c[:a.n_eval]
        fits["C_human"]  = _c[a.n_eval:a.n_eval + a.n_fit]
        # D: pool A dinucleotide-shuffled. Preserves mono- and di-nucleotide composition and
        # destroys everything longer. Separates "HViLM learned viral composition" (edge
        # survives) from "HViLM learned viral structure" (edge collapses).
        rs = np.random.default_rng(a.seed + 7)
        pools["D_shuffledA"] = [dinuc_shuffle(x, rs) for x in A]
        fits["D_shuffledA"]  = [dinuc_shuffle(x, rs) for x in fitA]
    return pools, fits, sorted(set(fams))


def human_chunks(L, n, seed, fa="/data/nvidia/data/hg38/hg38.fa"):
    """hg38 chunks as the non-viral control.

    Random access straight off the .fai index -- pysam is not in this env and the index makes
    it a two-line seek, so adding the dependency would buy nothing.
    """
    idx = {}
    for line in open(fa + ".fai"):
        nm, ln, off, lb, wb = line.split("\t")[:5]
        idx[nm] = (int(ln), int(off), int(lb), int(wb))
    chroms = [c for c in idx if "_" not in c and c not in ("chrM", "chrY", "chrEBV")]
    rng = np.random.default_rng(seed)
    out = []
    with open(fa, "rb") as fh:
        guard = 0
        while len(out) < n and guard < 50 * n:
            guard += 1
            c = chroms[int(rng.integers(len(chroms)))]
            ln, off, lb, wb = idx[c]
            p = int(rng.integers(0, max(1, ln - L)))
            fh.seek(off + p // lb * wb + p % lb)
            raw = fh.read(L + L // lb + 8).replace(b"\n", b"")[:L].decode().upper()
            if len(raw) == L and set(raw) <= set("ACGT"):
                out.append(raw)
    if len(out) < n:
        raise RuntimeError(f"hg38: only {len(out)}/{n} ACGT-clean chunks of {L}bp")
    return out               # caller slices into disjoint eval / fit


def dinuc_shuffle(s, rng):
    """Altschul-Erickson dinucleotide shuffle: an Eulerian walk on the dinucleotide graph
    keeps every dinucleotide count exactly, so k-mer composition to k=2 is preserved."""
    from collections import defaultdict
    if len(s) < 3:
        return s
    edges = defaultdict(list)
    for i in range(len(s) - 1):
        edges[s[i]].append(s[i + 1])
    last = s[-1]
    # pick a random last-exit edge per vertex forming a tree to `last` (ensures Eulerian)
    for v in edges:
        rng.shuffle(edges[v])
    out = [s[0]]; cur = s[0]
    avail = {k: list(v) for k, v in edges.items()}
    for _ in range(len(s) - 1):
        if not avail.get(cur):
            break
        nxt = avail[cur].pop()
        out.append(nxt); cur = nxt
    res = "".join(out)
    return res if len(res) == len(s) else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dnabert2", "hvilm", "gena_lm", "nt_v2_500m"])
    ap.add_argument("--hvue_cfg", default="hard_70_1000bp")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--n_fit", type=int, default=2000)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--minlen", type=int, default=900)
    ap.add_argument("--orders", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_pos", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poolB_prime_csv", default="",
                    help="CSV of viral chunks with NO easy-search hit to HVUE "
                         "train; adds pool B' for the corrected DiD")
    ap.add_argument("--extra_pools", action="store_true",
                    help="add C=hg38 (non-viral) and D=dinuc-shuffled A")
    ap.add_argument("--cap_family", type=int, default=3,
                    help="max genomes per viral family in pool B; 0 disables. "
                         "Unbalanced ViroBench is ~74%% Orthomyxoviridae.")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pools, fits, fams = load_pools(a)
    for k, v in pools.items():
        print(f"pool {k:16s} n={len(v)} len={len(v[0])}"
              + (f"  ({len(fams)} families, cap {a.cap_family})" if k.startswith("B") else ""))
    print(flush=True)

    res = dict(hvue_cfg=a.hvue_cfg, cap_family=a.cap_family, n_eval=a.n_eval,
               window=a.window, poolB_families=len(fams),
               instrument="masked-LM pseudo-log-likelihood (bidirectional context)",
               not_comparable_to="autoregressive bits/base in viral_likelihood*.json",
               pools={k: len(v) for k, v in pools.items()}, bits={}, per_seq={})

    for name, ev in pools.items():
        res["bits"].setdefault(name, {})
        for k in a.orders:
            b, _ = bidir_markov_bits_per_base(fits[name], ev, k)
            res["bits"][name][f"bidir_markov_k{k}"] = round(float(b), 4)
        print(f"  bidir-markov {name}: "
              + "  ".join(f"k{k}={res['bits'][name][f'bidir_markov_k{k}']:.4f}" for k in a.orders),
              flush=True)

    for key in a.models:
        for name, ev in pools.items():
            try:
                b, nb, sk, ps = pll_bits_per_base(key, ev, a.device, a.batch, a.stride, a.max_pos)
                res["bits"][name][key] = round(float(b), 4)
                res["per_seq"].setdefault(key, {})[name] = [round(v, 5) for v in ps]
                print(f"  {key} @ {name}: {b:.4f}  (bases {nb}, skipped {sk})", flush=True)
            except Exception as e:
                print(f"  {key} @ {name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                res["bits"][name][key] = None

    # paired probe: DNABERT-2 vs HViLM, same architecture + tokeniser, only pretraining differs
    if all(res["bits"][p].get("dnabert2") and res["bits"][p].get("hvilm") for p in pools):
        res["delta_dnabert2_minus_hvilm"] = {
            p: round(res["bits"][p]["dnabert2"] - res["bits"][p]["hvilm"], 4) for p in pools}
        d = res["delta_dnabert2_minus_hvilm"]
        print("\n  delta (DNABERT-2 - HViLM), + means HViLM is the better model:")
        for p, v in d.items():
            print(f"    {p:16s} {v:+.4f}")
        if "A_hvue_ht" in d and "Bprime_hvue_disjoint" in d:
            res["did_A_minus_Bprime"] = round(
                d["A_hvue_ht"] - d["Bprime_hvue_disjoint"], 4)
            print(f"  DiD (A - B') = {res['did_A_minus_Bprime']:+.4f}   "
                  f"<-- THE VALID memorisation test (B' has no HVUE overlap)", flush=True)
        if "A_hvue_ht" in d and "B_virobench_bal" in d:
            res["did_A_minus_B"] = round(d["A_hvue_ht"] - d["B_virobench_bal"], 4)
            res["did_A_minus_B_INVALID"] = ("pool B overlaps HVUE; a null DiD here is "
                                            "uninformative. Use did_A_minus_Bprime.")
            print(f"  DiD (A - B) = {res['did_A_minus_B']:+.4f}   "
                  f"(~0 -> general viral modelling; >>0 -> HVUE-corpus memorisation)", flush=True)

    tag = a.tag or f"{a.hvue_cfg}__" + "_".join(sorted(a.models))
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
    print(f"\nwrote {OUT}/{tag}.json")


if __name__ == "__main__":
    main()
