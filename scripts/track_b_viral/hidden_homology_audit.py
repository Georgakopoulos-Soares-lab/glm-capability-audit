"""Nested hidden-homology audit: how much uplift survives as evolutionary familiarity is stripped away.

The weakness this closes
------------------------
`homolog_regime_split.py` calls a test row "homolog-free" when MMseqs2 finds no
NUCLEOTIDE alignment to training. That establishes "no detectable nucleotide
homology under this search", not "no evolutionary relationship". In viruses a
protein can stay recognisably conserved long after synonymous substitution has
eroded nucleotide identity, which is exactly why translated search (tBLASTx) is
the standard homology baseline in viral host prediction.

So the current claim rests on the implication

    no nucleotide hit  =>  lookup unavailable  =>  uplift is not homology

whose first step is too strong.

The strata
----------
    L0  all test rows
    L1  no nucleotide hit                        (N-)          the old definition
    L2  no nucleotide AND no translated hit      (N-/T-)       the real novelty test
    L3  L2 and no protein-level hit              (six-frame protein search)
    L4  L2 and the genus is absent from training
    L5  L2 and the family is absent from training

L1\\L2 -- rows the nucleotide search called novel but translated search does not --
is the diagnostic group. If the positives live there, nucleotide-only cleaning
systematically understates exploitable viral relatedness, which is itself a
benchmark-validity result.

Reported at every level: n, each baseline, the floor, each model, the excess and
a bootstrap CI over the correct unit. NN-lookup is excluded from the floor at
every level below L0 because it is structurally disabled there.

Outcomes, all informative, committed before looking:
  A  uplift survives in L2/L3      -> the strongest form of the positive claim
  B  uplift exists only in L1\\L2   -> the positive was translated homology
  C  L2 is nearly empty            -> nucleotide screens vastly overstate novelty
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _arm_guard import sweep_arm_ready
from nn_lookup_probe import search_hits, MMSEQS, SCRATCH          # noqa: E402

R = os.environ.get("VB_VIRAL_OUT",
                   "/data/nvidia/genomic-capability-unlearning/results_viral_bench")
PRED, GLM = f"{R}/preds_rescore", f"{R}/geneb_finetune"
MODELS = ["hvilm", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna", "lucavirus"]


def translated_hits(train, test, threads=24, sens=7.5, evalue=1e-3, max_seqs=20):
    """MMseqs2 --search-type 2: six-frame translated search, the TBLASTX equivalent.

    No identity floor is imposed. At the protein level a 30% nucleotide cut is
    meaningless, and the point here is maximum sensitivity: we want to find the
    homology our nucleotide search missed, not to reproduce its threshold.
    """
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        for nm, df in (("tr", train), ("te", test)):
            with open(f"{d}/{nm}.fa", "w") as f:
                for i, s in enumerate(df.sequence.astype(str)):
                    f.write(f">{i}\n{s}\n")
        cmd = [MMSEQS, "easy-search", f"{d}/te.fa", f"{d}/tr.fa", f"{d}/h.tsv", f"{d}/tmp",
               "--search-type", "2", "-s", str(sens), "-e", str(evalue),
               "--max-seqs", str(max_seqs), "--threads", str(threads), "-v", "1",
               "--format-output", "query,target,pident,alnlen,qlen,evalue,bits"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"translated easy-search failed: {r.stderr[-500:]}")
        if not os.path.getsize(f"{d}/h.tsv"):
            return pd.DataFrame(columns=["q", "t", "pident", "alnlen", "qlen", "evalue", "bits"])
        return pd.read_csv(f"{d}/h.tsv", sep="\t", header=None,
                           names=["q", "t", "pident", "alnlen", "qlen", "evalue", "bits"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def protein_hits(train, test, threads=24, sens=7.5, evalue=1e-3):
    """--search-type 4: translated nucleotide alignment, i.e. protein-level scoring.

    Kept separate from search-type 2 so L3 is a genuinely different query, not a
    relabelling of L2. Prodigal is not installed here, so this is ORF-annotation
    free -- which is the conservative choice anyway: it cannot be wrong about
    where the genes are.
    """
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        for nm, df in (("tr", train), ("te", test)):
            with open(f"{d}/{nm}.fa", "w") as f:
                for i, s in enumerate(df.sequence.astype(str)):
                    f.write(f">{i}\n{s}\n")
        r = subprocess.run(
            [MMSEQS, "easy-search", f"{d}/te.fa", f"{d}/tr.fa", f"{d}/h.tsv", f"{d}/tmp",
             "--search-type", "4", "-s", str(sens), "-e", str(evalue),
             "--threads", str(threads), "-v", "1",
             "--format-output", "query,target,pident,alnlen,qlen,evalue,bits"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ! protein search failed, L3 skipped: {r.stderr[-200:]}")
            return None
        if not os.path.getsize(f"{d}/h.tsv"):
            return pd.DataFrame(columns=["q", "t", "pident", "alnlen", "qlen", "evalue", "bits"])
        return pd.read_csv(f"{d}/h.tsv", sep="\t", header=None,
                           names=["q", "t", "pident", "alnlen", "qlen", "evalue", "bits"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def load(pat):
    ps = sorted(glob.glob(pat))
    if not ps:
        return None, []
    yt, pr = None, []
    for p in ps:
        z = np.load(p)
        if yt is None:
            yt = z["y_true"]
        elif not np.array_equal(yt, z["y_true"]):
            return None, []
        pr.append(z["y_pred"])
    return yt, pr


def boot(y, arms, models, mask, units, reps, seed=0):
    """Cluster bootstrap within one stratum; the floor is re-maximised per replicate."""
    rng = np.random.default_rng(seed)
    rows = np.where(mask)[0]
    u = pd.unique(units[rows])
    idx = {c: rows[units[rows] == c] for c in u}
    draws = {m: [] for m in models}
    for _ in range(reps):
        r = np.concatenate([idx[c] for c in rng.choice(u, size=len(u), replace=True)])
        yy = y[r]
        if len(np.unique(yy)) < 2:
            continue
        b = max(matthews_corrcoef(yy, v[rng.integers(len(v))][r]) for v in arms.values())
        for m, pr in models.items():
            draws[m].append(matthews_corrcoef(yy, pr[rng.integers(len(pr))][r]) - b)
    out = {}
    for m, dd in draws.items():
        if dd:
            dd = np.asarray(dd)
            out[m] = dict(ci_lo=round(float(np.percentile(dd, 2.5)), 4),
                          ci_hi=round(float(np.percentile(dd, 97.5)), 4),
                          p_le_0=round(float((dd <= 0).mean()), 4))
    return out, len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-csv", required=True); ap.add_argument("--test-csv", required=True)
    ap.add_argument("--esc", required=True); ap.add_argument("--kmer-esc", default="")
    ap.add_argument("--cluster-csv", default="", help="test-row-order csv with a `cluster` column")
    ap.add_argument("--tax-train", default="", help="train-row-order csv with family/genus")
    ap.add_argument("--tax-test", default="", help="test-row-order csv with family/genus")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--sweep-tag", default="",
                    help="capacity_sweep json stem; enables the completion guard on the "
                         "CNN/transformer arms (they are written mid-run and a live sweep "
                         "leaves a mixture of fresh and stale seeds on disk)")
    ap.add_argument("--require-arms", action="store_true",
                    help="hard-fail rather than silently dropping an incomplete ladder arm")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--skip-protein", action="store_true", help="skip L3 (search-type 4 is slow)")
    ap.add_argument("--cache", default="", help="npz to persist the search masks in")
    ap.add_argument("--out", default=os.environ.get(
        "VB_REPORTS", "/home/nvidia/genomic-capability-unlearning/reports") + "/hidden_homology")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    rd = lambda p: pd.read_csv(p).rename(columns={"text": "sequence"}).reset_index(drop=True)
    tr, te = rd(a.train_csv), rd(a.test_csv)
    y = te.label.values.astype(int); n = len(te)
    ke = a.kmer_esc or a.esc

    # ---- searches -----------------------------------------------------------
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache)
        hasN, hasT, hasP = z["hasN"], z["hasT"], z["hasP"]
        tbits = z["tbits"]
        print(f"  [masks reused from {a.cache}]")
    else:
        print("  nucleotide search (search-type 3) ...", flush=True)
        hN = search_hits(tr, te, threads=a.threads)
        hasN = np.zeros(n, bool)
        hasN[hN.q.values.astype(int)] = True

        print("  translated search (search-type 2, the tBLASTx equivalent) ...", flush=True)
        hT = translated_hits(tr, te, threads=a.threads)
        hasT = np.zeros(n, bool); tbits = np.zeros(n)
        # Several continuous measures, not just the raw bitscore: bitscore scales
        # with alignment length, so a long weak hit and a short strong one are not
        # distinguished by bits alone. The graded curve reads all of them.
        extra = {k: np.zeros(n) for k in
                 ("bits_per_aa", "qcov", "alnlen", "neglog10E", "bits_x_cov")}
        if len(hT):
            bt = hT.sort_values("bits", ascending=False).drop_duplicates("q")
            qi = bt.q.values.astype(int)
            hasT[qi] = True; tbits[qi] = bt.bits.values
            aln = np.maximum(bt.alnlen.values.astype(float), 1.0)
            qaa = np.maximum(bt.qlen.values.astype(float) / 3.0, 1.0)
            extra["alnlen"][qi] = aln
            extra["bits_per_aa"][qi] = bt.bits.values / aln
            cov = np.clip(aln / qaa, 0, 1)
            extra["qcov"][qi] = cov
            extra["neglog10E"][qi] = -np.log10(np.maximum(bt.evalue.values.astype(float), 1e-300))
            extra["bits_x_cov"][qi] = bt.bits.values * cov

        hasP = np.zeros(n, bool)
        if not a.skip_protein:
            print("  protein-level search (search-type 4) ...", flush=True)
            hP = protein_hits(tr, te, threads=a.threads)
            if hP is not None and len(hP):
                hasP[hP.q.values.astype(int)] = True
        if a.cache:
            os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
            np.savez_compressed(a.cache, hasN=hasN, hasT=hasT, hasP=hasP, tbits=tbits, **extra)

    # ---- taxonomy -----------------------------------------------------------
    unseen = {}
    if a.tax_train and a.tax_test:
        tt, tx = pd.read_csv(a.tax_train), pd.read_csv(a.tax_test)
        if len(tx) == n:
            # species is included because VHDB is a WITHIN-FAMILY task by
            # construction: 0% of its test families and only 9.2% of its genera
            # are unseen, so family-level novelty is untestable there and species
            # (73.9% unseen) is the only lineage axis with real range.
            for col in ("species", "genus", "family"):
                if col in tt.columns and col in tx.columns:
                    seen = set(tt[col].dropna().astype(str))
                    unseen[col] = ~tx[col].astype(str).isin(seen).values
        else:
            print(f"  ! taxonomy has {len(tx)} rows but test has {n}; levels 4-5 skipped")

    # ---- arms ---------------------------------------------------------------
    SWEEP = f"{R}/capacity_sweep"
    arms = {}
    for tag, pat in (("kmer", f"{PRED}/hvue_v2__{ke}__kmer__s*__preds.npz"),
                     ("kmer_gbt", f"{PRED}/hvue_v2__{ke}__kmergbt__s*__preds.npz"),
                     ("aa_gbt", f"{PRED}/hvue_v2__{ke}__aagbt__s*__preds.npz"),
                     ("tax_prior", f"{PRED}/hvue_v2__{ke}__taxprior__s*__preds.npz"),
                     ("cnn", f"{PRED}/*{a.esc}*cw_balanced__cnn_ceiling__s*__preds.npz"),
                     ("transformer", f"{PRED}/*{a.esc}*transformer__cnn_ceiling__s*__preds.npz")):
        # The two ladder arms are written incrementally by capacity_sweep.py, so
        # they must be completion-gated. The composition arms are single-shot.
        if tag in ("cnn", "transformer") and a.sweep_tag:
            sj = f"{SWEEP}/{a.sweep_tag}{'' if tag == 'cnn' else '__transformer'}.json"
            ok, why = sweep_arm_ready(sj, pat)
            if not ok:
                print(f"  ! {tag} arm EXCLUDED: {why}")
                if a.require_arms:
                    sys.exit(f"--require-arms: refusing to score without a complete {tag} arm")
                continue
        yy, pp = load(pat)
        if pp and yy is not None and len(yy) == n:
            arms[tag] = pp
    if not arms:
        sys.exit(f"no baseline arms found for esc={a.esc} (kmer_esc={ke})")

    models = {}
    for m in MODELS:
        c = []
        for j in sorted(glob.glob(f"{GLM}/{a.esc}__{m}__lr*__fullft.json")):
            if any(x in j for x in ("_meanpool", "_ep4", "_truncated")):
                continue
            d = json.load(open(j))
            if len(d.get("runs", [])) >= 3:
                sfx = "".join(s for s in ("__probs", "__ep16", "__ep32", "__ep64") if s in j)
                c.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"], sfx))
        if not c:
            continue
        dev, lr, sfx = max(c)
        ym, pr = load(f"{GLM}/{a.esc}__{m}__lr{lr:g}{sfx}__s*__preds.npz")
        if pr and ym is not None and len(ym) == n:
            ok = [p for p in pr if len(np.unique(p)) > 1]
            if ok:
                models[m] = ok

    units = (pd.read_csv(a.cluster_csv)["cluster"].values if a.cluster_csv
             else np.arange(n))
    usrc = "cluster" if a.cluster_csv else "ROW (anti-conservative)"

    # ---- the ladder ---------------------------------------------------------
    L = [("L0_all", np.ones(n, bool)),
         ("L1_no_nucleotide", ~hasN),
         ("L2_no_nucleotide_no_translated", ~hasN & ~hasT),
         ("L1_minus_L2_translated_only", ~hasN & hasT)]
    if hasP.any():
        L.append(("L3_no_protein_either", ~hasN & ~hasT & ~hasP))
    for i, col in enumerate(("species", "genus", "family"), start=4):
        if col in unseen:
            L.append((f"L{i}_L2_and_unseen_{col}", ~hasN & ~hasT & unseen[col]))

    print(f"\n=== {a.name}: n={n:,}")
    print(f"  nucleotide hit  {hasN.sum():>7,} ({100*hasN.mean():5.1f}%)")
    print(f"  translated hit  {hasT.sum():>7,} ({100*hasT.mean():5.1f}%)")
    print(f"  HIDDEN (N-/T+)  {(~hasN & hasT).sum():>7,} "
          f"({100*(~hasN & hasT).mean():5.1f}% of all; "
          f"{100*(~hasN & hasT).sum()/max(1,(~hasN).sum()):.1f}% of the old 'homolog-free' set)")
    print(f"  bootstrap unit  {usrc}\n")

    out = dict(name=a.name, n_test=int(n), bootstrap_unit=usrc,
               n_nucleotide_hit=int(hasN.sum()), n_translated_hit=int(hasT.sum()),
               n_hidden_translated_only=int((~hasN & hasT).sum()),
               frac_of_old_homolog_free_that_is_hidden=round(
                   float((~hasN & hasT).sum() / max(1, (~hasN).sum())), 4),
               unseen_lineage_frac={k: round(float(v.mean()), 4) for k, v in unseen.items()},
               levels={})
    for lab, mask in L:
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            print(f"  {lab:34s} n={int(mask.sum()):>6,}  -- too few rows / single class")
            out["levels"][lab] = dict(n=int(mask.sum()), skipped="too few rows or single class")
            continue
        fl = {k: float(np.mean([matthews_corrcoef(y[mask], p[mask]) for p in v]))
              for k, v in arms.items()}
        floor = max(fl.values()); farm = max(fl, key=fl.get)
        print(f"  {lab:34s} n={int(mask.sum()):>6,} pos={y[mask].mean():.3f} "
              f"floor={floor:.4f} ({farm})")
        ci, nu = boot(y, arms, models, mask, units, a.reps) if models else ({}, 0)
        rr = {}
        for m, pr in sorted(models.items(),
                            key=lambda kv: -np.mean([matthews_corrcoef(y[mask], p[mask])
                                                     for p in kv[1]])):
            sc = float(np.mean([matthews_corrcoef(y[mask], p[mask]) for p in pr]))
            e = sc - floor
            c = ci.get(m, {})
            tail = (f"  95% CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  P(<=0)={c['p_le_0']:.4f}"
                    if c else "")
            print(f"      {m:11s} {sc:.4f}  excess {e:+.4f}{tail}")
            rr[m] = dict(mcc=round(sc, 4), excess=round(e, 4), **c)
        out["levels"][lab] = dict(n=int(mask.sum()), n_units=nu, floor=round(floor, 4),
                                  floor_arm=farm,
                                  baselines={k: round(v, 4) for k, v in fl.items()},
                                  models=rr)

    # ---- excess vs translated similarity ------------------------------------
    # Not another nearest-neighbour classifier: the question is whether model
    # EXCESS decays as hidden protein-level similarity to training disappears.
    nz = ~hasN
    if nz.sum() > 200 and models:
        qs = np.quantile(tbits[nz & hasT], [0, .25, .5, .75, 1]) if (nz & hasT).sum() > 40 else None
        bins = [("no translated hit", nz & ~hasT)]
        if qs is not None:
            for i in range(4):
                lo, hi = qs[i], qs[i + 1]
                sel = nz & hasT & (tbits >= lo) & (tbits <= hi if i == 3 else tbits < hi)
                bins.append((f"tbits {lo:.0f}-{hi:.0f}", sel))
        print("\n  excess vs translated-hit bitscore (within the nucleotide-no-hit rows):")
        tb = {}
        for lab, sel in bins:
            if sel.sum() < 50 or len(np.unique(y[sel])) < 2:
                continue
            f2 = max(np.mean([matthews_corrcoef(y[sel], p[sel]) for p in v]) for v in arms.values())
            row = {m: round(float(np.mean([matthews_corrcoef(y[sel], p[sel]) for p in pr])) - f2, 4)
                   for m, pr in models.items()}
            best = max(row, key=row.get)
            print(f"    {lab:22s} n={int(sel.sum()):>6,}  floor={f2:.4f}  "
                  f"best={best} excess={row[best]:+.4f}")
            tb[lab] = dict(n=int(sel.sum()), floor=round(float(f2), 4), excess=row)
        out["excess_vs_translated_similarity"] = tb

    p = f"{a.out}/{a.name}__hidden_homology.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
