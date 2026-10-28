"""Stratify evaluation by homolog availability, and use the APPROPRIATE floor in each regime.

The error this fixes
--------------------
We reported NN-lookup as a single full-test-set number with a majority-class fallback on rows
that have no training neighbour. On GUE phage that is 59% of rows. The fallback scores ~0
there, dragging the "NN-lookup floor" from 0.8901 (where it can actually operate) down to
0.6492. Every model was then compared against that blended 0.6492.

That is unfair to the BASELINE and therefore flatters the MODELS: it forces label transfer to
guess on rows where label transfer is impossible by construction, then treats the resulting
low score as the bar to clear.

The correct design
------------------
Split the test set by whether a row has an MMseqs2 training neighbour, and use the floor that
is valid in each regime:

  homolog-available rows : floor = max(NN-lookup, k-mer GBT, CNN, k-mer, transformer)
                           -- NN-lookup is a legitimate competitor here
  homolog-free rows      : floor = max(k-mer GBT, CNN, k-mer, transformer)
                           -- NN-lookup is EXCLUDED: structurally disabled, not a weak baseline

The homolog-free regime is the strongest available test of memorisation-free capability: no
model can succeed there by recalling a training relative, because there isn't one to recall.
For GUE phage (41% homolog coverage, GC-matched classes) that regime is the cleanest
capability measurement in the whole study.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
# moved out of track_b_viral/ when this stopped feeding a reported
# number; the siblings it imports still live there.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "track_b_viral"))
from _arm_guard import sweep_arm_ready
from nn_lookup_probe import search_hits
R = "/data/nvidia/genomic-capability-unlearning/results_viral_bench"
PRED, GLM = f"{R}/preds_rescore", f"{R}/geneb_finetune"
MODELS = ["hvilm", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna", "lucavirus"]
# Arms whose absence would silently LOWER the floor. Missing one is an error, not a warning.
REQUIRED_ARMS = {"kmer", "kmer_gbt", "cnn"}


def load(pat):
    ps = sorted(glob.glob(pat))
    if not ps: return None, []
    yt, pr = None, []
    for p in ps:
        z = np.load(p)
        if yt is None: yt = z["y_true"]
        elif not np.array_equal(yt, z["y_true"]): return None, []
        pr.append(z["y_pred"])
    return yt, pr


def bootstrap(y, arms, models, mask, use_nn, nn, units, reps, seed):
    """Cluster bootstrap of `model - max(valid baseline arms)` WITHIN one regime.

    The floor is re-maximised inside every replicate. Taking the max once on the observed
    data and holding it fixed would understate the interval: which arm is strongest is itself
    a random quantity, and on the homolog-free rows -- where every baseline sits near chance
    -- the identity of the argmax flips between draws.

    Seed indices are resampled alongside rows so the interval covers seed variance too; the
    point estimate is the mean over seeds, i.e. the same quantity the replicates average to.
    """
    rng = np.random.default_rng(seed)
    rows_all = np.where(mask)[0]
    u = pd.unique(units[rows_all])
    idx = {c: rows_all[units[rows_all] == c] for c in u}
    draws, skipped = {m: [] for m in models}, 0
    for _ in range(reps):
        r = np.concatenate([idx[c] for c in rng.choice(u, size=len(u), replace=True)])
        yy = y[r]
        if len(np.unique(yy)) < 2:
            skipped += 1; continue
        fl = [matthews_corrcoef(yy, v[rng.integers(len(v))][r]) for v in arms.values()]
        if use_nn:
            fl.append(matthews_corrcoef(yy, nn[r]))
        b = max(fl)
        for m, pr in models.items():
            draws[m].append(matthews_corrcoef(yy, pr[rng.integers(len(pr))][r]) - b)
    out = {}
    for m, d in draws.items():
        if not d: continue
        d = np.asarray(d)
        out[m] = dict(ci_lo=round(float(np.percentile(d, 2.5)), 4),
                      ci_hi=round(float(np.percentile(d, 97.5)), 4),
                      p_le_0=round(float((d <= 0).mean()), 4))
    return out, len(u), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True); ap.add_argument("--test-csv", required=True)
    ap.add_argument("--esc", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--kmer-esc", default="", help="if the k-mer/GBT preds use another tag")
    ap.add_argument("--allow-missing", action="store_true",
                    help="permit a REQUIRED_ARMS baseline to be absent (it will not be in the floor)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--reps", type=int, default=0,
                    help="bootstrap replicates for the excess CI (0 = point estimates only)")
    ap.add_argument("--cluster-csv", default="",
                    help="csv in TEST ROW ORDER carrying a `cluster` column -- the unit of "
                         "statistical dependence. Without it the bootstrap resamples ROWS, "
                         "which is anti-conservative wherever chunks share a genome.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-cache", default="",
                    help="npz to persist the homolog mask in. MMseqs2's prefilter is not "
                         "bit-reproducible across invocations -- two runs of this script on "
                         "GUE phage returned 41.0%% and 37.9%% homolog coverage. Caching "
                         "pins the regime definition so every arm is scored on the SAME "
                         "row split, which is the property that actually has to hold.")
    ap.add_argument("--out", default="/home/nvidia/genomic-capability-unlearning/reports/regime")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rd = lambda p: pd.read_csv(p).rename(columns={"text": "sequence"}).reset_index(drop=True)
    tr, te = rd(a.train_csv), rd(a.test_csv)
    y = te.label.values.astype(int); n = len(te)
    ke = a.kmer_esc or a.esc

    if a.cluster_csv:
        cu = pd.read_csv(a.cluster_csv)
        if len(cu) != n:
            sys.exit(f"--cluster-csv has {len(cu)} rows but the test set has {n}")
        units, unit_src = cu["cluster"].values, "cluster"
    else:
        units, unit_src = np.arange(n), "ROW (anti-conservative)"

    if a.mask_cache and os.path.exists(a.mask_cache):
        z = np.load(a.mask_cache)
        has, nn = z["has"], z["nn"]
        if len(has) != n:
            sys.exit(f"--mask-cache holds {len(has)} rows but the test set has {n}")
        print(f"  [mask reused from {a.mask_cache}]")
    else:
        h = search_hits(tr, te, threads=a.threads)
        best = h.sort_values("bits", ascending=False).drop_duplicates("q")
        has = np.zeros(n, bool); has[best.q.values.astype(int)] = True
        nn = np.full(n, -1, int)
        nn[best.q.values.astype(int)] = tr.label.values[best.t.values.astype(int)]
        if a.mask_cache:
            os.makedirs(os.path.dirname(a.mask_cache) or ".", exist_ok=True)
            np.savez_compressed(a.mask_cache, has=has, nn=nn)
    print(f"\n=== {a.name}: n={n:,}  homolog-available={has.sum():,} ({100*has.mean():.1f}%)  "
          f"homolog-free={(~has).sum():,}", flush=True)

    arms = {}
    for tag, pat in (("kmer", f"{PRED}/hvue_v2__{ke}__kmer__s*__preds.npz"),
                     ("kmer_gbt", f"{PRED}/hvue_v2__{ke}__kmergbt__s*__preds.npz"),
                     ("cnn", f"{PRED}/*{a.esc}*cnn_ceiling__s*__preds.npz"),
                     ("transformer", f"{PRED}/*{a.esc}*transformer*cnn_ceiling*__preds.npz")):
        yy, pp = load(pat)
        if pp and len(yy) == n:
            arms[tag] = pp
        elif tag in REQUIRED_ARMS and not a.allow_missing:
            # A silently-absent arm lowers the floor and inflates every excess. The phage
            # k-mer/GBT predictions are filed under `gue_phage_fragments` while the FT arm
            # uses `leannmlindsey_GUE_phage_fragments`; without --kmer-esc the glob missed
            # them and the homolog-free floor read 0.7619 (CNN) instead of 0.7667 (GBT).
            sys.exit(f"required baseline arm `{tag}` not found for esc={a.esc} "
                     f"(kmer_esc={ke}); pattern was {pat}. Pass --kmer-esc, or --allow-missing "
                     f"if the arm genuinely has not been run.")
    models = {}
    for m in MODELS:
        c = []
        for j in sorted(glob.glob(f"{GLM}/{a.esc}__{m}__lr*__fullft.json")):
            if any(x in j for x in ("_meanpool", "_ep4", "_truncated")): continue
            d = json.load(open(j))
            if len(d.get("runs", [])) >= 3:
                c.append((np.mean([r["dev_mcc"] for r in d["runs"]]), d["lr"],
                          "__probs" if "__probs" in j else ("__ep16" if "__ep16" in j else "")))
        if not c: continue
        dev, lr, sfx = max(c)
        ym, pr = load(f"{GLM}/{a.esc}__{m}__lr{lr:g}{sfx}__s*__preds.npz")
        if pr and len(ym) == n:
            ok = [p for p in pr if len(np.unique(p)) > 1]
            if ok: models[m] = ok

    out = dict(name=a.name, n_test=int(n), n_homolog_available=int(has.sum()),
               bootstrap_unit=unit_src,
               frac_homolog_available=round(float(has.mean()), 4), regimes={})
    for reg, mask, use_nn in (("homolog_available", has, True),
                              ("homolog_free", ~has, False)):
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            print(f"  {reg}: too few rows / single class, skipped"); continue
        fl = {k: float(np.mean([matthews_corrcoef(y[mask], p[mask]) for p in v]))
              for k, v in arms.items()}
        if use_nn:
            fl["nn_lookup"] = float(matthews_corrcoef(y[mask], nn[mask]))
        floor = max(fl.values()); farm = max(fl, key=fl.get)
        print(f"\n  [{reg}] n={int(mask.sum()):,} pos_rate={y[mask].mean():.3f}")
        print(f"    baselines: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(fl.items())))
        print(f"    FLOOR = {floor:.4f} ({farm})" +
              ("" if use_nn else "   [NN-lookup EXCLUDED: structurally disabled here]"))
        rr = {}
        for m, pr in sorted(models.items(),
                            key=lambda kv: -np.mean([matthews_corrcoef(y[mask], p[mask])
                                                     for p in kv[1]])):
            sc = float(np.mean([matthews_corrcoef(y[mask], p[mask]) for p in pr]))
            print(f"    {m:11s} {sc:.4f}  excess {sc-floor:+.4f}")
            rr[m] = dict(mcc=round(sc, 4), excess=round(sc - floor, 4))
        ci = {}
        if a.reps and models:
            ci, nu, sk = bootstrap(y, arms, models, mask, use_nn, nn, units, a.reps, a.seed)
            print(f"    -- {a.reps:,} {unit_src} replicates ({nu:,} units, {sk} single-class draws skipped)")
            for m in rr:
                if m in ci:
                    rr[m].update(ci[m])
                    sig = "" if ci[m]["ci_lo"] > 0 else ("  [CI spans 0]" if ci[m]["ci_hi"] > 0 else "  [below floor]")
                    print(f"    {m:11s} excess {rr[m]['excess']:+.4f}  "
                          f"95% CI [{ci[m]['ci_lo']:+.4f}, {ci[m]['ci_hi']:+.4f}]  "
                          f"P(<=0)={ci[m]['p_le_0']:.4f}{sig}")
        out["regimes"][reg] = dict(n=int(mask.sum()), floor=round(floor, 4), floor_arm=farm,
                                   baselines={k: round(v, 4) for k, v in fl.items()},
                                   models=rr, nn_lookup_valid=use_nn,
                                   bootstrap_unit=unit_src if a.reps else None,
                                   bootstrap_reps=a.reps or None)
    p = f"{a.out}/{a.name}__regime.json"
    json.dump(out, open(p, "w"), indent=2); print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
