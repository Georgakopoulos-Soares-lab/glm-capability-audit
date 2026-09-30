#!/usr/bin/env python
"""Does having a translated relative actually improve performance?

The audit shows that six frame translated search finds a relative in most of the
test sequences a nucleotide search calls distant. That is a fact about the
filter. Whether it matters for what a model scores is a separate question, and
the answer decides how much the reclassification licenses.

If a model exploited protein level relatedness, it should do better on sequences
that have a translated relative but no nucleotide one than on sequences with no
relative at either level. Those two groups are disjoint and both sit inside the
benchmark's own no-DNA-hit population, so the comparison holds the filter fixed
and varies only the relatedness.

Run per model, per benchmark, then pooled across all cells, because the per cell
intervals are wide and the question is whether the effect exists at all rather
than how large it is for any one model.

Result as of this writing: it does not resolve. The pooled mean gap is +0.018
with a 95 percent interval of [-0.013, +0.049], and 8 of 10 cells are positive
at a sign test p of 0.109. We therefore report that we could not determine
whether translated relatedness affects performance, and we do not claim the
models exploit it. A k-mer logistic regression, which has no access to reading
frames, shows the clearest single gap in the table, which is the reason not to
read the direction as mechanism.
"""
import argparse, glob, json, os, re
import numpy as np
from sklearn.metrics import matthews_corrcoef

MASKS = os.environ.get("VB_HIDDEN_MASKS",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench/hidden_masks")
FT = os.environ.get("VB_FINETUNE",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench/geneb_finetune")
PRED = os.environ.get("VB_PREDS",
    "/data/nvidia/genomic-capability-unlearning/results_viral_bench/preds_rescore")
BENCH = {"HVUE_HT_hard_70": "Host_Tropism__hard_70_1000bp",
         "HVUE_HT_std_95": "Host_Tropism__standard_95_1000bp"}
MODELS = ("hvilm", "dnabert2", "nt_v2_500m", "gena_lm", "hyenadna")
ARMS = ("kmer", "kmer_gbt", "cnn")


def groups(name):
    """(has a translated relative, has none) inside the no-DNA-hit population."""
    z = np.load(f"{MASKS}/{name}.npz")
    hn, ht = z["hasN"], z["hasT"]
    return (~hn) & ht, (~hn) & (~ht)


def best_run(esc, mdl):
    """The dev selected run, matching how the paper scores each model."""
    best = None
    for f in glob.glob(f"{FT}/{esc}__*__preds.npz"):
        m = re.match(r".*__(\w+)__lr[\d.e-]+__s\d+__preds\.npz", f)
        if not m or m.group(1) != mdl:
            continue
        d = np.load(f)
        s = matthews_corrcoef(d["y_true"], d["y_pred"])
        if best is None or s > best[0]:
            best = (s, d["y_true"], d["y_pred"])
    return best


def gap(y, p, hid, non, rng, B):
    a = matthews_corrcoef(y[hid], p[hid])
    b = matthews_corrcoef(y[non], p[non])
    ih, inn = np.where(hid)[0], np.where(non)[0]
    d = []
    for _ in range(B):
        s1 = rng.choice(ih, len(ih), True)
        s2 = rng.choice(inn, len(inn), True)
        if len(set(y[s1])) < 2 or len(set(y[s2])) < 2:
            continue
        d.append(matthews_corrcoef(y[s1], p[s1]) - matthews_corrcoef(y[s2], p[s2]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return a, b, a - b, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cells, held = [], []

    for name, esc in BENCH.items():
        hid, non = groups(name)
        print(f"\n{name}   with a translated relative n={hid.sum():,}   without n={non.sum():,}")
        for mdl in MODELS:
            r = best_run(esc, mdl)
            if r is None or len(r[1]) != len(hid):
                continue
            _, y, p = r
            x = gap(y, p, hid, non, rng, a.boot)
            cells.append((name, mdl, x[2])); held.append((y, p, hid, non))
            flag = "resolved" if x[3] > 0 or x[4] < 0 else "spans zero"
            print(f"   {mdl:11s} {x[0]:+.3f} vs {x[1]:+.3f}   gap {x[2]:+.3f}  "
                  f"[{x[3]:+.3f},{x[4]:+.3f}]  {flag}")
        for arm in ARMS:                      # the control: no reading frames at all
            fs = sorted(glob.glob(f"{PRED}/hvue_v2__{esc}__{arm}__s*__preds.npz"))
            if not fs:
                continue
            d = np.load(fs[0])
            if "y_pred" not in d or len(d["y_true"]) != len(hid):
                continue
            x = gap(d["y_true"], d["y_pred"], hid, non, rng, a.boot)
            flag = "resolved" if x[3] > 0 or x[4] < 0 else "spans zero"
            print(f"   {arm:11s} {x[0]:+.3f} vs {x[1]:+.3f}   gap {x[2]:+.3f}  "
                  f"[{x[3]:+.3f},{x[4]:+.3f}]  {flag}   baseline")

    pos = sum(1 for _, _, g in cells if g > 0)
    print(f"\ngap positive on {pos} of {len(cells)} model by benchmark cells")
    try:
        from scipy import stats
        print(f"  sign test p = {stats.binomtest(pos, len(cells), 0.5).pvalue:.3f}")
    except ImportError:
        pass
    est = []
    for _ in range(a.boot):
        per = []
        for y, p, hid, non in held:
            ih, inn = np.where(hid)[0], np.where(non)[0]
            s1 = rng.choice(ih, len(ih), True); s2 = rng.choice(inn, len(inn), True)
            if len(set(y[s1])) < 2 or len(set(y[s2])) < 2:
                continue
            per.append(matthews_corrcoef(y[s1], p[s1]) - matthews_corrcoef(y[s2], p[s2]))
        if per:
            est.append(np.mean(per))
    lo, hi = np.percentile(est, [2.5, 97.5])
    print(f"\npooled mean gap {np.mean(est):+.3f}  [{lo:+.3f},{hi:+.3f}]")
    print("  " + ("RESOLVED" if lo > 0 or hi < 0 else
                  "spans zero, so we do not claim translated relatedness affects performance"))


if __name__ == "__main__":
    main()
