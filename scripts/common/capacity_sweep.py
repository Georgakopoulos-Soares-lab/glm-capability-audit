
"""
BASELINE CAPACITY CEILING sweep.

Every "a small CNN matches the FM" claim in this project rests on ONE architecture at
0.64M params. That is an arbitrary point, not a ceiling. This sweeps architecture family
x capacity (0.1M -> ~10M) so we can state whether the supervised baseline is saturated.

Two possible outcomes, both scientifically useful:
  - CNN plateaus well below 10M  -> baseline is saturated, FM comparisons stand as fair
  - CNN keeps improving to 10M   -> our published baseline was UNDER-POWERED, and any FM
                                    win inside the improvement margin is void

Architectures:
  dilated : the incumbent (exponentially dilated conv stack, max+mean pool)
  unet    : encoder/decoder with skip connections, strided down / transposed up
  resnet  : residual conv blocks, no dilation

Discipline: architecture AND capacity are selected on DEV only; test is scored once per
(arch, capacity) cell and the selection is reported separately so the reader can see both
the dev-selected number and the full curve.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import paths as P
from sklearn.metrics import (matthews_corrcoef, f1_score, accuracy_score, roc_auc_score,
                             average_precision_score)
import warnings; warnings.filterwarnings("ignore")

OUT = P.sub("capacity_sweep")
NT_DIR = os.environ.get("VB_NT_DIR", "/data/nvidia/data/ntv3")
MAPI = np.full(256, 4, np.int64)
for i, c in enumerate("ACGT"):
    MAPI[ord(c)] = i; MAPI[ord(c.lower())] = i

# ---------------- architectures ----------------
class Dilated(nn.Module):
    """Kernel size k and dilation base b are SWEPT, not fixed.

    arXiv:2411.02796 finds that tuning kernel size and dilation rate is what lets a plain CNN
    match genomic FMs; our own receptive-field result says the same thing mechanistically.
    RF = 1 + sum_i (k-1) * b**min(i,8).
    """
    def __init__(self, ncls, ch, nb, drop=0.2, k=9, dbase=2):
        super().__init__(); self.emb = nn.Embedding(5, 16); L, cin = [], 16
        for i in range(nb):
            dil = dbase ** min(i, 8)
            L += [nn.Conv1d(cin, ch, k, padding=dil * (k // 2), dilation=dil),
                  nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(drop)]; cin = ch
        self.conv = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Linear(2*ch, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, ncls))
    def forward(self, x):
        h = self.conv(self.emb(x).transpose(1, 2))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

class UNet(nn.Module):
    """Encoder/decoder with skip connections -- multi-scale receptive field without
    exponential dilation, the natural 'more capacity' alternative for long sequences."""
    def __init__(self, ncls, ch, depth, drop=0.2, k=9, dbase=2):
        super().__init__(); self.emb = nn.Embedding(5, 16)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        cin = 16; pad = k // 2
        chans = [ch * (2**min(i, 3)) for i in range(depth)]
        for c in chans:
            self.down.append(nn.Sequential(nn.Conv1d(cin, c, k, padding=pad), nn.BatchNorm1d(c),
                                           nn.GELU(), nn.Conv1d(c, c, k, stride=4, padding=pad),
                                           nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(drop)))
            cin = c
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.Sequential(nn.Conv1d(chans[i] + chans[i-1], chans[i-1], k, padding=pad),
                                         nn.BatchNorm1d(chans[i-1]), nn.GELU(), nn.Dropout(drop)))
        self.head = nn.Sequential(nn.Linear(2*chans[0], 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, ncls))
    def forward(self, x):
        h = self.emb(x).transpose(1, 2); skips = []
        for d in self.down:
            skips.append(h); h = d(h)
        for k, u in enumerate(self.up):
            s = skips[-(k+1)]
            h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = u(torch.cat([h, s], 1))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

class ResBlk(nn.Module):
    def __init__(self, c, drop, k=9, dil=1):
        super().__init__(); pad = dil * (k // 2)
        self.b = nn.Sequential(nn.Conv1d(c, c, k, padding=pad, dilation=dil), nn.BatchNorm1d(c),
                               nn.GELU(), nn.Dropout(drop),
                               nn.Conv1d(c, c, k, padding=pad, dilation=dil), nn.BatchNorm1d(c))
    def forward(self, x): return F.gelu(x + self.b(x))

class ResNet(nn.Module):
    """dbase>1 makes this a DILATED residual net (SpliceAI/ChromBPNet-style); dbase=1 is plain."""
    def __init__(self, ncls, ch, nb, drop=0.2, k=9, dbase=1):
        super().__init__(); self.emb = nn.Embedding(5, 16)
        self.stem = nn.Sequential(nn.Conv1d(16, ch, k, padding=k//2), nn.BatchNorm1d(ch), nn.GELU())
        self.body = nn.Sequential(*[ResBlk(ch, drop, k, dbase ** min(i, 8)) for i in range(nb)])
        self.head = nn.Sequential(nn.Linear(2*ch, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, ncls))
    def forward(self, x):
        h = self.body(self.stem(self.emb(x).transpose(1, 2)))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

class Transformer(nn.Module):
    """From-scratch transformer baseline -- the control the CNN ladder cannot provide.

    Every gLM in this study differs from the CNN baselines in TWO ways at once: transformer
    architecture AND genomic pretraining. A randomly-initialised transformer trained only on
    the task data separates them: if it matches a pretrained gLM, the gain was architectural,
    not from pretraining.

    Single-nucleotide embedding + learned positional encoding + pre-LN encoder + masked mean
    pool. `ch` is d_model, `nb` is depth, `k` is reused as n_heads so the existing three-stage
    search (capacity -> kernel/dilation -> LR) sweeps depth/width/heads with no new machinery.
    """
    def __init__(self, ncls, ch, nb, k=8, dbase=None, maxlen=8192, dropout=0.1):
        super().__init__()
        nhead = max(1, min(k, ch // 32))
        while ch % nhead: nhead -= 1
        self.emb = nn.Embedding(5, ch)
        self.pos = nn.Parameter(torch.zeros(1, maxlen, ch)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(ch, nhead, dim_feedforward=4 * ch, dropout=dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nb)
        self.norm = nn.LayerNorm(ch)
        self.head = nn.Linear(ch, ncls)
    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.shape[1]]
        h = self.norm(self.enc(h))
        return self.head(h.mean(1))


def build(arch, ncls, ch, nb, k=9, dbase=None):
    if dbase is None: dbase = 1 if arch == "resnet" else 2
    return {"dilated": Dilated, "unet": UNet, "resnet": ResNet,
            "transformer": Transformer}[arch](ncls, ch, nb, k=k, dbase=dbase)


def receptive_field(arch, nb, k, dbase):
    if arch == "transformer": return "global"   # full self-attention
    """RF in bp. U-Net downsamples 4x per level so it is effectively global."""
    if arch == "unet": return None
    if arch == "dilated": return 1 + sum((k - 1) * dbase ** min(i, 8) for i in range(nb))
    return 1 + (k - 1) + sum(2 * (k - 1) * dbase ** min(i, 8) for i in range(nb))

# capacity ladder: (arch, ch, blocks) chosen to span ~0.1M -> ~10M
LADDER = [
    # (arch, ch, blocks) calibrated to span ~0.04M -> ~10M params.
    # dilated ch=128 nb=5 (0.68M) is the INCUMBENT baseline used in every prior result.
    ("dilated", 32, 3), ("dilated", 64, 4), ("dilated", 128, 5), ("dilated", 256, 5), ("dilated", 448, 6),
    ("unet",    24, 3), ("unet",   48, 3), ("unet",   112, 3), ("unet",    64, 4),
    ("resnet",  64, 3), ("resnet", 128, 4), ("resnet", 256, 5), ("resnet", 320, 5),
]
# from-scratch transformer ladder, ~0.2M -> ~12M params, spanning the CNN range so the
# comparison is at matched capacity rather than matched name.
TF_LADDER = [
    ("transformer", 64, 2), ("transformer", 128, 2), ("transformer", 128, 4),
    ("transformer", 256, 4), ("transformer", 256, 6), ("transformer", 384, 6),
]
INCUMBENT = ("dilated", 128, 5)

def enc(seqs, L):
    X = np.full((len(seqs), L), 4, np.int64)
    for i, s in enumerate(seqs):
        a = MAPI[np.frombuffer(s[:L].encode(), np.uint8)]; X[i, :len(a)] = a
    return X

def load(ds, task, level, seqcap, min_count=1):
    if ds == "splice":
        tr = pd.read_parquet(f"{NT_DIR}/{task}/train.parquet")
        te = pd.read_parquet(f"{NT_DIR}/{task}/test.parquet")
        rng = np.random.default_rng(0); m = np.zeros(len(tr), bool)
        m[rng.permutation(len(tr))[:int(.15*len(tr))]] = True
        return tr[~m].reset_index(drop=True), tr[m].reset_index(drop=True), te, "label", "mcc"
    if ds == "hvue":
        # allow an explicit split file so the ladder can run on the homology-strict splits,
        # not just splits_ungated/*identity_disjoint_hsd0
        _sd = os.environ.get("VB_SPLIT_DIR") or P.sub("splits_ungated")
        _sx = os.environ.get("VB_SPLIT_SUFFIX", "identity_disjoint_hsd0")
        d = pd.read_parquet(f"{_sd}/{task}__{_sx}.parquet")
        trall = d[d.partition == "train"].reset_index(drop=True)
        te = d[d.partition == "val"].reset_index(drop=True)
        # group-disjoint dev, so capacity/arch selection faces the same homology holdout as test
        from sklearn.model_selection import GroupShuffleSplit
        gi, di = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                      .split(trall, groups=trall.group.values))
        return (trall.iloc[gi].reset_index(drop=True), trall.iloc[di].reset_index(drop=True),
                te, "label", "auroc")
    if ds == "geneb":
        # GENEB ships tasks/<id>/{train,test}.csv with columns text,label and NO dev split.
        # Dev carve is a stratified 15% of train at seed 42 -- identical to
        # scripts/geneb/fair_kmer_sentinel.py and scripts/track_a_benchmarks/geneb_finetune.py,
        # so the CNN ladder, the fair k-mer and the FT arm all select on the same rows.
        G = os.environ.get("VB_GENEB_DIR", "/data/nvidia/geneb_data/tasks")
        tr_all = pd.read_csv(f"{G}/{task}/train.csv").rename(columns={"text": "sequence"})
        te = pd.read_csv(f"{G}/{task}/test.csv").rename(columns={"text": "sequence"})
        # --train_cap used to be IGNORED for geneb (passed through as `seqlen`), so the GUE
        # virus_covid CNN ladder trained on all 62,334 rows while every gLM saw the
        # --cap_train 40000 subset -- a 1.56x data advantage to the baseline. Cap here using
        # the SAME unstratified sample(random_state=0) and the SAME pre-carve ordering as
        # geneb_finetune.py, or the two arms are not reading the same rows.
        if seqcap and len(tr_all) > seqcap:
            tr_all = tr_all.sample(n=seqcap, random_state=0).reset_index(drop=True)
        from sklearn.model_selection import train_test_split
        i_tr, i_dv = train_test_split(np.arange(len(tr_all)), test_size=0.15,
                                      stratify=tr_all.label.values, random_state=42)
        return (tr_all.iloc[i_tr].reset_index(drop=True),
                tr_all.iloc[i_dv].reset_index(drop=True), te, "label", "mcc")
    if ds == "hvue_v2":
        # HVUE v2 (HF duttaprat/HVUE-v2 rev 9d7bfa32). `task` is "<Task>/<config>", e.g.
        # "Pathogenicity/standard_capped_1000bp". v2 ships its OWN dev split, so no carve is
        # made -- using the authors' partitions as provided is the honest test of their split.
        V2 = os.environ.get("HVUE_V2_DIR", "/data/nvidia/hvue_v2")
        f = lambda part: pd.read_csv(f"{V2}/{task}/{part}.csv").rename(columns={"sequence": "sequence"})
        tr, dv, te = f("train"), f("dev"), f("test")
        if seqcap and len(tr) > seqcap:
            # pandas 3.0 drops the grouping column from apply(); select it back so `label`
            # survives. Sampled rows are unchanged. See geneb_finetune.py.
            tr = tr.groupby("label", group_keys=False)[list(tr.columns)].apply(
                lambda g: g.sample(n=max(1, int(round(seqcap * len(g) / len(tr)))), random_state=0)
            ).reset_index(drop=True)
        return tr, dv, te, "label", "mcc"
    if ds == "csvdir":
        # Generic: <CSVDIR>/{train,dev,test}.csv with columns sequence,label. Used for
        # externally-built tasks that ship their own dev split (e.g. the Virus-Host DB
        # host-tropism task in build_vhdb_tropism.py).
        root = os.environ["CSVDIR"]
        f = lambda part: pd.read_csv(f"{root}/{part}.csv")
        tr, dv, te = f("train"), f("dev"), f("test")
        if seqcap and len(tr) > seqcap:
            # pandas 3.0 drops the grouping column from apply(); select it back so `label`
            # survives. Sampled rows are unchanged. See geneb_finetune.py.
            tr = tr.groupby("label", group_keys=False)[list(tr.columns)].apply(
                lambda g: g.sample(n=max(1, int(round(seqcap * len(g) / len(tr)))), random_state=0)
            ).reset_index(drop=True)
        return tr, dv, te, "label", "mcc"
    if ds == "virobench":
        # identical construction to virobench_baselines.py so the ladder is directly comparable
        # virobench_baselines lives in scripts/track_b_viral/, not scripts/common/
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'track_b_viral'))
        import virobench_baselines as VBB
        tr, dv, te = (VBB.load("ALL", "times", x) for x in ("train", "val", "test"))
        # min_count MUST match virobench_frozen_probe.py (default 1) or the label space
        # differs and the ladder is not comparable to the frozen-probe numbers
        keep = tr[level].value_counts()
        keep = set(keep[keep >= min_count].index) & set(te[level].dropna())
        tr, dv, te = (d[d[level].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
        ixm = {c: i for i, c in enumerate(sorted(keep))}
        for d in (tr, dv, te):
            d["label"] = d[level].map(ixm)
        return tr, dv, te, "label", "macro_f1"
    raise ValueError(ds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["splice","hvue","virobench","geneb","hvue_v2","csvdir"])
    ap.add_argument("--task", default="ALL_times")
    ap.add_argument("--level", default="family")
    ap.add_argument("--seqlen", type=int, default=0, help="0 = native length")
    ap.add_argument("--min_count", type=int, default=1, help="virobench: match frozen probe (1)")
    ap.add_argument("--class_weight", default="none", choices=["none", "balanced"],
                    help="'balanced' mirrors sklearn's class_weight='balanced' used by the "
                         "k-mer baseline. On HVUE v2 Host_Tropism (3.5-5.5%% positive) an "
                         "unweighted CNN is handicapped on exactly the axis under test, which "
                         "makes max(k-mer, CNN) an unfair baseline. Default 'none' preserves "
                         "all pre-existing results; pass 'balanced' for ALL tasks in a "
                         "comparison, never for one task only.")
    ap.add_argument("--train_cap", type=int, default=0,
                    help="hvue_v2: class-stratified cap on TRAIN rows (0 = all). HVUE v2 has up "
                         "to 454k train rows per config; capping is recorded in the output json.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,43])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-4, 3e-3],
                    help="LR grid; the FMs get one, so the baseline must too")
    ap.add_argument("--refit_seeds", type=int, default=0, help="re-run the dev-selected cell at N seeds")
    ap.add_argument("--arch_family", default="cnn", choices=["cnn", "transformer", "both"],
                    help="which ladder to search; transformer = from-scratch "
                         "control isolating pretraining from architecture")
    ap.add_argument("--only_selected", action="store_true",
                    help="skip the ladder; re-run ONLY the dev-selected cell recorded in the "
                         "existing output json. For regenerating saved predictions without "
                         "repeating a full search -- the cell config is READ from the json, so "
                         "this cannot re-select on test.")
    ap.add_argument("--force", action="store_true", help="redo even if the output json exists")
    ap.add_argument("--predictions_dir", default="",
                    help="save per-seed test predictions (needed for bootstrap CIs and for "
                         "homology-filtered re-scoring)")
    ap.add_argument("--kernels", nargs="+", type=int, default=[5, 9, 15, 21])
    ap.add_argument("--dbases", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--topk2", type=int, default=3, help="RF configs carried into the LR sweep")
    ap.add_argument("--topk", type=int, default=3, help="architectures carried into the LR sweep"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    _tag = (f"{a.dataset}__" + a.task.replace("/", "__")
            + ("__cw_balanced" if a.class_weight == "balanced" else "")
            + ("" if a.arch_family == "cnn" else f"__{a.arch_family}")) + (f"__{os.environ['VB_SPLIT_SUFFIX']}" if os.environ.get('VB_SPLIT_SUFFIX') else "") + (f"__{a.level}" if a.dataset == "virobench" else "")
    if a.only_selected and not os.path.exists(f"{OUT}/{_tag}.json"):
        print(f"--only_selected needs {_tag}.json to exist (it reads the cell from it)", flush=True)
        return
    if not a.force and not a.only_selected and os.path.exists(f"{OUT}/{_tag}.json"):
        print(f"SKIP {_tag}.json already exists (use --force to redo)", flush=True); return
    tr, dv, te, ycol, metric = load(a.dataset, a.task, a.level,
                                    a.train_cap if a.dataset in ("hvue_v2", "csvdir", "geneb") else a.seqlen,
                                    a.min_count)
    L = a.seqlen or int(pd.concat([tr, dv, te]).sequence.str.len().max())
    ncls = int(pd.concat([tr,dv,te])[ycol].nunique())
    print(f"{a.dataset}/{a.task}: classes={ncls} train={len(tr)} dev={len(dv)} test={len(te)} L={L} metric={metric}", flush=True)
    Itr = torch.from_numpy(enc(tr.sequence.tolist(), L)); ytr = torch.tensor(tr[ycol].values, dtype=torch.long)
    Idv = torch.from_numpy(enc(dv.sequence.tolist(), L)); ydv = dv[ycol].values.astype(int)
    Ite = torch.from_numpy(enc(te.sequence.tolist(), L)); yte = te[ycol].values.astype(int)

    # Class weights from the TRAIN split only (never dev/test), matching sklearn's
    # 'balanced': n_samples / (n_classes * count_c).
    cls_w = None
    if a.class_weight == "balanced":
        cnt = np.bincount(ytr.numpy(), minlength=ncls).astype(np.float64)
        cls_w = torch.tensor(len(ytr) / (ncls * np.maximum(cnt, 1)), dtype=torch.float32,
                             device=a.device)
        print(f"  class_weight=balanced -> counts {cnt.astype(int).tolist()} "
              f"weights {[round(float(w), 3) for w in cls_w]}", flush=True)

    def score(y, p, prob=None):
        if metric == "auroc": return float(roc_auc_score(y, prob))
        if metric == "auprc": return float(average_precision_score(y, prob))
        if metric == "macro_f1": return float(f1_score(y, p, average="macro", zero_division=0))
        return float(matthews_corrcoef(y, p))

    def aux_metrics(y, p, prob):
        """Always reported alongside the selection metric. AUPRC is the informative one on
        an imbalanced task; accuracy is deliberately NOT reported (96.5%% by predicting
        all-negative on Host_Tropism)."""
        out = {"mcc": round(float(matthews_corrcoef(y, p)), 4)}
        if ncls == 2 and prob is not None:
            try: out["auprc"] = round(float(average_precision_score(y, prob)), 4)
            except ValueError: out["auprc"] = None
            try: out["auroc"] = round(float(roc_auc_score(y, prob)), 4)
            except ValueError: out["auroc"] = None
        out["pos_rate_pred"] = round(float((np.asarray(p) == 1).mean()), 4) if ncls == 2 else None
        return out
    def run_cell(arch, ch, nb, lr, seeds, k=9, dbase=None):
        """Train one (architecture, lr) cell over `seeds`; return per-seed dev/test."""
        per = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            m = build(arch, ncls, ch, nb, k=k, dbase=dbase).to(a.device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
            def pred(I):
                m.eval(); pr=[]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(0, len(I), 256):
                        pr.append(torch.softmax(m(I[i:i+256].to(a.device)).float(), -1).cpu().numpy())
                m.train(); pr=np.vstack(pr); return pr.argmax(1), pr
            best=(-1,0,None,None); noimp=0
            try:
                for ep in range(1, a.epochs+1):
                    perm = torch.randperm(len(Itr))
                    for i in range(0, len(perm)-a.bs+1, a.bs):
                        b = perm[i:i+a.bs]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = F.cross_entropy(m(Itr[b].to(a.device)), ytr[b].to(a.device),
                                                   weight=cls_w)
                        opt.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                    sch.step()
                    pd_, pp_ = pred(Idv); sc = score(ydv, pd_, pp_[:,1] if ncls==2 else None)
                    if sc > best[0]+5e-4:
                        pt_, ptp_ = pred(Ite); best=(sc, ep, pt_, ptp_); noimp=0
                    else:
                        noimp += 1
                        if noimp >= 6: break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); del m; return None
            prob_te = best[3][:,1] if ncls==2 else None
            ts = score(yte, best[2], prob_te)
            rec = dict(seed=seed, dev=round(best[0],4), test=round(ts,4), best_epoch=best[1],
                       test_aux=aux_metrics(yte, best[2], prob_te))
            if a.predictions_dir:
                os.makedirs(a.predictions_dir, exist_ok=True)
                _cw = "__cw_balanced" if a.class_weight == "balanced" else ""
                # `__cnn_ceiling` was a fixed literal regardless of --arch_family, so a
                # transformer refit silently replaced the CNN's predictions under the CNN's
                # name. Downstream, every arm labelled `cnn` was then the transformer.
                _af = "" if a.arch_family == "cnn" else f"__{a.arch_family}"
                _pp = (f"{a.predictions_dir}/{a.dataset}__{a.task.replace('/','__')}{_cw}{_af}"
                       f"__cnn_ceiling__s{seed}__preds.npz")
                np.savez_compressed(_pp, y_true=yte, y_pred=best[2],
                                    y_prob=(prob_te if prob_te is not None else best[3]))
                rec["predictions_path"] = _pp
            per.append(rec)
            del m; torch.cuda.empty_cache()
        return per

    def rec(arch, ch, nb, k, dbase, lr, per):
        npar = sum(p.numel() for p in build(arch, ncls, ch, nb, k=k, dbase=dbase).parameters())
        dv = float(np.mean([q['dev'] for q in per])); te = float(np.mean([q['test'] for q in per]))
        return dict(arch=arch, ch=ch, blocks=nb, k=k, dbase=dbase, lr=lr,
                    params=npar, params_M=round(npar/1e6, 3),
                    rf_bp=receptive_field(arch, nb, k, dbase), dev=round(dv, 4), test=round(te, 4),
                    test_sd=round(float(np.std([q['test'] for q in per], ddof=1)), 4) if len(per) > 1 else None,
                    runs=per)

    LADDER_USE = {'cnn': LADDER, 'transformer': TF_LADDER,
                  'both': LADDER + TF_LADDER}[a.arch_family]
    rows = []
    if a.only_selected:
        prev = json.load(open(f"{OUT}/{_tag}.json"))
        ds = prev["dev_selected"]
        # train_cap is part of the CELL DEFINITION, not a runtime knob. The HVUE CNNs were
        # built at cap 40000 (matching the gLM arms); re-running --only_selected without it
        # silently trains on 230,934 rows -- a different, stronger CNN than the one whose
        # score is in the table, and no longer matched to the gLMs it is compared against.
        if prev.get("train_cap") not in (None, a.train_cap):
            raise SystemExit(
                f"--only_selected: cell was built with train_cap={prev['train_cap']} but you "
                f"passed {a.train_cap}. Pass --train_cap {prev['train_cap']} to reproduce it.")
        src = next((r for r in prev["ladder"]
                    if r["arch"] == ds["arch"] and r.get("lr") == ds["lr"]
                    and r.get("k") == ds["k"] and r.get("dbase") == ds["dbase"]
                    and r["params_M"] == ds["params_M"]), None)
        if src is None:
            raise SystemExit("--only_selected: dev_selected cell not found in ladder")
        seeds = list(range(42, 42 + (a.refit_seeds or ds.get("n_seeds") or 3)))
        print(f"  [only_selected] {ds['arch']} ch={src['ch']} nb={src['blocks']} k={ds['k']} "
              f"d={ds['dbase']} lr={ds['lr']} seeds={seeds}", flush=True)
        per = run_cell(ds["arch"], src["ch"], src["blocks"], ds["lr"], seeds,
                       k=ds["k"], dbase=ds["dbase"])
        if not per:
            raise SystemExit("--only_selected: cell OOMed")
        t = [q["test"] for q in per]
        print(f"  [only_selected] TEST={np.mean(t):.4f} +/-{np.std(t, ddof=1):.4f} "
              f"(json recorded {ds['test']}) over {len(t)} seeds", flush=True)
        for q in per:
            print(f"     seed {q['seed']}: test={q['test']:.4f} aux={q['test_aux']}", flush=True)
        return

    # ---- STAGE 1: architecture x capacity, default kernel/dilation, base LR, 1 seed ----
    s1 = []
    for arch, ch, nb in LADDER_USE:
        db = 1 if arch in ("resnet", "transformer") else 2
        per = run_cell(arch, ch, nb, a.lrs[0], a.seeds[:1], k=9, dbase=db)
        if per is None:
            print(f"  [s1] {arch}-{ch}x{nb}: OOM", flush=True); continue
        r = rec(arch, ch, nb, 9, db, a.lrs[0], per); s1.append(r); rows.append(r)
        print(f"  [s1] {arch:<8} ch={ch:<4} nb={nb} {r['params_M']:>6}M RF={str(r['rf_bp']):>6} "
              f"dev={r['dev']:.4f} test={r['test']:.4f}", flush=True)
    top1 = sorted(s1, key=lambda r: -r['dev'])[:a.topk]
    print(f"  [s1] top-{a.topk}: {[(r['arch'], r['params_M']) for r in top1]}", flush=True)

    # ---- STAGE 2: RECEPTIVE FIELD -- sweep kernel size x dilation base ----
    # arXiv:2411.02796: tuning kernel size and dilation is what lets a CNN match genomic FMs.
    # These change RF at ~no parameter cost (dilated 0.68M: RF 249 at dbase=2 vs 2729 at dbase=4).
    s2 = list(top1)
    for r0 in top1:
        combos = ([(k, 1) for k in a.kernels] if r0['arch'] in ("unet", "transformer")
                  else [(k, db) for k in a.kernels for db in a.dbases])
        for k, db in combos:
            if k == 9 and db == (1 if r0['arch'] == "resnet" else 2): continue   # already in s1
            per = run_cell(r0['arch'], r0['ch'], r0['blocks'], a.lrs[0], a.seeds[:1], k=k, dbase=db)
            if per is None: continue
            r = rec(r0['arch'], r0['ch'], r0['blocks'], k, db, a.lrs[0], per); s2.append(r); rows.append(r)
            print(f"  [s2] {r0['arch']:<8} {r0['params_M']:>6}M k={k:<3} d={db} RF={str(r['rf_bp']):>6} "
                  f"{r['params_M']:>6}M dev={r['dev']:.4f} test={r['test']:.4f}", flush=True)
    top2 = sorted(s2, key=lambda r: -r['dev'])[:a.topk2]
    print(f"  [s2] top-{a.topk2}: {[(r['arch'], r['k'], r['dbase'], r['rf_bp']) for r in top2]}", flush=True)

    # ---- STAGE 3: LR grid on the best RF configs, all seeds ----
    for r0 in top2:
        for lr in a.lrs:
            seeds = a.seeds if lr != a.lrs[0] else a.seeds[1:]
            if not seeds: continue
            per = run_cell(r0['arch'], r0['ch'], r0['blocks'], lr, seeds, k=r0['k'], dbase=r0['dbase'])
            if per is None: continue
            if lr == a.lrs[0]: per = r0['runs'] + per
            r = rec(r0['arch'], r0['ch'], r0['blocks'], r0['k'], r0['dbase'], lr, per); rows.append(r)
            print(f"  [s3] {r0['arch']:<8} k={r0['k']:<3} d={r0['dbase']} lr={lr:<7g} "
                  f"dev={r['dev']:.4f} test={r['test']:.4f}", flush=True)

    best_by_dev = max(rows, key=lambda r: r['dev'])
    if a.refit_seeds and len(best_by_dev.get('runs', [])) < a.refit_seeds:
        need = list(range(42, 42 + a.refit_seeds))
        print(f"  [refit] dev-selected cell at {a.refit_seeds} seeds "
              f"({best_by_dev['arch']} k={best_by_dev.get('k')} d={best_by_dev.get('dbase')} lr={best_by_dev.get('lr')})",
              flush=True)
        per = run_cell(best_by_dev['arch'], best_by_dev['ch'], best_by_dev['blocks'],
                       best_by_dev['lr'], need, k=best_by_dev.get('k', 9), dbase=best_by_dev.get('dbase'))
        if per:
            t = [q['test'] for q in per]
            best_by_dev = dict(best_by_dev, runs=per, dev=round(float(np.mean([q['dev'] for q in per])), 4),
                               test=round(float(np.mean(t)), 4),
                               test_sd=round(float(np.std(t, ddof=1)), 4) if len(t) > 1 else None,
                               refit_seeds=a.refit_seeds)
            rows.append(best_by_dev)
            print(f"  [refit] TEST={best_by_dev['test']:.4f} +/-{best_by_dev['test_sd']} over {len(t)} seeds", flush=True)
    res = dict(dataset=a.dataset, task=a.task, metric=metric, n_classes=ncls, seqlen=L,
               n_train=len(tr), n_dev=len(dv), n_test=len(te), seeds=a.seeds,
               train_cap=(a.train_cap if a.dataset in ("hvue_v2", "csvdir", "geneb") else None),
               class_weight=a.class_weight,
               ladder=rows,
               dev_selected=dict(n_seeds=len(best_by_dev.get('runs',[])), test_sd=best_by_dev.get('test_sd'), arch=best_by_dev['arch'], params_M=best_by_dev['params_M'], lr=best_by_dev.get('lr'),
                                 k=best_by_dev.get('k'), dbase=best_by_dev.get('dbase'), rf_bp=best_by_dev.get('rf_bp'),
                                 dev=best_by_dev['dev'], test=best_by_dev['test']),
               oracle_best_test=max(r['test'] for r in rows),
               incumbent=next(({k: r[k] for k in ('arch','params_M','dev','test')} for r in rows
                               if (r['arch'], r['ch'], r['blocks']) == INCUMBENT), None))
    os.makedirs(OUT, exist_ok=True)
    # MUST be _tag, not a second independently-built string. These two were built
    # separately and drifted: _tag carried `__{arch_family}` and the writer did not, so the
    # existence guard tested `..__transformer.json` while the write landed on the CNN's
    # `...json`. The transformer sweeps destroyed the VHDB and HVUE HT hard_70 CNN ladders
    # (json + 6 prediction files each) before this was caught.
    tag = _tag
    json.dump(res, open(f"{OUT}/{tag}.json","w"), indent=2)
    print(f"\n  DEV-SELECTED: {best_by_dev['arch']} @ {best_by_dev['params_M']}M k={best_by_dev.get('k')} "
          f"dbase={best_by_dev.get('dbase')} RF={best_by_dev.get('rf_bp')} lr={best_by_dev.get('lr')} "
          f"-> test {best_by_dev['test']:.4f}")
    print(f"  oracle best test across ladder: {res['oracle_best_test']:.4f} (report as oracle, not as the baseline)")
    print(f"  wrote {OUT}/{tag}.json", flush=True)

if __name__ == "__main__":
    main()
