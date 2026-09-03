"""Evo-1-8k LoRA + frozen probe on GENEB tasks.

WHY THIS EXISTS
---------------
Every erasable-excess measurement in this project used NT-v2-500M, DNABERT-2, GENA-LM or
HyenaDNA. The unlearning pipeline in /home/nvidia/unlearning operates on **Evo-1-8k-base**.
An excess measured on NT-v2 says nothing about whether Evo-1 holds removable capability on
the same task -- so the two proof-of-concept targets have to be re-measured on the model
that will actually be unlearned, or the PoC has no validated target.

Protocol is held identical to every other arm so the numbers are comparable:
  * stratified 15% dev carve at seed 42 (same rows as `geneb_finetune.py`,
    `fair_kmer_sentinel.py` and `capacity_sweep.py`);
  * LR selected on dev only, test scored once;
  * MCC as the metric;
  * multiple seeds at the dev-selected LR.

Two regimes, because they answer different questions and differ by ~0.6 MCC on some
tasks: `probe` (frozen encoder, head only) and `lora` (LoRA adapters + head).

Reuses the Evo machinery from /home/nvidia/glm-locking rather than reimplementing it, so
the LoRA rank/alpha/targets and the final-norm hook match the HVUE runs exactly.
"""
import os, sys, json, argparse, warnings, gc
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
warnings.filterwarnings("ignore")

ROOT = P.LOCK_ROOT
sys.path.insert(0, ROOT)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils import load_evo_model
from src.lora import inject_lora
from scripts.hvue_lora_finetune import (
    HVUEDataset, collate_pad, ClassificationHead, find_norm, forward_with_hook,
    LORA_RANK, LORA_ALPHA, LORA_TARGETS)

GENEB = P.GENEB_DIR
OUT = P.sub("geneb_evo")


def load_task(task, max_len, cap=0):
    """Stratified 15% dev carve at seed 42 -- identical rows to every other arm."""
    tr_all = pd.read_csv(f"{GENEB}/{task}/train.csv").rename(columns={"text": "sequence"})
    te = pd.read_csv(f"{GENEB}/{task}/test.csv").rename(columns={"text": "sequence"})
    if cap and len(tr_all) > cap:
        tr_all = tr_all.sample(n=cap, random_state=0).reset_index(drop=True)
    i_tr, i_dv = train_test_split(np.arange(len(tr_all)), test_size=0.15,
                                  stratify=tr_all.label.values, random_state=42)
    tr, dv = tr_all.iloc[i_tr].reset_index(drop=True), tr_all.iloc[i_dv].reset_index(drop=True)
    for d in (tr, dv, te):
        d["sequence"] = d.sequence.str.upper().str.slice(0, max_len)
    return tr, dv, te


@torch.no_grad()
def evaluate(model, head, loader, norm, device, amp):
    model.eval(); head.eval()
    logits, ys = [], []
    for ids, lab in loader:
        with torch.autocast(device_type="cuda", dtype=amp):
            h = forward_with_hook(model, ids.to(device), norm)
            logits.append(head(h.float()).float().cpu())
        ys.append(lab)
    model.train(); head.train()
    lg = torch.cat(logits).numpy(); y = torch.cat(ys).numpy().astype(int)
    pred = (lg > 0).astype(int)
    try:
        auroc = float(roc_auc_score(y, lg))
    except ValueError:
        auroc = float("nan")
    return float(matthews_corrcoef(y, pred)), auroc, pred, lg


def run(task, regime, lr, seed, tr, dv, te, args, tok, device, amp):
    torch.manual_seed(seed); np.random.seed(seed)
    model, _ = load_evo_model(args.model, device)
    for p in model.parameters():
        p.requires_grad = False
    n_lora = 0
    if regime == "lora":
        n_lora, _ = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA,
                                target_substrings=LORA_TARGETS)
        model.to(device)
    norm = find_norm(model)
    head = ClassificationHead(hidden=args.hidden).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    n_train_params = sum(p.numel() for p in trainable)
    print(f"    regime={regime} lora_modules={n_lora} trainable_params={n_train_params/1e6:.2f}M",
          flush=True)

    dl = dict(collate_fn=collate_pad, num_workers=2, pin_memory=True)
    tr_dl = DataLoader(HVUEDataset(tr, tok), batch_size=args.bs, shuffle=True, **dl)
    dv_dl = DataLoader(HVUEDataset(dv, tok), batch_size=args.bs, shuffle=False, **dl)
    te_dl = DataLoader(HVUEDataset(te, tok), batch_size=args.bs, shuffle=False, **dl)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()
    best = (-2.0, 0, None, None); noimp = 0
    step = 0
    for ep in range(1, args.epochs + 1):
        for i, (ids, lab) in enumerate(tr_dl):
            with torch.autocast(device_type="cuda", dtype=amp):
                h = forward_with_hook(model, ids.to(device), norm)
                loss = lossf(head(h.float()), lab.to(device)) / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0:
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(); step += 1
            if args.max_steps and step >= args.max_steps:
                break
        dv_mcc, dv_auroc, _, _ = evaluate(model, head, dv_dl, norm, device, amp)
        print(f"    ep{ep} dev_mcc={dv_mcc:.4f} dev_auroc={dv_auroc:.4f}", flush=True)
        if dv_mcc > best[0] + args.min_delta:
            te_mcc, te_auroc, pred, lg = evaluate(model, head, te_dl, norm, device, amp)
            best = (dv_mcc, ep, te_mcc, te_auroc); noimp = 0
            if args.predictions_dir:
                os.makedirs(args.predictions_dir, exist_ok=True)
                np.savez_compressed(
                    f"{args.predictions_dir}/{task}__evo1__{regime}__lr{lr:g}__s{seed}__preds.npz",
                    y_true=te.label.values.astype(int), y_pred=pred, logit=lg)
        else:
            noimp += 1
            if ep >= args.min_epochs and noimp >= args.patience:
                break
        if args.max_steps and step >= args.max_steps:
            break

    del model, head
    gc.collect(); torch.cuda.empty_cache()
    return dict(seed=seed, dev_mcc=round(best[0], 4), best_epoch=best[1],
                test_mcc=round(best[2], 4) if best[2] is not None else None,
                test_auroc=round(best[3], 4) if best[3] is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--regimes", nargs="+", default=["probe", "lora"],
                    choices=["probe", "lora"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--model", default="evo-1-8k-base")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--max_len", type=int, default=4096, help="nt truncation (Evo-1-8k)")
    ap.add_argument("--cap_train", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=5e-4)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = no cap")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--predictions_dir", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{OUT}/{a.task}__evo1.json"
    if os.path.exists(tag) and not a.force:
        print(f"SKIP {tag} exists (use --force)"); return

    amp = torch.bfloat16
    _, tok = load_evo_model(a.model, "cpu")          # tokenizer only; model reloaded per run
    gc.collect()
    tr, dv, te = load_task(a.task, a.max_len, a.cap_train)
    ncls = int(pd.concat([tr, dv, te]).label.nunique())
    if ncls != 2:
        raise SystemExit(f"ABORT: {a.task} has {ncls} classes; this harness is binary "
                         "(ClassificationHead emits one logit)")
    print(f"{a.task}: train={len(tr)} dev={len(dv)} test={len(te)} "
          f"maxlen={a.max_len}nt regimes={a.regimes}", flush=True)

    results = {}
    for regime in a.regimes:
        # Stage 1 -- LR search at one seed, selected on dev.
        search = []
        for lr in a.lrs:
            print(f"  [{regime}] lr={lr:g} seed={a.seeds[0]}", flush=True)
            search.append(run(a.task, regime, lr, a.seeds[0], tr, dv, te, a, tok, a.device, amp))
            search[-1]["lr"] = lr
            json.dump({**results, regime: {"search": search}}, open(tag, "w"), indent=2)
        best_lr = max(search, key=lambda r: r["dev_mcc"])["lr"]
        # Stage 2 -- remaining seeds at the dev-selected LR.
        runs = [r for r in search if r["lr"] == best_lr]
        for seed in a.seeds[1:]:
            print(f"  [{regime}] refit lr={best_lr:g} seed={seed}", flush=True)
            r = run(a.task, regime, best_lr, seed, tr, dv, te, a, tok, a.device, amp)
            r["lr"] = best_lr; runs.append(r)
            json.dump({**results, regime: {"search": search, "runs": runs}},
                      open(tag, "w"), indent=2)
        tests = [r["test_mcc"] for r in runs if r["test_mcc"] is not None]
        results[regime] = dict(
            search=search, selected_lr=best_lr, runs=runs,
            test_mcc_mean=round(float(np.mean(tests)), 4) if tests else None,
            test_mcc_sd=round(float(np.std(tests, ddof=1)), 4) if len(tests) > 1 else None,
            n_seeds=len(tests))
        print(f"  [{regime}] SELECTED lr={best_lr:g} -> TEST MCC "
              f"{results[regime]['test_mcc_mean']} +/- {results[regime]['test_mcc_sd']} "
              f"({len(tests)} seeds)", flush=True)

    payload = dict(task=a.task, model="evo-1-8k-base", max_len_nt=a.max_len,
                   n_train=len(tr), n_dev=len(dv), n_test=len(te),
                   dev_carve="stratified 15% seed 42", metric="mcc",
                   lora=dict(rank=LORA_RANK, alpha=LORA_ALPHA, targets=LORA_TARGETS),
                   regimes=results)
    json.dump(payload, open(tag, "w"), indent=2)
    print(f"wrote {tag}")


if __name__ == "__main__":
    main()
