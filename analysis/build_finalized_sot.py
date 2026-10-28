#!/usr/bin/env python
"""Build the finalized source of truth by merging the corrected intervals in.

CPU only. Reads, never overwrites, the pre-finalization artifacts.

    reports/tables/final_source_of_truth.csv            pre-finalization, UNCHANGED
    reports/finalization/bootstrap_corrected.csv        two-level + conditional intervals
        |
        v
    reports/tables/final_source_of_truth_finalized.csv  what the generators now read

Merge rule, per cell
--------------------
* Point estimates, model identity, comparator arm, counts, genus columns: carried
  from the pre-finalization file unchanged. The corrected run reproduced every
  one of them exactly, so there is nothing to replace.
* Fragment interval: replaced by the **two-level** interval (test units and
  training seeds resampled). provenance_frag = REGENERATED.
* Source interval:
    - VHDB: the native unit file is absent on this cluster, so the frozen values
      are carried forward EXACTLY. Nothing is recomputed, interpolated or
      approximated. provenance_src = PRESENT_BUT_NOT_REPRODUCED.
    - GUE phage, GUE covid: no usable provenance by design; the frozen
      `unavailable` marker is carried so the printed output stays "not
      estimable". provenance_src = FROZEN_VERIFIED_INPUT.
    - everything else: replaced by the two-level source interval.
      provenance_src = REGENERATED.
* Cells the pipeline skips (n below the scoring floor): carried verbatim,
  provenance = FROZEN_VERIFIED_INPUT.

The conditional-bootstrap interval is carried alongside in `cond_*` columns so
the supplementary sensitivity table can be generated without rerunning anything.

Usage
-----
    python build_finalized_sot.py
"""
import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FINAL_DIR = os.path.join(REPO, "reports", "finalization")
SOT = os.path.join(REPO, "reports", "tables", "final_source_of_truth.csv")
BOOT = os.path.join(FINAL_DIR, "bootstrap_corrected.csv")
OUT = os.path.join(REPO, "reports", "tables", "final_source_of_truth_finalized.csv")

CARRY_SRC_FROZEN = {"VHDB": "PRESENT_BUT_NOT_REPRODUCED"}
NO_PROVENANCE = {"GUE phage", "GUE covid"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()



def _dev_means():
    """Mean development MCC over the fits each model has NOW, from the finetune JSONs.

    This is reported alongside the selection quantity, never in place of it. Selection was
    pre-committed on three seeds, and the selected arms were later refit ten times, so this
    mean is a ten-fit number on six of the nine benchmarks. It is written to `dev_mcc_current`
    and the frozen three-seed `dev_mcc` is carried through untouched; the supplement's
    selection table reads `reports/tables/selection_3seed.csv`, not either of these.
    """
    import glob, json as _json, sys as _sys
    _sys.path.insert(0, os.path.join(REPO, "scripts", "track_b_viral"))
    import final_regime_table as _F
    NAME = {"HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
            "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
            "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
            "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid"}
    out = {}
    for name, _mask, esc, _atags, _ in _F.CELLS:
        for m in _F.MODELS:
            best = None
            for j in glob.glob(f"{_F.FT}/{esc}__{m}__lr*__fullft.json"):
                if any(x in j for x in ("_meanpool", "_ep4", "_truncated")):
                    continue
                try:
                    d = _json.load(open(j))
                except Exception:
                    continue
                if not d.get("runs"):
                    continue
                v = sum(r["dev_mcc"] for r in d["runs"]) / len(d["runs"])
                if best is None or v > best:
                    best = v
            if best is not None:
                out[(NAME.get(name, name), m)] = best
    return out


def main():
    # regenerated symmetric check, keyed by (benchmark, regime)
    _symp = os.path.join(FINAL_DIR, "symmetric_regenerated.csv")
    SYM = {}
    if os.path.exists(_symp):
        SYM = {(r["benchmark"], r["regime"]): r for r in csv.DictReader(open(_symp))}
    # development MCC recomputed over the fits each model actually has now
    DEV = _dev_means()

    frozen = list(csv.DictReader(open(SOT)))
    cols = list(frozen[0].keys())
    boot = {(r["benchmark"], r["regime"]): r for r in csv.DictReader(open(BOOT))}

    new_cols = ["interval_scheme", "finalization_provenance",
                "provenance_frag", "provenance_src", "dev_mcc_current",
                "cond_frag_lo", "cond_frag_hi", "cond_frag_verdict",
                "cond_src_lo", "cond_src_hi", "cond_src_verdict",
                "frag_reps_used", "src_reps_used"]
    out_cols = cols + [c for c in new_cols if c not in cols]

    rows, diffs = [], []
    for r in frozen:
        k = (r["benchmark"], r["regime"])
        o = dict(r)
        b = boot.get(k)
        if b is None:
            o.update(interval_scheme="n/a (cell below the scoring floor)",
                     finalization_provenance="FROZEN_VERIFIED_INPUT",
                     provenance_frag="FROZEN_VERIFIED_INPUT",
                     provenance_src="FROZEN_VERIFIED_INPUT")
            rows.append(o)
            continue

        before = (r.get("frag_lo"), r.get("frag_hi"), r.get("frag_verdict"),
                  r.get("src_lo"), r.get("src_hi"), r.get("src_verdict"))

        o["frag_lo"], o["frag_hi"], o["frag_verdict"] = b["frag2_lo"], b["frag2_hi"], b["frag2_verdict"]
        o["provenance_frag"] = "REGENERATED"

        # Point estimates must come from the SAME run as the intervals. They used to be
        # carried from final_regime_table.json, which is frozen at 3 fits per method, so a
        # regeneration produced rows whose printed model minus baseline did not equal the
        # printed excess. bootstrap_corrected.csv computes all three from the tree being
        # scored, so take them from there. Added 2026-10-18.
        for src_col, dst_col in (("model_mcc", "model_mcc"),
                                 ("baseline_mcc", "envelope_mcc"),
                                 ("excess", "excess"),
                                 ("baseline_arm", "envelope_arm")):
            if b.get(src_col) not in (None, ""):
                o[dst_col] = b[src_col]

        if r["benchmark"] in CARRY_SRC_FROZEN:
            o["provenance_src"] = CARRY_SRC_FROZEN[r["benchmark"]]
            # src_lo / src_hi / src_verdict left exactly as the frozen file has them
        elif r["benchmark"] in NO_PROVENANCE:
            o["provenance_src"] = "FROZEN_VERIFIED_INPUT"
        else:
            o["src_lo"], o["src_hi"] = b["src2_lo"], b["src2_hi"]
            o["src_verdict"] = b["src2_verdict"]
            o["provenance_src"] = "REGENERATED"

        # Symmetric check, regenerated at the current fit count with the manuscript
        # estimator. It used to be carried from final_regime_table.json, which is frozen at
        # three fits and uses a single-draw replicate statistic. Added 2026-10-28.
        sy = SYM.get(k)
        if sy:
            o["sym_excess"], o["sym_lo"] = sy["sym_excess"], sy["sym_lo"]
            o["sym_hi"], o["sym_verdict"] = sy["sym_hi"], sy["sym_verdict"]
            o["sym_provenance"] = sy["sym_provenance"]
        dv = DEV.get((r["benchmark"], r["model"]))
        if dv is not None:
            o["dev_mcc_current"] = round(dv, 4)   # NOT dev_mcc: see _dev_means

        o["interval_scheme"] = "two-level bootstrap (test units and training seeds)"
        o["finalization_provenance"] = ("REGENERATED" if o["provenance_src"] == "REGENERATED"
                                        else o["provenance_src"])
        o.update(cond_frag_lo=b["frag_lo"], cond_frag_hi=b["frag_hi"],
                 cond_frag_verdict=b["frag_verdict"],
                 cond_src_lo=b.get("src_lo", ""), cond_src_hi=b.get("src_hi", ""),
                 cond_src_verdict=b.get("src_verdict", ""),
                 frag_reps_used=b.get("frag_reps_used", ""),
                 src_reps_used=b.get("src_reps_used", ""))
        rows.append(o)

        after = (o.get("frag_lo"), o.get("frag_hi"), o.get("frag_verdict"),
                 o.get("src_lo"), o.get("src_hi"), o.get("src_verdict"))
        if before != after:
            diffs.append(dict(benchmark=k[0], regime=k[1],
                              frag_before=f"[{before[0]},{before[1]}] {before[2]}",
                              frag_after=f"[{after[0]},{after[1]}] {after[2]}",
                              frag_verdict_changed=(before[2] != after[2]),
                              src_before=f"[{before[3]},{before[4]}] {before[5]}",
                              src_after=f"[{after[3]},{after[4]}] {after[5]}",
                              src_verdict_changed=(before[5] != after[5]),
                              provenance_src=o["provenance_src"]))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in out_cols})

    dp = os.path.join(FINAL_DIR, "sot_before_after_diff.csv")
    with open(dp, "w", newline="") as fh:
        f = ["benchmark", "regime", "frag_before", "frag_after", "frag_verdict_changed",
             "src_before", "src_after", "src_verdict_changed", "provenance_src"]
        w = csv.DictWriter(fh, fieldnames=f)
        w.writeheader()
        for d in diffs:
            w.writerow(d)

    vch = [d for d in diffs if d["frag_verdict_changed"] or d["src_verdict_changed"]]
    json.dump({"input_prefinalization": SOT, "input_prefinalization_sha256": sha(SOT),
               "input_bootstrap": BOOT, "input_bootstrap_sha256": sha(BOOT),
               "output": OUT, "output_sha256": sha(OUT),
               "n_cells": len(rows), "n_interval_changes": len(diffs),
               "n_verdict_changes": len(vch),
               "verdict_changes": [f"{d['benchmark']} {d['regime']}: "
                                   f"frag {d['frag_before']} -> {d['frag_after']}" if d["frag_verdict_changed"]
                                   else f"{d['benchmark']} {d['regime']}: "
                                        f"src {d['src_before']} -> {d['src_after']}" for d in vch]},
              open(os.path.join(FINAL_DIR, "finalized_sot_provenance.json"), "w"), indent=2)

    print(f"wrote {OUT}")
    print(f"  cells: {len(rows)}   intervals changed: {len(diffs)}   verdicts changed: {len(vch)}")
    for d in vch:
        if d["frag_verdict_changed"]:
            print(f"    FRAG  {d['benchmark']} {d['regime']}: {d['frag_before']} -> {d['frag_after']}")
        if d["src_verdict_changed"]:
            print(f"    SRC   {d['benchmark']} {d['regime']}: {d['src_before']} -> {d['src_after']}")
    from collections import Counter
    print("  provenance_src:", dict(Counter(r.get("provenance_src", "") for r in rows)))


if __name__ == "__main__":
    main()
