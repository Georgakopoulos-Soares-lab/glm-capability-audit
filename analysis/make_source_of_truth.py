#!/usr/bin/env python
"""Consolidate every frozen analysis into one file. Nothing downstream reads anything else.

Every figure and table generator in this paper reads final_source_of_truth.csv, so a
number cannot appear in the manuscript without appearing here first. The analyses
themselves are frozen; this only joins them.
"""
import json, os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
R, T = os.path.join(REPO, "reports", "layered"), os.path.join(REPO, "reports", "tables")


def require(path, produced_by, what):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise SystemExit(f"missing or empty input: {path}\n  holds: {what}\n"
                         f"  produce it with: {produced_by}")
    return path


# canonical short names, used everywhere downstream
NAME = {
    "HVUE host tropism hard_70": "HT hard_70", "HVUE host tropism std_95": "HT std_95",
    "HVUE pathogenicity hard": "Patho hard", "HVUE pathogenicity std": "Patho std",
    "HVUE transmissibility hard": "Transm hard", "HVUE transmissibility std": "Transm std",
    "VHDB within family": "VHDB", "GUE phage": "GUE phage", "GUE virus_covid": "GUE covid",
}
GENUS_KEY = {
    "HT hard_70": "HVUE_HT_hard_70", "HT std_95": "HVUE_HT_std_95",
    "Patho hard": "HVUE_Patho_hard", "Patho std": "HVUE_Patho_std",
    "Transm hard": "HVUE_Transm_hard", "Transm std": "HVUE_Transm_std",
    "VHDB": "VHDB_within_family", "GUE phage": "GUE_phage", "GUE covid": "GUE_covid",
}
# an L1/L2 population this small cannot carry a headline; flagged, never silently dropped
ADEQUATE_MIN = 100


def main():
    reg = json.load(open(require(f"{R}/final_regime_table.json",
        "scripts/track_b_viral/final_regime_table.py", "dual-estimand results")))
    gen = json.load(open(require(f"{T}/genus_concentration.json",
        "analysis/make_genus_concentration.py", "per-stratum genus composition")))
    prov = json.load(open(require(f"{T}/inference_provenance.json",
        "analysis/make_inference_provenance.py", "unit provenance and coverage")))
    units = {(r["benchmark"], r["stratum"]): r for r in prov["units"]}

    rows = []
    for r in reg:
        b = NAME[r["benchmark"]]
        rec = dict(benchmark=b, regime=r["regime"], model=r["model"],
                   n_rows=r["n_rows"], skipped=bool(r.get("skipped")))
        if r.get("skipped"):
            rows.append(rec); continue
        rec.update(
            model_mcc=r["model_mcc"], dev_mcc=r["dev_mcc"],
            envelope_mcc=r["envelope"], envelope_arm=r["envelope_arm"], excess=r["excess"],
            frag_lo=r["frag_lo"], frag_hi=r["frag_hi"], frag_verdict=r["frag_verdict"],
            src_lo=r["src_lo"], src_hi=r["src_hi"], src_verdict=r["src_verdict"],
            n_units=r["n_units"], assign_cover=r["assign_cover"],
            n_real_units=r["n_real_units"], provenance=r["provenance"],
            nn_available=r["nn_available"],
            sym_excess=r["sym"]["excess"], sym_lo=r["sym"]["lo"],
            sym_hi=r["sym"]["hi"], sym_verdict=r["sym"]["verdict"])
        g = gen.get(GENUS_KEY[b], {}).get("strata", {}).get(r["regime"])
        if g:
            cov = (g["positives_with_genus"] / g["positives"]) if g["positives"] else None
            rec.update(positives=g["positives"], positives_with_genus=g["positives_with_genus"],
                       genus_cover=(round(cov, 4) if cov is not None else None),
                       n_positive_genera=g["n_positive_genera"],
                       top_genus_share=g["top_positive_share"],
                       genus_share_readable=not g["share_underpowered"],
                       genus_labels=gen[GENUS_KEY[b]]["labels"])
        rows.append(rec)

    df = pd.DataFrame(rows)
    # adequacy is about evaluation sample, and is reported, never used to hide a cell
    df["adequate_population"] = df.n_rows >= ADEQUATE_MIN
    # L0 fractions
    l0 = df[df.regime == "L0"].set_index("benchmark").n_rows
    df["pct_of_L0"] = [round(100 * r.n_rows / l0[r.benchmark], 2) for r in df.itertuples()]
    out = os.path.join(REPO, "reports", "tables", "final_source_of_truth.csv")
    df.to_csv(out, index=False)

    # translated-relative direct test, parsed from its frozen text output
    tr = open(require(f"{R}/translated_relatedness_effect.txt",
        "scripts/track_b_viral/translated_relatedness_effect.py",
        "N-/T+ vs N-/T- contrasts")).read()
    cells, bench = [], None
    for ln in tr.splitlines():
        m = re.match(r"^(\S+)\s+with a translated relative n=([\d,]+)\s+without n=([\d,]+)", ln)
        if m:
            bench = m.group(1)
            continue
        m = re.match(r"^\s+(\w+)\s+([+-][\d.]+) vs ([+-][\d.]+)\s+gap\s+([+-][\d.]+)\s+"
                     r"\[([+-][\d.]+),([+-][\d.]+)\]\s+(\S+)(\s+baseline)?", ln)
        if m and bench:
            cells.append(dict(benchmark=bench, arm=m.group(1), with_rel=float(m.group(2)),
                              without_rel=float(m.group(3)), gap=float(m.group(4)),
                              lo=float(m.group(5)), hi=float(m.group(6)),
                              resolved=(m.group(7) == "resolved"),
                              is_baseline=bool(m.group(8))))
    pooled = re.search(r"pooled mean gap ([+-][\d.]+)\s+\[([+-][\d.]+),([+-][\d.]+)\]", tr)
    sign = re.search(r"gap positive on (\d+) of (\d+).*?sign test p = ([\d.]+)", tr, re.S)
    td = pd.DataFrame(cells)
    td.to_csv(f"{T}/translated_effect_cells.csv", index=False)
    meta = dict(pooled_gap=float(pooled.group(1)), pooled_lo=float(pooled.group(2)),
                pooled_hi=float(pooled.group(3)), n_positive=int(sign.group(1)),
                n_cells=int(sign.group(2)), sign_p=float(sign.group(3)))
    json.dump(meta, open(f"{T}/translated_effect_pooled.json", "w"), indent=2)

    print(f"wrote {out}   {len(df)} benchmark-regime rows")
    print(f"      {T}/translated_effect_cells.csv   {len(td)} contrasts")
    print(f"      {T}/translated_effect_pooled.json  pooled {meta['pooled_gap']:+.3f} "
          f"[{meta['pooled_lo']:+.3f},{meta['pooled_hi']:+.3f}]")
    sc = df[~df.skipped]
    print(f"\n  fragment-resolved cells: {(sc.frag_verdict.isin(['SUPERIOR','INFERIOR'])).sum()}"
          f"  equivalence: {(sc.frag_verdict=='EQUIVALENT').sum()}")
    print(f"  source-resolved cells:   "
          f"{(sc.src_verdict.isin(['SUPERIOR','INFERIOR'])).sum()}")
    print(f"  source not estimable:    {(sc.src_verdict=='unavailable').sum()}")


if __name__ == "__main__":
    main()
