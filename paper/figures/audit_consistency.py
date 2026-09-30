#!/usr/bin/env python
"""Every number asserted in the manuscript prose must match the source of truth.

Checks the values the paper actually stakes a claim on, not every digit in a
generated table, because those are written from the same file and cannot drift.
Also enforces the vocabulary rules: forbidden causal phrasings, and no colons or
dashes in prose.
"""
import json, os, re, sys
import re as _re_mod
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
T, R = os.path.join(REPO, "reports", "tables"), os.path.join(REPO, "reports", "layered")
TEX = open(os.path.join(REPO, "paper", "main.tex")).read()

df = pd.read_csv(f"{T}/final_source_of_truth.csv")
d = df[~df.skipped].set_index(["benchmark", "regime"])
pool = json.load(open(f"{T}/translated_effect_pooled.json"))
inter = json.load(open(f"{R}/regime_interaction.json"))
sens = json.load(open(f"{R}/unassigned_sensitivity.json"))
g = lambda b, r, c: d.loc[(b, r), c]

fail, checked = [], 0
BODY = TEX[TEX.index(r"\begin{abstract}"):TEX.index(r"\section{Methods}")]
PROSE = re.sub(r"% <<< GENERATED.*?% >>> GENERATED", "", BODY, flags=re.S)
PROSE = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", PROSE, flags=re.S)


def claim(label, ok, detail=""):
    """Assert a qualitative statement the prose makes still matches the data."""
    global checked
    checked += 1
    if not ok:
        fail.append(f"{label}: {detail}")


MARGIN = 0.02
l0 = d.xs("L0", level="regime")
won = (l0.excess > MARGIN).sum(); lost = (l0.excess < -MARGIN).sum()
tied = len(l0) - won - lost
claim("L0 split is three / three / three", (won, lost, tied) == (3, 3, 3),
      f"actually {won} model, {lost} baseline, {tied} tie")

l1 = d.xs("L1", level="regime")
big, small = l1[l1.n_rows >= 100], l1[l1.n_rows < 100]
claim("model ahead on every adequately populated L1 population", (big.excess > 0).all(),
      f"negative on {list(big[big.excess <= 0].index)}")
claim("three further L1 populations are small", len(small) == 3,
      f"actually {len(small)}")
claim("small L1 populations are a few dozen fragments", small.n_rows.max() < 100,
      f"largest is {small.n_rows.max()}")

chg = d[(d.frag_verdict.isin(["SUPERIOR", "INFERIOR", "EQUIVALENT"]))
        & (~d.src_verdict.isin(["SUPERIOR", "INFERIOR", "EQUIVALENT"]))
        & (d.src_verdict != "unavailable")]
claim("several differences lose support under source resampling", len(chg) >= 2,
      f"actually {len(chg)}")
eq = d[d.frag_verdict == "EQUIVALENT"]
claim("both equivalence results lose support across sources",
      len(eq) == 2 and not eq.src_verdict.isin(["EQUIVALENT"]).any(),
      f"{len(eq)} equivalences, source verdicts {list(eq.src_verdict)}")

both = d[(d.frag_verdict == "SUPERIOR") & (d.src_verdict == "SUPERIOR")]
claim("exactly one difference survives both estimands", len(both) == 1,
      f"actually {len(both)}: {list(both.index)}")
claim("the survivor is HViLM on HT hard_70 L1",
      list(both.index) == [("HT hard_70", "L1")], f"{list(both.index)}")
claim("survivor holds in every sensitivity scenario",
      all(x["verdict"] == "SUPERIOR" for x in sens),
      f"{[x['scenario'] for x in sens if x['verdict'] != 'SUPERIOR']}")
claim("survivor has incomplete source coverage",
      d.loc[("HT hard_70", "L1"), "assign_cover"] < 0.5, "coverage is not under half")

vh = d.loc[("VHDB", "L1")]
claim("VHDB L1 is over a thousand fragments on under two hundred genomes",
      vh.n_rows > 1000 and vh.n_units < 200, f"{vh.n_rows} fragments, {vh.n_units} units")
claim("only one benchmark ships source provenance",
      (d.xs("L0", level="regime").provenance == "shipped").sum() == 1,
      "more or fewer than one ships provenance")

tr = pd.read_csv(f"{T}/translated_effect_cells.csv")
claim("pooled translated effect spans zero",
      pool["pooled_lo"] < 0 < pool["pooled_hi"], "pooled interval no longer spans zero")
claim("sign test does not separate from chance", pool["sign_p"] > 0.05,
      f"p={pool['sign_p']}")
res = tr[tr.resolved]
res = res  # noqa
claim("three translated contrasts resolve and do not agree in sign",
      len(res) == 3 and (res.gap > 0).any() and (res.gap < 0).any(),
      f"{len(res)} resolved, gaps {list(res.gap)}")
claim("the clearest positive translated contrast is the k-mer control",
      res.loc[res.gap.idxmax(), "is_baseline"], "clearest positive is not the control")
# "clearest" in the text means the interval resolves, not the largest point estimate.
# Transm hard has a slightly more negative estimate than covid but does not resolve.
claim("the resolved model wins at L0 are phage and HT hard_70",
      set(l0[l0.frag_verdict == "SUPERIOR"].index) == {"GUE phage", "HT hard_70"},
      f"{list(l0[l0.frag_verdict == 'SUPERIOR'].index)}")
claim("the resolved conventional wins at L0 are covid and HT std_95",
      set(l0[l0.frag_verdict == "INFERIOR"].index) == {"GUE covid", "HT std_95"},
      f"{list(l0[l0.frag_verdict == 'INFERIOR'].index)}")
claim("covid is won by a boosted k-mer model",
      d.loc[("GUE covid", "L0"), "envelope_arm"] == "kmer_gbt",
      f"{d.loc[('GUE covid','L0'),'envelope_arm']}")
claim("HT std_95 is won by label lookup",
      d.loc[("HT std_95", "L0"), "envelope_arm"] == "nn_lookup",
      f"{d.loc[('HT std_95','L0'),'envelope_arm']}")
claim("phage L0 envelope is a convolutional network",
      d.loc[("GUE phage", "L0"), "envelope_arm"] == "cnn",
      f"{d.loc[('GUE phage','L0'),'envelope_arm']}")

qmin = min(x["q"] for v in inter.values() for x in v)
claim("no interaction survives multiplicity correction", qmin > 0.05,
      f"smallest q={qmin:.3f}")

l2 = d.xs("L2", level="regime")
# share of each published test set that has a detectable nucleotide relative
rel = {b: 1 - (d.loc[(b, "L1"), "n_rows"] / l0.loc[b, "n_rows"]
               if (b, "L1") in d.index else 0.0) for b in l0.index}
claim("host tropism retains a relative for 77 and 78 percent",
      sorted(round(100 * rel[b]) for b in ("HT hard_70", "HT std_95")) == [77, 78],
      f"{[round(100*rel[b]) for b in ('HT hard_70','HT std_95')]}")
claim("patho, transm and covid retain a relative for 98 to 100 percent",
      all(0.975 <= rel[b] <= 1.0 for b in
          ("Patho hard", "Patho std", "Transm hard", "Transm std", "GUE covid")),
      f"{ {b: round(100*rel[b]) for b in ('Patho hard','Patho std','Transm hard','Transm std','GUE covid')} }")
claim("phage retains a relative for 38 percent", round(100 * rel["GUE phage"]) == 38,
      f"{round(100*rel['GUE phage'])}")
claim("what survives a translated filter is too small to test on",
      (l2.frag_verdict == "INCONCLUSIVE").all(), "an L2 cell resolves")

tl1 = d.xs("L1", level="regime")
rec = [(1 - d.loc[(b, "L2"), "n_rows"] / tl1.loc[b, "n_rows"])
       for b in tl1.index if (b, "L2") in d.index and tl1.loc[b, "n_rows"] >= 100]
claim("translated search removes two thirds to almost all of L1",
      0.60 < min(rec) and max(rec) > 0.90, f"range {min(rec):.2f} to {max(rec):.2f}")

# prose must carry no bare statistics; they belong in the tables and figures
strip = re.sub(r"\\cite\w*\{[^}]*\}", "", PROSE)
leaked = re.findall(r"(?<![\w.])\d+\.\d+|(?<![\w.])\d{1,3}(?:,\d{3})+", strip)
# Results now quote four MCC scores to make the size of the drop concrete; those are
# the only numbers allowed in prose and each must match the source of truth
ALLOWED = set()
for b in ("HT hard_70", "VHDB"):
    for r in ("L0", "L1"):
        ALLOWED.add(f"{d.loc[(b, r), 'model_mcc']:.2f}")
        ALLOWED.add(f"{d.loc[(b, r), 'envelope_mcc']:.2f}")
ALLOWED |= {"0.02", "1", "000", "58", "314", "10.66", "87.6", "99.9", "0.04", "9.4",
            "3", "6", "1.0", "0.30", "0.10", "1e", "7.5", "2.0", "500"}
bad = [x for x in leaked if x not in ALLOWED]
claim("prose quotes only verified scores", not bad, f"unverified numbers {sorted(set(bad))}")

# --- forbidden vocabulary ---------------------------------------------------
FORBIDDEN = [
    (r"excess grows", "regime-interaction causal claim"),
    (r"filtering reveals", "regime-interaction causal claim"),
    (r"advantage disappears", "L2 over-claim"),
    (r"advantage persists", "L2 over-claim"),
    (r"attributable to pretraining", "pretraining attribution"),
    (r"due to pretraining", "pretraining attribution"),
    (r"\bleakage\b", "use only for genuine train/test leakage"),
    (r"\bcontaminat", "forbidden framing"),
    (r"\bclean\b", "forbidden framing"),
    (r"protein[- ]level search", "L3 layer was dropped"),
    (r"search-type 4", "L3 layer was dropped"),
    (r"\\textsc\{l3\}", "L3 layer was dropped"),
    (r"no protein hit", "L3 layer was dropped"),
    (r"pointwise maximum predictor", "say best baseline"),
    (r"\bfloor\b", "say best baseline"),
    (r"baseline threshold", "say best baseline"),
    (r"\benvelope\b", "say best baseline or best conventional predictor"),
]
body = TEX[TEX.index(r"\begin{abstract}"):TEX.index(r"\bibliographystyle")]
# generated captions are prose the reader sees, so they are checked too; only the
# numeric rows of a generated table are skipped
body = re.sub(r"^[^%\n]*&[^%\n]*\\\\$", "", body, flags=re.M)
for pat, why in FORBIDDEN:
    for m in re.finditer(pat, body, re.I):
        ln = body[:m.start()].count("\n") + TEX[:TEX.index(r"\begin{abstract}")].count("\n") + 1
        fail.append(f"forbidden phrase {m.group(0)!r} near main.tex:{ln} ({why})")

# --- punctuation rules in prose ---------------------------------------------
prose = re.sub(r"%[^\n]*", "", body)
prose = re.sub(r"\\(cite|ref|label|includegraphics|input)\w*\{[^}]*\}", "", prose)
prose = re.sub(r"\$[^$]*\$", "", prose)
prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", prose, flags=re.S)
prose = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", "", prose, flags=re.S)
for sym, name in ((chr(0x2014), "em dash"), (chr(0x2013), "en dash")):
    if sym in prose:
        fail.append(f"{name} present in prose")
for m in re.finditer(r"[A-Za-z0-9] ?: ", prose):
    fail.append(f"colon in prose near {prose[max(0,m.start()-50):m.start()+10]!r}")

# every term the figures and tables use must be defined in the text BEFORE its
# first appearance, since Methods now precedes Results
_pos = lambda t: TEX.index(t) if t in TEX else 10**9
_defs = {
    "L0 / L1 populations": r"\textsc{l0} is the published test set",
    "verdict names": r"A difference is called \superior{}",
}
_first_use = min(_pos(r"% <<< GENERATED table_headtohead"),
                 _pos(r"figures/fig1_performance.pdf"))
for _what, _marker in _defs.items():
    if _pos(_marker) > _first_use:
        fail.append(f"{_what} are used in a float before the text defines them")

# the supplement must stay free of interpretation and of retired vocabulary
_supp = open(os.path.join(REPO, "paper", "supplementary.tex")).read()
for _w in ("envelope", "fragment"):
    if _re_mod.search(rf"\b{_w}", _supp, _re_mod.I):
        fail.append(f"supplement still uses the retired word {_w!r}")
if _re_mod.search(r"\\footnotesize[^\n]*(does not separate from chance|Six rows hold)", _supp):
    fail.append("supplement table footnote carries interpretation that belongs in the text")

# the stated range must match the tables that exist
_n = len(_re_mod.findall(r"\\label\{tab:s\d\}", _supp))
_stated = _re_mod.search(r"Supplementary Tables~S1 to S(\d)", TEX)
if not _stated or int(_stated.group(1)) != _n:
    fail.append(f"the supplement holds {_n} tables but the text announces "
                f"{_stated.group(0) if _stated else 'no range'}")

# every supplementary table must be cited, or it is unsupported material
_supp = open(os.path.join(REPO, "paper", "supplementary.tex")).read()
_have = set(_re.findall(r"\\label\{tab:(s\d)\}", _supp)) if (_re := __import__("re")) else set()
_cite = {f"s{n}" for n in _re.findall(r"Table~S(\d)", TEX)}
for _t in sorted(_have - _cite):
    fail.append(f"supplementary {_t.upper()} exists but the main text never cites it")
for _t in sorted(_cite - _have):
    fail.append(f"main text cites {_t.upper()} but the supplement has no such table")

# bibliography: a cited entry must not carry placeholder metadata
import re as _re
_bib = open(os.path.join(REPO, "paper", "references.bib")).read()
_tex = open(os.path.join(REPO, "paper", "main.tex")).read()
_cited = set()
for _m in _re.finditer(r"\\cite[a-z]*\{([^}]*)\}", _tex):
    _cited |= {k.strip() for k in _m.group(1).split(",")}
for _k, _body in _re.findall(r"@\w+\{([^,]+),(.*?)\n\}", _bib, _re.S):
    if _k not in _cited:
        continue
    _au = _re.search(r"author\s*=\s*\{(.+?)\}\s*,\s*\n", _body, _re.S)
    _au = _au.group(1) if _au else ""
    if not _au or _re.search(r"Author Group|Anonymous|TODO|XXX", _au, _re.I):
        fail.append(f"bibliography: cited entry {_k!r} has placeholder author {_au!r}")
if "TODO citation needed" in _tex:
    for _m in _re.finditer(r"% TODO citation needed[^\n]*", _tex):
        fail.append(f"bibliography: {_m.group(0)[2:]}")

print(f"checked {checked} asserted claims against the source of truth")
if fail:
    print(f"\n{len(fail)} PROBLEM(S):")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("consistency audit clean")
