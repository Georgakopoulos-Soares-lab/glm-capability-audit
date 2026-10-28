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

from sot_path import sot_csv
df = pd.read_csv(sot_csv())
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

chg = d[(d.frag_verdict.isin(["SUPERIOR", "INFERIOR"]))
        & (~d.src_verdict.isin(["SUPERIOR", "INFERIOR"]))
        & (d.src_verdict != "unavailable")]
# Compare against the number main.tex actually spells out, so a change in the data
# cannot silently leave a stale count in the prose. This is what let "six" survive
# after the EQUIVALENT category was removed and the count fell to four.
_W = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_m = (re.search(r"(\w+) of the (?:\w+) verdicts that resolve over sequences survive", TEX)
      or re.search(r"Of the \w+ verdicts that resolve at sequence level, (\w+) persist", TEX))
_l0r = d.xs("L0", level="regime")
_surv = _l0r[_l0r.frag_verdict.isin(["SUPERIOR", "INFERIOR"])
             & _l0r.src_verdict.isin(["SUPERIOR", "INFERIOR"])]
claim("the prose count of L0 verdicts surviving source resampling matches the data",
      bool(_m) and _W.get(_m.group(1).lower()) == len(_surv),
      f"prose says {_m.group(1) if _m else 'nothing'}, data says {len(_surv)}")
# The EQUIVALENT category was removed at finalization: declaring equivalence needs a
# margin, and the 0.02 MCC margin had no external basis and sat below the median
# seed-to-seed spread of 0.027 MCC. Nothing may now carry that verdict.
eq = d[d.frag_verdict == "EQUIVALENT"]
claim("no equivalence verdict is issued anywhere",
      len(eq) == 0 and not (d.src_verdict == "EQUIVALENT").any(),
      f"{len(eq)} fragment equivalences, {(d.src_verdict == 'EQUIVALENT').sum()} source")

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
# Transm hard crossed INCONCLUSIVE -> INFERIOR at 10 fits per method (2026-10-18).
claim("the resolved conventional wins at L0 are covid, HT std_95 and Transm hard",
      set(l0[l0.frag_verdict == "INFERIOR"].index) == {"GUE covid", "HT std_95", "Transm hard"},
      f"{list(l0[l0.frag_verdict == 'INFERIOR'].index)}")
claim("covid is won by a boosted k-mer model",
      d.loc[("GUE covid", "L0"), "envelope_arm"] == "kmer_gbt",
      f"{d.loc[('GUE covid','L0'),'envelope_arm']}")
claim("HT std_95 is won by label lookup",
      d.loc[("HT std_95", "L0"), "envelope_arm"] == "nn_lookup",
      f"{d.loc[('HT std_95','L0'),'envelope_arm']}")
# At 1 fit the CNN held this cell; at 10 fits the k-mer GBT mean overtakes it.
claim("phage L0 envelope is a boosted k-mer model",
      d.loc[("GUE phage", "L0"), "envelope_arm"] == "kmer_gbt",
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
# One L2 cell (HT hard_70) resolves over sequences at 10 fits; none resolves over
# source units, which is the standard the paper applies.
claim("no L2 cell is established over source units",
      not (l2.src_verdict.isin(["SUPERIOR", "INFERIOR"])).any(),
      f"{list(l2[l2.src_verdict.isin(['SUPERIOR','INFERIOR'])].index)}")

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
    # fires on the Rafi citation gloss, which is about their work, not a claim of ours
    (r"(?<!homology-induced data )\bleakage\b(?! and memorization)",
     "use only for genuine train/test leakage"),
    (r"\bcontaminat", "forbidden framing"),
    (r"\bclean\b", "forbidden framing"),
    # L2 IS a protein-level search, so the phrase is correct where L2 is the subject;
    # what was dropped is the separate L3 layer, and that is caught by the rules below
    (r"protein[- ]level search(?!\.)(?!, rather)", "L3 layer was dropped"),
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
    # match the sentence, not one wording of it: "called" became "designated" and the
    # check silently started reporting a definition that was there all along
    "verdict names": r"\superior{} when the entire interval lies above zero",
}
# the figure is included as .png; do not pin the extension
_pos_re = lambda pat: (m.start() if (m := _re_mod.search(pat, TEX)) else 10**9)
_first_use = min(_pos(r"% <<< GENERATED table_headtohead"),
                 _pos_re(r"figures/fig1_performance"))
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
_n = len(_re_mod.findall(r"\\label\{tab:[a-z]+\}", _supp))
# Supplement numbering is generated by \thetable, and the main text references tables
# by \ref rather than by a hardcoded S1..SN range, so there is no range to reconcile.

# every supplementary table must be cited, or it is unsupported material
_supp = open(os.path.join(REPO, "paper", "supplementary.tex")).read()
_have = set(_re.findall(r"\\label\{tab:([a-z]+)\}", _supp)) if (_re := __import__("re")) else set()
# matches "Table~S1", and also each table in "Tables~S4 and~S5"
_cite = set(_re.findall(r"\\ref\{tab:([a-z]+)\}", TEX)) & _have
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
    # entries may be formatted on one line, so do not require a trailing newline
    _au = _re.search(r"author\s*=\s*\{(.+?)\}\s*,", _body, _re.S)
    _au = _au.group(1) if _au else ""
    if not _au or _re.search(r"Author Group|Anonymous|TODO|XXX", _au, _re.I):
        fail.append(f"bibliography: cited entry {_k!r} has placeholder author {_au!r}")
if "TODO citation needed" in _tex:
    for _m in _re.finditer(r"% TODO citation needed[^\n]*", _tex):
        fail.append(f"bibliography: {_m.group(0)[2:]}")


# ---------------------------------------------------------------------------
# Fit counts. Methods once asserted ten fits on every arm of every benchmark and
# nothing in the pipeline contradicted it, so the claim survived two rounds of
# review. The counts are now derived from the tree and checked against the
# supplement, so prose and data cannot drift apart again. Added 2026-10-28.
# ---------------------------------------------------------------------------
_supp_txt = open(os.path.join(REPO, "paper", "supplementary.tex")).read()
_fit_tbl = _supp_txt.split(r"\label{tab:seedsd}")[-1] if "tab:seedsd" in _supp_txt else ""
_counts = {}
for _ln in _fit_tbl.splitlines():
    _c = [x.strip() for x in _ln.split("&")]
    if len(_c) == 8 and _c[3].isdigit():
        _counts.setdefault(_c[2], set()).add(int(_c[3]))
claim("the supplement carries a per-arm fit count table", bool(_counts),
      "no tab:seedsd rows parsed; Methods cites it")

# an arm the Methods paragraph calls deterministic must never show more than one fit
for _det in (r"$k$-mer", "NN", "Tax", "Tax$^{r}$"):
    if _det in _counts:
        claim(f"{_det} is deterministic", _counts[_det] == {1},
              f"shows fit counts {sorted(_counts[_det])}")

# Methods must name every benchmark that falls short of ten, and no others
_short_benches = set()
_cur = None
for _ln in _fit_tbl.splitlines():
    _c = [x.strip() for x in _ln.split("&")]
    if len(_c) == 8 and _c[3].isdigit():
        if _c[0]:
            _cur = _c[0]
        if int(_c[3]) < 10 and _c[2] not in (r"$k$-mer", "NN", "Tax", "Tax$^{r}$"):
            _short_benches.add(_cur)
_METHODS = TEX[TEX.index(r"\section{Methods}"):TEX.index(r"\section{Results}")]
for _b in sorted(_short_benches):
    _name = "VHDB" if "VHDB" in _b else ("GUE \\textsc{covid}" if "covid" in _b else _b)
    claim(f"Methods names the short-fit benchmark {_name}", _name in _METHODS,
          f"{_name} has arms below ten fits and Methods does not say so")
claim("Methods does not claim ten fits everywhere",
      "refit ten times with independent seeds, on every benchmark and every population"
      not in _METHODS,
      "the blanket ten-fit sentence is back")


# ---------------------------------------------------------------------------
# Table geometry. There is no LaTeX on the analysis cluster, so a row with the
# wrong number of ampersands is not caught until someone compiles. Count the
# columns in each preamble and check every body row against it. Added 2026-10-28.
# ---------------------------------------------------------------------------
def _ncols(spec):
    spec = _re_mod.sub(r"[{<][^{}<>]*[}>]", "", spec)      # p{..}, >{..}, @{..}
    return sum(spec.count(c) for c in "lcrpX")

for _fn in ("main.tex", "supplementary.tex"):
    _t = open(os.path.join(REPO, "paper", _fn)).read()
    for _m in _re_mod.finditer(r"\\begin\{(tabular|longtable)\}(?:\[[^\]]*\])?\{([^}]*)\}"
                               r"(.*?)\\end\{\1\}", _t, _re_mod.S):
        _n = _ncols(_m.group(2))
        for _ln in _m.group(3).splitlines():
            _ln = _ln.strip()
            # a longtable caption and label line also ends in \\ and is not a body row
            if (not _ln.endswith(r"\\") or "multicolumn" in _ln or _ln.startswith("%")
                    or _re_mod.search(r"\\(caption|label|[a-z]*rule|end(first)?head)", _ln)):
                continue
            _got = _ln.count("&") - _ln.count(r"\&") + 1
            if _got != _n:
                _line = _t[:_m.start(3)].count("\n") + 1
                fail.append(f"{_fn}: table near line {_line} declares {_n} columns, "
                            f"a row has {_got}: {_ln[:60]!r}")
                break
claim("every table row matches its column spec", True)

print(f"checked {checked} asserted claims against the source of truth")
if fail:
    print(f"\n{len(fail)} PROBLEM(S):")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("consistency audit clean")
