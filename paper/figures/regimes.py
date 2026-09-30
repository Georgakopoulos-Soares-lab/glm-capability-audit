"""Canonical names for the extrapolation regimes.

A regime is defined by which source of predictive information the test population
withholds, not by how "clean" it is. The old names described a search outcome;
these describe the question. Import this rather than hard coding a label.
"""

# ordered, with the withheld information and the legacy key that produced it
REGIMES = [
    ("L0", "all", "the benchmark as published, nothing withheld"),
    ("L1", "nucleotide_no_hit", "a nucleotide level training relationship is withheld"),
    ("L2", "translated_no_hit", "a six frame translated relationship is withheld"),
    ("L3", "protein_no_hit", "a protein level relationship is withheld"),
    ("L4", "unseen_species", "membership of a training species is withheld"),
    ("L5", "unseen_genus", "membership of a training genus is withheld"),
]

# legacy keys that must map onto the same regime
ALIASES = {"unseen_genus_recovered": "L5", "all_rows": "L0", "L5_recovered": "L5"}

TO_L = {legacy: name for name, legacy, _ in REGIMES}
TO_L.update(ALIASES)
TO_L.update({name: name for name, _, _ in REGIMES})
ORDER = [name for name, _, _ in REGIMES]
WITHHELD = {name: text for name, _, text in REGIMES}
LEGACY = {name: legacy for name, legacy, _ in REGIMES}


def to_L(key):
    """Regime name for any key this project has ever used. Unknown keys pass through."""
    return TO_L.get(key, key)


def legend_rows():
    """(regime, what it withholds) for a table or figure legend."""
    return [(n, WITHHELD[n]) for n in ORDER]


def legend_latex():
    return " ".join("\\textbf{%s} %s." % (n, w) for n, w in legend_rows())
