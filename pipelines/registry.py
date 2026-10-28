"""Load, expand and validate pipelines/config/*.yaml.

Config is data, not code: `--tasks`, `--models` and `--stages` on the CLI select
rows from these files, so changing what the paper measures is an edit to YAML
rather than to a command line buried in a shell history.

`${VAR}` in any string is expanded from the environment, falling back to the
defaults in scripts/common/paths.py so a fresh clone resolves somewhere sane.
"""
from __future__ import annotations
import os, re, sys
from dataclasses import dataclass, field

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "config")
sys.path.insert(0, os.path.join(REPO, "scripts"))

_VAR = re.compile(r"\$\{(\w+)\}")

# Fallbacks for the variables the config interpolates. Environment wins; these
# only stop a fresh clone from producing a literal "${VB_DATA}" in a path.
_DEFAULTS = {
    "VB_DATA": os.environ.get("VB_DATA", os.path.join(REPO, "data")),
}


def _expand(o):
    if isinstance(o, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), _DEFAULTS.get(m.group(1), "")), o)
    if isinstance(o, dict):
        return {k: _expand(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_expand(v) for v in o]
    return o


# YAML 1.1 reads `1e-5` as a STRING (a float needs `1.0e-5`), which surfaces far
# downstream as a format error or, worse, as a literal "1e-5" in a filename that
# no longer matches the one the run actually wrote. Coerce the numeric keys here.
_NUMERIC = {"lrs", "transformer_lrs", "C_grid", "identity", "coverage",
            "min_seq_id", "min_cov", "evalue"}


def _numify(o, key=None):
    if isinstance(o, dict):
        return {k: _numify(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [_numify(v, key) for v in o]
    if key in _NUMERIC and isinstance(o, str):
        try:
            return float(o)
        except ValueError:
            return o
    return o


def _load(name):
    with open(os.path.join(CONFIG, f"{name}.yaml")) as f:
        return _numify(_expand(yaml.safe_load(f)))


@dataclass
class Task:
    key: str
    name: str
    loader: str
    esc: str
    in_paper: bool = True
    dir: str = ""
    task_arg: str = ""
    kmer_esc: str = ""
    sweep_tag: str = ""
    cluster_csv: str = ""
    train_cap: int = 0
    imbalanced: bool = False
    level: str = ""
    seqlen: int = 0
    tax_train: str = ""      # train-row-order csv carrying species/genus/family
    tax_test: str = ""       # test-row-order, same columns
    author_ckpt: str = ""    # the authors' released task model, for the secondary contrast
    focus_model: str = ""    # model the equivalence verdict is computed for
    note: str = ""

    @property
    def kmer_tag(self):
        """Prediction-file tag for the k-mer/GBT arms; differs from `esc` on GUE."""
        return self.kmer_esc or self.esc

    def csv(self, part):
        return os.path.join(self.dir, f"{part}.csv") if self.dir else ""

    @property
    def epochs_key(self):
        return "epochs_gue" if self.loader == "geneb" else "epochs"


@dataclass
class Model:
    key: str
    hf_id: str
    loader: str
    role: str = ""
    max_tokens: int = 0
    needs_stubs: bool = False
    tasks: list = field(default_factory=list)   # empty = all in-paper tasks
    stages: list = field(default_factory=list)  # empty = all model stages
    note: str = ""

    def runs_on(self, task_key):
        return not self.tasks or task_key in self.tasks

    def runs_in(self, stage):
        return not self.stages or stage in self.stages


class Registry:
    def __init__(self):
        t = _load("tasks")
        m = _load("models")
        self.protocol = _load("protocol")
        self.tasks = {k: Task(key=k, **v) for k, v in t.items()}
        self.models = {k: Model(key=k, **v) for k, v in m["models"].items()}
        self.baselines = m["baselines"]
        self._validate()

    # -- selection -------------------------------------------------------------
    def select_tasks(self, spec):
        """`spec` is a comma list of keys, or "paper" (default) / "all".

        Naming an in_paper=false task explicitly always includes it, but says so:
        those tasks were excluded for a stated reason and a silent inclusion
        would put a known-saturated benchmark back into the headline table.
        """
        if spec in (None, "", "paper"):
            return [t for t in self.tasks.values() if t.in_paper]
        if spec == "all":
            return list(self.tasks.values())
        out = []
        for k in spec.split(","):
            k = k.strip()
            if k not in self.tasks:
                raise SystemExit(f"unknown task {k!r}; known: {', '.join(self.tasks)}")
            t = self.tasks[k]
            if not t.in_paper:
                print(f"  note: {k} is in_paper=false ({t.note.strip().splitlines()[0]}); "
                      f"included because you named it explicitly")
            out.append(t)
        return out

    def select_models(self, spec):
        if spec in (None, "", "all"):
            return list(self.models.values())
        out = []
        for k in spec.split(","):
            k = k.strip()
            if k not in self.models:
                raise SystemExit(f"unknown model {k!r}; known: {', '.join(self.models)}")
            out.append(self.models[k])
        return out

    def python(self):
        """The interpreter every job runs under.

        One environment covers the whole pipeline. It did not always: the earlier
        split existed because Evo 2 needed a dependency set that could not coexist
        with the rest, and picking the wrong interpreter was a silent failure that
        killed several runs. Evo 2 is gone, so the remaining work -- pandas,
        scikit-learn, xgboost, torch and transformers -- fits in one place.
        """
        p = os.environ.get("VB_PYTHON") or sys.executable
        if not os.path.exists(p):
            raise SystemExit(
                f"$VB_PYTHON is {p!r}, which does not exist.\n"
                f"Point it at the python of the environment built from environment.yml, "
                f"or leave it unset to use the interpreter running this script.")
        return p

    # -- integrity -------------------------------------------------------------
    def _validate(self):
        seen = {}
        for t in self.tasks.values():
            # Two tasks sharing a sweep_tag would overwrite each other's ladder.
            if t.sweep_tag:
                if t.sweep_tag in seen:
                    raise SystemExit(f"sweep_tag collision: {t.key} and {seen[t.sweep_tag]} "
                                     f"both write {t.sweep_tag}")
                seen[t.sweep_tag] = t.key
        for m in self.models.values():
            for k in m.tasks:
                if k not in self.tasks:
                    raise SystemExit(f"model {m.key} restricted to unknown task {k!r}")


if __name__ == "__main__":
    r = Registry()
    print(f"{len(r.tasks)} tasks ({sum(t.in_paper for t in r.tasks.values())} in paper), "
          f"{len(r.models)} models, {len(r.baselines)} baseline arms\n")
    for t in r.tasks.values():
        flag = "" if t.in_paper else "   [excluded from headline table]"
        unit = "cluster" if t.cluster_csv else "ROW (anti-conservative)"
        print(f"  {t.key:14s} {t.name:26s} loader={t.loader:10s} cap={t.train_cap or '-':>6} "
              f"unit={unit}{flag}")
