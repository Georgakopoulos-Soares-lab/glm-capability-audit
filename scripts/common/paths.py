"""
Central path/config resolution so every script in this repository is portable.

Nothing here is machine-specific. Every value comes from an environment variable
with a repository-relative default, so a fresh clone resolves to somewhere sane
and a missing input fails with a readable message instead of a stale absolute
path belonging to someone else's cluster.

Minimum setup on a new machine:

    export VB_ROOT=/scratch/$USER/glm-capability-audit   # working root
    export VB_DATA=$VB_ROOT/data                         # benchmark corpora
    export VB_MMSEQS=$(which mmseqs)
    export HF_HOME=$VB_ROOT/hf_cache

Then fetch the corpora with `python scripts/common/download_data.py`, or point
the per-benchmark variables below at copies you already have.

Call `require(path, what, how)` before reading an input so the failure names the
variable to set. Call `sub(name)` for a result subdirectory.
"""
import os

# --- roots -------------------------------------------------------------------
# REPO is the checkout itself; the output trees (results/, reports/) are written
# here by default and are not tracked.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ROOT is the working root for anything large and uncommitted.
ROOT = os.environ.get("VB_ROOT", REPO)
OUT = os.environ.get("VB_OUT", os.path.join(REPO, "results"))
SCRATCH = os.environ.get("VB_SCRATCH", os.path.join(ROOT, "scratch"))
REPORTS = os.environ.get("VB_REPORTS", os.path.join(REPO, "reports"))
DATA = os.environ.get("VB_DATA", os.path.join(ROOT, "data"))

# --- benchmark corpora --------------------------------------------------------
# HVUE: {task}_{train,validation,test}.parquet, columns [sequence, label];
# tasks Host_Tropism, Pathogenecity, Transmissibility.
HVUE_DIR = os.environ.get("VB_HVUE_DIR", os.path.join(DATA, "hvue"))
GUE_DIR = os.environ.get("VB_GUE_DIR", os.path.join(DATA, "gue_viral"))
VIRO_DIR = os.environ.get("VB_VIRO_DIR", os.path.join(DATA, "virobench"))
GENEB_DIR = os.environ.get("VB_GENEB_DIR", os.path.join(DATA, "geneb", "tasks"))
GENEB_SPEC = os.environ.get("VB_GENEB_SPEC", os.path.join(DATA, "geneb", "benchmark_spec.json"))
NT_DIR = os.environ.get("VB_NT_DIR", os.path.join(DATA, "ntv3"))
GENEB_HARNESS = os.environ.get("VB_GENEB_HARNESS", os.path.join(DATA, "geneb", "harness"))
EPI_DIR = os.environ.get("VB_EPI_DIR", os.path.join(GUE_DIR, "EPI"))

# Protein-side corpora (escape / variant-effect comparators).
PROTEINGYM_DIR = os.environ.get("VB_PROTEINGYM_DIR", os.path.join(DATA, "proteingym"))
PROTEINGYM_DMS = os.environ.get(
    "VB_PROTEINGYM_DMS", os.path.join(PROTEINGYM_DIR, "DMS_ProteinGym_substitutions")
)
PROTEINGYM_ZEROSHOT = os.environ.get(
    "VB_PROTEINGYM_ZEROSHOT", os.path.join(PROTEINGYM_DIR, "zeroshot")
)
ESCAPE_DIR = os.environ.get("VB_ESCAPE_DIR", os.path.join(DATA, "escape"))
ESCAPE_CSV = os.environ.get("VB_ESCAPE_CSV", os.path.join(ESCAPE_DIR, "escape_data.csv"))
ESCAPE_MSA = os.environ.get(
    "VB_ESCAPE_MSA",
    os.path.join(
        ESCAPE_DIR, "DMS_msa_files", "SPIKE_SARS2_theta0.99_full_11-26-2021_b01.a2m"
    ),
)

# Pre-built composition-cluster splits from the original project. OPTIONAL and
# baseline-hostile; prefer the identity-disjoint splits from
# scripts/track_b_viral/build_identity_splits.py. See SPLIT_DESIGN_EXPLAINED.md.
SPLITS_V2 = os.environ.get("VB_SPLITS_V2", os.path.join(DATA, "hvue_splits_v2"))

# --- external tools -----------------------------------------------------------
# mmseqs2 is required for every homology audit. `which mmseqs` if on PATH.
MMSEQS = os.environ.get("VB_MMSEQS", "mmseqs")
MMSEQS_SCRATCH = os.environ.get("VB_MMSEQS_SCRATCH", os.path.join(SCRATCH, "mmseqs"))
# Shared HuggingFace cache. Set HF_HOME to keep model downloads off the home volume.
HF_CACHE = os.environ.get("HF_HOME", os.path.join(ROOT, "hf_cache"))

# --- optional external project ------------------------------------------------
# Only scripts/track_b_viral/{hvue,geneb}_evo_lora.py need this: they import
# src.utils, src.lora and scripts.hvue_lora_finetune from the `glm-locking`
# project. Every other script runs without it.
LOCK_ROOT = os.environ.get("VB_LOCK_ROOT", os.path.join(ROOT, "glm-locking"))

# --- corpora added after the viral audit widened ------------------------------
# HVUE v2 (HF duttaprat/HVUE-v2): <Task>/<config>/{train,dev,test}.csv, the
# authors' own splits used as shipped.
HVUE_V2_DIR = os.environ.get("HVUE_V2_DIR", os.path.join(DATA, "hvue_v2"))
# Virus-Host DB host tropism, built here by track_b_viral/build_vhdb_tropism.py.
# The only corpus that ships a genome `cluster` column, i.e. a correct bootstrap unit.
VHDB_DIR = os.environ.get("VB_VHDB_DIR", os.path.join(DATA, "vhdb_tropism"))
VHDB_TSV = os.environ.get("VB_VHDB_TSV", os.path.join(DATA, "virushostdb", "virushostdb.tsv"))
# GUE with a dev split carved off train (the released GUE has none).
GUE_DEVCARVE = os.environ.get("VB_GUE_DEVCARVE", os.path.join(DATA, "gue_devcarve"))
# Viral chunks vs hg38 background, built by track_b_viral/build_viral_vs_host.py.
VIRAL_VS_HOST = os.environ.get("VB_VIRAL_VS_HOST", os.path.join(DATA, "viral_vs_host"))
# Local model weights that are not on the hub.

# --- viral-audit outputs ------------------------------------------------------
# Kept separate from OUT because these are large and regenerable: per-arm
# prediction arrays, capacity-sweep ladders, cached homology masks.
VIRAL_OUT = os.environ.get("VB_VIRAL_OUT", os.path.join(OUT, "viral_bench"))
PREDS = os.environ.get("VB_PREDS", os.path.join(VIRAL_OUT, "preds"))
MULTIMODEL = os.environ.get("VB_MULTIMODEL", os.path.join(SCRATCH, "multimodel"))


def sub(name):
    """Result subdirectory under $VB_OUT, created on demand."""
    p = os.path.join(OUT, name)
    os.makedirs(p, exist_ok=True)
    return p


def scratch(name):
    """Scratch subdirectory under $VB_SCRATCH, created on demand."""
    p = os.path.join(SCRATCH, name)
    os.makedirs(p, exist_ok=True)
    return p


def require(path, what, how=None):
    """Return `path`, or raise FileNotFoundError naming what to set.

    Use this at the top of a script instead of letting a missing corpus surface
    as an opaque error deep inside a loader.
    """
    if os.path.exists(path):
        return path
    msg = f"{what} not found at {path!r}."
    if how:
        msg += f" {how}"
    raise FileNotFoundError(msg)


def describe():
    """Print the resolved configuration. `python -m scripts.common.paths`."""
    for key, value in sorted(globals().items()):
        if key.isupper() and isinstance(value, str):
            mark = "" if os.path.exists(value) else "   (missing)"
            print(f"{key:22s} {value}{mark}")


if __name__ == "__main__":
    describe()
