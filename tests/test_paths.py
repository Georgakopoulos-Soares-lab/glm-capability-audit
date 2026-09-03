"""Tests for the central path resolver.

Every script in scripts/ reads its inputs through scripts/common/paths.py. If
that module silently resolves to a stale absolute path, a run either crashes deep
inside a loader or -- worse -- reads the wrong corpus. These tests pin the two
properties that matter: nothing is machine-absolute by default, and every value
is overridable by environment variable.
"""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PATHS_PY = REPO / "scripts" / "common" / "paths.py"
SHIM_DIRS = ["track_a_benchmarks", "track_b_viral", "geneb", "positive_control"]


def load_paths(env=None, name="vb_paths_under_test"):
    """Import paths.py fresh, optionally with a patched environment."""
    spec = importlib.util.spec_from_file_location(name, PATHS_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paths():
    return load_paths()


def test_no_default_points_outside_the_repository(paths, monkeypatch):
    """With no VB_* set, every path must be inside the checkout.

    A default like /home/someone/tools/mmseqs is the failure mode this guards:
    it works on exactly one machine and fails confusingly everywhere else.
    """
    for key in [k for k in vars(paths) if k.isupper()]:
        value = getattr(paths, key)
        if not isinstance(value, str) or not value.startswith("/"):
            continue  # bare command names such as "mmseqs" are fine
        if key == "HF_CACHE":
            continue  # legitimately follows a pre-existing $HF_HOME
        assert value.startswith(str(REPO)), f"{key} defaults outside the repo: {value}"


def test_mmseqs_defaults_to_a_path_lookup(paths):
    # Not an absolute path to one person's build.
    assert paths.MMSEQS == "mmseqs"


@pytest.mark.parametrize(
    "var,attr",
    [
        ("VB_ROOT", "ROOT"),
        ("VB_OUT", "OUT"),
        ("VB_DATA", "DATA"),
        ("VB_REPORTS", "REPORTS"),
        ("VB_HVUE_DIR", "HVUE_DIR"),
        ("VB_GENEB_DIR", "GENEB_DIR"),
        ("VB_NT_DIR", "NT_DIR"),
        ("VB_MMSEQS", "MMSEQS"),
        ("VB_PROTEINGYM_DMS", "PROTEINGYM_DMS"),
        ("VB_ESCAPE_CSV", "ESCAPE_CSV"),
    ],
)
def test_every_knob_is_env_overridable(var, attr, monkeypatch):
    monkeypatch.setenv(var, "/tmp/override-sentinel")
    module = load_paths(name=f"vb_paths_{attr}")
    assert getattr(module, attr) == "/tmp/override-sentinel"


def test_data_root_cascades_to_per_corpus_defaults(monkeypatch):
    monkeypatch.setenv("VB_DATA", "/tmp/corpora")
    module = load_paths(name="vb_paths_cascade")
    assert module.HVUE_DIR == "/tmp/corpora/hvue"
    assert module.GENEB_DIR == "/tmp/corpora/geneb/tasks"
    assert module.NT_DIR == "/tmp/corpora/ntv3"


def test_require_names_the_variable_to_set(paths, tmp_path):
    missing = str(tmp_path / "absent")
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.require(missing, "GENEB tasks", "Set $VB_GENEB_DIR.")
    message = str(excinfo.value)
    assert "GENEB tasks" in message
    assert missing in message
    assert "VB_GENEB_DIR" in message


def test_require_returns_the_path_when_present(paths, tmp_path):
    present = tmp_path / "here"
    present.mkdir()
    assert paths.require(str(present), "thing") == str(present)


def test_sub_creates_the_directory(paths, tmp_path, monkeypatch):
    monkeypatch.setenv("VB_OUT", str(tmp_path / "out"))
    module = load_paths(name="vb_paths_sub")
    created = module.sub("demo")
    assert pathlib.Path(created).is_dir()
    assert created.endswith("/demo")


@pytest.mark.parametrize("subdir", SHIM_DIRS)
def test_shims_reexport_the_canonical_resolver(subdir):
    """`import paths` from any script directory must yield the same values.

    Each of these directories contains a paths.py shim, because a plain
    `from paths import *` inside them resolves to the shim itself and silently
    exports nothing.
    """
    code = "import paths as P; print(P.GENEB_DIR); print(callable(P.sub)); print(callable(P.require))"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO / "scripts" / subdir,
        capture_output=True,
        text=True,
        check=True,
    )
    geneb_dir, sub_callable, require_callable = proc.stdout.split()
    assert geneb_dir == load_paths(name=f"vb_paths_shim_{subdir}").GENEB_DIR
    assert sub_callable == "True"
    assert require_callable == "True"


def test_every_script_directory_can_resolve_paths():
    """No script directory is left without a working `import paths`."""
    for subdir in SHIM_DIRS:
        assert (REPO / "scripts" / subdir / "paths.py").exists(), subdir
