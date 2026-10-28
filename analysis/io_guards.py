"""Input guards and the output directory shared by the analysis scripts.

A generator that silently emits an empty table is worse than one that crashes:
the table looks finished and the numbers are simply absent. That happened once
in this project, when a renamed mask file left the genus columns of the
capability matrix blank while the table still typeset. Every analysis script
routes its inputs through `require`.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reports", "tables")


def data_path(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)


def require(path, produced_by, what):
    """Return `path`, or fail loudly if it is missing or empty."""
    if not os.path.exists(path):
        raise SystemExit(
            f"missing input: {path}\n  holds: {what}\n  produce it with: {produced_by}")
    if os.path.getsize(path) == 0:
        raise SystemExit(f"empty input: {path}\n  produce it again with: {produced_by}")
    return path


def require_rows(rows, what, produced_by):
    """Fail rather than emit a table with no data rows."""
    if not rows:
        raise SystemExit(
            f"no rows to write for {what}\n  the inputs parsed but yielded nothing; "
            f"re-run: {produced_by}")
    return rows
