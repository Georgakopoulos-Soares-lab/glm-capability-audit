"""Splice generated LaTeX straight into main.tex or supplementary.tex.

The paper is two files. A generator writes its table into the document between
a pair of marker comments rather than into a side file, so there is nothing to
\input and nothing to forget to ship. Regenerating is idempotent: the block
between the markers is replaced, everything else is untouched.

Numeric intermediates still go to reports/, because those are pipeline outputs
that other generators read, not part of the manuscript.
"""
import io, os, re

PAPER = os.path.dirname(os.path.abspath(__file__)) + "/.."
ROOT = PAPER + "/.."
DATA = ROOT + "/reports/tables"


def begin(key):
    return f"% <<< GENERATED {key} (do not edit by hand)"


def end(key):
    return f"% >>> GENERATED {key}"


def inject(doc, key, latex):
    """Replace the marked block in `doc` with `latex`. Creates nothing implicitly."""
    path = doc if os.path.isabs(doc) else os.path.join(PAPER, doc)
    s = io.open(path, encoding="utf-8").read()
    b, e = begin(key), end(key)
    if b not in s or e not in s:
        raise SystemExit(f"{os.path.basename(path)} has no markers for {key!r}; "
                         f"add\n{b}\n{e}\nwhere the table belongs")
    pat = re.compile(re.escape(b) + r".*?" + re.escape(e), re.S)
    s = pat.sub(lambda _: b + "\n" + latex.rstrip("\n") + "\n" + e, s, count=1)
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"  injected {key} into {os.path.basename(path)}")


def data_path(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)


def require(path, produced_by, what):
    """Return `path`, or fail loudly if it is missing or empty.

    A generator that silently emits an empty table is worse than one that
    crashes: the table looks finished and the numbers are simply absent. This
    happened once in this project, when a renamed mask file left the genus
    columns of the capability matrix blank while the table still typeset. Every
    generator routes its inputs through here.
    """
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
