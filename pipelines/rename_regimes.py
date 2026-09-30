"""Rewrite legacy stratum keys to the regime names L0 to L5, in place.

The old keys named a search outcome, which invited reading a population as
"clean". The regime names say which source of predictive information the test
population withholds, which is the thing that varies. Originals are kept beside
each file with a .prerename suffix so the migration is reversible.

Run once per report tree. Re-running is a no-op.
"""
import argparse, glob, json, os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "paper", "figures"))
from regimes import to_L                                             # noqa: E402

LEVEL_FIELDS = ("level", "stratum", "regime")


def walk(o):
    """Rename dict keys that are regime names, and values of level-ish fields."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            nk = to_L(k) if isinstance(k, str) else k
            nv = walk(v)
            if isinstance(k, str) and k in LEVEL_FIELDS and isinstance(v, str):
                nv = to_L(v)
            if nk in out and nk != k:
                raise SystemExit(f"rename would collide on {nk!r}; resolve by hand")
            out[nk] = nv
        return out
    if isinstance(o, list):
        return [walk(x) for x in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "reports/layered", "reports/hidden_homology"],
        help="pipeline outputs only; paper/tables are regenerated from these")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    n_changed = 0
    for r in a.roots:
        for f in sorted(glob.glob(os.path.join(root, r, "*.json"))):
            before = open(f).read()
            try:
                new = json.dumps(walk(json.loads(before)), indent=1)
            except json.JSONDecodeError:
                print(f"  skipped, not JSON: {f}")
                continue
            if json.loads(new) == json.loads(before):
                continue
            n_changed += 1
            print(f"  {'would rewrite' if a.dry_run else 'rewrote'} {os.path.relpath(f, root)}")
            if not a.dry_run:
                if not os.path.exists(f + ".prerename"):
                    shutil.copy2(f, f + ".prerename")
                open(f, "w").write(new + "\n")
    print(f"\n  {n_changed} file(s) {'would change' if a.dry_run else 'changed'}")


if __name__ == "__main__":
    main()
