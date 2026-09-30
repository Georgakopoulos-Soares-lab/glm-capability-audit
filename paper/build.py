#!/usr/bin/env python
"""Rebuild every figure and table in the paper, then compile it.

The manuscript's tables are injected into main.tex and supplementary.tex between
marker comments rather than kept as separate files, so this is the only supported
way to regenerate them. Order matters: genus concentration feeds the capability
matrix, and the arm decay grid feeds the supplement.

    python paper/build.py            regenerate everything and compile
    python paper/build.py --no-pdf   regenerate only, skip LaTeX
    python paper/build.py --quick    skip the slow bootstrap calibration

Every step fails loudly on a missing input rather than emitting an empty table,
so a partial reports/ tree stops the build instead of producing a paper with
holes in it.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
PY = sys.executable

# (script, why it runs here, slow?)
STEPS = [
    ("make_genus_concentration.py", "genus breadth per population, feeds the matrix", False),
    ("explain_genus_floor.py", "why the two lineage verdicts differ", False),
    ("virobench_power.py", "detectable difference on the temporal populations", False),
    ("make_figdata.py", "figure data, derived from the ladder", False),
    ("make_figures.py", "figures 1 to 4 and S1", False),
    ("make_arm_decay_table.py", "main table 2, and the full grid for the supplement", False),
    ("make_capability_matrix.py", "main table 1", False),
    ("make_virobench_table.py", "main table 3", False),
    ("make_probe_table.py", "supplementary probe table", False),
    ("make_mde_calibration.py", "main table 4, bootstraps a synthetic predictor", True),
]


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true", help="regenerate only")
    ap.add_argument("--quick", action="store_true", help="skip the slow steps")
    a = ap.parse_args()

    failed = []
    for script, why, slow in STEPS:
        if slow and a.quick:
            print(f"  skipped  {script:32s} {why}")
            continue
        t0 = time.time()
        rc, out = run([PY, os.path.join(FIG, script)])
        if rc:
            failed.append(script)
            print(f"  FAILED   {script:32s} {why}")
            print("".join("      " + l + "\n" for l in out.strip().splitlines()[-6:]))
        else:
            print(f"  ok       {script:32s} {why}  ({time.time() - t0:.0f}s)")
    if failed:
        raise SystemExit(f"\n{len(failed)} step(s) failed, paper not rebuilt: "
                         + ", ".join(failed))

    # the supplement is written whole, so it runs last and needs --force once the
    # file has been edited by hand
    rc, out = run([PY, os.path.join(FIG, "make_supplementary.py")])
    if rc and "refusing to overwrite" in out:
        # expected: the supplement is hand maintained and the guard protects it
        print("  kept     supplementary.tex is hand maintained, left as is "
              "(make_supplementary.py --force to regenerate)")
    elif rc:
        print("  FAILED   make_supplementary.py")
        print("".join("      " + l + "\n" for l in out.strip().splitlines()[-3:]))
        raise SystemExit("\n  supplement generation failed, paper not rebuilt")
    else:
        print("  ok       make_supplementary.py")

    if a.no_pdf:
        print("\n  figures and tables rebuilt, PDF skipped")
        return
    rc, _ = run(["which", "pdflatex"])
    if rc:
        raise SystemExit("\n  pdflatex not found, install a TeX distribution or use --no-pdf")
    for i, cmd in enumerate([["pdflatex", "-interaction=nonstopmode", "main.tex"],
                             ["bibtex", "main"],
                             ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                             ["pdflatex", "-interaction=nonstopmode", "main.tex"]]):
        rc, out = run(cmd, cwd=HERE)
        if rc and cmd[0] == "pdflatex":
            bad = [l for l in out.splitlines() if l.startswith("! ")][:5]
            raise SystemExit("\n  LaTeX failed:\n" + "\n".join("    " + b for b in bad))
    log = open(os.path.join(HERE, "main.log")).read()
    for flag, label in (("Overfull", "overfull boxes"), ("undefined", "undefined references")):
        n = log.count(flag)
        print(f"  {label}: {n}")
    print("\n  wrote paper/main.pdf")


if __name__ == "__main__":
    main()
