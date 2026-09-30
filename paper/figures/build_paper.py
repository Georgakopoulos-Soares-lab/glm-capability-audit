#!/usr/bin/env python
"""One command: source of truth -> figures -> tables -> injection -> PDF.

Order matters. Everything downstream of make_source_of_truth.py reads only the
consolidated CSV, so a number cannot enter the manuscript without passing through it.
"""
import io, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.join(REPO, "paper")
PY = sys.executable

STEPS = [
    ("source of truth", "make_source_of_truth.py"),
    ("main figures", "make_main_figures.py"),
    ("main tables", "make_main_tables.py"),
    ("supplement", "make_supplement_v2.py"),
    ("claim ledger", "make_claim_ledger.py"),
]
INJECT = [("table_headtohead", "main.tex")]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r


def main():
    no_pdf = "--no-pdf" in sys.argv
    for label, script in STEPS:
        r = run([PY, os.path.join(HERE, script)])
        if r.returncode:
            print(f"FAILED at {label}\n{r.stdout}\n{r.stderr}"); sys.exit(1)
        print(f"  {label}: ok")

    sys.path.insert(0, HERE)
    from inject import inject
    for key, doc in INJECT:
        body = io.open(os.path.join(REPO, "reports", "tables", "generated",
                                    f"{key}.tex"), encoding="utf-8").read()
        inject(doc, key, body)

    if no_pdf:
        print("\n  --no-pdf, stopping before latex"); return
    for i in range(3):
        r = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=PAPER)
        if i == 0:
            run(["bibtex", "main"], cwd=PAPER)
        if r.returncode and i == 2:
            tail = "\n".join(r.stdout.splitlines()[-40:])
            print(f"pdflatex failed\n{tail}"); sys.exit(1)
    log = io.open(os.path.join(PAPER, "main.log"), encoding="utf-8",
                  errors="ignore").read()
    over = len(re.findall(r"Overfull \\hbox", log))
    undef = len(re.findall(r"LaTeX Warning: Reference .* undefined", log))
    cite = len(re.findall(r"Citation .* undefined", log))
    pages = re.search(r"Output written on main\.pdf \((\d+) page", log)
    print(f"\n  pages: {pages.group(1) if pages else '?'}")
    print(f"  overfull hboxes: {over}   undefined refs: {undef}   undefined citations: {cite}")
    if undef or cite:
        for m in re.findall(r"(LaTeX Warning: Reference [^\n]*|Citation [^\n]*)", log)[:10]:
            print("   ", m.strip())
    print(f"  wrote paper/main.pdf")


if __name__ == "__main__":
    main()
