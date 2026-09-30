"""Port of the dataviz skill's validate_palette.js (no node on this host).

Same five checks, same constants, same Machado 2009 CVD matrices at severity 1.0.
Delta E is Euclidean distance in OKLab x100.
"""
import math, sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
M = {
 "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998]],
 "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881]],
}
h2s = lambda h: [int(h.strip().lstrip("#")[i:i+2], 16)/255 for i in (0, 2, 4)]
s2lin = lambda c: c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
lin = lambda h: [s2lin(c) for c in h2s(h)]
def relLum(h):
    r, g, b = lin(h); return 0.2126*r + 0.7152*g + 0.0722*b
def contrast(a, b):
    hi, lo = sorted([relLum(a), relLum(b)], reverse=True); return (hi+0.05)/(lo+0.05)
def oklab_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return [0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s]
oklab = lambda h: oklab_lin(lin(h))
def oklch(h):
    L, a, b = oklab(h); return L, math.hypot(a, b)
def simulate(h, kind):
    r, g, b = lin(h); m = M[kind]
    return oklab_lin([sum(m[i][j]*v for j, v in enumerate((r, g, b))) for i in range(3)])
dE = lambda p, q: 100*math.dist(p, q)

def report(pal, mode="light", allpairs=False, ordinal=False):
    """`ordinal=True` applies the ORDINAL gate instead of the categorical one.

    A nested or ordered ramp is one hue stepped light to dark, so the lightness
    band and the chroma floor (both of which police categorical hue identity) do
    not apply. What does apply is that the step nearest the surface must still be
    visible against it: >= 2:1 on light, per the skill's ordinal rule.
    """
    surf = SURFACE[mode]; lo, hi = BAND[mode]; ok = True
    print(f"  palette ({mode}): {', '.join(pal)}")
    for h in pal:
        L, C = oklch(h); c = contrast(h, surf)
        if ordinal:
            f1 = f2 = True
            if c < 2.0:
                ok = False
                print(f"    {h}  contrast={c:.2f} below the 2:1 ordinal floor  [FAIL]")
                continue
        else:
            f1 = lo <= L <= hi; f2 = C >= CHROMA_FLOOR
        tag = "PASS" if (f1 and f2) else "FAIL"
        if not (f1 and f2): ok = False
        print(f"    {h}  L={L:.3f} {'ok ' if f1 else 'OUT'}  C={C:.3f} {'ok ' if f2 else 'LOW'}  "
              f"contrast={c:.2f}{'  (relief rule: needs visible labels)' if c < 3 else ''}  [{tag}]")
    # The active pairlist is ADJACENT by default (all-pairs only under --pairs all),
    # matching validate_palette.js; the 8-slot reference default fails an all-pairs
    # normal-vision check by design, so applying it here would be wrong.
    pairs = [(pal[i], pal[i+1]) for i in range(len(pal)-1)]
    if allpairs:
        pairs = [(a, b) for i, a in enumerate(pal) for b in pal[i+1:]]
    worst_n = min(((dE(oklab(a), oklab(b)), a, b) for a, b in pairs), default=None)
    if worst_n:
        d, a, b = worst_n
        st = "PASS" if d >= NORMAL_FLOOR else "FAIL"
        if d < NORMAL_FLOOR: ok = False
        print(f"    normal-vision worst pair: {a} vs {b}  dE={d:.1f} (floor {NORMAL_FLOOR}) [{st}]")
    for i in range(len(pal)-1):
        a, b = pal[i], pal[i+1]
        d = min(dE(simulate(a, k), simulate(b, k)) for k in M)
        st = "PASS" if d >= CVD_TARGET else ("WARN" if d >= CVD_FLOOR else "FAIL")
        if d < CVD_FLOOR: ok = False
        print(f"    adjacent CVD {a} vs {b}: dE={d:.1f} (target {CVD_TARGET}, floor {CVD_FLOOR}) [{st}]")
    print(f"  => {'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'}\n")
    return ok

if __name__ == "__main__":
    report([c.strip() for c in sys.argv[1].split(",")],
           next((a for a in sys.argv[2:] if not a.startswith("--") and a in SURFACE), "light"),
           "--pairs" in sys.argv and "L0" in sys.argv,
           "--ordinal" in sys.argv)
