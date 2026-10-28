"""Resolve which source-of-truth table the generators read.

The pre-finalization table `final_source_of_truth.csv` is produced by
make_source_of_truth.py and is never modified. The finalization pass writes
`final_source_of_truth_finalized.csv` beside it, carrying the two-level
bootstrap intervals as the primary scheme plus per-cell provenance. When that
file is present it is what the manuscript is built from; otherwise the
generators fall back to the pre-finalization table unchanged.

Set VB_SOT to force a specific file.
"""
import os

_T = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "reports", "tables")
FINALIZED = os.path.join(_T, "final_source_of_truth_finalized.csv")
PREFINAL = os.path.join(_T, "final_source_of_truth.csv")


def sot_csv():
    forced = os.environ.get("VB_SOT")
    if forced:
        return forced
    return FINALIZED if os.path.exists(FINALIZED) else PREFINAL
