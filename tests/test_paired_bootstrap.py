"""Tests for the equivalence verdict used in the paired bootstrap.

The negative result in this project rests on a distinction that is easy to get
wrong: "we failed to show a difference" is not "we showed equivalence". The
verdict function implements the second claim (two one-sided tests against a
pre-declared margin), and these tests pin that a wide interval cannot be read as
equivalence.

paired_bootstrap imports sklearn and a sibling module at import time, so the pure
function is loaded from source rather than by importing the module.
"""
from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = (REPO / "scripts" / "common" / "paired_bootstrap.py").read_text()


def _load_verdict():
    start = SRC.index("def equivalence_verdict")
    end = SRC.index("DELTAS = [")
    namespace: dict = {}
    exec(SRC[start:end], namespace)  # noqa: S102 - trusted first-party source
    return namespace["equivalence_verdict"]


equivalence_verdict = _load_verdict()


def test_interval_inside_the_margin_is_equivalent():
    assert equivalence_verdict(-0.005, 0.004, 0.01) is True


def test_wide_interval_is_not_equivalence():
    """The central guard.

    A delta of +0.001 with CI [-0.20, +0.20] is uninformative, not equivalent.
    Treating it as equivalence would let an underpowered run masquerade as a
    demonstrated null.
    """
    assert equivalence_verdict(-0.20, 0.20, 0.01) is False


def test_interval_touching_the_margin_is_not_equivalent():
    # Strict inequality on both sides: the boundary is not inside the region.
    assert equivalence_verdict(-0.01, 0.009, 0.01) is False
    assert equivalence_verdict(-0.009, 0.01, 0.01) is False


def test_clearly_positive_interval_is_not_equivalent():
    # A real advantage must not be reported as equivalence.
    assert equivalence_verdict(0.05, 0.12, 0.03) is False


def test_verdict_is_monotonic_in_the_margin():
    """Widening the pre-declared margin can only make equivalence easier.

    A non-monotonic verdict would mean the margin could be shopped for a
    conclusion in either direction.
    """
    lo, hi = -0.025, 0.018
    verdicts = [equivalence_verdict(lo, hi, m) for m in (0.01, 0.02, 0.03, 0.05)]
    assert verdicts == [False, False, True, True]


def test_returns_plain_bool_for_json_serialisation():
    # Results are written to JSON; a numpy bool_ would not serialise.
    verdict = equivalence_verdict(-0.001, 0.001, 0.01)
    assert type(verdict) is bool


@pytest.mark.parametrize("margin", [0.01, 0.02, 0.03, 0.05])
def test_declared_margins_all_reject_a_degenerate_interval(margin):
    # A CI spanning the whole metric range is never equivalence, at any margin.
    assert equivalence_verdict(-1.0, 1.0, margin) is False


def test_margins_are_declared_in_source_before_use():
    """The margins must be literals in the file, not derived from the data.

    Choosing a margin after seeing the CI is the failure this project's protocol
    forbids; keeping them as a module-level literal makes that auditable.
    """
    assert "DELTAS = [0.01, 0.02, 0.03, 0.05]" in SRC
    assert SRC.index("DELTAS = [") < SRC.index("equivalence_verdict(lo, hi, dd)")
