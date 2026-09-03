"""Tests for the CNN capacity ladder.

`receptive_field` is scientifically load-bearing: the paper's "baseline validity"
finding is that a published FM-vs-CNN gap collapses once the CNN can actually see
the whole input. These tests pin the arithmetic to the numbers quoted in
reports/PAPER_OUTLINE.md and reports/STATE_OF_EVIDENCE.md, so a refactor cannot
silently move them.

capacity_sweep imports torch at module scope, so the pure functions are loaded by
exec-ing the relevant source region rather than importing the module. That keeps
this test CPU-only and dependency-light.
"""
from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = (REPO / "scripts" / "common" / "capacity_sweep.py").read_text()


def _load_pure():
    """Exec just the receptive_field + LADDER region, with no torch import."""
    start = SRC.index("def receptive_field")
    end = SRC.index("def enc(")
    namespace: dict = {}
    exec(SRC[start:end], namespace)  # noqa: S102 - trusted first-party source
    return namespace


PURE = _load_pure()
receptive_field = PURE["receptive_field"]
LADDER = PURE["LADDER"]
INCUMBENT = PURE["INCUMBENT"]


def test_resnet_receptive_field_matches_the_published_89bp():
    # reports/PAPER_OUTLINE.md: "ResNet 9.44M (RF 89bp) = 0.336 vs U-Net 0.26M
    # (global) = 0.951 on 600bp splice". The 89 bp is this configuration.
    assert receptive_field("resnet", 5, 9, 1) == 89


def test_unet_is_reported_as_global():
    # U-Net downsamples 4x per level, so a single RF number is meaningless; the
    # ladder records None and the analysis treats it as spanning the input.
    assert receptive_field("unet", 3, 9, 2) is None


def test_dilated_receptive_field_grows_geometrically():
    # 1 + sum_{i<nb} (k-1) * dbase^i  with k=9, dbase=2
    assert receptive_field("dilated", 3, 9, 2) == 57
    assert receptive_field("dilated", 4, 9, 2) == 121
    assert receptive_field("dilated", 5, 9, 2) == 249
    assert receptive_field("dilated", 6, 9, 2) == 505


def test_resnet_receptive_field_grows_only_linearly():
    # Undilated residual stacks add a constant per block, which is exactly why
    # they lose to a global model on long inputs at far higher parameter count.
    rfs = [receptive_field("resnet", nb, 9, 1) for nb in (3, 4, 5, 6)]
    assert rfs == [57, 73, 89, 105]
    deltas = {b - a for a, b in zip(rfs, rfs[1:])}
    assert deltas == {16}


def test_incumbent_baseline_cannot_span_a_600bp_input():
    """The finding, as an assertion.

    The incumbent baseline used in every prior result is dilated/128ch/5 blocks.
    Its receptive field is 249 bp against the 600 bp splice input, so it
    structurally cannot use most of the sequence. Any FM-vs-CNN gap measured
    against it is confounded with receptive field.
    """
    arch, _channels, blocks = INCUMBENT
    rf = receptive_field(arch, blocks, 9, 2)
    assert rf == 249
    assert rf < 600


def test_ladder_spans_architectures_and_includes_the_incumbent():
    assert INCUMBENT in LADDER
    assert {arch for arch, _c, _b in LADDER} == {"dilated", "unet", "resnet"}
    # A ladder with only one architecture cannot separate capacity from
    # architecture, which is the confound this experiment exists to break.
    assert len(LADDER) >= 9


@pytest.mark.parametrize("arch", ["dilated", "resnet"])
def test_receptive_field_is_monotonic_in_depth(arch):
    dbase = 2 if arch == "dilated" else 1
    rfs = [receptive_field(arch, nb, 9, dbase) for nb in range(1, 8)]
    assert rfs == sorted(rfs)
    assert len(set(rfs)) == len(rfs)


def test_dilation_cap_prevents_unbounded_growth():
    # dbase ** min(i, 8) caps the dilation, so blocks past the 9th add a
    # constant rather than doubling. Without the cap a deep stack would report
    # an absurd receptive field.
    deep = receptive_field("dilated", 12, 9, 2)
    deeper = receptive_field("dilated", 13, 9, 2)
    assert deeper - deep == 8 * 2 ** 8
