"""Guards on the pipeline configuration.

These test the failure modes that produced wrong numbers in this project, not
the YAML parser. Each one maps to a specific incident.
"""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pipelines"))
from registry import Registry  # noqa: E402


@pytest.fixture(scope="module")
def R():
    return Registry()


def test_sweep_tags_unique(R):
    """Two tasks sharing a sweep_tag silently overwrite each other's ladder."""
    tags = [t.sweep_tag for t in R.tasks.values() if t.sweep_tag]
    assert len(tags) == len(set(tags))


def test_learning_rates_are_numbers(R):
    """YAML 1.1 reads `1e-5` as a string; unnoticed it lands in a filename as
    the literal text and the skip-check then never matches the real artifact."""
    for key in ("lrs",):
        for v in R.protocol["finetune"][key]:
            assert isinstance(v, float), f"{key} entry {v!r} is {type(v).__name__}, not float"
    for v in R.protocol["baselines"]["transformer_lrs"]:
        assert isinstance(v, float)


def test_nn_lookup_excluded_from_homolog_free_floor(R):
    """NN-lookup is structurally disabled without a neighbour. Including it in
    that floor forces label transfer to guess and flatters every model."""
    free = R.protocol["regime"]["floor_homolog_free"]
    avail = R.protocol["regime"]["floor_homolog_available"]
    assert "nn_lookup" not in free
    assert "nn_lookup" in avail


def test_floor_is_never_a_single_arm(R):
    """The reported floor is max() over several baselines; k-mer alone understated
    the achievable floor by up to 0.185 MCC."""
    for k in ("floor_homolog_available", "floor_homolog_free"):
        assert len(R.protocol["regime"][k]) >= 3
        assert "kmer_gbt" in R.protocol["regime"][k]
        assert "cnn" in R.protocol["regime"][k]


def test_imbalanced_tasks_report_auprc(R):
    assert R.protocol["reporting"]["imbalanced_also"] == "auprc"
    assert R.protocol["reporting"]["primary_metric"] == "mcc"
    assert any(t.imbalanced for t in R.tasks.values())


def test_homology_search_includes_reverse_complement(R):
    """--search-type 3 is nucleotide mode. Protein mode would miss revcomp
    homology entirely and understate leakage."""
    assert R.protocol["homology"]["search_type"] == 3


def test_homology_uses_query_coverage(R):
    """--cov-mode 2 covers the query. Mode 1 asks a 1 kb chunk to cover a 40 kb
    genome and matched 0.3% of rows."""
    assert R.protocol["homology"]["cov_mode"] == 2


def test_masks_are_cached(R):
    """MMseqs2's prefilter is not reproducible: 41.0% vs 37.9% coverage on
    identical inputs. Without the cache, arms are scored on different splits."""
    assert R.protocol["homology"]["cache_masks"] is True


def test_three_seeds_minimum(R):
    """No single-seed-vs-multi-seed comparisons."""
    assert len(R.protocol["finetune"]["seeds"]) >= 3
    assert R.protocol["baselines"]["refit_seeds"] >= 3


def test_selection_never_uses_test(R):
    assert R.protocol["finetune"]["select_on"] == "dev_mcc"


def test_excluded_tasks_state_a_reason(R):
    for t in R.tasks.values():
        if not t.in_paper:
            assert t.note.strip(), f"{t.key} is excluded but gives no reason"


def test_restricted_models_state_a_reason(R):
    for m in R.models.values():
        if m.tasks or m.stages:
            assert m.note.strip(), f"{m.key} is restricted but gives no reason"


def test_gue_kmer_tag_override(R):
    """GUE k-mer/GBT predictions are filed under a short name while the FT arm
    uses the long one. Without the override the glob finds nothing, the arm is
    silently absent, and the floor drops."""
    assert R.tasks["gue_phage"].kmer_tag != R.tasks["gue_phage"].esc
