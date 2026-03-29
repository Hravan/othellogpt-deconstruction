"""
tests/analysis/test_correlations.py
"""

import numpy as np
import pytest

from othellogpt_deconstruction.analysis.correlations import (
    correlate, print_correlation_table, summarise_correlations,
    CorrelationResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_records(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate synthetic records with known correlation structure."""
    rng = np.random.default_rng(seed)
    n_diff_cells = rng.integers(1, 5, size=n).tolist()
    records = []
    for i in range(n):
        nd = n_diff_cells[i]
        records.append({
            "ply":                  int(rng.integers(5, 40)),
            "n_diff_cells":         nd,
            "max_color_dist":       1,
            "total_color_dist":     nd,
            "tv_distance_full":     float(nd * 0.1 + rng.uniform(0, 0.02)),
            "js_divergence_full":   float(nd * 0.08 + rng.uniform(0, 0.02)),
            "spearman_rho_full":    float(1.0 - nd * 0.1 + rng.uniform(-0.05, 0.05)),            "tv_distance_legal":    float(nd * 0.09 + rng.uniform(0, 0.05)),
            "js_divergence_legal":  float(nd * 0.07 + rng.uniform(0, 0.05)),
            "spearman_rho_legal":   float(1.0 - nd * 0.09 + rng.uniform(-0.05, 0.05)),
        })
    return records


def make_records_with_nones(records: list[dict]) -> list[dict]:
    result = [r.copy() for r in records]
    result[0]["tv_distance_full"] = None
    result[1]["spearman_rho_full"] = None
    return result


# ---------------------------------------------------------------------------
# correlate
# ---------------------------------------------------------------------------

def test_correlate_returns_list():
    records = make_records()
    results = correlate(records)
    assert isinstance(results, list)


def test_correlate_result_count():
    records = make_records()
    results = correlate(records,
                        predictors=["n_diff_cells", "ply"],
                        metrics=["tv_distance_full", "js_divergence_full"])
    assert len(results) == 4


def test_correlate_result_type():
    records = make_records()
    results = correlate(records)
    assert all(isinstance(r, CorrelationResult) for r in results)


def test_correlate_known_positive():
    """n_diff_cells should correlate positively with tv_distance_full."""
    records = make_records()
    results = correlate(records,
                        predictors=["n_diff_cells"],
                        metrics=["tv_distance_full"])
    assert len(results) == 1
    assert results[0].rho is not None
    assert results[0].rho > 0.5


def test_correlate_constant_predictor_returns_none():
    """max_color_dist is constant in real data — should return None rho."""
    records = make_records()
    results = correlate(records,
                        predictors=["max_color_dist"],
                        metrics=["tv_distance_full"])
    assert results[0].rho is None
    assert results[0].p_value is None


def test_correlate_filters_nones():
    records = make_records_with_nones(make_records())
    results = correlate(records,
                        predictors=["n_diff_cells"],
                        metrics=["tv_distance_full"])
    assert results[0].n < len(records)


def test_correlate_spearman_rho_direction():
    """Higher n_diff_cells should mean more distributional difference,
    i.e. lower spearman_rho. After negation the correlation should be positive."""
    records = make_records()
    # Manually check the direction: rho between n_diff_cells and -spearman_rho_full
    import numpy as np
    from scipy.stats import spearmanr
    x = np.array([r["n_diff_cells"] for r in records])
    y = -np.array([r["spearman_rho_full"] for r in records])
    rho, _ = spearmanr(x, y)
    assert rho > 0


def test_correlate_empty_records():
    results = correlate([])
    assert results == []


def test_correlate_significance_flag():
    records = make_records(n=200)
    results = correlate(records,
                        predictors=["n_diff_cells"],
                        metrics=["tv_distance_full"])
    assert results[0].significant is True


# ---------------------------------------------------------------------------
# CorrelationResult.__str__
# ---------------------------------------------------------------------------

def test_correlation_result_str_with_values():
    r = CorrelationResult(
        predictor="n_diff_cells", metric="tv_distance_full",
        rho=0.234, p_value=1.5e-10, n=1000, significant=True,
    )
    s = str(r)
    assert "0.234" in s
    assert "**" in s


def test_correlation_result_str_none():
    r = CorrelationResult(
        predictor="max_color_dist", metric="tv_distance_full",
        rho=None, p_value=None, n=1000, significant=False,
    )
    s = str(r)
    assert "nan" in s


# ---------------------------------------------------------------------------
# summarise_correlations
# ---------------------------------------------------------------------------

def test_summarise_empty():
    assert summarise_correlations([]) == {}


def test_summarise_keys():
    records = make_records()
    results = correlate(records)
    s = summarise_correlations(results)
    assert "n_tests" in s
    assert "n_significant" in s
    assert "mean_abs_rho" in s
    assert "max_abs_rho" in s


def test_summarise_counts():
    records = make_records()
    results = correlate(records,
                        predictors=["n_diff_cells", "ply"],
                        metrics=["tv_distance_full", "js_divergence_full"])
    s = summarise_correlations(results)
    assert s["n_tests"] == 4


def test_summarise_significant_count():
    records = make_records(n=200)
    results = correlate(records,
                        predictors=["n_diff_cells"],
                        metrics=["tv_distance_full"])
    s = summarise_correlations(results)
    assert s["n_significant"] >= 1


# ---------------------------------------------------------------------------
# print_correlation_table (smoke test)
# ---------------------------------------------------------------------------

def test_print_correlation_table_runs(capsys):
    records = make_records()
    results = correlate(records)
    print_correlation_table(results)
    captured = capsys.readouterr()
    assert "Predictor" in captured.out
    assert "rho" in captured.out
