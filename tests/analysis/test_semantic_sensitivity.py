"""
tests/analysis/test_semantic_sensitivity.py
"""

import numpy as np
import pytest

from othellogpt_deconstruction.analysis.semantic_sensitivity import (
    SemanticSensitivityResult,
    group_sensitivity,
    group_contradiction_rate,
    from_groups,
    from_pairwise_records,
    pairwise_tv_distance,
    sensitivity_from_prob_matrix,
)


# ---------------------------------------------------------------------------
# pairwise_tv_distance
# ---------------------------------------------------------------------------

def test_tv_distance_identical():
    p = np.array([0.2, 0.5, 0.3])
    assert pairwise_tv_distance(p, p) == pytest.approx(0.0)


def test_tv_distance_disjoint():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    assert pairwise_tv_distance(p, q) == pytest.approx(1.0)


def test_tv_distance_symmetric():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1.0, 2.0, 3.0])
    q = rng.dirichlet([3.0, 1.0, 2.0])
    assert pairwise_tv_distance(p, q) == pytest.approx(pairwise_tv_distance(q, p))


def test_tv_distance_unnormalized_input():
    # Should normalise internally
    p = np.array([2.0, 4.0, 4.0])   # sums to 10
    q = np.array([1.0, 1.0, 1.0])   # sums to 3
    # After normalisation: p=[0.2, 0.4, 0.4], q=[1/3, 1/3, 1/3]
    expected = 0.5 * (abs(0.2 - 1/3) + abs(0.4 - 1/3) + abs(0.4 - 1/3))
    assert pairwise_tv_distance(p, q) == pytest.approx(expected, abs=1e-9)


def test_tv_distance_in_unit_interval():
    rng = np.random.default_rng(42)
    for _ in range(20):
        p = rng.dirichlet(rng.uniform(0.1, 2.0, size=10))
        q = rng.dirichlet(rng.uniform(0.1, 2.0, size=10))
        tv = pairwise_tv_distance(p, q)
        assert 0.0 <= tv <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# group_sensitivity
# ---------------------------------------------------------------------------

def test_group_sensitivity_empty():
    assert group_sensitivity([]) == pytest.approx(0.0)


def test_group_sensitivity_single_pair():
    assert group_sensitivity([0.3]) == pytest.approx(0.3)


def test_group_sensitivity_multiple_pairs():
    assert group_sensitivity([0.2, 0.4, 0.6]) == pytest.approx(0.4)


def test_group_sensitivity_identical_distributions():
    assert group_sensitivity([0.0, 0.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# group_contradiction_rate
# ---------------------------------------------------------------------------

def test_group_contradiction_rate_empty():
    assert group_contradiction_rate([]) == pytest.approx(0.0)


def test_group_contradiction_rate_all_agree():
    assert group_contradiction_rate([True, True, True]) == pytest.approx(0.0)


def test_group_contradiction_rate_all_disagree():
    assert group_contradiction_rate([False, False, False]) == pytest.approx(1.0)


def test_group_contradiction_rate_half():
    assert group_contradiction_rate([True, False]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# sensitivity_from_prob_matrix
# ---------------------------------------------------------------------------

def test_sensitivity_single_sequence():
    prob_matrix = np.array([[0.1, 0.7, 0.2]])
    result = sensitivity_from_prob_matrix(prob_matrix)
    assert result.semantic_sensitivity == pytest.approx(0.0)
    assert result.contradiction_rate == pytest.approx(0.0)
    assert result.n_pairs == 0


def test_sensitivity_identical_rows():
    row = np.array([0.1, 0.6, 0.3])
    prob_matrix = np.stack([row, row, row])
    result = sensitivity_from_prob_matrix(prob_matrix)
    assert result.semantic_sensitivity == pytest.approx(0.0)
    assert result.contradiction_rate == pytest.approx(0.0)
    assert result.n_pairs == 3   # 3*(3-1)/2


def test_sensitivity_two_disjoint_distributions():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    prob_matrix = np.stack([p, q])
    result = sensitivity_from_prob_matrix(prob_matrix)
    assert result.semantic_sensitivity == pytest.approx(1.0)
    assert result.contradiction_rate == pytest.approx(1.0)
    assert result.semantic_stability_score == pytest.approx(0.0)


def test_sensitivity_three_sequences():
    # Two sequences agree, one disagrees
    p1 = np.array([0.8, 0.1, 0.1])
    p2 = np.array([0.7, 0.2, 0.1])   # rank-1 same as p1
    p3 = np.array([0.0, 0.0, 1.0])   # rank-1 different
    prob_matrix = np.stack([p1, p2, p3])
    result = sensitivity_from_prob_matrix(prob_matrix)
    assert result.n_pairs == 3
    # pairs: (p1,p2) agree, (p1,p3) disagree, (p2,p3) disagree -> CR = 2/3
    assert result.contradiction_rate == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert 0.0 < result.semantic_sensitivity < 1.0


def test_sensitivity_result_sss_plus_ss_equals_one():
    rng = np.random.default_rng(7)
    prob_matrix = rng.dirichlet(np.ones(5), size=4)
    result = sensitivity_from_prob_matrix(prob_matrix)
    assert result.semantic_sensitivity + result.semantic_stability_score == pytest.approx(1.0)


def test_sensitivity_unnormalized_rows():
    # Each row unnormalized — should give same result as normalized
    p = np.array([2.0, 6.0, 2.0])   # normalized: [0.2, 0.6, 0.2]
    q = np.array([1.0, 1.0, 8.0])   # normalized: [0.1, 0.1, 0.8]
    result_unnorm = sensitivity_from_prob_matrix(np.stack([p, q]))
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    result_norm = sensitivity_from_prob_matrix(np.stack([p_norm, q_norm]))
    assert result_unnorm.semantic_sensitivity == pytest.approx(result_norm.semantic_sensitivity)


# ---------------------------------------------------------------------------
# from_groups
# ---------------------------------------------------------------------------

def test_from_groups_empty():
    result = from_groups([], [])
    assert result.n_groups == 0
    assert result.semantic_sensitivity == pytest.approx(0.0)
    assert result.semantic_stability_score == pytest.approx(1.0)


def test_from_groups_single_group():
    result = from_groups([[0.3, 0.5]], [[True, False]])
    assert result.n_groups == 1
    assert result.semantic_sensitivity == pytest.approx(0.4)
    assert result.contradiction_rate == pytest.approx(0.5)


def test_from_groups_two_groups():
    result = from_groups([[0.2], [0.8]], [[True], [False]])
    assert result.n_groups == 2
    # mean of group SS values: (0.2 + 0.8) / 2
    assert result.semantic_sensitivity == pytest.approx(0.5)
    # mean of group CR values: (0.0 + 1.0) / 2
    assert result.contradiction_rate == pytest.approx(0.5)


def test_from_groups_mismatched_lengths():
    with pytest.raises(AssertionError):
        from_groups([[0.1]], [])


def test_from_groups_group_weighting_equal_sizes():
    # Groups with the same number of pairs — mean over groups = mean over pairs
    result = from_groups([[0.1, 0.2], [0.5, 0.6]], [[True, True], [False, False]])
    assert result.semantic_sensitivity == pytest.approx((0.15 + 0.55) / 2)
    assert result.contradiction_rate == pytest.approx((0.0 + 1.0) / 2)


# ---------------------------------------------------------------------------
# from_pairwise_records
# ---------------------------------------------------------------------------

def _make_pairwise_records(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {
            "group_id":              i // 2,
            "tv_distance_full":      float(rng.uniform(0.0, 0.5)),
            "rank1_agreement_full":  bool(rng.integers(0, 2)),
        }
        for i in range(20)
    ]


def test_from_pairwise_records_empty():
    result = from_pairwise_records([])
    assert result.n_groups == 0
    assert result.n_pairs == 0


def test_from_pairwise_records_no_group_key():
    records = _make_pairwise_records()
    result = from_pairwise_records(records, group_key=None)
    assert result.n_groups == len(records)
    assert result.n_pairs == len(records)
    assert result.mean_group_size == pytest.approx(2.0)
    expected_ss = np.mean([r["tv_distance_full"] for r in records])
    assert result.semantic_sensitivity == pytest.approx(float(expected_ss))


def test_from_pairwise_records_with_group_key():
    records = _make_pairwise_records()
    result_grouped   = from_pairwise_records(records, group_key="group_id")
    result_ungrouped = from_pairwise_records(records, group_key=None)
    # Both should be valid SS values in [0, 1]
    assert 0.0 <= result_grouped.semantic_sensitivity <= 1.0
    assert 0.0 <= result_ungrouped.semantic_sensitivity <= 1.0
    # Grouped has fewer groups than pairs
    assert result_grouped.n_groups < result_grouped.n_pairs


def test_from_pairwise_records_filters_nones():
    records = _make_pairwise_records()
    records[0]["tv_distance_full"] = None
    result = from_pairwise_records(records)
    assert result.n_pairs == len(records) - 1


def test_from_pairwise_records_custom_fields():
    records = [
        {"my_tv": 0.1, "my_r1": True},
        {"my_tv": 0.3, "my_r1": False},
    ]
    result = from_pairwise_records(records, tv_field="my_tv", rank1_field="my_r1")
    assert result.semantic_sensitivity == pytest.approx(0.2)
    assert result.contradiction_rate == pytest.approx(0.5)


def test_from_pairwise_records_sss_complement():
    records = _make_pairwise_records()
    result = from_pairwise_records(records)
    assert result.semantic_sensitivity + result.semantic_stability_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SemanticSensitivityResult.__str__
# ---------------------------------------------------------------------------

def test_result_str_contains_key_fields():
    result = SemanticSensitivityResult(
        semantic_sensitivity=0.230,
        semantic_stability_score=0.770,
        contradiction_rate=0.256,
        n_groups=12894,
        n_pairs=12894,
        mean_group_size=2.0,
    )
    text = str(result)
    assert "0.2300" in text
    assert "0.7700" in text
    assert "0.2560" in text
    assert "12894" in text
