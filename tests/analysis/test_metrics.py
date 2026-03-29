"""
tests/analysis/test_metrics.py
"""

import numpy as np
import torch
import pytest

from othellogpt_deconstruction.core.tokenizer import stoi, itos, VOCAB_SIZE, alg_to_pos
from othellogpt_deconstruction.core.board import legal_moves_after
from othellogpt_deconstruction.analysis.metrics import (
    metrics_full, metrics_legal, all_metrics,
    rank_of, top1_info,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEQ = ["f5", "f6", "d3", "f4", "g5"]
LEGAL_POSITIONS = legal_moves_after(SEQ)
LEGAL_TOKENS = [stoi[p] for p in LEGAL_POSITIONS]

assert len(LEGAL_POSITIONS) >= 2


def uniform_probs() -> torch.Tensor:
    """Uniform over all 60 playable tokens."""
    p = torch.zeros(VOCAB_SIZE)
    for t in range(1, VOCAB_SIZE):
        p[t] = 1.0 / (VOCAB_SIZE - 1)
    return p


def peaked_probs_a() -> torch.Tensor:
    p = torch.zeros(VOCAB_SIZE)
    p[LEGAL_TOKENS[0]] = 1.0
    return p


def peaked_probs_b() -> torch.Tensor:
    p = torch.zeros(VOCAB_SIZE)
    p[LEGAL_TOKENS[1]] = 1.0
    return p


def weighted_probs() -> torch.Tensor:
    """Non-uniform distribution over legal moves only."""
    p = torch.zeros(VOCAB_SIZE)
    n = len(LEGAL_TOKENS)
    weights = np.array([1.0 / (i + 1) for i in range(n)])
    weights /= weights.sum()
    for t, w in zip(LEGAL_TOKENS, weights):
        p[t] = float(w)
    return p


def weighted_probs_full() -> torch.Tensor:
    """Non-uniform distribution over all 60 tokens."""
    p = torch.zeros(VOCAB_SIZE)
    weights = np.array([1.0 / (i + 1) for i in range(1, VOCAB_SIZE)])
    weights /= weights.sum()
    for t, w in zip(range(1, VOCAB_SIZE), weights):
        p[t] = float(w)
    return p


FULL_METRIC_KEYS = {
    "tv_distance_full", "js_divergence_full", "kl_ab_full", "kl_ba_full",
    "spearman_rho_full", "rank1_agreement_full",
}
LEGAL_METRIC_KEYS = {
    "tv_distance_legal", "js_divergence_legal", "kl_ab_legal", "kl_ba_legal",
    "spearman_rho_legal", "rank1_agreement_legal",
}


# ---------------------------------------------------------------------------
# metrics_full
# ---------------------------------------------------------------------------

def test_metrics_full_keys():
    result = metrics_full(weighted_probs_full(), uniform_probs())
    assert set(result.keys()) == FULL_METRIC_KEYS


def test_metrics_full_identical():
    p = weighted_probs_full()
    result = metrics_full(p, p)
    assert result["tv_distance_full"] == pytest.approx(0.0)
    assert result["js_divergence_full"] == pytest.approx(0.0, abs=1e-6)
    assert result["rank1_agreement_full"] is True


def test_metrics_full_orthogonal():
    pa = peaked_probs_a()
    pb = peaked_probs_b()
    result = metrics_full(pa, pb)
    assert result["tv_distance_full"] > 0.0
    assert result["rank1_agreement_full"] is False


def test_metrics_full_tv_range():
    result = metrics_full(weighted_probs_full(), uniform_probs())
    assert 0.0 <= result["tv_distance_full"] <= 1.0


def test_metrics_full_js_range():
    result = metrics_full(weighted_probs_full(), uniform_probs())
    assert 0.0 <= result["js_divergence_full"] <= 1.0


def test_metrics_full_kl_nonnegative():
    result = metrics_full(weighted_probs_full(), uniform_probs())
    assert result["kl_ab_full"] >= 0.0
    assert result["kl_ba_full"] >= 0.0


def test_metrics_full_spearman_range():
    result = metrics_full(weighted_probs_full(), peaked_probs_a())
    rho = result["spearman_rho_full"]
    assert rho is not None
    assert -1.0 <= rho <= 1.0


def test_metrics_full_spearman_none_for_uniform():
    result = metrics_full(weighted_probs_full(), uniform_probs())
    assert result["spearman_rho_full"] is None


def test_metrics_full_no_legal_positions_needed():
    """Full metrics require no legal positions — always computable."""
    pa = torch.zeros(VOCAB_SIZE)
    pa[1] = 1.0
    pb = torch.zeros(VOCAB_SIZE)
    pb[2] = 1.0
    result = metrics_full(pa, pb)
    assert result["tv_distance_full"] is not None


# ---------------------------------------------------------------------------
# metrics_legal
# ---------------------------------------------------------------------------

def test_metrics_legal_keys():
    result = metrics_legal(weighted_probs(), peaked_probs_a(), LEGAL_POSITIONS)
    assert set(result.keys()) == LEGAL_METRIC_KEYS


def test_metrics_legal_identical():
    p = weighted_probs()
    result = metrics_legal(p, p, LEGAL_POSITIONS)
    assert result["tv_distance_legal"] == pytest.approx(0.0)
    assert result["rank1_agreement_legal"] is True


def test_metrics_legal_none_for_single_move():
    result = metrics_legal(peaked_probs_a(), peaked_probs_b(), [LEGAL_POSITIONS[0]])
    assert all(v is None for v in result.values())


def test_metrics_legal_missing_position_raises():
    from othellogpt_deconstruction.core.tokenizer import START_SQUARES
    invalid = list(START_SQUARES)[:2]
    with pytest.raises(ValueError):
        metrics_legal(uniform_probs(), uniform_probs(), invalid)


def test_metrics_legal_tv_range():
    result = metrics_legal(weighted_probs(), peaked_probs_a(), LEGAL_POSITIONS)
    assert 0.0 <= result["tv_distance_legal"] <= 1.0


def test_metrics_legal_js_range():
    result = metrics_legal(weighted_probs(), peaked_probs_a(), LEGAL_POSITIONS)
    assert 0.0 <= result["js_divergence_legal"] <= 1.0


# ---------------------------------------------------------------------------
# all_metrics
# ---------------------------------------------------------------------------

def test_all_metrics_keys():
    result = all_metrics(weighted_probs(), peaked_probs_a(), LEGAL_POSITIONS)
    assert set(result.keys()) == FULL_METRIC_KEYS | LEGAL_METRIC_KEYS


def test_all_metrics_identical():
    p = weighted_probs_full()
    result = all_metrics(p, p, LEGAL_POSITIONS)
    assert result["tv_distance_full"] == pytest.approx(0.0)
    assert result["tv_distance_legal"] == pytest.approx(0.0)


def test_all_metrics_full_never_none():
    """Full metrics should never be None regardless of legal move count."""
    result = all_metrics(peaked_probs_a(), peaked_probs_b(), [LEGAL_POSITIONS[0]])
    assert result["tv_distance_full"] is not None
    assert result["js_divergence_full"] is not None


# ---------------------------------------------------------------------------
# rank_of
# ---------------------------------------------------------------------------

def test_rank_of_top():
    p = weighted_probs()
    assert rank_of(LEGAL_POSITIONS[0], p) == 1


def test_rank_of_invalid():
    from othellogpt_deconstruction.core.tokenizer import START_SQUARES
    p = weighted_probs()
    invalid = next(iter(START_SQUARES))
    assert rank_of(invalid, p) is None


def test_rank_of_range():
    p = weighted_probs()
    for pos in LEGAL_POSITIONS:
        r = rank_of(pos, p)
        assert 1 <= r <= VOCAB_SIZE


# ---------------------------------------------------------------------------
# top1_info
# ---------------------------------------------------------------------------

def test_top1_info_unrestricted():
    p = peaked_probs_a()
    pos, prob = top1_info(p)
    assert pos == LEGAL_POSITIONS[0]
    assert prob == pytest.approx(1.0)


def test_top1_info_legal_only():
    p = peaked_probs_a()
    result = top1_info(p, LEGAL_POSITIONS)
    assert result is not None
    pos, prob = result
    assert pos == LEGAL_POSITIONS[0]


def test_top1_info_none_for_empty_legal():
    assert top1_info(weighted_probs(), []) is None
