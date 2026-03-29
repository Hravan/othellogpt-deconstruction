"""
tests/analysis/test_metrics.py
"""

import numpy as np
import torch
import pytest

from othellogpt_deconstruction.core.tokenizer import stoi, itos, VOCAB_SIZE, alg_to_pos
from othellogpt_deconstruction.core.board import legal_moves_after
from othellogpt_deconstruction.analysis.metrics import (
    tv_distance, js_divergence, kl_divergence, spearman_rho,
    rank1_agreement, rank_of, top1_info, all_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures — use verified legal moves from the known pair
# ---------------------------------------------------------------------------

SEQ = ["f5", "f6", "d3", "f4", "g5"]
LEGAL_POSITIONS = legal_moves_after(SEQ)   # computed from actual board state
LEGAL_TOKENS = [stoi[p] for p in LEGAL_POSITIONS]

assert len(LEGAL_POSITIONS) >= 2, "Need at least 2 legal moves for tests"


def uniform_probs() -> torch.Tensor:
    p = torch.zeros(VOCAB_SIZE)
    for t in LEGAL_TOKENS:
        p[t] = 1.0 / len(LEGAL_TOKENS)
    return p


def peaked_probs_a() -> torch.Tensor:
    """All mass on first legal move."""
    p = torch.zeros(VOCAB_SIZE)
    p[LEGAL_TOKENS[0]] = 1.0
    return p


def peaked_probs_b() -> torch.Tensor:
    """All mass on second legal move."""
    p = torch.zeros(VOCAB_SIZE)
    p[LEGAL_TOKENS[1]] = 1.0
    return p


def weighted_probs() -> torch.Tensor:
    """Non-uniform distribution over legal moves."""
    p = torch.zeros(VOCAB_SIZE)
    n = len(LEGAL_TOKENS)
    weights = np.array([1.0 / (i + 1) for i in range(n)])
    weights /= weights.sum()
    for t, w in zip(LEGAL_TOKENS, weights):
        p[t] = float(w)
    return p


# ---------------------------------------------------------------------------
# tv_distance
# ---------------------------------------------------------------------------

def test_tv_identical():
    p = weighted_probs()
    assert tv_distance(p, p, LEGAL_POSITIONS) == pytest.approx(0.0)


def test_tv_orthogonal():
    pa = peaked_probs_a()
    pb = peaked_probs_b()
    assert tv_distance(pa, pb, LEGAL_POSITIONS) == pytest.approx(1.0)


def test_tv_symmetric():
    pa = weighted_probs()
    pb = uniform_probs()
    assert tv_distance(pa, pb, LEGAL_POSITIONS) == pytest.approx(
        tv_distance(pb, pa, LEGAL_POSITIONS)
    )


def test_tv_range():
    pa = weighted_probs()
    pb = uniform_probs()
    result = tv_distance(pa, pb, LEGAL_POSITIONS)
    assert 0.0 <= result <= 1.0


def test_tv_none_for_single_legal_move():
    assert tv_distance(peaked_probs_a(), peaked_probs_b(), [LEGAL_POSITIONS[0]]) is None


def test_tv_missing_position_raises():
    from othellogpt_deconstruction.core.tokenizer import START_SQUARES
    invalid = list(START_SQUARES)[:2]
    with pytest.raises(ValueError):
        tv_distance(uniform_probs(), uniform_probs(), invalid)


# ---------------------------------------------------------------------------
# js_divergence
# ---------------------------------------------------------------------------

def test_js_identical():
    p = weighted_probs()
    assert js_divergence(p, p, LEGAL_POSITIONS) == pytest.approx(0.0, abs=1e-6)


def test_js_orthogonal():
    pa = peaked_probs_a()
    pb = peaked_probs_b()
    assert js_divergence(pa, pb, LEGAL_POSITIONS) == pytest.approx(1.0, abs=1e-6)


def test_js_symmetric():
    pa = weighted_probs()
    pb = uniform_probs()
    assert js_divergence(pa, pb, LEGAL_POSITIONS) == pytest.approx(
        js_divergence(pb, pa, LEGAL_POSITIONS)
    )


def test_js_range():
    pa = weighted_probs()
    pb = uniform_probs()
    result = js_divergence(pa, pb, LEGAL_POSITIONS)
    assert 0.0 <= result <= 1.0


def test_js_none_for_single_legal_move():
    assert js_divergence(peaked_probs_a(), peaked_probs_b(), [LEGAL_POSITIONS[0]]) is None


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

def test_kl_identical():
    p = weighted_probs()
    kl_ab, kl_ba = kl_divergence(p, p, LEGAL_POSITIONS)
    assert kl_ab == pytest.approx(0.0, abs=1e-6)
    assert kl_ba == pytest.approx(0.0, abs=1e-6)


def test_kl_asymmetric():
    pa = weighted_probs()
    pb = uniform_probs()
    kl_ab, kl_ba = kl_divergence(pa, pb, LEGAL_POSITIONS)
    assert kl_ab != pytest.approx(kl_ba, abs=1e-6)


def test_kl_nonnegative():
    pa = weighted_probs()
    pb = uniform_probs()
    kl_ab, kl_ba = kl_divergence(pa, pb, LEGAL_POSITIONS)
    assert kl_ab >= 0.0
    assert kl_ba >= 0.0


def test_kl_none_for_single_legal_move():
    kl_ab, kl_ba = kl_divergence(peaked_probs_a(), peaked_probs_b(), [LEGAL_POSITIONS[0]])
    assert kl_ab is None
    assert kl_ba is None


# ---------------------------------------------------------------------------
# spearman_rho
# ---------------------------------------------------------------------------

def test_spearman_identical():
    p = weighted_probs()
    assert spearman_rho(p, p, LEGAL_POSITIONS) == pytest.approx(1.0)


def test_spearman_reversed():
    pa = torch.zeros(VOCAB_SIZE)
    pb = torch.zeros(VOCAB_SIZE)
    n = len(LEGAL_TOKENS)
    weights = np.array([1.0 / (i + 1) for i in range(n)])
    weights /= weights.sum()
    for t, w in zip(LEGAL_TOKENS, weights):
        pa[t] = float(w)
    for t, w in zip(reversed(LEGAL_TOKENS), weights):
        pb[t] = float(w)
    rho = spearman_rho(pa, pb, LEGAL_POSITIONS)
    assert rho == pytest.approx(-1.0, abs=1e-6)


def test_spearman_range():
    pa = weighted_probs()
    pb = peaked_probs_a()
    rho = spearman_rho(pa, pb, LEGAL_POSITIONS)
    assert rho is not None
    assert -1.0 <= rho <= 1.0


def test_spearman_none_for_constant_input():
    p = weighted_probs()
    assert spearman_rho(p, uniform_probs(), LEGAL_POSITIONS) is None


def test_spearman_none_for_single_legal_move():
    assert spearman_rho(peaked_probs_a(), peaked_probs_b(), [LEGAL_POSITIONS[0]]) is None


# ---------------------------------------------------------------------------
# rank1_agreement
# ---------------------------------------------------------------------------

def test_rank1_agreement_identical():
    p = weighted_probs()
    assert rank1_agreement(p, p, LEGAL_POSITIONS) is True


def test_rank1_agreement_different():
    pa = peaked_probs_a()
    pb = peaked_probs_b()
    assert rank1_agreement(pa, pb, LEGAL_POSITIONS) is False


def test_rank1_agreement_none_for_single():
    assert rank1_agreement(peaked_probs_a(), peaked_probs_b(), [LEGAL_POSITIONS[0]]) is None


# ---------------------------------------------------------------------------
# rank_of
# ---------------------------------------------------------------------------

def test_rank_of_top():
    p = weighted_probs()
    top_pos = LEGAL_POSITIONS[0]
    assert rank_of(top_pos, p) == 1


def test_rank_of_invalid_position():
    p = weighted_probs()
    from othellogpt_deconstruction.core.tokenizer import START_SQUARES
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
    assert prob == pytest.approx(1.0)


def test_top1_info_none_for_empty_legal():
    p = weighted_probs()
    assert top1_info(p, []) is None


# ---------------------------------------------------------------------------
# all_metrics
# ---------------------------------------------------------------------------

def test_all_metrics_keys():
    pa = weighted_probs()
    pb = uniform_probs()
    result = all_metrics(pa, pb, LEGAL_POSITIONS)
    assert set(result.keys()) == {
        "tv_distance", "js_divergence", "kl_ab", "kl_ba",
        "spearman_rho", "rank1_agreement",
    }


def test_all_metrics_identical():
    p = weighted_probs()
    result = all_metrics(p, p, LEGAL_POSITIONS)
    assert result["tv_distance"] == pytest.approx(0.0)
    assert result["js_divergence"] == pytest.approx(0.0, abs=1e-6)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["rank1_agreement"] is True


def test_all_metrics_orthogonal():
    pa = peaked_probs_a()
    pb = peaked_probs_b()
    result = all_metrics(pa, pb, LEGAL_POSITIONS)
    assert result["tv_distance"] == pytest.approx(1.0)
    assert result["rank1_agreement"] is False
