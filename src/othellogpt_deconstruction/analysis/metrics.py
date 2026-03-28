"""
src/othellogpt_deconstruction/analysis/metrics.py

Distribution comparison metrics for OthelloGPT output distributions.

All metrics operate on probability distributions restricted to legal moves.
Inputs are raw probability vectors over the full 61-token vocabulary;
legal move filtering and renormalization are handled internally.
"""

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from othellogpt_deconstruction.core.tokenizer import stoi, PAD_ID


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_legal(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract and renormalize probabilities for legal moves only.

    Parameters
    ----------
    probs_a, probs_b : probability vectors of shape (VOCAB_SIZE,)
    legal_positions  : list of board positions (not token ids)

    Returns
    -------
    (pa, pb) renormalized numpy arrays, or None if fewer than 2 legal moves.

    Raises
    ------
    ValueError if any legal position is missing from stoi.
    """
    if len(legal_positions) < 2:
        return None

    missing = [p for p in legal_positions if p not in stoi]
    if missing:
        raise ValueError(f"Legal positions missing from vocabulary: {missing}")

    token_ids = [stoi[p] for p in legal_positions]
    pa = probs_a[token_ids].cpu().numpy().astype(np.float64)
    pb = probs_b[token_ids].cpu().numpy().astype(np.float64)
    pa /= pa.sum()
    pb /= pb.sum()
    return pa, pb


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tv_distance(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> float | None:
    """
    Total variation distance over legal moves.

    TV(p, q) = 0.5 * sum |p_i - q_i|

    Range: [0, 1]. 0 = identical distributions, 1 = no overlap.
    Captures probability mass shifts that rank-based metrics miss.
    """
    result = _extract_legal(probs_a, probs_b, legal_positions)
    if result is None:
        return None
    pa, pb = result
    return float(0.5 * np.sum(np.abs(pa - pb)))


def js_divergence(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> float | None:
    """
    Jensen-Shannon divergence over legal moves (base-2 log).

    Symmetric version of KL divergence.
    Range: [0, 1]. 0 = identical, 1 = maximally different.
    """
    result = _extract_legal(probs_a, probs_b, legal_positions)
    if result is None:
        return None
    pa, pb = result
    return float(jensenshannon(pa, pb, base=2))


def kl_divergence(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> tuple[float, float] | tuple[None, None]:
    """
    KL divergences D(a||b) and D(b||a) over legal moves (base-2 log).

    Asymmetric — both directions returned as (kl_ab, kl_ba).
    Uses epsilon clipping to avoid log(0).
    """
    result = _extract_legal(probs_a, probs_b, legal_positions)
    if result is None:
        return None, None
    pa, pb = result
    eps = 1e-10
    pa = np.clip(pa, eps, None)
    pb = np.clip(pb, eps, None)
    kl_ab = float(np.sum(pa * np.log2(pa / pb)))
    kl_ba = float(np.sum(pb * np.log2(pb / pa)))
    return kl_ab, kl_ba


def spearman_rho(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> float | None:
    """
    Spearman rank correlation over legal moves.

    Range: [-1, 1]. 1 = identical ranking, -1 = reversed ranking.
    Captures reordering of move preferences but is insensitive to
    the magnitude of probability differences.
    """
    result = _extract_legal(probs_a, probs_b, legal_positions)
    if result is None:
        return None
    pa, pb = result
    if np.allclose(pa, pa[0]) or np.allclose(pb, pb[0]):
        return None   # constant input — rho undefined
    rho, _ = spearmanr(pa, pb)
    return float(rho)


def rank1_agreement(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> bool | None:
    """
    Whether the top-predicted legal move is the same in both distributions.
    Returns None if fewer than 2 legal moves.
    """
    result = _extract_legal(probs_a, probs_b, legal_positions)
    if result is None:
        return None
    pa, pb = result
    return bool(np.argmax(pa) == np.argmax(pb))


# ---------------------------------------------------------------------------
# Rank utilities
# ---------------------------------------------------------------------------

def rank_of(target_pos: int, probs: torch.Tensor) -> int | None:
    """
    1-based rank of target_pos in the full probability distribution.
    Returns None if target_pos is not in the vocabulary.
    Ties are broken by counting strictly higher probabilities.
    """
    if target_pos not in stoi:
        return None
    token = stoi[target_pos]
    target_prob = float(probs[token])
    return int((probs > target_prob).sum().item()) + 1


def top1_info(
    probs: torch.Tensor,
    legal_positions: list[int] | None = None,
) -> tuple[int, float] | None:
    """
    Return (board_position, probability) for the top predicted move.

    If legal_positions is provided, restricts to legal moves only.
    Returns None if no legal moves or vocabulary is empty.
    """
    from othellogpt_deconstruction.core.tokenizer import itos

    if legal_positions is not None:
        result = _extract_legal(probs, probs, legal_positions)
        if result is None:
            return None
        pa, _ = result
        token_ids = [stoi[p] for p in legal_positions]
        best_idx = int(np.argmax(pa))
        best_token = token_ids[best_idx]
        best_pos = itos[best_token]
        return int(best_pos), float(probs[best_token])
    else:
        best_token = int(probs.argmax())
        if best_token == PAD_ID:
            return None
        return int(itos[best_token]), float(probs[best_token])


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def all_metrics(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> dict:
    """
    Compute all metrics for a single pair and return as a dict.
    Convenient for building results tables.
    """
    kl_ab, kl_ba = kl_divergence(probs_a, probs_b, legal_positions)
    return {
        "tv_distance":     tv_distance(probs_a, probs_b, legal_positions),
        "js_divergence":   js_divergence(probs_a, probs_b, legal_positions),
        "kl_ab":           kl_ab,
        "kl_ba":           kl_ba,
        "spearman_rho":    spearman_rho(probs_a, probs_b, legal_positions),
        "rank1_agreement": rank1_agreement(probs_a, probs_b, legal_positions),
    }
