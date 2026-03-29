"""
src/othellogpt_deconstruction/analysis/metrics.py

Distribution comparison metrics for OthelloGPT output distributions.

Two modes:
  - Full distribution: all 60 playable tokens (no filtering)
  - Legal moves only: restricted to legal moves, renormalized

Full distribution is the primary mode for the paper's argument —
two sequences with identical Othello board states should produce
identical distributions if the model has a world model. Any difference,
on legal or illegal moves, is evidence of sensitivity to token history.

Legal-only mode mirrors Li et al.'s methodology for direct comparison.
"""

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from othellogpt_deconstruction.core.tokenizer import stoi, itos, PAD_ID, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _full_probs(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract probability vectors over all 60 playable tokens (excluding PAD).
    Renormalized to sum to 1.
    """
    token_ids = [t for t in range(1, VOCAB_SIZE)]   # skip PAD_ID=0
    pa = probs_a[token_ids].cpu().numpy().astype(np.float64)
    pb = probs_b[token_ids].cpu().numpy().astype(np.float64)
    pa /= pa.sum()
    pb /= pb.sum()
    return pa, pb


def _legal_probs(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract and renormalize probabilities for legal moves only.
    Returns None if fewer than 2 legal moves.
    Raises ValueError if any legal position is missing from stoi.
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


def _compute_metrics(pa: np.ndarray, pb: np.ndarray) -> dict:
    """Compute all metrics given two renormalized probability arrays."""
    eps = 1e-10
    pa_c = np.clip(pa, eps, None)
    pb_c = np.clip(pb, eps, None)

    tv  = float(0.5 * np.sum(np.abs(pa - pb)))
    js  = float(jensenshannon(pa, pb, base=2))
    kl_ab = float(np.sum(pa_c * np.log2(pa_c / pb_c)))
    kl_ba = float(np.sum(pb_c * np.log2(pb_c / pa_c)))

    rho: float | None
    if np.allclose(pa, pa[0]) or np.allclose(pb, pb[0]):
        rho = None
    else:
        rho = float(spearmanr(pa, pb).statistic)

    r1 = bool(np.argmax(pa) == np.argmax(pb))

    return {
        "tv_distance":     tv,
        "js_divergence":   js,
        "kl_ab":           kl_ab,
        "kl_ba":           kl_ba,
        "spearman_rho":    rho,
        "rank1_agreement": r1,
    }


# ---------------------------------------------------------------------------
# Full-distribution metrics
# ---------------------------------------------------------------------------

def metrics_full(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
) -> dict:
    """
    Compute all metrics over the full 60-token distribution.

    This is the primary mode for the paper — no filtering, captures
    distributional differences on both legal and illegal moves.
    Keys are suffixed with '_full'.
    """
    pa, pb = _full_probs(probs_a, probs_b)
    return {f"{k}_full": v for k, v in _compute_metrics(pa, pb).items()}


# ---------------------------------------------------------------------------
# Legal-only metrics
# ---------------------------------------------------------------------------

def metrics_legal(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> dict:
    """
    Compute all metrics restricted to legal moves only.

    Mirrors Li et al.'s methodology. Returns None values if
    fewer than 2 legal moves exist.
    Keys are suffixed with '_legal'.
    """
    result = _legal_probs(probs_a, probs_b, legal_positions)
    if result is None:
        return {f"{k}_legal": None for k in [
            "tv_distance", "js_divergence", "kl_ab", "kl_ba",
            "spearman_rho", "rank1_agreement",
        ]}
    pa, pb = result
    return {f"{k}_legal": v for k, v in _compute_metrics(pa, pb).items()}


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def all_metrics(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    legal_positions: list[int],
) -> dict:
    """
    Compute both full and legal-only metrics and return as a single dict.
    """
    return {**metrics_full(probs_a, probs_b),
            **metrics_legal(probs_a, probs_b, legal_positions)}


# ---------------------------------------------------------------------------
# Rank utilities
# ---------------------------------------------------------------------------

def rank_of(target_pos: int, probs: torch.Tensor) -> int | None:
    """
    1-based rank of target_pos in the full probability distribution.
    Returns None if target_pos is not in the vocabulary.
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
    If legal_positions provided, restricts to legal moves only.
    """
    if legal_positions is not None:
        result = _legal_probs(probs, probs, legal_positions)
        if result is None:
            return None
        pa, _ = result
        token_ids = [stoi[p] for p in legal_positions]
        best_token = token_ids[int(np.argmax(pa))]
        return int(itos[best_token]), float(probs[best_token])
    else:
        best_token = int(probs.argmax())
        if best_token == PAD_ID:
            return None
        return int(itos[best_token]), float(probs[best_token])
