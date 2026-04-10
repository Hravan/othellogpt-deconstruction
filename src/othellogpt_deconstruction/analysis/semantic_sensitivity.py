"""
src/othellogpt_deconstruction/analysis/semantic_sensitivity.py

Semantic Sensitivity metric for language models.

Formalises the degree to which a model's output distribution varies across
semantically equivalent inputs — inputs that represent identical world states.

Definitions
-----------
Let M be a model, Γ be a distribution over semantic equivalence classes,
and [x] = {x₁, ..., xₙ} denote the equivalence class of size n containing
sequence x.

**Semantic Sensitivity (SS)**:

    SS(M, [x]) = (2 / n(n-1)) * Σᵢ<ⱼ TV(M(xᵢ), M(xⱼ))

    SS(M, Γ)   = E_{[x] ~ Γ}[SS(M, [x])]

where TV(p, q) = ½ Σᵥ |p(v) - q(v)| is total variation distance.
SS ∈ [0, 1]: 0 means the model is perfectly invariant to path; 1 means
maximally inconsistent.

**Semantic Stability Score (SSS)**:

    SSS(M, Γ) = 1 - SS(M, Γ)

SSS ∈ [0, 1]: 1 means perfectly stable; 0 means maximally inconsistent.

**Contradiction Rate (CR)**:

    CR(M, [x]) = (2 / n(n-1)) * Σᵢ<ⱼ 1[argmax M(xᵢ) ≠ argmax M(xⱼ)]

    CR(M, Γ)   = E_{[x] ~ Γ}[CR(M, [x])]

CR is the fraction of equivalence-class pairs where the top-1 predicted
token disagrees. CR ≤ SS always (by Markov: rank-1 disagreement implies
TV ≥ probability mass of the winning token, which bounds TV from below).

**Mechanistic predictor**:
    When ground-truth semantic equivalence classes are available, we can
    regress SS per group on properties of the underlying state difference
    to ask: what surface features of the input history predict how much the
    model's distribution varies? For OthelloGPT, the Trichrome diff
    (number of cells with different flip counts) is a natural predictor,
    since it encodes exactly what path-dependent information the model has
    access to.

For OthelloGPT, semantic equivalence classes are transposition groups —
sets of move sequences reaching the same Othello board state at the same
ply. A model with a genuine world model of board state should have SS ≈ 0,
since the board uniquely determines the legal move set.

API
---
from_pairwise_records(records)
    Compute SS, SSS, CR from a list of pairwise records.
    Each record corresponds to one pair from a transposition group.
    If a group_key is provided, SS is computed as the mean of group-level
    SS values (proper expectation over groups). Otherwise it is the mean
    over all pairs (equivalent when all groups have size 2).

from_groups(group_tv_distances, group_cr_rates)
    Compute SS, SSS, CR from group-level pre-aggregated values.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SemanticSensitivityResult:
    """
    Aggregated Semantic Sensitivity metrics for a model on a corpus of
    semantic equivalence classes.
    """
    semantic_sensitivity:       float   # SS(M, Γ) — mean TV distance
    semantic_stability_score:   float   # SSS = 1 - SS
    contradiction_rate:         float   # CR(M, Γ) — fraction of disagreeing pairs
    n_groups:                   int     # number of equivalence classes
    n_pairs:                    int     # total number of pairwise comparisons
    mean_group_size:            float   # average |[x]|

    def __str__(self) -> str:
        lines = [
            f"Semantic Sensitivity (SS):       {self.semantic_sensitivity:.4f}",
            f"Semantic Stability Score (SSS):  {self.semantic_stability_score:.4f}",
            f"Contradiction Rate (CR):         {self.contradiction_rate:.4f}",
            f"Groups:                          {self.n_groups}",
            f"Pairs:                           {self.n_pairs}",
            f"Mean group size:                 {self.mean_group_size:.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Group-level aggregation
# ---------------------------------------------------------------------------

def group_sensitivity(tv_distances: Sequence[float]) -> float:
    """
    Compute SS for a single equivalence class from its pairwise TV distances.

    Parameters
    ----------
    tv_distances : sequence of TV distances between all pairs in the group
                   Length must equal n*(n-1)/2 for a group of size n.

    Returns
    -------
    Mean pairwise TV distance (= SS for this group).
    SS ∈ [0, 1].
    """
    if not tv_distances:
        return 0.0
    return float(np.mean(tv_distances))


def group_contradiction_rate(rank1_agreements: Sequence[bool]) -> float:
    """
    Compute CR for a single equivalence class from pairwise rank-1 agreement flags.

    Parameters
    ----------
    rank1_agreements : sequence of bools, True if argmax agrees for a pair.
                       Length must equal n*(n-1)/2 for a group of size n.

    Returns
    -------
    Fraction of pairs where rank-1 disagrees = 1 - mean(rank1_agreements).
    CR ∈ [0, 1].
    """
    if not rank1_agreements:
        return 0.0
    return float(1.0 - np.mean(rank1_agreements))


def _n_pairs_from_group_size(group_size: int) -> int:
    """Number of distinct pairs in a group of size n."""
    return group_size * (group_size - 1) // 2


# ---------------------------------------------------------------------------
# from_groups
# ---------------------------------------------------------------------------

def from_groups(
    group_tv_distances: list[list[float]],
    group_rank1_agreements: list[list[bool]],
) -> SemanticSensitivityResult:
    """
    Compute SS, SSS, CR from group-level data (pre-aggregated pairwise values).

    Parameters
    ----------
    group_tv_distances       : for each group, list of pairwise TV distances
    group_rank1_agreements   : for each group, list of pairwise rank-1 agreement flags

    Returns
    -------
    SemanticSensitivityResult with proper expectation over groups.
    """
    assert len(group_tv_distances) == len(group_rank1_agreements), (
        "group_tv_distances and group_rank1_agreements must have the same length"
    )

    if not group_tv_distances:
        return SemanticSensitivityResult(
            semantic_sensitivity=0.0,
            semantic_stability_score=1.0,
            contradiction_rate=0.0,
            n_groups=0,
            n_pairs=0,
            mean_group_size=0.0,
        )

    group_ss_values = [group_sensitivity(tv_dist) for tv_dist in group_tv_distances]
    group_cr_values = [group_contradiction_rate(r1a) for r1a in group_rank1_agreements]

    corpus_ss = float(np.mean(group_ss_values))
    corpus_cr = float(np.mean(group_cr_values))

    group_sizes = [
        int(round((1 + (1 + 8 * len(tv_dist)) ** 0.5) / 2))
        for tv_dist in group_tv_distances
    ]
    total_pairs = sum(_n_pairs_from_group_size(s) for s in group_sizes)

    return SemanticSensitivityResult(
        semantic_sensitivity=corpus_ss,
        semantic_stability_score=1.0 - corpus_ss,
        contradiction_rate=corpus_cr,
        n_groups=len(group_tv_distances),
        n_pairs=total_pairs,
        mean_group_size=float(np.mean(group_sizes)),
    )


# ---------------------------------------------------------------------------
# from_pairwise_records
# ---------------------------------------------------------------------------

def from_pairwise_records(
    records: list[dict],
    tv_field: str = "tv_distance_full",
    rank1_field: str = "rank1_agreement_full",
    group_key: str | None = None,
) -> SemanticSensitivityResult:
    """
    Compute SS, SSS, CR from a list of pairwise records.

    Parameters
    ----------
    records     : list of dicts, each representing one pair of sequences
    tv_field    : dict key for TV distance (float)
    rank1_field : dict key for rank-1 agreement (bool or 0/1)
    group_key   : optional dict key identifying which equivalence class a
                  pair belongs to. If provided, SS is the mean of group-level
                  SS values (proper expectation over groups). If None, each
                  pair is treated as its own group of size 2 — equivalent to
                  the mean over all pairs when all groups have size 2.

    Returns
    -------
    SemanticSensitivityResult
    """
    valid = [
        r for r in records
        if r.get(tv_field) is not None and r.get(rank1_field) is not None
    ]

    if not valid:
        return SemanticSensitivityResult(
            semantic_sensitivity=0.0,
            semantic_stability_score=1.0,
            contradiction_rate=0.0,
            n_groups=0,
            n_pairs=0,
            mean_group_size=0.0,
        )

    if group_key is None:
        # Each pair is its own group of size 2 (one pair per group).
        # SS = mean TV distance; CR = fraction with rank-1 disagreement.
        tv_values   = [float(r[tv_field]) for r in valid]
        rank1_agree = [bool(r[rank1_field]) for r in valid]
        corpus_ss   = float(np.mean(tv_values))
        corpus_cr   = float(np.mean([not a for a in rank1_agree]))
        return SemanticSensitivityResult(
            semantic_sensitivity=corpus_ss,
            semantic_stability_score=1.0 - corpus_ss,
            contradiction_rate=corpus_cr,
            n_groups=len(valid),
            n_pairs=len(valid),
            mean_group_size=2.0,
        )

    # Group records by group_key and compute group-level SS and CR.
    groups: dict[object, list[dict]] = {}
    for record in valid:
        key = record[group_key]
        groups.setdefault(key, []).append(record)

    group_tv_distances = [
        [float(r[tv_field]) for r in group_records]
        for group_records in groups.values()
    ]
    group_rank1_agreements = [
        [bool(r[rank1_field]) for r in group_records]
        for group_records in groups.values()
    ]

    return from_groups(group_tv_distances, group_rank1_agreements)


# ---------------------------------------------------------------------------
# Pairwise computation from raw probability arrays
# ---------------------------------------------------------------------------

def pairwise_tv_distance(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> float:
    """
    TV distance between two probability distributions over a common support.

    TV(p, q) = ½ Σᵥ |p(v) - q(v)|

    Parameters
    ----------
    probs_a, probs_b : 1-D probability arrays (need not be normalised; will be
                       normalised internally)

    Returns
    -------
    TV distance in [0, 1].
    """
    pa = np.asarray(probs_a, dtype=np.float64)
    pb = np.asarray(probs_b, dtype=np.float64)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(0.5 * np.sum(np.abs(pa - pb)))


def sensitivity_from_prob_matrix(
    prob_matrix: np.ndarray,
) -> SemanticSensitivityResult:
    """
    Compute SS, SSS, CR for a single equivalence class directly from a
    probability matrix.

    Parameters
    ----------
    prob_matrix : array of shape (n_sequences, vocab_size)
                  Each row is the model's output distribution for one sequence
                  in the equivalence class. Rows will be normalised internally.

    Returns
    -------
    SemanticSensitivityResult with n_groups=1.
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    n_sequences = prob_matrix.shape[0]

    if n_sequences < 2:
        return SemanticSensitivityResult(
            semantic_sensitivity=0.0,
            semantic_stability_score=1.0,
            contradiction_rate=0.0,
            n_groups=1,
            n_pairs=0,
            mean_group_size=float(n_sequences),
        )

    # Normalise each row
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = prob_matrix / row_sums

    top1_tokens = np.argmax(prob_matrix, axis=1)

    tv_distances  = []
    rank1_agrees  = []
    for i, j in combinations(range(n_sequences), 2):
        tv = float(0.5 * np.sum(np.abs(prob_matrix[i] - prob_matrix[j])))
        tv_distances.append(tv)
        rank1_agrees.append(bool(top1_tokens[i] == top1_tokens[j]))

    ss = group_sensitivity(tv_distances)
    cr = group_contradiction_rate(rank1_agrees)

    return SemanticSensitivityResult(
        semantic_sensitivity=ss,
        semantic_stability_score=1.0 - ss,
        contradiction_rate=cr,
        n_groups=1,
        n_pairs=len(tv_distances),
        mean_group_size=float(n_sequences),
    )
