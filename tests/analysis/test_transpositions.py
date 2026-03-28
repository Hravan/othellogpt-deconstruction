"""
tests/analysis/test_transpositions.py
"""

import numpy as np
import pytest

from othellogpt_deconstruction.core.tokenizer import pos_to_alg
from othellogpt_deconstruction.core.board import legal_moves
from othellogpt_deconstruction.analysis.transpositions import (
    find_transpositions, summarise,
    TranspositionGroup, TrichroneSubgroup, TrichromeGroupDiff,
)

# ---------------------------------------------------------------------------
# Minimal corpus containing the known transposition pair
# ---------------------------------------------------------------------------

KNOWN_PAIR = [
    ["f5", "f6", "d3", "f4", "g5"],
    ["f5", "f4", "d3", "f6", "g5"],
]

# A few extra games to pad the corpus
EXTRA_GAMES = [
    ["f5", "d6", "c5", "f4", "e3"],
    ["c4", "c5", "e6", "c3", "b4"],
    ["f5", "f6", "c4", "c3", "e6"],
]

CORPUS = KNOWN_PAIR + EXTRA_GAMES


# ---------------------------------------------------------------------------
# find_transpositions — basic properties
# ---------------------------------------------------------------------------

def test_returns_list():
    groups = find_transpositions(CORPUS)
    assert isinstance(groups, list)


def test_empty_corpus():
    assert find_transpositions([]) == []


def test_single_game_no_groups():
    groups = find_transpositions([KNOWN_PAIR[0]])
    assert groups == []


def test_sorted_by_ply():
    groups = find_transpositions(CORPUS)
    plies = [g.ply for g in groups]
    assert plies == sorted(plies)


def test_all_groups_have_at_least_two_sequences():
    groups = find_transpositions(CORPUS)
    for g in groups:
        assert g.n_sequences >= 2


def test_no_duplicate_groups():
    groups = find_transpositions(CORPUS)
    seen = set()
    for g in groups:
        key = frozenset(tuple(s) for s in g.sequences)
        assert key not in seen
        seen.add(key)


# ---------------------------------------------------------------------------
# Known transposition pair
# ---------------------------------------------------------------------------

def test_known_pair_found():
    groups = find_transpositions(CORPUS)
    seq_sets = [frozenset(tuple(s) for s in g.sequences) for g in groups]
    known = frozenset(tuple(s) for s in KNOWN_PAIR)
    assert any(known.issubset(ss) for ss in seq_sets)


def test_known_pair_ply():
    groups = find_transpositions(CORPUS)
    for g in groups:
        if any(s == KNOWN_PAIR[0] for s in g.sequences):
            assert g.ply == 5
            break


def test_known_pair_same_othello_board():
    """Both sequences in the known pair must reach the same Othello board."""
    from othellogpt_deconstruction.core.board import replay as othello_replay
    board_a, _ = othello_replay(KNOWN_PAIR[0])
    board_b, _ = othello_replay(KNOWN_PAIR[1])
    assert (board_a == board_b).all()


def test_known_pair_same_legal_moves():
    """Both sequences must have identical legal move sets."""
    from othellogpt_deconstruction.core.board import legal_moves_after
    legal_a = set(legal_moves_after(KNOWN_PAIR[0]))
    legal_b = set(legal_moves_after(KNOWN_PAIR[1]))
    assert legal_a == legal_b


# ---------------------------------------------------------------------------
# Trichrome annotation
# ---------------------------------------------------------------------------

def test_known_pair_has_trichrome_diffs():
    groups = find_transpositions(CORPUS)
    for g in groups:
        if any(s == KNOWN_PAIR[0] for s in g.sequences):
            assert g.has_trichrome_diffs
            break


def test_known_pair_trichrome_diff_cells():
    groups = find_transpositions(CORPUS)
    for g in groups:
        if any(s == KNOWN_PAIR[0] for s in g.sequences):
            assert g.n_trichrome_states == 2
            diff = g.trichrome_diffs[0]
            squares = {c["square"] for c in diff.differing_cells}
            assert squares == {"e4", "e5"}
            break


def test_trichrome_groups_cover_all_sequences():
    groups = find_transpositions(CORPUS)
    for g in groups:
        all_seqs = [
            s
            for tg in g.trichrome_groups
            for s in tg.sequences
        ]
        assert len(all_seqs) == g.n_sequences


def test_trichrome_subgroup_boards_shape():
    groups = find_transpositions(CORPUS)
    for g in groups:
        for tg in g.trichrome_groups:
            assert tg.trichrome_board.shape == (64, 2)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

def test_summarise_empty():
    assert summarise([]) == {}


def test_summarise_keys():
    groups = find_transpositions(CORPUS)
    s = summarise(groups)
    assert "total_groups" in s
    assert "mixed_trichrome" in s
    assert "same_trichrome" in s
    assert "color_dist_counts" in s


def test_summarise_counts_add_up():
    groups = find_transpositions(CORPUS)
    s = summarise(groups)
    assert s["mixed_trichrome"] + s["same_trichrome"] == s["total_groups"]


def test_summarise_mixed_at_least_one():
    groups = find_transpositions(CORPUS)
    s = summarise(groups)
    assert s["mixed_trichrome"] >= 1
