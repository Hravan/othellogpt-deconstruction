"""
tests/core/test_board.py
"""

import pytest
import numpy as np

from othellogpt_deconstruction.core.tokenizer import alg_to_pos, pos_to_alg
from othellogpt_deconstruction.core.board import (
    EMPTY, BLACK, WHITE,
    start_board, flipped_by, is_legal, legal_moves,
    apply_move, replay, legal_moves_after,
)


# ---------------------------------------------------------------------------
# Starting position
# ---------------------------------------------------------------------------

def test_start_board_shape():
    b = start_board()
    assert b.shape == (64,)


def test_start_board_center():
    b = start_board()
    assert b[alg_to_pos("d4")] == WHITE
    assert b[alg_to_pos("e4")] == BLACK
    assert b[alg_to_pos("d5")] == BLACK
    assert b[alg_to_pos("e5")] == WHITE


def test_start_board_disc_count():
    b = start_board()
    assert (b == BLACK).sum() == 2
    assert (b == WHITE).sum() == 2
    assert (b == EMPTY).sum() == 60


def test_start_board_is_mutable():
    b = start_board()
    b[0] = BLACK   # should not raise
    assert b[0] == BLACK


def test_start_board_copies_are_independent():
    b1 = start_board()
    b2 = start_board()
    b1[0] = BLACK
    assert b2[0] == EMPTY


# ---------------------------------------------------------------------------
# Legal moves from starting position
# ---------------------------------------------------------------------------

def test_start_legal_moves_black():
    b = start_board()
    moves = set(pos_to_alg(p) for p in legal_moves(b, BLACK))
    assert moves == {"c4", "d3", "e6", "f5"}


def test_start_legal_moves_white():
    b = start_board()
    moves = set(pos_to_alg(p) for p in legal_moves(b, WHITE))
    assert moves == {"c5", "d6", "e3", "f4"}


def test_start_legal_moves_count():
    b = start_board()
    assert len(legal_moves(b, BLACK)) == 4
    assert len(legal_moves(b, WHITE)) == 4


# ---------------------------------------------------------------------------
# flipped_by
# ---------------------------------------------------------------------------

def test_flipped_by_f5():
    b = start_board()
    flips = flipped_by(b, alg_to_pos("f5"), BLACK)
    assert set(pos_to_alg(p) for p in flips) == {"e5"}


def test_flipped_by_illegal_occupied():
    b = start_board()
    assert flipped_by(b, alg_to_pos("d4"), BLACK) == []


def test_flipped_by_illegal_no_flank():
    b = start_board()
    assert flipped_by(b, alg_to_pos("a1"), BLACK) == []


# ---------------------------------------------------------------------------
# apply_move
# ---------------------------------------------------------------------------

def test_apply_move_f5():
    b = start_board()
    b2 = apply_move(b, alg_to_pos("f5"), BLACK)
    assert b2[alg_to_pos("f5")] == BLACK
    assert b2[alg_to_pos("e5")] == BLACK   # flipped
    assert b2[alg_to_pos("d5")] == BLACK   # unchanged
    assert b2[alg_to_pos("d4")] == WHITE   # unchanged


def test_apply_move_does_not_modify_input():
    b = start_board()
    _ = apply_move(b, alg_to_pos("f5"), BLACK)
    assert b[alg_to_pos("f5")] == EMPTY
    assert b[alg_to_pos("e5")] == WHITE


def test_apply_move_illegal_raises():
    b = start_board()
    with pytest.raises(ValueError):
        apply_move(b, alg_to_pos("a1"), BLACK)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def test_replay_empty():
    board, player = replay([])
    assert (board == start_board()).all()
    assert player == BLACK


def test_replay_one_move():
    board, player = replay(["f5"])
    assert board[alg_to_pos("f5")] == BLACK
    assert board[alg_to_pos("e5")] == BLACK
    assert player == WHITE


def test_replay_alternates_players():
    board, player = replay(["f5", "f6"])
    assert board[alg_to_pos("f5")] == BLACK
    assert board[alg_to_pos("f6")] == WHITE
    assert player == BLACK


def test_replay_illegal_raises():
    with pytest.raises(ValueError):
        replay(["a1"])


# ---------------------------------------------------------------------------
# legal_moves_after — the key function used in statistics pipeline
# ---------------------------------------------------------------------------

def test_legal_moves_after_known_pair():
    """
    The two sequences from the paper's first transposition pair must have
    identical legal move sets after replay.
    """
    seq_a = ["f5", "f6", "d3", "f4", "g5"]
    seq_b = ["f5", "f4", "d3", "f6", "g5"]
    legal_a = set(legal_moves_after(seq_a))
    legal_b = set(legal_moves_after(seq_b))
    assert legal_a == legal_b


def test_legal_moves_after_nonempty():
    seq = ["f5", "f6", "d3", "f4", "g5"]
    assert len(legal_moves_after(seq)) > 0


def test_legal_moves_after_known_pair_algebraic():
    seq_a = ["f5", "f6", "d3", "f4", "g5"]
    moves = set(pos_to_alg(p) for p in legal_moves_after(seq_a))
    assert "c4" in moves   # confirmed legal from earlier debugging
