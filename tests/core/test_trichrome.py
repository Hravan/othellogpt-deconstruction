"""
tests/core/test_trichrome.py
"""

import pytest

from othellogpt_deconstruction.core.tokenizer import alg_to_pos
from othellogpt_deconstruction.core.board import EMPTY, BLACK, WHITE
from othellogpt_deconstruction.core.trichrome import (
    RED, GREEN, BLUE, color_distance, start_board, othello_projection,
    apply_move, replay, diff, trichrome_key,
)


# ---------------------------------------------------------------------------
# color_distance
# ---------------------------------------------------------------------------

def test_color_distance_same():
    assert color_distance(RED, RED) == 0
    assert color_distance(GREEN, GREEN) == 0
    assert color_distance(BLUE, BLUE) == 0


def test_color_distance_adjacent():
    assert color_distance(RED, GREEN) == 1
    assert color_distance(GREEN, BLUE) == 1
    assert color_distance(BLUE, RED) == 1


def test_color_distance_symmetric():
    assert color_distance(RED, GREEN) == color_distance(GREEN, RED)
    assert color_distance(RED, BLUE) == color_distance(BLUE, RED)


def test_color_distance_circular():
    # R->B is distance 1 going backwards (R->B via B->R)
    assert color_distance(RED, BLUE) == 1


def test_color_distance_max():
    # Maximum distance on a 3-cycle is 1
    for c1 in (RED, GREEN, BLUE):
        for c2 in (RED, GREEN, BLUE):
            assert color_distance(c1, c2) <= 1


# ---------------------------------------------------------------------------
# start_board
# ---------------------------------------------------------------------------

def test_start_board_shape():
    b = start_board()
    assert b.shape == (64, 2)


def test_start_board_center_owners():
    b = start_board()
    assert b[alg_to_pos("d4"), 0] == WHITE
    assert b[alg_to_pos("e4"), 0] == BLACK
    assert b[alg_to_pos("d5"), 0] == BLACK
    assert b[alg_to_pos("e5"), 0] == WHITE


def test_start_board_center_colors():
    b = start_board()
    for sq in ("d4", "e4", "d5", "e5"):
        assert b[alg_to_pos(sq), 1] == RED


def test_start_board_empty_cells():
    b = start_board()
    assert (b[:, 0] == EMPTY).sum() == 60


def test_start_board_is_mutable():
    b = start_board()
    b[0, 0] = BLACK  # should not raise


# ---------------------------------------------------------------------------
# othello_projection
# ---------------------------------------------------------------------------

def test_othello_projection_shape():
    b = start_board()
    assert othello_projection(b).shape == (64,)


def test_othello_projection_values():
    b = start_board()
    proj = othello_projection(b)
    assert proj[alg_to_pos("d4")] == WHITE
    assert proj[alg_to_pos("e4")] == BLACK
    assert proj[alg_to_pos("a1")] == EMPTY


def test_othello_projection_does_not_modify():
    b = start_board()
    proj = othello_projection(b)
    proj[0] = BLACK
    assert b[0, 0] == EMPTY


# ---------------------------------------------------------------------------
# apply_move
# ---------------------------------------------------------------------------

def test_apply_move_new_disc_is_red():
    b = start_board()
    b2 = apply_move(b, alg_to_pos("f5"), BLACK)
    assert b2[alg_to_pos("f5"), 0] == BLACK
    assert b2[alg_to_pos("f5"), 1] == RED


def test_apply_move_captured_disc_advances_color():
    b = start_board()
    b2 = apply_move(b, alg_to_pos("f5"), BLACK)
    # e5 was White/Red, captured by Black -> should be Green
    assert b2[alg_to_pos("e5"), 0] == BLACK
    assert b2[alg_to_pos("e5"), 1] == GREEN


def test_color_advances_on_capture():
    """Each capture advances color by exactly one step mod 3."""
    b = start_board()
    assert b[alg_to_pos("e5"), 1] == RED
    b = apply_move(b, alg_to_pos("f5"), BLACK)
    assert b[alg_to_pos("e5"), 1] == GREEN


def test_apply_move_does_not_modify_input():
    b = start_board()
    _ = apply_move(b, alg_to_pos("f5"), BLACK)
    assert b[alg_to_pos("f5"), 0] == EMPTY
    assert b[alg_to_pos("e5"), 1] == RED


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
    assert board[alg_to_pos("f5"), 0] == BLACK
    assert board[alg_to_pos("f5"), 1] == RED
    assert board[alg_to_pos("e5"), 0] == BLACK
    assert board[alg_to_pos("e5"), 1] == GREEN
    assert player == WHITE


def test_replay_othello_projection_matches_board():
    """Trichrome replay must produce same Othello state as board.replay."""
    from othellogpt_deconstruction.core.board import replay as othello_replay
    moves = ["f5", "f6", "d3", "f4", "g5"]
    tc_board, _ = replay(moves)
    ot_board, _ = othello_replay(moves)
    assert (othello_projection(tc_board) == ot_board).all()


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_known_pair():
    """
    The known transposition pair must produce exactly the expected diff.
    seq_a: f5 f6 d3 f4 g5  -> e4=Red, e5=Red
    seq_b: f5 f4 d3 f6 g5  -> e4=Blue, e5=Green
    """
    seq_a = ["f5", "f6", "d3", "f4", "g5"]
    seq_b = ["f5", "f4", "d3", "f6", "g5"]
    board_a, _ = replay(seq_a)
    board_b, _ = replay(seq_b)
    diffs = diff(board_a, board_b)
    squares = {d["square"] for d in diffs}
    assert squares == {"e4", "e5"}


def test_diff_same_board():
    seq = ["f5", "f6", "d3", "f4", "g5"]
    board, _ = replay(seq)
    assert diff(board, board) == []


def test_diff_distance():
    seq_a = ["f5", "f6", "d3", "f4", "g5"]
    seq_b = ["f5", "f4", "d3", "f6", "g5"]
    board_a, _ = replay(seq_a)
    board_b, _ = replay(seq_b)
    diffs = diff(board_a, board_b)
    assert all(d["distance"] >= 1 for d in diffs)


# ---------------------------------------------------------------------------
# trichrome_key
# ---------------------------------------------------------------------------

def test_trichrome_key_same_board():
    seq = ["f5", "f6", "d3", "f4", "g5"]
    board, _ = replay(seq)
    assert trichrome_key(board) == trichrome_key(board)


def test_trichrome_key_different_boards():
    seq_a = ["f5", "f6", "d3", "f4", "g5"]
    seq_b = ["f5", "f4", "d3", "f6", "g5"]
    board_a, _ = replay(seq_a)
    board_b, _ = replay(seq_b)
    assert trichrome_key(board_a) != trichrome_key(board_b)


def test_trichrome_key_is_hashable():
    board, _ = replay(["f5"])
    key = trichrome_key(board)
    d = {key: "test"}
    assert d[key] == "test"
