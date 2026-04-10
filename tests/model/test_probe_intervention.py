"""
tests/model/test_probe_intervention.py

Tests for model/probe_intervention.py.
No model checkpoints or GPU required.
"""

import numpy as np
import pytest
import torch

from othellogpt_deconstruction.core.board import (
    BLACK, WHITE, EMPTY, legal_moves, replay as board_replay, start_board,
)
from othellogpt_deconstruction.model.probe_intervention import (
    board_to_probe_encoding, replay_nonvalidating,
)


# ---------------------------------------------------------------------------
# board_to_probe_encoding
# ---------------------------------------------------------------------------

def test_board_to_probe_encoding_shape():
    board = np.zeros(64, dtype=np.int8)
    result = board_to_probe_encoding(board)
    assert result.shape == (64,)


def test_board_to_probe_encoding_dtype():
    board = np.zeros(64, dtype=np.int8)
    result = board_to_probe_encoding(board)
    assert result.dtype == torch.long


def test_board_to_probe_encoding_empty_cell():
    # Our EMPTY=0 → Li's empty=1
    board = np.zeros(64, dtype=np.int8)
    result = board_to_probe_encoding(board)
    assert (result == 1).all()


def test_board_to_probe_encoding_black_cell():
    # Our BLACK=1 → Li's black=2
    board = np.zeros(64, dtype=np.int8)
    board[0] = BLACK  # 1
    result = board_to_probe_encoding(board)
    assert int(result[0]) == 2


def test_board_to_probe_encoding_white_cell():
    # Our WHITE=2 → Li's white=0
    board = np.zeros(64, dtype=np.int8)
    board[0] = WHITE  # 2
    result = board_to_probe_encoding(board)
    assert int(result[0]) == 0


def test_board_to_probe_encoding_formula():
    # Verify (board + 1) % 3 for all three values
    board = np.array([EMPTY, BLACK, WHITE] * 21 + [EMPTY], dtype=np.int8)
    result = board_to_probe_encoding(board)
    expected = torch.tensor([(v + 1) % 3 for v in board], dtype=torch.long)
    assert torch.equal(result, expected)


def test_board_to_probe_encoding_values_in_range():
    board = np.array([EMPTY, BLACK, WHITE] * 21 + [EMPTY], dtype=np.int8)
    result = board_to_probe_encoding(board)
    assert result.min() >= 0
    assert result.max() <= 2


def test_board_to_probe_encoding_starting_board():
    # Starting board has 4 pieces; rest are empty
    board = start_board()
    result = board_to_probe_encoding(board)
    # Corners (d4, d5, e4, e5) have pieces; everything else is empty
    empty_cells = result[result == 1]
    assert len(empty_cells) == 60


# ---------------------------------------------------------------------------
# replay_nonvalidating
# ---------------------------------------------------------------------------

def test_replay_nonvalidating_returns_board_and_player():
    positions = [alg_to_pos("f5"), alg_to_pos("d6")]
    board, player = replay_nonvalidating(positions)
    assert isinstance(board, np.ndarray)
    assert board.shape == (64,)
    assert player in (BLACK, WHITE)


def test_replay_nonvalidating_empty_sequence():
    board, player = replay_nonvalidating([])
    starting = start_board()
    assert np.array_equal(board, starting)
    assert player == BLACK


def test_replay_nonvalidating_agrees_with_legal_replay_for_legal_moves():
    # For a legal opening sequence the two replays should agree
    opening = ["f5", "d6", "c5", "f4", "e3"]
    positions = [alg_to_pos(move) for move in opening]
    board_nonval, player_nonval = replay_nonvalidating(positions)
    board_val,    player_val    = board_replay(opening)
    assert np.array_equal(board_nonval, board_val)
    assert player_nonval == player_val


def test_replay_nonvalidating_places_piece():
    # After one move the played position should be occupied
    pos = alg_to_pos("f5")
    board, _ = replay_nonvalidating([pos])
    assert board[pos] != EMPTY


def test_replay_nonvalidating_illegal_move_does_not_raise():
    # Positions 27, 28, 35, 36 are the starting squares — playing on an
    # occupied cell is illegal in Othello, but replay_nonvalidating should
    # not raise.
    occupied = 27
    replay_nonvalidating([occupied])  # should not raise


def test_replay_nonvalidating_player_alternates():
    # After an even number of legal moves Black should be next
    opening = ["f5", "d6"]
    positions = [alg_to_pos(move) for move in opening]
    _, player = replay_nonvalidating(positions)
    assert player == BLACK


# ---------------------------------------------------------------------------
# Helpers (local to test module)
# ---------------------------------------------------------------------------

def alg_to_pos(move: str) -> int:
    col = ord(move[0].lower()) - ord("a")
    row = int(move[1]) - 1
    return row * 8 + col
