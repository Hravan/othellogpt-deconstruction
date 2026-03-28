"""
src/othellogpt_deconstruction/core/trichrome.py

Trichrome Othello board simulator.

Trichrome state
---------------
Each cell carries (owner, color):
    owner : 0=empty, 1=Black, 2=White
    color : 0=Red, 1=Green, 2=Blue  (flip_count mod 3)

A newly placed disc is always Red (color=0).
A captured disc advances one color step: R->G->B->R.
Legal moves are determined from owner alone — identical to standard Othello.

Board representation
--------------------
A Trichrome board is a numpy array of shape (64, 2) where:
    board[pos, 0] = owner
    board[pos, 1] = color

The Othello projection is board[:, 0].
"""

import numpy as np

from othellogpt_deconstruction.core.tokenizer import alg_to_pos, pos_to_alg
from othellogpt_deconstruction.core.board import (
    EMPTY, BLACK, WHITE,
    flipped_by, legal_moves, apply_move,
)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

RED:   int = 0
GREEN: int = 1
BLUE:  int = 2

COLOR_NAMES: dict[int, str] = {RED: "Red", GREEN: "Green", BLUE: "Blue"}


def color_distance(c1: int, c2: int) -> int:
    """
    Minimum circular distance between two colors on the R->G->B->R cycle.

    Examples
    --------
    >>> color_distance(RED, GREEN)
    1
    >>> color_distance(RED, BLUE)
    1
    >>> color_distance(GREEN, BLUE)
    1
    >>> color_distance(RED, RED)
    0
    """
    d = abs(c1 - c2)
    return min(d, 3 - d)


# ---------------------------------------------------------------------------
# Board construction
# ---------------------------------------------------------------------------

def start_board() -> np.ndarray:
    """
    Return a fresh Trichrome board in the standard starting position.
    All four center discs start Red.
    """
    board = np.zeros((64, 2), dtype=np.int8)
    board[alg_to_pos("d4")] = [WHITE, RED]
    board[alg_to_pos("e4")] = [BLACK, RED]
    board[alg_to_pos("d5")] = [BLACK, RED]
    board[alg_to_pos("e5")] = [WHITE, RED]
    return board


def othello_projection(board: np.ndarray) -> np.ndarray:
    """Return the Othello board (owner only) from a Trichrome board."""
    return board[:, 0].copy()


# ---------------------------------------------------------------------------
# Move application
# ---------------------------------------------------------------------------

def apply_move(tc_board: np.ndarray, pos: int, player: int) -> np.ndarray:
    """
    Apply a move to a Trichrome board and return a new board.
    Does not modify the input board.
    Raises ValueError if the move is illegal.
    """
    othello = othello_projection(tc_board)
    flips = flipped_by(othello, pos, player)
    if not flips:
        raise ValueError(
            f"Illegal move at pos {pos} for player {player}"
        )
    new_board = tc_board.copy()
    new_board[pos] = [player, RED]   # new disc always Red
    for fp in flips:
        old_color = new_board[fp, 1]
        new_board[fp] = [player, (old_color + 1) % 3]
    return new_board


# ---------------------------------------------------------------------------
# Game replay
# ---------------------------------------------------------------------------

def replay(moves: list[str]) -> tuple[np.ndarray, int]:
    """
    Replay a sequence of algebraic moves on a Trichrome board.
    Returns (final_board, next_player).
    Raises ValueError on any illegal move.
    """
    board = start_board()
    player = BLACK

    for move in moves:
        pos = alg_to_pos(move)
        board = apply_move(board, pos, player)
        player = 3 - player
        othello = othello_projection(board)
        if not legal_moves(othello, player):
            player = 3 - player

    return board, player


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------

def diff(board_a: np.ndarray, board_b: np.ndarray) -> list[dict]:
    """
    Return a list of cells where two Trichrome boards differ in color
    but agree on owner (i.e. same Othello state, different Trichrome state).

    Each entry is a dict with keys:
        square, pos, owner, color_a, color_b, distance
    """
    diffs = []
    for pos in range(64):
        owner_a, color_a = board_a[pos]
        owner_b, color_b = board_b[pos]
        if owner_a == EMPTY and owner_b == EMPTY:
            continue
        if owner_a != owner_b:
            continue   # Othello states differ — not a valid Trichrome-only diff
        if color_a != color_b:
            diffs.append({
                "square":   pos_to_alg(pos),
                "pos":      pos,
                "owner":    int(owner_a),
                "color_a":  int(color_a),
                "color_b":  int(color_b),
                "distance": color_distance(int(color_a), int(color_b)),
            })
    return diffs


def trichrome_key(board: np.ndarray) -> tuple:
    """Return a hashable key representing the full Trichrome state."""
    return tuple(map(tuple, board))
