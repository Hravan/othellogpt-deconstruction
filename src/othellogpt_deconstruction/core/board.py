"""
src/othellogpt_deconstruction/core/board.py

Othello board simulator and legal move utilities.

Board representation
--------------------
A board is a flat numpy array of 64 integers in row-major order:
    0 = empty
    1 = Black
    2 = White

Position indexing is row-major:
    pos = (row - 1) * 8 + col_index
    where row in 1..8, col_index in 0..7  (a=0 .. h=7)

This matches alg_to_pos / pos_to_alg in tokenizer.py.
"""

import numpy as np

from othellogpt_deconstruction.core.tokenizer import alg_to_pos

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY: int = 0
BLACK: int = 1
WHITE: int = 2

DIRS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           (0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

# Starting position: d4=White, e4=Black, d5=Black, e5=White
_START: dict[str, int] = {"d4": WHITE, "e4": BLACK, "d5": BLACK, "e5": WHITE}
START_BOARD: np.ndarray = np.zeros(64, dtype=np.int8)
for _sq, _owner in _START.items():
    START_BOARD[alg_to_pos(_sq)] = _owner
START_BOARD.flags.writeable = False   # immutable sentinel


# ---------------------------------------------------------------------------
# Board construction
# ---------------------------------------------------------------------------

def start_board() -> np.ndarray:
    """Return a fresh mutable copy of the starting board."""
    return START_BOARD.copy()


# ---------------------------------------------------------------------------
# Legal move computation
# ---------------------------------------------------------------------------

def flipped_by(board: np.ndarray, pos: int, player: int) -> list[int]:
    """
    Return the list of positions that would be flipped if `player` plays at `pos`.
    Returns an empty list if the move is illegal.
    """
    if board[pos] != EMPTY:
        return []

    opponent = 3 - player
    row, col = divmod(pos, 8)
    result = []

    for dr, dc in DIRS:
        r, c = row + dr, col + dc
        line: list[int] = []
        while 0 <= r < 8 and 0 <= c < 8:
            p = r * 8 + c
            if board[p] == opponent:
                line.append(p)
                r += dr; c += dc
            elif board[p] == player and line:
                result.extend(line)
                break
            else:
                break

    return result


def is_legal(board: np.ndarray, pos: int, player: int) -> bool:
    """Return True if playing at pos is legal for player."""
    return len(flipped_by(board, pos, player)) > 0


def legal_moves(board: np.ndarray, player: int) -> list[int]:
    """Return all legal move positions for player."""
    return [pos for pos in range(64) if is_legal(board, pos, player)]


# ---------------------------------------------------------------------------
# Move application
# ---------------------------------------------------------------------------

def apply_move(board: np.ndarray, pos: int, player: int) -> np.ndarray:
    """
    Apply a move and return a new board. Does not modify the input board.
    Raises ValueError if the move is illegal.
    """
    flips = flipped_by(board, pos, player)
    if not flips:
        raise ValueError(
            f"Illegal move at pos {pos} for player {player}"
        )
    new_board = board.copy()
    new_board[pos] = player
    for fp in flips:
        new_board[fp] = player
    return new_board


# ---------------------------------------------------------------------------
# Game replay
# ---------------------------------------------------------------------------

def replay(moves: list[str]) -> tuple[np.ndarray, int]:
    """
    Replay a sequence of algebraic moves from the starting position.
    Returns (final_board, next_player).
    Raises ValueError on any illegal move.
    """
    board = start_board()
    player = BLACK

    for move in moves:
        pos = alg_to_pos(move)
        board = apply_move(board, pos, player)
        player = 3 - player
        if not legal_moves(board, player):
            player = 3 - player   # pass if no legal moves

    return board, player


def legal_moves_after(moves: list[str]) -> list[int]:
    """
    Return the legal move positions after replaying a sequence.
    Convenience wrapper around replay + legal_moves.
    """
    board, player = replay(moves)
    return legal_moves(board, player)
