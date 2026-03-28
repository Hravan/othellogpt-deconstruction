"""
src/othellogpt_deconstruction/core/tokenizer.py

Vocabulary and coordinate utilities for OthelloGPT.

Vocabulary
----------
The model uses 61 tokens:
  - Token 0        : PAD (padding)
  - Tokens 1..60   : the 60 playable board positions (all squares except the
                     4 center starting squares: d4=27, e4=28, d5=35, e5=36)

Board positions are integers 0..63 in row-major order:
  pos = (row - 1) * 8 + col_index
  where row in 1..8, col_index in 0..7  (a=0, b=1, ..., h=7)

Examples
--------
  alg_to_pos("c4") -> 26    (row 4, col c=2 -> (4-1)*8 + 2 = 26)
  pos_to_alg(26)   -> "c4"
  stoi[26]         -> 19    (token id for c4)
  itos[19]         -> 26    (board position for token 19)
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_SQUARES: frozenset[int] = frozenset({27, 28, 35, 36})
VALID_POSITIONS: list[int] = [p for p in range(64) if p not in START_SQUARES]

VOCAB_SIZE: int = 61   # 60 playable squares + 1 PAD token
BLOCK_SIZE: int = 59
PAD_ID:     int = 0

# ---------------------------------------------------------------------------
# Token <-> board position mappings
# ---------------------------------------------------------------------------

stoi: dict[int, int] = {pos: i + 1 for i, pos in enumerate(VALID_POSITIONS)}
itos: dict[int, int] = {i + 1: pos  for i, pos in enumerate(VALID_POSITIONS)}

itos[PAD_ID] = -1       # sentinel: PAD has no board position

# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def alg_to_pos(move: str) -> int:
    """
    Convert algebraic notation to board position.

    Examples
    --------
    >>> alg_to_pos("a1")
    0
    >>> alg_to_pos("c4")
    26
    >>> alg_to_pos("h8")
    63
    """
    col = ord(move[0].lower()) - ord("a")   # a=0 .. h=7
    row = int(move[1]) - 1                  # 1-indexed -> 0-indexed
    return row * 8 + col


def pos_to_alg(pos: int) -> str:
    """
    Convert board position to algebraic notation.

    Examples
    --------
    >>> pos_to_alg(0)
    'a1'
    >>> pos_to_alg(26)
    'c4'
    >>> pos_to_alg(63)
    'h8'
    """
    row, col = divmod(pos, 8)
    return f"{chr(ord('a') + col)}{row + 1}"


def seq_key(sequence: list[str]) -> str:
    """
    Convert a move sequence to a hashable string key.

    Examples
    --------
    >>> seq_key(["f5", "d6", "c5"])
    'f5 d6 c5'
    """
    return " ".join(sequence)


def seq_from_key(key: str) -> list[str]:
    """
    Inverse of seq_key.

    Examples
    --------
    >>> seq_from_key("f5 d6 c5")
    ['f5', 'd6', 'c5']
    """
    return key.split()


# ---------------------------------------------------------------------------
# Token sequence utilities
# ---------------------------------------------------------------------------

def encode(sequence: list[str]) -> list[int]:
    """
    Convert a list of algebraic moves to a list of token ids.

    Examples
    --------
    >>> encode(["c4"])
    [19]
    """
    return [stoi[alg_to_pos(m)] for m in sequence]


def pad(tokens: list[int], block_size: int = BLOCK_SIZE) -> list[int]:
    """
    Pad a token sequence to block_size with PAD_ID.
    Raises ValueError if the sequence is longer than block_size.
    """
    if len(tokens) > block_size:
        raise ValueError(
            f"Sequence length {len(tokens)} exceeds block_size {block_size}"
        )
    return tokens + [PAD_ID] * (block_size - len(tokens))
