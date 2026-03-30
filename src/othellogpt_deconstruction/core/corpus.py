"""
src/othellogpt_deconstruction/core/corpus.py

Corpus loading for championship (text) and synthetic (pickle) datasets.

Both loaders return the same format:
    list[list[str]]  — a list of games, each game a list of algebraic moves

Championship format
-------------------
Text files where each line is a game as space-separated algebraic moves.

Synthetic format
----------------
Pickle files containing list[list[int]] where each integer is a token id
in the OthelloGPT vocabulary (stoi/itos from tokenizer.py).
"""

import pickle
import random
from pathlib import Path

from othellogpt_deconstruction.core.tokenizer import START_SQUARES
from othellogpt_deconstruction.core.tokenizer import itos, pos_to_alg


# ---------------------------------------------------------------------------
# Championship loader
# ---------------------------------------------------------------------------

def _parse_pgn(path: Path) -> list[list[str]]:
    """
    Parse an Othello PGN file into a list of games.
    Each game is a list of algebraic moves in order.

    PGN format:
        [Header "value"]   <- skip
        1. d3 c5           <- move number, black move, white move
        3. f6 f5
        ...
    """
    games: list[list[str]] = []
    current: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                # Header line — start new game if we have moves accumulated
                if current:
                    games.append(current)
                    current = []
                continue
            # Move line: "1. d3 c5" or "1. d3" (last move may be single)
            parts = line.split()
            # parts[0] is move number like "1.", skip it
            moves = [p.lower() for p in parts[1:] if not p.endswith(".")]
            current.extend(moves)

    if current:
        games.append(current)

    return games


def load_championship(path: str | Path) -> list[list[str]]:
    """
    Load championship games from a PGN or text file or directory.
    """
    path = Path(path)
    files = sorted(f for f in path.glob("*") if f.suffix in (".txt", ".pgn")) \
        if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .txt or .pgn files found in {path}")

    games = []
    for fpath in files:
        if fpath.suffix == ".pgn":
            games.extend(_parse_pgn(fpath))
        else:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        moves = [m.lower() for m in line.split()]
                        if moves:
                            games.append(moves)
    return games


def load_synthetic(path: str | Path) -> list[list[str]]:
    """
    Load synthetic games from a pickle file or directory of pickle files.
    Each pickle contains list[list[int]] of board positions (0-63).
    Center squares (START_SQUARES) are excluded from moves.
    """
    path = Path(path)
    files = sorted(path.glob("*.pickle")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .pickle files found in {path}")

    games = []
    for fpath in files:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        for positions in data:
            # positions are raw board positions 0-63
            # filter out center squares (should never appear but be safe)
            moves = [pos_to_alg(p) for p in positions if p not in START_SQUARES]
            if moves:
                games.append(moves)
    return games


# ---------------------------------------------------------------------------
# Auto-detecting loader
# ---------------------------------------------------------------------------

def load_corpus(path: str | Path) -> list[list[str]]:
    """
    Load a corpus from a file or directory, auto-detecting format.

    Detects format by file extension:
        .txt    -> championship format
        .pickle -> synthetic format

    For directories, uses the first file found to detect format.
    """
    path = Path(path)
    if path.is_dir():
        files = list(path.iterdir())
        extensions = {f.suffix for f in files if f.is_file()}
        if ".pickle" in extensions:
            return load_synthetic(path)
        elif extensions & {".txt", ".pgn"}:
            return load_championship(path)
        else:
            raise ValueError(f"Cannot detect corpus format in {path} (extensions: {extensions})")
    else:
        if path.suffix == ".pickle":
            return load_synthetic(path)
        elif path.suffix in (".txt", ".pgn"):
            return load_championship(path)
        else:
            raise ValueError(f"Cannot detect corpus format for file {path}")


# ---------------------------------------------------------------------------
# Multi-corpus loader
# ---------------------------------------------------------------------------

def load_corpora(paths: list[str | Path]) -> list[list[str]]:
    """
    Load and concatenate multiple corpora.
    Each path can be a file or directory in any supported format.
    """
    games = []
    for path in paths:
        games.extend(load_corpus(path))
    return games


# ---------------------------------------------------------------------------
# File-level utilities (for memory-efficient processing)
# ---------------------------------------------------------------------------

def list_corpus_files(paths: list[str | Path]) -> list[Path]:
    """
    Return a stable ordered list of all individual files across the given paths.
    """
    files = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.pickle")))
            files.extend(sorted(f for f in path.glob("*") if f.suffix in (".txt", ".pgn")))
        else:
            files.append(path)
    return files


def load_file(path: str | Path) -> list[list[str]]:
    """Load games from a single file, auto-detecting format."""
    path = Path(path)
    if path.suffix == ".pickle":
        with open(path, "rb") as f:
            data = pickle.load(f)
        games = []
        for positions in data:
            moves = [pos_to_alg(p) for p in positions if p not in START_SQUARES]
            if moves:
                games.append(moves)
        return games
    elif path.suffix == ".pgn":
        return _parse_pgn(path)
    elif path.suffix == ".txt":
        games = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    moves = [m.lower() for m in line.split()]
                    if moves:
                        games.append(moves)
        return games
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def iter_corpus_files(paths: list[str | Path]):
    """
    Yield one file's worth of games at a time across all given paths.

    Each yield is a list[list[str]] for a single file, so callers can
    process and discard it before loading the next file.
    """
    for path in list_corpus_files(paths):
        yield load_file(path)


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split(
    games: list[list[str]],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Split games into train and test sets.
    Shuffles before splitting for reproducibility.
    """
    rng = random.Random(seed)
    shuffled = games.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample(
    games: list[list[str]],
    n: int,
    seed: int = 42,
) -> list[list[str]]:
    """
    Return a random sample of n games without replacement.
    If n >= len(games), returns all games shuffled.
    """
    rng = random.Random(seed)
    if n >= len(games):
        result = games.copy()
        rng.shuffle(result)
        return result
    return rng.sample(games, n)
