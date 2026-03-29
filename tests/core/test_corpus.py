"""
tests/core/test_corpus.py

Tests for corpus loading. Uses in-memory fixtures rather than real files
so no corpus data is required to run the test suite.
"""

import pickle
import random
import tempfile
from pathlib import Path

import pytest

from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos
from othellogpt_deconstruction.core.corpus import (
    load_championship, load_synthetic, load_corpus, load_corpora,
    split, sample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_GAMES_ALG = [
    ["f5", "d6", "c5", "f4"],
    ["f5", "f6", "d3", "f4", "g5"],
    ["c4", "c5", "e6", "c3", "b4"],
]

SAMPLE_GAMES_TOKENS = [
    [stoi[alg_to_pos(m)] for m in game]
    for game in SAMPLE_GAMES_ALG
]


def write_championship_file(path: Path, games: list[list[str]]) -> None:
    with open(path, "w") as f:
        for game in games:
            f.write(" ".join(game) + "\n")


def write_synthetic_file(path: Path, games: list[list[int]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(games, f)


# ---------------------------------------------------------------------------
# load_championship
# ---------------------------------------------------------------------------

def test_load_championship_file():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "games.txt"
        write_championship_file(fpath, SAMPLE_GAMES_ALG)
        games = load_championship(fpath)
        assert len(games) == 3
        assert games[0] == ["f5", "d6", "c5", "f4"]


def test_load_championship_directory():
    with tempfile.TemporaryDirectory() as tmp:
        write_championship_file(Path(tmp) / "a.txt", SAMPLE_GAMES_ALG[:2])
        write_championship_file(Path(tmp) / "b.txt", SAMPLE_GAMES_ALG[2:])
        games = load_championship(tmp)
        assert len(games) == 3


def test_load_championship_lowercase():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "games.txt"
        write_championship_file(fpath, [["F5", "D6"]])
        games = load_championship(fpath)
        assert games[0] == ["f5", "d6"]


def test_load_championship_empty_lines():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "games.txt"
        with open(fpath, "w") as f:
            f.write("f5 d6\n\n\nc4 c5\n")
        games = load_championship(fpath)
        assert len(games) == 2


def write_pgn_file(path: Path, games: list[list[str]]) -> None:
    with open(path, "w") as f:
        for i, game in enumerate(games):
            f.write(f'[Event "Test"]\n')
            f.write(f'[Round "{i+1}"]\n')
            for j in range(0, len(game), 2):
                move_num = j // 2 + 1
                black = game[j]
                white = game[j+1] if j+1 < len(game) else ""
                f.write(f"{move_num}. {black} {white}\n".strip() + "\n")
            f.write("\n")


def test_load_championship_pgn():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "games.pgn"
        write_pgn_file(fpath, SAMPLE_GAMES_ALG)
        games = load_championship(fpath)
        assert len(games) == 3
        assert games[0] == SAMPLE_GAMES_ALG[0]
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_championship(tmp)


# ---------------------------------------------------------------------------
# load_synthetic
# ---------------------------------------------------------------------------

def test_load_synthetic_file():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "games.pickle"
        write_synthetic_file(fpath, SAMPLE_GAMES_TOKENS)
        games = load_synthetic(fpath)
        assert len(games) == 3
        assert games[0] == SAMPLE_GAMES_ALG[0]


def test_load_synthetic_directory():
    with tempfile.TemporaryDirectory() as tmp:
        write_synthetic_file(Path(tmp) / "a.pickle", SAMPLE_GAMES_TOKENS[:2])
        write_synthetic_file(Path(tmp) / "b.pickle", SAMPLE_GAMES_TOKENS[2:])
        games = load_synthetic(tmp)
        assert len(games) == 3


def test_load_synthetic_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "games.pickle"
        write_synthetic_file(fpath, SAMPLE_GAMES_TOKENS)
        games = load_synthetic(fpath)
        assert games == SAMPLE_GAMES_ALG


def test_load_synthetic_no_files_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_synthetic(tmp)


# ---------------------------------------------------------------------------
# load_corpus (auto-detection)
# ---------------------------------------------------------------------------

def test_load_corpus_detects_txt():
    with tempfile.TemporaryDirectory() as tmp:
        write_championship_file(Path(tmp) / "games.txt", SAMPLE_GAMES_ALG)
        games = load_corpus(tmp)
        assert len(games) == 3


def test_load_corpus_detects_pickle():
    with tempfile.TemporaryDirectory() as tmp:
        write_synthetic_file(Path(tmp) / "games.pickle", SAMPLE_GAMES_TOKENS)
        games = load_corpus(tmp)
        assert len(games) == 3


def test_load_corpus_unknown_format_raises():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "games.csv").write_text("f5,d6\n")
        with pytest.raises(ValueError):
            load_corpus(tmp)


# ---------------------------------------------------------------------------
# load_corpora
# ---------------------------------------------------------------------------

def test_load_corpora_combines():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "champ.txt"
        p2 = Path(tmp) / "synth.pickle"
        write_championship_file(p1, SAMPLE_GAMES_ALG[:2])
        write_synthetic_file(p2, SAMPLE_GAMES_TOKENS[2:])
        games = load_corpora([p1, p2])
        assert len(games) == 3


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def test_split_sizes():
    games = SAMPLE_GAMES_ALG * 10  # 30 games
    train, test = split(games, test_fraction=0.2)
    assert len(train) + len(test) == 30
    assert len(test) == 6


def test_split_no_overlap():
    games = [["f5", str(i)] for i in range(100)]
    train, test = split(games)
    train_set = {tuple(g) for g in train}
    test_set  = {tuple(g) for g in test}
    assert train_set.isdisjoint(test_set)


def test_split_reproducible():
    games = SAMPLE_GAMES_ALG * 20
    train1, test1 = split(games, seed=42)
    train2, test2 = split(games, seed=42)
    assert train1 == train2
    assert test1  == test2


def test_split_different_seeds():
    games = SAMPLE_GAMES_ALG * 20
    _, test1 = split(games, seed=42)
    _, test2 = split(games, seed=99)
    assert test1 != test2


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def test_sample_size():
    games = SAMPLE_GAMES_ALG * 10
    s = sample(games, 5)
    assert len(s) == 5


def test_sample_reproducible():
    games = SAMPLE_GAMES_ALG * 10
    s1 = sample(games, 5, seed=42)
    s2 = sample(games, 5, seed=42)
    assert s1 == s2


def test_sample_larger_than_corpus():
    games = SAMPLE_GAMES_ALG
    s = sample(games, 100)
    assert len(s) == len(games)


def test_sample_no_duplicate_indices():
    games = [["f5", str(i)] for i in range(20)]  # 20 unique games
    s = sample(games, 5)
    assert len(s) == len(set(tuple(g) for g in s))
