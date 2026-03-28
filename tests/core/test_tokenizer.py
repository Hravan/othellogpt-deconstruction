"""
tests/core/test_tokenizer.py
"""

import pytest
from othellogpt_deconstruction.core.tokenizer import (
    START_SQUARES, VALID_POSITIONS, VOCAB_SIZE, BLOCK_SIZE, PAD_ID,
    stoi, itos,
    alg_to_pos, pos_to_alg, seq_key, seq_from_key,
    encode, pad,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_start_squares():
    assert alg_to_pos("d4") == 27
    assert alg_to_pos("e4") == 28
    assert alg_to_pos("d5") == 35
    assert alg_to_pos("e5") == 36
    assert START_SQUARES == frozenset({alg_to_pos(s) for s in ("d4", "e4", "d5", "e5")})


def test_valid_positions_count():
    assert len(VALID_POSITIONS) == 60


def test_valid_positions_excludes_start_squares():
    assert not any(p in START_SQUARES for p in VALID_POSITIONS)


def test_vocab_size():
    assert VOCAB_SIZE == 61


# ---------------------------------------------------------------------------
# stoi / itos
# ---------------------------------------------------------------------------

def test_stoi_itos_roundtrip():
    for pos in VALID_POSITIONS:
        assert itos[stoi[pos]] == pos


def test_stoi_covers_all_valid_positions():
    assert set(VALID_POSITIONS).issubset(stoi.keys())


def test_stoi_excludes_start_squares():
    for sq in START_SQUARES:
        assert sq not in stoi


def test_token_range():
    tokens = [stoi[p] for p in VALID_POSITIONS]
    assert min(tokens) == 1
    assert max(tokens) == 60
    assert len(set(tokens)) == 60


def test_pad_token():
    assert PAD_ID == 0
    assert itos[PAD_ID] == -1


# ---------------------------------------------------------------------------
# alg_to_pos
# ---------------------------------------------------------------------------

def test_alg_to_pos_a1():
    assert alg_to_pos("a1") == 0


def test_alg_to_pos_h8():
    assert alg_to_pos("h8") == 63


def test_alg_to_pos_c4():
    assert alg_to_pos("c4") == 26


def test_alg_to_pos_d4():
    assert alg_to_pos("d4") == 27


def test_alg_to_pos_e4():
    assert alg_to_pos("e4") == 28


def test_alg_to_pos_d5():
    assert alg_to_pos("d5") == 35


def test_alg_to_pos_e5():
    assert alg_to_pos("e5") == 36


def test_alg_to_pos_case_insensitive():
    assert alg_to_pos("F5") == alg_to_pos("f5")


def test_alg_to_pos_all_squares():
    cols = "abcdefgh"
    for r in range(1, 9):
        for c in cols:
            pos = alg_to_pos(f"{c}{r}")
            assert 0 <= pos <= 63


# ---------------------------------------------------------------------------
# pos_to_alg
# ---------------------------------------------------------------------------

def test_pos_to_alg_0():
    assert pos_to_alg(0) == "a1"


def test_pos_to_alg_63():
    assert pos_to_alg(63) == "h8"


def test_pos_to_alg_26():
    assert pos_to_alg(26) == "c4"


def test_alg_pos_roundtrip():
    cols = "abcdefgh"
    for r in range(1, 9):
        for c in cols:
            move = f"{c}{r}"
            assert pos_to_alg(alg_to_pos(move)) == move


# ---------------------------------------------------------------------------
# seq_key / seq_from_key
# ---------------------------------------------------------------------------

def test_seq_key():
    assert seq_key(["f5", "d6", "c5"]) == "f5 d6 c5"


def test_seq_key_single():
    assert seq_key(["f5"]) == "f5"


def test_seq_key_empty():
    assert seq_key([]) == ""


def test_seq_from_key_roundtrip():
    seq = ["f5", "d6", "c5", "b4"]
    assert seq_from_key(seq_key(seq)) == seq


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def test_encode_single():
    token = encode(["c4"])[0]
    assert itos[token] == alg_to_pos("c4")


def test_encode_roundtrip():
    moves = ["f5", "d6", "c5"]
    tokens = encode(moves)
    recovered = [pos_to_alg(itos[t]) for t in tokens]
    assert recovered == moves


def test_encode_length():
    moves = ["f5", "d6", "c5", "b4", "e3"]
    assert len(encode(moves)) == len(moves)


# ---------------------------------------------------------------------------
# pad
# ---------------------------------------------------------------------------

def test_pad_length():
    tokens = encode(["f5", "d6"])
    padded = pad(tokens)
    assert len(padded) == BLOCK_SIZE


def test_pad_content():
    tokens = encode(["f5", "d6"])
    padded = pad(tokens)
    assert padded[:2] == tokens
    assert all(t == PAD_ID for t in padded[2:])


def test_pad_exact_length():
    tokens = [1] * BLOCK_SIZE
    padded = pad(tokens)
    assert len(padded) == BLOCK_SIZE
    assert padded == tokens


def test_pad_too_long():
    tokens = [1] * (BLOCK_SIZE + 1)
    with pytest.raises(ValueError):
        pad(tokens)
