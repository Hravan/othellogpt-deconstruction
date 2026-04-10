"""
tests/model/test_representation_alignment.py

Tests for model/representation_alignment.py.
Only tests build_token_batch (load_othello_gpt requires an external checkpoint).
"""

import pytest
import torch

from othellogpt_deconstruction.core.tokenizer import BLOCK_SIZE, PAD_ID, stoi
from othellogpt_deconstruction.model.representation_alignment import build_token_batch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Games are sequences of board positions (0-63).
# Positions 27, 28, 35, 36 are the center starting squares and are excluded
# from stoi. Use only playable positions here.
GAME_A = [20, 19, 26, 37, 44]   # f5, f4 area (valid positions)
GAME_B = [20, 19, 26]
SINGLE_MOVE_GAME = [20]


# ---------------------------------------------------------------------------
# build_token_batch
# ---------------------------------------------------------------------------

def test_build_token_batch_token_shape():
    token_ids, _ = build_token_batch([GAME_A], [2], device=torch.device("cpu"))
    assert token_ids.shape == (1, BLOCK_SIZE)


def test_build_token_batch_batch_shape():
    games = [GAME_A, GAME_B, SINGLE_MOVE_GAME]
    steps = [2, 1, 0]
    token_ids, seq_lens = build_token_batch(games, steps, device=torch.device("cpu"))
    assert token_ids.shape == (3, BLOCK_SIZE)
    assert len(seq_lens) == 3


def test_build_token_batch_dtype():
    token_ids, _ = build_token_batch([GAME_A], [2], device=torch.device("cpu"))
    assert token_ids.dtype == torch.long


def test_build_token_batch_seq_lens():
    # Step 2 means positions 0..2 → seq_len = 3
    games = [GAME_A, GAME_B, SINGLE_MOVE_GAME]
    steps = [2, 1, 0]
    _, seq_lens = build_token_batch(games, steps, device=torch.device("cpu"))
    assert seq_lens == [3, 2, 1]


def test_build_token_batch_padding():
    # Positions after seq_len should be PAD
    token_ids, seq_lens = build_token_batch([GAME_A], [2], device=torch.device("cpu"))
    seq_len = seq_lens[0]
    padded_region = token_ids[0, seq_len:]
    assert (padded_region == PAD_ID).all()


def test_build_token_batch_no_pad_in_prefix():
    token_ids, seq_lens = build_token_batch([GAME_A], [2], device=torch.device("cpu"))
    seq_len = seq_lens[0]
    prefix = token_ids[0, :seq_len]
    assert (prefix != PAD_ID).all()


def test_build_token_batch_tokens_match_stoi():
    # Verify first token of GAME_A at step 0
    token_ids, _ = build_token_batch([GAME_A], [0], device=torch.device("cpu"))
    expected_token = stoi[GAME_A[0]]
    assert int(token_ids[0, 0]) == expected_token


def test_build_token_batch_single_step():
    # Step 0 → only the first position is included, rest is padding
    token_ids, seq_lens = build_token_batch([GAME_A], [0], device=torch.device("cpu"))
    assert seq_lens == [1]
    assert (token_ids[0, 1:] == PAD_ID).all()


def test_build_token_batch_step_capped_at_block_size():
    # A step beyond BLOCK_SIZE-1 should be capped
    long_game = list(range(60))  # only playable positions
    # Use positions from stoi to ensure they are valid
    valid_positions = sorted(stoi.keys())
    long_game = valid_positions[:BLOCK_SIZE + 5]
    step = BLOCK_SIZE + 2  # beyond BLOCK_SIZE
    token_ids, seq_lens = build_token_batch([long_game], [step], device=torch.device("cpu"))
    assert seq_lens[0] <= BLOCK_SIZE
    assert token_ids.shape == (1, BLOCK_SIZE)


def test_build_token_batch_multiple_games_independent():
    games = [GAME_A, GAME_B]
    steps = [len(GAME_A) - 1, len(GAME_B) - 1]
    token_ids_together, _ = build_token_batch(games, steps, device=torch.device("cpu"))
    token_ids_a, _ = build_token_batch([GAME_A], [len(GAME_A) - 1], device=torch.device("cpu"))
    token_ids_b, _ = build_token_batch([GAME_B], [len(GAME_B) - 1], device=torch.device("cpu"))
    assert torch.equal(token_ids_together[0], token_ids_a[0])
    assert torch.equal(token_ids_together[1], token_ids_b[0])


def test_build_token_batch_empty_batch():
    token_ids, seq_lens = build_token_batch([], [], device=torch.device("cpu"))
    assert token_ids.shape == (0, BLOCK_SIZE)
    assert seq_lens == []
