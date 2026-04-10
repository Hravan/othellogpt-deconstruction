"""
tests/model/test_utils.py

Tests for model/utils.py.
Uses mode="random" so no checkpoint files or GPU are required.
"""

import pytest
import torch
import numpy as np

from othellogpt_deconstruction.core.board import legal_moves, apply_move, BLACK, WHITE
from othellogpt_deconstruction.core.board import replay as board_replay
from othellogpt_deconstruction.core.tokenizer import (
    BLOCK_SIZE, PAD_ID, VOCAB_SIZE, itos, stoi, alg_to_pos, pos_to_alg,
)
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.model.utils import (
    encode_sequence, forward_pass, rollout, top1_position, topn_errors, topn_positions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_model():
    return load_model(mode="random", device="cpu")


# A short but legal Othello opening (first 5 moves of a common line)
OPENING = ["f5", "d6", "c5", "f4", "e3"]

# Board state reached after OPENING
@pytest.fixture(scope="module")
def opening_board():
    board, next_player = board_replay(OPENING)
    return board, next_player


# ---------------------------------------------------------------------------
# encode_sequence
# ---------------------------------------------------------------------------

def test_encode_sequence_shape():
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    assert x.shape == (1, BLOCK_SIZE)


def test_encode_sequence_dtype():
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    assert x.dtype == torch.long


def test_encode_sequence_first_tokens():
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    expected_first = stoi[alg_to_pos(OPENING[0])]
    assert int(x[0, 0]) == expected_first


def test_encode_sequence_padding():
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    # All positions after len(OPENING) should be PAD
    assert (x[0, len(OPENING):] == PAD_ID).all()


def test_encode_sequence_no_padding_in_prefix():
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    assert (x[0, :len(OPENING)] != PAD_ID).all()


def test_encode_sequence_empty():
    x = encode_sequence([], device=torch.device("cpu"))
    assert x.shape == (1, BLOCK_SIZE)
    assert (x == PAD_ID).all()


def test_encode_sequence_full_game():
    # 59-move game should fill the whole block without error
    game = OPENING * (BLOCK_SIZE // len(OPENING) + 1)
    game = game[:BLOCK_SIZE]
    # Just verify it doesn't crash and has right shape
    # (the moves won't be legal but encode_sequence doesn't validate)
    x = encode_sequence(game, device=torch.device("cpu"))
    assert x.shape == (1, BLOCK_SIZE)


# ---------------------------------------------------------------------------
# forward_pass
# ---------------------------------------------------------------------------

def test_forward_pass_shape(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    assert probs.shape == (VOCAB_SIZE,)


def test_forward_pass_sums_to_one(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_forward_pass_nonnegative(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    assert (probs >= 0).all()


def test_forward_pass_pad_is_zero(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    assert float(probs[PAD_ID]) == pytest.approx(0.0, abs=1e-10)


def test_forward_pass_single_move(random_model):
    x = encode_sequence(["f5"], device=torch.device("cpu"))
    probs = forward_pass(random_model, x, 1)
    assert probs.shape == (VOCAB_SIZE,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_forward_pass_different_lengths_differ(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs_short = forward_pass(random_model, x, 1)
    probs_long  = forward_pass(random_model, x, len(OPENING))
    assert not torch.allclose(probs_short, probs_long)


# ---------------------------------------------------------------------------
# top1_position
# ---------------------------------------------------------------------------

def test_top1_position_returns_int(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    result = top1_position(probs)
    assert isinstance(result, int)


def test_top1_position_valid_board_position(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    pos = top1_position(probs)
    # Should be a valid board position (0-63) or -1 for PAD
    assert pos == -1 or (0 <= pos < 64)


def test_top1_position_not_pad(random_model):
    # For a random model, the argmax is almost certainly not PAD since
    # forward_pass already zeroes out PAD before softmax.
    # Test on multiple sequences to be robust.
    for seq in [["f5"], ["f5", "d6"], OPENING]:
        x = encode_sequence(seq, device=torch.device("cpu"))
        probs = forward_pass(random_model, x, len(seq))
        assert top1_position(probs) != PAD_ID


def test_top1_position_pad_probs_returns_minus_one():
    # If the argmax token is PAD (token 0), return -1
    probs = torch.zeros(VOCAB_SIZE)
    probs[PAD_ID] = 1.0
    assert top1_position(probs) == -1


def test_top1_position_matches_argmax(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    top_token = int(probs.argmax())
    expected = int(itos[top_token]) if top_token != PAD_ID else -1
    assert top1_position(probs) == expected


# ---------------------------------------------------------------------------
# topn_positions
# ---------------------------------------------------------------------------

def test_topn_positions_returns_set(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    result = topn_positions(probs, 5)
    assert isinstance(result, set)


def test_topn_positions_count(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    for n in [1, 3, 5, 10]:
        result = topn_positions(probs, n)
        assert len(result) == n


def test_topn_positions_no_pad(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    result = topn_positions(probs, 10)
    assert PAD_ID not in result


def test_topn_positions_valid_board_positions(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    result = topn_positions(probs, 10)
    for pos in result:
        assert 0 <= pos < 64


def test_topn_positions_top1_consistent(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    top1 = top1_position(probs)
    top5 = topn_positions(probs, 5)
    if top1 != -1:
        assert top1 in top5


# ---------------------------------------------------------------------------
# topn_errors
# ---------------------------------------------------------------------------

def test_topn_errors_empty_legal_set_returns_zero(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    assert topn_errors(probs, set()) == pytest.approx(0.0)


def test_topn_errors_returns_float(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    result = topn_errors(probs, {10, 20, 30})
    assert isinstance(result, float)


def test_topn_errors_nonnegative(random_model):
    x = encode_sequence(OPENING, device=torch.device("cpu"))
    probs = forward_pass(random_model, x, len(OPENING))
    result = topn_errors(probs, {10, 20, 30})
    assert result >= 0.0


def test_topn_errors_perfect_match_is_zero():
    # Put all probability mass on tokens for positions 10, 20, 30
    probs = torch.zeros(VOCAB_SIZE)
    positions = {10, 20, 30}
    for pos in positions:
        probs[stoi[pos]] = 1.0 / len(positions)
    assert topn_errors(probs, positions) == pytest.approx(0.0)


def test_topn_errors_complete_mismatch():
    # Top-3 tokens predict positions 10, 20, 30; legal set is {40, 50, 60}
    probs = torch.zeros(VOCAB_SIZE)
    predicted = {10, 20, 30}
    legal = {40, 50, 60}
    for pos in predicted:
        probs[stoi[pos]] = 1.0 / len(predicted)
    # All 3 predicted are false positives, all 3 legal are false negatives
    assert topn_errors(probs, legal) == pytest.approx(6.0)


def test_topn_errors_symmetric():
    # Swapping predicted and legal should not matter for symmetric set difference
    probs_a = torch.zeros(VOCAB_SIZE)
    probs_a[stoi[10]] = 0.6
    probs_a[stoi[20]] = 0.3
    probs_a[stoi[30]] = 0.1
    legal_a = {10, 20, 40}  # 10 and 20 match, 30 vs 40 mismatch
    assert topn_errors(probs_a, legal_a) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------

def test_rollout_returns_list(random_model, opening_board):
    board, next_player = opening_board
    result = rollout(random_model, OPENING, board, next_player, n_steps=3, device=torch.device("cpu"))
    assert isinstance(result, list)


def test_rollout_length_at_most_n_steps(random_model, opening_board):
    board, next_player = opening_board
    for n_steps in [1, 3, 5]:
        result = rollout(random_model, OPENING, board, next_player, n_steps=n_steps, device=torch.device("cpu"))
        assert len(result) <= n_steps


def test_rollout_elements_are_bool(random_model, opening_board):
    board, next_player = opening_board
    result = rollout(random_model, OPENING, board, next_player, n_steps=5, device=torch.device("cpu"))
    for element in result:
        assert isinstance(element, bool)


def test_rollout_zero_steps(random_model, opening_board):
    board, next_player = opening_board
    result = rollout(random_model, OPENING, board, next_player, n_steps=0, device=torch.device("cpu"))
    assert result == []


def test_rollout_does_not_modify_board(random_model, opening_board):
    board, next_player = opening_board
    board_before = board.copy()
    rollout(random_model, OPENING, board, next_player, n_steps=5, device=torch.device("cpu"))
    assert np.array_equal(board, board_before)


def test_rollout_does_not_modify_sequence(random_model, opening_board):
    board, next_player = opening_board
    seq_before = list(OPENING)
    rollout(random_model, OPENING, board, next_player, n_steps=5, device=torch.device("cpu"))
    assert OPENING == seq_before
