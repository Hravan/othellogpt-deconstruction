"""
src/othellogpt_deconstruction/model/utils.py

Shared inference utilities used across experiment scripts.

These helpers sit below the level of inference.py (which wraps the full
get_distribution pipeline) and above raw mingpt calls. They operate on
tensors rather than algebraic strings and are designed to be composed
inside loops that intervene on the residual stream between steps.
"""

from __future__ import annotations

import numpy as np
import torch

from othellogpt_deconstruction.core.board import apply_move, legal_moves
from othellogpt_deconstruction.core.tokenizer import (
    BLOCK_SIZE, PAD_ID, itos, pos_to_alg, stoi, alg_to_pos,
)


def encode_sequence(sequence: list[str], device: torch.device) -> torch.Tensor:
    """
    Encode a list of algebraic move strings into a padded token tensor.

    Returns
    -------
    Shape (1, BLOCK_SIZE) long tensor, padded with PAD_ID.
    """
    tokens = [stoi[alg_to_pos(move)] for move in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


def forward_pass(
    model: torch.nn.Module,
    x: torch.Tensor,
    seq_length: int,
) -> torch.Tensor:
    """
    Run a single forward pass and return the next-move probability distribution.

    Parameters
    ----------
    model      : OthelloGPT model
    x          : (1, BLOCK_SIZE) token tensor
    seq_length : number of real tokens in x (used to index the last position)

    Returns
    -------
    1-D softmax probability tensor of shape (VOCAB_SIZE,).
    PAD token is set to -inf before softmax.
    """
    with torch.no_grad():
        logits, _ = model(x)
    last_logits = logits[0, seq_length - 1, :].clone()
    last_logits[PAD_ID] = float("-inf")
    return torch.softmax(last_logits, dim=-1)


def top1_position(probs: torch.Tensor) -> int:
    """
    Return the board position of the highest-probability predicted token.

    Returns -1 if the top token is PAD.
    """
    token = int(probs.argmax())
    return int(itos[token]) if token != PAD_ID else -1


def topn_positions(probs: torch.Tensor, n: int) -> set[int]:
    """
    Return the board positions of the top-n predicted tokens, skipping PAD.
    """
    top_tokens = probs.topk(n + 1).indices  # +1 in case PAD slips in
    positions: set[int] = set()
    for token in top_tokens:
        token = int(token)
        if token != PAD_ID:
            positions.add(int(itos[token]))
        if len(positions) == n:
            break
    return positions


def topn_errors(probs: torch.Tensor, legal_set: set[int]) -> float:
    """
    Compute the top-N error rate used by Nanda et al.

    Takes the top-N predicted positions (N = |legal_set|) and counts
    false positives + false negatives against legal_set.

    Returns 0.0 if legal_set is empty.
    """
    n = len(legal_set)
    if n == 0:
        return 0.0
    predicted = topn_positions(probs, n)
    return float(len(predicted - legal_set) + len(legal_set - predicted))


def rollout(
    model: torch.nn.Module,
    sequence: list[str],
    starting_board: np.ndarray,
    next_player: int,
    n_steps: int,
    device: torch.device,
) -> list[bool]:
    """
    Roll out n_steps moves from starting_board with no intervention.

    At each step feed the current sequence to the model, take top-1, and check
    legality. Always appends the predicted move to the sequence (even if
    illegal) so the model sees its own outputs.

    Returns
    -------
    List of booleans of length ≤ n_steps, where True = legal move predicted.
    """
    board = starting_board.copy()
    player = next_player
    seq = list(sequence)
    results = []

    for _ in range(n_steps):
        if len(seq) >= BLOCK_SIZE:
            break
        x = encode_sequence(seq, device)
        probs = forward_pass(model, x, len(seq))
        pos = top1_position(probs)
        if pos < 0:
            break

        legal_set = set(legal_moves(board, player))
        is_legal = pos in legal_set
        results.append(is_legal)

        if is_legal:
            board = apply_move(board, pos, player)
            player = 3 - player
            if not legal_moves(board, player):
                player = 3 - player

        seq.append(pos_to_alg(pos))

    return results
