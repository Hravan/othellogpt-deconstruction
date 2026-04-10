"""
src/othellogpt_deconstruction/model/representation_alignment.py

Utilities for representation alignment experiments (Yuan et al. 2025).

These helpers load GPTforProbing and build padded token batches from
raw board-position sequences (as opposed to algebraic strings). The
per-experiment extract_representations functions are left in each script
because the two extraction loops differ in model type and shuffling logic.

This module is the only place where mingpt.model.GPTforProbing is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# GPTforProbing lives in the mingpt subtree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "mingpt"))
from mingpt.model import GPTConfig, GPTforProbing  # noqa: E402

from othellogpt_deconstruction.core.tokenizer import BLOCK_SIZE, PAD_ID, VOCAB_SIZE, stoi

N_EMBD: int = 512


def load_othello_gpt(
    checkpoint_path: str | Path,
    device: torch.device,
) -> GPTforProbing:
    """
    Load OthelloGPT as GPTforProbing so that ln_f hidden states are accessible.

    Parameters
    ----------
    checkpoint_path : path to the .ckpt file
    device          : target device

    Returns
    -------
    GPTforProbing model in eval mode.
    """
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=8,
        n_head=8,
        n_embd=N_EMBD,
    )
    model = GPTforProbing(config, probe_layer=8, ln=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_token_batch(
    games: list[list[int]],
    positions: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """
    Build a padded token batch for a list of (game, step) pairs.

    Parameters
    ----------
    games     : list of games (each game is a list of board positions 0-63)
    positions : for each game, the step index (0-based) to extract at

    Returns
    -------
    token_ids : (N, BLOCK_SIZE) long tensor
    seq_lens  : list of actual sequence lengths (= step + 1)
    """
    token_ids_list = []
    seq_lens = []
    for game, step in zip(games, positions):
        prefix = game[: step + 1]
        seq_len = min(len(prefix), BLOCK_SIZE)
        tokens = [stoi[pos] for pos in prefix[:seq_len]]
        padded = tokens + [PAD_ID] * (BLOCK_SIZE - seq_len)
        token_ids_list.append(padded)
        seq_lens.append(seq_len)
    if token_ids_list:
        token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)
    else:
        token_ids = torch.zeros((0, BLOCK_SIZE), dtype=torch.long, device=device)
    return token_ids, seq_lens
