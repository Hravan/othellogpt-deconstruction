"""
src/othellogpt_deconstruction/model/probe_intervention.py

Utilities for gradient-based probe intervention experiments.

These helpers load and configure the models and probes used in Li et al.'s
intervention setup, plus a non-validating board replay needed for their
"unnatural" benchmark of illegal-move sequences.

Board encoding note
-------------------
Li's probes use absolute board encoding:
    0 = white piece
    1 = empty
    2 = black piece

Our board uses:
    EMPTY = 0, BLACK = 1, WHITE = 2

Conversion: li_label = (our_label + 1) % 3

This module is the only place where mingpt.model.GPTforProbeIA and
mingpt.probe_model.BatteryProbeClassificationTwoLayer are imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Li's model and probe classes live in the mingpt subtree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "mingpt"))
from mingpt.model import GPTConfig, GPTforProbeIA  # noqa: E402
from mingpt.probe_model import BatteryProbeClassificationTwoLayer  # noqa: E402

from othellogpt_deconstruction.core.board import (
    BLACK, flipped_by, legal_moves, start_board,
)


def board_to_probe_encoding(board: np.ndarray) -> torch.Tensor:
    """
    Convert our board array to Li's probe encoding as a long tensor of shape (64,).

    Our encoding: EMPTY=0, BLACK=1, WHITE=2
    Li's encoding: white=0, empty=1, black=2

    Conversion: li_label = (our_label + 1) % 3
    """
    return torch.tensor((board + 1) % 3, dtype=torch.long)


def load_probe_intervention_model(
    checkpoint_path: str | Path,
    probe_layer: int,
    device: torch.device,
) -> GPTforProbeIA:
    """
    Load GPTforProbeIA (Li's intervention-capable model variant).

    Parameters
    ----------
    checkpoint_path : path to the .ckpt file
    probe_layer     : which layer to expose for gradient-based intervention
    device          : target device
    """
    mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
    model = GPTforProbeIA(mconf, probe_layer=probe_layer)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_cell_ownership_probes(
    probe_dir: str | Path,
    layers: list[int],
    device: torch.device,
    mid_dim: int = 128,
) -> dict[int, BatteryProbeClassificationTwoLayer]:
    """
    Load per-layer cell ownership probes (BatteryProbeClassificationTwoLayer).

    Parameters
    ----------
    probe_dir : directory containing layer subdirectories
    layers    : which layers to load
    device    : target device
    mid_dim   : hidden dimension of the two-layer probe (default 128, per Li et al.)

    Returns
    -------
    Dict mapping layer index → loaded probe in eval mode.
    """
    probe_dir = Path(probe_dir)
    probes: dict[int, BatteryProbeClassificationTwoLayer] = {}
    for layer in layers:
        probe = BatteryProbeClassificationTwoLayer(
            device=device, probe_class=3, num_task=64, mid_dim=mid_dim,
        )
        checkpoint_path = probe_dir / f"layer{layer}" / "checkpoint.ckpt"
        probe.load_state_dict(torch.load(checkpoint_path, map_location=device))
        probe.eval()
        probes[layer] = probe
    return probes


def replay_nonvalidating(board_positions: list[int]) -> tuple[np.ndarray, int]:
    """
    Replay a sequence of board positions without legality checks.

    Used for Li's unnatural benchmark, where sequences include illegal moves.
    Pieces are placed and flanks flipped as normal, but the move itself need
    not be legal on the current board.

    Parameters
    ----------
    board_positions : list of board positions (0-63)

    Returns
    -------
    (board, player) — final board state and next player to move (BLACK=1 or WHITE=2).
    """
    board = start_board()
    player = BLACK
    for pos in board_positions:
        flips = flipped_by(board, pos, player)
        new_board = board.copy()
        new_board[pos] = player
        for fp in flips:
            new_board[fp] = player
        board = new_board
        player = 3 - player
        if not legal_moves(board, player):
            player = 3 - player
    return board, player
