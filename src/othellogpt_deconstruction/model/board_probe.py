"""
src/othellogpt_deconstruction/model/board_probe.py

Board state probe: predicts cell ownership (EMPTY / MINE / YOURS) from the
residual stream, using the same structure as TrichromeProbe.

MINE  = cells owned by the player who moves next at this position
YOURS = cells owned by the opponent

This is the relative-player encoding Nanda et al. use.  The probe direction
for flipping cell pos from MINE to YOURS is W[pos, YOURS] - W[pos, MINE].

Shared helpers make_labels() and labels_to_absolute_board() are used by both
train_board_probes.py and nanda_intervention.py.
"""

import numpy as np
import torch

from othellogpt_deconstruction.model.probes import TrichromeProbe, save_probes, load_probes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY: int = 0
MINE:  int = 1
YOURS: int = 2

# BoardStateProbe has identical structure to TrichromeProbe:
#   weights : (64, 3, d_model)  — W[pos, class] is the score vector for EMPTY/MINE/YOURS
#   biases  : (64, 3)
BoardStateProbe = TrichromeProbe


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def make_labels(board: np.ndarray, next_player: int) -> np.ndarray:
    """
    Convert an absolute board (EMPTY=0, BLACK=1, WHITE=2) and the player who
    moves next into a relative-player label array (EMPTY=0, MINE=1, YOURS=2).

    Parameters
    ----------
    board       : (64,) int array from board.replay
    next_player : 1 (BLACK) or 2 (WHITE)

    Returns
    -------
    (64,) int8 array with values in {EMPTY, MINE, YOURS}
    """
    labels = np.zeros(64, dtype=np.int8)
    opponent = 3 - next_player
    for pos in range(64):
        if board[pos] == next_player:
            labels[pos] = MINE
        elif board[pos] == opponent:
            labels[pos] = YOURS
        # else: EMPTY stays 0
    return labels


def labels_to_absolute_board(labels: np.ndarray, next_player: int) -> np.ndarray:
    """
    Convert a relative-player label array back to an absolute board.

    Parameters
    ----------
    labels      : (64,) int8 array with values in {EMPTY, MINE, YOURS}
    next_player : 1 (BLACK) or 2 (WHITE)

    Returns
    -------
    (64,) int8 array with values in {0=EMPTY, 1=BLACK, 2=WHITE}
    """
    board = np.zeros(64, dtype=np.int8)
    opponent = 3 - next_player
    for pos in range(64):
        if labels[pos] == MINE:
            board[pos] = next_player
        elif labels[pos] == YOURS:
            board[pos] = opponent
    return board


# ---------------------------------------------------------------------------
# Decode helper
# ---------------------------------------------------------------------------

def decode_board(probe: TrichromeProbe, activation: torch.Tensor) -> np.ndarray:
    """
    Run the probe on an activation vector and return per-cell class predictions.

    Parameters
    ----------
    probe      : trained BoardStateProbe
    activation : (d_model,) float tensor

    Returns
    -------
    (64,) int8 array with values in {EMPTY, MINE, YOURS}
    """
    logits = probe.logits(activation)          # (64, 3)
    return logits.argmax(dim=-1).numpy().astype(np.int8)
