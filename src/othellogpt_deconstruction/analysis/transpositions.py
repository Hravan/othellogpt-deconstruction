"""
src/othellogpt_deconstruction/analysis/transpositions.py

Transposition group finder with Trichrome annotation.

A transposition group is a set of distinct move sequences that all reach
the same Othello board state (S_O) at the same ply.

Each group is annotated with Trichrome subgroups — partitions of the
sequences by their Trichrome state (S_T). Groups where sequences differ
in S_T are flagged with cell-level diffs.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from othellogpt_deconstruction.core.tokenizer import alg_to_pos
from othellogpt_deconstruction.core.board import (
    BLACK, legal_moves, apply_move as othello_apply_move,
    start_board as othello_start_board,
)
from othellogpt_deconstruction.core.trichrome import (
    replay as trichrome_replay,
    diff as trichrome_diff,
    trichrome_key,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrichroneSubgroup:
    trichrome_board: np.ndarray          # (64, 2) array
    sequences:       list[list[str]]     # all sequences in this S_T class


@dataclass
class TrichromeGroupDiff:
    group_a:         int                 # index into TranspositionGroup.trichrome_groups
    group_b:         int
    differing_cells: list[dict]          # from trichrome.diff()


@dataclass
class TranspositionGroup:
    ply:              int
    board:            np.ndarray         # (64,) Othello board
    sequences:        list[list[str]]    # all sequences reaching this S_O
    trichrome_groups: list[TrichroneSubgroup]
    trichrome_diffs:  list[TrichromeGroupDiff]

    @property
    def has_trichrome_diffs(self) -> bool:
        return len(self.trichrome_diffs) > 0

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    @property
    def n_trichrome_states(self) -> int:
        return len(self.trichrome_groups)


# ---------------------------------------------------------------------------
# Othello board snapshot (for grouping)
# ---------------------------------------------------------------------------

def _othello_snapshot(board: np.ndarray) -> tuple:
    return tuple(int(x) for x in board)


# ---------------------------------------------------------------------------
# Core finder
# ---------------------------------------------------------------------------

def find_transpositions(
    games:   list[list[str]],
    min_ply: int = 2,
    max_ply: int = 59,
) -> list[TranspositionGroup]:
    """
    Find all transposition groups in a corpus.

    A transposition group is a set of 2+ distinct sequences that reach
    the same Othello board state at the same ply.

    Parameters
    ----------
    games:   list of games, each a list of algebraic move strings
    min_ply: minimum sequence length to consider
    max_ply: maximum sequence length to consider

    Returns
    -------
    List of TranspositionGroup, sorted by ply.
    """
    # Map (ply, board_snapshot) -> set of sequences
    state_index: dict[tuple, set[tuple]] = defaultdict(set)

    for game in games:
        board = othello_start_board()
        player = BLACK

        for ply, move in enumerate(game):
            if ply > max_ply:
                break

            pos = alg_to_pos(move)
            try:
                board = othello_apply_move(board, pos, player)
            except ValueError:
                break
            player = 3 - player
            if not legal_moves(board, player):
                player = 3 - player

            # Record after applying move — ply is 1-indexed here
            # (ply=1 means 1 move has been played)
            actual_ply = ply + 1
            if min_ply <= actual_ply <= max_ply:
                snapshot = (actual_ply,) + _othello_snapshot(board)
                state_index[snapshot].add(tuple(game[:actual_ply]))

    results = []
    seen: set[frozenset] = set()

    for key, seqs in state_index.items():
        if len(seqs) < 2:
            continue

        canonical = frozenset(seqs)
        if canonical in seen:
            continue
        seen.add(canonical)

        ply = key[0]
        board = np.array(key[1:], dtype=np.int8)
        seq_list = [list(s) for s in seqs]

        # Partition sequences by Trichrome state
        tc_groups: dict[tuple, list[list[str]]] = defaultdict(list)
        for seq in seq_list:
            tc_board, _ = trichrome_replay(seq)
            tc_groups[trichrome_key(tc_board)].append(seq)

        tc_snap_list = list(tc_groups.keys())
        trichrome_groups = [
            TrichroneSubgroup(
                trichrome_board=np.array(tc_snap_list[i], dtype=np.int8).reshape(64, 2),
                sequences=tc_groups[tc_snap_list[i]],
            )
            for i in range(len(tc_snap_list))
        ]

        # Compute pairwise diffs between Trichrome subgroups
        trichrome_diffs = []
        for i in range(len(trichrome_groups)):
            for j in range(i + 1, len(trichrome_groups)):
                cells = trichrome_diff(
                    trichrome_groups[i].trichrome_board,
                    trichrome_groups[j].trichrome_board,
                )
                if cells:
                    trichrome_diffs.append(TrichromeGroupDiff(
                        group_a=i,
                        group_b=j,
                        differing_cells=cells,
                    ))

        results.append(TranspositionGroup(
            ply=ply,
            board=board,
            sequences=seq_list,
            trichrome_groups=trichrome_groups,
            trichrome_diffs=trichrome_diffs,
        ))

    return sorted(results, key=lambda g: g.ply)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise(groups: list[TranspositionGroup]) -> dict:
    """Return summary statistics over a list of transposition groups."""
    if not groups:
        return {}

    mixed = [g for g in groups if g.has_trichrome_diffs]
    sizes = [g.n_sequences for g in groups]
    plies = [g.ply for g in groups]

    all_dists = [
        cell["distance"]
        for g in groups
        for diff in g.trichrome_diffs
        for cell in diff.differing_cells
    ]

    return {
        "total_groups":        len(groups),
        "mixed_trichrome":     len(mixed),
        "same_trichrome":      len(groups) - len(mixed),
        "ply_min":             min(plies),
        "ply_max":             max(plies),
        "group_size_min":      min(sizes),
        "group_size_max":      max(sizes),
        "group_size_avg":      sum(sizes) / len(sizes),
        "color_dist_counts":   _count(all_dists),
    }


def _count(values: list) -> dict:
    result: dict = {}
    for v in values:
        result[v] = result.get(v, 0) + 1
    return dict(sorted(result.items()))
