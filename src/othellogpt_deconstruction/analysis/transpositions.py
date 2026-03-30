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
# Board encoding helpers
# ---------------------------------------------------------------------------

def _othello_snapshot(board: np.ndarray) -> tuple:
    return tuple(int(x) for x in board)


def _board_to_masks(board: np.ndarray) -> tuple[np.uint64, np.uint64]:
    """Encode a board as (black_mask, white_mask) numpy uint64 scalars."""
    white = np.int8(3 - BLACK)
    black_mask = np.packbits((board == BLACK).astype(np.uint8), bitorder='little').view(np.uint64)[0]
    white_mask = np.packbits((board == white).astype(np.uint8), bitorder='little').view(np.uint64)[0]
    return black_mask, white_mask


def _masks_to_board(black_mask: int, white_mask: int) -> np.ndarray:
    """Reconstruct a board array from bitmasks."""
    board = np.zeros(64, dtype=np.int8)
    white = 3 - BLACK
    for i in range(64):
        if black_mask & (1 << i):
            board[i] = BLACK
        elif white_mask & (1 << i):
            board[i] = white
    return board


# ---------------------------------------------------------------------------
# Core finder — two-phase API
# ---------------------------------------------------------------------------

def index_games(
    games:   list[list[str]],
    min_ply: int = 2,
    max_ply: int = 59,
) -> dict[tuple, set[tuple]]:
    """
    Index games into a state dict.

    Returns a dict mapping (ply, *board_snapshot) -> set of sequence tuples.
    Pass the result to build_groups to get TranspositionGroup objects.
    """
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

            actual_ply = ply + 1
            if min_ply <= actual_ply <= max_ply:
                snapshot = (actual_ply,) + _othello_snapshot(board)
                state_index[snapshot].add(tuple(game[:actual_ply]))

    return state_index


def _annotate_group(
    ply: int,
    board: np.ndarray,
    seq_list: list[list[str]],
) -> TranspositionGroup:
    """Build a TranspositionGroup with trichrome annotation from a sequence list."""
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

    trichrome_diffs = []
    for i in range(len(trichrome_groups)):
        for j in range(i + 1, len(trichrome_groups)):
            cells = trichrome_diff(
                trichrome_groups[i].trichrome_board,
                trichrome_groups[j].trichrome_board,
            )
            if cells:
                trichrome_diffs.append(TrichromeGroupDiff(
                    group_a=i, group_b=j, differing_cells=cells,
                ))

    return TranspositionGroup(
        ply=ply,
        board=board,
        sequences=seq_list,
        trichrome_groups=trichrome_groups,
        trichrome_diffs=trichrome_diffs,
    )


def build_groups(
    state_index: dict[tuple, set[tuple]],
) -> list[TranspositionGroup]:
    """
    Build TranspositionGroup objects from a completed state_index.

    Returns a list sorted by ply.
    """
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
        results.append(_annotate_group(ply, board, seq_list))

    return sorted(results, key=lambda g: g.ply)


def find_transpositions(
    games:   list[list[str]],
    min_ply: int = 2,
    max_ply: int = 59,
) -> list[TranspositionGroup]:
    """
    Find all transposition groups in a corpus already loaded into memory.

    Convenience wrapper around index_games + build_groups.
    For large corpora, use index_games per file then build_groups instead.
    """
    return build_groups(index_games(games, min_ply=min_ply, max_ply=max_ply))


# ---------------------------------------------------------------------------
# Memory-efficient two-pass API for large corpora
# ---------------------------------------------------------------------------

def compact_index_games(
    games:         list[list[str]],
    compact_index: dict[tuple, set[tuple[int, int]]],
    file_idx:      int,
    min_ply:       int = 2,
    max_ply:       int = 59,
) -> None:
    """
    Index games into compact_index in-place using bitmask board keys.

    compact_index maps (ply, black_mask, white_mask) -> set of (file_idx, game_idx).
    Stores only game references rather than full sequences, so memory usage is
    proportional to unique board states rather than total sequence data.

    Use build_groups_from_compact after a second pass to retrieve full sequences
    for transposition groups.
    """
    for game_idx, game in enumerate(games):
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

            actual_ply = ply + 1
            if min_ply <= actual_ply <= max_ply:
                black_mask, white_mask = _board_to_masks(board)
                compact_index[(actual_ply, black_mask, white_mask)].add((file_idx, game_idx))


def build_groups_from_compact(
    compact_index: dict[tuple, set[tuple[int, int]]],
    game_lookup:   dict[tuple[int, int], list[str]],
) -> list[TranspositionGroup]:
    """
    Build TranspositionGroup objects from a compact index and a game lookup table.

    compact_index: output of compact_index_games, filtered to entries with 2+ refs.
    game_lookup: maps (file_idx, game_idx) -> full game sequence (list of moves).
    """
    results = []
    seen: set[frozenset] = set()

    for key, refs in compact_index.items():
        ply, black_mask, white_mask = key

        seqs = set()
        for file_idx, game_idx in refs:
            game = game_lookup.get((file_idx, game_idx))
            if game is not None:
                seqs.add(tuple(game[:ply]))

        if len(seqs) < 2:
            continue

        canonical = frozenset(seqs)
        if canonical in seen:
            continue
        seen.add(canonical)

        board = _masks_to_board(black_mask, white_mask)
        seq_list = [list(s) for s in seqs]
        results.append(_annotate_group(ply, board, seq_list))

    return sorted(results, key=lambda g: g.ply)


# ---------------------------------------------------------------------------
# Numpy chunked API for very large corpora (10M+ games)
# ---------------------------------------------------------------------------

def index_chunk_games(
    games:     list[list[str]],
    file_idx:  int,
    ply_start: int,
    ply_end:   int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode game states at plies [ply_start, ply_end] as numpy arrays.

    Returns five parallel arrays, one entry per (game, ply) observation:
        ply_array      : int16  — ply number
        black_mask_arr : uint64 — black occupancy bitmask
        white_mask_arr : uint64 — white occupancy bitmask
        file_idx_arr   : int32  — index of source file
        game_idx_arr   : int32  — index of game within that file

    All arrays have the same length (number of game-ply pairs in range).
    Using numpy arrays here costs ~32 bytes/entry vs. ~430 bytes/entry for
    a Python dict, making it feasible to process 17M games at a time.
    """
    ply_list       = []
    black_mask_list = []
    white_mask_list = []
    file_idx_list  = []
    game_idx_list  = []

    for game_idx, game in enumerate(games):
        board  = othello_start_board()
        player = BLACK

        for move_num, move in enumerate(game):
            if move_num >= ply_end:
                break

            pos = alg_to_pos(move)
            try:
                board = othello_apply_move(board, pos, player)
            except ValueError:
                break
            player = 3 - player
            if not legal_moves(board, player):
                player = 3 - player

            actual_ply = move_num + 1
            if ply_start <= actual_ply < ply_end:
                black_mask, white_mask = _board_to_masks(board)
                ply_list.append(actual_ply)
                black_mask_list.append(black_mask)
                white_mask_list.append(white_mask)
                file_idx_list.append(file_idx)
                game_idx_list.append(game_idx)

    if not ply_list:
        empty_int16  = np.empty(0, dtype=np.int16)
        empty_uint64 = np.empty(0, dtype=np.uint64)
        empty_int32  = np.empty(0, dtype=np.int32)
        return empty_int16, empty_uint64, empty_uint64, empty_int32, empty_int32

    return (
        np.array(ply_list,        dtype=np.int16),
        np.array(black_mask_list, dtype=np.uint64),
        np.array(white_mask_list, dtype=np.uint64),
        np.array(file_idx_list,   dtype=np.int32),
        np.array(game_idx_list,   dtype=np.int32),
    )


def find_candidates_from_arrays(
    ply_array:       np.ndarray,
    black_mask_array: np.ndarray,
    white_mask_array: np.ndarray,
    file_idx_array:  np.ndarray,
    game_idx_array:  np.ndarray,
) -> dict[tuple, set[tuple[int, int]]]:
    """
    Find transposition candidates using a sort-and-group approach.

    Groups rows by (ply, black_mask, white_mask) using numpy lexsort,
    then collects (file_idx, game_idx) refs for each group with 2+ distinct
    games.

    Returns a dict mapping (ply, black_mask, white_mask) -> set of (file_idx, game_idx),
    compatible with build_groups_from_compact.
    """
    if len(ply_array) == 0:
        return {}

    # Sort by (ply, black_mask, white_mask) — lexsort keys are applied right-to-left
    sort_order = np.lexsort((white_mask_array, black_mask_array, ply_array))

    ply_sorted        = ply_array[sort_order]
    black_mask_sorted = black_mask_array[sort_order]
    white_mask_sorted = white_mask_array[sort_order]
    file_idx_sorted   = file_idx_array[sort_order]
    game_idx_sorted   = game_idx_array[sort_order]

    candidates: dict[tuple, set[tuple[int, int]]] = {}

    # Find boundaries where the (ply, black_mask, white_mask) key changes
    key_changed = np.ones(len(ply_sorted), dtype=bool)
    key_changed[1:] = (
        (ply_sorted[1:]        != ply_sorted[:-1]) |
        (black_mask_sorted[1:] != black_mask_sorted[:-1]) |
        (white_mask_sorted[1:] != white_mask_sorted[:-1])
    )
    group_starts = np.where(key_changed)[0]

    for ordinal, group_start in enumerate(group_starts):
        group_end = group_starts[ordinal + 1] if ordinal + 1 < len(group_starts) else len(ply_sorted)

        if group_end - group_start < 2:
            continue

        refs = set(
            zip(
                file_idx_sorted[group_start:group_end].tolist(),
                game_idx_sorted[group_start:group_end].tolist(),
            )
        )
        if len(refs) < 2:
            continue

        key = (
            int(ply_sorted[group_start]),
            int(black_mask_sorted[group_start]),
            int(white_mask_sorted[group_start]),
        )
        if key in candidates:
            candidates[key].update(refs)
        else:
            candidates[key] = refs

    return candidates


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
