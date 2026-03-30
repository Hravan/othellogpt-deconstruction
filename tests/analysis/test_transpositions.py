"""
tests/analysis/test_transpositions.py
"""
import numpy as np

from othellogpt_deconstruction.analysis.transpositions import (
    find_transpositions, index_games, build_groups, summarise,
    index_chunk_games, find_candidates_from_arrays, build_groups_from_compact,
)

# ---------------------------------------------------------------------------
# Minimal corpus containing the known transposition pair
# ---------------------------------------------------------------------------

KNOWN_PAIR = [
    ["f5", "f6", "d3", "f4", "g5"],
    ["f5", "f4", "d3", "f6", "g5"],
]

# A few extra games to pad the corpus
EXTRA_GAMES = [
    ["f5", "d6", "c5", "f4", "e3"],
    ["c4", "c5", "e6", "c3", "b4"],
    ["f5", "f6", "c4", "c3", "e6"],
]

CORPUS = KNOWN_PAIR + EXTRA_GAMES


# ---------------------------------------------------------------------------
# find_transpositions — basic properties
# ---------------------------------------------------------------------------

def test_returns_list():
    groups = find_transpositions(CORPUS)
    assert isinstance(groups, list)


def test_empty_corpus():
    assert find_transpositions([]) == []


def test_single_game_no_groups():
    groups = find_transpositions([KNOWN_PAIR[0]])
    assert groups == []


def test_sorted_by_ply():
    groups = find_transpositions(CORPUS)
    plies = [g.ply for g in groups]
    assert plies == sorted(plies)


def test_all_groups_have_at_least_two_sequences():
    groups = find_transpositions(CORPUS)
    for g in groups:
        assert g.n_sequences >= 2


def test_no_duplicate_groups():
    groups = find_transpositions(CORPUS)
    seen = set()
    for g in groups:
        key = frozenset(tuple(s) for s in g.sequences)
        assert key not in seen
        seen.add(key)


# ---------------------------------------------------------------------------
# Known transposition pair
# ---------------------------------------------------------------------------

def test_known_pair_found():
    groups = find_transpositions(CORPUS)
    seq_sets = [frozenset(tuple(s) for s in g.sequences) for g in groups]
    known = frozenset(tuple(s) for s in KNOWN_PAIR)
    assert any(known.issubset(ss) for ss in seq_sets)


def test_known_pair_ply():
    groups = find_transpositions(CORPUS)
    for g in groups:
        if any(s == KNOWN_PAIR[0] for s in g.sequences):
            assert g.ply == 5
            break


def test_known_pair_same_othello_board():
    """Both sequences in the known pair must reach the same Othello board."""
    from othellogpt_deconstruction.core.board import replay as othello_replay
    board_a, _ = othello_replay(KNOWN_PAIR[0])
    board_b, _ = othello_replay(KNOWN_PAIR[1])
    assert (board_a == board_b).all()


def test_known_pair_same_legal_moves():
    """Both sequences must have identical legal move sets."""
    from othellogpt_deconstruction.core.board import legal_moves_after
    legal_a = set(legal_moves_after(KNOWN_PAIR[0]))
    legal_b = set(legal_moves_after(KNOWN_PAIR[1]))
    assert legal_a == legal_b


# ---------------------------------------------------------------------------
# Trichrome annotation
# ---------------------------------------------------------------------------

def test_known_pair_has_trichrome_diffs():
    groups = find_transpositions(CORPUS)
    for g in groups:
        if any(s == KNOWN_PAIR[0] for s in g.sequences):
            assert g.has_trichrome_diffs
            break


def test_known_pair_trichrome_diff_cells():
    groups = find_transpositions(CORPUS)
    for g in groups:
        if any(s == KNOWN_PAIR[0] for s in g.sequences):
            assert g.n_trichrome_states == 2
            diff = g.trichrome_diffs[0]
            squares = {c["square"] for c in diff.differing_cells}
            assert squares == {"e4", "e5"}
            break


def test_trichrome_groups_cover_all_sequences():
    groups = find_transpositions(CORPUS)
    for g in groups:
        all_seqs = [
            s
            for tg in g.trichrome_groups
            for s in tg.sequences
        ]
        assert len(all_seqs) == g.n_sequences


def test_trichrome_subgroup_boards_shape():
    groups = find_transpositions(CORPUS)
    for g in groups:
        for tg in g.trichrome_groups:
            assert tg.trichrome_board.shape == (64, 2)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

def test_summarise_empty():
    assert summarise([]) == {}


def test_summarise_keys():
    groups = find_transpositions(CORPUS)
    s = summarise(groups)
    assert "total_groups" in s
    assert "mixed_trichrome" in s
    assert "same_trichrome" in s
    assert "color_dist_counts" in s


def test_summarise_counts_add_up():
    groups = find_transpositions(CORPUS)
    s = summarise(groups)
    assert s["mixed_trichrome"] + s["same_trichrome"] == s["total_groups"]


def test_summarise_mixed_at_least_one():
    groups = find_transpositions(CORPUS)
    s = summarise(groups)
    assert s["mixed_trichrome"] >= 1


# ---------------------------------------------------------------------------
# index_games + build_groups
# ---------------------------------------------------------------------------

def test_index_build_matches_find_transpositions():
    """index_games + build_groups should produce the same groups as find_transpositions."""
    groups = build_groups(index_games(CORPUS))
    expected = find_transpositions(CORPUS)

    assert len(groups) == len(expected)
    result_keys   = {frozenset(tuple(s) for s in g.sequences) for g in groups}
    expected_keys = {frozenset(tuple(s) for s in g.sequences) for g in expected}
    assert result_keys == expected_keys


def test_index_games_empty():
    assert index_games([]) == {}


def test_build_groups_empty_index():
    assert build_groups({}) == []


def test_build_groups_no_duplicates():
    groups = build_groups(index_games(CORPUS))
    seen = set()
    for g in groups:
        key = frozenset(tuple(s) for s in g.sequences)
        assert key not in seen
        seen.add(key)


def test_build_groups_sorted_by_ply():
    groups = build_groups(index_games(CORPUS))
    plies = [g.ply for g in groups]
    assert plies == sorted(plies)


def test_index_games_min_max_ply():
    """min_ply/max_ply should restrict which positions are indexed."""
    state_index_restricted = index_games(CORPUS, min_ply=6, max_ply=59)
    for key in state_index_restricted:
        assert key[0] >= 6


# ---------------------------------------------------------------------------
# index_chunk_games + find_candidates_from_arrays + build_groups_from_compact
# ---------------------------------------------------------------------------

def test_index_chunk_games_returns_five_arrays():
    ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr = \
        index_chunk_games(CORPUS, file_idx=0, ply_start=2, ply_end=6)
    assert len(ply_arr) == len(black_mask_arr) == len(white_mask_arr) \
        == len(file_idx_arr) == len(game_idx_arr)


def test_index_chunk_games_dtypes():
    ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr = \
        index_chunk_games(CORPUS, file_idx=0, ply_start=2, ply_end=6)
    assert ply_arr.dtype        == np.int16
    assert black_mask_arr.dtype == np.uint64
    assert white_mask_arr.dtype == np.uint64
    assert file_idx_arr.dtype   == np.int32
    assert game_idx_arr.dtype   == np.int32


def test_index_chunk_games_ply_range():
    ply_arr, *_ = index_chunk_games(CORPUS, file_idx=0, ply_start=3, ply_end=5)
    assert (ply_arr >= 3).all()
    assert (ply_arr < 5).all()


def test_index_chunk_games_empty_corpus():
    ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr = \
        index_chunk_games([], file_idx=0, ply_start=2, ply_end=6)
    assert len(ply_arr) == 0


def test_find_candidates_empty():
    empty_int16  = np.empty(0, dtype=np.int16)
    empty_uint64 = np.empty(0, dtype=np.uint64)
    empty_int32  = np.empty(0, dtype=np.int32)
    result = find_candidates_from_arrays(
        empty_int16, empty_uint64, empty_uint64, empty_int32, empty_int32,
    )
    assert result == {}


def test_numpy_pipeline_matches_find_transpositions():
    """index_chunk_games + find_candidates_from_arrays + build_groups_from_compact
    must find the same transposition groups as find_transpositions."""
    # Build candidates via numpy pipeline (one chunk covering all plies)
    ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr = \
        index_chunk_games(CORPUS, file_idx=0, ply_start=2, ply_end=60)
    candidates = find_candidates_from_arrays(
        ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr,
    )

    # game_lookup: all games are in file 0
    game_lookup = {(0, game_idx): game for game_idx, game in enumerate(CORPUS)}

    groups = build_groups_from_compact(candidates, game_lookup)
    expected = find_transpositions(CORPUS)

    assert len(groups) == len(expected)
    result_keys   = {frozenset(tuple(seq) for seq in group.sequences) for group in groups}
    expected_keys = {frozenset(tuple(seq) for seq in group.sequences) for group in expected}
    assert result_keys == expected_keys


def test_numpy_pipeline_known_pair():
    """The numpy pipeline must detect the known transposition pair."""
    ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr = \
        index_chunk_games(CORPUS, file_idx=0, ply_start=2, ply_end=60)
    candidates = find_candidates_from_arrays(
        ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr,
    )
    game_lookup = {(0, game_idx): game for game_idx, game in enumerate(CORPUS)}
    groups = build_groups_from_compact(candidates, game_lookup)

    known = frozenset(tuple(seq) for seq in KNOWN_PAIR)
    seq_sets = [frozenset(tuple(seq) for seq in group.sequences) for group in groups]
    assert any(known.issubset(seq_set) for seq_set in seq_sets)


def test_numpy_pipeline_chunked_matches_single_pass():
    """Processing in two ply chunks must give the same result as one big chunk."""
    game_lookup = {(0, game_idx): game for game_idx, game in enumerate(CORPUS)}

    # Single pass
    ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr = \
        index_chunk_games(CORPUS, file_idx=0, ply_start=2, ply_end=60)
    candidates_single = find_candidates_from_arrays(
        ply_arr, black_mask_arr, white_mask_arr, file_idx_arr, game_idx_arr,
    )
    groups_single = build_groups_from_compact(candidates_single, game_lookup)

    # Two chunks
    ply_arr_a, black_mask_arr_a, white_mask_arr_a, file_idx_arr_a, game_idx_arr_a = \
        index_chunk_games(CORPUS, file_idx=0, ply_start=2, ply_end=5)
    ply_arr_b, black_mask_arr_b, white_mask_arr_b, file_idx_arr_b, game_idx_arr_b = \
        index_chunk_games(CORPUS, file_idx=0, ply_start=5, ply_end=60)

    candidates_chunked: dict = {}
    for chunk_candidates in [
        find_candidates_from_arrays(
            ply_arr_a, black_mask_arr_a, white_mask_arr_a, file_idx_arr_a, game_idx_arr_a,
        ),
        find_candidates_from_arrays(
            ply_arr_b, black_mask_arr_b, white_mask_arr_b, file_idx_arr_b, game_idx_arr_b,
        ),
    ]:
        for key, refs in chunk_candidates.items():
            if key in candidates_chunked:
                candidates_chunked[key].update(refs)
            else:
                candidates_chunked[key] = refs

    groups_chunked = build_groups_from_compact(candidates_chunked, game_lookup)

    result_keys  = {frozenset(tuple(seq) for seq in group.sequences) for group in groups_single}
    chunked_keys = {frozenset(tuple(seq) for seq in group.sequences) for group in groups_chunked}
    assert result_keys == chunked_keys
