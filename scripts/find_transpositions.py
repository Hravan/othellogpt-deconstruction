"""
scripts/find_transpositions.py

Find transposition groups in an Othello corpus and annotate with
Trichrome state diffs.

Uses a sort-merge-on-disk approach to handle large corpora (10M+ games)
without running out of memory:

  Pass 1 — for each corpus file, replay all games once across all plies,
            sort the board-state rows by (ply, black_mask, white_mask),
            and write to a temp binary file.

  Merge  — k-way stream-merge all sorted temp files; collect board states
            seen in 2+ distinct games as transposition candidates.

  Pass 2 — reload only the games that appear in candidate groups; retrieve
            their full move sequences for build_groups_from_compact.

Peak memory is bounded by one file's rows at a time (~150 MB for 100k games).
Temp disk usage is roughly 26 bytes × total game-ply observations (~22 GB for
17M games × 58 plies).

Usage
-----
    # Championship corpus only
    uv run python scripts/find_transpositions.py \\
        --corpus path/to/championship/ \\
        --output data/transpositions.json

    # Both corpora combined
    uv run python scripts/find_transpositions.py \\
        --corpus path/to/championship/ path/to/synthetic/ \\
        --output data/transpositions.json

    # Full synthetic only
    uv run python scripts/find_transpositions.py \\
        --corpus path/to/synthetic/ \\
        --output data/transpositions_synthetic.json
"""

import argparse
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from othellogpt_deconstruction.core.corpus import list_corpus_files, load_file
from othellogpt_deconstruction.analysis.transpositions import (
    write_sorted_file, merge_sorted_files,
    build_groups_from_compact, summarise,
    TranspositionGroup,
)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_group(group: TranspositionGroup) -> dict:
    return {
        "ply":      group.ply,
        "board":    group.board.tolist(),
        "sequences": group.sequences,
        "trichrome_groups": [
            {
                "trichrome_board": tg.trichrome_board.tolist(),
                "sequences":       tg.sequences,
            }
            for tg in group.trichrome_groups
        ],
        "trichrome_diffs": [
            {
                "group_a":         diff.group_a,
                "group_b":         diff.group_b,
                "differing_cells": diff.differing_cells,
            }
            for diff in group.trichrome_diffs
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Othello transposition groups with Trichrome annotation."
    )
    parser.add_argument(
        "--corpus", nargs="+", required=True,
        help="Path(s) to corpus file(s) or directory(s)",
    )
    parser.add_argument(
        "--output", default="data/transpositions.json",
        help="Output JSON path (default: data/transpositions.json)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for group-level train/test split (default: 42)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.1,
        help="Fraction of transposition groups to hold out as test set (default: 0.1)",
    )
    parser.add_argument(
        "--min-ply", type=int, default=2,
        help="Minimum ply to consider (default: 2)",
    )
    parser.add_argument(
        "--max-ply", type=int, default=59,
        help="Maximum ply to consider (default: 59)",
    )
    parser.add_argument(
        "--tmp-dir", default=None,
        help="Directory for temporary sort files (default: system temp). "
             "Use a path with enough free space (~26 bytes × total game-ply observations).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    file_paths = list_corpus_files(args.corpus)
    n_files = len(file_paths)
    print(f"Found {n_files} file(s).")

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Pass 1: replay each file once, write a sorted binary chunk
        print("Pass 1: writing sorted chunk files...")
        temp_paths = []
        total_rows = 0
        for file_idx, file_path in enumerate(file_paths):
            games = load_file(file_path)
            temp_file = tmp_path / f"chunk_{file_idx:05d}.bin"
            n_rows = write_sorted_file(
                games, file_idx, temp_file,
                min_ply=args.min_ply, max_ply=args.max_ply,
            )
            temp_paths.append(temp_file)
            total_rows += n_rows
            print(
                f"  [{file_idx + 1}/{n_files}] {file_path.name}: "
                f"{len(games)} games, {n_rows} rows",
                end="\r",
            )
        print(f"\n  {total_rows:,} total rows written ({total_rows * 26 / 1e9:.1f} GB).")

        # Merge: stream-merge all sorted files to find candidates
        print("Merging sorted files...")
        all_candidates = merge_sorted_files(temp_paths)
        print(f"  {len(all_candidates):,} candidate board states with 2+ game references.")

    # Determine which (file_idx, game_idx) pairs need to be reloaded
    needed: dict[int, set[int]] = defaultdict(set)
    for refs in all_candidates.values():
        for file_idx, game_idx in refs:
            needed[file_idx].add(game_idx)

    # Pass 2: load only the games that appear in candidate groups
    print("Pass 2: loading sequences for candidate games...")
    game_lookup: dict[tuple[int, int], list[str]] = {}
    for file_idx, game_idxs in sorted(needed.items()):
        games = load_file(file_paths[file_idx])
        for game_idx in game_idxs:
            if game_idx < len(games):
                game_lookup[(file_idx, game_idx)] = games[game_idx]
        print(
            f"  [{file_idx + 1}/{n_files}] {len(game_idxs)} games "
            f"from {file_paths[file_idx].name}",
            end="\r",
        )
    print()

    # Build groups
    print("Building transposition groups...")
    groups = build_groups_from_compact(all_candidates, game_lookup)
    del game_lookup, all_candidates, needed

    # Split groups into train/test
    rng = random.Random(args.seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * args.test_fraction))
    test_groups  = shuffled[:n_test]
    train_groups = shuffled[n_test:]
    print(f"  Train: {len(train_groups)}  |  Test: {len(test_groups)}")

    # Summary
    summary = summarise(train_groups)
    if summary:
        print(f"  Found {summary['total_groups']} transposition groups")
        print(f"  Ply range         : {summary['ply_min']}–{summary['ply_max']}")
        print(f"  Group size        : min={summary['group_size_min']}, "
              f"max={summary['group_size_max']}, avg={summary['group_size_avg']:.1f}")
        print(f"  Same trichrome    : {summary['same_trichrome']} / {summary['total_groups']}")
        print(f"  Mixed trichrome   : {summary['mixed_trichrome']} / {summary['total_groups']}")
        print(f"  Color distances   : {summary['color_dist_counts']}")

        mixed = [group for group in train_groups if group.has_trichrome_diffs]
        if mixed:
            print("\nSample mixed trichrome groups (first 3):")
            for group in mixed[:3]:
                print(f"  ply={group.ply}, {group.n_sequences} sequences, "
                      f"{group.n_trichrome_states} trichrome states:")
                for diff in group.trichrome_diffs[:1]:
                    for cell in diff.differing_cells[:3]:
                        names = ["Red", "Green", "Blue"]
                        print(f"    {cell['square']}: "
                              f"{names[cell['color_a']]} vs {names[cell['color_b']]}")
    else:
        print("  No transposition groups found.")

    # Save train and test separately
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_path = output_path.with_stem(output_path.stem + "_test")

    with open(output_path, "w") as f:
        json.dump([serialize_group(group) for group in train_groups], f, indent=2)
    print(f"\nSaved {len(train_groups)} train groups to '{output_path}'")

    with open(test_path, "w") as f:
        json.dump([serialize_group(group) for group in test_groups], f, indent=2)
    print(f"Saved {len(test_groups)} test groups to '{test_path}'")


if __name__ == "__main__":
    main()
