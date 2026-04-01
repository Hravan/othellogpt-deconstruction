"""
scripts/split_corpus.py

Split a corpus into train and test game sets and save them as JSON.

Downstream scripts (train_board_probes.py, nanda_intervention.py) load the
pre-split files directly, ensuring no overlap between probe training data and
evaluation data.

Streams through corpus files one at a time so it never loads the full corpus
into memory. Use --max-games to cap the total number of games sampled (required
for large corpora like othello_synthetic).

Usage
-----
    uv run python scripts/split_corpus.py \\
        --corpus data/sequence_data/othello_championship \\
        --train-output data/games_train.json \\
        --test-output data/games_test.json

    uv run python scripts/split_corpus.py \\
        --corpus data/sequence_data/othello_synthetic \\
        --train-output data/games_synthetic_train.json \\
        --test-output data/games_synthetic_test.json \\
        --max-games 250000
"""

import argparse
import json
import random
from pathlib import Path

from othellogpt_deconstruction.core.corpus import iter_corpus_files, list_corpus_files


def reservoir_sample(
    paths: list[str],
    max_games: int | None,
    seed: int,
) -> list[list[str]]:
    """
    Stream through all corpus files and reservoir-sample up to max_games games.

    If max_games is None, loads everything (only safe for small corpora).
    Uses Vitter's Algorithm R so peak memory is O(max_games), not O(corpus).
    """
    rng = random.Random(seed)
    reservoir: list[list[str]] = []
    total_seen = 0

    for file_games in iter_corpus_files(paths):
        for game in file_games:
            total_seen += 1
            if max_games is None or len(reservoir) < max_games:
                reservoir.append(game)
            else:
                replace_index = rng.randrange(total_seen)
                if replace_index < max_games:
                    reservoir[replace_index] = game

        print(f"  {total_seen} games seen, {len(reservoir)} in reservoir", end="\r")

    print()
    return reservoir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split corpus into train and test JSON files."
    )
    parser.add_argument(
        "--corpus", nargs="+", required=True,
        help="Path(s) to corpus file(s) or directory(s)",
    )
    parser.add_argument(
        "--train-output", default="data/games_train.json",
        help="Output path for training games (default: data/games_train.json)",
    )
    parser.add_argument(
        "--test-output", default="data/games_test.json",
        help="Output path for test games (default: data/games_test.json)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.1,
        help="Fraction of games to hold out as test set (default: 0.1)",
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Maximum number of games to sample from the corpus. "
             "Required for large corpora (e.g. othello_synthetic). "
             "Uses reservoir sampling so peak memory is O(max_games).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading corpus from {args.corpus}...")
    files = list_corpus_files(args.corpus)
    print(f"  {len(files)} file(s) found.")

    sampled_games = reservoir_sample(args.corpus, args.max_games, args.seed)
    print(f"  {len(sampled_games)} games sampled.")

    rng = random.Random(args.seed)
    rng.shuffle(sampled_games)

    n_test  = max(1, int(len(sampled_games) * args.test_fraction))
    n_train = len(sampled_games) - n_test
    train_games = sampled_games[:n_train]
    test_games  = sampled_games[n_train:]
    print(f"  Train: {len(train_games)}  |  Test: {len(test_games)}")

    train_path = Path(args.train_output)
    test_path  = Path(args.test_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as train_file:
        json.dump(train_games, train_file)
    print(f"  Saved {len(train_games)} train games to '{train_path}'")

    with open(test_path, "w") as test_file:
        json.dump(test_games, test_file)
    print(f"  Saved {len(test_games)} test games to '{test_path}'")


if __name__ == "__main__":
    main()
