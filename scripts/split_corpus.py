"""
scripts/split_corpus.py

Split a corpus into train and test game sets and save them as JSON.

Downstream scripts (train_board_probes.py, nanda_intervention.py) load the
pre-split files directly, ensuring no overlap between probe training data and
evaluation data.

Usage
-----
    uv run python scripts/split_corpus.py \\
        --corpus data/sequence_data/othello_championship \\
        --train-output data/games_train.json \\
        --test-output data/games_test.json
"""

import argparse
import json
import random
from pathlib import Path

from othellogpt_deconstruction.core.corpus import list_corpus_files, load_file


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
        "--seed", type=int, default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading corpus from {args.corpus}...")
    file_paths = list_corpus_files(args.corpus)
    all_games: list[list[str]] = []
    for file_path in file_paths:
        all_games.extend(load_file(file_path))
    print(f"  {len(all_games)} games loaded from {len(file_paths)} file(s).")

    rng = random.Random(args.seed)
    rng.shuffle(all_games)

    n_test  = max(1, int(len(all_games) * args.test_fraction))
    n_train = len(all_games) - n_test
    train_games = all_games[:n_train]
    test_games  = all_games[n_train:]
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
