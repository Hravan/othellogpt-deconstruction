"""
scripts/negation_prepare.py

Split ss_pairs.json into train and held-out sets for the negation depth
fine-tuning experiment. Saves held_out_pairs.json to the output directory.

For train depths: groups 80-99 are held out (new capital facts, same phrasings).
For test depths: all 100 groups are used (none were trained on).
No overlap: train groups are always indices 0-79 of train-depth categories.

Usage
-----
    python scripts/negation_prepare.py --train-depths 0,1,2 --output-dir ckpts/negation_012
    python scripts/negation_prepare.py --train-depths 0,1,2,3 --output-dir ckpts/negation_0123
    python scripts/negation_prepare.py --train-depths 0,1,2,3,4 --output-dir ckpts/negation_01234
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare held-out pairs for the negation depth fine-tuning experiment."
    )
    parser.add_argument("--pairs", default="data/ss_pairs.json",
                        help="ss_pairs.json path (default: data/ss_pairs.json)")
    parser.add_argument("--train-depths", default="0,1,2",
                        help="Comma-separated depth indices to train on (default: 0,1,2)")
    parser.add_argument("--train-size", type=int, default=80,
                        help="Groups per depth used for training; rest held out (default: 80)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save held_out_pairs.json")
    args = parser.parse_args()

    train_depths = [int(depth_str) for depth_str in args.train_depths.split(",")]

    with open(args.pairs, encoding="utf-8") as pairs_file:
        all_groups = json.load(pairs_file)

    held_out_groups: list[dict] = []
    for depth in range(7):
        depth_groups = [g for g in all_groups if g["category"] == f"negation_depth_{depth}"]
        if depth in train_depths:
            train_groups  = depth_groups[:args.train_size]
            held_out_part = depth_groups[args.train_size:]
            print(f"  negation_depth_{depth}: {len(train_groups)} train, {len(held_out_part)} eval  [TRAIN]")
        else:
            train_groups  = []
            held_out_part = depth_groups  # all 100 — none were trained on
            print(f"  negation_depth_{depth}: {len(held_out_part)} eval  [TEST]")
        held_out_groups.extend(held_out_part)

    parity_groups = [
        g for g in all_groups if g["category"] in ("negation_even", "negation_odd")
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    held_out_path = output_dir / "held_out_pairs.json"
    with open(held_out_path, "w", encoding="utf-8") as output_file:
        json.dump(held_out_groups, output_file, indent=2, ensure_ascii=False)

    parity_path = output_dir / "parity_pairs.json"
    with open(parity_path, "w", encoding="utf-8") as output_file:
        json.dump(parity_groups, output_file, indent=2, ensure_ascii=False)

    eval_path = output_dir / "eval_pairs.json"
    with open(eval_path, "w", encoding="utf-8") as output_file:
        json.dump(held_out_groups + parity_groups, output_file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(held_out_groups)} held-out groups  → {held_out_path}")
    print(f"Saved {len(parity_groups)} parity groups     → {parity_path}")
    print(f"Saved {len(held_out_groups) + len(parity_groups)} total eval groups → {eval_path}")


if __name__ == "__main__":
    main()
