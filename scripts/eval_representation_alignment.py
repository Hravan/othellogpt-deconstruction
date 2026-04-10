"""
scripts/eval_representation_alignment.py

Compute direct test-pair cosine similarity from MUSE-aligned embeddings,
plus a shuffled-pairs baseline to establish the noise floor.

Usage
-----
    python scripts/eval_representation_alignment.py \
        --aligned-src muse/dumped/debug/muse_run/vectors-othello_gpt.txt \
        --aligned-tgt muse/dumped/debug/muse_run/vectors-board_predictor.txt \
        --dict-test data/muse/dict_test.txt
"""

import argparse
import random
from pathlib import Path

import numpy as np


def load_muse_vecs(path: Path) -> dict[str, np.ndarray]:
    vecs = {}
    with open(path) as f:
        next(f)  # skip header line (N D)
        for line in f:
            parts = line.split()
            vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return vecs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MUSE alignment via direct test-pair cosine similarity."
    )
    parser.add_argument(
        "--aligned-src",
        default="muse/dumped/debug/muse_run/vectors-othello_gpt.txt",
        help="Aligned source embeddings (default: muse/dumped/debug/muse_run/vectors-othello_gpt.txt)",
    )
    parser.add_argument(
        "--aligned-tgt",
        default="muse/dumped/debug/muse_run/vectors-board_predictor.txt",
        help="Aligned target embeddings (default: muse/dumped/debug/muse_run/vectors-board_predictor.txt)",
    )
    parser.add_argument(
        "--dict-test",
        default="data/muse/dict_test.txt",
        help="Test dictionary of aligned pairs (default: data/muse/dict_test.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffled-pairs baseline (default: 42)",
    )
    return parser.parse_args()


def compute_cosines(
    source_vecs: dict[str, np.ndarray],
    target_vecs: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
) -> tuple[list[float], int]:
    cosines = []
    missing = 0
    for source_id, target_id in pairs:
        if source_id not in source_vecs or target_id not in target_vecs:
            missing += 1
            continue
        source_vec = source_vecs[source_id]
        target_vec = target_vecs[target_id]
        source_vec = source_vec / np.linalg.norm(source_vec)
        target_vec = target_vec / np.linalg.norm(target_vec)
        cosines.append(float(np.dot(source_vec, target_vec)))
    return cosines, missing


def print_cosine_stats(label: str, cosines: list[float]) -> None:
    print(f"\n{label}  (n={len(cosines)})")
    print(f"  mean   = {np.mean(cosines):.4f}")
    print(f"  median = {np.median(cosines):.4f}")
    print(f"  std    = {np.std(cosines):.4f}")
    print(f"  min    = {np.min(cosines):.4f}")
    print(f"  max    = {np.max(cosines):.4f}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading aligned source embeddings from {args.aligned_src}...")
    aligned_src = load_muse_vecs(Path(args.aligned_src))

    print(f"Loading aligned target embeddings from {args.aligned_tgt}...")
    aligned_tgt = load_muse_vecs(Path(args.aligned_tgt))

    print(f"Loading test pairs from {args.dict_test}...")
    with open(args.dict_test) as f:
        test_pairs = [tuple(line.split()) for line in f if line.strip()]

    # Correct pairs
    print(f"Computing cosine similarity over {len(test_pairs)} test pairs...")
    cosines, missing = compute_cosines(aligned_src, aligned_tgt, test_pairs)
    if missing:
        print(f"  (skipped {missing} pairs with missing embeddings)")
    print_cosine_stats("Aligned test-pair cosine similarity", cosines)

    # Shuffled-pairs baseline: same source IDs, randomly permuted target IDs
    source_ids = [pair[0] for pair in test_pairs]
    target_ids = [pair[1] for pair in test_pairs]
    shuffled_target_ids = target_ids[:]
    random.shuffle(shuffled_target_ids)
    shuffled_pairs = list(zip(source_ids, shuffled_target_ids))

    shuffled_cosines, _ = compute_cosines(aligned_src, aligned_tgt, shuffled_pairs)
    print_cosine_stats("Shuffled-pairs baseline (noise floor)", shuffled_cosines)


if __name__ == "__main__":
    main()
