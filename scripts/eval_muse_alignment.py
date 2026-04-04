"""
scripts/eval_muse_alignment.py

Compute direct test-pair cosine similarity from MUSE-aligned embeddings.

Usage
-----
    python scripts/eval_muse_alignment.py \
        --aligned-src muse/dumped/debug/muse_run/vectors-othello_gpt.txt \
        --aligned-tgt muse/dumped/debug/muse_run/vectors-board_predictor.txt \
        --dict-test data/muse/dict_test.txt
"""

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading aligned source embeddings from {args.aligned_src}...")
    aligned_src = load_muse_vecs(Path(args.aligned_src))

    print(f"Loading aligned target embeddings from {args.aligned_tgt}...")
    aligned_tgt = load_muse_vecs(Path(args.aligned_tgt))

    print(f"Loading test pairs from {args.dict_test}...")
    with open(args.dict_test) as f:
        test_pairs = [line.split() for line in f if line.strip()]

    print(f"Computing cosine similarity over {len(test_pairs)} test pairs...")
    cosines = []
    missing = 0
    for source_id, target_id in test_pairs:
        if source_id not in aligned_src or target_id not in aligned_tgt:
            missing += 1
            continue
        source_vec = aligned_src[source_id]
        target_vec = aligned_tgt[target_id]
        source_vec = source_vec / np.linalg.norm(source_vec)
        target_vec = target_vec / np.linalg.norm(target_vec)
        cosines.append(np.dot(source_vec, target_vec))

    if missing:
        print(f"  (skipped {missing} pairs with missing embeddings)")

    print()
    print(f"Test-pair cosine similarity  (n={len(cosines)})")
    print(f"  mean   = {np.mean(cosines):.4f}")
    print(f"  median = {np.median(cosines):.4f}")
    print(f"  std    = {np.std(cosines):.4f}")
    print(f"  min    = {np.min(cosines):.4f}")
    print(f"  max    = {np.max(cosines):.4f}")


if __name__ == "__main__":
    main()
