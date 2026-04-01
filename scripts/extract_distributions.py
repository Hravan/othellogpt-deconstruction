"""
scripts/extract_distributions.py

Run OthelloGPT on every unique sequence from trichrome-differing
transposition pairs and save the softmax distributions to disk.

Run this once — it is the only script that requires the model and GPU.
All subsequent analysis loads from the saved distributions.

Usage
-----
    uv run python scripts/extract_distributions.py \\
        --transpositions data/transpositions.json \\
        --mode championship \\
        --output data/distributions.pt

    # With explicit checkpoint path
    uv run python scripts/extract_distributions.py \\
        --transpositions data/transpositions.json \\
        --checkpoint ckpts/gpt_championship.ckpt \\
        --output data/distributions.pt
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from othellogpt_deconstruction.core.tokenizer import seq_key
from othellogpt_deconstruction.model.inference import load_model, get_distributions_batch


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def collect_sequences(groups: list[dict]) -> list[list[str]]:
    """
    Extract all unique sequences that appear in trichrome-differing pairs.
    Only sequences from groups that have trichrome diffs are included.
    """
    seen: set[str] = set()
    sequences: list[list[str]] = []

    for group in groups:
        if not group.get("trichrome_diffs"):
            continue
        t_groups = group["trichrome_groups"]
        for diff in group["trichrome_diffs"]:
            for idx in (diff["group_a"], diff["group_b"]):
                for seq in t_groups[idx]["sequences"]:
                    k = seq_key(seq)
                    if k not in seen:
                        seen.add(k)
                        sequences.append(seq)

    return sequences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OthelloGPT distributions for transposition sequences."
    )
    parser.add_argument(
        "--transpositions", default="data/transpositions.json",
        help="Path to transpositions JSON (default: data/transpositions.json)",
    )
    parser.add_argument(
        "--mode", choices=["championship", "synthetic", "random"],
        default="championship",
        help="Model checkpoint to use (default: championship)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Explicit checkpoint path (overrides --mode)",
    )
    parser.add_argument(
        "--output", default="data/distributions.pt",
        help="Output path for distributions (default: data/distributions.pt)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for inference (default: 64)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load transpositions
    print(f"Loading transpositions from '{args.transpositions}' ...")
    with open(args.transpositions) as f:
        groups = json.load(f)

    mixed = [g for g in groups if g.get("trichrome_diffs")]
    print(f"  {len(groups)} total groups, {len(mixed)} with trichrome diffs")

    # Collect unique sequences
    sequences = collect_sequences(groups)
    print(f"  Unique sequences to process: {len(sequences)}")

    if not sequences:
        print("Nothing to process.")
        return

    # Load model
    print(f"Loading model (mode={args.mode}) ...")
    model = load_model(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Extract distributions in batches
    print("Extracting distributions ...")
    distributions: dict[str, torch.Tensor] = {}

    for i in tqdm(range(0, len(sequences), args.batch_size), desc="Batches"):
        batch = sequences[i:i + args.batch_size]
        batch_results = get_distributions_batch(
            model, batch, device=device, batch_size=args.batch_size,
        )
        distributions.update(batch_results)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(distributions, output_path)
    print(f"\nSaved {len(distributions)} distributions to '{output_path}'")


if __name__ == "__main__":
    main()
