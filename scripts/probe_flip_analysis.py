"""
scripts/probe_flip_analysis.py

Per-layer probe accuracy split by flip history.

For each test position we collect residual stream activations at every layer,
decode the board state with the trained probe, and compute accuracy separately
for cells that have been flipped at least once vs cells that have never been
flipped (i.e. whose current owner is whoever placed them).

Hypothesis: if probe accuracy on flipped cells is already high at layer 0
(where the residual stream has only one pass of self-attention over the raw
sequence), then the probe is exploiting input-level sequence context, not any
computation the model performed.  If flip accuracy grows substantially across
layers, the model's layers are contributing.

Never-flipped cells are predictable from ply parity alone and serve as a
baseline that should be high across all layers.

Usage
-----
    uv run python scripts/probe_flip_analysis.py \\
        --games data/games_test.json \\
        --probes data/board_probes_all_layers.pt \\
        --mode championship
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from othellogpt_deconstruction.core.board import (
    start_board, apply_move, flipped_by, EMPTY as BOARD_EMPTY,
)
from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, itos, BLOCK_SIZE, PAD_ID
from othellogpt_deconstruction.intervention.hooks import collect_activations
from othellogpt_deconstruction.model.board_probe import (
    make_labels, decode_board, EMPTY, MINE, YOURS,
)
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.model.probes import load_probes
from othellogpt_deconstruction.model.utils import encode_sequence


# ---------------------------------------------------------------------------
# Replay with flip tracking
# ---------------------------------------------------------------------------

def replay_with_flip_mask(moves: list[str]) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Replay a sequence and return (board, next_player, ever_flipped_mask).

    ever_flipped_mask : boolean (64,) array, True for cells that changed owner
                        at least once during the game.  Cells that are occupied
                        but never flipped are predictable from ply parity.
    """
    from othellogpt_deconstruction.core.tokenizer import alg_to_pos
    from othellogpt_deconstruction.core.board import BLACK

    board = start_board()
    player = BLACK
    ever_flipped = np.zeros(64, dtype=bool)

    for move in moves:
        pos = alg_to_pos(move)
        flips = flipped_by(board, pos, player)
        for fp in flips:
            ever_flipped[fp] = True
        board = apply_move(board, pos, player)
        player = 3 - player
        from othellogpt_deconstruction.core.board import legal_moves
        if not legal_moves(board, player):
            player = 3 - player

    return board, player, ever_flipped


# ---------------------------------------------------------------------------
# Per-position analysis
# ---------------------------------------------------------------------------

def analyse_position(
    model:    torch.nn.Module,
    probes:   dict,
    sequence: list[str],
    device:   torch.device,
) -> dict | None:
    """
    For one position return per-layer accuracy on:
      - all occupied cells
      - never-flipped occupied cells
      - flipped-at-least-once occupied cells

    Returns None if unusable.
    """
    try:
        board, next_player, ever_flipped = replay_with_flip_mask(sequence)
    except ValueError:
        return None

    labels = make_labels(board, next_player)
    occupied_mask = labels != EMPTY

    if occupied_mask.sum() == 0:
        return None

    never_flipped_occupied = occupied_mask & ~ever_flipped
    flipped_occupied       = occupied_mask & ever_flipped

    x = encode_sequence(sequence, device)
    with collect_activations(model, len(sequence)) as store:
        with torch.no_grad():
            model(x)

    result = {}
    for layer_idx, activation in store.activations.items():
        if layer_idx not in probes:
            continue
        probe = probes[layer_idx]
        decoded = decode_board(probe, activation.cpu().float())

        correct = decoded == labels

        acc_all = float(correct[occupied_mask].mean()) if occupied_mask.any() else None
        acc_never = float(correct[never_flipped_occupied].mean()) if never_flipped_occupied.any() else None
        acc_flipped = float(correct[flipped_occupied].mean()) if flipped_occupied.any() else None

        result[layer_idx] = {
            "acc_all":            acc_all,
            "acc_never_flipped":  acc_never,
            "acc_flipped":        acc_flipped,
            "n_occupied":         int(occupied_mask.sum()),
            "n_never_flipped":    int(never_flipped_occupied.sum()),
            "n_flipped":          int(flipped_occupied.sum()),
        }

    return result if result else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(all_results: list[dict], layer_indices: list[int]) -> None:
    print(f"\n{'layer':>6}  {'n':>6}  {'acc_all':>9}  "
          f"{'acc_never_flipped':>18}  {'acc_flipped':>12}  {'gap':>6}")
    print("-" * 70)

    for layer_idx in layer_indices:
        entries = [r[layer_idx] for r in all_results if layer_idx in r]
        if not entries:
            continue

        n = len(entries)
        acc_all     = np.mean([e["acc_all"]           for e in entries if e["acc_all"]           is not None])
        acc_never   = np.mean([e["acc_never_flipped"] for e in entries if e["acc_never_flipped"] is not None])
        acc_flipped = np.mean([e["acc_flipped"]       for e in entries if e["acc_flipped"]       is not None])
        gap = acc_never - acc_flipped

        print(f"{layer_idx:>6}  {n:>6}  {acc_all:>9.3f}  "
              f"{acc_never:>18.3f}  {acc_flipped:>12.3f}  {gap:>+6.3f}")

    print()
    print("gap = acc_never_flipped - acc_flipped")
    print("  large gap across all layers → probe mainly exploits parity, not model computation")
    print("  gap shrinks with depth      → model's layers progressively compute flip information")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-layer probe accuracy split by flip history."
    )
    parser.add_argument(
        "--games", required=True,
        help="Path to test games JSON (output of split_corpus.py)",
    )
    parser.add_argument(
        "--probes", required=True,
        help="Path to trained board probes (output of train_board_probes.py)",
    )
    parser.add_argument(
        "--mode", default="championship",
        choices=["championship", "synthetic", "random"],
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Explicit checkpoint path (overrides --mode)",
    )
    parser.add_argument(
        "--n-games", type=int, default=None,
        help="Number of games to evaluate (default: all)",
    )
    parser.add_argument(
        "--n-timesteps-per-game", type=int, default=2,
        help="Positions sampled per game (default: 2)",
    )
    parser.add_argument(
        "--min-ply", type=int, default=10,
        help="Minimum ply to sample (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model ({args.mode})...")
    model = load_model(args.mode, args.checkpoint)
    device = next(model.parameters()).device
    print(f"  Device: {device}")

    print(f"Loading probes from {args.probes}...")
    probes = load_probes(args.probes)
    print(f"  Probes at layers: {sorted(probes.keys())}")

    print(f"Loading games from {args.games}...")
    with open(args.games) as games_file:
        all_games: list[list[str]] = json.load(games_file)
    print(f"  {len(all_games)} games loaded.")

    rng = random.Random(args.seed)
    if args.n_games is not None and len(all_games) > args.n_games:
        sampled_games = rng.sample(all_games, args.n_games)
        print(f"  Sampled {args.n_games} games.")
    else:
        sampled_games = all_games

    positions: list[list[str]] = []
    for game in sampled_games:
        available_plies = list(range(args.min_ply, min(len(game), BLOCK_SIZE) + 1))
        if not available_plies:
            continue
        sampled_plies = rng.sample(
            available_plies,
            min(args.n_timesteps_per_game, len(available_plies)),
        )
        for ply in sampled_plies:
            positions.append(game[:ply])

    print(f"  {len(positions)} positions to analyse.")

    all_results: list[dict] = []
    for position_idx, sequence in enumerate(positions):
        print(f"  Position {position_idx + 1}/{len(positions)}", end="\r")
        result = analyse_position(model, probes, sequence, device)
        if result is not None:
            all_results.append(result)
    print()

    print(f"\n{len(all_results)} positions analysed.")
    layer_indices = sorted(probes.keys())
    report(all_results, layer_indices)


if __name__ == "__main__":
    main()
