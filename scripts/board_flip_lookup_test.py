"""
scripts/board_flip_lookup_test.py

Lookup table test for Nanda et al.'s world model claim.

For each test position we apply a Nanda-style intervention: add the board state
probe direction for flipping N cells (Mine↔Yours) to the residual stream at a
given layer, creating an illegal board state that no legal Othello sequence
reaches.  We then compare two predictions:

  Lookup table prediction
    Decode the intervened residual stream with the probe → compute legal moves
    for the decoded board.  This is the full mechanism if the model is only
    exploiting the learned (board_encoding → legal_moves) correlation.

  Model prediction
    Run the remaining transformer layers on the patched residual stream and
    observe the actual output distribution.

If the model generalises beyond the lookup table — predicting legal moves for
the target board state even when the probe fails to decode that state — that
is evidence for genuine world-model-like generalisation.  If model accuracy
tracks probe decode accuracy throughout, the mechanism is fully explained by
learned correlation.

We vary N (number of cells flipped) from 1 to 5 as a proxy for "distance from
training distribution": more flips → more illegal → further from anything seen
during training.

Usage
-----
    # Train board probes first:
    uv run python scripts/train_board_probes.py \\
        --corpus data/sequence_data/othello_championship \\
        --output data/board_probes.pt

    # Then run this:
    uv run python scripts/board_flip_lookup_test.py \\
        --corpus data/sequence_data/othello_championship \\
        --probes data/board_probes.pt \\
        --layer 5
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from othellogpt_deconstruction.core.board import (
    replay as board_replay, legal_moves, EMPTY as BOARD_EMPTY,
)
from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, itos, BLOCK_SIZE, PAD_ID
from othellogpt_deconstruction.intervention.hooks import collect_activations, patch_activations
from othellogpt_deconstruction.model.board_probe import (
    make_labels, labels_to_absolute_board, decode_board, EMPTY, MINE, YOURS,
)
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.model.probes import load_probes
from othellogpt_deconstruction.model.utils import encode_sequence, forward_pass, top1_position


# ---------------------------------------------------------------------------
# Per-position experiment
# ---------------------------------------------------------------------------

def analyse_position(
    model:       torch.nn.Module,
    probe:       object,
    sequence:    list[str],
    layer:       int,
    n_flips_list: list[int],
    device:      torch.device,
    rng:         random.Random,
) -> dict | None:
    """
    Run baseline measurement and all flip-count interventions for one position.

    Returns a dict keyed by n_flips (0 = baseline), each containing:
        probe_cell_acc     : fraction of cells where probe predicts correctly
        model_target_legal : whether model top-1 is legal for target board
        model_source_legal : whether model top-1 is legal for source board
        both_correct       : probe_cell_acc > THRESHOLD and model_target_legal
        model_only         : model_target_legal but probe_cell_acc <= THRESHOLD
        probe_only         : probe_cell_acc > THRESHOLD but not model_target_legal
        neither            : neither correct

    Returns None if the position is unusable (replay fails, no legal moves).
    """
    PROBE_CORRECT_THRESHOLD = 0.90

    try:
        board, next_player = board_replay(sequence)
    except ValueError:
        return None

    source_legal_set = set(legal_moves(board, next_player))
    if not source_legal_set:
        return None

    source_labels = make_labels(board, next_player)
    flippable_positions = [
        pos for pos in range(64)
        if source_labels[pos] != EMPTY
    ]
    if len(flippable_positions) < max(n_flips_list):
        return None

    x = encode_sequence(sequence, device)
    with collect_activations(model, len(sequence)) as store:
        probs_original = forward_pass(model, x, len(sequence))

    activation = store.get(layer)
    if activation is None:
        return None

    results = {}

    # --- Baseline (n_flips=0): probe accuracy on actual board ---
    # Restrict to non-empty cells: probe is trained only on MINE/YOURS and never
    # predicts EMPTY, so including empty cells collapses apparent accuracy.
    decoded_labels = decode_board(probe, activation)
    occupied_mask = source_labels != EMPTY
    probe_baseline_acc = float((decoded_labels[occupied_mask] == source_labels[occupied_mask]).mean()) if occupied_mask.any() else 0.0
    model_source_legal = top1_position(probs_original) in source_legal_set

    results[0] = {
        "probe_cell_acc":     probe_baseline_acc,
        "model_target_legal": model_source_legal,
        "model_source_legal": model_source_legal,
        "both_correct":       probe_baseline_acc > PROBE_CORRECT_THRESHOLD and model_source_legal,
        "model_only":         model_source_legal and probe_baseline_acc <= PROBE_CORRECT_THRESHOLD,
        "probe_only":         not model_source_legal and probe_baseline_acc > PROBE_CORRECT_THRESHOLD,
        "neither":            not model_source_legal and probe_baseline_acc <= PROBE_CORRECT_THRESHOLD,
    }

    # --- Interventions ---
    for n_flips in n_flips_list:
        cells_to_flip = rng.sample(flippable_positions, n_flips)

        # Build target labels (Mine↔Yours for selected cells)
        target_labels = source_labels.copy()
        for pos in cells_to_flip:
            target_labels[pos] = YOURS if source_labels[pos] == MINE else MINE

        # Target board and its legal moves
        target_board = labels_to_absolute_board(target_labels, next_player)
        target_legal_set = set(legal_moves(target_board, next_player))

        # Intervention vector: sum of cell_direction for each flipped cell
        cell_diffs = [
            {
                "pos":     pos,
                "color_a": int(source_labels[pos]),
                "color_b": int(target_labels[pos]),
            }
            for pos in cells_to_flip
        ]
        intervention_vector = probe.trichrome_direction(cell_diffs).to(device)

        # Apply intervention to model
        with patch_activations(model, len(sequence), {layer: intervention_vector}):
            probs_intervened = forward_pass(model, x, len(sequence))

        # Probe on intervened activation (act + intervention, computed directly)
        # Restrict to non-empty cells in the target board.
        activation_intervened = (activation + intervention_vector.cpu()).float()
        decoded_target = decode_board(probe, activation_intervened)
        target_occupied_mask = target_labels != EMPTY
        probe_target_acc = float((decoded_target[target_occupied_mask] == target_labels[target_occupied_mask]).mean()) if target_occupied_mask.any() else 0.0

        model_target_legal = top1_position(probs_intervened) in target_legal_set
        model_source_legal_after = top1_position(probs_intervened) in source_legal_set

        probe_correct  = probe_target_acc > PROBE_CORRECT_THRESHOLD
        results[n_flips] = {
            "probe_cell_acc":     probe_target_acc,
            "model_target_legal": model_target_legal,
            "model_source_legal": model_source_legal_after,
            "both_correct":       probe_correct and model_target_legal,
            "model_only":         model_target_legal and not probe_correct,
            "probe_only":         not model_target_legal and probe_correct,
            "neither":            not model_target_legal and not probe_correct,
        }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(all_results: list[dict], n_flips_list: list[int]) -> None:
    print(f"\n{'n_flips':>8}  {'n':>6}  {'probe_acc':>10}  "
          f"{'model→target':>13}  {'both_ok':>8}  "
          f"{'model_only':>11}  {'probe_only':>11}  {'neither':>8}")
    print("-" * 90)

    for n_flips in [0] + n_flips_list:
        entries = [r[n_flips] for r in all_results if n_flips in r]
        if not entries:
            continue
        n = len(entries)
        probe_acc     = np.mean([e["probe_cell_acc"]     for e in entries])
        target_legal  = np.mean([e["model_target_legal"] for e in entries])
        both_correct  = np.mean([e["both_correct"]       for e in entries])
        model_only    = np.mean([e["model_only"]         for e in entries])
        probe_only    = np.mean([e["probe_only"]         for e in entries])
        neither       = np.mean([e["neither"]            for e in entries])

        label = "baseline" if n_flips == 0 else str(n_flips)
        print(f"{label:>8}  {n:>6}  {probe_acc:>10.3f}  "
              f"{target_legal:>13.3f}  {both_correct:>8.3f}  "
              f"{model_only:>11.3f}  {probe_only:>11.3f}  {neither:>8.3f}")

    print()
    print("Key columns:")
    print("  model→target : fraction of positions where model top-1 is legal for the target board")
    print("  model_only   : model correct but probe failed — evidence for generalisation beyond lookup")
    print("  probe_only   : probe correct but model wrong — evidence for lookup table failure")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lookup table test for Nanda-style board state interventions."
    )
    parser.add_argument(
        "--games", required=True,
        help="Path to pre-split test games JSON (output of split_corpus.py)",
    )
    parser.add_argument(
        "--probes", required=True,
        help="Path to trained board state probes (output of train_board_probes.py)",
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
        "--layer", type=int, default=5,
        help="Residual stream layer to intervene on (default: 5)",
    )
    parser.add_argument(
        "--n-flips", default="1,2,3,5",
        help="Comma-separated flip counts to test (default: 1,2,3,5)",
    )
    parser.add_argument(
        "--n-games", type=int, default=None,
        help="Number of games to test on (default: all)",
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
    n_flips_list = [int(n) for n in args.n_flips.split(",")]

    print(f"Loading model ({args.mode})...")
    model = load_model(args.mode, args.checkpoint)
    device = next(model.parameters()).device
    print(f"  Device: {device}")

    print(f"Loading probes from {args.probes}...")
    probes = load_probes(args.probes)
    if args.layer not in probes:
        raise ValueError(
            f"Layer {args.layer} not in probes. Available: {sorted(probes.keys())}"
        )
    probe = probes[args.layer]
    print(f"  Using probe at layer {args.layer}.")

    print(f"Loading games from {args.games}...")
    import json
    with open(args.games) as games_file:
        all_games: list[list[str]] = json.load(games_file)
    print(f"  {len(all_games)} games loaded.")

    rng = random.Random(args.seed)
    if args.n_games is not None and len(all_games) > args.n_games:
        sampled_games = rng.sample(all_games, args.n_games)
        print(f"  Sampled {args.n_games} games.")
    else:
        sampled_games = all_games

    # Sample positions from games
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

    print(f"  {len(positions)} positions to test.")
    print(f"  Flip counts: {n_flips_list}")

    all_results: list[dict] = []
    for position_idx, sequence in enumerate(positions):
        print(f"  Position {position_idx + 1}/{len(positions)}", end="\r")
        result = analyse_position(
            model, probe, sequence, args.layer, n_flips_list, device, rng
        )
        if result is not None:
            all_results.append(result)
    print()

    print(f"\n{len(all_results)} positions analysed.")
    report(all_results, n_flips_list)


if __name__ == "__main__":
    main()
