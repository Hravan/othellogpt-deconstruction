"""
scripts/nanda_experiment.py

Two combined experiments attacking Nanda et al.'s illegal-state intervention claim.

Experiment 1 — Legal-set overlap (2×2 table)
---------------------------------------------
Nanda et al. measure top-N predictions post-intervention against legal_moves(B'),
where N = |legal_moves(B')|. They report 0.10 average errors and claim this shows
the model predicts legal moves for the new (illegal) board state.

Missing comparison: how do top-N predictions from the *original* (unintervened) model
score against legal_moves(B')? If the legal sets of B and B' overlap heavily, the
original model's predictions are already nearly correct for B' — the intervention
adds little.

Full 2×2 table reported per n_flips:
    original model   vs legal(B)   → baseline (should be ~0)
    original model   vs legal(B')  → null baseline (Nanda's "null intervention")
    intervened model vs legal(B')  → Nanda's claim
    intervened model vs legal(B)   → key missing number

Also reports legal set overlap = |legal(B) ∩ legal(B')| / |legal(B')|.

Experiment 2 — Rollout persistence
------------------------------------
If the model genuinely internalised a world model of B', its predictions should
remain consistent with B' for many subsequent moves, not just the first one.

After the intervention, take m* = model's top-1 prediction. If m* is legal for B'
but not B (i.e. the move specifically follows the B' path), roll out 10 more moves
with no intervention:
    - Feed sequence [S + m*] to the model step by step
    - Check each prediction against the actual board state (starting from B' + m*)
    - Compare illegal-move rate to a baseline rollout from B (no intervention)

A model with no persistent world model will produce illegal moves rapidly after the
intervention is removed, because it has been fed a game history it was never trained
on.

Usage
-----
    uv run python scripts/nanda_experiment.py \\
        --games data/games_test.json \\
        --probes data/board_probes.pt \\
        --layer 5
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from othellogpt_deconstruction.core.board import (
    replay as board_replay, legal_moves, apply_move, EMPTY as BOARD_EMPTY,
)
from othellogpt_deconstruction.core.tokenizer import (
    stoi, alg_to_pos, pos_to_alg, itos, BLOCK_SIZE, PAD_ID,
)
from othellogpt_deconstruction.intervention.hooks import collect_activations, patch_activations
from othellogpt_deconstruction.model.board_probe import (
    make_labels, labels_to_absolute_board, EMPTY, MINE, YOURS,
)
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.model.probes import load_probes


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def encode_sequence(sequence: list[str], device: torch.device) -> torch.Tensor:
    tokens = [stoi[alg_to_pos(move)] for move in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


def forward_pass(model: torch.nn.Module, x: torch.Tensor, seq_length: int) -> torch.Tensor:
    with torch.no_grad():
        logits, _ = model(x)
    last_logits = logits[0, seq_length - 1, :].clone()
    last_logits[PAD_ID] = float("-inf")
    return torch.softmax(last_logits, dim=-1)


def top1_position(probs: torch.Tensor) -> int:
    token = int(probs.argmax())
    return int(itos[token]) if token != PAD_ID else -1


def topn_positions(probs: torch.Tensor, n: int) -> set[int]:
    """Return the board positions of the top-n predicted tokens."""
    top_tokens = probs.topk(n + 1).indices  # +1 in case PAD slips in
    positions = set()
    for token in top_tokens:
        token = int(token)
        if token != PAD_ID:
            positions.add(int(itos[token]))
        if len(positions) == n:
            break
    return positions


# ---------------------------------------------------------------------------
# Experiment 1: top-N error rate
# ---------------------------------------------------------------------------

def topn_errors(probs: torch.Tensor, legal_set: set[int]) -> float:
    """
    Nanda et al.'s metric: take top-N predictions (N = |legal_set|) and count
    false positives + false negatives against legal_set.
    """
    n = len(legal_set)
    if n == 0:
        return 0.0
    predicted = topn_positions(probs, n)
    false_positives = len(predicted - legal_set)
    false_negatives = len(legal_set - predicted)
    return float(false_positives + false_negatives)


# ---------------------------------------------------------------------------
# Experiment 2: rollout
# ---------------------------------------------------------------------------

def rollout(
    model:         torch.nn.Module,
    sequence:      list[str],
    starting_board: np.ndarray,
    next_player:   int,
    n_steps:       int,
    device:        torch.device,
) -> list[bool]:
    """
    Roll out n_steps moves from starting_board with no intervention.
    At each step feed the current sequence to the model, take top-1, check legality.
    Returns a list of booleans: True = legal move predicted.
    """
    board = starting_board.copy()
    player = next_player
    seq = list(sequence)
    results = []

    for _ in range(n_steps):
        if len(seq) >= BLOCK_SIZE:
            break
        x = encode_sequence(seq, device)
        probs = forward_pass(model, x, len(seq))
        pos = top1_position(probs)

        legal_set = set(legal_moves(board, player))
        is_legal = pos in legal_set
        results.append(is_legal)

        if is_legal:
            board = apply_move(board, pos, player)
            player = 3 - player
            if not legal_moves(board, player):
                player = 3 - player
        # Always append the move to the sequence so the model sees it
        if pos >= 0:
            seq.append(pos_to_alg(pos))
        else:
            break

    return results


# ---------------------------------------------------------------------------
# Per-position analysis
# ---------------------------------------------------------------------------

def analyse_position(
    model:         torch.nn.Module,
    probe:         object,
    sequence:      list[str],
    layer:         int,
    n_flips_list:  list[int],
    n_rollout:     int,
    device:        torch.device,
    rng:           random.Random,
) -> dict | None:
    try:
        board, next_player = board_replay(sequence)
    except ValueError:
        return None

    source_legal_set = set(legal_moves(board, next_player))
    if not source_legal_set:
        return None

    source_labels = make_labels(board, next_player)
    flippable_positions = [pos for pos in range(64) if source_labels[pos] != EMPTY]
    if len(flippable_positions) < max(n_flips_list):
        return None

    x = encode_sequence(sequence, device)
    with collect_activations(model, len(sequence)) as store:
        probs_original = forward_pass(model, x, len(sequence))

    activation = store.get(layer)
    if activation is None:
        return None

    results = {}

    for n_flips in n_flips_list:
        cells_to_flip = rng.sample(flippable_positions, n_flips)

        target_labels = source_labels.copy()
        for pos in cells_to_flip:
            target_labels[pos] = YOURS if source_labels[pos] == MINE else MINE

        target_board = labels_to_absolute_board(target_labels, next_player)
        target_legal_set = set(legal_moves(target_board, next_player))
        if not target_legal_set:
            continue

        cell_diffs = [
            {"pos": pos, "color_a": int(source_labels[pos]), "color_b": int(target_labels[pos])}
            for pos in cells_to_flip
        ]
        intervention_vector = probe.trichrome_direction(cell_diffs).to(device)

        with patch_activations(model, len(sequence), {layer: intervention_vector}):
            probs_intervened = forward_pass(model, x, len(sequence))

        # --- Experiment 1: 2x2 error table ---
        legal_overlap = len(source_legal_set & target_legal_set) / len(target_legal_set)

        original_vs_source  = topn_errors(probs_original,  source_legal_set)
        original_vs_target  = topn_errors(probs_original,  target_legal_set)
        intervened_vs_target = topn_errors(probs_intervened, target_legal_set)
        intervened_vs_source = topn_errors(probs_intervened, source_legal_set)

        # --- Experiment 2: rollout ---
        m_star = top1_position(probs_intervened)
        m_star_legal_source = m_star in source_legal_set
        m_star_legal_target = m_star in target_legal_set
        m_star_unique_to_target = m_star_legal_target and not m_star_legal_source

        rollout_illegal_rate = None
        baseline_illegal_rate = None

        if m_star_legal_target and m_star >= 0 and len(sequence) < BLOCK_SIZE - 1:
            # Apply m* to target board to get rollout starting state
            try:
                rollout_board = apply_move(target_board, m_star, next_player)
                rollout_player = 3 - next_player
                if not legal_moves(rollout_board, rollout_player):
                    rollout_player = 3 - rollout_player
                rollout_seq = sequence + [pos_to_alg(m_star)]

                rollout_results = rollout(
                    model, rollout_seq, rollout_board, rollout_player, n_rollout, device
                )
                rollout_illegal_rate = 1.0 - float(np.mean(rollout_results)) if rollout_results else None
            except ValueError:
                rollout_illegal_rate = None

        # Baseline rollout: no intervention, from source board
        if len(sequence) < BLOCK_SIZE - 1:
            baseline_results = rollout(
                model, sequence, board, next_player, n_rollout, device
            )
            baseline_illegal_rate = 1.0 - float(np.mean(baseline_results)) if baseline_results else None

        results[n_flips] = {
            # Legal set info
            "n_legal_source":         len(source_legal_set),
            "n_legal_target":         len(target_legal_set),
            "legal_overlap":          legal_overlap,

            # Experiment 1: 2x2 error table
            "original_vs_source":     original_vs_source,
            "original_vs_target":     original_vs_target,
            "intervened_vs_target":   intervened_vs_target,
            "intervened_vs_source":   intervened_vs_source,

            # Experiment 2: rollout
            "m_star_legal_source":    m_star_legal_source,
            "m_star_legal_target":    m_star_legal_target,
            "m_star_unique_to_target": m_star_unique_to_target,
            "rollout_illegal_rate":   rollout_illegal_rate,
            "baseline_illegal_rate":  baseline_illegal_rate,
        }

    return results if results else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(all_results: list[dict], n_flips_list: list[int], n_rollout: int) -> None:
    print("\n" + "=" * 80)
    print("Experiment 1: Legal-set overlap and top-N error rates")
    print("=" * 80)
    print(f"\n{'n_flips':>8}  {'n':>6}  {'overlap':>9}  "
          f"{'orig→B':>8}  {'orig→B′':>9}  {'intv→B′':>9}  {'intv→B':>8}")
    print("-" * 75)
    print(f"{'':8}  {'':6}  {'':9}  "
          f"{'(baseline)':>8}  {'(null)':>9}  {'(Nanda)':>9}  {'(key)':>8}")
    print("-" * 75)

    for n_flips in n_flips_list:
        entries = [r[n_flips] for r in all_results if n_flips in r]
        if not entries:
            continue
        n = len(entries)
        overlap          = np.mean([e["legal_overlap"]        for e in entries])
        orig_vs_source   = np.mean([e["original_vs_source"]   for e in entries])
        orig_vs_target   = np.mean([e["original_vs_target"]   for e in entries])
        intv_vs_target   = np.mean([e["intervened_vs_target"] for e in entries])
        intv_vs_source   = np.mean([e["intervened_vs_source"] for e in entries])

        print(f"{n_flips:>8}  {n:>6}  {overlap:>9.3f}  "
              f"{orig_vs_source:>8.3f}  {orig_vs_target:>9.3f}  "
              f"{intv_vs_target:>9.3f}  {intv_vs_source:>8.3f}")

    print()
    print("overlap  = |legal(B) ∩ legal(B′)| / |legal(B′)|")
    print("orig→B   = errors of original model vs legal(B)  [should be ~0]")
    print("orig→B′  = errors of original model vs legal(B′) [Nanda's null baseline]")
    print("intv→B′  = errors of intervened model vs legal(B′) [Nanda's claim]")
    print("intv→B   = errors of intervened model vs legal(B)  [key: still predicts B?]")

    print("\n" + "=" * 80)
    print(f"Experiment 2: Rollout persistence ({n_rollout} steps after intervention)")
    print("=" * 80)

    for n_flips in n_flips_list:
        entries = [r[n_flips] for r in all_results if n_flips in r]
        if not entries:
            continue

        all_with_rollout    = [e for e in entries if e["rollout_illegal_rate"] is not None]
        unique_to_target    = [e for e in entries if e["m_star_unique_to_target"]]
        unique_with_rollout = [e for e in unique_to_target if e["rollout_illegal_rate"] is not None]
        baseline_entries    = [e for e in entries if e["baseline_illegal_rate"] is not None]

        baseline_rate = np.mean([e["baseline_illegal_rate"] for e in baseline_entries]) if baseline_entries else float("nan")
        rollout_rate_all = np.mean([e["rollout_illegal_rate"] for e in all_with_rollout]) if all_with_rollout else float("nan")
        rollout_rate_unique = np.mean([e["rollout_illegal_rate"] for e in unique_with_rollout]) if unique_with_rollout else float("nan")

        frac_unique = len(unique_to_target) / len(entries) if entries else 0.0

        print(f"\nn_flips={n_flips}  (n={len(entries)}, "
              f"{len(unique_to_target)} positions where m* unique to B′ = {100*frac_unique:.1f}%)")
        print(f"  Baseline illegal rate (no intervention, {n_rollout} steps): "
              f"{100*baseline_rate:.1f}%")
        print(f"  Rollout illegal rate  (all positions with m* legal for B′): "
              f"{100*rollout_rate_all:.1f}%  (n={len(all_with_rollout)})")
        print(f"  Rollout illegal rate  (m* unique to B′ only):               "
              f"{100*rollout_rate_unique:.1f}%  (n={len(unique_with_rollout)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined legal-set overlap and rollout persistence experiments."
    )
    parser.add_argument("--games",   required=True)
    parser.add_argument("--probes",  required=True)
    parser.add_argument(
        "--mode", default="championship",
        choices=["championship", "synthetic", "random"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--layer", type=int, default=5,
        help="Layer to intervene on (default: 5)",
    )
    parser.add_argument(
        "--n-flips", default="1,2,3,5",
        help="Comma-separated flip counts (default: 1,2,3,5)",
    )
    parser.add_argument(
        "--n-rollout", type=int, default=10,
        help="Steps to roll out after intervention (default: 10)",
    )
    parser.add_argument("--n-games", type=int, default=None)
    parser.add_argument("--n-timesteps-per-game", type=int, default=2)
    parser.add_argument("--min-ply", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_flips_list = [int(n) for n in args.n_flips.split(",")]

    print(f"Loading model ({args.mode})...")
    model = load_model(args.mode, args.checkpoint)
    device = next(model.parameters()).device

    print(f"Loading probes from {args.probes}...")
    probes = load_probes(args.probes)
    if args.layer not in probes:
        raise ValueError(f"Layer {args.layer} not in probes. Available: {sorted(probes.keys())}")
    probe = probes[args.layer]
    print(f"  Using probe at layer {args.layer}.")

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

    print(f"  {len(positions)} positions to test.")
    print(f"  Flip counts: {n_flips_list}, rollout steps: {args.n_rollout}")

    all_results: list[dict] = []
    for position_idx, sequence in enumerate(positions):
        print(f"  Position {position_idx + 1}/{len(positions)}", end="\r")
        result = analyse_position(
            model, probe, sequence, args.layer,
            n_flips_list, args.n_rollout, device, rng,
        )
        if result is not None:
            all_results.append(result)
    print()

    print(f"\n{len(all_results)} positions analysed.")
    report(all_results, n_flips_list, args.n_rollout)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(all_results, output_file)
        print(f"\nSaved results to '{args.output}'")


if __name__ == "__main__":
    main()
