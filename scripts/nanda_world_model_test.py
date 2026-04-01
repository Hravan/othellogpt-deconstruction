"""
scripts/nanda_world_model_test.py

Two experiments attacking Nanda et al.'s world model claim via their
illegal-state intervention.

Experiment 1 — Legal-set overlap
---------------------------------
Nanda et al. measure top-N predictions post-intervention against legal_moves(B'),
where N = |legal_moves(B')|, and report ~0.10 average errors as evidence the model
tracks the new board state B'.

The missing comparison: how does the *original* (unintervened) model score against
legal_moves(B')? If legal(B) and legal(B') overlap heavily, the original model
already scores near 0.10 — the intervention adds nothing.

Full 2×2 table:
    original model   vs legal(B)   → baseline (~0, model plays legal Othello)
    original model   vs legal(B')  → null baseline (Nanda's "null intervention")
    intervened model vs legal(B')  → Nanda's claim
    intervened model vs legal(B)   → key: does the model still predict B after intervention?

Experiment 2 — Rollout persistence
------------------------------------
If the model genuinely internalised a world model of B', its predictions should
remain consistent with B' for subsequent moves, not just the first one.

After the intervention, take m* = model's top-1 prediction. If m* is legal for B'
but not B (i.e. the move specifically follows the B' path), roll out n_rollout more
moves with no intervention:
    - Feed [S + m*] to the model step by step (no patching)
    - Check each prediction against the actual board state (starting from B' + m*)
    - Compare illegal-move rate to a baseline rollout from B (no intervention)

A model with no persistent world model will produce illegal moves rapidly, because
it has been fed a game history it was never trained on and has no "memory" of B'.

Usage
-----
    uv run python scripts/nanda_world_model_test.py \\
        --games data/games_test.json \\
        --probes data/board_probes.pt \\
        --layer 5

    uv run python scripts/nanda_world_model_test.py \\
        --games data/games_test.json \\
        --probes data/board_probes.pt \\
        --layer 5 \\
        --n-flips 1 \\
        --n-rollout 10 \\
        --output data/nanda_world_model_results.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from othellogpt_deconstruction.core.board import (
    replay as board_replay, legal_moves, apply_move,
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
    top_tokens = probs.topk(n + 1).indices
    positions = set()
    for token in top_tokens:
        token = int(token)
        if token != PAD_ID:
            positions.add(int(itos[token]))
        if len(positions) == n:
            break
    return positions


# ---------------------------------------------------------------------------
# Experiment 1 helpers
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
    return float(len(predicted - legal_set) + len(legal_set - predicted))


# ---------------------------------------------------------------------------
# Experiment 2 helpers
# ---------------------------------------------------------------------------

def rollout(
    model:          torch.nn.Module,
    sequence:       list[str],
    starting_board: np.ndarray,
    next_player:    int,
    n_steps:        int,
    device:         torch.device,
) -> list[bool]:
    """
    Roll out n_steps moves from starting_board with no intervention.
    At each step feed the current sequence to the model, take top-1, check legality.
    Always appends the predicted move to the sequence (even if illegal) so the model
    sees its own outputs. Returns a list of booleans: True = legal prediction.
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
        if pos < 0:
            break

        legal_set = set(legal_moves(board, player))
        is_legal = pos in legal_set
        results.append(is_legal)

        if is_legal:
            board = apply_move(board, pos, player)
            player = 3 - player
            if not legal_moves(board, player):
                player = 3 - player

        seq.append(pos_to_alg(pos))

    return results


# ---------------------------------------------------------------------------
# Per-position analysis
# ---------------------------------------------------------------------------

def analyse_position(
    model:      torch.nn.Module,
    probe:      object,
    sequence:   list[str],
    layer:      int,
    n_flips:    int,
    n_rollout:  int,
    device:     torch.device,
    rng:        random.Random,
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
    if len(flippable_positions) < n_flips:
        return None

    x = encode_sequence(sequence, device)
    with collect_activations(model, len(sequence)) as store:
        probs_original = forward_pass(model, x, len(sequence))

    activation = store.get(layer)
    if activation is None:
        return None

    # Build target board B' by flipping n_flips cells Mine↔Yours
    cells_to_flip = rng.sample(flippable_positions, n_flips)
    target_labels = source_labels.copy()
    for pos in cells_to_flip:
        target_labels[pos] = YOURS if source_labels[pos] == MINE else MINE

    target_board = labels_to_absolute_board(target_labels, next_player)
    target_legal_set = set(legal_moves(target_board, next_player))
    if not target_legal_set:
        return None

    cell_diffs = [
        {"pos": pos, "color_a": int(source_labels[pos]), "color_b": int(target_labels[pos])}
        for pos in cells_to_flip
    ]
    intervention_vector = probe.trichrome_direction(cell_diffs).to(device)

    with patch_activations(model, len(sequence), {layer: intervention_vector}):
        probs_intervened = forward_pass(model, x, len(sequence))

    # --- Experiment 1: 2×2 error table ---
    legal_overlap        = len(source_legal_set & target_legal_set) / len(target_legal_set)
    original_vs_source   = topn_errors(probs_original,   source_legal_set)
    original_vs_target   = topn_errors(probs_original,   target_legal_set)
    intervened_vs_target = topn_errors(probs_intervened,  target_legal_set)
    intervened_vs_source = topn_errors(probs_intervened,  source_legal_set)

    # --- Experiment 2: rollout persistence ---
    m_star = top1_position(probs_intervened)
    m_star_legal_source     = m_star in source_legal_set
    m_star_legal_target     = m_star in target_legal_set
    m_star_unique_to_target = m_star_legal_target and not m_star_legal_source

    rollout_illegal_rate  = None
    baseline_illegal_rate = None

    if m_star_legal_target and m_star >= 0 and len(sequence) < BLOCK_SIZE - 1:
        try:
            rollout_board = apply_move(target_board, m_star, next_player)
            rollout_player = 3 - next_player
            if not legal_moves(rollout_board, rollout_player):
                rollout_player = 3 - rollout_player
            rollout_seq = sequence + [pos_to_alg(m_star)]
            rollout_results = rollout(
                model, rollout_seq, rollout_board, rollout_player, n_rollout, device
            )
            if rollout_results:
                rollout_illegal_rate = 1.0 - float(np.mean(rollout_results))
        except ValueError:
            pass

    if len(sequence) < BLOCK_SIZE - 1:
        baseline_results = rollout(model, sequence, board, next_player, n_rollout, device)
        if baseline_results:
            baseline_illegal_rate = 1.0 - float(np.mean(baseline_results))

    return {
        "n_legal_source":          len(source_legal_set),
        "n_legal_target":          len(target_legal_set),
        "legal_overlap":           legal_overlap,
        "original_vs_source":      original_vs_source,
        "original_vs_target":      original_vs_target,
        "intervened_vs_target":    intervened_vs_target,
        "intervened_vs_source":    intervened_vs_source,
        "m_star_legal_source":     m_star_legal_source,
        "m_star_legal_target":     m_star_legal_target,
        "m_star_unique_to_target": m_star_unique_to_target,
        "rollout_illegal_rate":    rollout_illegal_rate,
        "baseline_illegal_rate":   baseline_illegal_rate,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(all_results: list[dict], n_flips: int, n_rollout: int) -> None:
    n = len(all_results)

    print("\n" + "=" * 80)
    print(f"Experiment 1: Legal-set overlap and top-N error rates  (n_flips={n_flips}, n={n})")
    print("=" * 80)

    overlap          = np.mean([r["legal_overlap"]        for r in all_results])
    orig_vs_source   = np.mean([r["original_vs_source"]   for r in all_results])
    orig_vs_target   = np.mean([r["original_vs_target"]   for r in all_results])
    intv_vs_target   = np.mean([r["intervened_vs_target"] for r in all_results])
    intv_vs_source   = np.mean([r["intervened_vs_source"] for r in all_results])

    col = 28
    print(f"\n  {'legal set overlap':.<{col}} {overlap:.3f}")
    print(f"  {'original model vs legal(B)':.<{col}} {orig_vs_source:.3f}  [baseline: should be ~0]")
    print(f"  {'original model vs legal(B′)':.<{col}} {orig_vs_target:.3f}  [Nanda's null intervention]")
    print(f"  {'intervened model vs legal(B′)':.<{col}} {intv_vs_target:.3f}  [Nanda's claim]")
    print(f"  {'intervened model vs legal(B)':.<{col}} {intv_vs_source:.3f}  [key: still predicts B?]")

    print("\n" + "=" * 80)
    print(f"Experiment 2: Rollout persistence  ({n_rollout} steps, n_flips={n_flips})")
    print("=" * 80)

    baseline_entries    = [r for r in all_results if r["baseline_illegal_rate"] is not None]
    all_with_rollout    = [r for r in all_results if r["rollout_illegal_rate"]  is not None]
    unique_to_target    = [r for r in all_results if r["m_star_unique_to_target"]]
    unique_with_rollout = [r for r in unique_to_target if r["rollout_illegal_rate"] is not None]

    frac_unique      = len(unique_to_target) / n if n else 0.0
    baseline_rate    = np.mean([r["baseline_illegal_rate"] for r in baseline_entries]) if baseline_entries else float("nan")
    rollout_rate_all = np.mean([r["rollout_illegal_rate"]  for r in all_with_rollout]) if all_with_rollout else float("nan")
    rollout_rate_uniq = np.mean([r["rollout_illegal_rate"] for r in unique_with_rollout]) if unique_with_rollout else float("nan")

    print(f"\n  m* unique to B′ (legal for B′ but not B): "
          f"{len(unique_to_target)}/{n} positions ({100*frac_unique:.1f}%)")
    print(f"\n  {'Baseline illegal rate (no intervention)':.<45} {100*baseline_rate:.1f}%"
          f"  (n={len(baseline_entries)})")
    print(f"  {'Rollout illegal rate (m* legal for B′)':.<45} {100*rollout_rate_all:.1f}%"
          f"  (n={len(all_with_rollout)})")
    print(f"  {'Rollout illegal rate (m* unique to B′ only)':.<45} {100*rollout_rate_uniq:.1f}%"
          f"  (n={len(unique_with_rollout)})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Nanda et al.'s world model claim: legal-set overlap and rollout persistence."
    )
    parser.add_argument("--games",  required=True,
                        help="Path to test games JSON (output of split_corpus.py)")
    parser.add_argument("--probes", required=True,
                        help="Path to trained board probes (output of train_board_probes.py)")
    parser.add_argument("--mode", default="championship",
                        choices=["championship", "synthetic", "random"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--layer", type=int, default=5,
                        help="Layer to intervene on (default: 5)")
    parser.add_argument("--n-flips", type=int, default=1,
                        help="Number of cells to flip Mine↔Yours (default: 1, matching Nanda)")
    parser.add_argument("--n-rollout", type=int, default=10,
                        help="Steps to roll out after intervention (default: 10)")
    parser.add_argument("--n-games", type=int, default=None)
    parser.add_argument("--n-timesteps-per-game", type=int, default=2)
    parser.add_argument("--min-ply", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    print(f"  n_flips={args.n_flips}, n_rollout={args.n_rollout}")

    all_results: list[dict] = []
    for position_idx, sequence in enumerate(positions):
        print(f"  Position {position_idx + 1}/{len(positions)}", end="\r")
        result = analyse_position(
            model, probe, sequence, args.layer,
            args.n_flips, args.n_rollout, device, rng,
        )
        if result is not None:
            all_results.append(result)
    print()

    print(f"\n{len(all_results)} positions analysed.")
    report(all_results, args.n_flips, args.n_rollout)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(all_results, output_file)
        print(f"Saved results to '{args.output}'")


if __name__ == "__main__":
    main()
