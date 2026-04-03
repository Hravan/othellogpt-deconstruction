"""
scripts/li_intervention_test.py

Replication and attack of Li et al.'s intervention experiment.

Li et al. intervene on the model's residual stream using gradient descent
against a non-linear probe (BatteryProbeClassificationTwoLayer, mid_dim=128),
applying 1000 gradient steps at each of 5 successive layers (4-8). They claim
this causes the model to predict legal moves for an illegal board state B'.

This script:

1. Computes the null baseline (original model vs legal(B')) that Li do not
   report. If the legal set overlap is high, the original model already scores
   near Li's post-intervention error — revealing their metric is confounded.

2. Replicates Li's multi-layer gradient descent intervention and computes
   post-intervention errors, matching their setup as closely as possible.

3. Runs rollout persistence: after the intervention produces m*, the model
   plays 10 more steps without patching. A genuine world model of B' would
   sustain legal play; a disrupted model would degrade.

Encoding note
-------------
Li's probe uses absolute board encoding:
    0 = white piece
    1 = empty
    2 = black piece

Our board uses:
    EMPTY = 0, BLACK = 1, WHITE = 2

Conversion: li_label = (our_label + 1) % 3

Usage
-----
    uv run python scripts/li_intervention_test.py \\
        --games data/games_synthetic_test.json \\
        --layer-start 4 \\
        --mode synthetic \\
        --output data/li_intervention_results.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Li's model and probe code
sys.path.insert(0, str(Path(__file__).parent.parent / "mingpt"))
from mingpt.model import GPTforProbeIA, GPTConfig
from mingpt.probe_model import BatteryProbeClassificationTwoLayer

from othellogpt_deconstruction.core.board import (
    replay as board_replay, legal_moves, apply_move, EMPTY, BLACK, WHITE,
)
from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, pos_to_alg, itos, BLOCK_SIZE, PAD_ID


# ---------------------------------------------------------------------------
# Encoding conversion
# ---------------------------------------------------------------------------

def board_to_li(board: np.ndarray) -> torch.Tensor:
    """
    Convert our board (EMPTY=0, BLACK=1, WHITE=2) to Li's probe encoding
    (0=white, 1=empty, 2=black) as a long tensor of shape (64,).
    """
    return torch.tensor((board + 1) % 3, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model and probe loading
# ---------------------------------------------------------------------------

def load_li_model(checkpoint_path: str, probe_layer: int, device: torch.device) -> GPTforProbeIA:
    mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
    model = GPTforProbeIA(mconf, probe_layer=probe_layer)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_li_probes(
    probe_dir: str,
    layers: list[int],
    device: torch.device,
    mid_dim: int = 128,
) -> dict[int, BatteryProbeClassificationTwoLayer]:
    probes = {}
    for layer in layers:
        probe = BatteryProbeClassificationTwoLayer(
            device=device, probe_class=3, num_task=64, mid_dim=mid_dim,
        )
        checkpoint_path = Path(probe_dir) / f"layer{layer}" / "checkpoint.ckpt"
        probe.load_state_dict(torch.load(checkpoint_path, map_location=device))
        probe.eval()
        probes[layer] = probe
    return probes


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------

def encode_sequence(sequence: list[str], device: torch.device) -> torch.Tensor:
    tokens = [stoi[alg_to_pos(move)] for move in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def get_probs_original(model: GPTforProbeIA, x: torch.Tensor, seq_length: int) -> torch.Tensor:
    """Run standard forward pass, return softmax probs at last position."""
    with torch.no_grad():
        logits, _ = model(x)
    last_logits = logits[0, seq_length - 1, :].clone()
    last_logits[PAD_ID] = float("-inf")
    return torch.softmax(last_logits, dim=-1)


def topn_errors(probs: torch.Tensor, legal_set: set[int]) -> float:
    n = len(legal_set)
    if n == 0:
        return 0.0
    predicted_positions: set[int] = set()
    for token in probs.topk(n + 1).indices:
        token_int = int(token)
        if token_int != PAD_ID and token_int in itos:
            predicted_positions.add(int(itos[token_int]))
        if len(predicted_positions) == n:
            break
    return float(len(predicted_positions - legal_set) + len(legal_set - predicted_positions))


def top1_position(probs: torch.Tensor) -> int:
    token = int(probs.argmax())
    if token in itos:
        return int(itos[token])
    return -1


# ---------------------------------------------------------------------------
# Li's gradient descent intervention
# ---------------------------------------------------------------------------

def li_intervene(
    probe:            BatteryProbeClassificationTwoLayer,
    activation:       torch.Tensor,
    labels_current:   torch.Tensor,
    flip_position:    int,
    flip_to:          int,
    lr:               float = 1e-3,
    steps:            int = 1000,
    reg_strength:     float = 0.2,
) -> torch.Tensor:
    """
    Gradient descent intervention matching Li et al.'s `intervene()`.

    Modifies activation so that the probe predicts flip_to at flip_position,
    while regularising other cells to stay at labels_current.

    Parameters
    ----------
    probe           : BatteryProbeClassificationTwoLayer
    activation      : (512,) float tensor on device
    labels_current  : (64,) long tensor, Li's encoding (0=white, 1=empty, 2=black)
    flip_position   : board position 0-63 to flip
    flip_to         : target Li label (0 or 2)
    """
    new_activation = torch.tensor(
        activation.detach().cpu().numpy(), dtype=torch.float32,
    ).to(activation.device)
    new_activation.requires_grad = True

    optimizer = torch.optim.Adam([new_activation], lr=lr)

    target_labels = labels_current.clone()
    target_labels[flip_position] = flip_to

    weight_mask = reg_strength * torch.ones(64, device=activation.device)
    weight_mask[flip_position] = 1.0

    for _ in range(steps):
        optimizer.zero_grad()
        logits = probe(new_activation[None, :])[0][0]  # [64, 3]
        loss = F.cross_entropy(logits, target_labels, reduction="none")
        loss = torch.mean(weight_mask * loss)
        loss.backward()
        optimizer.step()

    return new_activation.detach()


# ---------------------------------------------------------------------------
# Full multi-layer intervention (matching Li's notebook)
# ---------------------------------------------------------------------------

def li_full_intervention(
    model:          GPTforProbeIA,
    probes:         dict[int, BatteryProbeClassificationTwoLayer],
    x:              torch.Tensor,
    seq_length:     int,
    labels_current: torch.Tensor,
    flip_position:  int,
    flip_to:        int,
    layer_start:    int = 4,
    layer_end:      int = 9,
    lr:             float = 1e-3,
    steps:          int = 1000,
    reg_strength:   float = 0.2,
) -> torch.Tensor:
    """
    Replicate Li's multi-layer gradient descent intervention.

    Intervenes at layer_start, propagates through each subsequent layer,
    re-intervening at each one. Returns final softmax probabilities.

    We intervene at position `seq_length - 1` (the last real token, not the
    last padded position). predict() is position-wise (ln_f + head only), so
    the output at seq_length - 1 only depends on the hidden state at that
    position — intervening at the wrong position would have zero effect.
    """
    last_pos = seq_length - 1

    # Stage 1: run blocks 0..layer_start-1
    with torch.no_grad():
        whole_mid_act = model.forward_1st_stage(x)  # [1, T, 512]

    # First intervention (before block layer_start)
    mid_act = whole_mid_act[0, last_pos]
    probe_labels = labels_current.clone()
    new_mid_act = li_intervene(
        probes[layer_start], mid_act, probe_labels,
        flip_position, flip_to, lr, steps, reg_strength,
    )
    delta_layer_start = float(torch.norm(new_mid_act - mid_act).item())
    whole_mid_act = whole_mid_act.detach().clone()
    whole_mid_act[0, last_pos] = new_mid_act

    # Stage 2: propagate through layers, re-intervening at each
    activation_deltas = {layer_start: delta_layer_start}
    for layer in range(layer_start, layer_end - 1):
        with torch.no_grad():
            whole_mid_act = model.forward_2nd_stage(whole_mid_act, layer, layer + 1)[0]

        mid_act = whole_mid_act[0, last_pos]
        probe_labels = labels_current.clone()
        new_mid_act = li_intervene(
            probes[layer + 1], mid_act, probe_labels,
            flip_position, flip_to, lr, steps, reg_strength,
        )
        activation_deltas[layer + 1] = float(torch.norm(new_mid_act - mid_act).item())
        whole_mid_act = whole_mid_act.detach().clone()
        whole_mid_act[0, last_pos] = new_mid_act

    # Final prediction
    with torch.no_grad():
        logits, _ = model.predict(whole_mid_act)
    last_logits = logits[0, last_pos, :].clone()
    last_logits[PAD_ID] = float("-inf")
    debug = {
        "activation_deltas": activation_deltas,
        "last_pos": last_pos,
    }
    return torch.softmax(last_logits, dim=-1), debug


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(
    model:          GPTforProbeIA,
    sequence:       list[str],
    starting_board: np.ndarray,
    next_player:    int,
    n_steps:        int,
    device:         torch.device,
) -> list[bool]:
    board = starting_board.copy()
    player = next_player
    seq = list(sequence)
    results = []

    for _ in range(n_steps):
        if len(seq) >= BLOCK_SIZE:
            break
        x = encode_sequence(seq, device)
        probs = get_probs_original(model, x, len(seq))
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
    model:       GPTforProbeIA,
    probes:      dict[int, BatteryProbeClassificationTwoLayer],
    sequence:    list[str],
    layer_start: int,
    layer_end:   int,
    n_rollout:   int,
    device:      torch.device,
    rng:         random.Random,
) -> dict | None:
    try:
        board, next_player = board_replay(sequence)
    except ValueError:
        return None

    source_legal_set = set(legal_moves(board, next_player))
    if not source_legal_set:
        return None

    # Find a non-empty cell to flip (BLACK↔WHITE only, not EMPTY)
    flippable = [pos for pos in range(64) if board[pos] != EMPTY]
    if not flippable:
        return None

    flip_position = rng.choice(flippable)
    original_color = int(board[flip_position])  # our encoding: BLACK=1 or WHITE=2
    li_original = (original_color + 1) % 3      # Li encoding: 2=black, 0=white
    li_target   = 2 - li_original               # flip: 2↔0

    # Build B' by flipping the cell in absolute encoding
    target_board = board.copy()
    target_board[flip_position] = WHITE if original_color == BLACK else BLACK
    target_legal_set = set(legal_moves(target_board, next_player))
    if not target_legal_set:
        return None

    legal_overlap = len(source_legal_set & target_legal_set) / len(target_legal_set)

    x = encode_sequence(sequence, device)
    probs_original = get_probs_original(model, x, len(sequence))

    original_vs_source = topn_errors(probs_original, source_legal_set)
    original_vs_target = topn_errors(probs_original, target_legal_set)

    # Li's current board labels in Li's encoding
    labels_current = board_to_li(board).to(device)

    # Run Li's multi-layer intervention
    probs_intervened, debug_info = li_full_intervention(
        model, probes, x, len(sequence),
        labels_current, flip_position, li_target,
        layer_start=layer_start, layer_end=layer_end,
    )

    prob_delta = float((probs_intervened - probs_original).abs().sum().item())
    print(f"    [debug] last_pos={debug_info['last_pos']}  prob_delta={prob_delta:.4f}  "
          f"act_deltas={[f'{v:.2f}' for v in debug_info['activation_deltas'].values()]}")

    intervened_vs_target = topn_errors(probs_intervened, target_legal_set)
    intervened_vs_source = topn_errors(probs_intervened, source_legal_set)

    # Rollout persistence
    m_star = top1_position(probs_intervened)
    m_star_legal_source     = m_star in source_legal_set
    m_star_legal_target     = m_star in target_legal_set
    m_star_unique_to_target = m_star_legal_target and not m_star_legal_source

    rollout_illegal_rate  = None
    baseline_illegal_rate = None

    if m_star_legal_target and m_star >= 0 and len(sequence) < BLOCK_SIZE - 1:
        try:
            rollout_board  = apply_move(target_board, m_star, next_player)
            rollout_player = 3 - next_player
            if not legal_moves(rollout_board, rollout_player):
                rollout_player = 3 - rollout_player
            rollout_seq = sequence + [pos_to_alg(m_star)]
            rollout_results = rollout(model, rollout_seq, rollout_board, rollout_player, n_rollout, device)
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

def report(all_results: list[dict], layer_start: int, layer_end: int, n_rollout: int) -> None:
    n = len(all_results)

    print("\n" + "=" * 80)
    print(f"Li et al. Intervention: null baseline + replication  (layers {layer_start}-{layer_end-1}, n={n})")
    print("=" * 80)

    mean_n_legal_source = np.mean([r["n_legal_source"] for r in all_results])
    mean_n_legal_target = np.mean([r["n_legal_target"] for r in all_results])
    overlap             = np.mean([r["legal_overlap"]        for r in all_results])
    orig_vs_source      = np.mean([r["original_vs_source"]   for r in all_results])
    orig_vs_target      = np.mean([r["original_vs_target"]   for r in all_results])
    intv_vs_target      = np.mean([r["intervened_vs_target"] for r in all_results])
    intv_vs_source      = np.mean([r["intervened_vs_source"] for r in all_results])

    col = 40
    print(f"\n  {'mean |legal(B)|':.<{col}} {mean_n_legal_source:.2f}")
    print(f"  {'mean |legal(B′)|':.<{col}} {mean_n_legal_target:.2f}")
    print(f"  {'legal set overlap':.<{col}} {overlap:.3f}")
    print(f"  {'original model vs legal(B)':.<{col}} {orig_vs_source:.3f}  [baseline: should be ~0]")
    print(f"  {'original model vs legal(B′)':.<{col}} {orig_vs_target:.3f}  [Li null baseline — NOT reported]")
    print(f"  {'intervened model vs legal(B′)':.<{col}} {intv_vs_target:.3f}  [Li's claim]")
    print(f"  {'intervened model vs legal(B)':.<{col}} {intv_vs_source:.3f}  [drift from B]")

    max_improvement = orig_vs_target
    improvement     = orig_vs_target - intv_vs_target
    drift           = intv_vs_source - orig_vs_source
    print(f"\n  Intervention improvement toward B′: {improvement:.3f} / {max_improvement:.3f} "
          f"= {100*improvement/max_improvement:.1f}% of maximum")
    print(f"  Model drift away from B:           {drift:.3f} / {max_improvement:.3f} "
          f"= {100*drift/max_improvement:.1f}% of maximum")

    print("\n" + "=" * 80)
    print(f"Rollout persistence  ({n_rollout} steps)")
    print("=" * 80)

    baseline_entries    = [r for r in all_results if r["baseline_illegal_rate"] is not None]
    all_with_rollout    = [r for r in all_results if r["rollout_illegal_rate"]  is not None]
    unique_to_target    = [r for r in all_results if r["m_star_unique_to_target"]]
    unique_with_rollout = [r for r in unique_to_target if r["rollout_illegal_rate"] is not None]

    frac_unique       = len(unique_to_target) / n if n else 0.0
    baseline_rate     = np.mean([r["baseline_illegal_rate"] for r in baseline_entries]) if baseline_entries else float("nan")
    rollout_rate_all  = np.mean([r["rollout_illegal_rate"]  for r in all_with_rollout]) if all_with_rollout else float("nan")
    rollout_rate_uniq = np.mean([r["rollout_illegal_rate"]  for r in unique_with_rollout]) if unique_with_rollout else float("nan")

    print(f"\n  m* unique to B′: {len(unique_to_target)}/{n} positions ({100*frac_unique:.1f}%)")
    print(f"\n  {'Baseline illegal rate (no intervention)':.<45} {100*baseline_rate:.1f}%  (n={len(baseline_entries)})")
    print(f"  {'Rollout illegal rate (m* legal for B′)':.<45} {100*rollout_rate_all:.1f}%  (n={len(all_with_rollout)})")
    print(f"  {'Rollout illegal rate (m* unique to B′ only)':.<45} {100*rollout_rate_uniq:.1f}%  (n={len(unique_with_rollout)})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate Li et al.'s intervention and compute the null baseline they omit."
    )
    parser.add_argument("--games", required=True,
                        help="Path to test games JSON")
    parser.add_argument("--checkpoint", default="ckpts/gpt_synthetic.ckpt",
                        help="Path to OthelloGPT checkpoint (default: ckpts/gpt_synthetic.ckpt)")
    parser.add_argument("--probe-dir", default="ckpts/battery_othello/state_tl128",
                        help="Path to Li's non-linear probe directory (default: ckpts/battery_othello/state_tl128)")
    parser.add_argument("--layer-start", type=int, default=4,
                        help="First layer to intervene on (default: 4, matching Li)")
    parser.add_argument("--layer-end", type=int, default=9,
                        help="Layer to stop at exclusive (default: 9, layers 4-8)")
    parser.add_argument("--n-games", type=int, default=None)
    parser.add_argument("--n-timesteps-per-game", type=int, default=2)
    parser.add_argument("--min-ply", type=int, default=10)
    parser.add_argument("--n-rollout", type=int, default=10)
    parser.add_argument("--intervention-steps", type=int, default=1000,
                        help="Gradient descent steps per layer intervention (default: 1000)")
    parser.add_argument("--intervention-lr", type=float, default=1e-3)
    parser.add_argument("--intervention-reg", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading Li model from {args.checkpoint}...")
    model = load_li_model(args.checkpoint, probe_layer=args.layer_start, device=device)

    layers = list(range(args.layer_start, args.layer_end))
    print(f"Loading Li probes from {args.probe_dir} (layers {layers})...")
    probes = load_li_probes(args.probe_dir, layers, device)
    print(f"  Loaded {len(probes)} probes.")

    print(f"Loading games from {args.games}...")
    with open(args.games) as games_file:
        all_games: list[list[str]] = json.load(games_file)
    print(f"  {len(all_games)} games loaded.")

    rng = random.Random(args.seed)
    if args.n_games is not None and len(all_games) > args.n_games:
        sampled_games = rng.sample(all_games, args.n_games)
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
    print(f"  Intervention: {len(layers)} layers × {args.intervention_steps} steps each")

    all_results: list[dict] = []
    for position_idx, sequence in enumerate(positions):
        print(f"  Position {position_idx + 1}/{len(positions)}", end="\r")
        result = analyse_position(
            model, probes, sequence,
            args.layer_start, args.layer_end,
            args.n_rollout, device, rng,
        )
        if result is not None:
            all_results.append(result)
    print()

    print(f"\n{len(all_results)} positions analysed.")
    report(all_results, args.layer_start, args.layer_end, args.n_rollout)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(all_results, output_file)
        print(f"Saved results to '{args.output}'")


if __name__ == "__main__":
    main()
