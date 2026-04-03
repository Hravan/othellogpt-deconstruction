"""
scripts/li_unnatural_test.py

Attack of Li et al.'s intervention on the unnatural benchmark.

Li et al. test their intervention on an "unnatural" subset of 1000 positions
that are unreachable by legal Othello play. They construct these by feeding
the model sequences of illegal moves (which the model processes as tokens
without validating legality). The resulting board states are unreachable.

Li report an average top-N error of 0.06 on this benchmark (vs 0.12 on
natural positions), interpreting the lower error as strong evidence that
the representation is causal even for board states never seen in training.

This script:
1. Loads Li's unnatural benchmark from intervention_benchmark.pkl.
2. Computes the null baseline (original model vs legal(B')) that Li do not
   report. If the legal set overlap is high, the confound applies here too.
3. Replicates Li's multi-layer gradient descent intervention.
4. Runs rollout persistence: after the intervention produces m*, the model
   plays 10 more steps without patching.

Benchmark format (verified empirically):
  history   : list of direct board positions (0-63), NOT token indices.
               Convert to model tokens via stoi[pos].
  pos_int   : board position (0-63) of the cell to flip.
  ori_color : Li encoding of the piece at pos_int (0=white, 1=empty, 2=black).

Board encoding note
-------------------
Li's probe uses absolute board encoding:
    0 = white piece
    1 = empty
    2 = black piece

Our board uses:
    EMPTY = 0, BLACK = 1, WHITE = 2

Conversion: li_label = (our_label + 1) % 3

Usage
-----
    uv run python scripts/li_unnatural_test.py \\
        --benchmark data/intervention_benchmark.pkl \\
        --output data/li_unnatural_results.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "mingpt"))
from mingpt.model import GPTforProbeIA, GPTConfig
from mingpt.probe_model import BatteryProbeClassificationTwoLayer

from othellogpt_deconstruction.core.board import (
    start_board, flipped_by, legal_moves, apply_move, EMPTY, BLACK, WHITE,
)
from othellogpt_deconstruction.core.tokenizer import stoi, itos, pos_to_alg, PAD_ID, BLOCK_SIZE


# ---------------------------------------------------------------------------
# Encoding conversion
# ---------------------------------------------------------------------------

def board_to_li(board: np.ndarray) -> torch.Tensor:
    """Convert our board (EMPTY=0, BLACK=1, WHITE=2) to Li's probe encoding."""
    return torch.tensor((board + 1) % 3, dtype=torch.long)


# ---------------------------------------------------------------------------
# Board simulation (non-validating)
# ---------------------------------------------------------------------------

def replay_nonvalidating(board_positions: list[int]) -> tuple[np.ndarray, int]:
    """
    Replay a sequence of board positions without legality checks.

    Used for Li's unnatural benchmark, where sequences include illegal moves.
    Pieces are placed and flanks flipped as normal, but the move itself need
    not be legal on the current board.
    """
    board = start_board()
    player = BLACK
    for pos in board_positions:
        flips = flipped_by(board, pos, player)
        new_board = board.copy()
        new_board[pos] = player
        for fp in flips:
            new_board[fp] = player
        board = new_board
        player = 3 - player
        if not legal_moves(board, player):
            player = 3 - player
    return board, player


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
# Sequence encoding (board positions, not algebraic strings)
# ---------------------------------------------------------------------------

def encode_sequence_from_positions(board_positions: list[int], device: torch.device) -> torch.Tensor:
    """Encode a sequence of board positions (direct 0-63) to model tokens."""
    tokens = [stoi[pos] for pos in board_positions]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def get_probs_original(model: GPTforProbeIA, x: torch.Tensor, seq_length: int) -> torch.Tensor:
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
# Li's gradient descent intervention (unchanged from natural benchmark)
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
        logits = probe(new_activation[None, :])[0][0]
        loss = F.cross_entropy(logits, target_labels, reduction="none")
        loss = torch.mean(weight_mask * loss)
        loss.backward()
        optimizer.step()

    return new_activation.detach()


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
    last_pos = seq_length - 1

    with torch.no_grad():
        whole_mid_act = model.forward_1st_stage(x)

    mid_act = whole_mid_act[0, last_pos]
    probe_labels = labels_current.clone()
    new_mid_act = li_intervene(
        probes[layer_start], mid_act, probe_labels,
        flip_position, flip_to, lr, steps, reg_strength,
    )
    whole_mid_act = whole_mid_act.detach().clone()
    whole_mid_act[0, last_pos] = new_mid_act

    for layer in range(layer_start, layer_end - 1):
        with torch.no_grad():
            whole_mid_act = model.forward_2nd_stage(whole_mid_act, layer, layer + 1)[0]

        mid_act = whole_mid_act[0, last_pos]
        probe_labels = labels_current.clone()
        new_mid_act = li_intervene(
            probes[layer + 1], mid_act, probe_labels,
            flip_position, flip_to, lr, steps, reg_strength,
        )
        whole_mid_act = whole_mid_act.detach().clone()
        whole_mid_act[0, last_pos] = new_mid_act

    with torch.no_grad():
        logits, _ = model.predict(whole_mid_act)
    last_logits = logits[0, last_pos, :].clone()
    last_logits[PAD_ID] = float("-inf")
    return torch.softmax(last_logits, dim=-1)


# ---------------------------------------------------------------------------
# Rollout (works with board-position sequences)
# ---------------------------------------------------------------------------

def rollout(
    model:          GPTforProbeIA,
    board_positions: list[int],
    starting_board: np.ndarray,
    next_player:    int,
    n_steps:        int,
    device:         torch.device,
) -> list[bool]:
    board = starting_board.copy()
    player = next_player
    seq = list(board_positions)
    results = []

    for _ in range(n_steps):
        if len(seq) >= BLOCK_SIZE:
            break
        x = encode_sequence_from_positions(seq, device)
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

        seq.append(pos)

    return results


# ---------------------------------------------------------------------------
# Per-position analysis
# ---------------------------------------------------------------------------

def analyse_position(
    model:        GPTforProbeIA,
    probes:       dict[int, BatteryProbeClassificationTwoLayer],
    history:      list[int],
    pos_int:      int,
    li_ori_color: int,
    layer_start:  int,
    layer_end:    int,
    n_rollout:    int,
    device:       torch.device,
) -> dict | None:
    board, next_player = replay_nonvalidating(history)

    source_legal_set = set(legal_moves(board, next_player))
    if not source_legal_set:
        return None

    # Sanity check: ori_color should match board[pos_int]
    our_encoding = board[pos_int]
    li_encoding_check = (our_encoding + 1) % 3
    if li_encoding_check != li_ori_color:
        return None

    li_target = 2 - li_ori_color  # flip: 2(black)<->0(white)

    target_board = board.copy()
    target_board[pos_int] = WHITE if board[pos_int] == BLACK else BLACK
    target_legal_set = set(legal_moves(target_board, next_player))
    if not target_legal_set:
        return None

    legal_overlap = len(source_legal_set & target_legal_set) / len(target_legal_set)

    x = encode_sequence_from_positions(history, device)
    probs_original = get_probs_original(model, x, len(history))

    original_vs_source = topn_errors(probs_original, source_legal_set)
    original_vs_target = topn_errors(probs_original, target_legal_set)

    labels_current = board_to_li(board).to(device)

    probs_intervened = li_full_intervention(
        model, probes, x, len(history),
        labels_current, pos_int, li_target,
        layer_start=layer_start, layer_end=layer_end,
    )

    intervened_vs_target = topn_errors(probs_intervened, target_legal_set)
    intervened_vs_source = topn_errors(probs_intervened, source_legal_set)

    m_star = top1_position(probs_intervened)
    m_star_legal_source     = m_star in source_legal_set
    m_star_legal_target     = m_star in target_legal_set
    m_star_unique_to_target = m_star_legal_target and not m_star_legal_source

    rollout_illegal_rate  = None
    baseline_illegal_rate = None

    if m_star_legal_target and m_star >= 0 and len(history) < BLOCK_SIZE - 1:
        try:
            rollout_board  = apply_move(target_board, m_star, next_player)
            rollout_player = 3 - next_player
            if not legal_moves(rollout_board, rollout_player):
                rollout_player = 3 - rollout_player
            rollout_seq = history + [m_star]
            rollout_results = rollout(model, rollout_seq, rollout_board, rollout_player, n_rollout, device)
            if rollout_results:
                rollout_illegal_rate = 1.0 - float(np.mean(rollout_results))
        except ValueError:
            pass

    if len(history) < BLOCK_SIZE - 1:
        baseline_results = rollout(model, history, board, next_player, n_rollout, device)
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
    print(f"Li et al. Unnatural Benchmark: null baseline + replication  (layers {layer_start}-{layer_end-1}, n={n})")
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
        description="Run Li et al.'s unnatural benchmark and compute the null baseline they omit."
    )
    parser.add_argument("--benchmark", default="data/intervention_benchmark.pkl",
                        help="Path to Li's unnatural benchmark pickle (default: data/intervention_benchmark.pkl)")
    parser.add_argument("--checkpoint", default="ckpts/gpt_synthetic.ckpt",
                        help="Path to OthelloGPT checkpoint (default: ckpts/gpt_synthetic.ckpt)")
    parser.add_argument("--probe-dir", default="ckpts/battery_othello/state_tl128",
                        help="Path to Li's non-linear probe directory")
    parser.add_argument("--layer-start", type=int, default=4)
    parser.add_argument("--layer-end", type=int, default=9,
                        help="Layer to stop at exclusive (default: 9, layers 4-8)")
    parser.add_argument("--n-entries", type=int, default=None,
                        help="Max entries to process (default: all)")
    parser.add_argument("--n-rollout", type=int, default=10)
    parser.add_argument("--intervention-steps", type=int, default=1000)
    parser.add_argument("--intervention-lr", type=float, default=1e-3)
    parser.add_argument("--intervention-reg", type=float, default=0.2)
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

    print(f"Loading unnatural benchmark from {args.benchmark}...")
    with open(args.benchmark, "rb") as benchmark_file:
        benchmark: list[dict] = pickle.load(benchmark_file)
    print(f"  {len(benchmark)} entries loaded.")

    if args.n_entries is not None:
        benchmark = benchmark[:args.n_entries]
        print(f"  Truncated to {len(benchmark)} entries.")

    print(f"  Intervention: {len(layers)} layers × {args.intervention_steps} steps each")

    all_results: list[dict] = []
    skipped = 0
    for entry_idx, entry in enumerate(benchmark):
        print(f"  Entry {entry_idx + 1}/{len(benchmark)}", end="\r")
        result = analyse_position(
            model, probes,
            entry["history"],
            entry["pos_int"],
            int(entry["ori_color"]),
            args.layer_start,
            args.layer_end,
            args.n_rollout,
            device,
        )
        if result is not None:
            all_results.append(result)
        else:
            skipped += 1
    print()

    print(f"\n{len(all_results)} positions analysed ({skipped} skipped).")
    if all_results:
        report(all_results, args.layer_start, args.layer_end, args.n_rollout)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(all_results, output_file)
        print(f"Saved results to '{args.output}'")


if __name__ == "__main__":
    main()
