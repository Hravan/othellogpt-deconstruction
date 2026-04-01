"""
scripts/train_board_probes.py

Train linear probes predicting cell ownership (EMPTY / MINE / YOURS) from
OthelloGPT residual stream activations.

Uses the same per-cell LogisticRegression approach as train_trichrome_probes.py.
Labels are relative to the player who moves next at each position (Nanda's
Mine/Yours encoding).

Usage
-----
    uv run python scripts/train_board_probes.py \\
        --corpus data/sequence_data/othello_championship \\
        --output data/board_probes.pt \\
        --mode championship
"""

import argparse
import random
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.linear_model import LogisticRegression

from othellogpt_deconstruction.core.board import replay as board_replay, EMPTY as BOARD_EMPTY
from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, BLOCK_SIZE, PAD_ID
from othellogpt_deconstruction.intervention.hooks import collect_activations
from othellogpt_deconstruction.model.board_probe import make_labels, EMPTY, MINE, YOURS
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.model.probes import TrichromeProbe, save_probes


# ---------------------------------------------------------------------------
# Memory monitor
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """
    Background thread that prints RSS when it increases by at least threshold_mb.
    Use as a context manager.
    """
    def __init__(self, threshold_mb: float = 100.0, poll_interval: float = 1.0):
        self._threshold_mb   = threshold_mb
        self._poll_interval  = poll_interval
        self._process        = psutil.Process()
        self._stop_event     = threading.Event()
        self._thread         = threading.Thread(target=self._run, daemon=True)
        self._last_reported  = 0.0

    def _run(self) -> None:
        while not self._stop_event.is_set():
            current_mb = self._process.memory_info().rss / 1024 ** 2
            if current_mb - self._last_reported >= self._threshold_mb:
                print(f"  [mem] {current_mb:.0f} MB RSS")
                self._last_reported = current_mb
            time.sleep(self._poll_interval)

    def __enter__(self) -> "MemoryMonitor":
        self._last_reported = self._process.memory_info().rss / 1024 ** 2
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop_event.set()
        self._thread.join()
        final_mb = self._process.memory_info().rss / 1024 ** 2
        print(f"  [mem] {final_mb:.0f} MB RSS (end)")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def encode_sequence(sequence: list[str], device: torch.device) -> torch.Tensor:
    tokens = [stoi[alg_to_pos(move)] for move in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


def collect_data(
    model:                torch.nn.Module,
    games:                list[list[str]],
    n_timesteps_per_game: int,
    min_ply:              int,
    device:               torch.device,
    rng:                  random.Random,
    layers:               list[int] | None = None,
    max_samples:          int | None = None,
) -> tuple[dict[int, list[np.ndarray]], list[np.ndarray]]:
    """
    Collect residual stream activations and board state labels.

    Returns
    -------
    activations_by_layer : dict[layer_idx → list of (d_model,) float arrays]
    board_labels         : list of (64,) int8 arrays; EMPTY=0, MINE=1, YOURS=2
                           relative to the player who moves next
    """
    activations_by_layer: dict[int, list[np.ndarray]] = {}
    board_labels: list[np.ndarray] = []
    total_samples = 0

    for game_idx, game in enumerate(games):
        if len(game) <= min_ply:
            continue

        available_plies = list(range(min_ply, min(len(game), BLOCK_SIZE) + 1))
        sampled_plies = rng.sample(
            available_plies,
            min(n_timesteps_per_game, len(available_plies)),
        )

        for ply in sampled_plies:
            if max_samples is not None and total_samples >= max_samples:
                break

            prefix = game[:ply]

            try:
                board, next_player = board_replay(prefix)
            except ValueError:
                continue

            x = encode_sequence(prefix, device)

            with collect_activations(model, len(prefix)) as store:
                with torch.no_grad():
                    model(x)

            labels = make_labels(board, next_player)
            board_labels.append(labels)

            for layer_idx, activation in store.activations.items():
                if layers is not None and layer_idx not in layers:
                    continue
                if layer_idx not in activations_by_layer:
                    activations_by_layer[layer_idx] = []
                activations_by_layer[layer_idx].append(activation.cpu().float().numpy())

            total_samples += 1

        if max_samples is not None and total_samples >= max_samples:
            print(f"  Reached max_samples={max_samples}, stopping early.")
            break

        if (game_idx + 1) % 50 == 0:
            print(f"  Processed {game_idx + 1}/{len(games)} games, "
                  f"{total_samples} samples", end="\r")

    print(f"\n  {total_samples} total samples collected.")
    return activations_by_layer, board_labels


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe_for_layer(
    activations: list[np.ndarray],
    labels:      list[np.ndarray],
    layer_idx:   int,
) -> TrichromeProbe:
    """
    Train per-cell linear probes predicting EMPTY/MINE/YOURS for one layer.
    """
    activation_matrix = np.stack(activations)
    n_samples, d_model = activation_matrix.shape

    weights_list = []
    biases_list  = []
    n_skipped    = 0
    train_accuracies = []

    for pos in range(64):
        cell_labels = np.array([lab[pos] for lab in labels], dtype=np.int8)
        occupied_mask = cell_labels != EMPTY

        n_occupied = int(occupied_mask.sum())
        observed_classes = np.unique(cell_labels[occupied_mask])

        if n_occupied < 10 or len(observed_classes) < 2:
            weights_list.append(np.zeros((3, d_model), dtype=np.float32))
            biases_list.append(np.zeros(3, dtype=np.float32))
            n_skipped += 1
            continue

        x_cell = activation_matrix[occupied_mask]
        y_cell = cell_labels[occupied_mask]

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            random_state=42,
        )
        clf.fit(x_cell, y_cell)
        train_accuracies.append(clf.score(x_cell, y_cell))

        coef      = np.zeros((3, d_model), dtype=np.float32)
        intercept = np.zeros(3, dtype=np.float32)
        if len(clf.classes_) == 2:
            pos_label = int(clf.classes_[1])
            neg_label = int(clf.classes_[0])
            coef[pos_label]      =  clf.coef_[0]
            coef[neg_label]      = -clf.coef_[0]
            intercept[pos_label] =  clf.intercept_[0]
            intercept[neg_label] = -clf.intercept_[0]
        else:
            for class_rank, class_label in enumerate(clf.classes_):
                coef[int(class_label)]      = clf.coef_[class_rank]
                intercept[int(class_label)] = clf.intercept_[class_rank]

        weights_list.append(coef)
        biases_list.append(intercept)

    mean_accuracy = float(np.mean(train_accuracies)) if train_accuracies else 0.0
    print(f"  Layer {layer_idx}: {64 - n_skipped}/64 cells trained, "
          f"mean train acc = {mean_accuracy:.3f}")

    weights_tensor = torch.tensor(np.stack(weights_list), dtype=torch.float32)
    biases_tensor  = torch.tensor(np.stack(biases_list),  dtype=torch.float32)
    return TrichromeProbe(weights=weights_tensor, biases=biases_tensor, layer=layer_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear board-state probes on OthelloGPT residual stream."
    )
    parser.add_argument(
        "--games", required=True,
        help="Path to pre-split train games JSON (output of split_corpus.py)",
    )
    parser.add_argument(
        "--output", default="data/board_probes.pt",
        help="Output path for probes (default: data/board_probes.pt)",
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
        help="Number of games to use (default: all)",
    )
    parser.add_argument(
        "--n-timesteps-per-game", type=int, default=4,
        help="Timesteps sampled per game (default: 4)",
    )
    parser.add_argument(
        "--min-ply", type=int, default=8,
        help="Minimum ply to sample (default: 8)",
    )
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated layer indices to train probes for (default: all layers)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200_000,
        help="Hard cap on total samples collected per layer (default: 200000). "
             "Pass 0 for no limit (unsafe on large corpora).",
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

    max_samples = args.max_samples if args.max_samples != 0 else None
    layers_to_train = [int(l) for l in args.layers.split(",")] if args.layers else list(range(len(model.blocks)))

    # Collect and train one layer at a time to keep memory usage constant.
    # Each pass through the data collects activations for a single layer only.
    probes: dict[int, TrichromeProbe] = {}
    for layer_idx in layers_to_train:
        print(f"\nLayer {layer_idx}: collecting activations "
              f"({len(sampled_games)} games × {args.n_timesteps_per_game} timesteps)...")
        rng_layer = random.Random(args.seed)  # same sample each pass
        with MemoryMonitor():
            activations_by_layer, board_labels = collect_data(
                model,
                sampled_games,
                n_timesteps_per_game=args.n_timesteps_per_game,
                min_ply=args.min_ply,
                device=device,
                rng=rng_layer,
                layers=[layer_idx],
                max_samples=max_samples,
            )
            probe = train_probe_for_layer(
                activations_by_layer[layer_idx],
                board_labels,
                layer_idx,
            )
        probes[layer_idx] = probe
        del activations_by_layer, board_labels

    output_path = Path(args.output)
    save_probes(probes, output_path)
    print(f"\nSaved {len(probes)} probes to '{output_path}'")


if __name__ == "__main__":
    main()
