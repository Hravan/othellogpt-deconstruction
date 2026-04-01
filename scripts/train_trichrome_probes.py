"""
scripts/train_trichrome_probes.py

Train linear probes predicting trichrome color (RED / GREEN / BLUE) per cell
from OthelloGPT residual stream activations.

For each transformer layer we fit one per-cell 3-class linear classifier
(sklearn LogisticRegression, multinomial cross-entropy) using residual stream
activations collected at random timesteps sampled from corpus games.

Usage
-----
    uv run python scripts/train_trichrome_probes.py \\
        --corpus path/to/championship/ \\
        --mode championship \\
        --output data/trichrome_probes.pt

    # Faster (fewer games)
    uv run python scripts/train_trichrome_probes.py \\
        --corpus path/to/championship/ \\
        --n-games 500 \\
        --n-timesteps-per-game 5 \\
        --output data/trichrome_probes.pt
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from othellogpt_deconstruction.core.board import EMPTY
from othellogpt_deconstruction.core.corpus import list_corpus_files, load_file
from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, BLOCK_SIZE, PAD_ID
import othellogpt_deconstruction.core.trichrome as trichrome
from othellogpt_deconstruction.intervention.hooks import collect_activations
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.model.probes import TrichromeProbe, save_probes


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def encode_sequence(sequence: list[str], device: torch.device) -> torch.Tensor:
    tokens = [stoi[alg_to_pos(move)] for move in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


def collect_data(
    model:                 torch.nn.Module,
    games:                 list[list[str]],
    n_timesteps_per_game:  int,
    min_ply:               int,
    device:                torch.device,
    rng:                   random.Random,
) -> tuple[dict[int, list[np.ndarray]], list[np.ndarray]]:
    """
    Collect residual stream activations and trichrome color labels.

    For each game we sample n_timesteps_per_game random plies, run the
    model on the prefix, and record activations at all layers alongside
    the trichrome state at that ply.

    Returns
    -------
    activations_by_layer : dict[layer_idx → list of (d_model,) float arrays]
    trichrome_labels     : list of (64,) int arrays; value is color (0/1/2)
                           or -1 for empty cells
    """
    activations_by_layer: dict[int, list[np.ndarray]] = {}
    trichrome_labels: list[np.ndarray] = []
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
            prefix = game[:ply]

            try:
                tc_board, _ = trichrome.replay(prefix)
            except ValueError:
                continue

            x = encode_sequence(prefix, device)

            with collect_activations(model, len(prefix)) as store:
                with torch.no_grad():
                    model(x)

            # Build per-cell color label array (-1 for empty)
            labels = np.full(64, -1, dtype=np.int8)
            for pos in range(64):
                owner = int(tc_board[pos, 0])
                if owner != EMPTY:
                    labels[pos] = int(tc_board[pos, 1])
            trichrome_labels.append(labels)

            for layer_idx, activation in store.activations.items():
                if layer_idx not in activations_by_layer:
                    activations_by_layer[layer_idx] = []
                activations_by_layer[layer_idx].append(activation.cpu().float().numpy())

            total_samples += 1

        if (game_idx + 1) % 50 == 0:
            print(f"  Processed {game_idx + 1}/{len(games)} games, "
                  f"{total_samples} samples", end="\r")

    print(f"\n  {total_samples} total samples collected.")
    return activations_by_layer, trichrome_labels


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe_for_layer(
    activations:     list[np.ndarray],
    labels:          list[np.ndarray],
    layer_idx:       int,
) -> TrichromeProbe:
    """
    Train per-cell linear probes for one layer.

    For each of the 64 cells, fits a 3-class LogisticRegression predicting
    trichrome color from the residual stream activation at that layer.
    Cells with fewer than 10 occupied samples or only one observed color
    receive zero-weight probes.

    Returns
    -------
    TrichromeProbe with weights (64, 3, d_model) and biases (64, 3).
    """
    activation_matrix = np.stack(activations)    # (n_samples, d_model)
    n_samples, d_model = activation_matrix.shape

    weights_list = []
    biases_list  = []
    n_skipped    = 0
    train_accuracies = []

    for pos in range(64):
        cell_labels = np.array([lab[pos] for lab in labels], dtype=np.int8)
        occupied_mask = cell_labels >= 0

        n_occupied = int(occupied_mask.sum())
        observed_colors = np.unique(cell_labels[occupied_mask])

        if n_occupied < 10 or len(observed_colors) < 2:
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

        # coef_ is (n_classes_seen, d_model) for multiclass, but (1, d_model)
        # for binary — the single row represents the positive class (classes_[1]).
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
        description="Train linear trichrome-color probes on OthelloGPT residual stream."
    )
    parser.add_argument(
        "--corpus", nargs="+", required=True,
        help="Path(s) to corpus file(s) or directory(s)",
    )
    parser.add_argument(
        "--output",
        help="Output path for probes (default: data/trichrome_probes.pt)",
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
        help="Number of games to use for data collection (default: 2000)",
    )
    parser.add_argument(
        "--n-timesteps-per-game", type=int, default=3,
        help="Timesteps sampled per game (default: 3)",
    )
    parser.add_argument(
        "--min-ply", type=int, default=8,
        help="Minimum ply to sample (default: 8)",
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

    print(f"Loading corpus from {args.corpus}...")
    file_paths = list_corpus_files(args.corpus)
    all_games: list[list[str]] = []
    for file_path in file_paths:
        all_games.extend(load_file(file_path))
    print(f"  {len(all_games)} games loaded from {len(file_paths)} file(s).")

    rng = random.Random(args.seed)
    if args.n_games is not None and len(all_games) > args.n_games:
        sampled_games = rng.sample(all_games, args.n_games)
        print(f"  Sampled {args.n_games} games.")
    else:
        sampled_games = all_games

    print(f"\nCollecting activations "
          f"({len(sampled_games)} games × {args.n_timesteps_per_game} timesteps)...")
    activations_by_layer, trichrome_labels = collect_data(
        model,
        sampled_games,
        n_timesteps_per_game=args.n_timesteps_per_game,
        min_ply=args.min_ply,
        device=device,
        rng=rng,
    )

    n_layers = len(activations_by_layer)
    print(f"\nTraining probes for {n_layers} layers...")
    probes: dict[int, TrichromeProbe] = {}
    for layer_idx in sorted(activations_by_layer.keys()):
        probe = train_probe_for_layer(
            activations_by_layer[layer_idx],
            trichrome_labels,
            layer_idx,
        )
        probes[layer_idx] = probe

    output_path = Path(args.output)
    save_probes(probes, output_path)
    print(f"\nSaved {len(probes)} probes to '{output_path}'")


if __name__ == "__main__":
    main()
