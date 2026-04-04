"""
scripts/extract_muse_embeddings.py

Extract final hidden-layer representations from OthelloGPT and the board state
predictor for the same game positions, and save in MUSE text format for
representation alignment (Yuan et al. 2025, Section 4).

For each game position we extract the ln_f output (shape 512) from both models
and assign it a unique ID. Both embedding files share the same IDs, so the
MUSE dictionary is trivial (pos_00001 → pos_00001, etc.).

Output
------
    data/muse_othello_gpt.txt       OthelloGPT embeddings in MUSE format
    data/muse_board_predictor.txt   Board predictor embeddings in MUSE format
    data/muse_dict_train.txt        80% of position IDs (aligned pairs)
    data/muse_dict_test.txt         20% of position IDs (aligned pairs)

Usage
-----
    uv run python scripts/extract_muse_embeddings.py \\
        --othello-gpt-ckpt ckpts/gpt_synthetic.ckpt \\
        --board-predictor-ckpt ckpts/board_predictor.pt \\
        --n-games 5000
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "mingpt"))
from mingpt.model import GPTConfig, GPTforProbing

from othellogpt_deconstruction.core.tokenizer import stoi, BLOCK_SIZE, PAD_ID, VOCAB_SIZE
from train_board_predictor import BoardStatePredictor

SYNTHETIC_DATA_DIR = Path("data/sequence_data/othello_synthetic")
N_EMBD = 512


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_othello_gpt(checkpoint_path: str, device: torch.device) -> GPTforProbing:
    """Load OthelloGPT as GPTforProbing to extract ln_f hidden states."""
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=8,
        n_head=8,
        n_embd=N_EMBD,
    )
    model = GPTforProbing(config, probe_layer=8, ln=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_board_predictor(checkpoint_path: str, device: torch.device, n_layer: int = 4) -> BoardStatePredictor:
    model = BoardStatePredictor(n_layer=n_layer)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def build_token_batch(
    games: list[list[int]],
    positions: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """
    Build a padded token batch for a list of (game, position) pairs.

    Parameters
    ----------
    games     : list of games (each game is a list of board positions 0-63)
    positions : for each game, the step index (0-based) to extract representation at

    Returns
    -------
    token_ids : (N, BLOCK_SIZE) long tensor
    seq_lens  : list of actual sequence lengths (= position + 1)
    """
    token_ids_list = []
    seq_lens = []
    for game, step in zip(games, positions):
        prefix = game[: step + 1]
        seq_len = min(len(prefix), BLOCK_SIZE)
        tokens = [stoi[pos] for pos in prefix[:seq_len]]
        padded = tokens + [PAD_ID] * (BLOCK_SIZE - seq_len)
        token_ids_list.append(padded)
        seq_lens.append(seq_len)
    token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)
    return token_ids, seq_lens


@torch.no_grad()
def extract_representations(
    othello_gpt: GPTforProbing,
    board_predictor: BoardStatePredictor,
    games: list[list[int]],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract final hidden states (ln_f output) from both models for every
    position in every game.

    Returns
    -------
    gpt_embeddings       : (N_positions, 512) float32
    predictor_embeddings : (N_positions, 512) float32
    """
    # Flatten all (game, step) pairs
    all_games: list[list[int]] = []
    all_steps: list[int] = []
    for game in games:
        n_steps = min(len(game), BLOCK_SIZE)
        for step in range(n_steps):
            all_games.append(game)
            all_steps.append(step)

    n_total = len(all_steps)
    gpt_embeddings       = np.zeros((n_total, N_EMBD), dtype=np.float32)
    predictor_embeddings = np.zeros((n_total, N_EMBD), dtype=np.float32)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_games = all_games[start:end]
        batch_steps = all_steps[start:end]

        token_ids, seq_lens = build_token_batch(batch_games, batch_steps, device)

        # OthelloGPT: GPTforProbing returns (B, T, 512)
        hidden_gpt = othello_gpt(token_ids)  # (B, BLOCK_SIZE, 512)

        # Board predictor: forward returns (logits, hidden), hidden is (B, BLOCK_SIZE, 512)
        _, hidden_pred = board_predictor(token_ids)

        # Extract at the last real token position for each sample
        for sample_index, seq_len in enumerate(seq_lens):
            pos = seq_len - 1
            gpt_embeddings[start + sample_index]       = hidden_gpt[sample_index, pos].cpu().float().numpy()
            predictor_embeddings[start + sample_index] = hidden_pred[sample_index, pos].cpu().float().numpy()

        if start % (batch_size * 20) == 0:
            print(f"  Extracted {end}/{n_total} positions", end="\r")

    print(f"  Extracted {n_total} positions          ")
    return gpt_embeddings, predictor_embeddings


# ---------------------------------------------------------------------------
# MUSE format output
# ---------------------------------------------------------------------------

def save_muse_embeddings(
    embeddings: np.ndarray,
    position_ids: list[str],
    output_path: Path,
) -> None:
    """
    Save embeddings in MUSE text format:
        N D
        id1 v1 v2 ... vD
        id2 v1 v2 ... vD
        ...
    """
    n, d = embeddings.shape
    assert len(position_ids) == n
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{n} {d}\n")
        for position_id, vector in zip(position_ids, embeddings):
            vector_str = " ".join(f"{value:.6f}" for value in vector)
            f.write(f"{position_id} {vector_str}\n")
    print(f"Saved {n} embeddings ({d}d) to {output_path}")


def save_muse_dict(
    position_ids: list[str],
    output_path: Path,
) -> None:
    """
    Save a trivial MUSE dictionary: each position ID maps to itself.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for position_id in position_ids:
            f.write(f"{position_id} {position_id}\n")
    print(f"Saved {len(position_ids)} dictionary pairs to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MUSE-format embeddings from OthelloGPT and board predictor."
    )
    parser.add_argument("--othello-gpt-ckpt",      default="ckpts/gpt_synthetic.ckpt",
                        help="OthelloGPT checkpoint (default: ckpts/gpt_synthetic.ckpt)")
    parser.add_argument("--board-predictor-ckpt",  default="ckpts/board_predictor.pt",
                        help="Board predictor checkpoint (default: ckpts/board_predictor.pt)")
    parser.add_argument("--data-file-index",       type=int, default=3,
                        help="Pickle file index to use (default: 3, held out from training)")
    parser.add_argument("--n-games",               type=int, default=5000,
                        help="Number of games to extract from (default: 5000)")
    parser.add_argument("--batch-size",            type=int, default=256,
                        help="Inference batch size (default: 256)")
    parser.add_argument("--train-frac",            type=float, default=0.8,
                        help="Fraction of positions for MUSE train dict (default: 0.8)")
    parser.add_argument("--n-layers",              type=int, default=4,
                        help="Board predictor layers, must match training (default: 4)")
    parser.add_argument("--shuffle",               action="store_true",
                        help="Shuffle game sequences before extraction (must match training)")
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--output-dir",            default="data/muse",
                        help="Output directory (default: data/muse)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load held-out games
    pickle_files = sorted(SYNTHETIC_DATA_DIR.glob("*.pickle"))
    if args.data_file_index >= len(pickle_files):
        raise ValueError(
            f"data_file_index={args.data_file_index} but only {len(pickle_files)} files available"
        )
    pickle_path = pickle_files[args.data_file_index]
    print(f"Loading games from {pickle_path.name}...")
    with open(pickle_path, "rb") as f:
        all_games = pickle.load(f)

    games = all_games[:args.n_games]
    if args.shuffle:
        print("  Shuffle mode: permuting move sequences to destroy Othello structure")
        for i in range(len(games)):
            games[i] = games[i][:]
            random.shuffle(games[i])
    print(f"  Using {len(games)} games")

    # Load models
    print("Loading OthelloGPT...")
    othello_gpt = load_othello_gpt(args.othello_gpt_ckpt, device)

    print("Loading board predictor...")
    board_predictor = load_board_predictor(args.board_predictor_ckpt, device, n_layer=args.n_layers)

    # Extract representations
    print("Extracting representations...")
    gpt_embeddings, predictor_embeddings = extract_representations(
        othello_gpt, board_predictor, games, args.batch_size, device,
    )

    # Build position IDs
    n_positions = gpt_embeddings.shape[0]
    position_ids = [f"pos_{i:06d}" for i in range(n_positions)]
    print(f"Total positions: {n_positions}")

    # Train/test split
    indices = list(range(n_positions))
    random.shuffle(indices)
    n_train = int(n_positions * args.train_frac)
    train_indices = sorted(indices[:n_train])
    test_indices  = sorted(indices[n_train:])

    train_ids = [position_ids[i] for i in train_indices]
    test_ids  = [position_ids[i] for i in test_indices]

    # Save embeddings
    output_dir = Path(args.output_dir)
    save_muse_embeddings(gpt_embeddings,       position_ids, output_dir / "othello_gpt.txt")
    save_muse_embeddings(predictor_embeddings, position_ids, output_dir / "board_predictor.txt")

    # Save dictionaries
    save_muse_dict(train_ids, output_dir / "dict_train.txt")
    save_muse_dict(test_ids,  output_dir / "dict_test.txt")

    print(f"\nDone. Train: {len(train_ids)} pairs, Test: {len(test_ids)} pairs")
    print(f"\nRun MUSE alignment:")
    print(f"  python muse/supervised.py \\")
    print(f"    --src_lang othello_gpt \\")
    print(f"    --tgt_lang board_predictor \\")
    print(f"    --src_emb {output_dir}/othello_gpt.txt \\")
    print(f"    --tgt_emb {output_dir}/board_predictor.txt \\")
    print(f"    --n_refinement 5 \\")
    print(f"    --dico_train {output_dir}/dict_train.txt \\")
    print(f"    --dico_eval {output_dir}/dict_test.txt")


if __name__ == "__main__":
    main()
