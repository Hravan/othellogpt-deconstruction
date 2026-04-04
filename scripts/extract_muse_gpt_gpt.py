"""
scripts/extract_muse_gpt_gpt.py

Extract MUSE embeddings from two OthelloGPT checkpoints (e.g. synthetic vs
championship) over the same game positions. This gives the OthelloGPT↔OthelloGPT
alignment ceiling to compare against the board-predictor↔OthelloGPT result.

Usage
-----
    uv run python scripts/extract_muse_gpt_gpt.py \\
        --src-ckpt ckpts/gpt_synthetic.ckpt \\
        --tgt-ckpt ckpts/gpt_championship.ckpt \\
        --n-games 5000 \\
        --output-dir data/muse_gpt_gpt
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

SYNTHETIC_DATA_DIR = Path("data/sequence_data/othello_synthetic")
N_EMBD = 512


def load_othello_gpt(checkpoint_path: str, device: torch.device) -> GPTforProbing:
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


def build_token_batch(
    games: list[list[int]],
    positions: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
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
    src_model: GPTforProbing,
    tgt_model: GPTforProbing,
    games: list[list[int]],
    batch_size: int,
    device: torch.device,
    shuffle_tgt: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    all_games: list[list[int]] = []
    all_steps: list[int] = []
    for game in games:
        n_steps = min(len(game), BLOCK_SIZE)
        for step in range(n_steps):
            all_games.append(game)
            all_steps.append(step)

    # For shuffled target: permute each game once, consistently across steps
    if shuffle_tgt:
        shuffled_game_cache: dict[int, list[int]] = {}
        all_shuffled_games: list[list[int]] = []
        for game_index, game in enumerate(all_games):
            game_id = id(game)
            if game_id not in shuffled_game_cache:
                shuffled = game[:]
                random.shuffle(shuffled)
                shuffled_game_cache[game_id] = shuffled
            all_shuffled_games.append(shuffled_game_cache[game_id])
    else:
        all_shuffled_games = all_games

    n_total = len(all_steps)
    src_embeddings = np.zeros((n_total, N_EMBD), dtype=np.float32)
    tgt_embeddings = np.zeros((n_total, N_EMBD), dtype=np.float32)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_games         = all_games[start:end]
        batch_shuffled_games = all_shuffled_games[start:end]
        batch_steps = all_steps[start:end]

        token_ids,         seq_lens  = build_token_batch(batch_games,         batch_steps, device)
        token_ids_shuffled, _        = build_token_batch(batch_shuffled_games, batch_steps, device)

        hidden_src = src_model(token_ids)           # (B, BLOCK_SIZE, 512)
        hidden_tgt = tgt_model(token_ids_shuffled)  # (B, BLOCK_SIZE, 512)

        for sample_index, seq_len in enumerate(seq_lens):
            position = seq_len - 1
            src_embeddings[start + sample_index] = hidden_src[sample_index, position].cpu().float().numpy()
            tgt_embeddings[start + sample_index] = hidden_tgt[sample_index, position].cpu().float().numpy()

        if start % (batch_size * 20) == 0:
            print(f"  Extracted {end}/{n_total} positions", end="\r")

    print(f"  Extracted {n_total} positions          ")
    return src_embeddings, tgt_embeddings


def save_muse_embeddings(embeddings: np.ndarray, position_ids: list[str], output_path: Path) -> None:
    n, d = embeddings.shape
    assert len(position_ids) == n
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{n} {d}\n")
        for position_id, vector in zip(position_ids, embeddings):
            vector_str = " ".join(f"{value:.6f}" for value in vector)
            f.write(f"{position_id} {vector_str}\n")
    print(f"Saved {n} embeddings ({d}d) to {output_path}")


def save_muse_dict(position_ids: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for position_id in position_ids:
            f.write(f"{position_id} {position_id}\n")
    print(f"Saved {len(position_ids)} dictionary pairs to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MUSE embeddings from two OthelloGPT checkpoints."
    )
    parser.add_argument("--src-ckpt",        default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--tgt-ckpt",        default="ckpts/gpt_synthetic.ckpt")
    parser.add_argument("--data-file-index", type=int, default=3)
    parser.add_argument("--n-games",         type=int, default=5000)
    parser.add_argument("--batch-size",      type=int, default=256)
    parser.add_argument("--train-frac",      type=float, default=0.8)
    parser.add_argument("--shuffle-tgt",     action="store_true",
                        help="Feed shuffled sequences to the target model (order-sensitivity test)")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output-dir",      default="data/muse_gpt_gpt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pickle_files = sorted(SYNTHETIC_DATA_DIR.glob("*.pickle"))
    pickle_path = pickle_files[args.data_file_index]
    print(f"Loading games from {pickle_path.name}...")
    with open(pickle_path, "rb") as f:
        all_games = pickle.load(f)
    games = all_games[: args.n_games]
    print(f"  Using {len(games)} games")

    print(f"Loading source model from {args.src_ckpt}...")
    src_model = load_othello_gpt(args.src_ckpt, device)

    print(f"Loading target model from {args.tgt_ckpt}...")
    tgt_model = load_othello_gpt(args.tgt_ckpt, device)

    if args.shuffle_tgt:
        print("  Shuffle-tgt mode: target model receives shuffled sequences")
    print("Extracting representations...")
    src_embeddings, tgt_embeddings = extract_representations(
        src_model, tgt_model, games, args.batch_size, device,
        shuffle_tgt=args.shuffle_tgt,
    )

    n_positions = src_embeddings.shape[0]
    position_ids = [f"pos_{i:06d}" for i in range(n_positions)]
    print(f"Total positions: {n_positions}")

    indices = list(range(n_positions))
    random.shuffle(indices)
    n_train = int(n_positions * args.train_frac)
    train_ids = [position_ids[i] for i in sorted(indices[:n_train])]
    test_ids  = [position_ids[i] for i in sorted(indices[n_train:])]

    output_dir = Path(args.output_dir)
    save_muse_embeddings(src_embeddings, position_ids, output_dir / "src.txt")
    save_muse_embeddings(tgt_embeddings, position_ids, output_dir / "tgt.txt")
    save_muse_dict(train_ids, output_dir / "dict_train.txt")
    save_muse_dict(test_ids,  output_dir / "dict_test.txt")

    print(f"\nDone. Train: {len(train_ids)} pairs, Test: {len(test_ids)} pairs")
    print(f"\nRun MUSE alignment:")
    print(f"  python muse/supervised.py \\")
    print(f"    --src_lang src --tgt_lang tgt \\")
    print(f"    --src_emb {output_dir}/src.txt \\")
    print(f"    --tgt_emb {output_dir}/tgt.txt \\")
    print(f"    --n_refinement 5 \\")
    print(f"    --dico_train {output_dir}/dict_train.txt \\")
    print(f"    --dico_eval {output_dir}/dict_test.txt \\")
    print(f"    --emb_dim 512 --exp_name debug --exp_id gpt_gpt_run")


if __name__ == "__main__":
    main()
