"""
scripts/train_board_predictor.py

Train a tiny board state predictor for MUSE representation alignment.

The model uses the same GPT backbone width as OthelloGPT (512 dim, 8 heads)
but with only 2 transformer layers and a board state prediction head
(64 cells × 3 classes: EMPTY=0, BLACK=1, WHITE=2) instead of a next-move head.

This is the null-hypothesis baseline for the MUSE alignment attack on Yuan et al.
(2025, Section 4): a model that is explicitly supervised to predict board state
(not to play Othello) should achieve the same high alignment scores as OthelloGPT,
showing that high alignment does not imply a world model.

Input data: synthetic game pickle files from data/sequence_data/othello_synthetic/.
Each pickle file contains a list of games; each game is a list of board positions
(integers 0-63).

Usage
-----
    uv run python scripts/train_board_predictor.py \
        --n-files 5 \
        --output ckpts/board_predictor.pt \
        --epochs 3
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "mingpt"))
from mingpt.model import GPTConfig, Block

from othellogpt_deconstruction.core.board import (
    start_board, flipped_by, legal_moves, EMPTY, BLACK, WHITE,
)
from othellogpt_deconstruction.core.tokenizer import stoi, BLOCK_SIZE, PAD_ID, VOCAB_SIZE

SYNTHETIC_DATA_DIR = Path("data/sequence_data/othello_synthetic")


# ---------------------------------------------------------------------------
# Board replay (non-validating, for positions-as-integers input)
# ---------------------------------------------------------------------------

def replay_positions(board_positions: list[int]) -> list[tuple]:
    """
    Replay a sequence of board positions (0-63) without legality checks.
    Returns a list of (board, player) tuples — one per move, after applying it.
    """
    board = start_board()
    player = BLACK
    states = []
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
        states.append(board.copy())
    return states


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BoardStatePredictor(nn.Module):
    """
    2-layer GPT-style transformer trained to predict board state from move sequences.

    Same width as OthelloGPT (512 dim, 8 heads) so MUSE Procrustes alignment
    can map between the two representation spaces (requires matching dimensions).

    The representation used for MUSE is the output of ln_f (the final layer norm),
    shape (B, T, 512) — the same extraction point used for OthelloGPT.
    """

    N_EMBD  = 512
    N_HEAD  = 8
    DROPOUT = 0.1

    def __init__(self, n_layer: int = 4):
        super().__init__()
        self.n_layer = n_layer
        config = GPTConfig(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            n_layer=n_layer,
            n_head=self.N_HEAD,
            n_embd=self.N_EMBD,
            embd_pdrop=self.DROPOUT,
            resid_pdrop=self.DROPOUT,
            attn_pdrop=self.DROPOUT,
        )
        self.tok_emb    = nn.Embedding(VOCAB_SIZE, self.N_EMBD)
        self.pos_emb    = nn.Parameter(torch.zeros(1, BLOCK_SIZE, self.N_EMBD))
        self.drop       = nn.Dropout(self.DROPOUT)
        self.blocks     = nn.ModuleList([Block(config) for _ in range(n_layer)])
        self.ln_f       = nn.LayerNorm(self.N_EMBD)
        self.board_head = nn.Linear(self.N_EMBD, 64 * 3)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"BoardStatePredictor ({n_layer} layers): {n_params:,} parameters ({n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        token_ids : (B, T) long tensor

        Returns
        -------
        logits : (B, T, 64, 3) board state logits at each position
        hidden : (B, T, 512) representations from ln_f (used for MUSE)
        """
        B, T = token_ids.shape
        x = self.drop(self.tok_emb(token_ids) + self.pos_emb[:, :T, :])
        for block in self.blocks:
            x = block(x)
        hidden = self.ln_f(x)
        logits = self.board_head(hidden).view(B, T, 64, 3)
        return logits, hidden


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BoardStateDataset(Dataset):
    """
    Each sample is one complete game represented as:
        token_ids  : (BLOCK_SIZE,) padded sequence of move tokens
        board_labels: (T, 64) board state at each move, values in {0, 1, 2}
        seq_len    : number of actual moves T

    If shuffle=True, the move sequence is randomly permuted before replay,
    destroying all Othello structure while preserving token distribution.
    """

    def __init__(self, games: list[list[int]], shuffle: bool = False):
        self.games = games
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, index: int):
        game = self.games[index]
        if self.shuffle:
            game = game[:]
            random.shuffle(game)
        board_states = replay_positions(game)

        # Truncate to BLOCK_SIZE (games are 60 moves, BLOCK_SIZE is 59)
        seq_len      = min(len(game), BLOCK_SIZE)
        tokens       = [stoi[pos] for pos in game[:seq_len]]
        padded       = tokens + [PAD_ID] * (BLOCK_SIZE - seq_len)
        board_states = board_states[:seq_len]

        token_ids    = torch.tensor(padded, dtype=torch.long)
        board_labels = torch.from_numpy(
            np.stack(board_states).astype(np.int64)
        )  # (seq_len, 64)

        return token_ids, board_labels, seq_len


def collate(batch):
    token_ids_list, board_labels_list, seq_lens = zip(*batch)
    token_ids = torch.stack(token_ids_list)  # (B, BLOCK_SIZE)
    seq_lens  = list(seq_lens)
    # Pad board labels to max seq len in batch
    max_len = max(seq_lens)
    board_labels = torch.zeros(len(batch), max_len, 64, dtype=torch.long)
    for i, (labels, length) in enumerate(zip(board_labels_list, seq_lens)):
        board_labels[i, :length] = labels
    return token_ids, board_labels, seq_lens


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: BoardStatePredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_index, (token_ids, board_labels, seq_lens) in enumerate(dataloader):
        token_ids    = token_ids.to(device)     # (B, BLOCK_SIZE)
        board_labels = board_labels.to(device)  # (B, max_T, 64)

        logits, _ = model(token_ids)  # (B, BLOCK_SIZE, 64, 3)

        # Compute loss only over actual (non-padded) positions
        batch_size = token_ids.shape[0]
        loss = torch.tensor(0.0, device=device)
        for sample_index in range(batch_size):
            seq_len = seq_lens[sample_index]
            sample_logits = logits[sample_index, :seq_len]          # (T, 64, 3)
            sample_labels = board_labels[sample_index, :seq_len]    # (T, 64)
            loss = loss + nn.functional.cross_entropy(
                sample_logits.reshape(-1, 3),
                sample_labels.reshape(-1),
            )
        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_index % 100 == 0:
            print(
                f"  Epoch {epoch}  batch {batch_index}/{len(dataloader)}"
                f"  loss={loss.item():.4f}",
                end="\r",
            )

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: BoardStatePredictor,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_correct = 0
    total_cells   = 0

    for token_ids, board_labels, seq_lens in dataloader:
        token_ids    = token_ids.to(device)
        board_labels = board_labels.to(device)

        logits, _ = model(token_ids)  # (B, BLOCK_SIZE, 64, 3)

        for sample_index in range(token_ids.shape[0]):
            seq_len       = seq_lens[sample_index]
            sample_logits = logits[sample_index, :seq_len]       # (T, 64, 3)
            sample_labels = board_labels[sample_index, :seq_len] # (T, 64)
            predictions   = sample_logits.argmax(dim=-1)
            total_correct += (predictions == sample_labels).sum().item()
            total_cells   += seq_len * 64

    return total_correct / total_cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_synthetic_games(n_files: int) -> list[list[int]]:
    pickle_files = sorted(SYNTHETIC_DATA_DIR.glob("*.pickle"))[:n_files]
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in {SYNTHETIC_DATA_DIR}")
    games = []
    for path in pickle_files:
        with open(path, "rb") as f:
            games.extend(pickle.load(f))
        print(f"  Loaded {path.name}  ({len(games):,} games total)")
    return games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny board state predictor for MUSE alignment."
    )
    parser.add_argument("--n-files",    type=int,   default=2,
                        help="Number of synthetic pickle files to load (default: 2 = 200K games)")
    parser.add_argument("--output",     default="ckpts/board_predictor.pt",
                        help="Output checkpoint path (default: ckpts/board_predictor.pt)")
    parser.add_argument("--epochs",     type=int,   default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int,   default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--val-frac",   type=float, default=0.02,
                        help="Fraction of games held out for validation (default: 0.02)")
    parser.add_argument("--n-layers",   type=int,   default=4,
                        help="Number of transformer layers (default: 4)")
    parser.add_argument("--shuffle",    action="store_true",
                        help="Shuffle move sequences to destroy Othello structure (baseline)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.n_files} synthetic pickle file(s)...")
    all_games = load_synthetic_games(args.n_files)

    n_val   = max(1000, int(len(all_games) * args.val_frac))
    n_train = len(all_games) - n_val
    train_games = all_games[:n_train]
    val_games   = all_games[n_train:]
    print(f"  {n_train:,} training games, {n_val:,} validation games")

    if args.shuffle:
        print("  Shuffle mode: move sequences will be randomly permuted (structure-free baseline)")
    train_dataset = BoardStateDataset(train_games, shuffle=args.shuffle)
    val_dataset   = BoardStateDataset(val_games,   shuffle=args.shuffle)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, collate_fn=collate,
    )

    model     = BoardStatePredictor(n_layer=args.n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_path  = Path(args.output)
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        accuracy   = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  Saved checkpoint to {output_path}")

    print(f"\nBest validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
