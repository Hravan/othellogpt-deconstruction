"""
src/othellogpt_deconstruction/model/inference.py

OthelloGPT model loading and inference.

This is the only module that imports from mingpt directly.
All other modules interact with the model through this interface.
"""

from pathlib import Path

import torch

from mingpt.model import GPT, GPTConfig

from othellogpt_deconstruction.core.tokenizer import (
    stoi, alg_to_pos, VOCAB_SIZE, BLOCK_SIZE, PAD_ID,
)


# ---------------------------------------------------------------------------
# Model modes
# ---------------------------------------------------------------------------

CHECKPOINT_PATHS: dict[str, str] = {
    "championship": "ckpts/gpt_championship.ckpt",
    "synthetic":    "ckpts/gpt_synthetic.ckpt",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def make_config() -> GPTConfig:
    """Return the GPTConfig used for all OthelloGPT checkpoints."""
    return GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=8,
        n_head=8,
        n_embd=512,
    )


def load_model(
    mode: str = "championship",
    checkpoint_path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> GPT:
    """
    Load an OthelloGPT model.

    Parameters
    ----------
    mode            : "championship", "synthetic", or "random"
    checkpoint_path : explicit path to checkpoint, overrides mode
    device          : torch device (defaults to cuda if available, else cpu)

    Returns
    -------
    GPT model in eval mode on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    mconf = make_config()
    model = GPT(mconf)

    if mode == "random":
        model.apply(model._init_weights)
    else:
        if checkpoint_path is None:
            if mode not in CHECKPOINT_PATHS:
                raise ValueError(
                    f"Unknown mode {mode!r}. "
                    f"Choose from {list(CHECKPOINT_PATHS)} or 'random'."
                )
            checkpoint_path = CHECKPOINT_PATHS[mode]
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_distribution(
    model: GPT,
    sequence: list[str],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Run a move sequence through the model and return the next-move
    probability distribution.

    Parameters
    ----------
    model    : loaded GPT model
    sequence : list of algebraic move strings
    device   : device to run on (inferred from model if None)

    Returns
    -------
    1-D tensor of shape (VOCAB_SIZE,) with softmax probabilities.
    PAD token probability is set to -inf before softmax.
    """
    if device is None:
        device = next(model.parameters()).device

    tokens = [stoi[alg_to_pos(m)] for m in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    x = torch.tensor([padded], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(x)

    last_logits = logits[0, len(tokens) - 1, :].clone()
    last_logits[PAD_ID] = float("-inf")
    return torch.softmax(last_logits, dim=-1)


def get_distributions_batch(
    model: GPT,
    sequences: list[list[str]],
    device: torch.device | None = None,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """
    Compute distributions for a list of sequences efficiently in batches.
    Deduplicates sequences so each unique sequence is run only once.

    Parameters
    ----------
    model      : loaded GPT model
    sequences  : list of move sequence lists
    device     : device to run on
    batch_size : number of sequences per forward pass

    Returns
    -------
    Dict mapping seq_key -> probability tensor of shape (VOCAB_SIZE,).
    """
    from othellogpt_deconstruction.core.tokenizer import seq_key, pad

    if device is None:
        device = next(model.parameters()).device

    # Deduplicate
    unique: dict[str, list[str]] = {}
    for seq in sequences:
        k = seq_key(seq)
        if k not in unique:
            unique[k] = seq

    keys = list(unique.keys())
    seqs = list(unique.values())
    results: dict[str, torch.Tensor] = {}

    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i + batch_size]
        batch_keys = keys[i:i + batch_size]
        seq_lengths = [len(s) for s in batch_seqs]

        # Build padded batch
        batch_tokens = [
            [stoi[alg_to_pos(m)] for m in seq] for seq in batch_seqs
        ]
        padded_batch = [
            tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
            for tokens in batch_tokens
        ]
        x = torch.tensor(padded_batch, dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(x)  # (batch, block_size, vocab_size)

        for j, (key, length) in enumerate(zip(batch_keys, seq_lengths)):
            last_logits = logits[j, length - 1, :].clone()
            last_logits[PAD_ID] = float("-inf")
            results[key] = torch.softmax(last_logits, dim=-1).cpu()

    return results
