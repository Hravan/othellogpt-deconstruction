"""
scripts/gpt2_sentiment/train_sentiment_probes.py

Train sentiment probes on GPT-2 hidden states from SST-2.

For each of the 12 transformer layers, trains a BatteryProbeClassificationTwoLayer
(num_task=1, probe_class=2, input_dim=768, mid_dim=128) to predict positive/negative
sentiment from the last-token hidden state at that layer.

Saves checkpoints to ckpts/gpt2_sentiment/layer{N}/checkpoint.ckpt.

Usage
-----
    uv run python scripts/gpt2_sentiment/train_sentiment_probes.py
    uv run python scripts/gpt2_sentiment/train_sentiment_probes.py --n-examples 1000

Extra dependencies (install once):
    pip install transformers datasets
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mingpt"))
from mingpt.probe_model import BatteryProbeClassificationTwoLayer

CKPT_DIR = Path("ckpts/gpt2_sentiment")
N_LAYERS = 12
INPUT_DIM = 768
MID_DIM = 128
MAX_SEQ_LEN = 128


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    sentences: list[str],
    labels: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Run GPT-2 on all sentences, collect last-token hidden state at each layer.

    Returns
    -------
    layer_acts    : list of N_LAYERS tensors, each (N, INPUT_DIM)
    label_tensor  : (N,) long tensor
    """
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    gpt2 = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    gpt2 = gpt2.to(device).eval()

    all_layer_acts: list[list[torch.Tensor]] = [[] for _ in range(N_LAYERS)]

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            end = min(start + batch_size, len(sentences))
            batch_sentences = sentences[start:end]

            encoding = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding_side="left",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = gpt2(input_ids=input_ids, attention_mask=attention_mask)
            # hidden_states: tuple of (N_LAYERS + 1) tensors, each (B, T, INPUT_DIM)
            # Index 0 = embedding output; index i+1 = output of block i
            seq_lengths = attention_mask.sum(dim=1)  # (B,)

            for layer_index in range(N_LAYERS):
                hidden = outputs.hidden_states[layer_index + 1]  # (B, T, INPUT_DIM)
                for sample_index in range(hidden.shape[0]):
                    last_pos = int(seq_lengths[sample_index]) - 1
                    all_layer_acts[layer_index].append(
                        hidden[sample_index, last_pos].cpu()
                    )

            if start % (batch_size * 10) == 0:
                print(f"  Processed {end}/{len(sentences)}", end="\r")

    print(f"  Processed {len(sentences)}                   ")

    layer_tensors = [torch.stack(acts) for acts in all_layer_acts]
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return layer_tensors, label_tensor


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe_for_layer(
    train_acts: torch.Tensor,
    train_labels: torch.Tensor,
    val_acts: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[BatteryProbeClassificationTwoLayer, float]:
    probe = BatteryProbeClassificationTwoLayer(
        device=device,
        probe_class=2,
        num_task=1,
        mid_dim=MID_DIM,
        input_dim=INPUT_DIM,
    )
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_dataset = TensorDataset(train_acts, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_accuracy = 0.0
    best_state: dict | None = None

    for _ in range(epochs):
        probe.train()
        for acts_batch, labels_batch in train_loader:
            acts_batch = acts_batch.to(device)
            # Probe expects labels shape (B, num_task) = (B, 1)
            labels_batch = labels_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            _, loss = probe(acts_batch, labels_batch)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits, _ = probe(val_acts.to(device))  # (N, 1, 2)
            predictions = val_logits[:, 0, :].argmax(dim=-1)  # (N,)
            accuracy = (predictions == val_labels.to(device)).float().mean().item()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.cpu().clone() for key, value in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe, best_accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 sentiment probes.")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Limit training examples (default: all ~67K)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--layers",     type=int, nargs="+", default=list(range(N_LAYERS)),
                        help="Which layers to train probes for (default: all 12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2")
    train_sentences = list(dataset["train"]["sentence"])
    train_labels_raw = list(dataset["train"]["label"])
    val_sentences = list(dataset["validation"]["sentence"])
    val_labels_raw = list(dataset["validation"]["label"])

    if args.n_examples is not None:
        train_sentences = train_sentences[:args.n_examples]
        train_labels_raw = train_labels_raw[:args.n_examples]

    print(f"  Train: {len(train_sentences)}, Val: {len(val_sentences)}")

    print("Extracting train hidden states...")
    train_layer_acts, train_label_tensor = extract_hidden_states(
        train_sentences, train_labels_raw, args.batch_size, device,
    )

    print("Extracting val hidden states...")
    val_layer_acts, val_label_tensor = extract_hidden_states(
        val_sentences, val_labels_raw, args.batch_size, device,
    )

    print("\nTraining probes:")
    print(f"  {'Layer':>5}  {'Val Acc':>8}")
    print(f"  {'-'*5}  {'-'*8}")
    for layer_index in args.layers:
        probe, accuracy = train_probe_for_layer(
            train_layer_acts[layer_index],
            train_label_tensor,
            val_layer_acts[layer_index],
            val_label_tensor,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        output_dir = CKPT_DIR / f"layer{layer_index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), output_dir / "checkpoint.ckpt")
        print(f"  {layer_index:>5}  {accuracy:.4f}  → {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
