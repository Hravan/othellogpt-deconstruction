"""
scripts/gpt2_sentiment/sentiment_intervention.py

Apply Li et al.'s probe-based gradient descent intervention to GPT-2 for
sentiment steering. Demonstrates that the technique is general activation-space
steering, not evidence of a world model.

Experiment: take SST-2 negative examples, intervene to flip sentiment probe
from negative (0) → positive (1), and measure whether the output distribution
shifts toward positive-sentiment tokens.

Usage
-----
    # quick test
    uv run python scripts/gpt2_sentiment/sentiment_intervention.py --n-examples 20

    # full run
    uv run python scripts/gpt2_sentiment/sentiment_intervention.py --n-examples 500

Extra dependencies (install once):
    pip install transformers datasets vaderSentiment

Probes must be trained first:
    uv run python scripts/gpt2_sentiment/train_sentiment_probes.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mingpt"))
from mingpt.probe_model import BatteryProbeClassificationTwoLayer

PROBE_DIR = Path("ckpts/gpt2_sentiment")
VOCAB_LABELS_PATH = Path("data/gpt2_sentiment_vocab.json")
INPUT_DIM = 768
MID_DIM = 128
N_LAYERS = 12

SENTIMENT_NEGATIVE = 0
SENTIMENT_POSITIVE = 1


# ---------------------------------------------------------------------------
# GPT-2 wrapper with intervention interface
# ---------------------------------------------------------------------------

class GPT2forIntervention:
    """
    Thin wrapper around HuggingFace GPT2LMHeadModel providing the
    forward_1st_stage / forward_2nd_stage / predict interface used by the
    gradient descent intervention loop.
    """

    def __init__(self, hf_model, probe_layer: int):
        self.model = hf_model
        self.probe_layer = probe_layer
        self.transformer = hf_model.transformer
        self.lm_head = hf_model.lm_head

    @torch.no_grad()
    def forward_1st_stage(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run token + position embedding, then transformer blocks 0..probe_layer-1.

        Returns hidden state (1, T, INPUT_DIM) at the input of block probe_layer.
        """
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.transformer.drop(
            self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        )
        for block in self.transformer.h[: self.probe_layer]:
            hidden = block(hidden)[0]
        return hidden  # (1, T, INPUT_DIM)

    @torch.no_grad()
    def forward_2nd_stage(
        self,
        hidden: torch.Tensor,
        layer_start: int,
        layer_end: int,
    ) -> tuple[torch.Tensor, None]:
        """
        Run transformer blocks layer_start..layer_end-1.

        Returns (hidden, None) to match the mingpt signature.
        """
        for block in self.transformer.h[layer_start:layer_end]:
            hidden = block(hidden)[0]
        return hidden, None

    @torch.no_grad()
    def predict(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Apply ln_f + lm_head.

        Returns (logits, None) where logits is (1, T, vocab_size).
        """
        hidden_normed = self.transformer.ln_f(hidden)
        logits = self.lm_head(hidden_normed)
        return logits, None


# ---------------------------------------------------------------------------
# Gradient descent intervention (standalone for num_task=1)
# ---------------------------------------------------------------------------

def gpt2_intervene(
    probe: BatteryProbeClassificationTwoLayer,
    activation: torch.Tensor,
    labels_current: torch.Tensor,
    flip_position: int,
    flip_to: int,
    lr: float = 1e-3,
    steps: int = 1000,
    reg_strength: float = 0.2,
    num_task: int = 1,
) -> torch.Tensor:
    """
    Gradient descent intervention on a single activation vector.

    Mirrors li_intervene() from scripts/li_intervention_test.py, but
    parameterized for num_task=1 (binary sentiment instead of 64 board cells).

    Parameters
    ----------
    probe           : BatteryProbeClassificationTwoLayer (num_task=1, probe_class=2)
    activation      : (INPUT_DIM,) float tensor
    labels_current  : (num_task,) long tensor
    flip_position   : task index to flip (0 for the single sentiment task)
    flip_to         : target class (SENTIMENT_POSITIVE = 1)
    num_task        : number of probe tasks (1 for sentiment)
    """
    new_activation = torch.tensor(
        activation.detach().cpu().numpy(), dtype=torch.float32,
    ).to(activation.device)
    new_activation.requires_grad = True

    optimizer = torch.optim.Adam([new_activation], lr=lr)

    target_labels = labels_current.clone()
    target_labels[flip_position] = flip_to

    weight_mask = reg_strength * torch.ones(num_task, device=activation.device)
    weight_mask[flip_position] = 1.0

    for _ in range(steps):
        optimizer.zero_grad()
        logits = probe(new_activation[None, :])[0][0]  # (num_task, probe_class)
        loss = F.cross_entropy(logits, target_labels, reduction="none")  # (num_task,)
        loss = torch.mean(weight_mask * loss)
        loss.backward()
        optimizer.step()

    return new_activation.detach()


# ---------------------------------------------------------------------------
# Full multi-layer intervention
# ---------------------------------------------------------------------------

def gpt2_full_intervention(
    model: GPT2forIntervention,
    probes: dict[int, BatteryProbeClassificationTwoLayer],
    input_ids: torch.Tensor,
    seq_length: int,
    labels_current: torch.Tensor,
    flip_position: int,
    flip_to: int,
    layer_start: int,
    layer_end: int,
    lr: float = 1e-3,
    steps: int = 1000,
    reg_strength: float = 0.2,
) -> torch.Tensor:
    """
    Replicate Li's multi-layer gradient descent intervention for GPT-2.

    Intervenes at layer_start, propagates through each subsequent layer,
    re-intervening at each one. Returns final softmax probabilities (vocab_size,).
    """
    last_pos = seq_length - 1

    with torch.no_grad():
        whole_mid_act = model.forward_1st_stage(input_ids)  # (1, T, INPUT_DIM)

    # First intervention (before block layer_start)
    mid_act = whole_mid_act[0, last_pos]
    new_mid_act = gpt2_intervene(
        probes[layer_start], mid_act, labels_current,
        flip_position, flip_to, lr, steps, reg_strength,
    )
    whole_mid_act = whole_mid_act.detach().clone()
    whole_mid_act[0, last_pos] = new_mid_act

    # Propagate through remaining layers, re-intervening at each
    for layer in range(layer_start, layer_end - 1):
        with torch.no_grad():
            whole_mid_act = model.forward_2nd_stage(whole_mid_act, layer, layer + 1)[0]

        mid_act = whole_mid_act[0, last_pos]
        new_mid_act = gpt2_intervene(
            probes[layer + 1], mid_act, labels_current,
            flip_position, flip_to, lr, steps, reg_strength,
        )
        whole_mid_act = whole_mid_act.detach().clone()
        whole_mid_act[0, last_pos] = new_mid_act

    with torch.no_grad():
        logits, _ = model.predict(whole_mid_act)  # (1, T, vocab_size)

    last_logits = logits[0, last_pos, :].clone()
    return torch.softmax(last_logits, dim=-1)


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def get_probs_original(
    model: GPT2forIntervention,
    input_ids: torch.Tensor,
    seq_length: int,
) -> torch.Tensor:
    """Standard forward pass — returns softmax probs at last position (vocab_size,)."""
    with torch.no_grad():
        whole_hidden = model.forward_1st_stage(input_ids)
        whole_hidden, _ = model.forward_2nd_stage(whole_hidden, model.probe_layer, N_LAYERS)
        logits, _ = model.predict(whole_hidden)
    last_logits = logits[0, seq_length - 1, :]
    return torch.softmax(last_logits, dim=-1)


def mean_rank_of_tokens(probs: torch.Tensor, token_ids: set[int]) -> float:
    """Return the mean 1-based rank of the given token IDs in the probability distribution."""
    if not token_ids:
        return float("nan")
    sorted_indices = probs.argsort(descending=True).tolist()
    rank_map = {token_id: rank + 1 for rank, token_id in enumerate(sorted_indices)}
    return float(np.mean([rank_map[token_id] for token_id in token_ids if token_id in rank_map]))


def tv_distance(probs_a: torch.Tensor, probs_b: torch.Tensor) -> float:
    """Total variation distance between two probability distributions."""
    return float(0.5 * (probs_a - probs_b).abs().sum().item())


# ---------------------------------------------------------------------------
# Persistence rollout
# ---------------------------------------------------------------------------

def measure_persistence(
    model: GPT2forIntervention,
    tokenizer,
    input_ids: torch.Tensor,
    probs_original: torch.Tensor,
    probs_intervened: torch.Tensor,
    n_steps: int = 5,
) -> list[float]:
    """
    Force the top-1 pre-intervention token, then roll out n_steps tokens
    from the modified hidden state (post-intervention) vs the original.

    Returns TV divergence at each rollout step (should decrease — the
    intervention effect fades as new context overwhelms it).
    """
    # Top-1 token from pre-intervention distribution (force same next token)
    forced_token_id = int(probs_original.argmax())
    forced_token_tensor = torch.tensor([[forced_token_id]], device=input_ids.device)

    base_ids = torch.cat([input_ids, forced_token_tensor], dim=1)

    tv_distances = []
    for _ in range(n_steps):
        if base_ids.shape[1] > 1024:  # GPT-2 context limit
            break
        probs_base = get_probs_original(model, base_ids, base_ids.shape[1])

        # Use intervened top-1 as context for this step
        intervened_token_id = int(probs_intervened.argmax())
        intervened_ids = torch.cat(
            [input_ids, torch.tensor([[intervened_token_id]], device=input_ids.device)],
            dim=1,
        )
        probs_int = get_probs_original(model, intervened_ids, intervened_ids.shape[1])

        tv_distances.append(tv_distance(probs_base, probs_int))

        # Advance base_ids by one (greedy)
        next_token = int(probs_base.argmax())
        base_ids = torch.cat(
            [base_ids, torch.tensor([[next_token]], device=input_ids.device)], dim=1
        )
        probs_intervened = probs_int  # update for next step

    return tv_distances


# ---------------------------------------------------------------------------
# Model and probe loading
# ---------------------------------------------------------------------------

def load_gpt2_for_intervention(probe_layer: int, device: torch.device) -> GPT2forIntervention:
    from transformers import GPT2LMHeadModel
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model = hf_model.to(device).eval()
    return GPT2forIntervention(hf_model, probe_layer=probe_layer)


def load_probes(
    layers: list[int],
    device: torch.device,
) -> dict[int, BatteryProbeClassificationTwoLayer]:
    probes = {}
    for layer_index in layers:
        probe = BatteryProbeClassificationTwoLayer(
            device=device,
            probe_class=2,
            num_task=1,
            mid_dim=MID_DIM,
            input_dim=INPUT_DIM,
        )
        checkpoint_path = PROBE_DIR / f"layer{layer_index}" / "checkpoint.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Probe checkpoint not found: {checkpoint_path}\n"
                f"Run train_sentiment_probes.py first."
            )
        probe.load_state_dict(torch.load(checkpoint_path, map_location=device))
        probe.eval()
        probes[layer_index] = probe
    return probes


def load_vocab_labels(path: Path) -> tuple[set[int], set[int]]:
    """Load VADER-labeled vocab, return (positive_token_ids, negative_token_ids)."""
    with open(path) as f:
        raw = json.load(f)
    positive_token_ids = {int(token_id) for token_id, label in raw.items() if label == "positive"}
    negative_token_ids = {int(token_id) for token_id, label in raw.items() if label == "negative"}
    return positive_token_ids, negative_token_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-2 sentiment steering via probe-based gradient descent intervention."
    )
    parser.add_argument("--n-examples",   type=int,   default=100,
                        help="Number of negative SST-2 examples to process (default: 100)")
    parser.add_argument("--layer-start",  type=int,   default=6,
                        help="First intervention layer (default: 6)")
    parser.add_argument("--layer-end",    type=int,   default=10,
                        help="Last intervention layer exclusive (default: 10)")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--steps",        type=int,   default=1000)
    parser.add_argument("--reg-strength", type=float, default=0.2)
    parser.add_argument("--n-rollout",    type=int,   default=5,
                        help="Rollout steps for persistence measurement (default: 5)")
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load vocabulary sentiment labels
    if not VOCAB_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{VOCAB_LABELS_PATH} not found. Run label_vocab_sentiment.py first."
        )
    positive_token_ids, negative_token_ids = load_vocab_labels(VOCAB_LABELS_PATH)
    print(f"Vocab labels: {len(positive_token_ids):,} positive, {len(negative_token_ids):,} negative tokens")

    # Load model and probes
    intervention_layers = list(range(args.layer_start, args.layer_end))
    print(f"Loading GPT-2 (probe_layer={args.layer_start})...")
    model = load_gpt2_for_intervention(probe_layer=args.layer_start, device=device)

    print(f"Loading probes for layers {intervention_layers}...")
    probes = load_probes(intervention_layers, device)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load negative SST-2 validation examples
    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2")
    negative_examples = [
        example["sentence"]
        for example in dataset["validation"]
        if example["label"] == SENTIMENT_NEGATIVE
    ][: args.n_examples]
    print(f"  {len(negative_examples)} negative examples")

    # Accumulators
    rank_positive_before_list: list[float] = []
    rank_positive_after_list: list[float] = []
    rank_negative_before_list: list[float] = []
    rank_negative_after_list: list[float] = []
    tv_distance_list: list[float] = []
    tv_persistence_steps: list[list[float]] = []

    print(f"\nRunning interventions (layers {args.layer_start}–{args.layer_end - 1})...")

    for example_index, sentence in enumerate(negative_examples):
        print(f"  Example {example_index + 1}/{len(negative_examples)}", end="\r")

        encoding = tokenizer(sentence, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        seq_length = input_ids.shape[1]

        # Clamp to GPT-2 context limit
        if seq_length > 1020:
            input_ids = input_ids[:, :1020]
            seq_length = 1020

        # Labels: this is a negative example
        labels_current = torch.tensor([SENTIMENT_NEGATIVE], dtype=torch.long, device=device)

        # Pre-intervention distribution
        probs_before = get_probs_original(model, input_ids, seq_length)

        # Intervention: flip negative → positive
        probs_after = gpt2_full_intervention(
            model=model,
            probes=probes,
            input_ids=input_ids,
            seq_length=seq_length,
            labels_current=labels_current,
            flip_position=0,
            flip_to=SENTIMENT_POSITIVE,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            lr=args.lr,
            steps=args.steps,
            reg_strength=args.reg_strength,
        )

        # Rank metrics
        rank_positive_before_list.append(
            mean_rank_of_tokens(probs_before, positive_token_ids)
        )
        rank_positive_after_list.append(
            mean_rank_of_tokens(probs_after, positive_token_ids)
        )
        rank_negative_before_list.append(
            mean_rank_of_tokens(probs_before, negative_token_ids)
        )
        rank_negative_after_list.append(
            mean_rank_of_tokens(probs_after, negative_token_ids)
        )
        tv_distance_list.append(tv_distance(probs_before, probs_after))

        # Persistence
        if args.n_rollout > 0:
            persistence = measure_persistence(
                model, tokenizer, input_ids, probs_before, probs_after, args.n_rollout,
            )
            tv_persistence_steps.append(persistence)

    print(f"\n  Done ({len(negative_examples)} examples)")

    # Aggregate results
    rank_pos_before = float(np.mean(rank_positive_before_list))
    rank_pos_after  = float(np.mean(rank_positive_after_list))
    rank_neg_before = float(np.mean(rank_negative_before_list))
    rank_neg_after  = float(np.mean(rank_negative_after_list))
    mean_tv         = float(np.mean(tv_distance_list))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Examples processed : {len(negative_examples)}")
    print(f"Layers             : {args.layer_start}–{args.layer_end - 1}")
    print()
    print(f"{'Metric':<40} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"{'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    print(
        f"{'Mean rank of POSITIVE tokens':<40} "
        f"{rank_pos_before:>10.1f} "
        f"{rank_pos_after:>10.1f} "
        f"{rank_pos_after - rank_pos_before:>+10.1f}"
    )
    print(
        f"{'Mean rank of NEGATIVE tokens':<40} "
        f"{rank_neg_before:>10.1f} "
        f"{rank_neg_after:>10.1f} "
        f"{rank_neg_after - rank_neg_before:>+10.1f}"
    )
    print(f"\nMean TV distance (before vs after): {mean_tv:.4f}")

    if tv_persistence_steps:
        max_steps = min(args.n_rollout, min(len(steps) for steps in tv_persistence_steps))
        if max_steps > 0:
            print(f"\nTV distance persistence over {max_steps} rollout steps:")
            for step_index in range(max_steps):
                step_tvs = [
                    steps[step_index]
                    for steps in tv_persistence_steps
                    if step_index < len(steps)
                ]
                print(f"  Step {step_index + 1}: {float(np.mean(step_tvs)):.4f}")

    print("=" * 60)
    print()
    print("Interpretation:")
    print("  Positive tokens moving UP (rank decreasing) = intervention")
    print("  steered distribution toward positive sentiment.")
    print("  Negative tokens moving DOWN (rank increasing) = intervention")
    print("  suppressed negative-sentiment tokens.")
    print("  TV persistence fading = one-shot residual perturbation,")
    print("  not a persistent state change — consistent with activation-space")
    print("  steering, not world model state switching.")


if __name__ == "__main__":
    main()
