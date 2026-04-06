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
    # quick test, single layer set
    uv run python scripts/gpt2_sentiment/sentiment_intervention.py --n-examples 20

    # multiple layer sets compared in one run
    uv run python scripts/gpt2_sentiment/sentiment_intervention.py \\
        --n-examples 100 --layer-sets 6,10 7,11 8,12 8,10 9,11

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

# Template prompts where the natural completion is a sentiment-laden whole word.
# Each entry: (prompt, negative_completions, positive_completions)
# Completions include the leading space as GPT-2 tokenizes " great" ≠ "great".
TEMPLATE_PROMPTS: list[tuple[str, list[str], list[str]]] = [
    ("The movie was absolutely",   [" terrible", " awful", " horrible"],        [" wonderful", " amazing", " brilliant"]),
    ("This film is",               [" boring", " dreadful", " awful"],           [" brilliant", " wonderful", " great"]),
    ("The acting was",             [" terrible", " awful", " horrible"],         [" brilliant", " excellent", " outstanding"]),
    ("I found this film",          [" boring", " dull", " disappointing"],       [" captivating", " wonderful", " brilliant"]),
    ("Overall the movie was",      [" disappointing", " terrible", " awful"],    [" excellent", " wonderful", " brilliant"]),
    ("The story was",              [" boring", " dreadful", " terrible"],        [" captivating", " wonderful", " brilliant"]),
    ("This movie was a",           [" disaster", " waste", " mess"],             [" masterpiece", " triumph", " gem"]),
    ("The film left me feeling",   [" bored", " depressed", " empty"],           [" inspired", " uplifted", " moved"]),
    ("The direction was",          [" terrible", " awful", " poor"],             [" brilliant", " masterful", " stunning"]),
    ("I would describe this as",   [" boring", " terrible", " awful"],           [" brilliant", " wonderful", " amazing"]),
]


# ---------------------------------------------------------------------------
# GPT-2 wrapper with intervention interface
# ---------------------------------------------------------------------------

class GPT2forIntervention:
    """
    Thin wrapper around HuggingFace GPT2LMHeadModel providing the
    forward_1st_stage / forward_2nd_stage / predict interface used by the
    gradient descent intervention loop.

    probe_layer is NOT stored on the class — it is passed per-call so the
    same model instance can be reused across different layer sets.
    """

    def __init__(self, hf_model):
        self.model = hf_model
        self.transformer = hf_model.transformer
        self.lm_head = hf_model.lm_head

    @torch.no_grad()
    def forward_1st_stage(self, input_ids: torch.Tensor, layer_start: int) -> torch.Tensor:
        """
        Run token + position embedding, then transformer blocks 0..layer_start-1.

        Returns hidden state (1, T, INPUT_DIM) at the input of block layer_start.
        """
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.transformer.drop(
            self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        )
        for block in self.transformer.h[:layer_start]:
            hidden = block(hidden)
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
            hidden = block(hidden)
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

def probe_confidence(
    probe: BatteryProbeClassificationTwoLayer,
    activation: torch.Tensor,
    target_class: int,
) -> float:
    """Return the probe's softmax probability for target_class given activation."""
    with torch.no_grad():
        logits = probe(activation[None, :])[0][0]  # (num_task, probe_class)
        prob = torch.softmax(logits[0], dim=-1)[target_class].item()
    return float(prob)


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
) -> tuple[torch.Tensor, float, float]:
    """
    Replicate Li's multi-layer gradient descent intervention for GPT-2.

    Intervenes at layer_start, propagates through each subsequent layer,
    re-intervening at each one.

    Returns
    -------
    probs_after       : softmax probabilities (vocab_size,)
    conf_before       : probe P(positive) at layer_start before intervention
    conf_after        : probe P(positive) at layer_start after intervention
    """
    last_pos = seq_length - 1

    with torch.no_grad():
        whole_mid_act = model.forward_1st_stage(input_ids, layer_start)  # (1, T, INPUT_DIM)

    # First intervention (before block layer_start)
    mid_act = whole_mid_act[0, last_pos]
    conf_before = probe_confidence(probes[layer_start], mid_act, flip_to)
    new_mid_act = gpt2_intervene(
        probes[layer_start], mid_act, labels_current,
        flip_position, flip_to, lr, steps, reg_strength,
    )
    conf_after = probe_confidence(probes[layer_start], new_mid_act, flip_to)
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
    return torch.softmax(last_logits, dim=-1), conf_before, conf_after


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def get_probs_original(
    hf_model,
    input_ids: torch.Tensor,
    seq_length: int,
) -> torch.Tensor:
    """Standard forward pass — returns softmax probs at last position (vocab_size,)."""
    with torch.no_grad():
        outputs = hf_model(input_ids)
    last_logits = outputs.logits[0, seq_length - 1, :]
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
# Top-5 diagnostic
# ---------------------------------------------------------------------------

def top_tokens_str(
    probs: torch.Tensor,
    tokenizer,
    vocab_labels: dict[int, str],
    n: int = 5,
) -> str:
    """Return a one-line string showing the top-n tokens with their VADER label."""
    top_ids = probs.argsort(descending=True)[:n].tolist()
    parts = []
    for token_id in top_ids:
        token_str = repr(tokenizer.decode([token_id]))
        label = vocab_labels.get(token_id, "neutral")[0].upper()  # P/N/U
        prob = probs[token_id].item()
        parts.append(f"{token_str}({label},{prob:.3f})")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Persistence rollout
# ---------------------------------------------------------------------------

def measure_persistence(
    hf_model,
    input_ids: torch.Tensor,
    probs_before: torch.Tensor,
    probs_after: torch.Tensor,
    n_steps: int = 5,
) -> list[float]:
    """
    Compare two greedy rollouts starting from the intervention point:
      - Original path:   appends top-1(probs_before), then greedy
      - Intervened path: appends top-1(probs_after),  then greedy

    Returns TV distance between the two distributions at each subsequent step.

    Note: once we move past the intervention point with a standard forward pass,
    the hidden-state perturbation is gone — any remaining divergence is purely
    due to different token choices propagating through context. Rising TV means
    the paths are compounding; falling TV means they converge back.
    """
    original_next = int(probs_before.argmax())
    intervened_next = int(probs_after.argmax())

    original_ids = torch.cat(
        [input_ids, torch.tensor([[original_next]], device=input_ids.device)], dim=1
    )
    intervened_ids = torch.cat(
        [input_ids, torch.tensor([[intervened_next]], device=input_ids.device)], dim=1
    )

    tv_distances = []
    for _ in range(n_steps):
        if original_ids.shape[1] > 1023 or intervened_ids.shape[1] > 1023:
            break

        probs_orig = get_probs_original(hf_model, original_ids, original_ids.shape[1])
        probs_int  = get_probs_original(hf_model, intervened_ids, intervened_ids.shape[1])

        tv_distances.append(tv_distance(probs_orig, probs_int))

        # Each path advances greedily from its own distribution
        next_orig = int(probs_orig.argmax())
        next_int  = int(probs_int.argmax())

        original_ids   = torch.cat(
            [original_ids,   torch.tensor([[next_orig]], device=input_ids.device)], dim=1
        )
        intervened_ids = torch.cat(
            [intervened_ids, torch.tensor([[next_int]],  device=input_ids.device)], dim=1
        )

    return tv_distances


# ---------------------------------------------------------------------------
# Model and probe loading
# ---------------------------------------------------------------------------

def load_gpt2(device: torch.device):
    from transformers import GPT2LMHeadModel
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model = hf_model.to(device).eval()
    return hf_model


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


def load_vocab_labels(path: Path) -> tuple[dict[int, str], set[int], set[int]]:
    """Load VADER-labeled vocab; return (full_dict, positive_ids, negative_ids)."""
    with open(path) as f:
        raw = json.load(f)
    full_dict = {int(token_id): label for token_id, label in raw.items()}
    positive_token_ids = {k for k, v in full_dict.items() if v == "positive"}
    negative_token_ids = {k for k, v in full_dict.items() if v == "negative"}
    return full_dict, positive_token_ids, negative_token_ids


# ---------------------------------------------------------------------------
# Per-layer-set experiment
# ---------------------------------------------------------------------------

def run_layer_set(
    hf_model,
    model: GPT2forIntervention,
    probes: dict[int, BatteryProbeClassificationTwoLayer],
    tokenizer,
    negative_examples: list[str],
    vocab_labels: dict[int, str],
    positive_token_ids: set[int],
    negative_token_ids: set[int],
    layer_start: int,
    layer_end: int,
    lr: float,
    steps: int,
    reg_strength: float,
    n_rollout: int,
    device: torch.device,
    show_diagnostic: bool = False,
) -> dict:
    """Run intervention for one layer set, return result dict."""
    rank_positive_before_list: list[float] = []
    rank_positive_after_list: list[float] = []
    rank_negative_before_list: list[float] = []
    rank_negative_after_list: list[float] = []
    tv_distance_list: list[float] = []
    tv_persistence_steps: list[list[float]] = []
    probe_conf_before_list: list[float] = []
    probe_conf_after_list: list[float] = []

    diagnostic_shown = False

    for example_index, sentence in enumerate(negative_examples):
        print(f"  [{layer_start},{layer_end}) Example {example_index + 1}/{len(negative_examples)}", end="\r")

        encoding = tokenizer(sentence, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        seq_length = input_ids.shape[1]
        if seq_length > 1020:
            input_ids = input_ids[:, :1020]
            seq_length = 1020

        labels_current = torch.tensor([SENTIMENT_NEGATIVE], dtype=torch.long, device=device)

        probs_before = get_probs_original(hf_model, input_ids, seq_length)

        probs_after, conf_before, conf_after = gpt2_full_intervention(
            model=model,
            probes=probes,
            input_ids=input_ids,
            seq_length=seq_length,
            labels_current=labels_current,
            flip_position=0,
            flip_to=SENTIMENT_POSITIVE,
            layer_start=layer_start,
            layer_end=layer_end,
            lr=lr,
            steps=steps,
            reg_strength=reg_strength,
        )

        probe_conf_before_list.append(conf_before)
        probe_conf_after_list.append(conf_after)
        rank_positive_before_list.append(mean_rank_of_tokens(probs_before, positive_token_ids))
        rank_positive_after_list.append(mean_rank_of_tokens(probs_after,  positive_token_ids))
        rank_negative_before_list.append(mean_rank_of_tokens(probs_before, negative_token_ids))
        rank_negative_after_list.append(mean_rank_of_tokens(probs_after,  negative_token_ids))
        tv_distance_list.append(tv_distance(probs_before, probs_after))

        if n_rollout > 0:
            tv_persistence_steps.append(
                measure_persistence(hf_model, input_ids, probs_before, probs_after, n_rollout)
            )

        if show_diagnostic and not diagnostic_shown:
            diagnostic_shown = True
            print(f"\n  [Diagnostic — first example, layers {layer_start}–{layer_end - 1}]")
            print(f"  Sentence: {sentence[:80]!r}")
            print(f"  Before: {top_tokens_str(probs_before, tokenizer, vocab_labels)}")
            print(f"  After:  {top_tokens_str(probs_after,  tokenizer, vocab_labels)}")
            print()

    print(f"  [{layer_start},{layer_end}) Done ({len(negative_examples)} examples)       ")

    return {
        "layer_start": layer_start,
        "layer_end": layer_end,
        "rank_pos_before": float(np.mean(rank_positive_before_list)),
        "rank_pos_after":  float(np.mean(rank_positive_after_list)),
        "rank_neg_before": float(np.mean(rank_negative_before_list)),
        "rank_neg_after":  float(np.mean(rank_negative_after_list)),
        "mean_tv":          float(np.mean(tv_distance_list)),
        "tv_persistence":   tv_persistence_steps,
        "probe_conf_before": float(np.mean(probe_conf_before_list)),
        "probe_conf_after":  float(np.mean(probe_conf_after_list)),
    }


# ---------------------------------------------------------------------------
# Template prompt experiment
# ---------------------------------------------------------------------------

def run_sst2_stem_prompts(
    hf_model,
    model: GPT2forIntervention,
    probes: dict[int, BatteryProbeClassificationTwoLayer],
    tokenizer,
    negative_examples: list[str],
    positive_token_ids: set[int],
    layer_start: int,
    layer_end: int,
    lr: float,
    steps: int,
    reg_strength: float,
    device: torch.device,
    max_prompts: int = 20,
    neg_threshold: float = 0.4,
) -> None:
    """
    Build prompts from negative SST-2 sentences by stripping the last word.

    Only uses sentences where the probe classifies the stem as negative
    (P(pos) < neg_threshold). The last word of the sentence is the expected
    negative completion — we measure whether it becomes less probable after
    intervention, and whether VADER-positive tokens become more probable.
    """
    print(f"\n{'=' * 70}")
    print(f"SST-2 STEM PROMPTS  (layers {layer_start}–{layer_end - 1}, P(pos) threshold < {neg_threshold})")
    print(f"{'=' * 70}")
    print(f"{'Stem':<34} {'Last word':<14} {'P+bef':>5} {'Rk−bef':>7} {'Rk−aft':>7} {'Rk+bef':>7} {'Rk+aft':>7}")
    print(f"{'-'*34} {'-'*14} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    rank_neg_before_list: list[float] = []
    rank_neg_after_list:  list[float] = []
    rank_pos_before_list: list[float] = []
    rank_pos_after_list:  list[float] = []
    prompts_run = 0

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    for sentence in negative_examples:
        if prompts_run >= max_prompts:
            break

        words = sentence.rstrip().split()
        if len(words) < 3:
            continue

        # Find the last word with VADER compound < -0.1, skipping punctuation
        split_index = None
        for word_index in range(len(words) - 1, 0, -1):
            word = words[word_index]
            if not any(char.isalpha() for char in word):
                continue
            if vader.polarity_scores(word)["compound"] < -0.1:
                split_index = word_index
                break

        if split_index is None:
            continue

        stem = " ".join(words[:split_index])
        last_word_str = " " + words[split_index]

        encoding = tokenizer(stem, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        seq_length = input_ids.shape[1]

        # Check probe confidence on stem — skip if not negative enough
        with torch.no_grad():
            hidden = model.forward_1st_stage(input_ids, layer_start)
        conf_before = probe_confidence(probes[layer_start], hidden[0, seq_length - 1], SENTIMENT_POSITIVE)
        if conf_before >= neg_threshold:
            continue

        last_word_ids = {tokenizer.encode(last_word_str)[0]}

        labels_current = torch.tensor([SENTIMENT_NEGATIVE], dtype=torch.long, device=device)
        probs_before = get_probs_original(hf_model, input_ids, seq_length)
        probs_after, _, conf_after = gpt2_full_intervention(
            model=model, probes=probes, input_ids=input_ids, seq_length=seq_length,
            labels_current=labels_current, flip_position=0, flip_to=SENTIMENT_POSITIVE,
            layer_start=layer_start, layer_end=layer_end,
            lr=lr, steps=steps, reg_strength=reg_strength,
        )

        rank_neg_before = mean_rank_of_tokens(probs_before, last_word_ids)
        rank_neg_after  = mean_rank_of_tokens(probs_after,  last_word_ids)
        rank_pos_before = mean_rank_of_tokens(probs_before, positive_token_ids)
        rank_pos_after  = mean_rank_of_tokens(probs_after,  positive_token_ids)

        rank_neg_before_list.append(rank_neg_before)
        rank_neg_after_list.append(rank_neg_after)
        rank_pos_before_list.append(rank_pos_before)
        rank_pos_after_list.append(rank_pos_after)
        prompts_run += 1

        print(
            f"{stem[:34]:<34} {words[split_index][:14]:<14} {conf_before:.3f} "
            f"{rank_neg_before:>7.0f} {rank_neg_after:>7.0f} "
            f"{rank_pos_before:>7.0f} {rank_pos_after:>7.0f}"
        )

    if not rank_neg_before_list:
        print(f"  No prompts passed the P(pos) < {neg_threshold} filter. Try a higher threshold.")
        return

    print(
        f"\n{'MEAN':<55}"
        f"{np.mean(rank_neg_before_list):>7.0f} {np.mean(rank_neg_after_list):>7.0f} "
        f"{np.mean(rank_pos_before_list):>7.0f} {np.mean(rank_pos_after_list):>7.0f}"
    )
    print(f"\nRk− = rank of last word of original sentence (the negative completion).")
    print(f"Rk+ = mean rank of all VADER-positive tokens in vocabulary.")
    print(f"Good result: Rk− rises (last word less probable), Rk+ falls (positive tokens more probable).")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_layer_sets(layer_sets_raw: list[str]) -> list[tuple[int, int]]:
    result = []
    for item in layer_sets_raw:
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"--layer-sets items must be 'start,end', got: {item!r}")
        result.append((int(parts[0]), int(parts[1])))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-2 sentiment steering via probe-based gradient descent intervention."
    )
    parser.add_argument("--n-examples",   type=int,   default=100,
                        help="Number of negative SST-2 examples (default: 100)")
    parser.add_argument("--layer-sets",   type=str,   nargs="+", default=["6,10"],
                        help="Layer ranges to try, e.g. --layer-sets 6,10 7,11 8,12 (default: 6,10)")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--steps",        type=int,   default=1000)
    parser.add_argument("--reg-strength", type=float, default=0.2)
    parser.add_argument("--n-rollout",    type=int,   default=5,
                        help="Rollout steps for persistence measurement (default: 5)")
    parser.add_argument("--use-templates", action="store_true",
                        help="Also run template-prompt experiment with known sentiment completions")
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


def print_results(results: list[dict], n_rollout: int) -> None:
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Layers':<10} {'P(pos) before':>14} {'P(pos) after':>13} {'P(pos) Δ':>9} {'TV':>8}")
    print(f"{'-'*10} {'-'*14} {'-'*13} {'-'*9} {'-'*8}")
    for result in results:
        layer_str = f"{result['layer_start']}–{result['layer_end'] - 1}"
        conf_b = result["probe_conf_before"]
        conf_a = result["probe_conf_after"]
        tv = result["mean_tv"]
        print(f"{layer_str:<10} {conf_b:>14.4f} {conf_a:>13.4f} {conf_a - conf_b:>+9.4f} {tv:>8.4f}")

    print()
    for result in results:
        layer_str = f"{result['layer_start']}–{result['layer_end'] - 1}"
        persistence = result["tv_persistence"]
        if not persistence:
            continue
        max_steps = min(n_rollout, min(len(steps) for steps in persistence))
        if max_steps == 0:
            continue
        step_means = []
        for step_index in range(max_steps):
            step_tvs = [steps[step_index] for steps in persistence if step_index < len(steps)]
            step_means.append(float(np.mean(step_tvs)))
        print(f"  Layers {layer_str} persistence: " + "  ".join(f"t+{i+1}={v:.3f}" for i, v in enumerate(step_means)))

    print()
    print("P(pos) = probe's P(positive) at layer_start. Should rise toward 1.0 after intervention.")
    print("TV = total variation distance between pre/post output distributions.")
    print("Persistence: TV between greedy rollouts at each step after intervention.")
    print("Rising TV = token choices diverged (not persistent hidden-state effect).")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    layer_sets = parse_layer_sets(args.layer_sets)
    all_layers_needed = sorted({layer for start, end in layer_sets for layer in range(start, end)})
    print(f"Layer sets: {layer_sets}")
    print(f"Probes needed: {all_layers_needed}")

    if not VOCAB_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{VOCAB_LABELS_PATH} not found. Run label_vocab_sentiment.py first."
        )
    vocab_labels, positive_token_ids, negative_token_ids = load_vocab_labels(VOCAB_LABELS_PATH)
    print(f"Vocab: {len(positive_token_ids):,} positive, {len(negative_token_ids):,} negative tokens")

    print("Loading GPT-2...")
    hf_model = load_gpt2(device)
    model = GPT2forIntervention(hf_model)

    print(f"Loading probes for layers {all_layers_needed}...")
    probes = load_probes(all_layers_needed, device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2")
    all_negative_examples = [
        example["sentence"]
        for example in dataset["validation"]
        if example["label"] == SENTIMENT_NEGATIVE
    ]
    negative_examples = all_negative_examples[: args.n_examples]
    print(f"  {len(negative_examples)} negative examples (intervention), {len(all_negative_examples)} total\n")

    all_results = []
    for set_index, (layer_start, layer_end) in enumerate(layer_sets):
        result = run_layer_set(
            hf_model=hf_model,
            model=model,
            probes=probes,
            tokenizer=tokenizer,
            negative_examples=negative_examples,
            vocab_labels=vocab_labels,
            positive_token_ids=positive_token_ids,
            negative_token_ids=negative_token_ids,
            layer_start=layer_start,
            layer_end=layer_end,
            lr=args.lr,
            steps=args.steps,
            reg_strength=args.reg_strength,
            n_rollout=args.n_rollout,
            device=device,
            show_diagnostic=True,
        )
        all_results.append(result)

    print_results(all_results, args.n_rollout)

    if args.use_templates:
        first_start, first_end = layer_sets[0]
        run_sst2_stem_prompts(
            hf_model=hf_model,
            model=model,
            probes=probes,
            tokenizer=tokenizer,
            negative_examples=all_negative_examples,
            positive_token_ids=positive_token_ids,
            layer_start=first_start,
            layer_end=first_end,
            lr=args.lr,
            steps=args.steps,
            reg_strength=args.reg_strength,
            device=device,
        )


if __name__ == "__main__":
    main()
