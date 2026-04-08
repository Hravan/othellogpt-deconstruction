"""
scripts/negation_finetune.py

Fine-tune an instruct model on depth-2 double negation examples, then evaluate
SS/CR at all negation depths to test whether the fix generalises.

Hypothesis: fine-tuning on depth-2 (¬¬P ↔ P) reduces CR at depth-2 but leaves
depth-3 and depth-4 unchanged — evidence of surface pattern learning, not rule
internalization.

Training data: first --train-size double_negation groups from ss_pairs.json.
  - questions[0]: depth-1 (direct question)  — NOT used for training by default
  - questions[1]: depth-2 phrasing 1          — trained with correct answer
  - questions[2]: depth-2 phrasing 2          — trained with correct answer

Held-out evaluation: the remaining double_negation and negation_depth groups
(same held-out capital facts, so depth-2 eval uses facts never seen in training).

Train/eval split rationale:
  Both double_negation and negation_depth use the same 100 capital facts.
  Training on the first 80 double_negation groups, evaluating on the last 20
  ensures: (a) depth-2 eval uses new facts, (b) depth-3/4 eval uses new facts.
  This rules out memorization of specific question-answer pairs.

Usage
-----
    # Train on depth-2 only (default), evaluate after
    uv run python scripts/negation_finetune.py

    # Larger model, more epochs
    uv run python scripts/negation_finetune.py \\
        --model Qwen/Qwen2-7B-Instruct \\
        --epochs 5 \\
        --output-dir ckpts/negation_finetuned_7b

    # Also include depth-1 in training (control condition)
    uv run python scripts/negation_finetune.py --include-depth-1

    # Skip evaluation after training
    uv run python scripts/negation_finetune.py --no-eval
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SYSTEM_PROMPT = (
    "Answer the following yes/no question with a single word: either 'Yes' or 'No'. "
    "Do not add any explanation."
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NegationDataset(Dataset):
    """
    Each example is a (prompt, answer_token_id) pair.
    Loss is computed only on the answer token.
    """

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def build_training_examples(
    pairs_path: str,
    tokenizer,
    train_size: int = 80,
    include_depth_1: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Build tokenized training examples from the first train_size double_negation groups.
    Returns (training_examples, held_out_groups).

    held_out_groups contains the remaining double_negation and negation_depth groups
    sharing the same held-out capital facts — used for evaluation after training.

    Each training example contributes:
      - questions[1] and questions[2] (depth-2 phrasings) always
      - questions[0] (depth-1) only if include_depth_1=True

    Labels: -100 for all tokens except the final answer token.
    """
    with open(pairs_path, encoding="utf-8") as pairs_file:
        all_groups = json.load(pairs_file)

    double_negation_groups = [g for g in all_groups if g["category"] == "double_negation"]
    negation_depth_groups  = [g for g in all_groups if g["category"] == "negation_depth"]
    print(f"Found {len(double_negation_groups)} double_negation groups, "
          f"{len(negation_depth_groups)} negation_depth groups")

    train_groups    = double_negation_groups[:train_size]
    held_out_double = double_negation_groups[train_size:]
    # negation_depth groups share the same ordering of capital facts —
    # take the same held-out slice for a fully unseen fact evaluation
    held_out_depth  = negation_depth_groups[train_size:]
    held_out_groups = held_out_double + held_out_depth

    print(f"Train: {len(train_groups)} double_negation groups "
          f"({train_size} facts)")
    print(f"Held-out: {len(held_out_double)} double_negation + "
          f"{len(held_out_depth)} negation_depth groups "
          f"({len(double_negation_groups) - train_size} facts)")

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

    examples = []
    for group in train_groups:
        questions = group["questions"]
        expected  = group["expected"]
        answer_token_id = yes_token_id if expected == "yes" else no_token_id

        training_indices = [1, 2]
        if include_depth_1:
            training_indices = [0, 1, 2]

        for question_index in training_indices:
            question = questions[question_index]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]

            answer_tensor = torch.tensor([answer_token_id])
            input_ids_with_answer = torch.cat([input_ids, answer_tensor])
            attention_mask_with_answer = torch.cat([attention_mask, torch.ones(1, dtype=torch.long)])

            labels = torch.full_like(input_ids_with_answer, fill_value=-100)
            labels[-1] = answer_token_id

            examples.append({
                "input_ids":      input_ids_with_answer,
                "attention_mask": attention_mask_with_answer,
                "labels":         labels,
            })

    print(f"Built {len(examples)} training examples")
    return examples, held_out_groups


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Pad input_ids, attention_mask, and labels to the longest sequence in the batch."""
    max_length = max(example["input_ids"].shape[0] for example in batch)

    padded_input_ids      = []
    padded_attention_masks = []
    padded_labels         = []

    for example in batch:
        sequence_length = example["input_ids"].shape[0]
        padding_length  = max_length - sequence_length

        padded_input_ids.append(
            torch.cat([example["input_ids"], torch.zeros(padding_length, dtype=torch.long)])
        )
        padded_attention_masks.append(
            torch.cat([example["attention_mask"], torch.zeros(padding_length, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([example["labels"], torch.full((padding_length,), -100, dtype=torch.long)])
        )

    return {
        "input_ids":      torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels":         torch.stack(padded_labels),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str,
    held_out_groups: list[dict],
    output_dir: str,
) -> None:
    """
    Evaluate the fine-tuned model on held-out groups only.

    Saves held-out groups to a temporary JSON file and passes it to hf_ss_test.py.
    held_out_groups contains both double_negation and negation_depth groups
    for the same unseen capital facts.
    """
    held_out_path = Path(output_dir) / "held_out_pairs.json"
    with open(held_out_path, "w", encoding="utf-8") as held_out_file:
        json.dump(held_out_groups, held_out_file, indent=2, ensure_ascii=False)

    output_path = Path(output_dir) / "eval_results_held_out.json"
    cmd = [
        sys.executable, "scripts/hf_ss_test.py",
        "--model", model_path,
        "--pairs", str(held_out_path),
        "--instruct",
        "--output", str(output_path),
    ]
    print(f"\nRunning held-out evaluation ({len(held_out_groups)} groups): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nHeld-out results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on depth-2 negation, test generalization to depth-3/4."
    )
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct",
                        help="Base model (default: Qwen/Qwen2-1.5B-Instruct)")
    parser.add_argument("--pairs", default="data/ss_pairs.json",
                        help="ss_pairs.json path (default: data/ss_pairs.json)")
    parser.add_argument("--output-dir", default="ckpts/negation_finetuned",
                        help="Directory to save fine-tuned model (default: ckpts/negation_finetuned)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--train-size", type=int, default=80,
                        help="Number of double_negation groups to train on (default: 80, held-out: remaining 20)")
    parser.add_argument("--include-depth-1", action="store_true",
                        help="Also train on depth-1 questions (control condition)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load base model in 8-bit quantization")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Training depth-2 only: {not args.include_depth_1}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build training data and held-out evaluation set
    examples, held_out_groups = build_training_examples(
        pairs_path=args.pairs,
        tokenizer=tokenizer,
        train_size=args.train_size,
        include_depth_1=args.include_depth_1,
    )
    dataset = NegationDataset(examples)

    # Load model
    print(f"Loading {args.model}...")
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model = model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=16,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )

    print(f"\nTraining on {len(dataset)} examples for {args.epochs} epochs...")
    trainer.train()

    # Save final model
    final_model_path = str(Path(args.output_dir) / "final")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nModel saved to {final_model_path}")

    # Evaluate on held-out groups
    if not args.no_eval:
        run_evaluation(
            model_path=final_model_path,
            held_out_groups=held_out_groups,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
