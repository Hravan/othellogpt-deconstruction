"""
scripts/negation_finetune.py

Fine-tune an instruct model on negation examples up to depth N, then evaluate
accuracy at all depths to test whether the fix generalises.

Hypothesis: fine-tuning on depths 0..N reduces errors at those depths but leaves
depth N+1 and N+2 unchanged — evidence of surface pattern learning, not rule
internalization.

Training data: negation_depth_0 through negation_depth_N groups from ss_pairs.json.
  Each group has 3 phrasings at exactly the same negation depth.
  All phrasings are used for training.

Held-out evaluation: the last 20 capital facts across all depth categories.
  (Same 20 facts are held out from every depth, so no capital-level leakage.)

Three planned experiments:
  --train-depths 0,1,2   → test depths 3,4  (trained on no/one/double negation)
  --train-depths 0,1,2,3 → test depths 4,5
  --train-depths 0,1,2,3,4 → test depths 5,6

Usage
-----
    uv run python scripts/negation_finetune.py --train-depths 0,1,2
    uv run python scripts/negation_finetune.py \\
        --train-depths 0,1,2,3 \\
        --model Qwen/Qwen2-7B-Instruct \\
        --epochs 5 \\
        --output-dir ckpts/negation_finetuned_0123
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
    Trainer,
    TrainingArguments,
)

SYSTEM_PROMPT = (
    "Answer the following yes/no question with a single word: either 'Yes' or 'No'. "
    "Do not add any explanation."
)

# Correct answer for each negation depth for expected="yes" (correct capital) groups.
# Even depths → yes (¬²ᵏP = P), odd depths → no (¬²ᵏ⁺¹P = ¬P).
CORRECT_ANSWER_AT_DEPTH = {
    0: "yes", 1: "no", 2: "yes", 3: "no", 4: "yes", 5: "no", 6: "yes",
}

DEPTH_LABELS = [
    "depth-0 (P)",
    "depth-1 (¬P)",
    "depth-2 (¬²P)",
    "depth-3 (¬³P)",
    "depth-4 (¬⁴P)",
    "depth-5 (¬⁵P)",
    "depth-6 (¬⁶P)",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NegationDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def build_training_examples(
    pairs_path: str,
    tokenizer,
    train_depths: list[int],
    train_size: int = 80,
) -> tuple[list[dict], list[dict]]:
    """
    Build tokenized training examples from negation_depth_N groups.

    Returns (training_examples, held_out_groups).

    Training groups: first train_size groups from each negation_depth_N category
    for N in train_depths. All 3 phrasings per group are used.

    Held-out groups: the remaining groups (last 100-train_size per category)
    across ALL depth categories (0..6), for evaluation.

    The same capital facts are held out from every depth category, so there is
    no capital-level leakage between train and test.
    """
    with open(pairs_path, encoding="utf-8") as pairs_file:
        all_groups = json.load(pairs_file)

    # Collect negation_depth_N groups (0..6) from the pairs file
    depth_groups: dict[int, list[dict]] = {depth: [] for depth in range(7)}
    for group in all_groups:
        category = group["category"]
        if category.startswith("negation_depth_"):
            try:
                depth = int(category.split("_")[-1])
                if depth in depth_groups:
                    depth_groups[depth].append(group)
            except ValueError:
                pass

    for depth, depth_group_list in depth_groups.items():
        print(f"  negation_depth_{depth}: {len(depth_group_list)} groups")

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

    training_examples: list[dict] = []
    held_out_groups: list[dict] = []

    for depth in range(7):
        groups_at_depth = depth_groups[depth]
        train_groups  = groups_at_depth[:train_size]
        held_out_part = groups_at_depth[train_size:]
        held_out_groups.extend(held_out_part)

        if depth not in train_depths:
            continue

        for group in train_groups:
            expected        = group["expected"]
            answer_token_id = yes_token_id if expected == "yes" else no_token_id

            for question in group["questions"]:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                encoding = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                )
                input_ids      = encoding["input_ids"][0]
                attention_mask = encoding["attention_mask"][0]

                answer_tensor              = torch.tensor([answer_token_id])
                input_ids_with_answer      = torch.cat([input_ids, answer_tensor])
                attention_mask_with_answer = torch.cat(
                    [attention_mask, torch.ones(1, dtype=torch.long)]
                )

                labels        = torch.full_like(input_ids_with_answer, fill_value=-100)
                labels[-1]    = answer_token_id

                training_examples.append({
                    "input_ids":      input_ids_with_answer,
                    "attention_mask": attention_mask_with_answer,
                    "labels":         labels,
                })

    print(f"Built {len(training_examples)} training examples "
          f"from depths {train_depths} ({train_size} groups each)")
    print(f"Held-out: {len(held_out_groups)} groups across all depths")
    return training_examples, held_out_groups


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    max_length = max(example["input_ids"].shape[0] for example in batch)

    padded_input_ids       = []
    padded_attention_masks = []
    padded_labels          = []

    for example in batch:
        sequence_length = example["input_ids"].shape[0]
        padding_length  = max_length - sequence_length

        padded_input_ids.append(
            torch.cat([example["input_ids"],
                       torch.zeros(padding_length, dtype=torch.long)])
        )
        padded_attention_masks.append(
            torch.cat([example["attention_mask"],
                       torch.zeros(padding_length, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([example["labels"],
                       torch.full((padding_length,), -100, dtype=torch.long)])
        )

    return {
        "input_ids":      torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels":         torch.stack(padded_labels),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def report_per_depth_accuracy(results_path: str, train_depths: list[int]) -> None:
    """
    For each negation_depth_N category in results_path, compute accuracy
    and report whether each depth was in the training set or held out.

    A model that has internalized the negation rule should achieve high accuracy
    on held-out depths. A model that learned surface patterns will fail at
    depths not seen during training.
    """
    with open(results_path, encoding="utf-8") as results_file:
        results = json.load(results_file)

    print()
    print("=" * 65)
    print(f"Per-depth accuracy (train depths: {train_depths})")
    print("=" * 65)
    print(f"  {'Depth':<18} {'Expected':<10} {'Status':<8} {'Accuracy':>8}  (correct/total)")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*8}")

    for depth in range(7):
        depth_results = [
            r for r in results
            if r["category"] == f"negation_depth_{depth}"
        ]
        if not depth_results:
            continue

        expected_answer = CORRECT_ANSWER_AT_DEPTH[depth]
        status = "TRAIN" if depth in train_depths else "TEST"

        num_correct = 0
        num_total   = 0
        for result in depth_results:
            for answer in result["metrics"]["answers"]:
                num_total += 1
                if answer.strip().lower() == expected_answer:
                    num_correct += 1

        accuracy = num_correct / num_total if num_total > 0 else 0.0
        label    = DEPTH_LABELS[depth] if depth < len(DEPTH_LABELS) else f"depth-{depth}"
        print(f"  {label:<18} {expected_answer:<10} {status:<8} {accuracy:>8.3f}  ({num_correct}/{num_total})")

    print()


def run_evaluation(
    model_path: str,
    held_out_groups: list[dict],
    output_dir: str,
    train_depths: list[int],
) -> None:
    """
    Save held-out groups and evaluate the fine-tuned model via hf_ss_test.py.
    Reports SS/CR per negation depth category plus per-question accuracy.
    """
    held_out_path = Path(output_dir) / "held_out_pairs.json"
    with open(held_out_path, "w", encoding="utf-8") as held_out_file:
        json.dump(held_out_groups, held_out_file, indent=2, ensure_ascii=False)

    output_path = Path(output_dir) / "eval_results_held_out.json"
    cmd = [
        sys.executable, "scripts/hf_ss_test.py",
        "--model",    model_path,
        "--pairs",    str(held_out_path),
        "--instruct",
        "--output",   str(output_path),
    ]
    print(f"\nRunning evaluation ({len(held_out_groups)} groups): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    report_per_depth_accuracy(str(output_path), train_depths)
    print(f"Held-out results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on negation depths 0..N, test generalisation to N+1 and N+2."
    )
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct",
                        help="Base model (default: Qwen/Qwen2-1.5B-Instruct)")
    parser.add_argument("--pairs", default="data/ss_pairs.json",
                        help="ss_pairs.json path (default: data/ss_pairs.json)")
    parser.add_argument("--train-depths", default="0,1,2",
                        help="Comma-separated depth indices to train on (default: 0,1,2)")
    parser.add_argument("--output-dir", default="ckpts/negation_finetuned",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--train-size", type=int, default=80,
                        help="Groups per depth used for training; rest are held out (default: 80)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load base model in 8-bit quantization")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after training")
    parser.add_argument("--report", default=None,
                        help="Skip training; just report accuracy from this results JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_depths = [int(depth_str) for depth_str in args.train_depths.split(",")]

    if args.report:
        report_per_depth_accuracy(args.report, train_depths)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:       {device}")
    print(f"Model:        {args.model}")
    print(f"Train depths: {train_depths}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading groups...")
    training_examples, held_out_groups = build_training_examples(
        pairs_path=args.pairs,
        tokenizer=tokenizer,
        train_depths=train_depths,
        train_size=args.train_size,
    )
    dataset = NegationDataset(training_examples)

    print(f"\nLoading {args.model}...")
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model = model.to(device)

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=16,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )

    print(f"\nTraining on {len(dataset)} examples for {args.epochs} epochs...")
    trainer.train()

    final_model_path = str(Path(args.output_dir) / "final")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nModel saved to {final_model_path}")

    if not args.no_eval:
        run_evaluation(
            model_path=final_model_path,
            held_out_groups=held_out_groups,
            output_dir=args.output_dir,
            train_depths=train_depths,
        )


if __name__ == "__main__":
    main()
