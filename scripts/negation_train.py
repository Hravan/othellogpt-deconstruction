"""
scripts/negation_train.py

Fine-tune an instruct model on negation_depth_N groups for N in --train-depths.
Saves the fine-tuned model to --output-dir/final.

Run negation_prepare.py first to generate held_out_pairs.json, then
negation_eval.py after to report per-depth accuracy.

Usage
-----
    python scripts/negation_train.py --train-depths 0,1,2 --output-dir ckpts/negation_012
    python scripts/negation_train.py \\
        --model Qwen/Qwen2-7B-Instruct \\
        --train-depths 0,1,2,3 \\
        --output-dir ckpts/negation_7b_0123
"""

import argparse
import json
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
) -> list[dict]:
    with open(pairs_path, encoding="utf-8") as pairs_file:
        all_groups = json.load(pairs_file)

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

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

    training_examples: list[dict] = []
    for depth in train_depths:
        groups_at_depth = depth_groups[depth][:train_size]
        for group in groups_at_depth:
            answer_token_id = yes_token_id if group["expected"] == "yes" else no_token_id
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
                labels      = torch.full_like(input_ids_with_answer, fill_value=-100)
                labels[-1]  = answer_token_id
                training_examples.append({
                    "input_ids":      input_ids_with_answer,
                    "attention_mask": attention_mask_with_answer,
                    "labels":         labels,
                })

    print(f"Built {len(training_examples)} training examples "
          f"from depths {train_depths} ({train_size} groups × 3 phrasings each)")
    return training_examples


def collate_fn(batch: list[dict]) -> dict:
    max_length = max(example["input_ids"].shape[0] for example in batch)
    padded_input_ids, padded_attention_masks, padded_labels = [], [], []
    for example in batch:
        padding_length = max_length - example["input_ids"].shape[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on negation depths 0..N."
    )
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct",
                        help="Base model (default: Qwen/Qwen2-1.5B-Instruct)")
    parser.add_argument("--pairs", default="data/ss_pairs.json",
                        help="ss_pairs.json path (default: data/ss_pairs.json)")
    parser.add_argument("--train-depths", default="0,1,2",
                        help="Comma-separated depth indices to train on (default: 0,1,2)")
    parser.add_argument("--train-size", type=int, default=80,
                        help="Groups per depth used for training (default: 80)")
    parser.add_argument("--output-dir", default="ckpts/negation_finetuned",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load base model in 8-bit quantization")
    args = parser.parse_args()

    train_depths = [int(depth_str) for depth_str in args.train_depths.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:       {device}")
    print(f"Model:        {args.model}")
    print(f"Train depths: {train_depths}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_examples = build_training_examples(
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
            args.model, quantization_config=quantization_config, device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

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


if __name__ == "__main__":
    main()
