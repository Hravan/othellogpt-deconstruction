"""
scripts/hf_ss_test.py

Test a HuggingFace causal language model on the Semantic Sensitivity (SS) metric.

For each equivalence group in data/ss_pairs.json, the script runs the model
on each semantically identical question and computes:

  SS  — mean pairwise TV distance between next-token output distributions
  SSS — 1 - SS
  CR  — fraction of pairs where top-1 answer disagrees

A model with stable semantic representations should produce nearly identical
distributions for all questions in a group, giving SS ≈ 0.

The model sees each question formatted as:

    {question}
    Answer:

and we record the full next-token probability distribution at that position.
TV distance is computed over the complete vocabulary (exact, not approximated).

Usage
-----
    uv run python scripts/hf_ss_test.py
    uv run python scripts/hf_ss_test.py --model gpt2-medium
    uv run python scripts/hf_ss_test.py --n-groups 100 --category capital_word_order
    uv run python scripts/hf_ss_test.py --output data/ss_results_gpt2.json
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

YES_TOKENS = {"Yes", "YES", "yes", " Yes", " YES", " yes", "Y", "y", "Ye"}
NO_TOKENS  = {"No",  "NO",  "no",  " No",  " NO",  " no",  "N", "n"}

ANSWER_PREFIX = "\nAnswer:"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device).eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Distribution extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_next_token_distribution(
    tokenizer,
    model,
    question: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Return the full next-token probability distribution (vocab_size,) after
    the prompt "{question}\nAnswer:".
    """
    prompt = question + ANSWER_PREFIX
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logits = model(input_ids).logits[0, -1, :]  # (vocab_size,)
    return torch.softmax(logits, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def tv_distance(dist_a: torch.Tensor, dist_b: torch.Tensor) -> float:
    return 0.5 * (dist_a - dist_b).abs().sum().item()


def top1_answer(dist: torch.Tensor, tokenizer, answer_type: str) -> str:
    top_token = tokenizer.decode([dist.argmax().item()]).strip()
    if answer_type == "word":
        return top_token
    if top_token in YES_TOKENS:
        return "yes"
    if top_token in NO_TOKENS:
        return "no"
    # Fall back to checking summed yes/no probability mass
    vocab = tokenizer.get_vocab()
    prob_yes = sum(dist[token_id].item() for token, token_id in vocab.items() if token in YES_TOKENS)
    prob_no  = sum(dist[token_id].item() for token, token_id in vocab.items() if token in NO_TOKENS)
    if prob_yes >= prob_no and prob_yes >= 0.1:
        return "yes"
    if prob_no > prob_yes and prob_no >= 0.1:
        return "no"
    return "other"


def group_metrics(
    distributions: list[torch.Tensor],
    tokenizer,
    answer_type: str,
) -> dict:
    n = len(distributions)
    answers = [top1_answer(dist, tokenizer, answer_type) for dist in distributions]
    if n < 2:
        return {"ss": 0.0, "cr": 0.0, "answers": answers}

    tv_distances = []
    contradictions = []
    for dist_a, dist_b in combinations(distributions, 2):
        tv_distances.append(tv_distance(dist_a, dist_b))
        contradictions.append(
            top1_answer(dist_a, tokenizer, answer_type) !=
            top1_answer(dist_b, tokenizer, answer_type)
        )

    return {
        "ss":      sum(tv_distances) / len(tv_distances),
        "cr":      sum(contradictions) / len(contradictions),
        "answers": answers,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[dict], model_name: str) -> None:
    if not results:
        print("No results.")
        return

    all_ss = [r["metrics"]["ss"] for r in results]
    all_cr = [r["metrics"]["cr"] for r in results]
    col = 30

    print()
    print("=" * 70)
    print(f"Semantic Sensitivity — {model_name}  (n={len(results)} groups)")
    print("=" * 70)
    print(f"\n  {'SS  (mean TV distance)':.<{col}} {sum(all_ss) / len(all_ss):.4f}")
    print(f"  {'SSS (1 - SS)':.<{col}} {1 - sum(all_ss) / len(all_ss):.4f}")
    print(f"  {'CR  (contradiction rate)':.<{col}} {sum(all_cr) / len(all_cr):.4f}")

    by_category: dict[str, list] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result["metrics"])

    print(f"\n  {'Category':<30}  {'n':>5}  {'SS':>6}  {'SSS':>6}  {'CR':>6}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}")
    for category in sorted(by_category):
        category_metrics = by_category[category]
        ss_cat = sum(m["ss"] for m in category_metrics) / len(category_metrics)
        cr_cat = sum(m["cr"] for m in category_metrics) / len(category_metrics)
        print(f"  {category:<30}  {len(category_metrics):>5}  {ss_cat:>6.4f}  {1-ss_cat:>6.4f}  {cr_cat:>6.4f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a HuggingFace causal LM on the Semantic Sensitivity metric."
    )
    parser.add_argument("--pairs",    default="data/ss_pairs.json",
                        help="Input pairs JSON (default: data/ss_pairs.json)")
    parser.add_argument("--model",    default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--n-groups", type=int, default=None,
                        help="Number of groups to test (default: all)")
    parser.add_argument("--category", default=None,
                        help="Test only this category (default: all)")
    parser.add_argument("--output",   default=None,
                        help="Save full results JSON to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.pairs, encoding="utf-8") as pairs_file:
        all_groups: list[dict] = json.load(pairs_file)

    if args.category:
        all_groups = [g for g in all_groups if g["category"] == args.category]
        print(f"Filtered to category '{args.category}': {len(all_groups)} groups")

    if args.n_groups and len(all_groups) > args.n_groups:
        all_groups = all_groups[:args.n_groups]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.model}...")
    tokenizer, model = load_model(args.model, device)

    total_questions = sum(len(g["questions"]) for g in all_groups)
    print(f"Testing {len(all_groups)} groups ({total_questions} questions)...")

    results: list[dict] = []
    for group_index, group in enumerate(all_groups):
        if (group_index + 1) % 50 == 0 or group_index == 0:
            print(f"  Group {group_index + 1}/{len(all_groups)}", end="\r")

        answer_type = group.get("answer_type", "yes_no")
        distributions = [
            get_next_token_distribution(tokenizer, model, question, device)
            for question in group["questions"]
        ]
        metrics = group_metrics(distributions, tokenizer, answer_type)
        results.append({
            "id":            group["id"],
            "category":      group["category"],
            "answer_type":   answer_type,
            "questions":     group["questions"],
            "expected":      group.get("expected"),
            "metrics":       metrics,
        })

    print(f"  Done ({len(results)} groups)              ")
    print_report(results, args.model)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
