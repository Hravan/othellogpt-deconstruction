"""
scripts/semantic_sensitivity_llm_test.py

Test an external LLM on the Semantic Sensitivity (SS) metric.

For each equivalence group in data/ss_pairs.json, the script queries
the model with each semantically identical question and computes:

  SS  — mean pairwise TV distance between output distributions
  SSS — 1 - SS
  CR  — fraction of pairs where top-1 answer disagrees

A model with stable semantic representations should produce nearly
identical distributions for all questions in a group, giving SS ≈ 0.

Requirements
------------
    pip install openai python-dotenv
    export OPENAI_API_KEY=...  (or set in .env)

Usage
-----
    python scripts/semantic_sensitivity_llm_test.py \\
        --model gpt-4o-mini \\
        --n-groups 100 \\
        --output data/ss_results_gpt4o_mini.json

    # Test a specific category only
    python scripts/semantic_sensitivity_llm_test.py \\
        --category capital_word_order

    # Resume from cache (skips already-queried questions)
    python scripts/semantic_sensitivity_llm_test.py \\
        --cache-file data/ss_cache.json

API details
-----------
Uses the OpenAI API with logprobs=True, top_logprobs=20.
TV distance is computed over the top-20 returned tokens.
This gives a lower bound on true TV — conservative but unbiased.

For a yes/no question, the top tokens are almost always "Yes"/"No"
variants, so the top-20 captures essentially all the probability mass.
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai package not found — run: pip install openai")

try:
    from dotenv import load_dotenv
except ImportError:
    raise SystemExit("python-dotenv not found — run: pip install python-dotenv")

load_dotenv()


# ---------------------------------------------------------------------------
# Tokens considered as "Yes" or "No" answers
# ---------------------------------------------------------------------------

YES_TOKENS = {"Yes", "YES", "yes", " Yes", " YES", " yes", "Y", "y", "Ye"}
NO_TOKENS  = {"No",  "NO",  "no",  " No",  " NO",  " no",  "N", "n"}


# ---------------------------------------------------------------------------
# LLM query
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_YES_NO = (
    "Answer the following yes/no question with a single word: either 'Yes' or 'No'. "
    "Do not add any explanation."
)

SYSTEM_PROMPT_WORD = (
    "Answer the following question with a single word or number. "
    "Do not add any explanation."
)


def query_distribution(
    client:       OpenAI,
    model:        str,
    question:     str,
    system_prompt: str = SYSTEM_PROMPT_YES_NO,
    top_logprobs: int = 20,
    max_retries:  int = 5,
) -> dict[str, float]:
    """
    Query the model and return a normalised probability distribution
    over the first generated token.

    Returns a dict mapping token string → probability, normalised
    over the top_logprobs tokens returned by the API.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            break
        except Exception as error:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {error} — waiting {wait}s")
            time.sleep(wait)

    raw_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    probs = {entry.token: math.exp(entry.logprob) for entry in raw_logprobs}

    # Normalise over observed tokens (tail mass is treated as shared)
    total = sum(probs.values())
    if total > 0:
        probs = {token: prob / total for token, prob in probs.items()}

    return probs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def tv_distance(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """TV distance between two token probability distributions."""
    all_tokens = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(token, 0.0) - dist_b.get(token, 0.0))
                     for token in all_tokens)


def prob_yes(dist: dict[str, float]) -> float:
    return sum(dist.get(token, 0.0) for token in YES_TOKENS)


def prob_no(dist: dict[str, float]) -> float:
    return sum(dist.get(token, 0.0) for token in NO_TOKENS)


def top1_answer(dist: dict[str, float], answer_type: str = "yes_no") -> str:
    """
    Return the model's top-1 answer as a string.

    For yes_no groups: returns 'yes', 'no', or 'other'.
    For word groups: returns the highest-probability token (stripped).
    """
    if answer_type == "word":
        if not dist:
            return "other"
        return max(dist, key=dist.get).strip()

    p_yes = prob_yes(dist)
    p_no  = prob_no(dist)
    if p_yes >= p_no and p_yes >= 0.3:
        return "yes"
    if p_no > p_yes and p_no >= 0.3:
        return "no"
    return "other"


def group_metrics(distributions: list[dict[str, float]], answer_type: str = "yes_no") -> dict:
    """
    Compute SS, CR, and binary P(Yes) stats for one equivalence group.

    distributions : one distribution per question in the group
    answer_type   : "yes_no" or "word"
    """
    n = len(distributions)
    answers = [top1_answer(d, answer_type) for d in distributions]
    if n < 2:
        return {"ss": 0.0, "cr": 0.0, "answers": answers,
                "p_yes": [prob_yes(d) for d in distributions]}

    tv_distances = []
    contradictions = []
    for dist_a, dist_b in combinations(distributions, 2):
        tv_distances.append(tv_distance(dist_a, dist_b))
        answer_a = top1_answer(dist_a, answer_type)
        answer_b = top1_answer(dist_b, answer_type)
        contradictions.append(answer_a != answer_b)

    metrics = {
        "ss":      sum(tv_distances) / len(tv_distances),
        "cr":      sum(contradictions) / len(contradictions),
        "answers": answers,
    }
    if answer_type == "yes_no":
        metrics["p_yes"] = [prob_yes(d) for d in distributions]
        metrics["p_no"]  = [prob_no(d) for d in distributions]
    return metrics


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def load_cache(cache_file: Path) -> dict[str, dict]:
    """Load cached query results. Key: (model, question) serialised as JSON."""
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as cache_fp:
            return json.load(cache_fp)
    return {}


def save_cache(cache: dict, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as cache_fp:
        json.dump(cache, cache_fp, ensure_ascii=False)


def cache_key(model: str, question: str) -> str:
    return json.dumps([model, question], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    results: list[dict],
    model:   str,
    col:     int = 30,
) -> None:
    if not results:
        print("No results.")
        return

    all_ss = [r["metrics"]["ss"] for r in results]
    all_cr = [r["metrics"]["cr"] for r in results]

    print()
    print("=" * 70)
    print(f"Semantic Sensitivity — {model}  (n={len(results)} groups)")
    print("=" * 70)
    print(f"\n  {'SS  (mean TV distance)':.<{col}} {sum(all_ss) / len(all_ss):.4f}")
    print(f"  {'SSS (1 - SS)':.<{col}} {1 - sum(all_ss) / len(all_ss):.4f}")
    print(f"  {'CR  (contradiction rate)':.<{col}} {sum(all_cr) / len(all_cr):.4f}")

    # Per category
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
        description="Test an LLM on the Semantic Sensitivity metric."
    )
    parser.add_argument("--pairs",      default="data/ss_pairs.json",
                        help="Input pairs JSON (default: data/ss_pairs.json)")
    parser.add_argument("--model",      default="gpt-4o-mini",
                        help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--n-groups",   type=int, default=None,
                        help="Number of groups to test (default: all)")
    parser.add_argument("--category",   default=None,
                        help="Test only this category (default: all)")
    parser.add_argument("--output",     default=None,
                        help="Save full results JSON to this path")
    parser.add_argument("--cache-file", default="data/ss_query_cache.json",
                        help="Cache file for LLM responses (default: data/ss_query_cache.json)")
    parser.add_argument("--top-logprobs", type=int, default=20,
                        help="Number of logprobs to request (default: 20)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.05,
                        help="Seconds to wait between API calls (default: 0.05)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load pairs
    with open(args.pairs, encoding="utf-8") as pairs_fp:
        all_groups: list[dict] = json.load(pairs_fp)

    # Filter by category
    if args.category:
        all_groups = [g for g in all_groups if g["category"] == args.category]
        print(f"Filtered to category '{args.category}': {len(all_groups)} groups")

    # Sample
    if args.n_groups and len(all_groups) > args.n_groups:
        all_groups = all_groups[: args.n_groups]

    total_questions = sum(len(g["questions"]) for g in all_groups)
    print(f"Testing {len(all_groups)} groups ({total_questions} questions) on {args.model}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cache  = load_cache(Path(args.cache_file))
    results: list[dict] = []
    cache_hits  = 0
    api_calls   = 0

    for group_idx, group in enumerate(all_groups):
        print(f"  Group {group_idx + 1}/{len(all_groups)}", end="\r")

        answer_type   = group.get("answer_type", "yes_no")
        system_prompt = SYSTEM_PROMPT_WORD if answer_type == "word" else SYSTEM_PROMPT_YES_NO

        distributions: list[dict[str, float]] = []
        for question in group["questions"]:
            key = cache_key(args.model, question)
            if key in cache:
                distributions.append(cache[key])
                cache_hits += 1
            else:
                dist = query_distribution(
                    client, args.model, question,
                    system_prompt=system_prompt,
                    top_logprobs=args.top_logprobs,
                )
                cache[key] = dist
                distributions.append(dist)
                api_calls += 1
                if args.rate_limit_delay > 0:
                    time.sleep(args.rate_limit_delay)

        metrics = group_metrics(distributions, answer_type)
        results.append({
            "id":            group["id"],
            "category":      group["category"],
            "answer_type":   answer_type,
            "questions":     group["questions"],
            "expected":      group.get("expected"),
            "distributions": distributions,
            "metrics":       metrics,
        })

    print()
    print(f"API calls: {api_calls}  |  Cache hits: {cache_hits}")

    # Save cache
    save_cache(cache, Path(args.cache_file))

    # Report
    print_report(results, args.model)

    # Save full results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_fp:
            json.dump(results, output_fp, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
