"""
scripts/negation_eval.py

Report per-depth accuracy from a negation fine-tuning evaluation results JSON
(produced by hf_ss_test.py).

Usage
-----
    python scripts/negation_eval.py \\
        --results ckpts/negation_012/eval_results_held_out.json \\
        --train-depths 0,1,2
"""

import argparse
import json

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report per-depth accuracy from a negation evaluation results JSON."
    )
    parser.add_argument("--results", required=True,
                        help="Path to eval_results_held_out.json from hf_ss_test.py")
    parser.add_argument("--train-depths", default="0,1,2",
                        help="Comma-separated depth indices used for training (default: 0,1,2)")
    args = parser.parse_args()

    train_depths = [int(depth_str) for depth_str in args.train_depths.split(",")]

    with open(args.results, encoding="utf-8") as results_file:
        results = json.load(results_file)

    print()
    print("=" * 65)
    print(f"Per-depth accuracy  (train depths: {train_depths})")
    print("=" * 65)
    print(f"  {'Depth':<18} {'Expected':<10} {'Status':<8} {'Accuracy':>8}  (correct/total)")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*8}")

    for depth in range(7):
        depth_results = [r for r in results if r["category"] == f"negation_depth_{depth}"]
        if not depth_results:
            continue

        expected_answer = CORRECT_ANSWER_AT_DEPTH[depth]
        status = "TRAIN" if depth in train_depths else "TEST"

        num_correct = 0
        num_total   = 0
        for result in depth_results:
            for answer in result["metrics"]["answers"]:
                num_total  += 1
                if answer.strip().lower() == expected_answer:
                    num_correct += 1

        accuracy = num_correct / num_total if num_total > 0 else 0.0
        label    = DEPTH_LABELS[depth] if depth < len(DEPTH_LABELS) else f"depth-{depth}"
        print(f"  {label:<18} {expected_answer:<10} {status:<8} {accuracy:>8.3f}  ({num_correct}/{num_total})")

    print()


if __name__ == "__main__":
    main()
