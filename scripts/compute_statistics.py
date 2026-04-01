"""
scripts/compute_statistics.py

Load pre-computed distributions and compute comparison statistics
for every trichrome-differing pair. Outputs a CSV for pandas analysis.

No model or GPU required. Rerun freely when adding new metrics.

Usage
-----
    uv run python scripts/compute_statistics.py \\
        --transpositions data/transpositions.json \\
        --distributions data/distributions.pt \\
        --output data/results.csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from othellogpt_deconstruction.core.tokenizer import seq_key
from othellogpt_deconstruction.core.board import legal_moves_after
from othellogpt_deconstruction.core.trichrome import color_distance
from othellogpt_deconstruction.analysis.metrics import all_metrics, rank_of, top1_info
from othellogpt_deconstruction.analysis.correlations import (
    correlate, print_correlation_table, summarise_correlations,
)


# ---------------------------------------------------------------------------
# Trichrome diff properties
# ---------------------------------------------------------------------------

def diff_properties(diff_cells: list[dict]) -> dict:
    if not diff_cells:
        return {"n_diff_cells": 0, "max_color_dist": 0, "total_color_dist": 0}
    dists = [color_distance(c["color_a"], c["color_b"]) for c in diff_cells]
    return {
        "n_diff_cells":     len(dists),
        "max_color_dist":   max(dists),
        "total_color_dist": sum(dists),
    }


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

COLUMNS = [
    # Pair identity
    "ply",
    "seq_a",
    "seq_b",
    "diff_squares",
    # Trichrome diff properties
    "n_diff_cells",
    "max_color_dist",
    "total_color_dist",
    # Rank-1 info
    "rank1_pos_a",
    "rank1_prob_a",
    "rank1_pos_b",
    "rank1_prob_b",
    "rank_of_a_in_b",
    "rank_of_b_in_a",
    # Full distribution metrics
    "tv_distance_full",
    "js_divergence_full",
    "kl_ab_full",
    "kl_ba_full",
    "spearman_rho_full",
    "rank1_agreement_full",
    # Legal-only metrics
    "tv_distance_legal",
    "js_divergence_legal",
    "kl_ab_legal",
    "kl_ba_legal",
    "spearman_rho_legal",
    "rank1_agreement_legal",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute distribution comparison statistics for transposition pairs."
    )
    parser.add_argument(
        "--transpositions", default="data/transpositions.json",
        help="Path to transpositions JSON (default: data/transpositions.json)",
    )
    parser.add_argument(
        "--distributions", default="data/distributions.pt",
        help="Path to distributions saved by extract_distributions.py "
             "(default: data/distributions.pt)",
    )
    parser.add_argument(
        "--output", default="data/results.csv",
        help="Output CSV path (default: data/results.csv)",
    )
    parser.add_argument(
        "--no-correlations", action="store_true",
        help="Skip correlation analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load
    print(f"Loading transpositions from '{args.transpositions}' ...")
    with open(args.transpositions) as f:
        groups = json.load(f)
    mixed = [g for g in groups if g.get("trichrome_diffs")]
    print(f"  {len(mixed)} groups with trichrome diffs")

    print(f"Loading distributions from '{args.distributions}' ...")
    distributions: dict[str, torch.Tensor] = torch.load(
        args.distributions, weights_only=True,
    )
    print(f"  {len(distributions)} distributions loaded")

    # Compute
    rows: list[dict] = []
    agree_count = 0
    total_pairs = 0
    missing_keys: list[str] = []

    for group in tqdm(mixed, desc="Pairs"):
        t_groups = group["trichrome_groups"]

        for diff in group["trichrome_diffs"]:
            seq_a = t_groups[diff["group_a"]]["sequences"][0]
            seq_b = t_groups[diff["group_b"]]["sequences"][0]

            key_a = seq_key(seq_a)
            key_b = seq_key(seq_b)

            if key_a not in distributions or key_b not in distributions:
                missing_keys.extend([key_a, key_b])
                continue

            probs_a = distributions[key_a]
            probs_b = distributions[key_b]

            legal = legal_moves_after(seq_a)
            metrics = all_metrics(probs_a, probs_b, legal)
            tc_props = diff_properties(diff["differing_cells"])

            # Rank-1 info
            t1_a = top1_info(probs_a, legal)
            t1_b = top1_info(probs_b, legal)
            pos_a = t1_a[0] if t1_a else None
            pos_b = t1_b[0] if t1_b else None

            agree = metrics["rank1_agreement_legal"]
            if agree:
                agree_count += 1
            total_pairs += 1

            rows.append({
                "ply":                group["ply"],
                "seq_a":              " ".join(seq_a),
                "seq_b":              " ".join(seq_b),
                "diff_squares":       " ".join(
                    c["square"] for c in diff["differing_cells"]
                ),
                **tc_props,
                "rank1_pos_a":        pos_a,
                "rank1_prob_a":       round(t1_a[1], 6) if t1_a else None,
                "rank1_pos_b":        pos_b,
                "rank1_prob_b":       round(t1_b[1], 6) if t1_b else None,
                "rank_of_a_in_b":     rank_of(pos_a, probs_b) if pos_a else None,
                "rank_of_b_in_a":     rank_of(pos_b, probs_a) if pos_b else None,
                **{k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in metrics.items()},
            })

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\nResults: {total_pairs} pairs analysed")
    if total_pairs > 0:
        print(f"  Rank-1 agrees  : {agree_count} / {total_pairs} "
              f"({100 * agree_count / total_pairs:.1f}%)")
        print(f"  Rank-1 differs : {total_pairs - agree_count} / {total_pairs} "
              f"({100 * (total_pairs - agree_count) / total_pairs:.1f}%)")

        disagree = [r for r in rows if r.get("rank1_agreement_legal") is False]
        if disagree:
            avg_cross = sum(
                (r["rank_of_a_in_b"] or 0) + (r["rank_of_b_in_a"] or 0)
                for r in disagree
            ) / (2 * len(disagree))
            print(f"  Mean cross-rank: {avg_cross:.1f}")

    if missing_keys:
        print(f"\nWarning: {len(set(missing_keys))} sequences missing from distributions")

    # Distribution divergence summary
    def _vals(key):
        return [r[key] for r in rows if r.get(key) is not None]

    import numpy as np
    print("\nDistribution divergence:")
    for label, key in [
        ("TV distance    (full)", "tv_distance_full"),
        ("TV distance    (legal)", "tv_distance_legal"),
        ("JS divergence  (full)", "js_divergence_full"),
        ("JS divergence  (legal)", "js_divergence_legal"),
        ("Spearman rho   (full)", "spearman_rho_full"),
        ("Spearman rho   (legal)", "spearman_rho_legal"),
    ]:
        vals = _vals(key)
        if vals:
            arr = np.array(vals)
            print(f"  {label:<28} mean={arr.mean():.3f}  "
                  f"median={np.median(arr):.3f}  std={arr.std():.3f}")

    # Correlation analysis
    if not args.no_correlations:
        print()
        results = correlate(rows)
        print_correlation_table(results)
        s = summarise_correlations(results)
        if s:
            print(f"\n  {s['n_significant']}/{s['n_tests']} correlations significant (p<0.01)")
            print(f"  Mean |rho|: {s['mean_abs_rho']:.3f}  Max |rho|: {s['max_abs_rho']:.3f}")

    print(f"\nSaved to '{output_path}'")


if __name__ == "__main__":
    main()
