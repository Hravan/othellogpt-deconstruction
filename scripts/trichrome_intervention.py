"""
scripts/trichrome_intervention.py

Trichrome-state intervention experiment.

For each mixed-trichrome transposition group — pairs of sequences reaching
the same Othello board state but different trichrome states — we run
delta_intervention: add the activation delta (act_b - act_a) to seq_a's
forward pass and measure two things:

  1. Does the output distribution shift toward seq_b? (TV distance reduction)
  2. Does the top-1 prediction become illegal in the actual board?

Point 2 is the key finding. The board state is unchanged by the intervention
(both sequences already reach the same board). The activation delta encodes
only path-dependent history — trichrome state and other sequence features.
If the intervention causes the model to predict illegal moves, it proves
the model's predictions are NOT solely determined by board state. A model
with a world model of Othello would never predict an illegal move for a
board it "knows."

Conditions
----------
  Mixed-trichrome  [experimental]
    seq_a and seq_b reach the same Othello board, different trichrome states.
    Delta encodes purely path-dependent history.

  Same-trichrome   [control]
    seq_a and seq_b reach the same Othello board AND same trichrome state.
    Delta should be near-zero; intervention should have negligible effect.

Usage
-----
    uv run python scripts/trichrome_intervention.py \\
        --transpositions data/transpositions_championship.json \\
        --mode championship

    uv run python scripts/trichrome_intervention.py \\
        --transpositions data/transpositions_championship.json \\
        --n-groups 300 \\
        --output data/trichrome_intervention_results.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from othellogpt_deconstruction.core.board import legal_moves_after
from othellogpt_deconstruction.core.tokenizer import itos, PAD_ID
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.intervention.hooks import delta_intervention
from othellogpt_deconstruction.analysis.metrics import metrics_full


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def analyse_pair(
    model:      torch.nn.Module,
    seq_a:      list[str],
    seq_b:      list[str],
    legal_set:  set[int],
    device:     torch.device,
    layers:     list[int] | None = None,
    alpha:      float = 1.0,
) -> dict:
    """
    Run delta_intervention(seq_a → seq_b) and return per-pair statistics.

    Returns a dict with:
        tv_before          : TV distance between original distributions
        tv_after           : TV distance between intervened and target
        tv_reduction       : tv_before - tv_after  (positive = success)
        top1_legal_before  : whether seq_a's top-1 is legal (should be True)
        top1_legal_after   : whether intervened top-1 is legal
        rank1_before       : whether rank-1 agreed before intervention
        rank1_after        : whether rank-1 agrees after intervention
    """
    result = delta_intervention(model, seq_a, seq_b, layers=layers, alpha=alpha, device=device)

    metrics_before = metrics_full(result.probs_original,   result.probs_target)
    metrics_after  = metrics_full(result.probs_intervened, result.probs_target)

    def top1_pos(probs: torch.Tensor) -> int:
        token = int(probs.argmax())
        return int(itos[token]) if token != PAD_ID else -1

    top1_before = top1_pos(result.probs_original)
    top1_after  = top1_pos(result.probs_intervened)

    return {
        "tv_before":         metrics_before["tv_distance_full"],
        "tv_after":          metrics_after["tv_distance_full"],
        "tv_reduction":      metrics_before["tv_distance_full"] - metrics_after["tv_distance_full"],
        "js_before":         metrics_before["js_divergence_full"],
        "js_after":          metrics_after["js_divergence_full"],
        "top1_legal_before": top1_before in legal_set,
        "top1_legal_after":  top1_after  in legal_set,
        "rank1_before":      metrics_before["rank1_agreement_full"],
        "rank1_after":       metrics_after["rank1_agreement_full"],
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_condition(
    model:    torch.nn.Module,
    pairs:    list[tuple[list[str], list[str], set[int]]],
    device:   torch.device,
    label:    str,
    layers:   list[int] | None = None,
    alpha:    float = 1.0,
) -> list[dict]:
    """Run both directions of delta_intervention for each pair."""
    results = []
    for pair_idx, (seq_a, seq_b, legal_set) in enumerate(pairs):
        print(f"  {label}: {pair_idx + 1}/{len(pairs)}", end="\r")
        results.append(analyse_pair(model, seq_a, seq_b, legal_set, device, layers=layers, alpha=alpha))
        results.append(analyse_pair(model, seq_b, seq_a, legal_set, device, layers=layers, alpha=alpha))
    print()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results: list[dict], label: str) -> None:
    if not results:
        print(f"\n{label}: no pairs.")
        return

    tv_before      = np.array([r["tv_before"]    for r in results])
    tv_after       = np.array([r["tv_after"]      for r in results])
    tv_reduction   = np.array([r["tv_reduction"]  for r in results])
    js_before      = np.array([r["js_before"]     for r in results])
    js_after       = np.array([r["js_after"]      for r in results])
    legal_before   = np.array([r["top1_legal_before"] for r in results])
    legal_after    = np.array([r["top1_legal_after"]  for r in results])

    n = len(results)
    initially_disagree = [r for r in results if not r["rank1_before"]]
    rank1_recovered = sum(r["rank1_after"] for r in initially_disagree)

    print(f"\n{label} ({n} directions):")
    print(f"  TV distance   before : {tv_before.mean():.3f}  "
          f"(median {np.median(tv_before):.3f})")
    print(f"  TV distance   after  : {tv_after.mean():.3f}  "
          f"(median {np.median(tv_after):.3f})")
    print(f"  TV reduction         : {tv_reduction.mean():+.3f}  "
          f"({100 * (tv_reduction > 0).mean():.1f}% of directions improved)")
    print(f"  JS divergence before : {js_before.mean():.3f}")
    print(f"  JS divergence after  : {js_after.mean():.3f}")
    print(f"  Top-1 legal  before  : {100 * legal_before.mean():.1f}%")
    print(f"  Top-1 legal  after   : {100 * legal_after.mean():.1f}%  "
          f"({100 * (legal_before & ~legal_after).mean():.1f}% became illegal)")
    if initially_disagree:
        print(
            f"  Rank-1 recovery      : {rank1_recovered}/{len(initially_disagree)} "
            f"({100 * rank1_recovered / len(initially_disagree):.1f}% of disagreeing pairs now agree)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trichrome-state intervention experiment."
    )
    parser.add_argument(
        "--transpositions", required=True,
        help="Path to transpositions JSON (output of find_transpositions.py)",
    )
    parser.add_argument(
        "--mode", default="championship",
        choices=["championship", "synthetic", "random"],
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Explicit checkpoint path (overrides --mode)",
    )
    parser.add_argument(
        "--n-groups", type=int, default=None,
        help="Max mixed-trichrome groups to analyse (default: all)",
    )
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated layer indices to intervene on (default: all layers)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Scaling factor for the activation delta (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save per-pair results as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model ({args.mode})...")
    model = load_model(args.mode, args.checkpoint)
    device = next(model.parameters()).device
    print(f"  Device: {device}")

    print(f"Loading transpositions from {args.transpositions}...")
    with open(args.transpositions) as transpositions_file:
        all_groups = json.load(transpositions_file)

    mixed_groups = [g for g in all_groups if g["trichrome_diffs"]]
    print(f"  {len(all_groups)} total groups, {len(mixed_groups)} mixed-trichrome.")

    rng = random.Random(args.seed)
    if args.n_groups and len(mixed_groups) > args.n_groups:
        mixed_groups = rng.sample(mixed_groups, args.n_groups)
        print(f"  Sampled {len(mixed_groups)} groups.")

    # Build pairs
    # Mixed-trichrome [experimental]: sequences from different trichrome subgroups
    mixed_pairs: list[tuple[list[str], list[str], set[int]]] = []
    # Same-trichrome [control]: sequences from the same subgroup (within mixed groups)
    control_pairs: list[tuple[list[str], list[str], set[int]]] = []

    for group in mixed_groups:
        trichrome_subgroups = group["trichrome_groups"]
        if len(trichrome_subgroups) < 2:
            continue

        seq_a = trichrome_subgroups[0]["sequences"][0]
        seq_b = trichrome_subgroups[1]["sequences"][0]
        # legal_moves_after handles passes correctly by replaying the sequence
        legal_set = set(legal_moves_after(seq_a))
        mixed_pairs.append((seq_a, seq_b, legal_set))

        # Control: two sequences within the same trichrome subgroup
        for subgroup in trichrome_subgroups:
            if len(subgroup["sequences"]) >= 2:
                ctrl_a = subgroup["sequences"][0]
                ctrl_b = subgroup["sequences"][1]
                ctrl_legal = set(legal_moves_after(ctrl_a))
                control_pairs.append((ctrl_a, ctrl_b, ctrl_legal))
                break

    print(f"\n  {len(mixed_pairs)} mixed-trichrome pairs  [experimental]")
    print(f"  {len(control_pairs)} same-trichrome pairs   [control]")

    layers = [int(l) for l in args.layers.split(",")] if args.layers else None
    if layers:
        print(f"  Intervening on layers: {layers}")
    print(f"  Alpha: {args.alpha}")

    print("\nRunning mixed-trichrome interventions [experimental]...")
    mixed_results = run_condition(model, mixed_pairs, device, "mixed", layers=layers, alpha=args.alpha)

    print("Running same-trichrome interventions [control]...")
    control_results = run_condition(model, control_pairs, device, "control", layers=layers, alpha=args.alpha)

    report(
        mixed_results,
        "Mixed-trichrome [experimental]  (same board, different trichrome)",
    )
    report(
        control_results,
        "Same-trichrome  [control]        (same board, same trichrome)",
    )

    # Summary
    if mixed_results and control_results:
        mixed_illegal_rate   = 1 - np.mean([r["top1_legal_after"]  for r in mixed_results])
        control_illegal_rate = 1 - np.mean([r["top1_legal_after"]  for r in control_results])
        baseline_illegal     = 1 - np.mean([r["top1_legal_before"] for r in mixed_results])
        mixed_tv_reduction   = np.mean([r["tv_reduction"] for r in mixed_results])
        control_tv_reduction = np.mean([r["tv_reduction"] for r in control_results])

        print(f"\nSummary:")
        print(f"  Baseline illegal rate (before any intervention) : "
              f"{100 * baseline_illegal:.1f}%")
        print(f"  Post-intervention illegal rate — mixed          : "
              f"{100 * mixed_illegal_rate:.1f}%")
        print(f"  Post-intervention illegal rate — control        : "
              f"{100 * control_illegal_rate:.1f}%")
        print(f"  TV reduction — mixed                            : "
              f"{mixed_tv_reduction:+.3f}")
        print(f"  TV reduction — control                          : "
              f"{control_tv_reduction:+.3f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump({
                "mixed":   mixed_results,
                "control": control_results,
            }, output_file, indent=2)
        print(f"\nSaved results to '{args.output}'")


if __name__ == "__main__":
    main()
