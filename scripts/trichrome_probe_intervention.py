"""
scripts/trichrome_probe_intervention.py

Trichrome probe intervention experiment.

For each mixed-trichrome transposition pair we perform two norm-matched
interventions at a single layer and compare their effects:

  Trichrome-direction [experimental]
    Add the probe-derived trichrome direction — the sum of
    W[pos, color_b] - W[pos, color_a] across all cells where the two
    sequences' trichrome states differ — scaled to the same L2 norm as
    the actual activation delta at that layer.

  Random-direction [control]
    Add a random unit vector drawn from an isotropic Gaussian, scaled to
    the same L2 norm as the actual activation delta.

If adding the trichrome-probe direction causes significantly more illegal
predictions than the norm-matched random direction, the probe direction
specifically encodes information the model uses causally — the effect is
not just a consequence of perturbation magnitude.

Usage
-----
    # Train probes first:
    uv run python scripts/train_trichrome_probes.py \\
        --corpus path/to/championship/ \\
        --output data/trichrome_probes.pt

    # Then run this:
    uv run python scripts/trichrome_probe_intervention.py \\
        --transpositions data/transpositions_championship.json \\
        --probes data/trichrome_probes.pt \\
        --layer 5 \\
        --n-groups 300
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from othellogpt_deconstruction.core.board import legal_moves_after
from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, itos, BLOCK_SIZE, PAD_ID
from othellogpt_deconstruction.intervention.hooks import collect_activations, patch_activations
from othellogpt_deconstruction.model.inference import load_model
from othellogpt_deconstruction.model.probes import load_probes
from othellogpt_deconstruction.analysis.metrics import metrics_full


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def encode_sequence(sequence: list[str], device: torch.device) -> torch.Tensor:
    tokens = [stoi[alg_to_pos(move)] for move in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


def forward_pass(model: torch.nn.Module, x: torch.Tensor, seq_length: int) -> torch.Tensor:
    """Return softmax distribution at the last token position."""
    with torch.no_grad():
        logits, _ = model(x)
    last_logits = logits[0, seq_length - 1, :].clone()
    last_logits[PAD_ID] = float("-inf")
    return torch.softmax(last_logits, dim=-1)


# ---------------------------------------------------------------------------
# Per-pair experiment
# ---------------------------------------------------------------------------

def analyse_pair(
    model:       torch.nn.Module,
    seq_a:       list[str],
    seq_b:       list[str],
    cell_diffs:  list[dict],
    probe:       object,
    layer:       int,
    legal_set:   set[int],
    device:      torch.device,
    rng:         random.Random,
) -> dict | None:
    """
    Run both norm-matched interventions for one pair and return statistics.

    Returns None if the probe direction is near-zero (no trainable cells).
    """
    x_a = encode_sequence(seq_a, device)
    x_b = encode_sequence(seq_b, device)

    # Collect activations for both sequences
    with collect_activations(model, len(seq_a)) as store_a:
        probs_a = forward_pass(model, x_a, len(seq_a))

    with collect_activations(model, len(seq_b)) as store_b:
        probs_b = forward_pass(model, x_b, len(seq_b))

    act_a = store_a.get(layer)
    act_b = store_b.get(layer)
    if act_a is None or act_b is None:
        return None

    # Actual activation delta at the probe layer
    actual_delta = (act_b - act_a).float()
    delta_norm = actual_delta.norm().item()

    if delta_norm < 1e-8:
        return None

    # Trichrome probe direction for these cell diffs
    probe_direction = probe.trichrome_direction(cell_diffs).to(device)
    probe_direction_norm = probe_direction.norm().item()

    if probe_direction_norm < 1e-8:
        return None

    # Scale to match actual delta norm
    trichrome_patch = probe_direction * (delta_norm / probe_direction_norm)

    # Random direction at same norm
    random_direction = torch.randn_like(probe_direction)
    random_patch = random_direction * (delta_norm / random_direction.norm())

    # Cosine similarity between probe direction and actual delta
    cosine_similarity = float(
        torch.dot(probe_direction.float(), actual_delta.to(device).float())
        / (probe_direction_norm * delta_norm)
    )

    # Intervene with trichrome-probe direction
    with patch_activations(model, len(seq_a), {layer: trichrome_patch}):
        probs_trichrome = forward_pass(model, x_a, len(seq_a))

    # Intervene with random direction
    with patch_activations(model, len(seq_a), {layer: random_patch}):
        probs_random = forward_pass(model, x_a, len(seq_a))

    def top1_position(probs: torch.Tensor) -> int:
        token = int(probs.argmax())
        return int(itos[token]) if token != PAD_ID else -1

    top1_original  = top1_position(probs_a)
    top1_trichrome = top1_position(probs_trichrome)
    top1_random    = top1_position(probs_random)

    metrics_original  = metrics_full(probs_a, probs_b)
    metrics_trichrome = metrics_full(probs_trichrome, probs_b)
    metrics_random    = metrics_full(probs_random,    probs_b)

    return {
        "top1_legal_original":   top1_original  in legal_set,
        "top1_legal_trichrome":  top1_trichrome in legal_set,
        "top1_legal_random":     top1_random    in legal_set,
        "tv_original":           metrics_original["tv_distance_full"],
        "tv_trichrome":          metrics_trichrome["tv_distance_full"],
        "tv_random":             metrics_random["tv_distance_full"],
        "delta_norm":            delta_norm,
        "probe_direction_norm":  probe_direction_norm,
        "cosine_similarity":     cosine_similarity,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results: list[dict], label: str) -> None:
    if not results:
        print(f"\n{label}: no results.")
        return

    n = len(results)
    legal_original  = np.mean([r["top1_legal_original"]  for r in results])
    legal_trichrome = np.mean([r["top1_legal_trichrome"] for r in results])
    legal_random    = np.mean([r["top1_legal_random"]    for r in results])

    tv_original  = np.mean([r["tv_original"]  for r in results])
    tv_trichrome = np.mean([r["tv_trichrome"] for r in results])
    tv_random    = np.mean([r["tv_random"]    for r in results])

    cos_sims = np.array([r["cosine_similarity"] for r in results])

    print(f"\n{label} ({n} intervention directions):")
    print(f"  Top-1 legal  — original         : {100 * legal_original:.1f}%")
    print(f"  Top-1 legal  — trichrome patch  : {100 * legal_trichrome:.1f}%  "
          f"({100 * (legal_original - legal_trichrome):+.1f}pp)")
    print(f"  Top-1 legal  — random patch     : {100 * legal_random:.1f}%  "
          f"({100 * (legal_original - legal_random):+.1f}pp)")
    print(f"  TV dist      — original         : {tv_original:.3f}")
    print(f"  TV dist      — trichrome patch  : {tv_trichrome:.3f}  "
          f"(delta {tv_trichrome - tv_original:+.3f})")
    print(f"  TV dist      — random patch     : {tv_random:.3f}  "
          f"(delta {tv_random - tv_original:+.3f})")
    print(f"  Cosine sim (probe vs delta)     : {cos_sims.mean():.3f}  "
          f"(std {cos_sims.std():.3f})")

    illegal_trichrome = 100 * (legal_original - legal_trichrome)
    illegal_random    = 100 * (legal_original - legal_random)
    trichrome_vs_random = illegal_trichrome - illegal_random
    print(f"\n  Key comparison:")
    print(f"    Trichrome direction caused {illegal_trichrome:+.1f}pp more illegal predictions")
    print(f"    Random direction caused    {illegal_random:+.1f}pp more illegal predictions")
    print(f"    Trichrome excess           {trichrome_vs_random:+.1f}pp")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trichrome probe intervention: norm-matched comparison."
    )
    parser.add_argument(
        "--transpositions", required=True,
        help="Path to transpositions JSON (output of find_transpositions.py)",
    )
    parser.add_argument(
        "--probes", required=True,
        help="Path to trained trichrome probes (output of train_trichrome_probes.py)",
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
        "--layer", type=int, default=5,
        help="Residual stream layer to intervene on (default: 5)",
    )
    parser.add_argument(
        "--n-groups", type=int, default=None,
        help="Max mixed-trichrome groups to analyse (default: all)",
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

    print(f"Loading probes from {args.probes}...")
    probes = load_probes(args.probes)
    available_layers = sorted(probes.keys())
    print(f"  Probes available for layers: {available_layers}")

    if args.layer not in probes:
        raise ValueError(
            f"Layer {args.layer} not in probes. Available: {available_layers}"
        )
    probe = probes[args.layer]
    print(f"  Using probe at layer {args.layer}.")

    print(f"Loading transpositions from {args.transpositions}...")
    with open(args.transpositions) as transpositions_file:
        all_groups = json.load(transpositions_file)

    mixed_groups = [group for group in all_groups if group["trichrome_diffs"]]
    print(f"  {len(all_groups)} total groups, {len(mixed_groups)} mixed-trichrome.")

    rng = random.Random(args.seed)
    if args.n_groups and len(mixed_groups) > args.n_groups:
        mixed_groups = rng.sample(mixed_groups, args.n_groups)
        print(f"  Sampled {len(mixed_groups)} groups.")

    # Build pairs: one pair per group, both intervention directions
    results: list[dict] = []
    n_skipped = 0

    for group_idx, group in enumerate(mixed_groups):
        print(f"  Group {group_idx + 1}/{len(mixed_groups)}", end="\r")

        trichrome_subgroups = group["trichrome_groups"]
        if len(trichrome_subgroups) < 2:
            n_skipped += 1
            continue

        seq_a = trichrome_subgroups[0]["sequences"][0]
        seq_b = trichrome_subgroups[1]["sequences"][0]

        # cell_diffs: cells where trichrome states differ (seq_a=color_a, seq_b=color_b)
        cell_diffs = group["trichrome_diffs"][0]["differing_cells"]
        legal_set = set(legal_moves_after(seq_a))

        # Both directions: seq_a → seq_b and seq_b → seq_a
        result_ab = analyse_pair(
            model, seq_a, seq_b, cell_diffs, probe, args.layer, legal_set, device, rng
        )
        if result_ab is not None:
            results.append(result_ab)

        # Reverse: cell_diffs swap color_a/color_b
        reversed_cell_diffs = [
            {**diff, "color_a": diff["color_b"], "color_b": diff["color_a"]}
            for diff in cell_diffs
        ]
        legal_set_b = set(legal_moves_after(seq_b))
        result_ba = analyse_pair(
            model, seq_b, seq_a, reversed_cell_diffs, probe, args.layer,
            legal_set_b, device, rng,
        )
        if result_ba is not None:
            results.append(result_ba)

    print()
    if n_skipped:
        print(f"  Skipped {n_skipped} groups (fewer than 2 trichrome subgroups).")

    report(results, f"Layer {args.layer} intervention (n={len(results)} directions)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_results = [
            {k: float(v) if isinstance(v, (float, np.floating)) else bool(v) if isinstance(v, (bool, np.bool_)) else v
             for k, v in result.items()}
            for result in results
        ]
        with open(output_path, "w") as output_file:
            json.dump(serializable_results, output_file, indent=2)
        print(f"\nSaved results to '{args.output}'")


if __name__ == "__main__":
    main()
