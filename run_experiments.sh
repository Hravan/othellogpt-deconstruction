#!/usr/bin/env bash
# run_experiments.sh
#
# Full reproduction pipeline for the OthelloGPT paper.
# Runs every experiment from scratch in dependency order.
#
# GPU stages run sequentially to avoid OOM.
# CPU-only stages that are independent run in parallel.
#
# Usage:
#   bash run_experiments.sh
#
# Outputs land in data/. Logs for stdout-only scripts go to data/logs/.

set -euo pipefail

LOGS=data/logs
mkdir -p "$LOGS"

echo "============================================================"
echo " Stage 1: Split corpora"
echo "============================================================"

uv run python scripts/split_corpus.py \
    --corpus data/sequence_data/othello_championship \
    --train-output data/games_train.json \
    --test-output data/games_test.json

uv run python scripts/split_corpus.py \
    --corpus data/sequence_data/othello_synthetic \
    --train-output data/games_synthetic_train.json \
    --test-output data/games_synthetic_test.json \
    --max-games 250000

echo "============================================================"
echo " Stage 2: Find transpositions (slow, disk-intensive)"
echo "============================================================"

uv run python scripts/find_transpositions.py \
    --corpus data/sequence_data/othello_championship \
    --output data/transpositions_championship.json &

uv run python scripts/find_transpositions.py \
    --corpus data/sequence_data/othello_synthetic \
    --output data/transpositions_synthetic.json &

wait
echo "Transpositions done."

echo "============================================================"
echo " Stage 3: Train probes  [GPU]"
echo "============================================================"

# Championship board probes (for Nanda attack on championship model)
uv run python scripts/train_board_probes.py \
    --games data/games_train.json \
    --output data/board_probes.pt \
    --mode championship

# Synthetic board probes (for Nanda attack on synthetic model)
uv run python scripts/train_board_probes.py \
    --games data/games_synthetic_train.json \
    --output data/board_probes_synthetic.pt \
    --mode synthetic

# Trichrome probes (for supplementary trichrome section)
uv run python scripts/train_trichrome_probes.py \
    --corpus data/sequence_data/othello_championship \
    --mode championship \
    --output data/trichrome_probes.pt

echo "============================================================"
echo " Stage 4: Extract distributions  [GPU]"
echo "============================================================"

uv run python scripts/extract_distributions.py \
    --transpositions data/transpositions_championship.json \
    --mode championship \
    --output data/distributions_championship.pt

echo "============================================================"
echo " Stage 5: Compute statistics  (Section 1 — path-dependence)"
echo "============================================================"

uv run python scripts/compute_statistics.py \
    --transpositions data/transpositions_championship.json \
    --distributions data/distributions_championship.pt \
    --output data/results_championship.csv

echo "============================================================"
echo " Stage 6: Intervention + rollout  [GPU]  (Section 2 — core)"
echo "============================================================"

# --- Nanda attack: board-flip (3 setups) ---

# Synthetic model on championship games (cross-dataset)
uv run python scripts/board_flip_world_model_test.py \
    --games data/games_test.json \
    --probes data/board_probes_synthetic.pt \
    --mode synthetic --layer 5 \
    --output data/board_flip_syn_on_champ.json

# Synthetic model on synthetic games (Nanda's own conditions)
uv run python scripts/board_flip_world_model_test.py \
    --games data/games_synthetic_test.json \
    --probes data/board_probes_synthetic.pt \
    --mode synthetic --layer 5 \
    --output data/board_flip_syn_on_syn.json

# Championship model on championship games
uv run python scripts/board_flip_world_model_test.py \
    --games data/games_test.json \
    --probes data/board_probes.pt \
    --mode championship --layer 5 \
    --output data/board_flip_champ_on_champ.json

# --- Nanda attack: n_flips sweep (1-5) ---

uv run python scripts/board_flip_intervention_test.py \
    --games data/games_test.json \
    --probes data/board_probes.pt \
    --layer 5 \
    --output data/board_flip_intervention_results.json

# --- Nanda attack: lookup table vs model ---

uv run python scripts/board_flip_lookup_test.py \
    --corpus data/sequence_data/othello_championship \
    --probes data/board_probes.pt \
    --layer 5 \
    | tee "$LOGS/board_flip_lookup_results.log"

# --- Li attack: gradient steering ---

# Natural positions — random flip (main replication)
uv run python scripts/probe_gradient_steering_test.py \
    --games data/games_synthetic_test.json \
    --layer-start 4 --mode synthetic \
    --flip-strategy random \
    --output data/probe_gradient_steering_random.json

# Natural positions — max-new-legal flip (best-case for Li)
uv run python scripts/probe_gradient_steering_test.py \
    --games data/games_synthetic_test.json \
    --layer-start 4 --mode synthetic \
    --flip-strategy max_new_legal \
    --output data/probe_gradient_steering_max_new_legal.json

# Unnatural benchmark (Li's strongest claim)
uv run python scripts/probe_gradient_steering_unreachable_test.py \
    --benchmark data/intervention_benchmark.pkl \
    --output data/probe_gradient_steering_unreachable.json

echo "============================================================"
echo " Stage 7: Trichrome  [GPU]  (Section 3 — supplementary)"
echo "============================================================"

# Delta intervention: mixed vs same-trichrome control
uv run python scripts/trichrome_intervention.py \
    --transpositions data/transpositions_championship.json \
    --mode championship \
    --output data/trichrome_intervention_results.json

# Probe-direction vs random (norm-matched)
uv run python scripts/trichrome_probe_intervention.py \
    --transpositions data/transpositions_championship.json \
    --probes data/trichrome_probes.pt \
    --layer 5 \
    --output data/trichrome_intervention_targeted.json

echo "============================================================"
echo " All experiments complete."
echo " JSON results: data/"
echo " Stdout logs:  $LOGS/"
echo "============================================================"
