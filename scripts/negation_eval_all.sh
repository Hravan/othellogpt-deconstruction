#!/usr/bin/env bash
# scripts/negation_eval_all.sh
#
# Prepare held-out sets, run SS evaluation, and report per-depth accuracy
# for all three negation fine-tuning experiments sequentially.
#
# Usage: bash scripts/negation_eval_all.sh

python scripts/negation_prepare.py --train-depths 0,1,2   --output-dir ckpts/negation_depth_7b_012
python scripts/negation_prepare.py --train-depths 0,1,2,3 --output-dir ckpts/negation_depth_7b_0123
python scripts/negation_prepare.py --train-depths 0,1,2,3,4 --output-dir ckpts/negation_depth_7b_01234

uv run python scripts/hf_ss_test.py \
    --model        ckpts/negation_depth_7b_012/final \
    --pairs        ckpts/negation_depth_7b_012/held_out_pairs.json \
    --instruct \
    --train-depths 0,1,2 \
    --output       ckpts/negation_depth_7b_012/eval_results_held_out.json

uv run python scripts/hf_ss_test.py \
    --model        ckpts/negation_depth_7b_0123/final \
    --pairs        ckpts/negation_depth_7b_0123/held_out_pairs.json \
    --instruct --load-in-8bit \
    --train-depths 0,1,2,3 \
    --output       ckpts/negation_depth_7b_0123/eval_results_held_out.json

uv run python scripts/hf_ss_test.py \
    --model        ckpts/negation_depth_7b_01234/final \
    --pairs        ckpts/negation_depth_7b_01234/held_out_pairs.json \
    --instruct --load-in-8bit \
    --train-depths 0,1,2,3,4 \
    --output       ckpts/negation_depth_7b_01234/eval_results_held_out.json
