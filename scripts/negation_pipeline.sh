#!/usr/bin/env bash
# scripts/negation_pipeline.sh
#
# Full negation depth fine-tuning pipeline: generate → prepare → train → evaluate.
# Covers 6 models (Qwen 3B/7B/14B, Mistral 7B/8B/12B) × 3 depth configs.
#
# Usage: bash scripts/negation_pipeline.sh

# ── 1. Generate data ──────────────────────────────────────────────────────────

python scripts/generate_ss_pairs.py

# ── 2. Prepare eval sets (shared across all models) ───────────────────────────

python scripts/negation_prepare.py \
    --train-depths 0,1,2 \
    --output-dir   ckpts/negation_012

python scripts/negation_prepare.py \
    --train-depths 0,1,2,3 \
    --output-dir   ckpts/negation_0123

python scripts/negation_prepare.py \
    --train-depths 0,1,2,3,4 \
    --output-dir   ckpts/negation_01234

# ── 3. Train + evaluate ───────────────────────────────────────────────────────
#
# Each block: negation_train.py then hf_ss_test.py on eval_pairs.json.
# eval_pairs.json contains held-out depth groups + all parity groups.
# hf_ss_test.py --train-depths prints per-depth accuracy + parity SS/CR/accuracy.

# ── Qwen 3B (full precision) ──────────────────────────────────────────────────

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-3B-Instruct \
    --train-depths 0,1,2 \
    --output-dir   ckpts/qwen_3b_012

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_3b_012/final \
    --pairs        ckpts/negation_012/eval_pairs.json \
    --instruct \
    --train-depths 0,1,2 \
    --output       ckpts/qwen_3b_012/eval_results.json

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-3B-Instruct \
    --train-depths 0,1,2,3 \
    --output-dir   ckpts/qwen_3b_0123

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_3b_0123/final \
    --pairs        ckpts/negation_0123/eval_pairs.json \
    --instruct \
    --train-depths 0,1,2,3 \
    --output       ckpts/qwen_3b_0123/eval_results.json

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-3B-Instruct \
    --train-depths 0,1,2,3,4 \
    --output-dir   ckpts/qwen_3b_01234

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_3b_01234/final \
    --pairs        ckpts/negation_01234/eval_pairs.json \
    --instruct \
    --train-depths 0,1,2,3,4 \
    --output       ckpts/qwen_3b_01234/eval_results.json

# ── Qwen 7B (8-bit LoRA) ──────────────────────────────────────────────────────

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-7B-Instruct \
    --train-depths 0,1,2 \
    --output-dir   ckpts/qwen_7b_012 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_7b_012/final \
    --pairs        ckpts/negation_012/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2 \
    --output       ckpts/qwen_7b_012/eval_results.json

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-7B-Instruct \
    --train-depths 0,1,2,3 \
    --output-dir   ckpts/qwen_7b_0123 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_7b_0123/final \
    --pairs        ckpts/negation_0123/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3 \
    --output       ckpts/qwen_7b_0123/eval_results.json

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-7B-Instruct \
    --train-depths 0,1,2,3,4 \
    --output-dir   ckpts/qwen_7b_01234 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_7b_01234/final \
    --pairs        ckpts/negation_01234/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3,4 \
    --output       ckpts/qwen_7b_01234/eval_results.json

# ── Qwen 14B (8-bit LoRA) ─────────────────────────────────────────────────────

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-14B-Instruct \
    --train-depths 0,1,2 \
    --output-dir   ckpts/qwen_14b_012 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_14b_012/final \
    --pairs        ckpts/negation_012/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2 \
    --output       ckpts/qwen_14b_012/eval_results.json

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-14B-Instruct \
    --train-depths 0,1,2,3 \
    --output-dir   ckpts/qwen_14b_0123 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_14b_0123/final \
    --pairs        ckpts/negation_0123/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3 \
    --output       ckpts/qwen_14b_0123/eval_results.json

uv run python scripts/negation_train.py \
    --model        Qwen/Qwen2.5-14B-Instruct \
    --train-depths 0,1,2,3,4 \
    --output-dir   ckpts/qwen_14b_01234 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/qwen_14b_01234/final \
    --pairs        ckpts/negation_01234/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3,4 \
    --output       ckpts/qwen_14b_01234/eval_results.json

# ── Mistral 7B (8-bit LoRA) ───────────────────────────────────────────────────

uv run python scripts/negation_train.py \
    --model        mistralai/Mistral-7B-Instruct-v0.3 \
    --train-depths 0,1,2 \
    --output-dir   ckpts/mistral_7b_012 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_7b_012/final \
    --pairs        ckpts/negation_012/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2 \
    --output       ckpts/mistral_7b_012/eval_results.json

uv run python scripts/negation_train.py \
    --model        mistralai/Mistral-7B-Instruct-v0.3 \
    --train-depths 0,1,2,3 \
    --output-dir   ckpts/mistral_7b_0123 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_7b_0123/final \
    --pairs        ckpts/negation_0123/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3 \
    --output       ckpts/mistral_7b_0123/eval_results.json

uv run python scripts/negation_train.py \
    --model        mistralai/Mistral-7B-Instruct-v0.3 \
    --train-depths 0,1,2,3,4 \
    --output-dir   ckpts/mistral_7b_01234 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_7b_01234/final \
    --pairs        ckpts/negation_01234/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3,4 \
    --output       ckpts/mistral_7b_01234/eval_results.json

# ── Mistral 8B / Ministral (8-bit LoRA) ───────────────────────────────────────

uv run python scripts/negation_train.py \
    --model        mistralai/Ministral-8B-Instruct-2410 \
    --train-depths 0,1,2 \
    --output-dir   ckpts/mistral_8b_012 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_8b_012/final \
    --pairs        ckpts/negation_012/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2 \
    --output       ckpts/mistral_8b_012/eval_results.json

uv run python scripts/negation_train.py \
    --model        mistralai/Ministral-8B-Instruct-2410 \
    --train-depths 0,1,2,3 \
    --output-dir   ckpts/mistral_8b_0123 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_8b_0123/final \
    --pairs        ckpts/negation_0123/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3 \
    --output       ckpts/mistral_8b_0123/eval_results.json

uv run python scripts/negation_train.py \
    --model        mistralai/Ministral-8B-Instruct-2410 \
    --train-depths 0,1,2,3,4 \
    --output-dir   ckpts/mistral_8b_01234 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_8b_01234/final \
    --pairs        ckpts/negation_01234/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3,4 \
    --output       ckpts/mistral_8b_01234/eval_results.json

# ── Mistral Nemo 12B (8-bit LoRA) ─────────────────────────────────────────────

uv run python scripts/negation_train.py \
    --model        mistralai/Mistral-Nemo-Instruct-2407 \
    --train-depths 0,1,2 \
    --output-dir   ckpts/mistral_12b_012 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_12b_012/final \
    --pairs        ckpts/negation_012/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2 \
    --output       ckpts/mistral_12b_012/eval_results.json

uv run python scripts/negation_train.py \
    --model        mistralai/Mistral-Nemo-Instruct-2407 \
    --train-depths 0,1,2,3 \
    --output-dir   ckpts/mistral_12b_0123 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_12b_0123/final \
    --pairs        ckpts/negation_0123/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3 \
    --output       ckpts/mistral_12b_0123/eval_results.json

uv run python scripts/negation_train.py \
    --model        mistralai/Mistral-Nemo-Instruct-2407 \
    --train-depths 0,1,2,3,4 \
    --output-dir   ckpts/mistral_12b_01234 \
    --load-in-8bit

uv run python scripts/hf_ss_test.py \
    --model        ckpts/mistral_12b_01234/final \
    --pairs        ckpts/negation_01234/eval_pairs.json \
    --instruct \
    --load-in-8bit \
    --train-depths 0,1,2,3,4 \
    --output       ckpts/mistral_12b_01234/eval_results.json
