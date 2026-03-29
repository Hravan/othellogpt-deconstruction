"""
tests/model/test_inference.py

Tests for model loading and inference.
Uses mode="random" so no checkpoint files or GPU are required.
"""

import pytest
import torch

from othellogpt_deconstruction.core.tokenizer import VOCAB_SIZE, seq_key
from othellogpt_deconstruction.model.inference import (
    make_config, load_model, get_distribution, get_distributions_batch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_model():
    """A randomly initialised model shared across all tests in this module."""
    return load_model(mode="random", device="cpu")


SEQ_A = ["f5", "f6", "d3", "f4", "g5"]
SEQ_B = ["f5", "f4", "d3", "f6", "g5"]
SHORT_SEQ = ["f5"]


# ---------------------------------------------------------------------------
# make_config
# ---------------------------------------------------------------------------

def test_make_config_vocab_size():
    cfg = make_config()
    assert cfg.vocab_size == VOCAB_SIZE


def test_make_config_block_size():
    from othellogpt_deconstruction.core.tokenizer import BLOCK_SIZE
    cfg = make_config()
    assert cfg.block_size == BLOCK_SIZE


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def test_load_model_random():
    model = load_model(mode="random", device="cpu")
    assert model is not None


def test_load_model_eval_mode():
    model = load_model(mode="random", device="cpu")
    assert not model.training


def test_load_model_on_cpu():
    model = load_model(mode="random", device="cpu")
    device = next(model.parameters()).device
    assert device.type == "cpu"


def test_load_model_unknown_mode_raises():
    with pytest.raises(ValueError):
        load_model(mode="unknown")


def test_load_model_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        load_model(mode="championship", checkpoint_path="nonexistent.ckpt")


# ---------------------------------------------------------------------------
# get_distribution
# ---------------------------------------------------------------------------

def test_get_distribution_shape(random_model):
    probs = get_distribution(random_model, SEQ_A)
    assert probs.shape == (VOCAB_SIZE,)


def test_get_distribution_sums_to_one(random_model):
    probs = get_distribution(random_model, SEQ_A)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_get_distribution_nonnegative(random_model):
    probs = get_distribution(random_model, SEQ_A)
    assert (probs >= 0).all()


def test_get_distribution_pad_is_zero(random_model):
    from othellogpt_deconstruction.core.tokenizer import PAD_ID
    probs = get_distribution(random_model, SEQ_A)
    assert float(probs[PAD_ID]) == pytest.approx(0.0, abs=1e-10)


def test_get_distribution_cpu_output(random_model):
    probs = get_distribution(random_model, SEQ_A)
    assert probs.device.type == "cpu"


def test_get_distribution_short_sequence(random_model):
    probs = get_distribution(random_model, SHORT_SEQ)
    assert probs.shape == (VOCAB_SIZE,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_get_distribution_different_sequences_differ(random_model):
    """Two different sequences should (almost certainly) produce different distributions."""
    probs_a = get_distribution(random_model, SEQ_A)
    probs_b = get_distribution(random_model, SEQ_B)
    assert not torch.allclose(probs_a, probs_b)


# ---------------------------------------------------------------------------
# get_distributions_batch
# ---------------------------------------------------------------------------

def test_get_distributions_batch_keys(random_model):
    seqs = [SEQ_A, SEQ_B, SHORT_SEQ]
    result = get_distributions_batch(random_model, seqs)
    assert set(result.keys()) == {seq_key(s) for s in seqs}


def test_get_distributions_batch_shapes(random_model):
    seqs = [SEQ_A, SEQ_B]
    result = get_distributions_batch(random_model, seqs)
    for probs in result.values():
        assert probs.shape == (VOCAB_SIZE,)


def test_get_distributions_batch_sums_to_one(random_model):
    seqs = [SEQ_A, SEQ_B]
    result = get_distributions_batch(random_model, seqs)
    for probs in result.values():
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_get_distributions_batch_deduplication(random_model):
    """Duplicate sequences should only appear once in output."""
    seqs = [SEQ_A, SEQ_A, SEQ_B]
    result = get_distributions_batch(random_model, seqs)
    assert len(result) == 2


def test_get_distributions_batch_matches_single(random_model):
    """Batch results must match single-sequence results exactly."""
    result = get_distributions_batch(random_model, [SEQ_A, SEQ_B])
    probs_a_single = get_distribution(random_model, SEQ_A)
    probs_a_batch  = result[seq_key(SEQ_A)]
    assert torch.allclose(probs_a_single.cpu(), probs_a_batch.cpu(), atol=1e-6)


def test_get_distributions_batch_small_batch_size(random_model):
    """batch_size=1 should give same results as default."""
    seqs = [SEQ_A, SEQ_B, SHORT_SEQ]
    result_default = get_distributions_batch(random_model, seqs)
    result_small   = get_distributions_batch(random_model, seqs, batch_size=1)
    for k in result_default:
        assert torch.allclose(result_default[k], result_small[k], atol=1e-6)
