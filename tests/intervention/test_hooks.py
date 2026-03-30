"""
tests/intervention/test_hooks.py

Tests for activation patching and intervention.
Uses mode="random" so no checkpoint or GPU required.
"""

import pytest
import torch

from othellogpt_deconstruction.core.tokenizer import VOCAB_SIZE
from othellogpt_deconstruction.core.board import legal_moves_after
from othellogpt_deconstruction.model.inference import load_model, get_distribution
from othellogpt_deconstruction.intervention.hooks import (
    collect_activations, patch_activations,
    delta_intervention, direct_intervention,
    intervention_success, ActivationStore, InterventionResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return load_model(mode="random", device="cpu")


SEQ_A = ["f5", "f6", "d3", "f4", "g5"]
SEQ_B = ["f5", "f4", "d3", "f6", "g5"]
SHORT_SEQ = ["f5"]


# ---------------------------------------------------------------------------
# collect_activations
# ---------------------------------------------------------------------------

def test_collect_activations_layer_count(model):
    with collect_activations(model, len(SEQ_A)) as store:
        get_distribution(model, SEQ_A)
    assert len(store) == len(model.blocks)


def test_collect_activations_shape(model):
    n_embd = model.tok_emb.embedding_dim
    with collect_activations(model, len(SEQ_A)) as store:
        get_distribution(model, SEQ_A)
    for layer in store.layers():
        act = store.get(layer)
        assert act.shape == (n_embd,)


def test_collect_activations_layers_sorted(model):
    with collect_activations(model, len(SEQ_A)) as store:
        get_distribution(model, SEQ_A)
    assert store.layers() == sorted(store.layers())


def test_collect_activations_different_sequences_differ(model):
    with collect_activations(model, len(SEQ_A)) as store_a:
        get_distribution(model, SEQ_A)
    with collect_activations(model, len(SEQ_B)) as store_b:
        get_distribution(model, SEQ_B)
    # At least one layer should differ
    diffs = [
        not torch.allclose(store_a.get(l), store_b.get(l))
        for l in store_a.layers()
    ]
    assert any(diffs)


def test_collect_activations_hooks_removed_after(model):
    """Hooks should be removed after context exits — model should behave normally."""
    with collect_activations(model, len(SEQ_A)):
        pass
    # Should run without error and produce valid output
    probs = get_distribution(model, SEQ_A)
    assert probs.shape == (VOCAB_SIZE,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# patch_activations
# ---------------------------------------------------------------------------

def test_patch_activations_changes_output(model):
    """Patching with a nonzero delta should change the output distribution."""
    with collect_activations(model, len(SEQ_A)) as store_a:
        probs_original = get_distribution(model, SEQ_A)
    with collect_activations(model, len(SEQ_B)) as store_b:
        get_distribution(model, SEQ_B)

    patches = {
        l: store_b.get(l) - store_a.get(l)
        for l in store_a.layers()
    }

    from othellogpt_deconstruction.model.inference import get_distribution as gd
    from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, BLOCK_SIZE, PAD_ID

    tokens = [stoi[alg_to_pos(m)] for m in SEQ_A]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    x = torch.tensor([padded], dtype=torch.long)

    with patch_activations(model, len(SEQ_A), patches, alpha=1.0):
        with torch.no_grad():
            logits, _ = model(x)
        last = logits[0, len(SEQ_A) - 1, :].clone()
        last[PAD_ID] = float("-inf")
        probs_patched = torch.softmax(last, dim=-1)

    assert not torch.allclose(probs_original, probs_patched)


def test_patch_zero_delta_no_change(model):
    """Patching with zero deltas should not change the output."""
    n_embd = model.tok_emb.embedding_dim
    with collect_activations(model, len(SEQ_A)) as store_a:
        probs_original = get_distribution(model, SEQ_A)

    patches = {l: torch.zeros(n_embd) for l in store_a.layers()}

    from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, BLOCK_SIZE, PAD_ID
    tokens = [stoi[alg_to_pos(m)] for m in SEQ_A]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    x = torch.tensor([padded], dtype=torch.long)

    with patch_activations(model, len(SEQ_A), patches, alpha=1.0):
        with torch.no_grad():
            logits, _ = model(x)
        last = logits[0, len(SEQ_A) - 1, :].clone()
        last[PAD_ID] = float("-inf")
        probs_patched = torch.softmax(last, dim=-1)

    assert torch.allclose(probs_original, probs_patched, atol=1e-6)


def test_patch_hooks_removed_after(model):
    """Hooks should be removed after patch context exits."""
    n_embd = model.tok_emb.embedding_dim
    with collect_activations(model, len(SEQ_A)) as store_a:
        get_distribution(model, SEQ_A)
    patches = {l: torch.zeros(n_embd) for l in store_a.layers()}
    with patch_activations(model, len(SEQ_A), patches):
        pass
    probs = get_distribution(model, SEQ_A)
    assert probs.shape == (VOCAB_SIZE,)


# ---------------------------------------------------------------------------
# delta_intervention
# ---------------------------------------------------------------------------

def test_delta_intervention_returns_result(model):
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    assert isinstance(result, InterventionResult)


def test_delta_intervention_distribution_shapes(model):
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    assert result.probs_original.shape == (VOCAB_SIZE,)
    assert result.probs_target.shape == (VOCAB_SIZE,)
    assert result.probs_intervened.shape == (VOCAB_SIZE,)


def test_delta_intervention_distributions_sum_to_one(model):
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    assert float(result.probs_original.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(result.probs_target.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(result.probs_intervened.sum()) == pytest.approx(1.0, abs=1e-5)


def test_delta_intervention_original_matches_get_distribution(model):
    """probs_original should match what get_distribution returns directly."""
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    probs_direct = get_distribution(model, SEQ_A, device=torch.device("cpu"))
    assert torch.allclose(result.probs_original, probs_direct.cpu(), atol=1e-6)


def test_delta_intervention_target_matches_get_distribution(model):
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    probs_direct = get_distribution(model, SEQ_B, device=torch.device("cpu"))
    assert torch.allclose(result.probs_target, probs_direct.cpu(), atol=1e-6)


def test_delta_intervention_changes_output(model):
    """Intervention should change the output from the original."""
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    assert not torch.allclose(result.probs_original, result.probs_intervened)


def test_delta_intervention_subset_of_layers(model):
    """Intervening on fewer layers should still work."""
    result = delta_intervention(
        model, SEQ_A, SEQ_B, layers=[0, 1, 2], device=torch.device("cpu")
    )
    assert result.layers_intervened == [0, 1, 2]
    assert result.probs_intervened.shape == (VOCAB_SIZE,)


def test_delta_intervention_alpha_zero_no_change(model):
    """alpha=0 means no patch applied — output should match original."""
    result = delta_intervention(
        model, SEQ_A, SEQ_B, alpha=0.0, device=torch.device("cpu")
    )
    assert torch.allclose(result.probs_original, result.probs_intervened, atol=1e-6)


# ---------------------------------------------------------------------------
# direct_intervention
# ---------------------------------------------------------------------------

def test_direct_intervention_returns_result(model):
    result = direct_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    assert isinstance(result, InterventionResult)


def test_direct_intervention_changes_output(model):
    result = direct_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    assert not torch.allclose(result.probs_original, result.probs_intervened)


def test_direct_full_intervention_approaches_target(model):
    """
    With alpha=1.0 on all layers, intervened output should be closer to
    target than original is.
    """
    result = direct_intervention(
        model, SEQ_A, SEQ_B, alpha=1.0, device=torch.device("cpu")
    )
    from othellogpt_deconstruction.analysis.metrics import metrics_full
    m_before = metrics_full(result.probs_original, result.probs_target)
    m_after  = metrics_full(result.probs_intervened, result.probs_target)
    # TV distance to target should decrease after full intervention
    assert m_after["tv_distance_full"] < m_before["tv_distance_full"]


# ---------------------------------------------------------------------------
# intervention_success
# ---------------------------------------------------------------------------

def test_intervention_success_keys(model):
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    legal = legal_moves_after(SEQ_A)
    s = intervention_success(result, legal)
    assert set(s.keys()) == {
        "tv_before", "tv_after", "tv_reduction",
        "js_before", "js_after",
        "rank1_before", "rank1_after",
    }


def test_intervention_success_tv_nonnegative(model):
    result = delta_intervention(model, SEQ_A, SEQ_B, device=torch.device("cpu"))
    legal = legal_moves_after(SEQ_A)
    s = intervention_success(result, legal)
    assert s["tv_before"] >= 0.0
    assert s["tv_after"] >= 0.0


def test_intervention_success_identical_sequences(model):
    """Intervening A->A should give zero TV reduction."""
    result = delta_intervention(model, SEQ_A, SEQ_A, device=torch.device("cpu"))
    legal = legal_moves_after(SEQ_A)
    s = intervention_success(result, legal)
    assert s["tv_before"] == pytest.approx(0.0, abs=1e-6)
    assert s["tv_reduction"] == pytest.approx(0.0, abs=1e-6)
