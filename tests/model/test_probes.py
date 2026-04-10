"""
tests/model/test_probes.py

Tests for TrichromeProbe, save_probes, and load_probes.
No model checkpoint or GPU required.
"""

import pytest
import torch

from othellogpt_deconstruction.model.probes import TrichromeProbe, save_probes, load_probes

N_CELLS  = 64
N_COLORS = 3
D_MODEL  = 16   # small for speed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def probe():
    weights = torch.randn(N_CELLS, N_COLORS, D_MODEL)
    biases  = torch.randn(N_CELLS, N_COLORS)
    return TrichromeProbe(weights=weights, biases=biases, layer=3)


@pytest.fixture
def zero_probe():
    weights = torch.zeros(N_CELLS, N_COLORS, D_MODEL)
    biases  = torch.zeros(N_CELLS, N_COLORS)
    return TrichromeProbe(weights=weights, biases=biases, layer=0)


# ---------------------------------------------------------------------------
# logits — single activation
# ---------------------------------------------------------------------------

def test_logits_single_shape(probe):
    activation = torch.randn(D_MODEL)
    result = probe.logits(activation)
    assert result.shape == (N_CELLS, N_COLORS)


def test_logits_batch_shape(probe):
    batch = torch.randn(8, D_MODEL)
    result = probe.logits(batch)
    assert result.shape == (8, N_CELLS, N_COLORS)


def test_logits_single_matches_batch(probe):
    activation = torch.randn(D_MODEL)
    single = probe.logits(activation)
    batched = probe.logits(activation.unsqueeze(0))
    assert torch.allclose(single, batched[0], atol=1e-6)


def test_logits_zero_weights_equals_bias(zero_probe):
    zero_probe.biases = torch.ones(N_CELLS, N_COLORS) * 5.0
    activation = torch.randn(D_MODEL)
    result = zero_probe.logits(activation)
    assert torch.allclose(result, torch.full((N_CELLS, N_COLORS), 5.0), atol=1e-6)


def test_logits_uses_float(probe):
    activation = torch.randn(D_MODEL, dtype=torch.float16)
    result = probe.logits(activation)
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# cell_direction
# ---------------------------------------------------------------------------

def test_cell_direction_shape(probe):
    direction = probe.cell_direction(pos=10, from_color=0, to_color=1)
    assert direction.shape == (D_MODEL,)


def test_cell_direction_dtype(probe):
    direction = probe.cell_direction(pos=0, from_color=0, to_color=2)
    assert direction.dtype == torch.float32


def test_cell_direction_same_color_is_zero(probe):
    direction = probe.cell_direction(pos=5, from_color=1, to_color=1)
    assert torch.allclose(direction, torch.zeros(D_MODEL), atol=1e-6)


def test_cell_direction_is_weight_difference(probe):
    pos, from_color, to_color = 7, 0, 2
    expected = (probe.weights[pos, to_color] - probe.weights[pos, from_color]).float()
    direction = probe.cell_direction(pos, from_color, to_color)
    assert torch.allclose(direction, expected, atol=1e-6)


def test_cell_direction_antisymmetric(probe):
    pos = 3
    forward  = probe.cell_direction(pos, from_color=0, to_color=1)
    backward = probe.cell_direction(pos, from_color=1, to_color=0)
    assert torch.allclose(forward, -backward, atol=1e-6)


def test_cell_direction_independent_of_other_cells(probe):
    """Changing cell 0 weights should not affect direction for cell 1."""
    pos = 1
    before = probe.cell_direction(pos, from_color=0, to_color=2).clone()
    probe.weights[0] = torch.randn(N_COLORS, D_MODEL)
    after = probe.cell_direction(pos, from_color=0, to_color=2)
    assert torch.allclose(before, after, atol=1e-6)


# ---------------------------------------------------------------------------
# trichrome_direction
# ---------------------------------------------------------------------------

def test_trichrome_direction_shape(probe):
    cell_diffs = [{"pos": 10, "color_a": 0, "color_b": 1}]
    direction = probe.trichrome_direction(cell_diffs)
    assert direction.shape == (D_MODEL,)


def test_trichrome_direction_dtype(probe):
    cell_diffs = [{"pos": 10, "color_a": 0, "color_b": 1}]
    direction = probe.trichrome_direction(cell_diffs)
    assert direction.dtype == torch.float32


def test_trichrome_direction_empty_diffs_is_zero(probe):
    direction = probe.trichrome_direction([])
    assert torch.allclose(direction, torch.zeros(D_MODEL), atol=1e-6)


def test_trichrome_direction_single_diff_matches_cell_direction(probe):
    pos, from_color, to_color = 12, 1, 2
    cell_diffs = [{"pos": pos, "color_a": from_color, "color_b": to_color}]
    trichrome = probe.trichrome_direction(cell_diffs)
    cell = probe.cell_direction(pos, from_color, to_color)
    assert torch.allclose(trichrome, cell, atol=1e-6)


def test_trichrome_direction_sums_multiple_diffs(probe):
    diffs = [
        {"pos": 5,  "color_a": 0, "color_b": 1},
        {"pos": 20, "color_a": 2, "color_b": 0},
    ]
    expected = (
        probe.cell_direction(5, 0, 1)
        + probe.cell_direction(20, 2, 0)
    )
    direction = probe.trichrome_direction(diffs)
    assert torch.allclose(direction, expected, atol=1e-6)


def test_trichrome_direction_same_color_diffs_cancel(probe):
    """Two diffs on the same cell in opposite directions should cancel."""
    diffs = [
        {"pos": 3, "color_a": 0, "color_b": 1},
        {"pos": 3, "color_a": 1, "color_b": 0},
    ]
    direction = probe.trichrome_direction(diffs)
    assert torch.allclose(direction, torch.zeros(D_MODEL), atol=1e-6)


def test_trichrome_direction_accepts_int_string_keys(probe):
    """Keys 'pos', 'color_a', 'color_b' should be int-castable."""
    cell_diffs = [{"pos": "10", "color_a": "0", "color_b": "1",
                   "square": "c4", "owner": 1, "distance": 1}]
    direction = probe.trichrome_direction(cell_diffs)
    assert direction.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# save_probes / load_probes
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(probe, tmp_path):
    path = tmp_path / "probes.pt"
    save_probes({3: probe}, path)
    loaded = load_probes(path)
    assert 3 in loaded
    assert torch.allclose(loaded[3].weights, probe.weights)
    assert torch.allclose(loaded[3].biases,  probe.biases)
    assert loaded[3].layer == 3


def test_save_load_multiple_layers(tmp_path):
    probes = {
        layer: TrichromeProbe(
            weights=torch.randn(N_CELLS, N_COLORS, D_MODEL),
            biases=torch.randn(N_CELLS, N_COLORS),
            layer=layer,
        )
        for layer in range(4)
    }
    path = tmp_path / "probes.pt"
    save_probes(probes, path)
    loaded = load_probes(path)
    assert set(loaded.keys()) == {0, 1, 2, 3}
    for layer in range(4):
        assert torch.allclose(loaded[layer].weights, probes[layer].weights)
        assert torch.allclose(loaded[layer].biases,  probes[layer].biases)


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "probes.pt"
    probe_single = TrichromeProbe(
        weights=torch.zeros(N_CELLS, N_COLORS, D_MODEL),
        biases=torch.zeros(N_CELLS, N_COLORS),
        layer=0,
    )
    save_probes({0: probe_single}, path)
    assert path.exists()


def test_load_preserves_layer_attribute(tmp_path):
    path = tmp_path / "probes.pt"
    probe_layer7 = TrichromeProbe(
        weights=torch.zeros(N_CELLS, N_COLORS, D_MODEL),
        biases=torch.zeros(N_CELLS, N_COLORS),
        layer=7,
    )
    save_probes({7: probe_layer7}, path)
    loaded = load_probes(path)
    assert loaded[7].layer == 7
