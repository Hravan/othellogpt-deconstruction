"""
src/othellogpt_deconstruction/model/probes.py

Linear probes predicting trichrome color per cell from residual stream activations.

A TrichromeProbe for a given layer contains weight vectors W[pos, color] of
shape (d_model,) for each (cell, color) pair.  The direction for changing cell
pos from color_a to color_b is W[pos, color_b] - W[pos, color_a].  Summing
these directions across all differing cells gives the probe-derived intervention
direction used in the trichrome_probe_intervention experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrichromeProbe:
    """
    Per-cell linear probe for trichrome color prediction.

    weights : (64, 3, d_model) — W[pos, color] is the weight vector that
              scores color `color` at cell `pos`
    biases  : (64, 3) — per-cell, per-color intercepts
    layer   : which residual stream layer these weights were trained on
    """

    weights: torch.Tensor   # (64, 3, d_model)
    biases:  torch.Tensor   # (64, 3)
    layer:   int

    def logits(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute per-cell color logits.

        Parameters
        ----------
        activation : (d_model,) or (batch, d_model)

        Returns
        -------
        (..., 64, 3) logits
        """
        weights_float = self.weights.float()   # (64, 3, d_model)
        biases_float  = self.biases.float()    # (64, 3)
        activation_float = activation.float()
        if activation_float.dim() == 1:
            return torch.einsum("pcd,d->pc", weights_float, activation_float) + biases_float
        else:
            return torch.einsum("pcd,bd->bpc", weights_float, activation_float) + biases_float

    def cell_direction(self, pos: int, from_color: int, to_color: int) -> torch.Tensor:
        """
        Direction vector in d_model space pointing from_color toward to_color at pos.

        Returns
        -------
        (d_model,) float tensor
        """
        return (self.weights[pos, to_color] - self.weights[pos, from_color]).float()

    def trichrome_direction(self, cell_diffs: list[dict]) -> torch.Tensor:
        """
        Sum of cell_direction vectors across all differing cells.

        cell_diffs entries must have keys 'pos', 'color_a', 'color_b'
        (compatible with trichrome.diff() output and the transpositions JSON).

        Returns
        -------
        (d_model,) unnormalized float tensor
        """
        d_model = self.weights.shape[2]
        direction = torch.zeros(d_model, dtype=torch.float32)
        for diff in cell_diffs:
            direction = direction + self.cell_direction(
                int(diff["pos"]), int(diff["color_a"]), int(diff["color_b"])
            )
        return direction


def save_probes(probes: dict[int, TrichromeProbe], path: str | Path) -> None:
    """Save a layer→TrichromeProbe mapping to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            layer: {
                "weights": probe.weights,
                "biases":  probe.biases,
                "layer":   probe.layer,
            }
            for layer, probe in probes.items()
        },
        path,
    )


def load_probes(path: str | Path) -> dict[int, TrichromeProbe]:
    """Load probes saved with save_probes."""
    data = torch.load(Path(path), map_location="cpu")
    return {
        layer: TrichromeProbe(
            weights=entry["weights"],
            biases=entry["biases"],
            layer=entry["layer"],
        )
        for layer, entry in data.items()
    }
