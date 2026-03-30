"""
src/othellogpt_deconstruction/intervention/hooks.py

Activation patching for OthelloGPT intervention experiments.

Two intervention modes:

1. Delta patching (primary)
   Compute the activation difference between two sequences at each layer,
   then add that delta to one sequence's activations. Tests whether the
   activation difference is causally upstream of the output difference.

2. Direct patching
   Replace activations from sequence A with activations from sequence B
   at specified layers. Tests whether specific layers carry the causally
   relevant information.

Both modes operate on the residual stream at the last token position,
matching Li et al. and Nanda & Lee's methodology.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from mingpt.model import GPT

from othellogpt_deconstruction.core.tokenizer import stoi, alg_to_pos, BLOCK_SIZE, PAD_ID


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

@dataclass
class ActivationStore:
    """Stores residual stream activations at the last token position per layer."""
    activations: dict[int, torch.Tensor] = field(default_factory=dict)

    def get(self, layer: int) -> torch.Tensor | None:
        return self.activations.get(layer)

    def layers(self) -> list[int]:
        return sorted(self.activations.keys())

    def __len__(self) -> int:
        return len(self.activations)


@contextmanager
def collect_activations(model: GPT, seq_length: int):
    """
    Context manager that hooks into all transformer blocks and collects
    the residual stream at the last token position after each block.

    Yields an ActivationStore populated after the forward pass.

    Usage
    -----
        store = ActivationStore()
        with collect_activations(model, len(sequence)) as store:
            probs = get_distribution(model, sequence)
        layer_0_acts = store.get(0)
    """
    store = ActivationStore()
    hooks = []
    last_pos = seq_length - 1

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # output is the residual stream after this block: (batch, seq, embd)
            store.activations[layer_idx] = output[0, last_pos, :].detach().clone()
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

@dataclass
class InterventionResult:
    """Result of an intervention experiment."""
    probs_original:    torch.Tensor    # original distribution for seq_a
    probs_target:      torch.Tensor    # original distribution for seq_b
    probs_intervened:  torch.Tensor    # distribution after intervention
    layers_intervened: list[int]       # which layers were patched
    alpha:             float           # scaling factor applied


@contextmanager
def patch_activations(
    model: GPT,
    seq_length: int,
    patches: dict[int, torch.Tensor],
    alpha: float = 1.0,
    replace: bool = False,
):
    """
    Context manager that modifies residual stream activations at specified
    layers during a forward pass.

    Parameters
    ----------
    model      : GPT model
    seq_length : length of the input sequence (for last-token indexing)
    patches    : dict mapping layer_index -> patch vector of shape (embd,)
    alpha      : scaling factor for patches (1.0 = full patch); ignored when replace=True
    replace    : if True, set the activation to patches[layer] directly instead
                 of adding alpha * patches[layer] to the existing activation

    Usage
    -----
        with patch_activations(model, len(seq), {0: delta_0, 1: delta_1}):
            probs = get_distribution(model, seq)
    """
    hooks = []
    last_pos = seq_length - 1

    def make_hook(layer_idx: int, patch: torch.Tensor):
        def hook(module, input, output):
            # output shape: (batch, seq, embd)
            patched = output.clone()
            if replace:
                patched[0, last_pos, :] = patch
            else:
                patched[0, last_pos, :] = patched[0, last_pos, :] + alpha * patch
            return patched
        return hook

    for layer_idx, patch in patches.items():
        block = model.blocks[layer_idx]
        hooks.append(block.register_forward_hook(make_hook(layer_idx, patch)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------------------------------------------------------
# High-level intervention API
# ---------------------------------------------------------------------------

def _encode_sequence(sequence: list[str], device: torch.device) -> torch.Tensor:
    tokens = [stoi[alg_to_pos(m)] for m in sequence]
    padded = tokens + [PAD_ID] * (BLOCK_SIZE - len(tokens))
    return torch.tensor([padded], dtype=torch.long, device=device)


def _forward(model: GPT, x: torch.Tensor, seq_length: int) -> torch.Tensor:
    """Run forward pass and return softmax distribution at last token position."""
    with torch.no_grad():
        logits, _ = model(x)
    last_logits = logits[0, seq_length - 1, :].clone()
    last_logits[PAD_ID] = float("-inf")
    return torch.softmax(last_logits, dim=-1)


def delta_intervention(
    model:      GPT,
    seq_a:      list[str],
    seq_b:      list[str],
    layers:     list[int] | None = None,
    alpha:      float = 1.0,
    device:     torch.device | None = None,
) -> InterventionResult:
    """
    Intervene on seq_a by adding the activation delta (seq_b - seq_a)
    at each specified layer, then observe whether predictions shift
    toward seq_b.

    Parameters
    ----------
    model   : loaded GPT model
    seq_a   : source sequence (activations to patch)
    seq_b   : target sequence (provides the delta direction)
    layers  : which layers to intervene on (default: all layers)
    alpha   : scaling factor (1.0 = full shift toward seq_b)
    device  : torch device

    Returns
    -------
    InterventionResult with original and intervened distributions.
    """
    if device is None:
        device = next(model.parameters()).device

    n_layers = len(model.blocks)
    if layers is None:
        layers = list(range(n_layers))

    x_a = _encode_sequence(seq_a, device)
    x_b = _encode_sequence(seq_b, device)

    # Collect activations for both sequences
    with collect_activations(model, len(seq_a)) as store_a:
        probs_a = _forward(model, x_a, len(seq_a))

    with collect_activations(model, len(seq_b)) as store_b:
        probs_b = _forward(model, x_b, len(seq_b))

    # Compute delta at each layer
    patches = {}
    for layer in layers:
        act_a = store_a.get(layer)
        act_b = store_b.get(layer)
        if act_a is not None and act_b is not None:
            patches[layer] = (act_b - act_a).to(device)

    # Apply intervention to seq_a
    with patch_activations(model, len(seq_a), patches, alpha=alpha):
        with torch.no_grad():
            probs_intervened = _forward(model, x_a, len(seq_a))

    return InterventionResult(
        probs_original=probs_a.cpu(),
        probs_target=probs_b.cpu(),
        probs_intervened=probs_intervened.cpu(),
        layers_intervened=layers,
        alpha=alpha,
    )


def direct_intervention(
    model:      GPT,
    seq_a:      list[str],
    seq_b:      list[str],
    layers:     list[int] | None = None,
    alpha:      float = 1.0,
    device:     torch.device | None = None,
) -> InterventionResult:
    """
    Intervene on seq_a by replacing its activations with seq_b's activations
    at each specified layer (rather than adding the delta).

    This is a stronger intervention — it fully transplants seq_b's
    representation into seq_a's forward pass.
    """
    if device is None:
        device = next(model.parameters()).device

    n_layers = len(model.blocks)
    if layers is None:
        layers = list(range(n_layers))

    x_a = _encode_sequence(seq_a, device)
    x_b = _encode_sequence(seq_b, device)

    with collect_activations(model, len(seq_a)) as store_a:
        probs_a = _forward(model, x_a, len(seq_a))

    with collect_activations(model, len(seq_b)) as store_b:
        probs_b = _forward(model, x_b, len(seq_b))

    patches = {}
    for layer in layers:
        act_b = store_b.get(layer)
        if act_b is not None:
            patches[layer] = act_b.to(device)

    with patch_activations(model, len(seq_a), patches, replace=True):
        with torch.no_grad():
            probs_intervened = _forward(model, x_a, len(seq_a))

    return InterventionResult(
        probs_original=probs_a.cpu(),
        probs_target=probs_b.cpu(),
        probs_intervened=probs_intervened.cpu(),
        layers_intervened=layers,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def intervention_success(
    result: InterventionResult,
    legal_positions: list[int],
) -> dict:
    """
    Measure how much the intervention moved predictions toward the target.

    Returns a dict with:
        tv_before  : TV distance between original and target
        tv_after   : TV distance between intervened and target
        tv_reduction : how much TV distance was reduced (positive = success)
        rank1_before : whether rank-1 agreed before intervention
        rank1_after  : whether rank-1 agrees after intervention
    """
    from othellogpt_deconstruction.analysis.metrics import metrics_full

    m_before = metrics_full(result.probs_original, result.probs_target)
    m_after  = metrics_full(result.probs_intervened, result.probs_target)

    return {
        "tv_before":     m_before["tv_distance_full"],
        "tv_after":      m_after["tv_distance_full"],
        "tv_reduction":  m_before["tv_distance_full"] - m_after["tv_distance_full"],
        "js_before":     m_before["js_divergence_full"],
        "js_after":      m_after["js_divergence_full"],
        "rank1_before":  m_before["rank1_agreement_full"],
        "rank1_after":   m_after["rank1_agreement_full"],
    }
