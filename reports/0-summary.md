# Summary: Attacking the OthelloGPT World Model Claim

## The Claim

Li et al. (2023) train OthelloGPT — an 8-layer GPT — to predict next moves in Othello given a sequence of prior moves. They observe that a non-linear probe can recover board state from the model's residual stream with ~99% accuracy, and interpret this as evidence of an emergent world model. Nanda et al. (2023) strengthen the claim using linear probes and a causal intervention: they add a probe direction vector to the residual stream, causing the model to predict moves consistent with an illegal board state B' rather than the true state B. They conclude that the model genuinely tracks and reasons from an internal board representation.

We conduct three lines of attack.

---

## Methodological Critique: Representation is in the Eye of the Beholder

Nanda et al. write, in their own paper (page 6):

> "what constitutes a natural feature may be in the eye of the beholder."

We take this as the motto of the entire research programme — and as an inadvertent admission of its core weakness.

### The framework is unfalsifiable

The world model claim is operationalised entirely through metrics that the authors design, control, and evaluate: probe accuracy, intervention alignment, top-N error rates. There is no external ground truth for what a world model actually is. Within this framework, every negative result has an escape hatch:

- Intervention fails → probe was bad; use a better probe
- Probe accuracy is low → wrong data distribution; use the right one
- Alignment is low → wrong layer; sweep more layers

No result can falsify the claim, because any failure can be attributed to a methodological shortcoming rather than the absence of a world model. Our attacks are valuable precisely because they use the authors' own metrics against them — showing the metric is confounded (legal set overlap), the alignment collapses immediately beyond one step (rollout persistence), and the same intervention fails entirely on the championship model.

### Representations are found by search, not predicted

Neither Li nor Nanda specify *a priori* where the world model representation should live. Instead, both papers sweep over layers, probe types, intervention configurations, and numbers of flipped cells, reporting the configuration that produces the best numbers. Li reports "L_τ = 4 gives the best result" — best out of how many configurations tried? Nanda identifies layer 5 as optimal. These are post-hoc discoveries, not predictions.

This is a structural form of p-hacking at the architectural level. With enough probes, enough layers, and enough intervention configurations, you will always find *something* that looks like board state encoding. A sufficiently expressive probe applied to any intermediate representation of a model trained on a structured task will find structure — because the model has learned structure. The question is whether the found representation is causally privileged in the model's computation, and their answer to that (the intervention) is itself found by the same brute-force search.

The result is that Li's layer 4 and Nanda's layer 5 may be measuring different things in the same model, with no principled basis for choosing between them. The representation is wherever the probe finds it — which is, as Nanda admits, in the eye of the beholder.

### The probe defines the claim it verifies

In Li's intervention, the non-linear probe is used both to **define** B' in activation space (by gradient descent against the probe's loss) and to **verify** that the model tracks B' (by checking the probe's output post-intervention). The probe is doing double duty. If the probe is even slightly misaligned with the model's actual board-state circuit, both the forcing and the verification are correlated errors — the system finds activations that satisfy the probe and then confirms that those activations satisfy the probe. This is circular.

---

## Attack 1: The Model is Path-Dependent (Report 1)

**Setup.** A transposition is a pair of distinct move sequences that reach the same Othello board state at the same ply. If OthelloGPT reasons from a world model of board state, identical boards should produce identical output distributions regardless of the path taken. We extracted 12,894 transposition pairs from the championship corpus and measured distributional divergence.

**Results.**
- 25.6% of pairs disagree on the top-1 predicted move despite identical board states.
- Mean TV distance between output distributions: 0.230.
- Divergence is predicted by trichrome state (how many times each cell has been flipped) and ply — path-dependent features that carry no information about the current board.

**Conclusion.** The model's predictions depend on move history, not just board state. A genuine world model would be path-independent.

---

## Attack 2: Path-Dependent Activations are Causally Active (Report 2)

**Setup.** We test whether the path-dependence found in Report 1 is causal. For each transposition pair (seq_a, seq_b) reaching the same board state, we compute the activation difference (act_b - act_a) — which encodes only path-dependent history, not board state — and add it to seq_a's forward pass. We compare mixed-trichrome pairs (different path histories) against same-trichrome pairs (same path history, control).

**Results.**
- Adding path-dependent activation deltas increases illegal move predictions by **22.1pp** (mixed) vs **14.9pp** (control).
- The **7.2pp differential** is the key finding: the trichrome-specific component of path-dependent activations causally influences legal move prediction.
- A model reasoning purely from board state would not be disrupted by adding activation differences from sequences with the same board state — the board is unchanged.

**Conclusion.** Non-board-state, path-dependent features are causally active in the model's move prediction circuit. Nanda's board-state representation is one of multiple causally relevant signals, not a privileged world model.

---

## Attack 3: Nanda's Intervention Metric is Confounded (Report 3)

### Finding 1: Legal Set Overlap

**Setup.** Nanda's intervention flips one cell from Mine to Yours (or vice versa) to create an illegal board state B', then measures how many of the model's top-N predictions are legal for B'. We show this metric is confounded by the fact that flipping one cell barely changes the legal move set.

**Results.**
- Legal set overlap |legal(B) ∩ legal(B')| / |legal(B')| = **0.942–0.948** across all setups.
- The intervention shows 65–70% of the maximum possible improvement toward B' — but the model drifts only 14–17% away from B.
- These two numbers should be equal if the model genuinely shifted its board representation. The large gap is explained by the 94% overlap: most of the apparent improvement is "free" from overlap, not from genuine B' tracking.
- Nanda do not report the null baseline (original model vs legal(B')), which would have made this confound visible.

### Finding 2: Rollout Persistence

**Setup.** A genuine world model of B' should sustain consistent play for subsequent moves, not just the first. After the intervention, we roll out 10 more steps with no patching and measure the illegal move rate.

**Results.**

| Setup | Baseline illegal rate | Post-intervention | Ratio |
|---|---|---|---|
| Synthetic model, synthetic games | 0.2% | 7.5% | 37× |
| Synthetic model, championship games | 0.8% | 6.7% | 8× |
| Championship model, championship games | 21.0% | 21.8% (26.5% for m* unique to B') | ~1× |

On the synthetic model, post-intervention play is 8–37× more illegal than baseline. A model that had genuinely internalised B' as its world state would play at least as legally as the baseline. Instead it becomes significantly more erratic — consistent with the intervention being a one-shot disruption, not a persistent state change.

For the championship model, the intervention has no detectable effect overall. The rare cases where the intervention forces a move genuinely unique to B' (1.7% of positions) show the worst subsequent play (26.5% illegal), the opposite of what a world model would predict.

### Finding 3: The Intervention Fails on the Championship Model

**Setup.** Li et al. trained two OthelloGPT models — one on 20M synthetic (random) games, one on ~140K championship (strategic) games. Both play Othello. Nanda tested only the synthetic model and never disclosed this choice as a limitation.

**Results.**

| Setup | Probe accuracy | Intervention improvement |
|---|---|---|
| Synthetic model, synthetic games | 0.974 | 70% |
| Championship model, championship games | 0.812 | <1% |

The intervention that produces a 70% improvement on the synthetic model produces no signal on the championship model. Two models trained on the same game, with the same intervention applied, give opposite results. This is not what a world model predicts — the rules of Othello do not change based on training distribution.

The lower probe accuracy on the championship model (0.812) is a partial confound for the intervention result. But the low accuracy is itself evidence: if the championship model had a genuine world model of Othello, board state should be just as linearly decodable from its representations. Nanda never tested this, despite the championship model being publicly available.

### Finding 4: Memorisation and the Cross-Dataset Test

OthelloGPT was trained on game sequences drawn from the same pool used for probing and intervention. Any probe accuracy on seen sequences is potentially confounded by sequence memorisation. The synthetic model, when evaluated on championship games it has never seen, shows reduced probe accuracy (0.927 vs 0.974). This suggests part of what the probe recovers on synthetic games is memorisation artifacts, not a general board state computation.

---

## Overall Conclusion

| Claim | Evidence against |
|---|---|
| Model reasons from board state | 25.6% rank-1 disagreement on transpositions; path-dependent activations causally increase illegal moves |
| Intervention proves causal board tracking | 94% legal set overlap explains 65–70% apparent improvement; model drifts only 14–17% from B |
| World model persists beyond one step | Post-intervention rollout is 8–37× more illegal than baseline on synthetic model |
| World model is general | Intervention is 70% effective on synthetic model, <1% on championship model — same game, same intervention |
| Representation reflects the game | Probe accuracy drops cross-dataset (0.974 → 0.927); championship model representations less linearly decodable |

## Li's Unnatural Board State Experiment (pending replication)

Li et al. run the same intervention on two benchmarks: a "natural" subset of 1,000 positions reachable by legal play, and an "unnatural" subset of 1,000 positions unreachable by any legal game sequence. They report average top-N errors of 0.12 (natural) and 0.06 (unnatural) after intervention, compared to baselines of 2.68 and 2.59. They interpret the low error on unnatural states as strong evidence that the representation is causal even for board states the model has never encountered.

Our attacks apply directly to this experiment:

**The same legal set overlap confound applies.** Li's top-N error metric has the same structure as Nanda's. If flipping cells to construct the unnatural state barely changes the legal move set, most of the apparent improvement is free from overlap. Li does not report the null baseline (original model vs legal(B')) for either benchmark.

**The 0.06 error on illegal states is suspicious.** It is lower than the 0.12 on legal states, despite illegal states being "far from anything encountered in the training distribution." If the model genuinely tracked an interventionally-imposed illegal board state, steering should be harder, not easier.

We tested whether this is explained by smaller legal move sets for illegal states. For single-cell flips (n_flips=1, n=49,961), the mean legal move set sizes are:
- mean |legal(B)| = 9.09
- mean |legal(B')| = 9.26

B' has slightly *more* legal moves than B on average — the set size hypothesis is not supported for single-cell flips. The metric confound for single-cell flips is purely about overlap (94.8%), not set size. This sharpens the argument: even without invoking set size effects, the 94.8% overlap alone explains the gap between 70% apparent improvement and 17% actual drift from B.

Whether the set size hypothesis holds for Li's specific unnatural construction (which likely flips many more cells) remains untested. Li does not report their exact construction procedure in sufficient detail to replicate precisely.

**Pending experiments:**
1. Replicate Li's natural/unnatural benchmark, computing legal set overlap, null baseline, and mean legal set sizes for both subsets.
2. Run rollout persistence on unnatural states: if the model genuinely internalised the illegal board state, subsequent play should remain consistent with it.

---

## Conclusion

What OthelloGPT has is a board-state-shaped projection of move history — a useful and human-legible intermediate representation that any powerful sequence model will learn when trained on Othello. This is not the same as a world model. Nanda's intervention appears to work because the two legal sets are nearly identical (94% overlap), not because the model genuinely shifted its internal board representation. When the metric is corrected for this — by measuring drift from B alongside improvement toward B', and by testing persistence over subsequent moves — the evidence for a world model largely disappears.
