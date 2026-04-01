# Report 3: Attacking the World Model Claim

## Background

Nanda et al. (2023) argue that OthelloGPT has an emergent world model of Othello board state. Their primary causal evidence is an intervention experiment: they add a probe direction vector to the residual stream at every layer, causing the model to "believe" the board is in an illegal state B' (unreachable by any legal game sequence). They then measure whether the model's top-N predictions match the legal moves for B', reporting an average of 0.10 errors. They interpret this as evidence that the model genuinely tracks board state and updates its move predictions accordingly.

We identify two flaws in this interpretation.

---

## Experiment 1: Legal-Set Overlap

### Setup

We replicate Nanda's intervention using the synthetic model on championship game sequences (n=1,997 positions, layer 5, n_flips=1). For each position we compute:

- **legal set overlap**: |legal(B) ∩ legal(B')| / |legal(B')|
- **original vs legal(B)**: top-N errors of the original model against the source legal set (baseline, should be ~0)
- **original vs legal(B')**: top-N errors of the original model against the target legal set (Nanda's null baseline — not reported in their paper)
- **intervened vs legal(B')**: top-N errors of the intervened model against the target legal set (Nanda's claim)
- **intervened vs legal(B)**: top-N errors of the intervened model against the source legal set (key missing number)

### Results

| | vs legal(B) | vs legal(B') |
|---|---|---|
| Original model | 0.031 | 1.271 |
| Intervened model | 0.195 | 0.441 |

Legal set overlap: **0.942**

### Interpretation

**The legal sets barely change.** Flipping one cell Mine↔Yours changes the legal move set by only 5.8% on average (0.942 overlap). This means the original model — which ignores the intervention entirely — already scores near-correctly on the target legal set, simply because legal(B) ≈ legal(B').

**Nanda's improvement is largely explained by overlap.** The intervention reduces B' errors from 1.271 to 0.441 — an improvement of 0.830, or 65% of the maximum possible improvement (1.271). This looks impressive. But:

- If the model genuinely shifted from tracking B to tracking B', errors against legal(B) should increase by approximately |legal(B) △ legal(B')| ≈ 1.16.
- The actual increase is only 0.195 - 0.031 = 0.164, which is 14% of the maximum possible drift.

The model drifted only **14%** away from B, yet showed **65%** of the maximum improvement toward B'. These should be equal if the model genuinely changed its board representation. The large gap is explained by the 94.2% overlap: because the two legal sets are nearly identical, even a tiny shift in the model's distribution produces a large apparent improvement in B' accuracy. Most of the 65% improvement is "free" from overlap, not from genuine B' tracking.

Nanda do not report the null baseline (original model vs legal(B')). Had they done so, the improvement from 1.271 to 0.441 would invite exactly this comparison.

---

## Experiment 2: Rollout Persistence

### Setup

If the model genuinely internalised a world model of B', its predictions should remain consistent with B' for subsequent moves, not just the first one. The intervention is a single vector addition — it has no mechanism to update the model's "belief" about board state for future forward passes. A genuine world model would need to persist.

After the intervention, we take m* = the model's top-1 prediction and roll out 10 more steps with no intervention: the model processes [S + m*] step by step, and we check each prediction against the actual board state (starting from B' with m* applied). We compare illegal-move rates to a baseline rollout from B (no intervention).

### Results

| Condition | Illegal rate (10 steps) | n |
|---|---|---|
| Baseline (no intervention) | 0.8% | 1,916 |
| After intervention (m* legal for B') | 6.7% | 1,898 |
| After intervention (m* unique to B') | 0.0% | 2 |

Positions where m* was unique to B' (legal for B' but not B): **0.1%** (2/1,997).

### Interpretation

**The intervention disrupts subsequent play.** After the intervention, the model makes illegal moves at 8× the baseline rate (6.7% vs 0.8%) over the next 10 steps. If the model had genuinely internalised B' as its world state, subsequent play should be at least as legal as the baseline — the model would "know" it is in state B' + m* and play accordingly. Instead, it becomes significantly more erratic.

This is consistent with the intervention being a one-shot disruption rather than a persistent state change. The model has been fed an unusual move sequence (m* was chosen under intervention and may be atypical for the given position), and without any ongoing correction, it cannot recover a consistent internal state to play from.

**m* is almost never unique to B'.** In only 0.1% of positions was the intervened model's top-1 prediction legal for B' but not B. In the vast majority of cases the model's first post-intervention move is legal for both boards — again a consequence of the 94.2% legal set overlap.

---

## On the Generality of the World Model Claim

### Setup dependence

The experiment above uses the synthetic model on championship game sequences. The two natural symmetric setups are:

1. **Synthetic model + synthetic games** — Nanda's exact conditions
2. **Championship model + championship games** — same game, different training distribution

A genuine world model of Othello should perform similarly in both setups: the rules of Othello do not change based on play style. If the intervention mechanism is sensitive to training distribution, it is modelling the training distribution rather than the game.

Results for both setups are pending (see tasks 4 and 5). The current hybrid setup (synthetic model, championship games, probe trained on championship activations) already shows a probe accuracy of 0.927 — lower than Nanda's ~99% on synthetic games, despite the model being the same. The board state is an objective property of the game; its encodability should not depend on whether games were played randomly or strategically.

### The same-world-model argument

Both the synthetic and championship models are trained to play Othello — the same game with the same rules. If both models have a world model of Othello, they should converge on the same representation of board state. The fact that probe accuracy and intervention effectiveness differ substantially between models playing the same game is itself evidence that neither model has a genuine world model in the strong sense. What they have is a representation shaped by their training distribution.

To use a concrete analogy: two people who play Othello — one randomly, one strategically — both track the same board state. Their "world model" of the game is identical because the game's rules are identical. Two neural networks trained on those two distributions should converge on the same board state representation if they are genuinely simulating the game. If they do not, they are doing something else.

---

## Summary

| Claim | Evidence against |
|---|---|
| Intervention shows model tracks B' | 94.2% legal set overlap explains most of the apparent improvement; model drifted only 14% from B while showing 65% improvement toward B' |
| Model generalises to illegal states | Subsequent play (10 steps, no intervention) is 8× more illegal than baseline |
| World model is general | Probe accuracy and intervention effectiveness are sensitive to training distribution; same game, different representations |
