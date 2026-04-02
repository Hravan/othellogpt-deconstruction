# Report 3: Attacking the World Model Claim

## Background

Nanda et al. (2023) argue that OthelloGPT has an emergent world model of Othello board state. Their primary causal evidence is an intervention experiment: they add a probe direction vector to the residual stream at every layer, causing the model to "believe" the board is in an illegal state B' (unreachable by any legal game sequence). They then measure whether the model's top-N predictions match the legal moves for B', reporting an average of 0.10 errors. They interpret this as evidence that the model genuinely tracks board state and updates its move predictions accordingly.

We identify two flaws in this interpretation. We test them across three setups: the synthetic model on championship games (cross-dataset), the synthetic model on synthetic games (Nanda's own conditions), and the championship model on championship games.

---

## Experiment 1: Legal-Set Overlap (cross-dataset: synthetic model on championship games)

### Setup

We replicate Nanda's intervention using the synthetic model on held-out championship game sequences (n=1,997 positions, layer 5, n_flips=1). This is a cross-dataset setup: the model was trained on synthetic games but evaluated on championship games it has never seen. For each position we compute:

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

## Experiment 2: Rollout Persistence (cross-dataset: synthetic model on championship games)

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

## Experiment 1 & 2 (clean replication): Synthetic model on synthetic games

### Setup

We repeat both experiments using the synthetic model on held-out synthetic game sequences (n=49,961 positions, layer 5, n_flips=1). The probe was trained on a separate synthetic train split (225K games, 200K sample cap) and achieves 0.974 mean train accuracy. This is the closest replication of Nanda's exact conditions.

### Results

| | vs legal(B) | vs legal(B') |
|---|---|---|
| Original model | 0.012 | 1.290 |
| Intervened model | 0.225 | 0.379 |

Legal set overlap: **0.948**

Rollout (10 steps):

| Condition | Illegal rate | n |
|---|---|---|
| Baseline (no intervention) | 0.2% | 47,929 |
| After intervention (m* legal for B') | 7.5% | 47,455 |
| After intervention (m* unique to B') | 6.3% | 183 |

Positions where m* was unique to B': **0.4%** (186/49,961)

### Interpretation

The pattern from the cross-dataset experiment holds and is more extreme on the model's own training distribution.

**Experiment 1:** The intervention shows 70% of the maximum possible improvement toward B' (1.290 → 0.379, improvement of 0.911). But the model drifted only 17% away from B (0.225 - 0.012 = 0.213, vs maximum ~1.290). The gap between apparent B' improvement and actual B drift is explained by the 94.8% legal set overlap.

**Experiment 2:** The post-intervention rollout illegal rate is **37× the baseline** (7.5% vs 0.2%). The baseline is lower than in the cross-dataset setup (0.2% vs 0.8%) because the synthetic model is maximally adapted to synthetic sequences — making the intervention's disruption even more severe by contrast. A model that genuinely internalised B' as its world state would play at least as legally as the baseline; instead it degrades dramatically.

These results replicate under Nanda's own conditions and strengthen both attacks.

---

## Experiment 1 & 2: Championship model on championship games

### Setup

We repeat both experiments using the championship model evaluated on held-out championship game sequences (n=26,478 positions, layer 5, n_flips=1). The probe was trained on the championship train split. This is the symmetric clean setup to the synthetic-on-synthetic experiment above.

### Results

| | vs legal(B) | vs legal(B') |
|---|---|---|
| Original model | 5.481 | 5.776 |
| Intervened model | 5.635 | 5.735 |

Legal set overlap: **0.941**

Rollout (10 steps):

| Condition | Illegal rate | n |
|---|---|---|
| Baseline (no intervention) | 21.0% | 25,436 |
| After intervention (m* legal for B') | 21.8% | 22,047 |
| After intervention (m* unique to B') | 26.5% | 453 |

Positions where m* was unique to B': **1.7%** (457/26,478)

### Interpretation

**Probe quality caveat.** The linear probe trained on championship model activations achieves 0.812 mean train accuracy, compared to 0.974 for the synthetic model. This is a methodological limitation: a weaker probe produces a noisier intervention direction, which could partially explain why the intervention fails. The championship intervention result should therefore be treated as inconclusive rather than definitive.

However, the low probe accuracy is itself informative. Board state is an objective property of the game — if the championship model had a genuine world model of Othello, its representations should be at least as linearly decodable as the synthetic model's. The 16-point gap in probe accuracy (0.974 vs 0.812) suggests the championship model encodes board state in a less linearly accessible form, or encodes something else entirely.

Notably, **Nanda et al. never tested their probes or intervention on the championship model**, despite Li et al. having trained and released it. All of Nanda's experiments use the synthetic model exclusively — the one where linear probes achieve ~99% accuracy. This is a significant omission: a claim about emergent world models should hold across both models trained on the same game, and the authors never checked whether it does.

**The rollout tells a sharper story.** The baseline illegal rate is already 21% — the championship model degrades quickly in free play even without intervention, suggesting it does not maintain consistent internal state across moves. After intervention, the overall rate barely changes (21.8%). But when m* is unique to B' — the only cases where the intervention actually forces a move that differs between B and B' — the illegal rate jumps to **26.5%**, 26% above baseline. These are precisely the cases where the intervention is doing what Nanda claims: pushing the model toward a genuinely different board state. And they are precisely the cases where the model's subsequent play is worst. A model with a genuine world model of B' would play more consistently after such a move, not less. This rollout result is less sensitive to probe quality than the top-N metric, since it measures the model's free play after the intervention rather than the intervention's direct effect on predictions.

---

## On the Generality of the World Model Claim

### The same-world-model argument

Both the synthetic and championship models are trained to play Othello — the same game with the same rules. If both models have a world model of Othello, they should converge on the same representation of board state and respond similarly to the same intervention. They do not:

| Setup | Baseline error | Intervention improvement | Rollout illegal rate |
|---|---|---|---|
| Synthetic model, synthetic games | 0.012 | 70% | 7.5% (37× baseline) |
| Synthetic model, championship games | 0.031 | 65% | 6.7% (8× baseline) |
| Championship model, championship games | 5.481 | <1% | 21.8% (~baseline) |

The intervention works on the synthetic model regardless of which game sequences are used, and fails entirely on the championship model. The difference is training distribution, not the game. A genuine world model of Othello — in the sense of a representation of the game's rules — would not behave this way.

To use a concrete analogy: two people who play Othello — one randomly, one strategically — both track the same board state. Their "world model" of the game is identical because the game's rules are identical. Two neural networks trained on those two distributions should converge on the same board state representation if they are genuinely simulating the game. If they do not, they are doing something else.

### Memorisation and the cross-dataset test

The synthetic model was trained on 20M synthetic game sequences drawn from the same pool as our test set. The championship model was trained on ~112K championship games from the same dataset. Any probe accuracy or intervention effect on seen sequences is potentially confounded by the model having memorised those sequences rather than computed board state.

The cross-dataset test operationalises this directly: a genuine world model would represent board state correctly for any legal game sequence, including sequences from a distribution it was never trained on. The synthetic model, when evaluated on championship games it has never seen, shows reduced probe accuracy (0.927 vs 0.974 on synthetic games). This drop is evidence that part of what the probe recovers on synthetic games is a memorisation artifact — the model's representation of seen sequences — rather than a general board state computation.

---

## Summary

| Claim | Evidence against |
|---|---|
| Intervention shows model tracks B' | 94% legal set overlap explains most of the apparent improvement; model drifted only 14–17% from B while showing 65–70% improvement toward B' |
| Model generalises to illegal states | Post-intervention rollout is 8–37× more illegal than baseline on the synthetic model; championship model is already at 21% and the intervention makes the unique-to-B' cases worst (26.5%) |
| World model is general | Intervention is 70% effective on synthetic model, <1% effective on championship model — same game, same intervention, opposite results |
| Representation reflects the game, not training data | Cross-dataset probe accuracy drop (0.974 → 0.927) and intervention failure on championship model indicate representations are shaped by training distribution, not the game's rules |
