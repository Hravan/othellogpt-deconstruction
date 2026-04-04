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
| Synthetic model, synthetic games (natural) | 0.3% | 6.8% | 23× |
| Synthetic model, championship games | 0.8% | 6.7% | 8× |
| Championship model, championship games (Nanda) | 21.0% | 21.8% (26.5% for m* unique to B') | ~1× |
| Championship model, championship games (Li) | 21.5% | 21.9% (28.5% for m* unique to B') | ~1× |
| Synthetic model, **unnatural** benchmark | **0.0%** | **20.4%** (30.9% for m* unique to B') | **∞** |

On the synthetic model with natural positions, post-intervention play is 8–23× more illegal than baseline. On the unnatural benchmark the model plays with 0.0% illegal rate before intervention — and 20.4% after. The ratio is effectively infinite: the intervention takes a model playing perfectly on these board states and makes it play illegally one in five times. The cases where m* is genuinely unique to B' (2.3% of positions) are the worst: 30.9% illegal rate.

A model that had genuinely internalised B' as its world state would play at least as legally as the baseline. Instead it becomes maximally erratic — consistent with the intervention being a one-shot disruption, not a persistent state change.

For the championship model, the intervention has no detectable effect overall. The rare cases where the intervention forces a move genuinely unique to B' (1.7% of positions) show the worst subsequent play (26.5% illegal), the opposite of what a world model would predict.

### Finding 3: The Intervention Fails on the Championship Model

**Setup.** Li et al. trained two OthelloGPT models — one on 20M synthetic (random) games, one on ~140K championship (strategic) games. Both play Othello. Nanda tested only the synthetic model and never disclosed this choice as a limitation. We run Li's own intervention (gradient descent against the non-linear probe, layers 4–8) on the championship model using the championship probe checkpoints.

**Results.**

| Setup | Intervention | Probe accuracy | topN baseline vs B | Improvement toward B' | Rollout baseline | Rollout post-intervention |
|---|---|---|---|---|---|---|
| Synthetic model, synthetic games | Li (gradient descent) | 0.974 | ~0 | 70–80% | 0.3% | 6.8–20.4% |
| Championship model, championship games | Nanda (linear probe add) | 0.812 | 5.29 | <1% | 21.0% | 21.8% (26.5% unique) |
| Championship model, championship games | Li (gradient descent) | 0.812 | 5.29 | 8.4% | 21.5% | 21.9% (28.5% unique) |

The intervention produces 8.4% improvement on the championship model versus 70–80% on the synthetic model. The rollout illegal rate is essentially unchanged (21.5% → 21.9%). Two models trained on the same game, with the same intervention applied, give opposite results.

**The topN metric is unsuitable for the championship model.** The topN_errors metric is a symmetric difference between the model's top-N predictions and the full legal move set. It penalises both illegal predictions and legal moves the model does not predict. The synthetic model, trained on 20M random games, sees all legal moves with roughly equal frequency and learns a distribution that covers the full legal set — so the topN baseline is near zero. The championship model, trained on strategic games, learns a concentrated distribution over a few good moves and ignores most legal moves that are strategically irrelevant. The high topN baseline (5.29) is therefore largely a metric artefact: the championship model does not predict all legal moves because it was not trained to. Li's metric implicitly assumes a uniform prior over legal moves, which is precisely what random-game training produces and strategic training does not.

The rollout illegal rate is the appropriate metric for the championship model: it measures whether the model's actual top-1 pick is legal, without penalising strategic selectivity. The rollout result is unambiguous — the intervention has no effect.

**Neither Li nor Nanda tested the championship model.** Li trained and published both models but ran all intervention experiments on the synthetic model only, without disclosing this as a limitation. This is not a minor omission: if the intervention proves a general world model of Othello, it should hold for any model trained on Othello, regardless of training distribution. The rules of the game do not change. Choosing to report only the model where the intervention works — while publishing the other model without testing it — is selective reporting. The consistent treatment would be: if you probe the championship model, you also intervene on it.

The lower probe accuracy on the championship model (0.812 vs 0.974) is a partial confound — a weaker probe produces a weaker intervention target. But the low accuracy is itself evidence: a genuine world model of Othello should be linearly decodable from the representations of any model trained on Othello.

### Finding 4: Memorisation and the Cross-Dataset Test

OthelloGPT was trained on game sequences drawn from the same pool used for probing and intervention. Any probe accuracy on seen sequences is potentially confounded by sequence memorisation. The synthetic model, when evaluated on championship games it has never seen, shows reduced probe accuracy (0.927 vs 0.974). This suggests part of what the probe recovers on synthetic games is memorisation artifacts, not a general board state computation.

---

---

## Yuan et al. (2025): Representation Alignment

Yuan et al. (2025, "Revisiting the Othello World Model Hypothesis") use MUSE representation alignment to show that independently trained Othello-playing models — across different architectures (GPT-2, Bart, Mistral, LLaMA-2) — converge on similar internal representations, with cosine similarity of 80–96% after Procrustes alignment. They also show that adjacent tiles tend to have similar embeddings (Section 5), which they interpret as evidence of learned spatial geometry.

We acknowledge these results. Something structurally similar is being computed across independently trained models. However, what that something is remains unknown. High representation similarity after alignment is consistent with both:

1. All models converge on a shared world model of board state
2. All models are non-lossy encoders of move sequences, and board state is recoverable from any non-lossy encoding of a structured sequential input — without requiring an explicit internal world state

These cannot be distinguished by alignment scores alone. Similarly, tile embedding proximity reflects co-occurrence statistics (adjacent tiles tend to be simultaneously legal throughout training) as much as any spatial world model. The alignment result is a necessary but not sufficient condition for the world model claim.

---

---

## The Unexplained Learning Mechanism

The world model claim, taken seriously, asserts that OthelloGPT learned — from sequences of integers in range 0–63 — a rich physical theory of Othello: that there are two players, a spatial 8×8 board, discs belonging to each player, and a flip mechanic that changes ownership across turns. No board image, no rules, no domain knowledge was injected into the training data or the model architecture. The model sees only token sequences.

If this is true, it demands an explanation of the learning mechanism. How does a sequence model construct a physical world model from token co-occurrence statistics alone? Nanda, Li, and Yuan do not address this question. They establish that board state is recoverable from representations and that interventions partially redirect behavior — and then conclude that a world model exists. The mechanism by which such a model could arise is left entirely unaccounted for.

The parsimonious alternative requires no such explanation: the model learns a compressed statistical summary of move sequences that happens to be human-interpretable as board state. Board state is a deterministic function of the move sequence; any model that learns the distribution of legal continuations will implicitly encode it as a predictive feature — without representing players, boards, or discs as explicit internal concepts. The probe finds this encoding and the authors interpret it as a world model. But a statistical regularity that is *consistent with* a world model is not the same as a world model.

The burden of proof lies with those making the stronger claim. Showing that board state is decodable, that representations align across models, and that interventions partially work does not discharge that burden — it shows the model has learned something structured from structured input, which is the null hypothesis for any competent sequence model trained on a rule-governed task.

---

## Overall Conclusion

| Claim | Evidence against |
|---|---|
| Model reasons from board state | SS = 0.230, CR = 25.6%; path-dependent activations causally increase illegal moves |
| Intervention proves causal board tracking | 94% legal set overlap explains 65–70% apparent improvement; model drifts only 14–17% from B |
| World model persists beyond one step | Post-intervention rollout is 8–37× more illegal than baseline on synthetic model |
| World model is general | Intervention is 70–80% effective on synthetic model, 8.4% on championship model (rollout: 21.5% → 21.9%, no effect) — same game, same intervention; Li never tested championship model |
| Representation reflects the game | Probe accuracy drops cross-dataset (0.974 → 0.927); championship model representations less linearly decodable |

## Li's Unnatural Board State Experiment

Li et al. run the same intervention on two benchmarks: a "natural" subset of 1,000 positions reachable by legal play, and an "unnatural" subset of 1,000 positions unreachable by any legal game sequence. They report average top-N errors of 0.12 (natural) and 0.06 (unnatural) after intervention, compared to baselines of 2.68 and 2.59. They interpret the low error on unnatural states as strong evidence that the representation is causal even for board states the model has never encountered.

Our attacks apply directly to this experiment:

**The same legal set overlap confound applies.** Li's top-N error metric has the same structure as Nanda's. If flipping cells to construct the unnatural state barely changes the legal move set, most of the apparent improvement is free from overlap. Li does not report the null baseline (original model vs legal(B')) for either benchmark.

**The 0.06 error on illegal states is suspicious.** It is lower than the 0.12 on legal states, despite illegal states being "far from anything encountered in the training distribution." If the model genuinely tracked an interventionally-imposed illegal board state, steering should be harder, not easier.

### Natural benchmark replication (n=998)

We replicated Li's intervention on positions drawn from legal synthetic game sequences. All metrics are computed with a single-cell flip (BLACK↔WHITE).

| Metric | Value |
|---|---|
| mean \|legal(B)\| | 9.06 |
| mean \|legal(B')\| | 9.24 |
| Legal set overlap | 0.949 |
| Original model vs legal(B) | 0.012 (baseline; expected ~0) |
| Original model vs legal(B') | 1.216 (null baseline — **not reported by Li**) |
| Intervened model vs legal(B') | 0.240 (Li's claimed metric) |
| Intervened model vs legal(B) | 0.479 (drift from B) |
| Improvement toward B' | 80.2% of maximum |
| Drift from B | 38.4% of maximum |

The confound is present here as in the Nanda case: 80.2% apparent improvement but only 38.4% drift from B, a 42pp gap explained by the 94.9% overlap. Li's claimed improvement is substantially inflated.

Rollout persistence (10 steps, n=998):

| | Rate |
|---|---|
| Baseline illegal rate (no intervention) | 0.3% (n=955) |
| Post-intervention illegal rate (m* legal for B') | 6.8% (n=953) |
| Post-intervention illegal rate (m* unique to B') | 0.0% (n=11; sample too small) |

Post-intervention play is ~23× more illegal than baseline. Only 1.1% of positions (11/998) produce an m* genuinely unique to B'. The 0.0% illegal rate for that subset is not interpretable — n=11 is too small to detect a 6.8% signal.

We tested whether this is explained by smaller legal move sets for illegal states. For single-cell flips (n_flips=1, n=49,961), the mean legal move set sizes are:
- mean |legal(B)| = 9.09
- mean |legal(B')| = 9.26

B' has slightly *more* legal moves than B on average — the set size hypothesis is not supported for single-cell flips. The metric confound for single-cell flips is purely about overlap (94.8%), not set size. This sharpens the argument: even without invoking set size effects, the 94.8% overlap alone explains the gap between apparent improvement and actual drift from B.

Whether the set size hypothesis holds for Li's specific unnatural construction (which likely flips many more cells) remains untested. Li does not report their exact construction procedure in sufficient detail to replicate precisely.

### Unnatural benchmark (pending)

**Pending experiments:**
1. Construct unnatural positions (board states unreachable by legal play) and run Li's intervention, computing legal set overlap, null baseline, and mean legal set sizes.
2. Run rollout persistence on unnatural states: if the model genuinely internalised the illegal board state, subsequent play should remain consistent with it.

---

## Semantic Sensitivity: A Formal Metric for Stochastic Parrots

Our experimental findings connect to a broader question: what does it mean for a language model to be a "stochastic parrot" (Bender et al. 2021)? A stochastic parrot does not track meaning — it tracks surface form. Our transposition analysis operationalises this precisely in OthelloGPT: *transpositions are identical in meaning (same board state) but differ in surface form (move history)*. The model's sensitivity to this difference is exactly what we want to quantify.

We define three metrics:

**Semantic Sensitivity (SS):**

$$SS(M, [x]) = \frac{2}{n(n-1)} \sum_{i < j} \mathrm{TV}(M(x_i), M(x_j))$$

$$SS(M, \Gamma) = \mathbb{E}_{[x] \sim \Gamma}[SS(M, [x])]$$

where $\mathrm{TV}(p, q) = \frac{1}{2} \sum_v |p(v) - q(v)|$ is total variation distance, $[x] = \{x_1, \ldots, x_n\}$ is a semantic equivalence class (a transposition group), and $\Gamma$ is the corpus distribution over equivalence classes. $SS \in [0, 1]$: 0 means the model is perfectly invariant to path; 1 means maximally inconsistent.

**Semantic Stability Score:**

$$SSS(M, \Gamma) = 1 - SS(M, \Gamma)$$

**Contradiction Rate (CR):**

$$CR(M, [x]) = \frac{2}{n(n-1)} \sum_{i < j} \mathbf{1}[\arg\max M(x_i) \neq \arg\max M(x_j)]$$

$$CR(M, \Gamma) = \mathbb{E}_{[x] \sim \Gamma}[CR(M, [x])]$$

CR is the fraction of equivalence-class pairs where the model's top-1 prediction disagrees. This is the stochastic parrot test operationalised as a metric: a model with no world model of meaning will exhibit high CR on semantic equivalence classes.

**OthelloGPT results** (12,894 transposition pairs, synthetic model):

| Metric | Value |
|---|---|
| SS | 0.230 |
| SSS | 0.770 |
| CR | 0.256 |

OthelloGPT's CR of 0.256 means that in 25.6% of pairs of games reaching the same board state, the model gives a different top-1 move prediction. Its output is sensitive to path, not meaning.

**Mechanistic predictor.** The Trichrome diff (number of cells with different flip counts between two paths to the same board) explains SS variance: Spearman ρ = 0.205 between number of differing Trichrome cells and TV distance (p < 0.01). This means the model's sensitivity to path-dependent history is predicted by how different the two sequences' histories are — consistent with the model encoding Trichrome state as a causally active signal alongside board state.

**Novelty.** The metric is novel on three dimensions relative to prior work:
1. *Full distributional comparison*: uses TV distance over the full output distribution, not log-likelihood ratios on a target token as in existing sensitivity measures.
2. *Ground-truth semantic equivalence*: uses exact game-theoretic equivalence (transpositions), not approximate paraphrases or back-translation.
3. *Mechanistic decomposition*: regresses SS against trichrome predictors to identify what path-dependent information drives inconsistency.

The connection to stochastic parrots is direct: the same failure mode that makes LLMs give inconsistent answers to semantically equivalent prompts (e.g. "Is X harmful?" vs "Is it harmful to do X?") is what drives OthelloGPT's path-dependence. SS formalises this as a testable property of any model with a defined semantic equivalence relation.

---

## Conclusion

What OthelloGPT has is a board-state-shaped projection of move history — a useful and human-legible intermediate representation that any powerful sequence model will learn when trained on Othello. This is not the same as a world model. Nanda's intervention appears to work because the two legal sets are nearly identical (94% overlap), not because the model genuinely shifted its internal board representation. When the metric is corrected for this — by measuring drift from B alongside improvement toward B', and by testing persistence over subsequent moves — the evidence for a world model largely disappears.
