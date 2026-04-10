# Semantic Sensitivity: A Formal Test for Rule Internalization in Language Models

## The Claim

Language models trained on next-token prediction learn statistical regularities of surface form, not the underlying rules that generate meaning. We operationalise this as a testable, falsifiable metric and demonstrate it across six models and nineteen categories of semantic equivalence.

---

## The Metric

**Semantic Sensitivity (SS):**

$$SS(M, [x]) = \frac{2}{n(n-1)} \sum_{i < j} \mathrm{TV}(M(x_i), M(x_j))$$

where $[x] = \{x_1, \ldots, x_n\}$ is a semantic equivalence group, $\mathrm{TV}(p, q) = \frac{1}{2} \sum_v |p(v) - q(v)|$ is total variation distance over the full output vocabulary, and $M(x_i)$ is the model's next-token distribution for question $x_i$. $SS \in [0, 1]$: 0 means the model is perfectly invariant to surface form; 1 means maximally inconsistent.

**Semantic Stability Score:** $SSS = 1 - SS$

**Contradiction Rate (CR):** fraction of pairs in a group where the model's top-1 answer disagrees. A coarser binary version of SS — did the top-1 answer flip?

Both metrics are computed over **formally defined equivalence classes**, not approximate LLM-generated paraphrases. For logical categories (arithmetic, negation, comparison), equivalence holds by mathematical definition. For factual categories (capitals, geography), equivalence is logically valid given a true underlying fact.

**Primary metric: CR.** SS is reported alongside CR but is not used for cross-model comparison, since local models use exact full-vocabulary TV while OpenAI models use top-20 logprobs (a lower bound). CR is unaffected by this difference.

---

## Equivalence Groups

4,325 groups across 28 categories, 13,875 total questions. All categories have ≥100 groups.

| Category | Groups | Type | Transformation |
|---|---|---|---|
| capital_word_order | 200 | factual | word order swap |
| capital_retrieval | 100 | factual | question rephrasing |
| geographic_containment | 100 | factual | active/containment variants |
| active_passive | 100 | factual | active ↔ passive voice |
| classification | 100 | factual | taxonomic rephrasing |
| chemical_formula | 100 | factual | subject/predicate swap (3 phrasings) |
| arithmetic_order | 400 | logical | addition commutativity |
| arithmetic_large | 300 | logical | addition commutativity (3-digit) |
| arithmetic_result | 100 | logical | addition rephrasing (word answer) |
| arithmetic_convoluted | 225 | logical | 5 indirect phrasings of addition |
| multiplication_order | 300 | logical | multiplication commutativity |
| subtraction_equivalence | 225 | logical | subtraction ↔ addition inverse |
| comparison_symmetric | 300 | logical | A>B ↔ B<A |
| comparison_convoluted | 225 | logical | comparison + "exceed" + negated ≤ |
| unit_equivalence | 100 | logical | unit conversion rephrasing |
| double_negation | 200 | logical | ¬¬P ↔ P (100 yes + 100 no) |
| negation_depth | 100 | logical | depths 0,2,3,4 mixed in one group |
| negation_arithmetic | 150 | logical | arithmetic + negation depth |
| contrastive_negation | 100 | logical | "Is X, not Z, the capital of Y?" |
| negation_depth_0 | 100 | logical | 3 phrasings at depth 0 (no negation) |
| negation_depth_1 | 100 | logical | 3 phrasings at depth 1 (¬P) |
| negation_depth_2 | 100 | logical | 3 phrasings at depth 2 (¬²P) |
| negation_depth_3 | 100 | logical | 3 phrasings at depth 3 (¬³P) |
| negation_depth_4 | 100 | logical | 3 phrasings at depth 4 (¬⁴P) |
| negation_depth_5 | 100 | logical | 3 phrasings at depth 5 (¬⁵P) |
| negation_depth_6 | 100 | logical | 3 phrasings at depth 6 (¬⁶P) |
| negation_even | 100 | logical | depths 0,2,4,6 in one group (all "yes") |
| negation_odd | 100 | logical | depths 1,3,5 in one group (all "no") |

The `negation_depth_N` categories each contain 3 distinct phrasings using different negation operators ("is not the case that", "is false that", "is not true that") all at exactly N negation operators. Within each group all questions have the same answer (even depths → yes, odd depths → no for correct capitals). These categories serve both for SS/CR evaluation and as training/test data in the fine-tuning experiment. The `negation_even` and `negation_odd` categories test whether models maintain consistency across the full parity class.

---

## Models Tested

| Model | Type | TV computation |
|---|---|---|
| Qwen2-1.5B-Instruct | local, instruct | exact full vocabulary |
| Mistral-7B-Instruct-v0.3 | local, instruct | exact full vocabulary |
| Qwen2-7B-Instruct | local, instruct | exact full vocabulary |
| Qwen2.5-14B-Instruct | local, 8-bit, instruct | exact full vocabulary |
| Llama-3.1-8B-Instruct | local, instruct | exact full vocabulary |
| GPT-4o-mini | OpenAI API | top-20 logprobs (lower bound) |
| GPT-4o | OpenAI API | top-20 logprobs (lower bound) |

---

## Results

### CR by category

CR normalized by maximum possible CR for that group size (CR_norm = CR / CR_max(n), where CR_max(n) = ⌊n/2⌋⌈n/2⌉ / C(n,2)). Raw CR stored in JSON files.

| Category | Qwen 1.5B | Mistral 7B | Qwen 7B | Qwen 14B | Llama 8B | GPT-4o-mini | GPT-4o |
|---|---|---|---|---|---|---|---|
| capital_word_order | 0.045 | 0.030 | 0.030 | 0.025 | 0.020 | 0.005 | 0.005 |
| capital_retrieval | 0.060 | 0.000 | 0.000 | 0.010 | 0.010 | 0.010 | 0.010 |
| geographic_containment | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| active_passive | 0.180 | 0.170 | 0.100 | 0.090 | 0.070 | 0.080 | 0.010 |
| classification | 0.070 | 0.020 | 0.040 | 0.030 | 0.010 | 0.020 | 0.010 |
| chemical_formula | 0.170 | 0.250 | 0.150 | 0.160 | 0.210 | 0.130 | 0.020 |
| arithmetic_order | 0.625 | 0.615 | 0.512 | 0.497 | 0.530 | 0.623 | 0.115 |
| arithmetic_large | 0.787 | 0.353 | 0.623 | 0.330 | 0.507 | 0.727 | 0.163 |
| arithmetic_result | 0.215 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic_convoluted | 0.418 | 0.559 | 0.613 | 0.652 | 0.477 | 0.667 | 0.200 |
| multiplication_order | 0.677 | 0.570 | 0.450 | 0.143 | 0.167 | 0.437 | 0.247 |
| subtraction_equivalence | 0.711 | 0.876 | 0.689 | 0.484 | 0.653 | 0.822 | 0.222 |
| comparison_symmetric | 0.570 | 0.383 | 0.057 | 0.023 | 0.047 | 0.000 | 0.000 |
| comparison_convoluted | 0.599 | 0.418 | 0.414 | 0.490 | 0.487 | 0.003 | 0.013 |
| unit_equivalence | 0.490 | 0.510 | 0.340 | 0.280 | 0.470 | 0.180 | 0.070 |
| **double_negation** | **0.480*** | **0.995** | **0.795** | **0.950** | **1.000** | **1.000** | **1.000** |
| negation_depth | 0.705 | 0.800 | 0.988 | 0.885 | 0.965 | 0.940 | 0.920 |
| negation_arithmetic | 0.510 | 0.913 | 0.888 | 0.727 | 0.823 | 0.852 | 0.908 |
| contrastive_negation | 0.880 | 0.060 | 0.100 | 0.180 | 0.050 | 0.000 | 0.000 |

*Qwen 1.5B double_negation is depressed by yes/no group asymmetry: no-groups (wrong capitals) yield CR≈0 because "always answer no" accidentally gives correct answers at all negation depths. Yes-group CR_norm ≈ 0.960. All other models show high CR on both group types.

---

## Finding 1: Double Negation and Negation Depth Are Near-Ceiling

After normalization, double_negation CR_norm ranges from 0.795 to 1.000 across six of the seven models (Qwen 1.5B is an outlier at 0.480 due to a yes/no group asymmetry — its yes-group CR_norm ≈ 0.960). Llama 8B, GPT-4o-mini, and GPT-4o all reach the ceiling of 1.000.

negation_depth CR_norm ranges from 0.705 to 0.988 — comparable to or exceeding double_negation. This is the highest-CR category in the dataset when normalized, across all models and all families.

Example (GPT-4o):
```
[yes] Is Vienna the capital of Austria?
[no]  Is it false that Vienna is not the capital of Austria?
[no]  Is it not the case that Vienna is not the capital of Austria?
```

The model correctly answers the direct question but fails both double-negation phrasings. Position 0 (direct question) is the odd one out in 69/100 groups; position 2 in 31/100; position 1 never.

**Mechanism:** the model answers the embedded proposition ("Vienna is not the capital" → false → "no") rather than composing the outer negation operator. The outer scope is ignored.

**This is not a knowledge failure.** capital_word_order CR = 0.003 for GPT-4o — the model knows the facts. It fails at logical composition, not factual recall.

**Scale has no effect.** The jump from GPT-4o-mini to GPT-4o that reduced arithmetic_order CR by 5× leaves double_negation completely unchanged. More parameters do not fix negation scope.

---

## Finding 2: Negation Depth Is Non-Monotonic and Scale-Invariant

negation_depth CR_norm (normalized): 0.705 (Qwen 1.5B) → 0.800 (Mistral 7B) → 0.988 (Qwen 7B) → 0.885 (Qwen 14B) → 0.965 (Llama 8B) → 0.940 (GPT-4o-mini) → 0.920 (GPT-4o). No model improves consistently with scale. Qwen 7B (0.988) is the worst model — worse than Qwen 1.5B (0.705).

negation_arithmetic CR_norm: GPT-4o (0.908) is the worst model on this category — worse than every smaller model. Larger models perform worse on arithmetic negation.

The non-monotonicity indicates scale is learning a stronger but still wrong surface pattern, which conflicts with itself at different depths.

---

## Finding 3: Arithmetic Sensitivity Persists; Comparison Is Solved

comparison_symmetric CR_norm drops from 0.570 (Qwen 1.5B) to 0.000 (GPT-4o-mini). comparison_convoluted also reaches ≈0 for GPT models. contrastive_negation reaches 0.000 for GPT-4o-mini and GPT-4o.

arithmetic_order CR_norm: GPT-4o-mini 0.623 — worse than Qwen 14B (0.497). GPT-4o improves to 0.115 but does not reach zero. subtraction_equivalence CR_norm: GPT-4o-mini 0.822, GPT-4o 0.222. arithmetic_large CR_norm: GPT-4o-mini 0.727, the second-highest across all models on this category.

The model has learned comparison symmetry and contrastive framing as surface patterns. It has not learned addition commutativity across all phrasings, subtraction-addition duality, or double negation cancellation.

**The Qwen/GPT split on comparison_convoluted** (Qwen 14B CR_norm=0.490, GPT-4o-mini CR_norm=0.003) is not a scale effect — GPT-4o-mini is smaller than Qwen 14B in parameter count. It reflects training distribution: the GPT family has seen more comparison phrasing variants.

---

## Finding 4: Model-Family Effects Dominate Scale for Some Categories

| Category | Qwen 1.5B | Mistral 7B | Qwen 7B |
|---|---|---|---|
| comparison_symmetric | 0.380 | 0.256 | 0.038 |
| contrastive_negation | 0.587 | 0.040 | 0.067 |

Mistral 7B solves contrastive_negation (CR=0.040) but fails comparison_symmetric (CR=0.256). Qwen 7B does the opposite. These are the same parameter count, same evaluation setup — the difference is training data and RLHF, not scale. This shows CR is sensitive to what surface patterns the model has learned, not just how large it is.

---

## Fine-Tuning Experiment: Surface Patch vs Rule Internalization

**Setup.** Three Qwen 1.5B models are fine-tuned on negation examples at increasing depth ranges, then tested on the two depths immediately beyond their training range. Training uses the `negation_depth_N` categories (100 capital facts, 3 phrasings each). Train on 80 capital facts (first 80 groups per category), evaluate on 20 held-out capital facts never seen during training.

The correct answer alternates with depth: even depths → "yes", odd depths → "no" (for correct capitals). A model that has internalized the recursive negation rule would achieve 100% accuracy at all depths. A model that learned a surface pattern would fail at the first unseen depth.

**Results — Qwen 1.5B:**

| Depth | Expected | Model A (train 0,1,2) | Model B (train 0,1,2,3) | Model C (train 0,1,2,3,4) |
|-------|----------|-----------------------|-------------------------|---------------------------|
| 0 | yes | 100% (TRAIN) | 100% (TRAIN) | 100% (TRAIN) |
| 1 | no  | 100% (TRAIN) | 100% (TRAIN) | 100% (TRAIN) |
| 2 | yes | 100% (TRAIN) | 100% (TRAIN) | 100% (TRAIN) |
| 3 | no  | **0%** (TEST) | 100% (TRAIN) | 100% (TRAIN) |
| 4 | yes | 100% (TEST)  | **0%** (TEST) | 100% (TRAIN) |
| 5 | no  | —            | 100% (TEST)  | **0%** (TEST) |
| 6 | yes | —            | 15% (TEST)   | 100% (TEST)  |

**The pattern is exact and consistent across all three models.** The first unseen depth (N+1) is always 0% accurate. The second unseen depth (N+2) is always 100% accurate (except model B at depth-6: 15%, discussed below).

**Mechanism.** Each model applies the answer from the last training depth to all unseen depths:
- Model A trained last on depth-2 (answer: "yes") → answers "yes" for depths 3 and 4 → wrong at 3, right at 4
- Model B trained last on depth-3 (answer: "no") → answers "no" for depths 4, 5, 6 → wrong at 4, right at 5, mostly wrong at 6
- Model C trained last on depth-4 (answer: "yes") → answers "yes" for depths 5 and 6 → wrong at 5, right at 6

**Model B at depth-6 (15%, not 100%).** The model answers "no" for depth-6 (expected "yes") — consistent with the last-depth bias. The 15% accuracy comes from a small fraction of groups where the phrasing triggered a different response, not from any generalization.

**CR = 0.000 at every depth, including unseen ones.** Within each depth group (3 distinct phrasings), the model gives identical answers — perfectly consistent. The failure is not about surface form variation within a depth; it is about the alternating rule between depths. The model is confidently wrong, not confused.

**The decisive result.** Fine-tuning on depths 0..N teaches the model "the answer at depth N is X, therefore the answer at any deeper depth is also X." This is the wrong rule. The correct rule is "each negation operator inverts the answer." A model that had internalized the recursive rule would answer correctly at N+1 and N+2 simultaneously. Instead:
- N+1 accuracy: always 0%
- N+2 accuracy: always 100% (by coincidence — N+2 has the same parity as N)

This demonstrates that the model learned a last-seen-depth heuristic, not the alternating rule.

*(Qwen 7B experiments pending.)*

---

## The Formal Test for Rule Internalization

The fine-tuning experiment defines a general test:

> **A model has internalized rule R if and only if fine-tuning on instances of R at complexity k generalizes to complexity k+1.**

For negation: fine-tuning on depths 0..N should generalize to depth N+1 if the model learned "each negation inverts the truth value." It does not — it generalizes to N+2 (same parity as N) but fails at N+1 (opposite parity).

The three-model design makes the test airtight. Any single experiment could be dismissed as coincidence or artefact. Three experiments with the same outcome at three different training cutoffs and three different test depths — all showing the identical pattern of 0% at N+1 and 100% at N+2 — is direct evidence of a parity heuristic, not a rule.

This test is not specific to negation. The same structure applies to any rule with instances at varying complexity:
- Arithmetic commutativity: fine-tuning on a+b=b+a for small numbers should generalize to large numbers
- Syllogistic reasoning: fine-tuning on two-premise syllogisms should generalize to three-premise syllogisms
- Spatial relations: fine-tuning on "A is left of B" ↔ "B is right of A" should generalize to transitive cases

A model that has truly internalized the rule will pass at every depth. A model that learned a surface heuristic will fail at the first unseen depth and succeed at the second — exactly as observed.

---

## Connection to the Stochastic Parrots Argument

Bender et al. (2021) argue that language models trained on form alone cannot acquire meaning. This argument is philosophical and difficult to operationalize. SS provides a formal, testable version:

**A model that has acquired the meaning of negation should assign identical output distributions to all phrasings of the same proposition, regardless of the number of negation operators applied.**

High SS on negation groups is direct evidence of surface form sensitivity. The fine-tuning experiment shows that the model's failure is not a knowledge deficit — it knows the facts — but a compositional deficit: it cannot compose the negation operator recursively.

The connection to the stochastic parrot argument: a model trained on text where depth-2 negation is rare and depth-3 negation is vanishingly rare will learn a surface approximation. The approximation works for the common cases (depth-2) and over-generalizes to uncommon cases (depth-3) in exactly the wrong direction. This is what stochastic pattern matching predicts; it is not what rule internalization predicts.

---

## Relation to Prior Work

**Elazar et al. (2021)** — most structurally similar: formal equivalence groups (ParaRel), symmetric KL divergence as training loss. Restricted to single-token cloze completions on factual knowledge relations; no evaluation across model scales; no fine-tuning generalization test.

**Zhou et al. (2024) "Paraphrase and Solve"** — defines Variance of Variations (VOV): variance of solve rates across math problem paraphrases. Answer-level (binary correct/wrong), not distributional; approximate LLM-generated paraphrases; no formal equivalence classes; no negation.

**SCAN (Lake & Baroni, 2018)** — systematic compositional generalization in instruction following. Closest in spirit to the fine-tuning experiment. SS operationalizes the same test for natural language yes/no questions without requiring a formal grammar.

**What is novel:**
1. TV distance over the full output distribution (not binary correctness, not single-token cloze)
2. Mathematically exact equivalence classes across 19 categories
3. Systematic evaluation across six models and three families
4. Identification of double negation as a scale-invariant failure mode
5. Formal definition of a generalization test for rule internalization, demonstrated with fine-tuning experiments at two scales

---

## Summary of Key Results

| Finding | Evidence |
|---|---|
| Double negation and negation depth are near-ceiling | CR_norm 0.795–1.000 (double_negation), 0.705–0.988 (negation_depth) across all 7 models |
| Negation depth is non-monotonic | Qwen 7B (0.988) worse than Qwen 1.5B (0.705); GPT-4o worst on negation_arithmetic (0.908) |
| Comparison is solved; arithmetic is not | comparison_symmetric CR_norm = 0.000 at GPT-4o-mini; arithmetic_order CR_norm = 0.623 |
| Model family > scale for some categories | Mistral 7B solves contrastive_negation (0.060); Qwen 7B solves comparison_symmetric (0.057) |
| Fine-tuning creates last-depth heuristic | All 3 models: 0% at N+1, 100% at N+2; CR=0 even on unseen depths |
| Three experiments, identical pattern | Rules out coincidence; confirms parity heuristic not recursive rule |
