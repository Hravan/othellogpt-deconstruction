# Semantic Sensitivity of Large Language Models

## Overview

We introduce **Semantic Sensitivity (SS)**, a metric that measures how much a model's output distribution shifts across semantically equivalent inputs. A model with genuine semantic understanding should assign nearly identical output distributions to all phrasings of the same question. High SS indicates the model is responding to surface form rather than meaning.

### Definitions

**Semantic Sensitivity (SS):**

$$SS(M, [x]) = \frac{2}{n(n-1)} \sum_{i < j} \mathrm{TV}(M(x_i), M(x_j))$$

where $[x] = \{x_1, \ldots, x_n\}$ is a semantic equivalence group, $\mathrm{TV}(p, q) = \frac{1}{2} \sum_v |p(v) - q(v)|$ is total variation distance over the full output vocabulary, and $M(x_i)$ is the model's next-token distribution for question $x_i$.

**Semantic Stability Score:** $SSS = 1 - SS$

**Contradiction Rate (CR):** fraction of pairs in a group where the model's top-1 answer disagrees.

SS captures distributional shifts including confidence changes. CR is a coarser binary version — did the top-1 answer flip? Both are computed over formally defined equivalence classes, not approximate LLM-generated paraphrases.

---

## Equivalence Groups

We construct 2,772 equivalence groups across 16 categories, testing different types of semantic transformation:

| Category | Groups | Type | Transformation |
|---|---|---|---|
| capital_word_order | 200 | factual | word order swap |
| capital_retrieval | 100 | factual | question rephrasing |
| geographic_containment | 100 | factual | active/containment variants |
| active_passive | 50 | factual | active ↔ passive voice |
| classification | 97 | factual | taxonomic rephrasing |
| chemical_formula | 20 | factual | subject/predicate swap |
| arithmetic_order | 400 | logical | addition commutativity (small numbers) |
| arithmetic_large | 300 | logical | addition commutativity (3-digit numbers) |
| arithmetic_result | 100 | logical | addition rephrasing (word answer) |
| arithmetic_convoluted | 225 | logical | 5 indirect phrasings of addition |
| multiplication_order | 300 | logical | multiplication commutativity |
| subtraction_equivalence | 225 | logical | subtraction ↔ addition inverse |
| comparison_symmetric | 300 | logical | A>B ↔ B<A |
| comparison_convoluted | 225 | logical | comparison + "exceed" + negated ≤ |
| unit_equivalence | 30 | logical | unit conversion rephrasing |
| double_negation | 100 | logical | ¬¬P ↔ P |

**Logically exact** groups (arithmetic, comparison, double negation) hold by mathematical definition. **Factual** groups hold given a true underlying fact; the syntactic transformation is logically valid.

---

## Results

### Models tested

- Qwen2-1.5B-Instruct (local, instruct)
- Qwen2-7B-Instruct (local, instruct)
- Qwen2.5-14B-Instruct (local, 8-bit, instruct)
- GPT-4o-mini (OpenAI API, top-20 logprobs)
- GPT-4o (OpenAI API, top-20 logprobs)

**Note:** Local models use exact TV over full vocabulary. OpenAI models use TV over top-20 logprobs (lower bound on true TV). CR is unaffected by this difference.

### SS by category

| Category | Qwen 1.5B | Qwen 7B | Qwen 14B | GPT-4o-mini | GPT-4o |
|---|---|---|---|---|---|
| capital_word_order | 0.0613 | 0.0183 | 0.0166 | 0.0033 | 0.0034 |
| capital_retrieval | 0.0609 | 0.0112 | 0.0059 | 0.0049 | 0.0044 |
| geographic_containment | 0.0330 | 0.0002 | 0.0000 | 0.0000 | 0.0013 |
| active_passive | 0.1573 | 0.1226 | 0.1314 | 0.0884 | 0.0238 |
| classification | 0.0436 | 0.0276 | 0.0240 | 0.0131 | 0.0053 |
| chemical_formula | 0.0206 | 0.0000 | 0.1011 | 0.0778 | 0.0225 |
| arithmetic_order | 0.1845 | 0.3468 | 0.3257 | 0.3922 | 0.0855 |
| arithmetic_large | 0.2173 | 0.4144 | 0.2215 | 0.4383 | 0.1188 |
| arithmetic_result | 0.1520 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| arithmetic_convoluted | 0.1869 | 0.3593 | 0.3826 | 0.3841 | 0.1363 |
| multiplication_order | 0.2067 | 0.2980 | 0.0942 | 0.2855 | 0.1745 |
| subtraction_equivalence | 0.2542 | 0.4586 | 0.3200 | 0.5188 | 0.1538 |
| comparison_symmetric | 0.2569 | 0.0381 | 0.0148 | 0.0000 | 0.0000 |
| comparison_convoluted | 0.2933 | 0.2812 | 0.3255 | 0.0022 | 0.0116 |
| unit_equivalence | 0.2147 | 0.0680 | 0.0693 | 0.0570 | 0.0512 |
| **double_negation** | **0.5541** | **0.6359** | **0.6534** | **0.6665** | **0.6454** |
| negation_depth | 0.4206 | 0.6409 | 0.5894 | 0.5910 | 0.5841 |
| negation_arithmetic | 0.2780 | 0.5617 | 0.4788 | 0.4461 | 0.5299 |
| contrastive_negation | 0.4563 | 0.0702 | 0.1146 | 0.0005 | 0.0000 |

### CR by category

| Category | Qwen 1.5B | Qwen 7B | Qwen 14B | GPT-4o-mini | GPT-4o |
|---|---|---|---|---|---|
| capital_word_order | 0.030 | 0.020 | 0.017 | 0.003 | 0.003 |
| capital_retrieval | 0.040 | 0.000 | 0.007 | 0.007 | 0.007 |
| geographic_containment | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 |
| active_passive | 0.160 | 0.140 | 0.140 | 0.080 | 0.020 |
| classification | 0.048 | 0.028 | 0.021 | 0.014 | 0.007 |
| chemical_formula | 0.000 | 0.000 | 0.100 | 0.100 | 0.000 |
| arithmetic_order | 0.417 | 0.342 | 0.332 | 0.415 | 0.077 |
| arithmetic_large | 0.524 | 0.416 | 0.220 | 0.484 | 0.109 |
| arithmetic_result | 0.143 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic_convoluted | 0.251 | 0.368 | 0.391 | 0.400 | 0.120 |
| multiplication_order | 0.451 | 0.300 | 0.096 | 0.291 | 0.164 |
| subtraction_equivalence | 0.474 | 0.459 | 0.323 | 0.548 | 0.148 |
| comparison_symmetric | 0.380 | 0.038 | 0.016 | 0.000 | 0.000 |
| comparison_convoluted | 0.399 | 0.276 | 0.327 | 0.002 | 0.009 |
| unit_equivalence | 0.222 | 0.067 | 0.067 | 0.067 | 0.044 |
| **double_negation** | **0.640** | **0.633** | **0.653** | **0.667** | **0.667** |
| negation_depth | 0.470 | 0.658 | 0.590 | 0.627 | 0.613 |
| negation_arithmetic | 0.340 | 0.592 | 0.484 | 0.552 | **0.606** |
| contrastive_negation | 0.587 | 0.067 | 0.120 | 0.000 | 0.000 |

---

## Finding 1: Arithmetic Sensitivity Plateaus with Scale

arithmetic_order CR drops from 0.417 (1.5B) to 0.332 (14B) then barely moves — and GPT-4o-mini (0.415) is *worse* than Qwen 14B despite being a far larger model. GPT-4o finally makes a large jump (0.077) but does not reach zero.

The specific failure: the model agrees on "Is 15+60=75?" and "Does 60+15=75?" but answers differently for "Is 75 the sum of 15 and 60?" This third phrasing is the odd one out in 146/249 contradiction cases. The model parses the syntactic form of an equation differently from the syntactic form of a sum assertion.

arithmetic_large (3-digit numbers) has higher CR than arithmetic_order on GPT-4o-mini (0.484 vs 0.415), confirming the model has memorised small-number arithmetic rather than computing it. GPT-4o largely closes this gap (0.109 vs 0.077).

subtraction_equivalence and arithmetic_convoluted are the hardest arithmetic categories: GPT-4o-mini CR = 0.548 and 0.400 respectively. GPT-4o improves substantially (0.148 and 0.120) but remains far from zero.

---

## Finding 2: Double Negation Is Scale-Invariant

**CR = 0.667 on both GPT-4o-mini and GPT-4o.** This is the maximum possible CR for 3-question groups — every group with a contradiction has exactly one question answered differently from the other two. Qwen models are not better: 0.640 (1.5B), 0.633 (7B), 0.653 (14B).

Example (GPT-4o):
```
[yes] Is Vienna the capital of Austria?
[no]  Is it false that Vienna is not the capital of Austria?
[no]  Is it not the case that Vienna is not the capital of Austria?
```

The model correctly answers the direct question but fails both double-negation phrasings. Analysis of which position is the odd one out:
- Position 0 (direct question): odd one out in 69/100 groups
- Position 2 (¬¬ phrasing 2): odd one out in 31/100 groups
- Position 1 (¬¬ phrasing 1): never the odd one out

**The mechanism:** the model answers the embedded proposition ("Vienna is not the capital of Austria" → false → answer "no") rather than composing the outer negation operator ("Is it false that [false]?" → yes). The outer scope of negation is ignored.

**This is not a knowledge failure.** The model knows Vienna is the capital of Austria (capital_word_order CR = 0.003). It fails at logical composition, not factual recall.

**Scale has no effect.** The jump from GPT-4o-mini to GPT-4o that reduced arithmetic_order CR by 5× leaves double_negation completely unchanged. More parameters trained on the same data distribution do not fix negation scope.

---

## Finding 3: Negation Depth Is Non-Monotonic

negation_depth tests questions at depths 1–4 (direct question, ¬¬P, ¬¬¬P, ¬¬¬¬P). CR across models: 0.470 (1.5B) → 0.658 (7B) → 0.590 (14B) → 0.627 (GPT-4o-mini) → 0.613 (GPT-4o). **No model improves monotonically with scale.** 7B is the worst model, worse than 1.5B.

negation_arithmetic (same depth hierarchy applied to arithmetic facts: "Is it false that 3+4=7?") shows an even more striking pattern: GPT-4o CR = **0.606**, the *worst* model on this category. Combining arithmetic and negation is harder for larger models, not easier.

The non-monotonicity is evidence that scale is not learning the recursive negation rule — it is learning a stronger but still wrong surface pattern, which then conflicts with itself at different depths.

---

## Finding 4: Comparison Is Solved; Arithmetic Is Not

comparison_symmetric (A>B ↔ B<A) drops from CR=0.380 at 1.5B to CR=0.000 at GPT-4o-mini. comparison_convoluted (adding "Does A exceed B?" and "Is it true that A is not ≤ B?") also reaches CR=0.002 at GPT-4o-mini. contrastive_negation ("Is X, not Z, the capital of Y?") reaches CR=0.000 for both GPT models.

The model has learned that comparison relations are symmetric and that contrastive framing is identifiable. It has not learned that addition is commutative across all phrasings, that subtraction and addition are inverses, or that negation composes recursively.

**The Qwen/GPT split on comparison_convoluted is notable:** Qwen 14B CR = 0.327, GPT-4o-mini CR = 0.002. This is not a scale effect — it is a training distribution effect. The GPT family has learned comparison phrasing variants that Qwen has not, despite Qwen 14B being a larger model than GPT-4o-mini in parameter count.

---

## Theoretical Interpretation: Surface Competence vs Rule Internalization

These findings connect to a broader failure mode visible across domains. OthelloGPT, trained on 8×8 Othello games, fails to generalize to 9×9 boards. Train it on 9×9 and it fails on 10×10. The model learns the statistical regularities of the specific board size, not the rules of Othello — which are board-size-invariant.

LLMs on negation follow the same pattern. Fine-tuning on double negation examples would likely teach the model to handle depth-2 patterns without learning the recursive rule (each negation inverts the truth value). Triple negation would then fail again — a new surface pattern the model has not seen.

**The formal connection:** SS measures the gap between surface competence and rule internalization. A model that had internalized the rule "negation inverts truth value" would have SS≈0 across all depths of negation, because the rule is invariant to surface form. High SS at depth-2 is direct evidence of surface pattern matching. If SS remains high at depth-3 after training on depth-2, it proves the training fix was a surface patch, not rule learning.

This is the LLM analogue of the SCAN benchmark (Lake & Baroni, 2018): systematic compositional generalization tests whether models learn rules or memorise training distributions. SS operationalises this test for natural language yes/no questions.

---

## Relation to Prior Work

**Elazar et al. (2021)** — most structurally similar: formal equivalence groups (ParaRel), symmetric KL divergence as training loss. Restricted to single-token cloze completions on factual knowledge relations; KL used only for training, not evaluation.

**Xie et al. (2020) UDA** — consistency training via KL divergence on paraphrase-augmented pairs. Approximate paraphrases (back-translation); evaluation-only metric not defined.

**Zhou et al. (2024) "Paraphrase and Solve"** — defines Variance of Variations (VOV): variance of solve rates across math problem paraphrases. Answer-level (binary correct/wrong), not distributional; approximate LLM-generated paraphrases; no formal equivalence classes.

**What is novel here:**
1. TV distance over the full output distribution (not KL on single-token cloze, not embedding cosine similarity, not binary solve rate)
2. Mathematically exact equivalence classes (not approximate paraphrases)
3. Systematic evaluation across model scales and families
4. Identification of double negation as a scale-invariant failure mode

---

## Dataset Augmentation Application

SS identifies which equivalence classes have high CR — i.e., which syntactic transformations the model treats inconsistently. These are exactly the training pairs that provide the most signal for consistency training:

1. **Supervised augmentation**: add all phrasings in high-CR groups with consistent correct labels. The model sees "Is 15+60=75?" and "Is 75 the sum of 15 and 60?" both labeled "yes."
2. **Consistency loss (Elazar-style)**: penalize KL divergence between distributions on high-CR pairs during fine-tuning. Low-CR pairs contribute no signal; high-CR pairs provide targeted gradient.

Since equivalence classes are mathematically exact (not LLM-generated), the training signal is clean — no label noise from imperfect paraphrase generation. For double negation specifically, the equivalence is a tautology: ¬¬P ↔ P holds for any proposition P, so training pairs can be generated infinitely without a knowledge base.

The metric also provides a direct feedback loop: retrain, re-evaluate SS, check which categories dropped. A genuine fix (rule internalization) would show SS≈0 across all depths of negation. A surface fix (pattern memorisation) would show SS≈0 at depth-2 but high SS at depth-3.

---

## Pending: Negation Depth Experiment

**Hypothesis:** fine-tuning on double negation will not generalize to triple negation, because the model learns a surface pattern rather than the recursive rule.

**Proposed experiment:**
1. Evaluate SS at negation depth 1, 2, 3, 4 on all models
2. Fine-tune a model on depth-2 examples
3. Re-evaluate at all depths
4. Expected result: SS drops at depth-2, remains high at depth-3 and depth-4

If confirmed, this is direct evidence that the model learned a depth-2 surface fix, not the negation rule — the same failure mode as OthelloGPT failing to generalize across board sizes.
