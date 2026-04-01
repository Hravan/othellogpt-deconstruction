# 2. Trichrome Intervention

## Setup

Report 1 shows the model's predictions are path-dependent. This report asks whether
that path-dependence is *causal*: do path-dependent activation differences actually
drive prediction differences, or are they merely correlated?

We use delta_intervention: for a transposition pair (seq_a, seq_b) reaching the
same Othello board state, compute the activation difference (act_b - act_a) at
every layer and add it to seq_a's forward pass, then measure how predictions change.

The key property: since both sequences reach the **same Othello board**, the
activation delta cannot encode board state differences. It encodes only
path-dependent history — trichrome state and other sequence features. If the
intervention shifts predictions or increases illegal move rates, that is causal
evidence that non-board-state features drive the model.

**Conditions**
- Mixed-trichrome [experimental]: seq_a and seq_b reach the same board via
  different trichrome states. Delta is path-dependent.
- Same-trichrome [control]: seq_a and seq_b reach the same board AND same trichrome
  state. Delta should be near-zero; minimal effect expected.

## Results (n=11,633 mixed-trichrome groups, championship corpus)

**Baseline (before intervention)**
- Top-1 illegal rate: ~12% in both conditions

**Distribution shift**

|                      | Mixed [experimental] | Same [control] |
|----------------------|---------------------|----------------|
| TV distance before   | 0.230               | 0.157          |
| TV distance after    | 0.515               | 0.394          |
| TV reduction         | -0.284              | -0.237         |
| % directions improved| 20.2%               | 18.5%          |

TV distance *increases* in both conditions — the intervention is destructive rather
than steering. Adding the full residual stream delta at all layers simultaneously is
too aggressive and pushes the model into incoherent states. This is consistent with
Nanda et al.'s Appendix B, which shows layer selection matters significantly.

**Illegal move rate**

|                          | Mixed [experimental] | Same [control] |
|--------------------------|---------------------|----------------|
| Top-1 legal before       | 88.2%               | 91.5%          |
| Top-1 legal after        | 71.0%               | 80.3%          |
| Became illegal           | +22.1pp             | +14.9pp        |

**Rank-1 recovery** (pairs where rank-1 initially disagreed)
- Mixed: 2434/5828 = 41.8% now agree after intervention
- Control: 512/990 = 51.7% now agree

## Interpretation

**The intervention is destructive, not steering.** TV increases in both conditions
because adding full-layer deltas is too aggressive. A cleaner experiment would use
targeted layer selection (layers 4–7 where Nanda finds board state is best encoded)
and a smaller alpha. This limits what can be concluded about distributional shift.

**The illegal move differential is the key finding.** Path-dependent activation
deltas (board state unchanged) cause significantly more illegal predictions than
same-board-same-trichrome deltas:
- Mixed: +22.1pp more illegal predictions
- Control: +14.9pp more illegal predictions
- **Differential: 7.2pp**

A model reasoning purely from a world model of Othello should not have its legal
move predictions disrupted by adding path-dependent activation differences — the
board state is unchanged, so a world model would still read off the same legal moves.

**The control condition is not a near-zero baseline — and that strengthens the
argument.** Same-trichrome transposition pairs differ only in finer-grained move
history features beyond trichrome. That they also produce +14.9pp of illegal
predictions confirms the broader claim: the model is sensitive to arbitrary
path-dependent features of move history, not just board state. Trichrome is one such
feature we can name and measure; the control shows there are many others. This is
what we expect from a sequence model that has learned to pattern-match on move
history. Nanda et al.'s board-state representation is one particular projection of
that history — the most human-legible one for people who know Othello — not evidence
of a privileged world model.

**The 12% baseline illegal rate is itself a finding.** The championship model
predicts illegal moves ~12% of the time on transposition positions (ply ~24),
substantially higher than its overall accuracy. This is consistent with sequence
pattern matching: transposition positions are exactly where pattern matching is most
ambiguous (multiple very different sequences converge here), so the model is most
likely to fail.

## Limitations and next steps

1. **Layer selection**: re-run with layers 4–7 only to get steering rather than
   destruction. This would allow a cleaner claim about whether path-dependent
   activations shift the *direction* of predictions.

2. **Probe intervention**: train linear probes predicting trichrome color per cell
   and intervene only along the probe direction at matched norm, compared to a random
   direction of equal norm. This isolates the trichrome subspace specifically.

3. **Alpha sweep**: smaller alpha (0.5, 0.25) may allow partial steering without
   full corruption, revealing the causal structure more cleanly.

## Relation to Nanda et al.

Nanda et al. demonstrate that adding board-state linear vectors successfully steers
the model (error rate 0.10 for flipping). Our result shows that adding
*path-dependent* vectors (from same-board transposition pairs) also perturbs
predictions — causing more illegal moves. The board state representation is one of
multiple causally relevant signals, not the sole mechanism. Crucially, the
board-state representation is itself a deterministic function of move history; any
powerful sequence model will encode it. Nanda's intervention works because the
model's move-prediction circuit reads from a board-state-shaped intermediate
representation — not because the model reasons from a genuine world model of Othello.
