# 1. Transposition Distribution Divergence

## Setup

A transposition is a pair of distinct move sequences that reach the same Othello board
state (S_O) at the same ply. If OthelloGPT has a world model — an internal board
representation it reasons from — then two sequences reaching identical board states
should produce identical output distributions. Any divergence is evidence that the
model's predictions depend on the path taken, not just the board state.

We extracted transposition groups from the championship corpus and ran the model on
each sequence pair, measuring distributional divergence.

## Results (n=12,894 pairs, championship corpus)

**Rank-1 agreement**
- 74.4% of pairs agree on the top predicted move
- 25.6% disagree — for 1 in 4 transpositions, the model predicts a different
  top move despite the board being identical

**Distribution divergence**

| Metric               | Mean  | Median | Std   |
|----------------------|-------|--------|-------|
| TV distance (full)   | 0.230 | 0.143  | 0.246 |
| TV distance (legal)  | 0.211 | 0.122  | 0.236 |
| JS divergence (full) | 0.256 | 0.189  | 0.230 |
| JS divergence (legal)| 0.236 | 0.168  | 0.219 |
| Spearman rho (full)  | 0.915 | 0.962  | 0.136 |
| Spearman rho (legal) | 0.918 | 0.957  | 0.120 |

**Correlation with trichrome and ply (n=12,484 pairs)**

| Predictor       | TV distance (full) | Spearman rho (full) |
|-----------------|--------------------|---------------------|
| n_diff_cells    | rho=0.205 **       | rho=-0.438 **       |
| total_color_dist| rho=0.205 **       | rho=-0.438 **       |
| max_color_dist  | nan (constant)     | nan                 |
| ply             | rho=-0.287 **      | rho=+0.419 **       |

All 18 non-nan correlations significant at p<0.01. Mean |rho|=0.288.

## Interpretation

**The model is not path-independent.** 25.6% rank-1 disagreement and mean TV
distance of 0.23 for identical board states is direct evidence against a world model.
A model reasoning from an internal board representation would produce identical
distributions for identical boards.

**Trichrome differences predict divergence.** Trichrome state is a path-dependent
coloring of cells tracking how many times each cell has been flipped. More differing
trichrome cells → more divergent distributions (rho 0.2–0.25 for TV/JS, -0.44 for
Spearman). The model is sensitive to move history beyond board state.

**n_diff_cells and total_color_dist are identical predictors**, meaning every
differing cell has distance exactly 1. max_color_dist is constant (always 1 when
any diffs exist), confirming all trichrome disagreements are single-step transitions.

**Ply is the stronger predictor than trichrome.** Later plies → more similar
distributions (rho -0.287 to -0.323 for TV/JS). This is consistent with
sequence pattern matching: later in the game there are fewer legal moves and less
variation in move history, so the model's outputs naturally converge.

## What this does not prove about Nanda et al.

Nanda et al. show that board state is linearly encoded and causally controllable
via intervention. These transposition results do not contradict that — they show
the board state representation exists but is not the *sole* determinant of
predictions. The model also uses path-dependent signals, and those signals are
causal (see Report 2).
