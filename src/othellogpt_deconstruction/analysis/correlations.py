"""
src/othellogpt_deconstruction/analysis/correlations.py

Correlation analysis between Trichrome diff properties and
OthelloGPT distribution divergence metrics.

Predictors (properties of the Trichrome diff):
    - n_diff_cells:     number of cells with different color
    - max_color_dist:   largest single-cell color distance
    - total_color_dist: sum of color distances across differing cells
    - ply:              game ply at which the transposition occurs

Metrics (distribution divergence between sequence pairs):
    - tv_distance:    total variation distance
    - js_divergence:  Jensen-Shannon divergence
    - spearman_rho:   rank correlation (negated so higher = more different)
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    predictor:    str
    metric:       str
    rho:          float | None
    p_value:      float | None
    n:            int
    significant:  bool   # p < 0.01

    def __str__(self) -> str:
        if self.rho is None:
            return f"{self.predictor:<25}  {self.metric:<22}  {'nan':>6}  {'nan':>8}"
        sig = "**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "")
        return (
            f"{self.predictor:<25}  {self.metric:<22}  "
            f"{self.rho:>6.3f}  {self.p_value:>8.2e} {sig}"
        )


# ---------------------------------------------------------------------------
# Core correlation function
# ---------------------------------------------------------------------------

def correlate(
    records: list[dict],
    predictors: list[str] | None = None,
    metrics: list[str] | None = None,
) -> list[CorrelationResult]:
    """
    Compute Spearman correlations between predictor and metric columns
    in a list of result records.

    Parameters
    ----------
    records    : list of dicts, each with predictor and metric keys
    predictors : predictor column names (default: standard set)
    metrics    : metric column names (default: standard set)

    Returns
    -------
    List of CorrelationResult, one per (predictor, metric) pair.
    """
    if predictors is None:
        predictors = ["n_diff_cells", "max_color_dist", "total_color_dist", "ply"]
    if metrics is None:
        metrics = [
            "tv_distance_full", "js_divergence_full", "spearman_rho_full",
            "tv_distance_legal", "js_divergence_legal", "spearman_rho_legal",
        ]

    # Drop records with any None in relevant columns
    valid = [
        r for r in records
        if all(r.get(k) is not None for k in predictors + metrics)
    ]

    if not valid:
        return []

    results = []
    for pred in predictors:
        x = np.array([r[pred] for r in valid], dtype=float)
        for metric in metrics:
            y = np.array([r[metric] for r in valid], dtype=float)
            # Negate spearman_rho so higher = more different (consistent direction)
            if metric == "spearman_rho":
                y = -y

            if np.all(x == x[0]) or np.all(y == y[0]):
                results.append(CorrelationResult(
                    predictor=pred, metric=metric,
                    rho=None, p_value=None,
                    n=len(valid), significant=False,
                ))
                continue

            rho, p = spearmanr(x, y)
            results.append(CorrelationResult(
                predictor=pred, metric=metric,
                rho=float(rho), p_value=float(p),
                n=len(valid), significant=bool(p < 0.01),
            ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_correlation_table(results: list[CorrelationResult]) -> None:
    """Print a formatted correlation table to stdout."""
    header = f"  {'Predictor':<25}  {'Metric':<22}  {'rho':>6}  {'p':>8}"
    sep    = f"  {'-'*25}  {'-'*22}  {'-'*6}  {'-'*8}"
    if results:
        print(f"\nCorrelation analysis (n={results[0].n} pairs):")
    print(header)
    print(sep)
    for r in results:
        print(f"  {r}")


def summarise_correlations(results: list[CorrelationResult]) -> dict:
    """Return summary statistics over a list of CorrelationResults."""
    valid = [r for r in results if r.rho is not None]
    if not valid:
        return {}
    rhos = [r.rho for r in valid]
    return {
        "n_tests":       len(results),
        "n_significant": sum(r.significant for r in results),
        "mean_abs_rho":  float(np.mean(np.abs(rhos))),
        "max_abs_rho":   float(np.max(np.abs(rhos))),
    }
