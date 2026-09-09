"""Small statistical-test helpers for replay evaluation.

The functions in this module intentionally return plain dictionaries so UI,
reporting, and optimizers can consume the same payload without importing scipy
objects or test-specific classes.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Sequence

import numpy as np

try:  # pragma: no cover - fallback is covered when scipy is unavailable.
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None  # type: ignore[assignment]


def _clean(values: Sequence[float] | None) -> np.ndarray:
    arr = np.asarray([] if values is None else list(values), dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _normal_pvalue_two_sided(z_score: float) -> float:
    z = abs(float(z_score))
    if scipy_stats is not None:
        return float(2.0 * scipy_stats.norm.sf(z))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, 2.0 * (1.0 - cdf))))


def ks_test(left: Sequence[float] | None, right: Sequence[float] | None) -> Dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test with deterministic empty handling."""

    a = _clean(left)
    b = _clean(right)
    if a.size == 0 or b.size == 0:
        return {"test": "ks_2samp", "statistic": 0.0, "p_value": 1.0, "valid": False}
    if scipy_stats is not None:
        result = scipy_stats.ks_2samp(a, b)
        return {"test": "ks_2samp", "statistic": float(result.statistic), "p_value": float(result.pvalue), "valid": True}

    values = np.sort(np.unique(np.concatenate([np.sort(a), np.sort(b)])))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / b.size
    stat = float(np.max(np.abs(cdf_a - cdf_b))) if values.size else 0.0
    n_eff = a.size * b.size / max(a.size + b.size, 1)
    p_value = float(min(1.0, 2.0 * math.exp(-2.0 * n_eff * stat * stat)))
    return {"test": "ks_2samp", "statistic": stat, "p_value": p_value, "valid": True}


def ad_test(left: Sequence[float] | None, right: Sequence[float] | None) -> Dict[str, Any]:
    """Anderson-Darling k-sample test where scipy is available."""

    a = _clean(left)
    b = _clean(right)
    if a.size < 2 or b.size < 2:
        return {"test": "anderson_ksamp", "statistic": 0.0, "p_value": 1.0, "valid": False}
    if scipy_stats is None:
        ks = ks_test(a, b)
        return {
            "test": "anderson_ksamp_fallback_ks",
            "statistic": float(ks["statistic"]),
            "p_value": float(ks["p_value"]),
            "valid": True,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = scipy_stats.anderson_ksamp([a, b])
        p_value = float(getattr(result, "pvalue", getattr(result, "significance_level", 100.0) / 100.0))
        return {"test": "anderson_ksamp", "statistic": float(result.statistic), "p_value": p_value, "valid": True}
    except Exception:
        return {"test": "anderson_ksamp", "statistic": 0.0, "p_value": 1.0, "valid": False}


def ljung_box(series: Sequence[float] | None, lags: int = 10) -> Dict[str, Any]:
    """Ljung-Box serial-dependence test for a one-dimensional series."""

    x = _clean(series)
    n = int(x.size)
    max_lag = int(max(1, min(lags, n - 1))) if n > 1 else 0
    if n < 3 or max_lag <= 0:
        return {"test": "ljung_box", "statistic": 0.0, "p_value": 1.0, "lags": 0, "valid": False}
    centered = x - float(np.mean(x))
    denom = float(np.dot(centered, centered))
    if denom <= 1e-18:
        return {"test": "ljung_box", "statistic": 0.0, "p_value": 1.0, "lags": max_lag, "valid": False}
    q = 0.0
    for lag in range(1, max_lag + 1):
        autocov = float(np.dot(centered[lag:], centered[:-lag]) / denom)
        q += autocov * autocov / max(n - lag, 1)
    statistic = float(n * (n + 2) * q)
    p_value = float(scipy_stats.chi2.sf(statistic, max_lag)) if scipy_stats is not None else float(math.exp(-0.5 * statistic))
    return {"test": "ljung_box", "statistic": statistic, "p_value": p_value, "lags": max_lag, "valid": True}


def bootstrap_ci(
    values: Sequence[float] | None,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap confidence interval for a statistic, defaulting to the mean."""

    x = _clean(values)
    stat_fn = statistic or (lambda arr: float(np.mean(arr)) if arr.size else 0.0)
    if x.size == 0:
        return {"method": "bootstrap_ci", "statistic": 0.0, "lower": 0.0, "upper": 0.0, "valid": False}
    rng = np.random.default_rng(int(seed))
    reps = max(1, int(n_bootstrap))
    samples = np.empty(reps, dtype=float)
    for idx in range(reps):
        draw = rng.choice(x, size=x.size, replace=True)
        samples[idx] = float(stat_fn(draw))
    alpha = max(0.0, min(1.0, 1.0 - float(ci)))
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "method": "bootstrap_ci",
        "statistic": float(stat_fn(x)),
        "lower": float(lower),
        "upper": float(upper),
        "valid": True,
    }


def permutation_test(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_permutations: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-sample permutation test for a difference in statistic."""

    a = _clean(left)
    b = _clean(right)
    stat_fn = statistic or (lambda arr: float(np.mean(arr)) if arr.size else 0.0)
    if a.size == 0 or b.size == 0:
        return {"test": "permutation", "statistic": 0.0, "p_value": 1.0, "valid": False}
    observed = float(stat_fn(a) - stat_fn(b))
    combined = np.concatenate([a, b])
    rng = np.random.default_rng(int(seed))
    count = 0
    reps = max(1, int(n_permutations))
    for _ in range(reps):
        perm = rng.permutation(combined)
        diff = float(stat_fn(perm[: a.size]) - stat_fn(perm[a.size :]))
        if abs(diff) >= abs(observed):
            count += 1
    p_value = float((count + 1) / (reps + 1))
    return {"test": "permutation", "statistic": observed, "p_value": p_value, "valid": True}


def diebold_mariano(loss_a: Sequence[float] | None, loss_b: Sequence[float] | None) -> Dict[str, Any]:
    """Lightweight Diebold-Mariano style comparison of two loss series.

    Positive ``mean_loss_diff`` means method A has higher loss than method B.
    """

    a = _clean(loss_a)
    b = _clean(loss_b)
    n = min(a.size, b.size)
    if n < 3:
        return {"test": "diebold_mariano", "statistic": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0, "valid": False}
    diff = a[:n] - b[:n]
    mean_diff = float(np.mean(diff))
    std = float(np.std(diff, ddof=1))
    if std <= 1e-18:
        stat = 0.0
        p_value = 1.0 if abs(mean_diff) <= 1e-18 else 0.0
    else:
        stat = float(mean_diff / (std / math.sqrt(n)))
        p_value = _normal_pvalue_two_sided(stat)
    return {
        "test": "diebold_mariano",
        "statistic": stat,
        "p_value": float(p_value),
        "mean_loss_diff": mean_diff,
        "valid": True,
    }


def block_bootstrap_excess_loss(
    loss_a: Sequence[float] | None,
    loss_b: Sequence[float] | None,
    *,
    block_size: int = 5,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Block bootstrap CI for average excess loss A minus B."""

    a = _clean(loss_a)
    b = _clean(loss_b)
    n = min(a.size, b.size)
    if n == 0:
        return {"method": "block_bootstrap_excess_loss", "mean_excess_loss": 0.0, "lower": 0.0, "upper": 0.0, "valid": False}
    diff = a[:n] - b[:n]
    block = max(1, min(int(block_size), n))
    starts = np.arange(0, n, dtype=int)
    rng = np.random.default_rng(int(seed))
    reps = max(1, int(n_bootstrap))
    samples = np.empty(reps, dtype=float)
    for idx in range(reps):
        pieces = []
        while sum(len(piece) for piece in pieces) < n:
            start = int(rng.choice(starts))
            piece = diff[start : min(start + block, n)]
            if piece.size < block:
                piece = np.concatenate([piece, diff[: block - piece.size]])
            pieces.append(piece)
        samples[idx] = float(np.mean(np.concatenate(pieces)[:n]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "method": "block_bootstrap_excess_loss",
        "mean_excess_loss": float(np.mean(diff)),
        "lower": float(lower),
        "upper": float(upper),
        "valid": True,
    }


__all__ = [
    "ad_test",
    "block_bootstrap_excess_loss",
    "bootstrap_ci",
    "diebold_mariano",
    "ks_test",
    "ljung_box",
    "permutation_test",
]
