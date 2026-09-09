"""Event-window alignment metrics for replay experiments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.eval.stat_tests import bootstrap_ci, permutation_test


def _clean_prices(values: Sequence[float] | None) -> np.ndarray:
    arr = np.asarray([] if values is None else list(values), dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _returns(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(prices), dtype=float)
    if arr.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(arr) / np.maximum(arr[:-1], 1e-12)


def _event_indices(
    *,
    event_points: Iterable[Any] | None,
    dates: Sequence[Any] | None,
    series_length: int,
) -> list[int]:
    if event_points is None:
        return []
    points = list(event_points)
    if not points:
        return []
    date_to_idx: dict[str, int] = {}
    if dates is not None:
        parsed_dates = pd.to_datetime(list(dates), errors="coerce")
        for idx, value in enumerate(parsed_dates):
            if pd.notna(value):
                date_to_idx[str(value.date())] = idx
    indices: list[int] = []
    for point in points:
        idx: int | None = None
        if isinstance(point, Mapping):
            if "index" in point:
                try:
                    idx = int(point["index"])
                except Exception:
                    idx = None
            elif "date" in point:
                parsed = pd.to_datetime(point["date"], errors="coerce")
                if pd.notna(parsed):
                    idx = date_to_idx.get(str(parsed.date()))
        elif isinstance(point, (int, np.integer)):
            idx = int(point)
        else:
            parsed = pd.to_datetime(point, errors="coerce")
            if pd.notna(parsed):
                idx = date_to_idx.get(str(parsed.date()))
        if idx is not None and 0 <= idx < series_length:
            indices.append(idx)
    return sorted(set(indices))


def _window_slice(series: np.ndarray, center_idx: int, start_offset: int, end_offset: int) -> np.ndarray:
    start = max(0, center_idx + start_offset)
    end = min(series.size, center_idx + end_offset + 1)
    if end <= start:
        return np.asarray([], dtype=float)
    return series[start:end]


def event_window_direction_hit_rate(
    sim_close: Sequence[float] | None,
    real_close: Sequence[float] | None,
    event_points: Iterable[Any] | None,
    *,
    dates: Sequence[Any] | None = None,
    pre_window: int = 3,
    post_window: int = 3,
) -> float:
    """Direction agreement of cumulative event-window returns."""

    summary = event_study(
        sim_close=sim_close,
        real_close=real_close,
        event_points=event_points,
        dates=dates,
        pre_window=pre_window,
        post_window=post_window,
    )
    return float(summary["event_window_direction_hit_rate"])


def abnormal_return_error(
    sim_close: Sequence[float] | None,
    real_close: Sequence[float] | None,
    event_points: Iterable[Any] | None = None,
    *,
    dates: Sequence[Any] | None = None,
    pre_window: int = 3,
    post_window: int = 3,
) -> float:
    """Mean absolute error of event-window abnormal returns."""

    summary = event_study(
        sim_close=sim_close,
        real_close=real_close,
        event_points=event_points,
        dates=dates,
        pre_window=pre_window,
        post_window=post_window,
    )
    return float(summary["abnormal_return_error"])


def event_study(
    *,
    sim_close: Sequence[float] | None,
    real_close: Sequence[float] | None,
    event_points: Iterable[Any] | None,
    dates: Sequence[Any] | None = None,
    pre_window: int = 3,
    post_window: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare simulated and real behavior around event windows.

    Event indices refer to price points. For each event, the pre window is the
    mean return before the event and the post window is the mean return after
    the event, producing an abnormal-return delta for real and simulated paths.
    """

    sim = _clean_prices(sim_close)
    real = _clean_prices(real_close)
    n_price = min(sim.size, real.size)
    if n_price < 2:
        return {
            "event_count": 0,
            "pre_window": int(pre_window),
            "post_window": int(post_window),
            "event_window_direction_hit_rate": 0.0,
            "abnormal_return_error": 0.0,
            "mean_sim_abnormal_return": 0.0,
            "mean_real_abnormal_return": 0.0,
            "bootstrap_ci": bootstrap_ci([], seed=seed),
            "permutation_test": permutation_test([], [], seed=seed),
            "events": [],
        }

    sim_ret = _returns(sim[:n_price])
    real_ret = _returns(real[:n_price])
    n_ret = min(sim_ret.size, real_ret.size)
    indices = _event_indices(event_points=event_points, dates=dates, series_length=n_price)
    event_rows: list[Dict[str, Any]] = []
    for price_idx in indices:
        ret_idx = max(0, min(price_idx - 1, n_ret - 1))
        sim_pre = _window_slice(sim_ret[:n_ret], ret_idx, -abs(int(pre_window)), -1)
        real_pre = _window_slice(real_ret[:n_ret], ret_idx, -abs(int(pre_window)), -1)
        sim_post = _window_slice(sim_ret[:n_ret], ret_idx, 0, abs(int(post_window)) - 1)
        real_post = _window_slice(real_ret[:n_ret], ret_idx, 0, abs(int(post_window)) - 1)
        sim_abnormal = float((np.mean(sim_post) if sim_post.size else 0.0) - (np.mean(sim_pre) if sim_pre.size else 0.0))
        real_abnormal = float((np.mean(real_post) if real_post.size else 0.0) - (np.mean(real_pre) if real_pre.size else 0.0))
        event_rows.append(
            {
                "event_index": int(price_idx),
                "sim_pre_return": float(np.mean(sim_pre)) if sim_pre.size else 0.0,
                "real_pre_return": float(np.mean(real_pre)) if real_pre.size else 0.0,
                "sim_post_return": float(np.mean(sim_post)) if sim_post.size else 0.0,
                "real_post_return": float(np.mean(real_post)) if real_post.size else 0.0,
                "sim_abnormal_return": sim_abnormal,
                "real_abnormal_return": real_abnormal,
                "abnormal_error": float(abs(sim_abnormal - real_abnormal)),
                "direction_hit": bool(np.sign(sim_abnormal) == np.sign(real_abnormal)),
            }
        )

    if not event_rows:
        return {
            "event_count": 0,
            "pre_window": int(pre_window),
            "post_window": int(post_window),
            "event_window_direction_hit_rate": 0.0,
            "abnormal_return_error": 0.0,
            "mean_sim_abnormal_return": 0.0,
            "mean_real_abnormal_return": 0.0,
            "bootstrap_ci": bootstrap_ci([], seed=seed),
            "permutation_test": permutation_test([], [], seed=seed),
            "events": [],
        }

    sim_abn = [float(row["sim_abnormal_return"]) for row in event_rows]
    real_abn = [float(row["real_abnormal_return"]) for row in event_rows]
    errors = [float(row["abnormal_error"]) for row in event_rows]
    return {
        "event_count": int(len(event_rows)),
        "pre_window": int(pre_window),
        "post_window": int(post_window),
        "event_window_direction_hit_rate": float(np.mean([bool(row["direction_hit"]) for row in event_rows])),
        "abnormal_return_error": float(np.mean(errors)),
        "mean_sim_abnormal_return": float(np.mean(sim_abn)),
        "mean_real_abnormal_return": float(np.mean(real_abn)),
        "bootstrap_ci": bootstrap_ci(errors, seed=seed),
        "permutation_test": permutation_test(sim_abn, real_abn, seed=seed),
        "events": event_rows,
    }


__all__ = ["abnormal_return_error", "event_study", "event_window_direction_hit_rate"]
