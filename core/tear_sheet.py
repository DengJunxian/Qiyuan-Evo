"""Standardized tear sheet builders for Civitas outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json

import numpy as np
import pandas as pd

from core.performance import compute_performance_metrics


@dataclass
class TearSheetPayload:
    scenario_name: str
    metrics: Dict[str, float]
    returns: List[float] = field(default_factory=list)
    benchmark_returns: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "metrics": self.metrics,
            "returns": self.returns,
            "benchmark_returns": self.benchmark_returns,
            "dates": self.dates,
            "metadata": self.metadata,
        }


def _to_list(values: Optional[Sequence[float]]) -> List[float]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(x) for x in arr.tolist()]


def build_standard_tear_sheet(
    *,
    scenario_name: str,
    returns: Sequence[float],
    benchmark_returns: Optional[Sequence[float]] = None,
    dates: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> TearSheetPayload:
    metrics = compute_performance_metrics(returns, benchmark_returns)
    return TearSheetPayload(
        scenario_name=scenario_name,
        metrics=metrics,
        returns=_to_list(returns),
        benchmark_returns=_to_list(benchmark_returns),
        dates=[str(d) for d in dates] if dates is not None else [],
        metadata=dict(metadata or {}),
    )


def compare_scenarios(payloads: Iterable[TearSheetPayload]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        row: Dict[str, Any] = {"scenario": payload.scenario_name}
        row.update(payload.metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_by = "sharpe_ratio" if "sharpe_ratio" in df.columns else df.columns[-1]
    return df.sort_values(sort_by, ascending=False).reset_index(drop=True)


def render_html_tear_sheet(payload: TearSheetPayload) -> str:
    m = payload.metrics
    scenario = payload.scenario_name
    return f"""
    <div style="font-family:Arial,sans-serif;padding:16px;border:1px solid #dbe4f0;border-radius:10px;background:#f7fbff;">
      <h3 style="margin:0 0 10px 0;">Civitas Tear Sheet: {scenario}</h3>
      <table style="border-collapse:collapse;width:100%;background:#ffffff;">
        <tr><th style="text-align:left;padding:8px;border-bottom:1px solid #e3e9f2;">Metric</th><th style="text-align:right;padding:8px;border-bottom:1px solid #e3e9f2;">Value</th></tr>
        <tr><td style="padding:8px;">Total Return</td><td style="padding:8px;text-align:right;">{m.get('total_return', 0.0):.2%}</td></tr>
        <tr><td style="padding:8px;">Annual Return</td><td style="padding:8px;text-align:right;">{m.get('annual_return', 0.0):.2%}</td></tr>
        <tr><td style="padding:8px;">Sharpe Ratio</td><td style="padding:8px;text-align:right;">{m.get('sharpe_ratio', 0.0):.3f}</td></tr>
        <tr><td style="padding:8px;">Sortino Ratio</td><td style="padding:8px;text-align:right;">{m.get('sortino_ratio', 0.0):.3f}</td></tr>
        <tr><td style="padding:8px;">Max Drawdown</td><td style="padding:8px;text-align:right;">{m.get('max_drawdown', 0.0):.2%}</td></tr>
        <tr><td style="padding:8px;">Calmar Ratio</td><td style="padding:8px;text-align:right;">{m.get('calmar_ratio', 0.0):.3f}</td></tr>
        <tr><td style="padding:8px;">Information Ratio</td><td style="padding:8px;text-align:right;">{m.get('information_ratio', 0.0):.3f}</td></tr>
        <tr><td style="padding:8px;">VaR 95%</td><td style="padding:8px;text-align:right;">{m.get('var_95', 0.0):.3%}</td></tr>
        <tr><td style="padding:8px;">CVaR 95%</td><td style="padding:8px;text-align:right;">{m.get('cvar_95', 0.0):.3%}</td></tr>
        <tr><td style="padding:8px;">Omega Ratio</td><td style="padding:8px;text-align:right;">{m.get('omega_ratio', 0.0):.3f}</td></tr>
      </table>
    </div>
    """


def export_tear_sheet_html(payload: TearSheetPayload, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html_tear_sheet(payload), encoding="utf-8")
    return str(out.resolve())


def export_tear_sheet_json(payload: TearSheetPayload, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out.resolve())


def export_quantstats_html(
    *,
    returns: Sequence[float],
    output_path: str,
    benchmark_returns: Optional[Sequence[float]] = None,
    dates: Optional[Sequence[str]] = None,
    title: str = "Civitas QuantStats Report",
) -> str:
    """Export a QuantStats HTML report if quantstats is installed."""

    try:
        import quantstats as qs  # pragma: no cover - optional dependency
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("quantstats is not installed. Install quantstats to export this report.") from exc

    returns_series = pd.Series(_to_list(returns))
    benchmark_series = pd.Series(_to_list(benchmark_returns)) if benchmark_returns is not None else None

    if dates is not None and len(dates) == len(returns_series):
        idx = pd.to_datetime(list(dates))
        returns_series.index = idx
        if benchmark_series is not None and len(benchmark_series) == len(idx):
            benchmark_series.index = idx
    else:
        idx = pd.date_range("2000-01-01", periods=len(returns_series), freq="B")
        returns_series.index = idx
        if benchmark_series is not None and len(benchmark_series) == len(idx):
            benchmark_series.index = idx

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(returns_series, benchmark=benchmark_series, output=str(out), title=title)
    return str(out.resolve())

