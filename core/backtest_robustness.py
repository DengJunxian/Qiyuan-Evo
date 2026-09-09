"""回测稳健性分析：参数网格扫描 + Walk-Forward 验证。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional
import json

import numpy as np
import pandas as pd

from core.backtester import BacktestConfig, HistoricalBacktester

ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class GridSearchResult:
    records: List[Dict[str, Any]] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    metric: str = "score"

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)


@dataclass
class WalkForwardResult:
    folds: List[Dict[str, Any]] = field(default_factory=list)
    avg_test_return: float = 0.0
    avg_test_sharpe: float = 0.0
    pass_rate: float = 0.0
    optimize_metric: str = "score"

    def to_frame(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        return pd.DataFrame(self.folds)


class RobustnessAnalyzer:
    def __init__(
        self,
        base_config: BacktestConfig,
        historical_data: Optional[pd.DataFrame] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
    ):
        self.base_config = base_config
        self.historical_data = historical_data.copy() if historical_data is not None else None
        self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else None

    @staticmethod
    def _clone_config(base: BacktestConfig, updates: Dict[str, Any]) -> BacktestConfig:
        payload = asdict(base)
        payload.update(updates)
        return BacktestConfig(**payload)

    @staticmethod
    def _normalize_param_grid(param_grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        norm: Dict[str, List[Any]] = {}
        for k, values in param_grid.items():
            cleaned = []
            for v in values:
                if v is None:
                    continue
                if isinstance(v, float) and np.isnan(v):
                    continue
                cleaned.append(v)
            if cleaned:
                norm[k] = cleaned
        return norm

    @staticmethod
    def _score(row: Dict[str, Any], metric: str) -> float:
        if metric == "total_return":
            return float(row.get("total_return", 0.0))
        if metric == "sharpe_ratio":
            return float(row.get("sharpe_ratio", 0.0))
        if metric == "calmar_ratio":
            return float(row.get("calmar_ratio", 0.0))
        if metric == "information_ratio":
            return float(row.get("information_ratio", 0.0))

        return (
            float(row.get("sharpe_ratio", 0.0))
            + 0.6 * float(row.get("total_return", 0.0))
            + 0.3 * float(row.get("information_ratio", 0.0))
            + 0.2 * float(row.get("calmar_ratio", 0.0))
            - 1.2 * abs(float(row.get("max_drawdown", 0.0)))
        )

    @staticmethod
    def _slice_by_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        sub = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
        return sub.reset_index(drop=True)

    def _create_backtester(self, config: BacktestConfig) -> HistoricalBacktester:
        bt = HistoricalBacktester(config)
        if self.historical_data is not None and not self.historical_data.empty:
            bt.historical_data = self.historical_data.copy()
        if self.benchmark_data is not None and not self.benchmark_data.empty:
            bt.benchmark_data = self.benchmark_data.copy()
        return bt

    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        optimize_metric: str = "score",
        progress_callback: ProgressCallback = None,
    ) -> GridSearchResult:
        grid = self._normalize_param_grid(param_grid)
        if not grid:
            return GridSearchResult(metric=optimize_metric)

        keys = list(grid.keys())
        combos = list(product(*[grid[k] for k in keys]))
        rows: List[Dict[str, Any]] = []

        total = len(combos)
        for idx, combo in enumerate(combos, start=1):
            updates = dict(zip(keys, combo))
            config = self._clone_config(self.base_config, updates)
            bt = self._create_backtester(config)
            result = bt.run_backtest()

            row = {
                "combo_id": idx,
                "params": json.dumps(updates, ensure_ascii=False),
                "total_return": float(result.total_return),
                "sharpe_ratio": float(result.sharpe_ratio),
                "max_drawdown": float(result.max_drawdown),
                "calmar_ratio": float(result.calmar_ratio),
                "information_ratio": float(result.information_ratio),
                "total_trades": int(result.total_trades),
            }
            for k, v in updates.items():
                row[k] = v

            row["score"] = self._score(row, optimize_metric)
            rows.append(row)

            if progress_callback:
                progress_callback(idx, total, f"网格扫描 {idx}/{total}")

        rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        best = rows[0] if rows else {}
        best_params: Dict[str, Any] = {}
        for key in keys:
            if key in best:
                best_params[key] = best[key]

        return GridSearchResult(
            records=rows,
            best_params=best_params,
            best_score=float(best.get("score", 0.0)) if best else 0.0,
            metric=optimize_metric,
        )

    def run_walk_forward(
        self,
        param_grid: Dict[str, List[Any]],
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 63,
        optimize_metric: str = "score",
        progress_callback: ProgressCallback = None,
    ) -> WalkForwardResult:
        if self.historical_data is None or self.historical_data.empty:
            bt = HistoricalBacktester(self.base_config)
            if not bt.load_data():
                return WalkForwardResult(optimize_metric=optimize_metric)
            all_data = bt.historical_data.copy()
            all_benchmark = bt.benchmark_data.copy()
        else:
            all_data = self.historical_data.copy()
            all_benchmark = self.benchmark_data.copy() if self.benchmark_data is not None else pd.DataFrame()

        all_data = all_data.sort_values("date").reset_index(drop=True)
        if all_data.empty:
            return WalkForwardResult(optimize_metric=optimize_metric)

        n = len(all_data)
        fold_starts = list(range(0, max(n - train_days - test_days + 1, 0), max(step_days, 1)))
        if not fold_starts:
            return WalkForwardResult(optimize_metric=optimize_metric)

        folds: List[Dict[str, Any]] = []
        total_folds = len(fold_starts)

        for fi, start_idx in enumerate(fold_starts, start=1):
            train_start = all_data.iloc[start_idx]["date"]
            train_end = all_data.iloc[start_idx + train_days - 1]["date"]
            test_start = all_data.iloc[start_idx + train_days]["date"]
            test_end = all_data.iloc[start_idx + train_days + test_days - 1]["date"]

            train_data = self._slice_by_dates(all_data, train_start, train_end)
            test_data = self._slice_by_dates(all_data, test_start, test_end)

            if train_data.empty or test_data.empty:
                continue

            if all_benchmark is not None and not all_benchmark.empty:
                train_bm = self._slice_by_dates(all_benchmark, train_start, train_end)
                test_bm = self._slice_by_dates(all_benchmark, test_start, test_end)
            else:
                train_bm = pd.DataFrame()
                test_bm = pd.DataFrame()


            train_analyzer = RobustnessAnalyzer(
                self.base_config,
                historical_data=train_data,
                benchmark_data=train_bm,
            )
            grid = train_analyzer.run_grid_search(param_grid, optimize_metric=optimize_metric)
            best_params = grid.best_params


            test_config = self._clone_config(self.base_config, best_params)
            test_bt = HistoricalBacktester(test_config)
            test_bt.historical_data = test_data.copy()
            if test_bm is not None and not test_bm.empty:
                test_bt.benchmark_data = test_bm.copy()
            test_res = test_bt.run_backtest()

            fold_row = {
                "fold": fi,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "best_params": json.dumps(best_params, ensure_ascii=False),
                "test_total_return": float(test_res.total_return),
                "test_sharpe_ratio": float(test_res.sharpe_ratio),
                "test_max_drawdown": float(test_res.max_drawdown),
                "test_calmar_ratio": float(test_res.calmar_ratio),
                "test_total_trades": int(test_res.total_trades),
            }
            folds.append(fold_row)

            if progress_callback:
                progress_callback(fi, total_folds, f"Walk-Forward 折 {fi}/{total_folds}")

        if not folds:
            return WalkForwardResult(optimize_metric=optimize_metric)

        df = pd.DataFrame(folds)
        avg_ret = float(df["test_total_return"].mean())
        avg_sharpe = float(df["test_sharpe_ratio"].mean())
        pass_rate = float((df["test_total_return"] > 0).mean())

        return WalkForwardResult(
            folds=folds,
            avg_test_return=avg_ret,
            avg_test_sharpe=avg_sharpe,
            pass_rate=pass_rate,
            optimize_metric=optimize_metric,
        )
