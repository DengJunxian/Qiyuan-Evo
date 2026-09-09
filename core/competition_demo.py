"""Competition demo mode helpers for Civitas-Economica-Demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional
import json

import pandas as pd
import plotly.graph_objects as go

from core.ui_text import translate_display_text

LIVE_MODE = "LIVE_MODE"
DEMO_MODE = "DEMO_MODE"
COMPETITION_DEMO_MODE = "COMPETITION_DEMO_MODE"

REQUIRED_SCENARIOS = (
    "tax_cut_liquidity_boost",
    "rumor_panic_selloff",
    "regulator_stabilization_intervention",
)

REQUIRED_METRICS_COLUMNS = {
    "step",
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "csad",
    "panic_level",
}
REQUIRED_SCENARIO_FILES = (
    "initial_config.yaml",
    "analyst_manager_output.json",
    "narration.json",
    "metrics.csv",
)


@dataclass
class DemoScenario:
    name: str
    config: Dict[str, Any]
    analyst_manager_output: Dict[str, Any]
    narration: List[Dict[str, Any]]
    metrics: pd.DataFrame
    root: Path


def _scenario_root(base_dir: Optional[Path] = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return Path(__file__).resolve().parents[1] / "demo_scenarios"


def _scenario_missing_files(scenario_dir: Path) -> List[str]:
    missing: List[str] = []
    for name in REQUIRED_SCENARIO_FILES:
        if not (scenario_dir / name).exists():
            missing.append(name)
    return missing


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_yaml_load(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            return data
        return {"value": data}
    except Exception:
        result: Dict[str, Any] = {}
        parents: List[tuple[int, Dict[str, Any]]] = [(0, result)]
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            while len(parents) > 1 and indent < parents[-1][0]:
                parents.pop()
            current = parents[-1][1]
            if value == "":
                node: Dict[str, Any] = {}
                current[key] = node
                parents.append((indent + 2, node))
            else:
                current[key] = _parse_scalar(value)
        return result


def list_competition_scenarios(base_dir: Optional[Path] = None, *, only_valid: bool = True) -> List[str]:
    root = _scenario_root(base_dir)
    if not root.exists():
        return []
    scenarios: List[str] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        if only_valid and _scenario_missing_files(item):
            continue
        scenarios.append(item.name)
    return sorted(scenarios)


def load_competition_scenario(name: str, base_dir: Optional[Path] = None) -> DemoScenario:
    root = _scenario_root(base_dir)
    scenario_dir = root / name
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_dir}")
    missing = _scenario_missing_files(scenario_dir)
    if missing:
        raise FileNotFoundError(f"Scenario incomplete: {scenario_dir}, missing={missing}")

    config = _minimal_yaml_load(scenario_dir / "initial_config.yaml")
    analyst_manager_output = json.loads((scenario_dir / "analyst_manager_output.json").read_text(encoding="utf-8-sig"))
    narration = json.loads((scenario_dir / "narration.json").read_text(encoding="utf-8-sig"))
    metrics = pd.read_csv(scenario_dir / "metrics.csv")

    missing_columns = REQUIRED_METRICS_COLUMNS - set(metrics.columns)
    if missing_columns:
        raise ValueError(f"Scenario metrics missing required columns: {sorted(missing_columns)}")

    metrics = metrics.sort_values("step").reset_index(drop=True)
    metrics["step"] = metrics["step"].astype(int)

    return DemoScenario(
        name=name,
        config=config,
        analyst_manager_output=analyst_manager_output,
        narration=narration if isinstance(narration, list) else [],
        metrics=metrics,
        root=scenario_dir,
    )


def switch_runtime_mode(state: MutableMapping[str, Any], mode: str) -> None:
    if mode not in {LIVE_MODE, DEMO_MODE}:
        raise ValueError(f"Unsupported runtime mode: {mode}")
    state["runtime_mode"] = mode
    state["competition_mode"] = COMPETITION_DEMO_MODE if mode == DEMO_MODE else ""
    if mode == LIVE_MODE:
        state["demo_autoplay"] = False
        state["is_running"] = False


def bootstrap_competition_demo(
    state: MutableMapping[str, Any],
    scenario_name: str,
    base_dir: Optional[Path] = None,
    auto_play: bool = True,
) -> DemoScenario:
    available = list_competition_scenarios(base_dir=base_dir, only_valid=True)
    selected_name = scenario_name if scenario_name in available else (available[0] if available else "")
    if not selected_name:
        raise FileNotFoundError("No valid competition scenario found.")
    scenario = load_competition_scenario(selected_name, base_dir=base_dir)

    state["controller"] = None
    state["runtime_mode"] = DEMO_MODE
    state["competition_mode"] = COMPETITION_DEMO_MODE
    state["demo_scenario_name"] = selected_name
    state["demo_scenario"] = scenario
    state["demo_step_cursor"] = 0
    state["demo_narration_cursor"] = 0
    state["demo_autoplay"] = bool(auto_play)
    state["is_running"] = bool(auto_play)
    state["market_history"] = []
    state["csad_history"] = []
    state["demo_last_narration"] = None
    state["demo_last_panic_level"] = 0.0
    state["demo_last_step"] = 0

    return scenario


def _collect_narration_for_step(scenario: DemoScenario, step: int) -> List[Dict[str, Any]]:
    return [item for item in scenario.narration if int(item.get("step", -1)) == int(step)]


def replay_next_narration(state: MutableMapping[str, Any]) -> Optional[Dict[str, Any]]:
    scenario = state.get("demo_scenario")
    if scenario is None:
        return None
    narration = scenario.narration
    idx = int(state.get("demo_narration_cursor", 0))
    if idx >= len(narration):
        return None
    entry = narration[idx]
    state["demo_narration_cursor"] = idx + 1
    state["demo_last_narration"] = entry
    return entry


def advance_competition_demo(state: MutableMapping[str, Any], steps: int = 1) -> Dict[str, Any]:
    scenario: Optional[DemoScenario] = state.get("demo_scenario")
    if scenario is None:
        return {"advanced": 0, "done": True, "narration": [], "latest_row": None}

    cursor = int(state.get("demo_step_cursor", 0))
    total = len(scenario.metrics)
    advanced = 0
    narration_hits: List[Dict[str, Any]] = []
    latest_row: Optional[Dict[str, Any]] = None

    while advanced < max(1, int(steps)) and cursor < total:
        row = scenario.metrics.iloc[cursor].to_dict()
        latest_row = row
        state["market_history"].append(
            {
                "time": row["time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "is_historical": False,
            }
        )
        state["csad_history"].append(float(row.get("csad", 0.0)))
        state["demo_last_panic_level"] = float(row.get("panic_level", 0.0))
        state["demo_last_step"] = int(row.get("step", cursor + 1))

        narration_hits.extend(_collect_narration_for_step(scenario, int(row["step"])))
        cursor += 1
        advanced += 1

    state["demo_step_cursor"] = cursor
    done = cursor >= total
    if done:
        state["demo_autoplay"] = False
        state["is_running"] = False
    if narration_hits:
        state["demo_last_narration"] = narration_hits[-1]

    return {
        "advanced": advanced,
        "done": done,
        "narration": narration_hits,
        "latest_row": latest_row,
    }


def build_competition_metrics_figure(metrics: pd.DataFrame, upto_step: Optional[int] = None) -> go.Figure:
    frame = metrics.copy()
    if upto_step is not None:
        frame = frame[frame["step"] <= int(upto_step)]
    if frame.empty:
        frame = metrics.head(1).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["close"],
            mode="lines+markers",
            name="指数收盘",
            line=dict(color="#4DA6FF", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["csad"],
            mode="lines",
            name="CSAD",
            yaxis="y2",
            line=dict(color="#FFD60A", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title=translate_display_text("综合展示指标总览"),
        template="plotly_dark",
        yaxis=dict(title="指数点位"),
        yaxis2=dict(title="CSAD", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
    )
    return fig
