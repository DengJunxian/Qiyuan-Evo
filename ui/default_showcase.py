"""Default, reproducible showcase for the policy and history experiment chain."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from core.backtester import BacktestConfig, FactorBacktestEngine
from core.data.market_data_provider import MarketDataProvider, MarketDataQuery
from ui.history_replay import _build_replay_metrics, _compile_policy_score
from ui.policy_lab import (
    _build_regulation_counterfactual_worlds,
    _compile_policy_bundle,
    _compute_policy_summary,
    _generate_policy_metrics,
    _load_policy_templates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYMBOL = "sh000300"
DEFAULT_SYMBOL_LABEL = "沪深 300"
DEFAULT_HISTORY_START = "2024-09-24"
DEFAULT_HISTORY_END = "2026-05-29"
DEFAULT_SEED = 42

CHANNEL_LABELS = {
    "compliance_intensity": "交易摩擦重定价",
    "liquidity_supply": "流动性供给",
    "risk_appetite": "风险偏好",
    "sector_preference": "板块配置",
    "liquidity": "流动性",
    "tax_frictions": "税费摩擦",
}


def _json_records(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    if frame.empty:
        return []
    safe = frame.replace([np.inf, -np.inf], np.nan)
    return json.loads(safe.to_json(orient="records", force_ascii=False, date_format="iso"))


def _frame_digest(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return sha256(payload).hexdigest()


def _candidate_market_frames(cache_dir: Path) -> Iterable[Tuple[Path, pd.DataFrame]]:
    for path in sorted(cache_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"date", "open", "high", "low", "close", "volume", "symbol"}
        if not required.issubset(frame.columns):
            continue
        if not (frame["symbol"].astype(str) == DEFAULT_SYMBOL).any():
            continue
        frame = frame[frame["symbol"].astype(str) == DEFAULT_SYMBOL].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame[
            (frame["date"] >= DEFAULT_HISTORY_START)
            & (frame["date"] <= DEFAULT_HISTORY_END)
        ]
        if len(frame) >= 200:
            yield path, frame.reset_index(drop=True)


def _load_internal_market_snapshot(project_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load only project-local data; never reach an external market endpoint."""

    candidates = list(_candidate_market_frames(project_root / "data" / "cache" / "market"))
    if candidates:
        # Prefer a real cached provider, then the longest usable snapshot.
        def rank(item: Tuple[Path, pd.DataFrame]) -> Tuple[int, int]:
            _, frame = item
            providers = set(frame.get("provider", pd.Series(dtype=str)).astype(str).str.lower())
            real_rank = 1 if providers and providers != {"synthetic"} else 0
            return real_rank, len(frame)

        path, frame = max(candidates, key=rank)
        providers = sorted(set(frame.get("provider", pd.Series(["local_cache"])).astype(str)))
        return frame, {
            "kind": "local_cache",
            "label": "项目内置历史行情缓存",
            "providers": providers,
            "path": str(path.relative_to(project_root)),
            "external_network": False,
        }

    # A deterministic internal fallback keeps the showcase runnable in a clean clone.
    provider = MarketDataProvider(
        cache_dir=str(project_root / "tmp" / "default_showcase" / "market_cache"),
        snapshot_dir=str(project_root / "tmp" / "default_showcase" / "snapshots"),
        provider_priority=("synthetic",),
    )
    query = MarketDataQuery(
        symbol=DEFAULT_SYMBOL,
        interval="1d",
        start=DEFAULT_HISTORY_START,
        end=DEFAULT_HISTORY_END,
        period_days=0,
        adjust="",
        market="CN",
    )
    frame = provider.get_ohlcv(query, use_cache=True, freeze_snapshot=False)
    return frame.reset_index(drop=True), {
        "kind": "deterministic_fallback",
        "label": "内部确定性行情回退",
        "providers": ["synthetic"],
        "path": "tmp/default_showcase/market_cache",
        "external_network": False,
    }


def _normalize_path(values: pd.Series | list[float]) -> list[float]:
    series = pd.Series(values, dtype=float)
    if series.empty or not np.isfinite(series.iloc[0]) or abs(float(series.iloc[0])) < 1e-12:
        return [100.0 for _ in range(len(series))]
    return (series / float(series.iloc[0]) * 100.0).round(4).tolist()


def _pick_default_policy_template() -> Dict[str, Any]:
    templates = _load_policy_templates()
    if not templates:
        raise RuntimeError("未找到默认政策模板。")
    for item in templates:
        if str(item.get("id", "")) == "stamp-tax-liquidity":
            return dict(item)
    return dict(templates[0])


def build_default_showcase_payload(project_root: Path | None = None) -> Dict[str, Any]:
    """Run the default policy experiment and historical validation via internal APIs."""

    root = Path(project_root or PROJECT_ROOT).resolve()
    market_frame, source_meta = _load_internal_market_snapshot(root)
    if market_frame.empty:
        raise RuntimeError("内部行情快照为空，无法运行默认展示实验。")

    market_frame = market_frame.sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        market_frame[column] = pd.to_numeric(market_frame[column], errors="coerce")
    market_frame = market_frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    snapshot_digest = _frame_digest(market_frame)

    template = _pick_default_policy_template()
    duration_days = int(template.get("recommended_duration", 30) or 30)
    intensity = float(template.get("recommended_intensity", 1.0) or 1.0)
    policy_text = str(template.get("policy_text", ""))
    reference = market_frame.tail(duration_days).copy().reset_index(drop=True)
    reference["time"] = reference["date"].astype(str)
    reference["step"] = np.arange(1, len(reference) + 1)
    reference = reference[["step", "time", "open", "high", "low", "close", "volume"]]

    _, package = _compile_policy_bundle(
        policy_text,
        intensity,
        policy_type_hint=str(template.get("policy_type", "")) or None,
        market_regime="policy_support",
        enable_structured_parser=True,
    )
    package_dict = package.to_dict()
    policy_frame = _generate_policy_metrics(
        policy_text=policy_text,
        intensity=intensity,
        duration_days=duration_days,
        rumor_noise=bool(template.get("default_rumor_noise", False)),
        scenario_key=str(template.get("id", "default_policy")),
        market_history=reference,
    )
    policy_summary = _compute_policy_summary(policy_frame)
    counterfactual = _build_regulation_counterfactual_worlds(policy_frame, intensity=intensity)

    channel_rows = []
    for channel in list(package_dict.get("channels", []) or []):
        if str(channel.get("channel_type", "")) == "legacy_alias":
            continue
        name = str(channel.get("name", ""))
        channel_rows.append(
            {
                "name": name,
                "label": CHANNEL_LABELS.get(name, name),
                "impact": float(channel.get("impact", 0.0) or 0.0),
                "lag_days": int(channel.get("lag_days", 0) or 0),
                "direction": str(channel.get("direction", "")),
            }
        )

    policy_score = _compile_policy_score(policy_text, 1.5, "中性")
    backtest_config = BacktestConfig(
        symbol=DEFAULT_SYMBOL,
        benchmark_symbol=DEFAULT_SYMBOL,
        start_date=DEFAULT_HISTORY_START,
        end_date=DEFAULT_HISTORY_END,
        period_days=0,
        strategy_name="portfolio_system",
        lookback=20,
        rebalance_frequency=5,
        max_position=1.0,
        policy_shock=policy_score,
        policy_text=policy_text,
        sentiment_weight=0.55,
        civitas_factor_weight=0.45,
        news_source_strategy="local",
        persist_news_events=False,
        auth_score_mode="strict",
        random_seed=DEFAULT_SEED,
        runtime_mode="SMART",
        feature_flags={
            "agent_replay": False,
            "strict_history_replay": True,
            "default_showcase": True,
            "external_network": False,
        },
    )
    backtest_frame = market_frame[["date", "open", "high", "low", "close", "volume"]].copy()
    engine = FactorBacktestEngine(backtest_config)
    # Supplying the project-local snapshot explicitly guarantees that the run stays internal.
    engine.historical_data = backtest_frame.copy()
    engine.benchmark_data = backtest_frame[["date", "close"]].rename(
        columns={"close": "benchmark_close"}
    )
    history_result = engine.run_backtest()
    if history_result.total_days <= 0 or not history_result.real_prices:
        raise RuntimeError("历史验证未生成有效序列。")
    replay_metrics = _build_replay_metrics(history_result)

    history_path = pd.DataFrame(
        {
            "date": history_result.dates,
            "real": history_result.real_prices,
            "simulated": history_result.simulated_prices,
        }
    )
    history_path["real_normalized"] = _normalize_path(history_path["real"])
    history_path["simulated_normalized"] = _normalize_path(history_path["simulated"])

    policy_path = policy_frame[["step", "time", "close", "panic_level", "csad"]].copy()
    policy_path["policy_normalized"] = _normalize_path(policy_path["close"])
    policy_path["reference_normalized"] = _normalize_path(reference["close"])

    worlds_payload: Dict[str, list[Dict[str, Any]]] = {}
    for world_name, rows in dict(counterfactual.get("worlds", {}) or {}).items():
        world_frame = pd.DataFrame(rows)
        if world_frame.empty:
            continue
        world_frame["normalized"] = _normalize_path(world_frame["close"])
        worlds_payload[str(world_name)] = _json_records(
            world_frame[["step", "time", "close", "panic_level", "normalized"]]
        )

    run_material = {
        "policy_template": template.get("id"),
        "policy_text": policy_text,
        "policy_intensity": intensity,
        "history_window": [DEFAULT_HISTORY_START, DEFAULT_HISTORY_END],
        "history_symbol": DEFAULT_SYMBOL,
        "random_seed": DEFAULT_SEED,
        "snapshot_sha256": snapshot_digest,
    }
    run_id = sha256(
        json.dumps(run_material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12].upper()

    return {
        "run": {
            "run_id": f"CE-{run_id}",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "status": "completed",
            "api_scope": "internal_only",
            "random_seed": DEFAULT_SEED,
            "snapshot_sha256": snapshot_digest,
            "snapshot_short": snapshot_digest[:12],
            "source": source_meta,
        },
        "policy": {
            "template_id": str(template.get("id", "")),
            "title": str(template.get("title", "默认政策实验")),
            "category": str(template.get("category", "政策场景")),
            "policy_type": str(package_dict.get("event", {}).get("policy_type", "")),
            "policy_label": str(package_dict.get("event", {}).get("policy_label", "")),
            "policy_text": policy_text,
            "goal": str(template.get("policy_goal", "")),
            "intensity": intensity,
            "duration_days": duration_days,
            "confidence": float(package_dict.get("uncertainty", {}).get("confidence", 0.0) or 0.0),
            "expected_lag_days": int(package_dict.get("event", {}).get("expected_lag_days", 0) or 0),
            "summary": {key: float(value) for key, value in policy_summary.items()},
            "channels": channel_rows,
            "path": _json_records(policy_path),
            "worlds": worlds_payload,
            "counterfactual_scorecards": counterfactual.get("scorecards", {}),
            "recommended_timing": str(counterfactual.get("recommended_timing", "")),
            "intervention_steps": counterfactual.get("intervention_steps", {}),
        },
        "history": {
            "symbol": DEFAULT_SYMBOL,
            "symbol_label": DEFAULT_SYMBOL_LABEL,
            "start_date": DEFAULT_HISTORY_START,
            "end_date": DEFAULT_HISTORY_END,
            "total_days": int(history_result.total_days),
            "total_trades": int(history_result.total_trades),
            "total_return": float(history_result.total_return),
            "benchmark_return": float(history_result.benchmark_return),
            "excess_return": float(history_result.excess_return),
            "max_drawdown": float(history_result.max_drawdown),
            "annual_volatility": float(history_result.annual_volatility),
            "credibility_score": float(history_result.credibility_score),
            "metrics": {key: float(value) for key, value in replay_metrics.items()},
            "path": _json_records(history_path),
            "config_hash": str(history_result.metadata.get("config_hash", "")),
            "data_snapshot": dict(history_result.metadata.get("data_snapshot", {}) or {}),
        },
    }


@st.cache_data(show_spinner=False)
def _cached_default_showcase_payload() -> Dict[str, Any]:
    return build_default_showcase_payload(PROJECT_ROOT)


def write_default_showcase_artifact(
    output_path: Path | None = None,
    *,
    project_root: Path | None = None,
) -> Path:
    root = Path(project_root or PROJECT_ROOT).resolve()
    target = Path(output_path or root / "outputs" / "default_showcase" / "default_showcase_run.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_default_showcase_payload(root)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


SHOWCASE_GALLERY = [
    {
        "file": "01-policy-input.png",
        "index": "01",
        "stage": "政策输入",
        "title": "从自然语言进入结构化实验",
        "summary": "默认政策文本、强度、期限与对照基线在同一入口完成配置，形成可执行的政策冲击。",
        "class": "gallery-tall",
    },
    {
        "file": "02-live-session.png",
        "index": "02",
        "stage": "会话推演",
        "title": "资金、情绪与订单流同步演化",
        "summary": "会话推进到第 46 个交易日，实时观察买卖主导、情绪热度、指数动作与 K 线反馈。",
        "class": "gallery-feature",
    },
    {
        "file": "03-mechanism-analysis.png",
        "index": "03",
        "stage": "机制解释",
        "title": "把政策影响拆到主体与板块",
        "summary": "投资者分歧、角色订单流、板块热度和大模型结构化解读共同解释价格结果。",
        "class": "gallery-tall",
    },
    {
        "file": "04-research-workbench.png",
        "index": "04",
        "stage": "研究工作台",
        "title": "不同干预方案进入同一比较坐标",
        "summary": "场景对比、风险贡献与指标重要性用于判断干预时机及政策组合的稳健性。",
        "class": "gallery-wide",
    },
    {
        "file": "07-counterfactual-objectives.png",
        "index": "05",
        "stage": "反事实评估",
        "title": "比较不介入、提前介入与延后介入",
        "summary": "多条世界线同时展示收益、回撤、恐慌度和波动率，并沉淀候选指标池。",
        "class": "gallery-tall",
    },
    {
        "file": "05-policy-report.png",
        "index": "06",
        "stage": "结果归档",
        "title": "自动形成可复盘的政策报告",
        "summary": "会话摘要、政策时间轴、影响评价、风险副作用与分阶段建议汇总为报告。",
        "class": "gallery-document",
    },
    {
        "file": "06-history-validation.png",
        "index": "07",
        "stage": "历史验证",
        "title": "让仿真路径与真实市场同图对照",
        "summary": "自动提取历史窗口政策与新闻，比较真实走势和仿真走势的方向、转折与滞后。",
        "class": "gallery-feature",
    },
    {
        "file": "08-authenticity-overview.png",
        "index": "08",
        "stage": "综合结论",
        "title": "从拟真评分回到证据边界",
        "summary": "综合拟真、新闻覆盖、智能体行为与偏差解读共同说明结果可信度及使用边界。",
        "class": "gallery-tall",
    },
]


def _showcase_image_url(file_name: str) -> str:
    path = PROJECT_ROOT / "static" / "showcase_gallery" / file_name
    if not path.exists():
        return ""
    # Keep this URL relative so Streamlit's configured baseUrlPath is preserved
    # when the app is served behind a proxy or from a hosted subpath.
    return f"app/static/showcase_gallery/{file_name}"


def _gallery_figure(item: Dict[str, str], *, featured: bool = False) -> str:
    source = _showcase_image_url(str(item["file"]))
    if not source:
        return ""
    feature_class = " gallery-frame-featured" if featured else ""
    return (
        f'<figure class="gallery-frame {escape(str(item["class"]))}{feature_class}">'
        f'<a href="{source}" target="_blank" aria-label="查看{escape(str(item["title"]))}原图">'
        f'<img src="{source}" alt="{escape(str(item["title"]))}" loading="lazy" /></a>'
        '<figcaption>'
        f'<span>{escape(str(item["index"]))} · {escape(str(item["stage"]))}</span>'
        f'<h3>{escape(str(item["title"]))}</h3>'
        f'<p>{escape(str(item["summary"]))}</p>'
        '</figcaption></figure>'
    )


def _render_showcase_css() -> None:
    st.markdown(
        """
        <style>
        .showcase-hero { position: relative; overflow: hidden; min-height: 320px; margin: 0 0 34px; padding: 46px 48px 38px; border: 1px solid rgba(56,189,248,.18); border-radius: 20px; background: linear-gradient(110deg, rgba(3,10,20,.96), rgba(7,24,43,.94) 58%, rgba(4,17,31,.98)); animation: showcaseRise .58s ease both; }
        .showcase-hero::before {
            content: ""; position: absolute; inset: 0; pointer-events: none;
            background:
                linear-gradient(90deg, transparent 0 49.8%, rgba(56,189,248,.06) 50%, transparent 50.2%),
                linear-gradient(0deg, transparent 0 49.8%, rgba(56,189,248,.05) 50%, transparent 50.2%);
            background-size: 84px 84px; mask-image: linear-gradient(90deg, transparent 28%, #000 100%);
        }
        .showcase-orbit {
            position: absolute; width: 430px; height: 430px; right: -116px; top: -124px;
            border: 1px solid rgba(56,189,248,.17); border-radius: 50%;
            box-shadow: 0 0 0 46px rgba(56,189,248,.025), 0 0 0 92px rgba(56,189,248,.018);
        }
        .showcase-eyebrow { color: #38bdf8; letter-spacing: .18em; font-size: 12px; font-weight: 700; }
        .showcase-hero h1 { max-width: 800px; margin: 18px 0 14px; font-size: clamp(42px, 5vw, 68px); line-height: 1.03; letter-spacing: -.045em; }
        .showcase-hero p { max-width: 700px; color: #9eb0c7; font-size: 16px; line-height: 1.75; }
        .showcase-status { display: flex; gap: 18px; align-items: center; margin-top: 26px; color: #cfe7f7; font-size: 13px; }
        .showcase-status-dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 16px rgba(52,211,153,.8); }
        .showcase-section { margin: 0 0 34px; padding-top: 8px; }
        .showcase-section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 24px; padding-bottom: 18px; border-bottom: 1px solid rgba(148,163,184,.14); }
        .showcase-index { color: #38bdf8; font-size: 12px; letter-spacing: .18em; }
        .showcase-section h2 { margin: 6px 0 0; font-size: 34px; letter-spacing: -.03em; }
        .showcase-section-note { max-width: 520px; color: #7f91a9; font-size: 13px; line-height: 1.65; text-align: right; }
        .showcase-route { display: grid; grid-template-columns: repeat(4, 1fr); margin: 0 0 54px; border-top: 1px solid rgba(148,163,184,.15); border-bottom: 1px solid rgba(148,163,184,.15); }
        .showcase-route div { padding: 20px; border-right: 1px solid rgba(148,163,184,.12); color: #8ca0b9; font-size: 13px; }
        .showcase-route div:last-child { border-right: 0; }
        .showcase-route span { display: block; color: #38bdf8; font-size: 11px; letter-spacing: .16em; margin-bottom: 7px; }
        .gallery-feature-wrap { margin: 0 0 72px; }
        .gallery-grid { columns: 2; column-gap: 26px; margin-bottom: 72px; }
        .gallery-frame { break-inside: avoid; margin: 0 0 28px; border: 1px solid rgba(56,189,248,.16); border-radius: 18px; background: rgba(5,15,29,.72); overflow: hidden; box-shadow: 0 18px 44px rgba(0,0,0,.24); animation: showcaseRise .65s ease both; }
        .gallery-frame a { display: block; overflow: hidden; background: #050b14; }
        .gallery-frame img { display: block; width: 100%; height: auto; transition: transform .45s cubic-bezier(.22,.61,.36,1), filter .3s ease; }
        .gallery-frame:hover img { transform: scale(1.012); filter: brightness(1.05); }
        .gallery-frame-featured { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(280px, .7fr); align-items: center; margin-bottom: 70px; }
        .gallery-frame-featured img { width: 100%; height: auto; }
        .gallery-frame figcaption { padding: 22px 24px 25px; border-top: 1px solid rgba(148,163,184,.12); }
        .gallery-frame-featured figcaption { border-top: 0; border-left: 1px solid rgba(148,163,184,.12); padding: 34px; }
        .gallery-frame figcaption span { color: #38bdf8; font-size: 11px; letter-spacing: .16em; }
        .gallery-frame figcaption h3 { margin: 10px 0 8px; font-size: 22px; line-height: 1.35; color: #f1f6fc; }
        .gallery-frame-featured figcaption h3 { font-size: 30px; }
        .gallery-frame figcaption p { margin: 0; color: #879ab2; line-height: 1.7; font-size: 14px; }
        .gallery-closing { margin: 20px 0 34px; padding: 32px 34px; border-top: 1px solid rgba(56,189,248,.2); border-bottom: 1px solid rgba(56,189,248,.2); background: linear-gradient(90deg, rgba(56,189,248,.06), transparent); }
        .gallery-closing h3 { margin: 0 0 10px; font-size: 28px; }
        .gallery-closing p { margin: 0; max-width: 820px; color: #91a4ba; line-height: 1.75; }
        @keyframes showcaseRise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 900px) {
            .showcase-hero { min-height: auto; padding: 34px 26px; }
            .showcase-route { grid-template-columns: repeat(2, 1fr); }
            .gallery-grid { columns: 1; }
            .gallery-frame-featured { display: block; }
            .gallery-frame-featured img { max-height: none; }
            .gallery-frame-featured figcaption { border-left: 0; border-top: 1px solid rgba(148,163,184,.12); }
            .showcase-section-head { display: block; }
            .showcase-section-note { margin-top: 10px; text-align: left; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_default_showcase() -> None:
    _render_showcase_css()
    st.markdown(
        """
        <section class="showcase-hero">
          <div class="showcase-orbit"></div>
          <div class="showcase-eyebrow">PROJECT INTERFACE RECORD · 08 SCREENS</div>
          <h1>从一项政策，<br>看到完整证据链。</h1>
          <p>以下全部为项目真实运行界面。按政策输入、会话推演、机制解释、反事实评估与历史验证的顺序，呈现平台如何把政策文本转化为可观察、可比较、可归档的研究结果。</p>
          <div class="showcase-status"><span class="showcase-status-dot"></span><span>8 张项目实录</span><span>点击任意界面可查看原图</span></div>
        </section>
        <div class="showcase-route">
          <div><span>01</span>政策配置与编译</div>
          <div><span>02</span>会话推演与市场反馈</div>
          <div><span>03</span>反事实与机制评估</div>
          <div><span>04</span>历史验证与结果归档</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="showcase-section">
          <div class="showcase-section-head">
            <div><div class="showcase-index">01 · LIVE WORKSPACE</div><h2>市场正在回应政策</h2></div>
            <div class="showcase-section-note">先看运行中的主工作台，再沿证据链回看输入、机制与验证。</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(_gallery_figure(SHOWCASE_GALLERY[1], featured=True), unsafe_allow_html=True)
    st.markdown(
        """
        <section class="showcase-section">
          <div class="showcase-section-head">
            <div><div class="showcase-index">02 · EVIDENCE CHAIN</div><h2>从输入到评估</h2></div>
            <div class="showcase-section-note">界面按研究逻辑排序；纵向长图保留完整信息密度，宽图承担阶段转场。</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    first_grid = "".join(_gallery_figure(item) for item in [SHOWCASE_GALLERY[0], SHOWCASE_GALLERY[2], SHOWCASE_GALLERY[4], SHOWCASE_GALLERY[5]])
    st.markdown(f'<div class="gallery-grid">{first_grid}</div>', unsafe_allow_html=True)
    st.markdown(_gallery_figure(SHOWCASE_GALLERY[3], featured=True), unsafe_allow_html=True)
    st.markdown(
        """
        <section class="showcase-section">
          <div class="showcase-section-head">
            <div><div class="showcase-index">03 · HISTORICAL CLOSE</div><h2>回到真实市场检验</h2></div>
            <div class="showcase-section-note">历史走势对照与综合拟真评分构成最后一道证据边界。</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    closing_grid = "".join(_gallery_figure(item) for item in [SHOWCASE_GALLERY[6], SHOWCASE_GALLERY[7]])
    st.markdown(f'<div class="gallery-grid">{closing_grid}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="gallery-closing">
          <h3>一条完整的政策研究闭环</h3>
          <p>平台不是只给出一条曲线，而是保留政策输入、主体响应、市场反馈、反事实比较、历史验证与报告归档。每张界面都对应一段可检查的研究证据。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_a, action_b = st.columns(2)
    if action_a.button("进入政策实验", type="primary", width="stretch"):
        st.session_state.entry = "政策实验"
        st.rerun()
    if action_b.button("进入历史验证", width="stretch"):
        st.session_state.entry = "历史验证"
        st.rerun()
    st.caption("图片均为项目实际界面记录；点击图片可在新窗口查看原始尺寸。仅供教学科研与仿真评估，不构成投资建议。")
