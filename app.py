"""Civitas front-end in Streamlit: Policy Lab / History Replay / Advanced Analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, cast

import pandas as pd
import streamlit as st

from core.competition_demo import (
    LIVE_MODE,
    DEMO_MODE,
    COMPETITION_DEMO_MODE,
    REQUIRED_SCENARIOS,
    list_competition_scenarios,
    load_competition_scenario,
    bootstrap_competition_demo,
)
from core.competition_compliance import (
    COMPETITION_MODE_FLAG,
    competition_mode_enabled,
    write_competition_compliance_artifacts,
)
from core.model_router import ModelRouter
from core.runtime_mode import merge_mode_feature_flags, normalize_runtime_mode, resolve_runtime_mode_profile
from core.ui_text import display_runtime_mode, display_scenario_name, localize_dataframe_columns, zh_world_name
from ui.backtest_panel import render_backtest_panel
from ui.behavioral_diagnostics import render_behavioral_diagnostics
from ui.default_showcase import render_default_showcase
from ui.team_evolution import TEAM_PAGES, render_team_evolution, navigate
from ui.components.replay_scrubber import render_replay_scrubber
from ui.components.repro_meta import (
    build_experiment_registry_entry,
    build_reproducibility_meta,
    render_experiment_registry,
    render_reproducibility_panel,
    stable_payload_hash,
)
from ui.components.scenario_diff import render_scenario_diff
from ui.components.scorecard_panel import render_scorecard_panel
from ui.demo_wind_tunnel import render_demo_tab
from ui.history_replay import render_history_replay
from ui.policy_lab import _build_regulation_counterfactual_worlds, render_policy_lab
from ui.regulator_optimization import render_regulator_optimization
from ui.reporting import export_defense_bundle
from ui import dashboard as dashboard_ui


st.set_page_config(
    page_title="启元 Qiyuan-Evo | 可信自演进金融政策数字风洞",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


TEAM_ENTRY = "自演进驾驶舱"
SHOWCASE_ENTRY = "成果展示"
OVERVIEW_ENTRY = "系统总览"
ENTRY_POINTS = [TEAM_ENTRY, "协作过程", "演进实验", "三类任务", "展示页面"]
RESEARCH_ENTRIES = ["金融政策风洞", "历史验证", "技术与复现"]
DOMAIN_ENTRIES = [SHOWCASE_ENTRY, OVERVIEW_ENTRY, "政策实验", "研判分析"]
VALID_ENTRIES = ENTRY_POINTS + RESEARCH_ENTRIES + DOMAIN_ENTRIES
ENTRY_ALIASES = {
    "团队自演进": TEAM_ENTRY,
    "默认展示": SHOWCASE_ENTRY,
    "展示窗口": SHOWCASE_ENTRY,
    "默认展示窗口": SHOWCASE_ENTRY,
    "系统说明": OVERVIEW_ENTRY,
    "总览首页": OVERVIEW_ENTRY,
    "政策试验台": "政策实验",
    "历史政策回放": "历史验证",
    "历史Agent回放": "历史验证",
    "历史因子回测": "历史验证",
    "历史智能体回放": "历史验证",
    "新闻驱动历史回测": "历史验证",
    "历史回测": "历史验证",
    "政策A/B推演": "政策实验",
    "政策甲乙对照推演": "政策实验",
    "真实性报告": "研判分析",
    "高级分析": "研判分析",
    "监管优化": "研判分析",
}
ENTRY_DESCRIPTIONS = {
    TEAM_ENTRY: "运行真实团队协作、Prompt/工具/拓扑演进与优化前后对照。",
    SHOWCASE_ENTRY: "按研究链路浏览政策实验、反事实评估与历史验证的真实界面记录。",
    OVERVIEW_ENTRY: "用一屏建立平台定位、能力链路、工程特性与代表性结果。",
    "政策实验": "输入政策文本，查看结构化编译、多智能体响应与市场路径。",
    "历史验证": "基于真实历史窗口自动汇总政策与新闻，验证仿真与真实走势的一致性。",
    "研判分析": "聚合大模型证据、行为诊断、监管优化、研究验证与结果归档能力。",
}
ENTRY_PURPOSE = {
    TEAM_ENTRY: "展示 openJiuwen 编排、团队演进与可复现实验",
    SHOWCASE_ENTRY: "浏览项目实录与完整政策研究证据链",
    OVERVIEW_ENTRY: "快速建立平台认知并进入核心工作流",
    "政策实验": "进行政策仿真、影响评估与结果归档",
    "历史验证": "验证历史区间下的拟真效果与误差边界",
    "研判分析": "查看证据链、机制诊断、策略建议与研究验证",
}
THEME_PATH = Path("theme") / "competition_dark.css"
MATERIALS_ROOT = Path("outputs") / "competition_materials"


def _render_policy_lab_compatible(presentation_mode: str) -> None:
    """Call policy lab in a backward-compatible way across mixed deployments."""
    try:
        render_policy_lab(presentation_mode=presentation_mode)
    except TypeError:

        render_policy_lab()


def _normalize_entry(entry: str) -> str:
    key = str(entry or "")
    return ENTRY_ALIASES.get(key, key)


def _feature_flag_enabled(flag_name: str, *, default: bool = False) -> bool:
    flags = st.session_state.get("feature_flags", {})
    if isinstance(flags, MutableMapping):
        if flag_name in flags:
            return bool(flags[flag_name])
    return bool(default)


def _sync_runtime_mode_profile() -> None:
    mode = normalize_runtime_mode(str(st.session_state.get("simulation_mode", "SMART")))
    st.session_state.simulation_mode = mode
    profile = resolve_runtime_mode_profile(mode)
    st.session_state.runtime_mode_profile = profile.to_dict()
    flags = st.session_state.get("feature_flags", {})
    st.session_state.feature_flags = merge_mode_feature_flags(
        mode,
        flags if isinstance(flags, MutableMapping) else {},
    )


def _load_theme() -> None:
    if THEME_PATH.exists():
        css = THEME_PATH.read_text(encoding="utf-8")
    else:
        css = """
        .stApp { background: #050b14; color: #e2e8f0; }
        .kpi-card { background: rgba(10, 25, 49, 0.65); border: 1px solid #1f365c; border-radius: 12px; padding: 16px; backdrop-filter: blur(8px); }
        .kpi-title { font-size: 13px; color: #8aa0c2; }
        .kpi-value { font-size: 28px; font-weight: 700; color: #ffffff; }
        .kpi-note { font-size: 12px; color: #7995bc; }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _runtime_router_summary() -> Dict[str, Any]:
    summary = ModelRouter.runtime_observability_summary()
    if isinstance(summary, dict):
        return summary
    return {}


def _load_data_flywheel_summary(max_records: int = 500) -> Dict[str, Any]:
    seed_path = Path("data") / "seed_events.jsonl"
    if not seed_path.exists():
        return {"available": False, "reason": f"{seed_path} not found"}

    total = 0
    impact_counter: Dict[str, int] = {}
    topic_counter: Dict[str, int] = {}
    source_counter: Dict[str, int] = {}
    recent: List[Dict[str, Any]] = []
    for raw in seed_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        total += 1
        impact = str(item.get("impact_level", "unknown"))
        impact_counter[impact] = impact_counter.get(impact, 0) + 1
        source = str(item.get("source", item.get("source_name", "unknown")))
        source_counter[source] = source_counter.get(source, 0) + 1
        topic = str(item.get("topic", item.get("event_type", "unknown")))
        topic_counter[topic] = topic_counter.get(topic, 0) + 1
        if len(recent) < 5:
            recent.append(
                {
                    "time": item.get("published_at", item.get("created_at", "")),
                    "topic": topic,
                    "impact": impact,
                    "source": source,
                }
            )
        if total >= max_records:
            break

    top_topics = sorted(topic_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_sources = sorted(source_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "available": True,
        "total_events": total,
        "impact_counter": impact_counter,
        "top_topics": top_topics,
        "top_sources": top_sources,
        "recent_events": recent,
    }


def _render_value_bridge_tab() -> None:
    st.markdown("### 数据飞轮与反事实价值卡")
    flywheel = _load_data_flywheel_summary()
    if not flywheel.get("available", False):
        st.info(f"数据飞轮暂不可用：{flywheel.get('reason', 'unknown')}")
    else:
        a, b, c = st.columns(3)
        a.metric("已采集事件", int(flywheel.get("total_events", 0)))
        high_impact = int(dict(flywheel.get("impact_counter", {})).get("high", 0))
        b.metric("高影响事件", high_impact)
        c.metric("主要来源数", len(list(flywheel.get("top_sources", []))))
        st.markdown("#### 事件来源与主题摘要")
        left, right = st.columns(2)
        with left:
            st.dataframe(
                localize_dataframe_columns(pd.DataFrame(flywheel.get("top_sources", []), columns=["source", "count"])),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.dataframe(
                localize_dataframe_columns(pd.DataFrame(flywheel.get("top_topics", []), columns=["topic", "count"])),
                hide_index=True,
                use_container_width=True,
            )

    _ensure_demo_loaded()
    scenario = st.session_state.get("demo_scenario")
    if scenario is None:
        st.warning("当前未加载分析场景，无法生成反事实对照。")
        return
    metrics = scenario.metrics if hasattr(scenario, "metrics") else pd.DataFrame()
    if metrics.empty:
        st.warning("当前场景缺少指标数据。")
        return

    try:
        counterfactual = _build_regulation_counterfactual_worlds(metrics, intensity=1.0)
    except Exception as exc:
        st.warning(f"反事实计算失败：{exc}")
        return

    st.markdown("#### 监管反事实甲乙对照")
    worlds = dict(counterfactual.get("worlds", {}) or {})
    scorecards = dict(counterfactual.get("scorecards", {}) or {})
    rec = str(counterfactual.get("recommended_timing", ""))
    if rec:
        st.caption(f"推荐干预时机：{zh_world_name(rec)}")
    cards_df = pd.DataFrame(
        [
            {"world": key, **(value if isinstance(value, dict) else {})}
            for key, value in scorecards.items()
        ]
    )
    if not cards_df.empty:
        st.dataframe(localize_dataframe_columns(cards_df), use_container_width=True, hide_index=True)
    world_keys = list(worlds.keys())
    if len(world_keys) >= 2:
        base = pd.DataFrame(worlds[world_keys[0]])
        alt = pd.DataFrame(worlds[world_keys[1]])
        if not base.empty and not alt.empty and "step" in base.columns and "close" in base.columns:
            merged = base[["step", "close"]].merge(
                alt[["step", "close"]],
                on="step",
                suffixes=("_base", "_alt"),
                how="inner",
            )
            if not merged.empty:
                merged["delta_close"] = merged["close_alt"] - merged["close_base"]
                st.line_chart(merged.set_index("step")[["delta_close"]], use_container_width=True)


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "entry": st.query_params.get("page", TEAM_ENTRY),
        "controller": None,
        "runtime_mode": LIVE_MODE,
        "competition_mode": "",
        "market_history": [],
        "csad_history": [],
        "demo_scenario_name": REQUIRED_SCENARIOS[0],
        "demo_scenario": None,
        "demo_step_cursor": 0,
        "demo_narration_cursor": 0,
        "demo_last_narration": None,
        "demo_last_step": 0,
        "demo_last_panic_level": 0.0,
        "demo_autoplay": False,
        "is_running": False,
        "last_demo_figures": {},
        "materials_last_export": None,
        "policy_lab_result": None,
        "history_replay_result": None,
        "simulation_mode": "SMART",
        "runtime_mode_profile": resolve_runtime_mode_profile("SMART").to_dict(),
        "feature_flags": {
            COMPETITION_MODE_FLAG: True,
            "competition_safe_mode": True,
            "market_pipeline_v2": True,
            "regulator_real_env_v1": True,
            "regulator_toy_fallback_v1": True,
        },
        "competition_mode_requested": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.entry = _normalize_entry(str(st.session_state.get("entry", TEAM_ENTRY)))
    if st.session_state.entry not in VALID_ENTRIES:
        st.session_state.entry = TEAM_ENTRY
    _sync_runtime_mode_profile()


def _render_top_entry_selector() -> None:
    st.html('<div class="ev-masthead"><div><b>启元</b><span>QIYUAN–EVO</span></div>'
            '<span>基于 openJiuwen 的自演进多智能体系统</span></div>')
    columns = st.columns([1, 1, 1, 1, 1, .8], gap="small")
    for column, entry in zip(columns, ENTRY_POINTS):
        display = {"自演进驾驶舱":"演示总览", "协作过程":"团队协作", "演进实验":"自演进", "三类任务":"三类实验", "展示页面":"运行截图"}.get(entry, entry)
        column.button(display, key="top_entry_" + entry, width="stretch",
                      type="primary" if st.session_state.entry == entry else "secondary",
                      on_click=navigate, args=(entry,))
    with columns[-1].popover("研究工具", width="stretch"):
        for entry in RESEARCH_ENTRIES:
            st.button(entry, key="top_entry_" + entry, width="stretch", on_click=navigate, args=(entry,))



def _ensure_demo_loaded() -> None:
    if st.session_state.get("demo_scenario") is None:
        name = st.session_state.get("demo_scenario_name", REQUIRED_SCENARIOS[0])
        try:
            state = cast(MutableMapping[str, Any], st.session_state)
            bootstrap_competition_demo(state, name, auto_play=False)
            st.session_state.runtime_mode = DEMO_MODE
            st.session_state.competition_mode = COMPETITION_DEMO_MODE
        except Exception as exc:
            st.session_state.demo_scenario = None
            st.error(f"场景加载失败：{exc}")


def _material_summary_from_metrics(metrics: pd.DataFrame) -> Dict[str, float]:
    if metrics.empty:
        return {"return_pct": 0.0, "volatility": 0.0, "panic_max": 0.0, "herding_avg": 0.0}
    ret = (float(metrics.iloc[-1]["close"]) - float(metrics.iloc[0]["close"])) / max(1.0, float(metrics.iloc[0]["close"]))
    vol = float(metrics["close"].pct_change().std()) if len(metrics) > 1 else 0.0
    panic_max = float(metrics["panic_level"].max())
    herding_avg = float(metrics["csad"].mean())
    return {
        "return_pct": ret,
        "volatility": vol,
        "panic_max": panic_max,
        "herding_avg": herding_avg,
    }


def _competition_mode_active() -> bool:
    flags = st.session_state.get("feature_flags", {})
    requested = bool(st.session_state.get("competition_mode_requested", True))
    if not isinstance(flags, MutableMapping):
        flags = {}
    return competition_mode_enabled(feature_flags=flags, requested_mode=requested)


def _generate_competition_materials() -> Dict[str, Path]:
    scenario = st.session_state.get("demo_scenario")
    if scenario is None:
        scenario_name = st.session_state.get("demo_scenario_name", REQUIRED_SCENARIOS[0])
        scenario = load_competition_scenario(scenario_name)

    MATERIALS_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = MATERIALS_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = scenario.metrics
    stat = _material_summary_from_metrics(metrics)
    runtime_summary = _runtime_router_summary()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    competition_summary = f"""# experiment_summary

生成时间：{now}
分析场景：{display_scenario_name(scenario.name)}

## 关键指标
- 区间收益率：{stat['return_pct']:.2%}
- 波动率：{stat['volatility']:.2%}
- 最大风险热度：{stat['panic_max']:.2f}
- 平均羊群度（CSAD）：{stat['herding_avg']:.3f}

## 运行模式证据
- 在线调用成功次数：{int(runtime_summary.get('online_success_total', 0))}
- 自动回退次数：{int(runtime_summary.get('fallback_total', 0))}
- 在线成功率：{float(runtime_summary.get('online_success_rate', 0.0)):.1%}
- 最近回退原因：{str(runtime_summary.get('last_fallback_reason', '') or '-')}

## 结论摘要
1. 系统能够围绕政策冲击、市场反应和风险扩散形成闭环推演。
2. 多智能体行为与市场微观结构结果可以同步展示，便于机制解释与方案汇报。
3. 监管动作、甲乙对照和归档材料可直接服务复盘、留痕与策略研判。
"""

    design_outline = """# design_outline

## 功能模块
- 政策实验
- 历史验证
- 研究验证
- 反事实甲乙对照推演
- 监管优化

## 技术实现
- 使用 Streamlit 构建工程化分析前端与交互流程。
- 通过多智能体仿真引擎驱动市场演化与行为决策。
- 结合行为金融指标、历史验证分析和结果归档形成完整闭环。
"""

    analysis_playbook = f"""# analysis_playbook

1. 在系统总览页说明平台定位与分析对象
2. 展示系统工作流与核心能力链
3. 进入政策实验，查看政策输入、结构化编译与多智能体推演
4. 进入历史验证，比较仿真走势与真实市场路径
5. 进入行为与风险诊断，说明群体行为与风险扩散机制
6. 展示监管优化、甲乙差分与帕累托权衡
7. 沉淀实验报告、分析摘要、证据链与结果归档
8. 回到总览页，总结平台从政策理解到策略建议的闭环价值
"""

    figures_index = {
        "generated_at": now,
        "scenario": scenario.name,
        "scenario_display": display_scenario_name(scenario.name),
        "figures": st.session_state.get("last_demo_figures", {}),
    }

    file_map = {
        "experiment_summary.md": MATERIALS_ROOT / "experiment_summary.md",
        "system_design_outline.md": MATERIALS_ROOT / "system_design_outline.md",
        "analysis_playbook.md": MATERIALS_ROOT / "analysis_playbook.md",
        "figures/index.json": figures_dir / "index.json",
        "runtime_mode_evidence.json": MATERIALS_ROOT / "runtime_mode_evidence.json",
    }
    file_map["experiment_summary.md"].write_text(competition_summary, encoding="utf-8")
    file_map["system_design_outline.md"].write_text(design_outline, encoding="utf-8")
    file_map["analysis_playbook.md"].write_text(analysis_playbook, encoding="utf-8")
    file_map["figures/index.json"].write_text(json.dumps(figures_index, ensure_ascii=False, indent=2), encoding="utf-8")
    file_map["runtime_mode_evidence.json"].write_text(
        json.dumps(runtime_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if _competition_mode_active():
        feature_flags = dict(st.session_state.get("feature_flags", {}))
        compliance = write_competition_compliance_artifacts(
            root_dir=MATERIALS_ROOT,
            project_root=Path.cwd(),
            feature_flags=feature_flags,
            materials_context={
                "scenario": scenario.name,
                "generated_at": now,
            },
            app_flow=["政策实验", "历史验证", "研究验证", "反事实甲乙对照推演", "监管优化"],
        )
        realism_payload = {
            "title": "真实性评估摘要",
            "path_fit": {
                "enabled": True,
                "score": round(max(0.0, 1.0 - stat["volatility"]), 4),
                "price_correlation": round(max(0.0, 1.0 - stat["panic_max"] * 0.2), 4),
                "volatility_correlation": round(max(0.0, 1.0 - stat["volatility"]), 4),
                "price_rmse": round(abs(stat["return_pct"]), 4),
                "price_mae": round(abs(stat["return_pct"]) * 0.8, 4),
            },
            "microstructure_fit": {"enabled": True, "score": round(max(0.0, 1.0 - stat["panic_max"] * 0.2), 4)},
            "behavioral_fit": {"enabled": True, "score": round(max(0.0, 1.0 - stat["herding_avg"]), 4)},
            "reproducibility": {"seed": 42, "config_hash": json.dumps(feature_flags, sort_keys=True, ensure_ascii=False)},
            "snapshot_info": {"scenario": scenario.name},
            "charts": [],
        }
        agent_taxonomy_markdown = "\n".join(
            [
                "# agent_taxonomy",
                "",
                "- 散户资金：情绪与政策敏感的交易主体",
                "- 公募基金：受基准约束的机构配置资金",
                "- 量化择时资金：跟随趋势与波动状态切换的交易主体",
                "- 做市资金：受流动性与库存约束的交易主体",
                "- 稳定资金：承担市场稳定和承接功能的资金主体",
            ]
        )
        policy_causal_chain = {
            "policy": scenario.name,
            "path": ["policy_input", "expectation_shift", "order_flow_split", "lob_matching", "market_metrics"],
            "summary": {
                "return_pct": float(stat["return_pct"]),
                "volatility": float(stat["volatility"]),
                "panic_max": float(stat["panic_max"]),
                "herding_avg": float(stat["herding_avg"]),
            },
        }
        realism_metrics_csv = "\n".join(
            [
                "指标,数值",
                f"累计收益率,{float(stat['return_pct'])}",
                f"波动率,{float(stat['volatility'])}",
                f"最大恐慌度,{float(stat['panic_max'])}",
                f"平均羊群度,{float(stat['herding_avg'])}",
            ]
        )
        competition_snapshot = {
            "scenario": scenario.name,
            "generated_at": now,
            "seed": 42,
            "config_hash": json.dumps(feature_flags, sort_keys=True, ensure_ascii=False),
            "dataset_snapshot_id": f"demo:{scenario.name}",
            "runtime_mode_evidence": runtime_summary,
        }
        bundle = export_defense_bundle(
            root_dir=MATERIALS_ROOT,
            bundle_name=f"{scenario.name}_analysis_bundle",
            design_chapter_markdown=design_outline,
            realism_payload=realism_payload,
            policy_ab_markdown="# 反事实与甲乙对照说明\n\n- 系统会导出精简版对照材料，用于方案汇报、复盘说明与结果归档。",
            architecture_graph={"nodes": [], "edges": []},
            causal_chain_graph={"nodes": [], "edges": []},
            defense_outline_markdown=analysis_playbook,
            feature_flags=feature_flags,
            compliance_artifacts={
                "manifest": compliance["manifest"],
                "files": {
                    "ai_tool_usage_manifest": compliance["manifest_path"],
                    "technical_route_template": compliance["technical_route_template_path"],
                },
            },
            agent_taxonomy_markdown=agent_taxonomy_markdown,
            policy_causal_chain=policy_causal_chain,
            realism_metrics_csv=realism_metrics_csv,
            competition_snapshot=competition_snapshot,
        )
        file_map["bundle_manifest.json"] = bundle["manifest_path"]
        file_map["ai_tool_usage_manifest.json"] = compliance["manifest_path"]
        file_map["technical_route_template.md"] = compliance["technical_route_template_path"]

    return file_map


def _render_sidebar_global() -> None:
    with st.sidebar:
        st.markdown("### 研究工具")
        with st.expander("金融分析工具"):
            for entry in DOMAIN_ENTRIES:
                st.button(entry, key=f"entry_{entry}", width="stretch", on_click=navigate, args=(entry,))
        if st.session_state.entry in TEAM_PAGES or st.session_state.entry in {"金融政策风洞", "历史验证"}:
            return
        st.markdown("### 推演设置")
        sim_mode_display = {"SMART": "智能模式（仅智谱 GLM）", "DEEP": "高级模式（DeepSeek + 智谱混用）"}
        selected_mode_key = st.radio(
            "选择大模型调度策略",
            options=["SMART", "DEEP"],
            index=0 if st.session_state.simulation_mode == "SMART" else 1,
            format_func=lambda x: sim_mode_display.get(x, x),
            label_visibility="collapsed",
            help="智能模式只接入智谱模型；高级模式沿用 DeepSeek 与智谱混合调度。"
        )
        if selected_mode_key != st.session_state.simulation_mode:
            st.session_state.simulation_mode = selected_mode_key
            _sync_runtime_mode_profile()
            st.toast(f"已切换至 {sim_mode_display[selected_mode_key]}")
        else:
            _sync_runtime_mode_profile()

        runtime_profile = st.session_state.get("runtime_mode_profile", {})
        if isinstance(runtime_profile, MutableMapping):
            summary = str(runtime_profile.get("summary", ""))
            pause_seconds = float(runtime_profile.get("pause_for_llm_seconds", 0.0) or 0.0)
            st.caption(f"模式摘要：{summary}")

        runtime_summary = _runtime_router_summary()
        if runtime_summary:
            online_success = int(runtime_summary.get("online_success_total", 0))
            fallback_total = int(runtime_summary.get("fallback_total", 0))
            success_rate = float(runtime_summary.get("online_success_rate", 0.0))
            status_map = {
                "online_ok": "🟢 在线稳定",
                "degraded_with_fallback": "🟡 降级兜底",
                "offline_fallback": "🔴 离线兜底",
            }
            status = status_map.get(str(runtime_summary.get("status", "")), "⚪ 状态未知")
            st.markdown(f"<div style='font-size: 12px; margin-top: 10px; color: #8aa0c2;'>{status} | 在线成功={online_success} | 降级={fallback_total} | 成功率={success_rate:.0%}</div>", unsafe_allow_html=True)

        st.markdown("---")

        if st.session_state.entry in {OVERVIEW_ENTRY, "研判分析"}:
            scenarios = list_competition_scenarios()
            if scenarios:
                if st.session_state.demo_scenario_name not in scenarios:
                    st.session_state.demo_scenario_name = scenarios[0]
                st.session_state.demo_scenario_name = st.selectbox(
                    "研判分析默认场景",
                    options=scenarios,
                    index=scenarios.index(st.session_state.demo_scenario_name),
                    format_func=display_scenario_name,
                )
            else:
                st.warning("未发现可用分析场景，请检查 demo_scenarios 内容完整性。")

            if st.button("导出实验归档", width="stretch"):
                _ensure_demo_loaded()
                if st.session_state.get("demo_scenario") is None:
                    st.error("当前没有可用场景，无法生成归档材料。")
                else:
                    try:
                        file_map = _generate_competition_materials()
                        st.session_state.materials_last_export = {k: str(v) for k, v in file_map.items()}
                        st.success("实验归档已生成。")
                    except Exception as exc:
                        st.error(f"实验归档生成失败：{exc}")

            if st.session_state.materials_last_export:
                st.caption(f"最近已导出 {len(st.session_state.materials_last_export)} 份实验归档。")
                with st.expander("查看导出文件清单", expanded=False):
                    for name, path in st.session_state.materials_last_export.items():
                        st.markdown(f"- `{name}`")


def _render_ai_decision_tab() -> None:
    st.markdown("### 智能决策解读")
    _ensure_demo_loaded()
    scenario = st.session_state.demo_scenario
    metrics = scenario.metrics
    step = int(st.session_state.get("demo_last_step", 0))
    if step <= 0:
        step = int(metrics.iloc[min(3, len(metrics) - 1)]["step"])

    kpi = dashboard_ui.build_kpi_snapshot(metrics, step, regulation_hint="专家复盘")
    dashboard_ui.render_kpi_cards(kpi)
    dashboard_ui.render_decision_evidence_flow(scenario.narration, scenario.analyst_manager_output)

    upto = metrics[metrics["step"] <= step]
    latest = upto.tail(1).iloc[0] if not upto.empty else metrics.tail(1).iloc[0]
    chain_payload = {
        "policy": f"专家复盘 第 {int(latest['step'])} 步",
        "macro_variables": {
            "inflation": 0.02 + float(latest["panic_level"]) * 0.01,
            "unemployment": 0.05 + float(latest["panic_level"]) * 0.03,
            "wage_growth": 0.03 - float(latest["panic_level"]) * 0.01,
            "credit_spread": 0.015 + float(latest["panic_level"]) * 0.02,
            "liquidity_index": max(0.2, 1.15 - float(latest["panic_level"])),
            "policy_rate": 0.022,
            "fiscal_stimulus": 0.02,
            "sentiment_index": max(0.0, min(1.0, 0.65 - float(latest["panic_level"]) * 0.55)),
        },
        "social_sentiment": {"mean": 0.5 - float(latest["panic_level"])},
        "industry_agent": {
            "avg_household_risk": 0.50 - float(latest["panic_level"]) * 0.3,
            "avg_firm_hiring": 0.15 - float(latest["panic_level"]) * 0.5,
        },
        "market_microstructure": {
            "buy_volume": float(latest["volume"]) * (1.0 - float(latest["panic_level"])),
            "sell_volume": float(latest["volume"]) * (0.45 + float(latest["panic_level"])),
            "trade_count": int(max(1.0, float(latest["volume"]) / 800)),
            "matching_mode": "expert_replay",
        },
    }
    c1, c2 = st.columns(2)
    with c1:
        dashboard_ui.render_social_network_heatmap(upto, key_prefix="advanced_ai")
        dashboard_ui.render_lob_depth_animation(upto, key_prefix="advanced_ai")
    with c2:
        dashboard_ui.render_policy_transmission_chain(chain_payload, key_prefix="advanced_ai")
        dashboard_ui.render_risk_event_timeline(upto, key_prefix="advanced_ai")

    role_order_flows = {
        "retail_general": float(latest["volume"]) * (0.32 - 0.22 * float(latest["panic_level"])),
        "mutual_fund": float(latest["volume"]) * (0.22 - 0.12 * float(latest["panic_level"])),
        "quant_timing": float(latest["volume"]) * (0.10 + 0.18 * float(latest["panic_level"])),
        "market_maker": float(latest["volume"]) * (-0.08 + 0.04 * float(latest["panic_level"])),
    }
    spread = max(0.001, float(latest["panic_level"]) * 0.02)
    buy_vol = float(chain_payload["market_microstructure"]["buy_volume"])
    sell_vol = float(chain_payload["market_microstructure"]["sell_volume"])
    open_px = float(latest.get("open", latest["close"]))
    csad_val = float(latest.get("csad", latest.get("panic_level", 0.0)))
    micro_metrics = {
        "spread": spread,
        "spread_pct": spread / max(1.0, float(latest["close"])),
        "depth_imbalance": (buy_vol - sell_vol) / max(1.0, buy_vol + sell_vol),
        "impact": abs(float(latest["close"]) - open_px) / max(1.0, float(latest["close"])),
        "herding_proxy": abs(csad_val),
    }
    dashboard_ui.render_orderflow_microstructure_panel(
        role_order_flows=role_order_flows,
        microstructure_metrics=micro_metrics,
        key_prefix="advanced_ai",
    )


def _build_overview_chain_payload(metrics: pd.DataFrame) -> Dict[str, Any]:
    latest = metrics.tail(1).iloc[0]
    panic = float(latest.get("panic_level", 0.0))
    return {
        "policy": "默认分析场景传导链",
        "macro_variables": {
            "inflation": 0.02 + panic * 0.01,
            "unemployment": 0.05 + panic * 0.03,
            "credit_spread": 0.015 + panic * 0.02,
            "liquidity_index": max(0.2, 1.15 - panic),
            "sentiment_index": max(0.0, min(1.0, 0.68 - panic * 0.52)),
        },
        "social_sentiment": {"mean": 0.5 - panic},
        "industry_agent": {
            "avg_household_risk": 0.50 - panic * 0.3,
            "avg_firm_hiring": 0.15 - panic * 0.5,
        },
        "market_microstructure": {
            "buy_volume": float(latest["volume"]) * (1.0 - panic),
            "sell_volume": float(latest["volume"]) * (0.45 + panic),
            "trade_count": int(max(1.0, float(latest["volume"]) / 800)),
            "matching_mode": "overview_replay",
        },
    }


def _render_research_workbench_tab() -> None:
    st.markdown("### 研究工作台")
    _ensure_demo_loaded()
    scenario = st.session_state.get("demo_scenario")
    metrics = scenario.metrics.copy() if scenario is not None and hasattr(scenario, "metrics") else pd.DataFrame()
    if metrics.empty:
        dashboard_ui.render_empty_market_board(key_prefix="research_empty")
        return

    benchmark = st.selectbox(
        "基准指数",
        options=["sh000001", "sh000300", "sz399001", "sz399006"],
        index=0,
        key="research_workbench_benchmark",
    )
    if "sentiment_index" not in metrics.columns and "panic_level" in metrics.columns:
        metrics["sentiment_index"] = 1.0 - pd.to_numeric(metrics["panic_level"], errors="coerce").fillna(0.0)
    if "spread" not in metrics.columns:
        metrics["spread"] = pd.to_numeric(metrics["close"], errors="coerce").pct_change().abs().fillna(0.001)
    if "depth_imbalance" not in metrics.columns:
        panic_series = (
            pd.to_numeric(metrics["panic_level"], errors="coerce").fillna(0.0)
            if "panic_level" in metrics.columns
            else pd.Series(0.0, index=metrics.index)
        )
        metrics["depth_imbalance"] = (0.5 - panic_series).clip(-1.0, 1.0)
    tape = dashboard_ui.build_synthetic_trade_tape_from_market_frame(metrics, symbol=benchmark)
    events = [
        {"event_type": "policy", "title": "政策冲击", "effective_day": 1, "strength": 1.0, "source": "demo_scenario"},
        {"event_type": "major_news", "title": "新闻扰动", "effective_day": max(2, int(len(metrics) * 0.35)), "strength": 0.7, "source": "demo_scenario"},
        {"event_type": "regulatory_action", "title": "监管观察", "effective_day": max(3, int(len(metrics) * 0.70)), "strength": 0.8, "source": "demo_scenario"},
    ]
    fig, chart_frame = dashboard_ui.render_trade_tape_kline_workbench(
        tape,
        symbol=benchmark,
        benchmark_symbol=benchmark,
        market_frame=metrics,
        real_index_frame=pd.DataFrame(),
        events=events,
        key_prefix="research_workbench",
    )
    tabs = st.tabs(["场景对比", "回放", "评估卡", "可复现信息"])
    with tabs[0]:
        counterfactual = _build_regulation_counterfactual_worlds(metrics, intensity=1.0)
        worlds = dict(counterfactual.get("worlds", {}) or {})
        render_scenario_diff(
            {
                "baseline": pd.DataFrame(worlds.get("no_intervention", []) or []),
                "policy_a": pd.DataFrame(worlds.get("early_intervention", []) or []),
                "policy_b": pd.DataFrame(worlds.get("late_intervention", []) or []),
                "optimized_policy": pd.DataFrame(worlds.get(counterfactual.get("recommended_timing", "early_intervention"), []) or []),
            },
            base_frame=chart_frame,
            key_prefix="research_workbench_diff",
        )
    with tabs[1]:
        render_replay_scrubber(
            chart_frame,
            events=events,
            trade_tape=[item.to_dict() for item in tape],
            key_prefix="research_workbench_replay",
        )
    with tabs[2]:
        render_scorecard_panel(st.session_state.get("policy_lab_session", {}).get("scorecard") if isinstance(st.session_state.get("policy_lab_session"), dict) else None, key_prefix="research_workbench_scorecard")
    with tabs[3]:
        config_hash = stable_payload_hash({"scenario": getattr(scenario, "name", "demo"), "benchmark": benchmark})
        registry = build_experiment_registry_entry(
            scenario_name=display_scenario_name(getattr(scenario, "name", "demo")),
            config_hash=config_hash,
            data_snapshot_id=chart_frame.attrs.get("trade_tape_hash", "demo_synthetic_tape"),
            seed=42,
            selected_benchmark=benchmark,
            status="ready",
        )
        render_experiment_registry(registry, key_prefix="research_workbench_registry")
        render_reproducibility_panel(
            build_reproducibility_meta(
                data_snapshot_hash=str(registry["data_snapshot_id"]),
                config_hash=config_hash,
                random_seed=42,
                llm_provider_chain=["GLM-4-flashx", "DeepSeek", "offline_fallback"],
                calibration_parameter_set_id="demo_default_calibration_v1",
                extra={"figure_trace_count": len(fig.data)},
            ),
            key_prefix="research_workbench_repro",
        )


def _render_overview_home() -> None:
    st.markdown(
        """
        <div class="hero-panel" style="margin-top: 10px;">
            <div class="hero-kicker">系统总览</div>
            <h1 style="font-size: 30px; margin-bottom: 14px;">数治观澜：面向金融政策预评估的风洞推演沙箱</h1>
            <p style="font-size: 16px; max-width: 920px;">
                数治观澜面向的核心问题是：一项经济或金融政策在正式出台之前，能否先在数字环境中完成理解、推演、验证和优化，
                提前预判它可能引发的市场行为、风险扩散与监管效果。它不是单纯输出结论的问答系统，也不是静态展示平台，而是一套把
                自然语言政策输入、结构化编译、多智能体市场仿真、历史验证、因子研究、行为风险诊断、监管优化和结果归档串联起来的工程化闭环。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _ensure_demo_loaded()
    scenario = st.session_state.get("demo_scenario")
    metrics = scenario.metrics if scenario is not None and hasattr(scenario, "metrics") else pd.DataFrame()
    stat = _material_summary_from_metrics(metrics) if not metrics.empty else {"return_pct": 0.0, "volatility": 0.0, "panic_max": 0.0, "herding_avg": 0.0}
    runtime_summary = _runtime_router_summary()
    flywheel = _load_data_flywheel_summary()

    top_stats = st.columns(4)
    top_stats[0].metric("默认分析场景", display_scenario_name(getattr(scenario, "name", REQUIRED_SCENARIOS[0])))
    top_stats[1].metric("区间收益表现", f"{stat['return_pct']:.2%}")
    top_stats[2].metric("峰值风险热度", f"{stat['panic_max']:.2f}")
    top_stats[3].metric("在线稳定率", f"{float(runtime_summary.get('online_success_rate', 0.0)):.0%}")

    story_cols = st.columns(4)
    story_cards = [
        (
            "政策理解",
            "把自然语言政策解析为可推演、可验证、可归档的结构化政策冲击。",
            ["自然语言解析", "结构化编译", "传导渠道识别"],
        ),
        (
            "市场推演",
            "用多智能体行为、订单流和撮合机制生成市场路径与风险扩散过程。",
            ["主体分歧", "订单流", "价格路径"],
        ),
        (
            "历史验证",
            "对照真实指数窗口、重大新闻和方向命中，检验仿真逻辑的一致性。",
            ["真实走势", "新闻回放", "拟真评估"],
        ),
        (
            "监管优化",
            "比较反事实路径、帕累托权衡和候选方案，为政策节奏提供证据。",
            ["反事实对照", "帕累托权衡", "方案推荐"],
        ),
    ]
    for col, (title, summary, bullets) in zip(story_cols, story_cards):
        bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
        col.markdown(
            f"""
            <div class="ops-card">
              <div class="ops-card-title">{title}</div>
              <div class="ops-card-summary">{summary}</div>
              <ul class="ops-card-list">{bullet_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 大模型与智能底座")
    model_cols = st.columns(3)
    model_cards = [
        (
            "GLM-4-flashx 接入",
            "当前演示接入智谱 GLM-4-flashx，承担政策文本理解、结构化信息抽取、部分智能体认知推理以及解释性内容生成。",
            ["政策语义理解", "结构化编译辅助", "智能体认知分化", "解释生成"],
        ),
        (
            "多智能体市场仿真",
            "不同角色市场主体会基于风险偏好、资金约束和行为偏差做差异化响应，再映射到订单流、板块轮动和价格路径上。",
            ["散户与机构", "情绪与预期", "订单流与撮合", "风险热度"],
        ),
        (
            "结果沉淀与复现",
            "系统支持把实验结论、关键指标、候选方案和推荐建议沉淀为报告、图表索引和归档材料，便于复盘、汇报和留痕。",
            ["实验报告", "图表导出", "证据链", "可复现信息"],
        ),
    ]
    for col, (title, summary, bullets) in zip(model_cols, model_cards):
        bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
        col.markdown(
            f"""
            <div class="ops-card">
              <div class="ops-card-title">{title}</div>
              <div class="ops-card-summary">{summary}</div>
              <ul class="ops-card-list">{bullet_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 系统工作流")
    flow_cards = [
        ("1. 政策输入", "输入自然语言政策或事件描述"),
        ("2. 语义编译", "识别类型、强度、传导渠道与影响对象"),
        ("3. 多智能体推演", "形成角色分歧、行为意图与市场反馈"),
        ("4. 市场撮合", "生成价格、成交量与微观结构结果"),
        ("5. 历史验证", "校验走势一致性、响应时点与误差边界"),
        ("6. 监管优化", "比较干预成本、稳定性和流动性权衡"),
        ("7. 证据归档", "沉淀报告、图表、证据链与可复现信息"),
    ]
    for row_cards in (flow_cards[:4], flow_cards[4:]):
        flow_cols = st.columns(len(row_cards))
        for col, (title, summary) in zip(flow_cols, row_cards):
            col.markdown(
                f"""
                <div class="flow-card">
                  <div class="flow-step-index">{title}</div>
                  <div class="flow-step-label">{summary}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("### 当前可运行模块")
    route_cols = st.columns(4)
    route_cards = [
        ("系统总览", "建立平台定位、能力边界、工程特性与代表性结果。"),
        ("政策实验", "输入政策文本或选择模板，查看结构化编译与推演结果。"),
        ("历史验证", "用真实历史窗口验证仿真与现实走势的一致性。"),
        ("研判分析", "查看证据链、风险机制、监管优化与研究验证内容。"),
    ]
    for col, (title, summary) in zip(route_cols, route_cards):
        col.markdown(
            f"""
            <div class="story-card">
              <div class="story-card-title">{title}</div>
              <div class="story-card-summary">{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not metrics.empty:
        st.markdown("### 代表性结果卡片")
        snapshot = dashboard_ui.build_kpi_snapshot(metrics, int(metrics["step"].max()), regulation_hint="已具备干预评估条件")
        dashboard_ui.render_kpi_cards(snapshot)

        left, right = st.columns([1.5, 1.0])
        with left:
            dashboard_ui.render_market_overview(metrics, upto_step=None, key_prefix="overview", title="默认分析场景市场结果总览")
        with right:
            package_summary = [
                f"区间收益率：{stat['return_pct']:.2%}",
                f"波动率：{stat['volatility']:.2%}",
                f"羊群度均值：{stat['herding_avg']:.3f}",
                f"高影响事件：{int(dict(flywheel.get('impact_counter', {})).get('high', 0)) if flywheel.get('available') else 0}",
                f"自动回退次数：{int(runtime_summary.get('fallback_total', 0))}",
            ]
            st.markdown(
                """
                <div class="summary-card">
                  <div class="summary-label">代表性结果摘要</div>
                  <div class="summary-value">适合政策研判、结果复盘与证据留痕</div>
                  <div class="summary-note">建议优先查看主图、KPI、政策传导链与历史验证对照。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {item}" for item in package_summary))
            dashboard_ui.render_policy_transmission_chain(_build_overview_chain_payload(metrics), key_prefix="overview")

    st.markdown("### 汇报价值与应用边界")
    value_cols = st.columns(2)
    value_cols[0].markdown(
        """
        <div class="summary-card">
          <div class="summary-label">技术意义</div>
          <div class="summary-value">让大模型从“能理解政策”走向“能参与政策预演”</div>
          <div class="summary-note">通过结构化政策编译、多智能体认知决策、历史对照验证与反事实推演，把大模型能力落到可运行、可验证、可交付的系统中。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    value_cols[1].markdown(
        """
        <div class="summary-card">
          <div class="summary-label">社会价值</div>
          <div class="summary-value">服务公共治理，而不是娱乐化内容生成</div>
          <div class="summary-note">平台旨在帮助政策制定者更早识别潜在风险、比较不同干预方案效果，并支撑更透明、更稳健、更可复盘的公共决策过程。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 工程特性与交付能力")
    capability_cols = st.columns(3)
    capability_cards = [
        (
            "工作流主线",
            ["系统总览", "政策实验", "历史验证", "研判分析"],
            "支持从问题定义到策略建议的连续分析，不依赖单点功能拼接。",
        ),
        (
            "后端沉淀能力",
            ["政策结构化包", "向量记忆 / 事件库", "监管优化 / 帕累托", "多格式报告导出"],
            "很多能力已完成工程沉淀，前端负责把能力链按分析逻辑组织出来。",
        ),
        (
            "结果交付支持",
            ["实验报告", "分析摘要", "结果归档", "可复现信息"],
            "支持实验报告、图表导出、证据链和可复现信息沉淀。",
        ),
    ]
    for col, (title, items, note) in zip(capability_cols, capability_cards):
        item_html = "".join(f"<li>{item}</li>" for item in items)
        col.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">{title}</div>
              <ul class="ops-card-list">{item_html}</ul>
              <div class="summary-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_advanced_analysis() -> None:
    st.session_state.runtime_mode = DEMO_MODE
    st.markdown("## 研判分析中心")
    briefing_cols = st.columns(3)
    briefing_cards = [
        (
            "行为与风险诊断",
            "把价格涨跌进一步拆解成羊群效应、情绪不对称、波动聚集和风险扩散机制。",
        ),
        (
            "监管策略优化",
            "围绕候选动作、甲乙差分、帕累托权衡和推荐方案形成真正可汇报的决策支持过程。",
        ),
        (
            "结果归档与复用",
            "政策实验、历史验证、研究回测和监管优化都可以导出报告与证据链，支持留痕和复盘。",
        ),
    ]
    for col, (title, summary) in zip(briefing_cols, briefing_cards):
        col.markdown(
            f"""
            <div class="story-card">
              <div class="story-card-title">{title}</div>
              <div class="story-card-summary">{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["研究工作台", "大模型决策证据", "行为与风险诊断", "监管优化与甲乙对照", "研究验证与归档"])

    with tab0:
        _render_research_workbench_tab()
    with tab1:
        _render_ai_decision_tab()
    with tab2:
        render_behavioral_diagnostics()
    with tab3:
        reg_tab, demo_tab = st.tabs(["监管策略优化", "系统场景对照"])
        with reg_tab:
            render_regulator_optimization()
        with demo_tab:
            st.session_state.competition_mode = COMPETITION_DEMO_MODE
            render_demo_tab(ctrl=st.session_state.get("controller"))
    with tab4:
        verify_tab, value_tab = st.tabs(["研究回测与导出", "价值链路与数据飞轮"])
        with verify_tab:
            st.session_state.runtime_mode = LIVE_MODE
            render_backtest_panel(ctrl=st.session_state.get("controller"))
        with value_tab:
            _render_value_bridge_tab()


def _render_domain_hub() -> None:
    from ui.components.evolution import section, render_market_result
    from ui.team_evolution import load_context, TASK_LABELS
    section("", "金融政策风洞")
    st.write("团队调用政策解析、市场撮合、历史回放和风险计算工具，完成以下实验。")
    archive, comparisons, result = load_context()
    if comparisons:
        families=["policy_report","historical_analysis","regulatory_planning"]
        tabs=st.tabs([TASK_LABELS[f] for f in families])
        for tab, family in zip(tabs,families):
            with tab:
                case=next((c for c in comparisons if c["baseline"]["task"]["task_type"]==family),None)
                if case:
                    st.write(case["evolved"]["task"]["objective"])
                    render_market_result(case["evolved"],"domain_"+family)
                else: st.info("此类任务暂无结果。可在实验案例中运行。")
    st.markdown("### 研究工具")
    columns = st.columns(len(DOMAIN_ENTRIES))
    for column, entry in zip(columns, DOMAIN_ENTRIES):
        column.button(entry, key="domain_entry_" + entry, on_click=navigate, args=(entry,), width="stretch")
    st.button("查看历史验证", on_click=navigate, args=("历史验证",), width="stretch")


def _render_history_workspace() -> None:
    from ui.components.evolution import section, render_market_result
    from ui.team_evolution import load_context
    section("", "历史验证")
    completed, research = st.tabs(["已完成验证", "新建历史验证"])
    with completed:
        _, comparisons, _ = load_context()
        case = next((c for c in comparisons if c["baseline"]["task"]["task_type"] == "historical_analysis"), None)
        if case:
            st.write("2024 年 9 月政策窗口 · 已完成实验")
            render_market_result(case["evolved"], "history_reference")
        else:
            st.info("暂无已完成的历史验证。可在实验案例中运行历史分析任务。")
    with research:
        st.session_state.runtime_mode = LIVE_MODE
        st.session_state["history_replay_entry_mode"] = str(st.session_state.get("history_replay_entry_mode", "factor")).strip().lower() or "factor"
        render_history_replay(ctrl=st.session_state.get("controller"))
    st.button("查看运行环境与实验资料", on_click=navigate, args=("技术与复现",), width="stretch")


def main() -> None:
    _load_theme()
    _init_state()
    _render_sidebar_global()
    _render_top_entry_selector()
    entry = st.session_state.entry
    if entry in TEAM_PAGES:
        render_team_evolution(entry)
    elif entry == "金融政策风洞":
        _render_domain_hub()
    elif entry == SHOWCASE_ENTRY:
        render_default_showcase()
    elif entry == OVERVIEW_ENTRY:
        _render_overview_home()
    elif entry == "政策实验":
        st.session_state.runtime_mode = DEMO_MODE
        _render_policy_lab_compatible("standard")
    elif entry == "历史验证":
        _render_history_workspace()
    elif entry == "研判分析":
        _render_advanced_analysis()
    if st.session_state.pop("ev_scroll_to_top", False):
        # Streamlit preserves scroll position across reruns. A next-page button
        # at the bottom must reveal the next page's heading, not its footer.
        st.iframe('''<!doctype html><html><body style="margin:0;background:transparent"><script>
        requestAnimationFrame(() => {
          const main = window.parent.document.querySelector('[data-testid="stMain"]');
          if (main) main.scrollTo(0, 0);
          window.parent.scrollTo(0, 0);
        });
        </script></body></html>''', height=1)


if __name__ == "__main__":
    main()
