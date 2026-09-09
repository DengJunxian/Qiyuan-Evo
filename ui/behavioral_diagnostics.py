"""Behavioral-finance diagnostics view for stylized facts report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.components.scorecard_panel import mock_scorecard, render_scorecard_panel
from ui.narrative import narrate_payload, render_narrative_block


def _safe_get(report: Dict[str, Any], path: list[str], default: float = 0.0) -> float:
    cur: Any = report
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return float(default)
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return float(default)


def _load_social_propagation_report(report_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = report_path or Path("outputs") / "social_propagation_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_strategy_ecology_report(report_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = report_path or Path("outputs") / "strategy_ecology_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scorecard_behavioral_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    scorecard = report.get("replay_scorecard", report.get("scorecard", {}))
    if not isinstance(scorecard, dict):
        return {}
    metrics = scorecard.get("behavioral_metrics", {})
    return dict(metrics) if isinstance(metrics, dict) else {}


def _render_social_propagation_report(report: Dict[str, Any]) -> None:
    st.markdown("### 社会传播链")
    cols = st.columns(4)
    node_types = report.get("node_type_distribution", {})
    source_rankings = report.get("source_rankings", [])
    rumor = report.get("rumor_suppression", {})
    cascade = report.get("cascade_metrics", {})
    clarification = report.get("clarification_metrics", {})
    with cols[0]:
        st.metric("节点数", f"{int(report.get('snapshot_info', {}).get('node_count', 0))}")
    with cols[1]:
        st.metric("边数", f"{int(report.get('snapshot_info', {}).get('edge_count', 0))}")
    with cols[2]:
        st.metric("均值情绪", f"{float(report.get('mean_sentiment', 0.0)):.3f}")
    with cols[3]:
        st.metric("谣言压制差值", f"{float(rumor.get('delta', 0.0)):.3f}")

    ext_cols = st.columns(4)
    with ext_cols[0]:
        st.metric("瀑布覆盖率", f"{float(cascade.get('coverage', 0.0)):.2%}")
    with ext_cols[1]:
        st.metric("最大传播深度", f"{float(cascade.get('max_depth', 0.0)):.0f}")
    with ext_cols[2]:
        st.metric("推荐触达数", f"{int(float(cascade.get('recommendation_count', 0.0)))}")
    with ext_cols[3]:
        st.metric("澄清延迟", f"{float(clarification.get('clarification_latency', 0.0)):.1f}")

    st.dataframe(
        pd.DataFrame(
            [{"节点类型": key, "数量": value} for key, value in sorted(node_types.items(), key=lambda item: item[1], reverse=True)]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if source_rankings:
        st.markdown("#### 传播影响力")
        st.dataframe(
            pd.DataFrame(source_rankings[:10]),
            use_container_width=True,
            hide_index=True,
        )
    leaders = report.get("opinion_leaders", [])
    if leaders:
        st.markdown("#### 意见领袖")
        st.dataframe(pd.DataFrame(leaders[:10]), use_container_width=True, hide_index=True)
    recommendations = report.get("recommendation_trace", [])
    if recommendations:
        st.markdown("#### 媒体推荐触达")
        st.dataframe(pd.DataFrame(recommendations[:20]), use_container_width=True, hide_index=True)

    chain = report.get("propagation_chain", [])
    if chain:
        st.markdown("#### 传播链")
        chain_df = pd.DataFrame(chain)
        if {"source_id", "target_id", "topic", "kind"} <= set(chain_df.columns):
            cols = ["source_id", "target_id", "topic", "kind", "signal", "belief_delta", "delay", "received_tick"]
            cols = [col for col in cols if col in chain_df.columns]
            st.dataframe(chain_df[cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.dataframe(chain_df.head(20), use_container_width=True, hide_index=True)

    observation_packets = report.get("observation_packets", {})
    if observation_packets:
        st.markdown("#### 节点观察卡")
        packet_rows: List[Dict[str, Any]] = []
        for node_id, packet in list(observation_packets.items())[:10]:
            packet_rows.append(
                {
                    "node_id": node_id,
                    "node_type": packet.get("node_type", ""),
                    "sentiment": packet.get("sentiment", 0.0),
                    "first_seen_tick": packet.get("first_seen_tick", -1),
                    "rumor_pressure": packet.get("memory_seed", {}).get("rumor_pressure", 0.0),
                    "refutation_pressure": packet.get("memory_seed", {}).get("refutation_pressure", 0.0),
                }
            )
        st.dataframe(pd.DataFrame(packet_rows), use_container_width=True, hide_index=True)

    bdi_packets = report.get("bdi_observation_packets", {})
    if bdi_packets:
        st.markdown("#### BDI 观察卡")
        bdi_rows: List[Dict[str, Any]] = []
        for node_id, packet in list(bdi_packets.items())[:10]:
            belief = packet.get("belief", {})
            desire = packet.get("desire", {})
            intention = packet.get("intention", {})
            bdi_rows.append(
                {
                    "node_id": node_id,
                    "belief_sentiment": belief.get("sentiment", 0.0),
                    "rumor_pressure": belief.get("rumor_pressure", 0.0),
                    "information_need": desire.get("information_need", 0.0),
                    "trading_bias": intention.get("trading_bias", 0.0),
                    "communication_action": intention.get("communication_action", ""),
                }
            )
        st.dataframe(pd.DataFrame(bdi_rows), use_container_width=True, hide_index=True)

    st.markdown("#### 智能解读")
    st.markdown(
        narrate_payload(
            "社会传播报告解读",
            report,
            context="聚焦节点结构、传播链路与谣言压制效果。",
        )
    )


def _render_strategy_ecology_report(report: Dict[str, Any]) -> None:
    st.markdown("### 策略群体演化")
    latest = report.get("latest", {}) if isinstance(report.get("latest", {}), dict) else {}
    cohorts = latest.get("cohorts", {}) if isinstance(latest.get("cohorts", {}), dict) else {}
    dominant = latest.get("dominant_cohort", {}) if isinstance(latest.get("dominant_cohort", {}), dict) else {}
    force = latest.get("force_breakdown", {}) if isinstance(latest.get("force_breakdown", {}), dict) else {}
    cols = st.columns(4)
    with cols[0]:
        st.metric("群体数", f"{len(cohorts)}")
    with cols[1]:
        st.metric("主导群体", str(dominant.get("label", dominant.get("cohort_id", "-")))[:18])
    with cols[2]:
        st.metric("主导目标份额", f"{float(dominant.get('target_share', 0.0)):.2%}")
    with cols[3]:
        st.metric("政策环境", str(latest.get("policy_regime", "neutral")))

    rows = []
    for cohort in cohorts.values():
        if not isinstance(cohort, dict):
            continue
        rows.append(
            {
                "cohort_id": cohort.get("cohort_id", ""),
                "label": cohort.get("label", ""),
                "agent_count": cohort.get("agent_count", 0),
                "population_share": cohort.get("population_share", 0.0),
                "target_share": cohort.get("target_share", 0.0),
                "capital_share": cohort.get("capital_share", 0.0),
                "fitness": cohort.get("fitness", 0.0),
                "selection_force": cohort.get("selection_force", 0.0),
                "innovation_force": cohort.get("innovation_force", 0.0),
                "perturbation_force": cohort.get("perturbation_force", 0.0),
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        chart_df = df.set_index("label")[["population_share", "target_share"]]
        st.bar_chart(chart_df)

    force_rows = [
        {"演化力": "选择", "强度": float(force.get("selection", 0.0))},
        {"演化力": "创新", "强度": float(force.get("innovation", 0.0))},
        {"演化力": "扰动", "强度": float(force.get("perturbation", 0.0))},
    ]
    st.dataframe(pd.DataFrame(force_rows), use_container_width=True, hide_index=True)

    transitions = latest.get("transition_events", [])
    if transitions:
        st.markdown("#### 群体迁移事件")
        st.dataframe(pd.DataFrame(transitions), use_container_width=True, hide_index=True)

    st.markdown("#### 智能解读")
    st.markdown(
        narrate_payload(
            "策略群体演化报告解读",
            latest,
            context="请解释长期政策环境下策略群体份额、选择力、创新力与扰动力的变化。",
        )
    )


def render_behavioral_diagnostics(report_path: Path | None = None) -> None:
    st.markdown("## 行为金融诊断")
    st.caption("自动读取仿真输出的典型事实，聚焦 CSAD、PGR/PLR、波动聚集、回撤分布与历史新高异象。")

    path = report_path or Path("outputs") / "stylized_facts_report.json"
    if not path.exists():
        st.warning(f"未找到报告文件：{path.as_posix()}。请先运行仿真至少 1 步。")
        return

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"读取报告失败：{exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CSAD 均值", f"{_safe_get(report, ['csad', 'mean']):.4f}")
    with c2:
        st.metric("PGR", f"{_safe_get(report, ['pgr_plr', 'pgr']):.3f}")
    with c3:
        st.metric("PLR", f"{_safe_get(report, ['pgr_plr', 'plr']):.3f}")
    with c4:
        st.metric("波动聚集（|r| 一阶滞后）", f"{_safe_get(report, ['volatility_clustering', 'abs_return_lag1_autocorr']):.3f}")

    summary_rows = [
        {"指标": "CSAD 羊群回归 gamma2", "数值": _safe_get(report, ["csad", "herding_regression", "gamma2"])},
        {"指标": "处置效应差值（PGR-PLR）", "数值": _safe_get(report, ["pgr_plr", "disposition_gap"])},
        {"指标": "历史新高超额表现", "数值": _safe_get(report, ["all_time_high_effect", "ath_outperformance"])},
        {"指标": "最大回撤", "数值": _safe_get(report, ["drawdown_distribution", "max_drawdown"])},
        {"指标": "损失厌恶强度均值", "数值": _safe_get(report, ["loss_aversion_intensity", "mean"])},
    ]
    unified_behavior = _scorecard_behavioral_metrics(report)
    if unified_behavior:
        summary_rows.extend(
            [
                {"指标": "统一评估卡 CSAD", "数值": float(unified_behavior.get("csad_mean", unified_behavior.get("csad", 0.0)))},
                {"指标": "统一评估卡羊群强度", "数值": float(unified_behavior.get("herd_intensity", 0.0))},
            ]
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.markdown("### 回放评估卡行为分解")
    scorecard_payload = report.get("replay_scorecard", report.get("scorecard", mock_scorecard()))
    render_scorecard_panel(scorecard_payload if isinstance(scorecard_payload, dict) else mock_scorecard(), key_prefix="behavioral_scorecard")
    render_narrative_block(
        "行为金融关键指标解读",
        summary_rows,
        context="请解释羊群效应、处置效应、历史新高效应与回撤分布所反映的市场行为特征。",
        cache_namespace="behavioral_diag_narrative_cache",
    )

    st.markdown("### 报告解读")
    st.markdown(
        narrate_payload(
            "行为金融报告解读",
            report,
            context="请解释典型事实指标表现与潜在市场行为含义。",
        )
    )
    st.download_button(
        "下载典型事实报告 JSON",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="stylized_facts_report.json",
        mime="application/json",
        use_container_width=True,
    )


    sensitivity_path = Path("outputs") / "parameter_sensitivity.csv"
    st.markdown("### 参数敏感性（损失厌恶 / 参考适应 / 边权重）")
    if sensitivity_path.exists():
        try:
            sens_df = pd.read_csv(sensitivity_path)
            st.dataframe(sens_df, use_container_width=True, hide_index=True)
            if {"loss_aversion", "avg_trading_intent", "edge_weight"} <= set(sens_df.columns):
                pivot = sens_df.pivot_table(
                    index="loss_aversion",
                    columns="edge_weight",
                    values="avg_trading_intent",
                    aggfunc="mean",
                )
                st.caption("平均交易意图热力图（按损失厌恶 × 边权重）")
                st.dataframe(pivot, use_container_width=True)
            render_narrative_block(
                "参数敏感性解读",
                sens_df.head(15).to_dict(orient="records"),
                context="请解释损失厌恶、参考适应和边权重变化如何影响交易意图与系统稳定性。",
                cache_namespace="behavioral_diag_narrative_cache",
            )
        except Exception as exc:
            st.warning(f"参数敏感性文件读取失败：{exc}")
    else:
        st.info("未发现 outputs/parameter_sensitivity.csv，可先运行参数敏感性脚本。")


    ab_path = Path("outputs") / "intervention_effect_report.json"
    st.markdown("### 干预前后甲乙对照")
    if ab_path.exists():
        try:
            ab = json.loads(ab_path.read_text(encoding="utf-8"))
            before = ab.get("before", {})
            after = ab.get("after", {})
            rows = []
            for metric in ("volatility", "mean_abs_return", "abuse_event_rate"):
                b = float(before.get(metric, 0.0))
                a = float(after.get(metric, 0.0))
                rows.append({"指标": metric, "干预前": b, "干预后": a, "变化": a - b})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                f"干预是否开启={ab.get('intervention_active', False)}，"
                f"触发时点={ab.get('intervention_tick', '无')}"
            )
            render_narrative_block(
                "干预前后对照解读",
                {"rows": rows, "meta": {"intervention_active": ab.get("intervention_active", False), "intervention_tick": ab.get("intervention_tick", "无")}},
                context="请解释干预前后波动、绝对收益和异常事件率的变化，判断干预是否有效。",
                cache_namespace="behavioral_diag_narrative_cache",
            )
        except Exception as exc:
            st.warning(f"甲乙对照报告读取失败：{exc}")
    else:
        st.info("未发现 outputs/intervention_effect_report.json。")

    eco_path = Path("outputs") / "ecology_metrics.csv"
    abuse_path = Path("outputs") / "market_abuse_report.json"
    st.markdown("### 生态与滥用导出")
    cols = st.columns(2)
    with cols[0]:
        if eco_path.exists():
            st.download_button(
                "下载 ecology_metrics.csv",
                data=eco_path.read_bytes(),
                file_name="ecology_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with cols[1]:
        if abuse_path.exists():
            st.download_button(
                "下载 market_abuse_report.json",
                data=abuse_path.read_bytes(),
                file_name="market_abuse_report.json",
                mime="application/json",
                use_container_width=True,
            )

    strategy_report = _load_strategy_ecology_report()
    if strategy_report:
        _render_strategy_ecology_report(strategy_report)
    else:
        st.info("未发现 outputs/strategy_ecology_report.json。可在仿真运行后查看策略群体演化。")

    social_report = _load_social_propagation_report()
    st.markdown("### 社会传播报告")
    if social_report:
        _render_social_propagation_report(social_report)
    else:
        st.info("未发现 outputs/social_propagation_report.json。可在社交传播引擎运行后导出。")
