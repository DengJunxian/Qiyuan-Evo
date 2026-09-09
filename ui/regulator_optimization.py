"""Dedicated regulator optimization page with A/B and Pareto visualization."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.ui_text import localize_dataframe_columns, zh_action_name, zh_metric_name, zh_provider_name
from regulator_agent import run_regulatory_closed_loop
from ui.components.optimization_report_panel import render_optimization_report_panel
from ui.components.repro_meta import build_reproducibility_meta, render_reproducibility_panel
from ui.chart_theme import PLOTLY_DARK_LAYOUT
from ui.dashboard import render_discovered_metrics_panel
from ui.narrative import narrate_payload, render_narrative_block


REGULATOR_OPTIMIZATION_PAGE_FLAG = "regulator_optimization_page_v1"
_REGULATOR_RESULT_STATE_KEY = "regulator_optimization_result"


def _display_path(value: Any) -> str:
    text = str(value or "q_learning_baseline")
    return {
        "q_learning_baseline": "强化学习基线",
        "bayesian_optimization": "贝叶斯优化",
        "nsga_ii": "多目标进化搜索",
        "default_production_path": "默认优化路径",
    }.get(text, zh_provider_name(text))


def _safe_rows(items: Any) -> List[Dict[str, Any]]:
    if isinstance(items, list):
        return [dict(x) for x in items if isinstance(x, dict)]
    return []


def _build_regulator_result_frames(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    counterfactual = result.get("counterfactual_ab", {}) if isinstance(result, dict) else {}
    recommendation = result.get("recommendation", {}) if isinstance(result, dict) else {}
    baseline = counterfactual.get("baseline", {}) if isinstance(counterfactual, dict) else {}
    candidates = _safe_rows(counterfactual.get("candidates"))
    deltas = _safe_rows(counterfactual.get("deltas"))
    pareto = _safe_rows(result.get("pareto_frontier"))
    evidence = _safe_rows(recommendation.get("evidence_chain"))

    baseline_df = pd.DataFrame([baseline]) if isinstance(baseline, dict) and baseline else pd.DataFrame()
    candidates_df = pd.DataFrame(candidates)
    deltas_df = pd.DataFrame(deltas)
    pareto_df = pd.DataFrame(pareto)
    recommendation_df = (
        pd.DataFrame([recommendation.get("scorecard", {})])
        if isinstance(recommendation, dict) and recommendation.get("scorecard")
        else pd.DataFrame()
    )
    evidence_df = pd.DataFrame(evidence)

    for frame in (baseline_df, candidates_df, deltas_df, pareto_df, recommendation_df, evidence_df):
        for col in (
            "macro_stability",
            "liquidity",
            "intervention_cost",
            "avg_reward",
            "financing_function",
            "fairness_compliance",
            "composite_score",
            "delta",
        ):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    return {
        "baseline": baseline_df,
        "candidates": candidates_df,
        "deltas": deltas_df,
        "pareto": pareto_df,
        "recommendation": recommendation_df,
        "evidence": evidence_df,
    }


def render_regulator_optimization() -> None:
    st.markdown("## 监管策略优化")

    with st.form("regulator_optimization_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            episodes = st.slider("训练轮数", min_value=10, max_value=400, value=120, step=10)
            max_steps = st.slider("每轮最大步数", min_value=4, max_value=96, value=24, step=4)
            top_k = st.slider("候选动作数量", min_value=1, max_value=5, value=3, step=1)
        with right:
            seed = int(st.number_input("随机种子", min_value=0, max_value=2_147_483_647, value=42, step=1))
            use_toy_env = st.toggle("直接使用简化环境", value=False)
            st.caption("默认优先真实环境；真实环境初始化失败时，自动回退到简化环境。")
        submitted = st.form_submit_button("运行监管优化", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("正在运行监管闭环优化..."):
            result = run_regulatory_closed_loop(
                episodes=int(episodes),
                max_steps_per_episode=int(max_steps),
                seed=int(seed),
                top_k=int(top_k),
                use_toy_env=bool(use_toy_env),
            )
            st.session_state[_REGULATOR_RESULT_STATE_KEY] = result

    result = st.session_state.get(_REGULATOR_RESULT_STATE_KEY)
    if not isinstance(result, dict):
        st.info("先运行一次优化以查看甲乙反事实对照、帕累托前沿和推荐证据链。")
        return

    summary = result.get("training_summary", {}) if isinstance(result, dict) else {}
    reproducibility = result.get("reproducibility", {}) if isinstance(result, dict) else {}
    recommendation = result.get("recommendation", {}) if isinstance(result, dict) else {}
    blackbox = result.get("blackbox_optimization", {}) if isinstance(result, dict) else {}
    opt_report = result.get("optimization_report", {}) if isinstance(result, dict) else {}
    frames = _build_regulator_result_frames(result)

    metric_rows = [
        [("平均回合收益", f"{float(summary.get('avg_episode_reward', 0.0)):.4f}"), ("最优动作得分", f"{float(summary.get('best_action_score', 0.0)):.4f}"), ("帕累托点数", str(len(frames["pareto"])))],
        [("强化学习状态数量", str(int(summary.get("q_states", 0)))), ("默认优化路径", _display_path(result.get("default_production_path", "q_learning_baseline")))],
    ]
    for row in metric_rows:
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            col.metric(label, value)

    st.markdown("### 发现目标约束")
    render_discovered_metrics_panel(result.get("discovered_objectives", {}), key_prefix="regulator_objectives")

    env_selection = reproducibility.get("env_selection", {}) if isinstance(reproducibility, dict) else {}
    with st.expander("导出与复现信息", expanded=False):
        st.caption(
            "可复现信息："
            f"随机种子={reproducibility.get('seed', 0)} | "
            f"配置哈希={reproducibility.get('config_hash', '')} | "
            f"训练轮数={reproducibility.get('episodes', 0)} | "
            f"最大步数={reproducibility.get('max_steps_per_episode', 0)}"
        )
        if isinstance(env_selection, dict) and env_selection:
            st.caption(
                "环境信息："
                f"路径={env_selection.get('selected_path', '')} | "
                f"是否回退={env_selection.get('fallback_used', False)}"
            )
        render_reproducibility_panel(
            build_reproducibility_meta(
                data_snapshot_hash=str(reproducibility.get("data_snapshot_hash", reproducibility.get("dataset_snapshot_id", "regulator_env_snapshot"))),
                config_hash=str(reproducibility.get("config_hash", "")),
                random_seed=int(reproducibility.get("seed", 42) or 42),
                llm_provider_chain=["regulator_q_learning", "bayesian_optimization", "nsga_ii"],
                calibration_parameter_set_id=str(reproducibility.get("calibration_parameter_set_id", "regulator_default_v1")),
                extra={"env_selection": env_selection},
            ),
            key_prefix="regulator_repro",
        )

    st.markdown("### 黑箱多目标优化")
    if isinstance(blackbox, dict) and blackbox:
        bo_best = dict(blackbox.get("bayesian_optimization", {}).get("best", {}) or {})
        validation = dict(blackbox.get("validation", {}) or {})
        report_best = dict(opt_report.get("best_solution", {}) or {})
        st.caption(
            "贝叶斯优化用于静态政策包搜索；多目标进化搜索输出多目标帕累托前沿；晋升默认路径需在固定回放窗口中同时优于规则基线。"
        )
        cols = st.columns(4)
        cols[0].metric("贝叶斯优化最优分", f"{float(bo_best.get('score', 0.0)):.4f}")
        cols[1].metric("稳定胜场", f"{int(validation.get('stable_win_count', 0))}/{int(validation.get('required_windows', 2))}")
        cols[2].metric("晋升默认", str(bool(validation.get("promote_blackbox_default", False))))
        cols[3].metric("多目标进化点数", str(len(blackbox.get("nsga_ii", {}).get("pareto_frontier", []) or [])))
        if report_best:
            st.dataframe(localize_dataframe_columns(pd.DataFrame([report_best.get("metrics", {})])), use_container_width=True, hide_index=True)
        if validation.get("windows"):
            st.dataframe(localize_dataframe_columns(pd.DataFrame(validation.get("windows", []))), use_container_width=True, hide_index=True)
        nsga_pareto = blackbox.get("nsga_ii", {}).get("pareto_frontier", [])
        if nsga_pareto:
            pareto_rows = []
            for row in nsga_pareto:
                if not isinstance(row, dict):
                    continue
                metrics = dict(row.get("metrics", {}) or {})
                objectives = dict(row.get("objectives", {}) or {})
                pareto_rows.append({"score": row.get("score", 0.0), **metrics, **objectives})
            if pareto_rows:
                st.dataframe(localize_dataframe_columns(pd.DataFrame(pareto_rows).head(20)), use_container_width=True, hide_index=True)
        render_narrative_block(
            "黑箱优化报告解读",
            {
                "best_solution": opt_report.get("best_solution", {}),
                "constraint_violations": opt_report.get("constraint_violations", {}),
                "validation": opt_report.get("validation", {}),
                "final_recommendation_text": opt_report.get("final_recommendation_text", ""),
            },
            context="请解释贝叶斯优化、多目标进化搜索、规则基线与强化学习的关系，并说明为什么当前默认生产路径可以或不可以晋升。",
            cache_namespace="regulator_opt_narrative_cache",
        )
    else:
        st.info("暂无黑箱优化输出。")

    render_optimization_report_panel(result, key_prefix="regulator_optimization_report")

    st.markdown("### 反事实甲乙对照")
    left, right = st.columns(2)
    with left:
        if frames["baseline"].empty:
            st.info("暂无基线结果。")
        else:
            st.dataframe(localize_dataframe_columns(frames["baseline"]), use_container_width=True, hide_index=True)
    with right:
        if frames["deltas"].empty:
            st.info("暂无候选动作差分。")
        else:
            st.dataframe(localize_dataframe_columns(frames["deltas"]), use_container_width=True, hide_index=True)
    render_narrative_block(
        "监管甲乙对照解读",
        {
            "baseline": frames["baseline"].to_dict(orient="records"),
            "deltas": frames["deltas"].head(10).to_dict(orient="records"),
        },
        context="请解释基线方案与候选动作之间的主要差异，说明哪些指标改善最大、代价最大。",
        cache_namespace="regulator_opt_narrative_cache",
    )

    st.markdown("### 帕累托前沿")
    pareto_df = frames["pareto"]
    if pareto_df.empty:
        st.info("暂无帕累托数据。")
    else:
        plot_df = pareto_df.copy()
        for col in ("intervention_cost", "macro_stability", "liquidity", "composite_score", "avg_reward"):
            if col not in plot_df.columns:
                plot_df[col] = 0.0
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
        color_col = "composite_score" if "composite_score" in pareto_df.columns else "avg_reward"
        size_values = (plot_df["liquidity"].clip(lower=0.01) * 42).clip(lower=10, upper=46)
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=plot_df["intervention_cost"],
                    y=plot_df["macro_stability"],
                    mode="markers",
                    name="候选方案",
                    marker=dict(
                        size=size_values,
                        color=plot_df[color_col],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="综合得分" if color_col == "composite_score" else "平均收益"),
                        line=dict(color="rgba(255,255,255,0.35)", width=1),
                    ),
                    text=plot_df.get("action_description", plot_df.get("action_signature", pd.Series(["候选方案"] * len(plot_df)))),
                    hovertemplate=(
                        "方案：%{text}<br>"
                        "干预成本：%{x:.3f}<br>"
                        "宏观稳定性：%{y:.3f}<br>"
                        "流动性：%{marker.size:.1f}<extra></extra>"
                    ),
                )
            ]
        )
        cost_mid = float(plot_df["intervention_cost"].median())
        stability_mid = float(plot_df["macro_stability"].median())
        fig.add_vline(x=cost_mid, line_dash="dot", line_color="rgba(148,163,184,0.55)")
        fig.add_hline(y=stability_mid, line_dash="dot", line_color="rgba(148,163,184,0.55)")
        annotations = [
            (plot_df["intervention_cost"].min(), plot_df["macro_stability"].max(), "低成本高稳定：优先候选"),
            (plot_df["intervention_cost"].max(), plot_df["macro_stability"].max(), "高成本高稳定：强干预"),
            (plot_df["intervention_cost"].min(), plot_df["macro_stability"].min(), "低成本低稳定：观察"),
            (plot_df["intervention_cost"].max(), plot_df["macro_stability"].min(), "高成本低稳定：谨慎"),
        ]
        for x, y, text in annotations:
            fig.add_annotation(x=float(x), y=float(y), text=text, showarrow=False, font=dict(size=12, color="#e2e8f0"), bgcolor="rgba(10,25,49,0.72)")
        fig.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="帕累托前沿：稳定性、成本与流动性权衡",
            xaxis_title="干预成本",
            yaxis_title="宏观稳定性",
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("原始指标表（可展开）", expanded=False):
            st.dataframe(localize_dataframe_columns(pareto_df), use_container_width=True, hide_index=True)
        render_narrative_block(
            "帕累托前沿解读",
            pareto_df.head(12).to_dict(orient="records"),
            context="请解释稳定性、成本与流动性之间的权衡关系，并指出推荐方案为何位于合理区间。",
            cache_namespace="regulator_opt_narrative_cache",
        )

    st.markdown("### 推荐方案与证据")
    left, right = st.columns(2)
    with left:
        st.markdown(
            narrate_payload(
                "推荐方案解读",
                {
                    "action_description": recommendation.get("action_description", ""),
                    "tradeoff_summary": recommendation.get("tradeoff_summary", ""),
                    "side_effects": recommendation.get("side_effects", []),
                },
                context="解释推荐动作、收益-成本权衡和可能副作用。",
            )
        )
        if not frames["recommendation"].empty:
            st.dataframe(localize_dataframe_columns(frames["recommendation"]), use_container_width=True, hide_index=True)
    with right:
        if not frames["evidence"].empty:
            st.dataframe(localize_dataframe_columns(frames["evidence"]), use_container_width=True, hide_index=True)
        else:
            st.info("暂无推荐证据。")

    if not frames["candidates"].empty:
        st.markdown("### 候选方案")
        st.dataframe(localize_dataframe_columns(frames["candidates"]), use_container_width=True, hide_index=True)
        render_narrative_block(
            "候选方案清单解读",
            frames["candidates"].head(12).to_dict(orient="records"),
            context="请概括候选方案的分布特征，并指出适合重点汇报的 1 到 2 个候选动作。",
            cache_namespace="regulator_opt_narrative_cache",
        )

    export_cols = st.columns(2)
    export_cols[0].download_button(
        "下载监管优化结果 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="regulator_optimization_result.json",
        mime="application/json",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "下载帕累托数据 CSV",
        data=frames["pareto"].to_csv(index=False).encode("utf-8"),
        file_name="regulator_pareto.csv",
        mime="text/csv",
        use_container_width=True,
    )


__all__ = ["REGULATOR_OPTIMIZATION_PAGE_FLAG", "_build_regulator_result_frames", "render_regulator_optimization"]
