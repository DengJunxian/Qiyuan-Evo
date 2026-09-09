"""Trace-backed cockpit visuals. HTML is escaped; SVG needs no remote assets."""
from __future__ import annotations

from datetime import datetime
from html import escape
import json
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.team_evolution.models import TeamTopology
from core.team_evolution.evaluation import valid_evidence
from ui.chart_theme import apply_dark_theme

from ui.presentation import (ROLE_NAMES, EVIDENCE_NAMES, STATE_NAMES, ACTION_NAMES,
                             TOOL_NAMES, PLAN_NAMES, chinese, evidence_progress, outcome_summary, action_summary)
ROLE_ZH = ROLE_NAMES
STATE_COLORS = {"IDLE": "#718399", "PLANNING": "#43c5d0", "RUNNING": "#43c5d0", "WAITING": "#8494a9",
                "REVIEWING": "#87aef5", "FAILED": "#ed8b82", "COMPLETED": "#67c7b3", "EVOLVING": "#dcb473"}


def html(value):
    return escape(str(value), quote=True)


def critique_label(critique):
    if critique["problem"].startswith("缺少有效的 "):
        return EVIDENCE_NAMES.get(critique["dimension"], chinese(critique["dimension"])) + "验证未完成"
    return chinese(critique["problem"])


def section(kicker, title, note=""):
    st.html(f'<div class="ev-section"><h2>{html(title)}</h2></div>')


def agent_states(trace):
    config = trace["team_config"]
    latest = {e["agent"]: e for e in trace.get("agent_events", [])}
    states = {}
    for name in config["topology"]["active_agents"]:
        event = latest.get(name)
        if event:
            status = event["status"].upper()
            if status == "RUNNING":
                status = {"planner": "PLANNING", "critic": "REVIEWING", "evolution_controller": "EVOLVING"}.get(name, status)
        else:
            status = "WAITING" if latest and not trace.get("evaluation") else "IDLE"
        states[name] = status
    return states


def edge_kind(a, b):
    if a == "planner":
        return "delegation"
    if b == "critic":
        return "review"
    if a == "critic":
        return "result"
    return "message"


def graph_svg(trace):
    """Stable serpentine topological layout, including genuine parallel branches."""
    config = trace["team_config"]
    topology = TeamTopology(**config["topology"])
    layers = topology.layers()
    nodes = [node for layer in layers for node in layer]
    states = agent_states(trace)
    positions = {}
    for index, name in enumerate(nodes):
        row, col = divmod(index, 3)
        if row % 2:
            col = 2 - col
        positions[name] = (14 + col * 220, 18 + row * 139)
    height = math.ceil(len(nodes) / 3) * 139 + 4
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 670 {height}" role="img" aria-label="团队协作图" class="ev-team-svg">',
             '<defs><marker id="ev-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#506984"/></marker></defs>']
    for a, b in topology.communication_edges:
        ax, ay = positions[a]; bx, by = positions[b]
        if ay == by:
            direction = 1 if bx > ax else -1
            x1, x2 = ax + (202 if direction == 1 else 0), bx + (0 if direction == 1 else 202)
            y1, y2 = ay + 52, by + 52
            path = f'M{x1},{y1} L{x2},{y2}'
        else:
            x1, x2 = ax + 101, bx + 101
            y1, y2 = ay + 100, by
            mid = (y1 + y2) / 2
            path = f'M{x1},{y1} C{x1},{mid} {x2},{mid} {x2},{y2}'
        parts.append(f'<path d="{path}" fill="none" stroke="#506984" stroke-width="1.5" marker-end="url(#ev-arrow)"><title>{html(a)} → {html(b)} · {edge_kind(a,b)}</title></path>')
        edge_label={"delegation":"分派", "review":"审查", "result":"结果", "message":"消息"}[edge_kind(a,b)]
        lx,ly=(x1+x2)/2,(y1+y2)/2
        parts.append(f'<text x="{lx + (20 if ay != by else 0)}" y="{ly - (7 if ay == by else 0)}" '
                     f'text-anchor="middle" fill="#829cb5" font-size="9">{edge_label}</text>')
    for name in nodes:
        x, y = positions[name]
        spec = config["roles"][name]
        status = states[name]; color = STATE_COLORS.get(status, STATE_COLORS["IDLE"])
        executed = list(dict.fromkeys(c["tool"] for c in trace.get("tool_calls",[]) if c.get("agent")==name))
        tools = "、".join(chinese(k) for k in (executed or spec.get("tools", []))) or "任务规划"
        parallel = any(name in layer and len(layer) > 1 for layer in layers)
        title = f'{ROLE_NAMES.get(name,name)} | {spec["role_type"]} | {status} | {trace["task_id"]} | Prompt {spec["prompt_version"]} | {tools}'
        display = ROLE_NAMES.get(name, name)
        if len(display) > 24:
            display = "Repro. Reviewer"
        parts.append(f'<g class="ev-node" role="group" aria-label="{html(title)}" tabindex="0"><title>{html(title)}</title><rect x="{x}" y="{y}" width="202" height="100" rx="10" fill="#101c2d" stroke="#2a4058"/>'
                     f'<rect x="{x}" y="{y+17}" width="3" height="65" rx="1.5" fill="{color}"/>'
                     f'<text x="{x+14}" y="{y+23}" fill="#eef4ff" font-size="20" font-weight="600">{html(display)}</text>'
                     f'<text x="{x+14}" y="{y+43}" fill="#a7b8cc" font-size="13">提示词 {html(spec["prompt_version"])}</text>'
                     f'<text x="{x+14}" y="{y+64}" fill="{color}" font-size="13" letter-spacing=".6">{html(STATE_NAMES.get(status,status))}{" · 并行" if parallel else ""}</text>'
                     f'<text x="{x+14}" y="{y+85}" fill="#a3b3c7" font-size="12">{html(tools[:28])}{"…" if len(tools)>28 else ""}</text></g>')
    parts.append('</svg>')
    return ''.join(parts)


def render_graph(trace, *, inspect=False, controller=None):
    height = math.ceil(len(trace["team_config"]["topology"]["active_agents"]) / 3) * 139 + 4
    # st.html sanitizes SVG in some Streamlit builds. All variable text below is
    # escaped; this static local document needs no CDN or executable scripts.
    document = ('<!doctype html><html><head><meta charset="utf-8"><style>'
                'body{margin:0;background:transparent;font-family:Arial,"Microsoft YaHei",sans-serif}'
                'svg{display:block;width:100%;height:auto;max-height:390px}'
                '.ev-node:hover rect{stroke:#58bac7}'
                '</style></head><body>' + graph_svg(trace) + '</body></html>')
    st.iframe(document, height="content")
    if controller:
        status = "EVOLVING" if controller.get("status") == "running" else controller.get("status", "IDLE").upper()
        st.html(f'<div class="ev-controller"><span class="ev-dot" style="background:{STATE_COLORS.get(status,"#718399")}"></span>'
                f'<b>演进控制器</b><span>{html(STATE_NAMES.get(status,status))}</span><span>读取反馈 → 调整配置 → 重新评估</span></div>')
    if inspect:
        with st.expander("节点与通信边详情"):
            states = agent_states(trace)
            st.dataframe(pd.DataFrame([{"成员": ROLE_NAMES.get(k,k), "职责": chinese(v["role_type"]), "状态": STATE_NAMES.get(states[k],states[k]),
                                        "任务": trace["task_id"], "提示词版本": v["prompt_version"], "工具": "、".join(chinese(t) for t in v["tools"])}
                                       for k,v in trace["team_config"]["roles"].items() if k in states]), hide_index=True, width="stretch")
            st.json(trace["team_config"]["topology"], expanded=False)


def events_from_trace(trace):
    rows = []
    for event in trace.get("agent_events", []):
        role = event["agent"]
        tools = [c for c in trace.get("tool_calls", []) if c.get("call_id") in event.get("tool_call_ids", [])]
        if not tools:
            tools = [c for c in trace.get("tool_calls", []) if c.get("agent") == role]
        evidence_ids = event.get("evidence_ids", [])
        summary = event.get("decision_summary") or f"{ROLE_ZH.get(role,role)}正在执行"
        if event["status"] == "completed":
            if role == "planner" and trace.get("plan"):
                summary = f'拆解为 {len(trace["plan"]["subtasks"])} 个子任务，安排依赖与执行分工'
            elif role == "critic":
                summary = f'完成风险与证据审查，发现 {len(trace.get("critiques", []))} 项反馈'
            elif role == "reporter":
                summary = f'生成 {len(trace.get("report", {}).get("conclusions", []))} 条绑定证据的报告结论'
            elif evidence_ids:
                summary = "完成 " + "、".join(EVIDENCE_NAMES.get(k,k) for k in evidence_ids)
        rows.append({"id": event.get("event_id", role), "timestamp": event.get("finished_at", event.get("started_at", "")),
                     "actor": ROLE_NAMES.get(role, role), "kind": "agent", "status": event["status"].upper(),
                     "summary": summary,
                     "input_summary": event.get("input_summary") or {"objective": trace.get("task", {}).get("objective"), "upstream": {}},
                     "output_summary": event.get("decision_summary", ""),
                     "evidence": evidence_ids, "tool": tools,
                     "artifact": {k: trace["artifacts"][k] for k in evidence_ids if k in trace["artifacts"]},
                     "evaluation": trace.get("evaluation", {}), "run_id": trace["run_id"]})
    for index, message in enumerate(trace.get("messages", [])):
        a,b = message["sender"], message["receiver"]
        payload = message["payload_summary"]
        evidence = payload.get("evidence_ids", [])
        summary = "下发任务计划" if a=="planner" else (
            "传递" + "、".join(EVIDENCE_NAMES.get(k,k) for k in evidence) + "结果" if evidence else "移交任务结果")
        rows.append({"id": f"message-{index}", "timestamp": message["timestamp"],
                     "actor": f"{ROLE_NAMES.get(a,a)} → {ROLE_NAMES.get(b,b)}", "kind": edge_kind(a,b),
                     "status": "COMPLETED", "summary": summary,
                     "input_summary": payload, "output_summary": message["message_type"],
                     "evidence": payload.get("evidence_ids", []), "tool": [], "artifact": {},
                     "evaluation": {}, "run_id": trace["run_id"]})
    critic_time = next((e.get("finished_at",e.get("started_at","")) for e in trace.get("agent_events", []) if e["agent"]=="critic"), "")
    for index, critique in enumerate(trace.get("critiques", [])):
        rows.append({"id": f"critique-{index}", "timestamp": critic_time, "actor": ROLE_NAMES["critic"], "kind": "feedback",
                     "status": "REVIEWING", "summary": critique["problem"], "input_summary": critique["evidence"],
                     "output_summary": critique["recommendation"], "evidence": critique["evidence"], "tool": [],
                     "artifact": {k: trace["artifacts"][k] for k in critique["evidence"] if k in trace["artifacts"]},
                     "evaluation": {"severity": critique["severity"], "attribution": critique["attribution"]}, "run_id": trace["run_id"]})
    return sorted(rows, key=lambda e: e["timestamp"])


def clock_text(timestamp):
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return "—"


def render_events(trace, *, compact=False, rows=None):
    events = events_from_trace(trace) if rows is None else rows
    if not events:
        st.info("还没有执行记录。运行一次实验即可查看成员分工。")
        return
    if compact:
        selected = [e for e in events if e["kind"] in {"agent", "feedback"}][-5:]
        for event in selected:
            st.html(f'<div class="ev-event"><time>{clock_text(event["timestamp"])}</time><div><b>{html(event["actor"])}</b>'
                    f'<p>{html(chinese(event["summary"])[:80])}</p></div><span class="ev-dot" style="background:{STATE_COLORS.get(event["status"],"#718399")}"></span></div>')
    else:
        for event in events:
            with st.expander(f'{clock_text(event["timestamp"])}  {event["actor"]} · {chinese(event["summary"])[:46]}'):
                st.write(chinese(event["summary"]))
                st.write(STATE_NAMES.get(event["status"],event["status"]))
                st.json({k:event[k] for k in ("input_summary","output_summary","evidence","tool","artifact","evaluation")}, expanded=False)


def metric_delta(before, after, *, lower=False, ratio=False):
    if before is None or after is None:
        return "— 未测量", "neutral"
    if isinstance(before, bool):
        return ("通过" if after else "未通过"), ("good" if after else "warn")
    change = after - before
    if abs(change) < 1e-12:
        return "— 持平", "neutral"
    direction = "↑" if change > 0 else "↓"
    text = f"{abs(change)*100:.1f} 个百分点" if ratio else (f"{abs(change/before)*100:.1f}%" if before else f"{abs(change):g}")
    good = change < 0 if lower else change > 0
    return f"{direction} {text}", "good" if good else "warn"


METRICS = [("task_success","任务验收",False,False),("quality_score","规则评分",False,True),
           ("evidence_coverage","报告结论证据覆盖",False,True),("steps","执行步骤",True,False),
           ("retries","重试次数",True,False),("token_usage","模型用量（词元）",True,False),("latency","运行耗时",True,False)]


def value_text(value, key):
    if value is None:
        return "未测量"
    if isinstance(value,bool):
        return "通过" if value else "未完成"
    if key in {"quality_score","evidence_coverage"}:
        return f"{value:.1%}"
    if key=="latency":
        return f"{value:.2f} 秒"
    return f"{value:g}"


def render_comparison(comparison, *, detail=True):
    if not comparison:
        st.info("暂无可复现实验结果")
        return
    a,b = comparison["baseline"],comparison["evolved"]
    x,y = a["evaluation"],b["evaluation"]
    ca, total = evidence_progress(a); cb, _ = evidence_progress(b)
    st.html('<div class="ev-comparison-head"><div><span>初始团队</span>'
            f'<strong>{ca}<em> / {total} 项</em></strong><p>已完成验证</p></div>'
            '<div class="ev-evolved"><span>改进后团队</span>'
            f'<strong>{cb}<em> / {total} 项</em></strong><p>已完成验证</p></div></div>')
    if not detail:
        st.write(f'执行成员 {x["active_agent_count"]} → {y["active_agent_count"]} 个 · '
                 f'用时 {x["latency"]:.2f} → {y["latency"]:.2f} 秒')
        return
    st.write(f'两轮使用相同任务、数据和评价规则，随机种子为 {comparison["seed"]}。')
    rows=[]
    for key,label,lower,ratio in METRICS:
        delta,tone=metric_delta(x.get(key),y.get(key),lower=lower,ratio=ratio)
        rows.append(f'<div class="ev-metric-row"><span>{label}</span><b>{value_text(x.get(key),key)}</b>'
                    f'<b>{value_text(y.get(key),key)}</b><span class="ev-{tone}">{delta}</span></div>')
    st.html('<div class="ev-metric-table"><div class="ev-metric-row"><span>指标</span><span>初始团队</span><span>改进后团队</span><span>变化</span></div>'+''.join(rows)+'</div>')
    with st.expander("评分细则与计算用量"):
        labels = {"completeness": "输出与结构完整度", "evidence": "结论证据覆盖", "consistency": "事实一致性",
                  "task_quality": "任务专属规则", "reproducibility": "独立复核"}
        weights = a["task"]["evaluation_rules"]["weights"]
        st.dataframe(pd.DataFrame([{"组成项": labels[k], "权重": f"{w:.0%}",
                                   "初始团队": f'{x["dimensions"][k]:.3f}',
                                   "改进后团队": f'{y["dimensions"][k]:.3f}'}
                                  for k,w in weights.items()]), hide_index=True, width="stretch")
        st.write(f'规则评分为各项加权之和，通过阈值为 {a["task"]["evaluation_rules"]["success_threshold"]:.0%}；'
                 '还须必需证据齐全、无严重审查问题和未恢复错误。评分衡量任务完成情况，不表示行情预测准确率。')
        st.write(f'审查通过率 {x["critic_pass_rate"]:.1%} → {y["critic_pass_rate"]:.1%}；'
                 f'工具调用 {x["tool_calls"]} → {y["tool_calls"]} 次；执行成员 {x["active_agent_count"]} → {y["active_agent_count"]} 个。')
        if y.get("token_usage") == 0: st.write("本轮任务角色使用规则执行，未调用语言模型，词元用量为零。")
    tabs=st.tabs(["分工变化", "提示词变化", "工具变化"])
    with tabs[0]:
        left,right=st.columns(2)
        with left: render_graph(a)
        with right: render_graph(b)
    with tabs[1]:
        import difflib
        changes=[]
        for name,role in b["team_config"]["roles"].items():
            old=a["team_config"]["roles"].get(name,{})
            if role["system_prompt"]!=old.get("system_prompt"):
                changes.append(name)
                with st.expander(f'{ROLE_NAMES.get(name,name)} · {old.get("prompt_version","新角色")} → {role["prompt_version"]}'):
                    st.code('\n'.join(difflib.unified_diff(old.get("system_prompt","").splitlines(),role["system_prompt"].splitlines(),fromfile="初始提示词",tofile="改进后提示词",lineterm="")),language="diff")
        if not changes: st.write("本轮沿用已有提示词。")
    with tabs[2]:
        st.dataframe(pd.DataFrame([{"配置": label,
                "首选工具":chinese(t["team_config"]["tool_policy"]["executor"]["preferred"]),
                "复用相同输入结果":"启用" if t["team_config"]["tool_policy"]["executor"].get("cache") else "关闭"}
                for label,t in (("初始团队",a),("改进后团队",b))]),hide_index=True,width="stretch")
        with st.expander("工具配置原始记录"):
            st.json({"before":a["team_config"]["tool_policy"],"after":b["team_config"]["tool_policy"]},expanded=False)
    st.download_button("下载两轮实验记录",json.dumps(comparison,ensure_ascii=False,indent=2),"before_after.json","application/json",key="ev_compare_download")


def render_timeline(comparison, *, compact=False, history=None):
    if not comparison:
        st.info("暂无演进记录。运行实验后生成。")
        return
    a,b = comparison["baseline"],comparison["evolved"]
    histories = history if history is not None else comparison.get("evolution_history",[])
    actions=[action for h in histories for action in h.get("actions",[])]
    if not actions:
        st.write("本轮沿用已验证的团队配置。")
        return
    for action in actions:
        st.html(f'<div class="ev-change"><b>{html(ACTION_NAMES.get(action["action_type"],action["action_type"]))}</b>'
                f'<span>{html(action_summary(action))}</span></div>')
    if compact: return
    for h in histories:
        accepted=h.get("promotion",{}).get("accepted",False)
        st.write("复评后采用了这组配置。" if accepted else "复评后未采用这组配置。")
        with st.expander("修改原因与提示词原文"):
            for action in h.get("actions",[]):
                st.markdown("**" + ACTION_NAMES.get(action["action_type"],action["action_type"]) + "**")
                st.write(chinese(action["rationale"]))
                if action.get("diff"): st.code(action["diff"],language="diff")
                else: st.json({"before":action["before"],"after":action["after"]},expanded=False)
            st.json({k:h.get(k) for k in ("source_trace_ids","controller_trace","training_cost","promotion")},expanded=False)


def render_market_result(trace, key, *, height=280):
    artifacts=trace.get("artifacts",{})
    simulation=artifacts.get("simulation",{}).get("data",{})
    history=artifacts.get("historical_comparison",{}).get("data")
    counter=artifacts.get("counterfactual_search",{}).get("data")
    fig=go.Figure()
    if history:
        for name,column,color in (("上证指数观测","observed_normalized","#a9b8cc"),("条件回放","simulated_normalized","#43c5d0")):
            fig.add_trace(go.Scatter(x=history["dates"],y=history[column],name=name,line={"color":color,"width":2}))
        apply_dark_theme(fig,title="政策窗口走势 · 首日归一为 1",height=height)
        fig.update_xaxes(tickformat="%m月%d日", hoverformat="%Y年%m月%d日")
        st.plotly_chart(fig,width="stretch",key=key)
        st.write(f'观测区间 {history["dates"][0]} 至 {history["dates"][-1]}，共 {history["observations"]} 个交易日。回放使用前一日观测收盘价与合成宏观条件。')
    elif counter:
        plans=counter["plans"]
        st.dataframe(pd.DataFrame([{"方案":PLAN_NAMES.get(p["name"],p["name"]),
                    "最大回撤":f'{p["risk"]["max_drawdown"]:.2%}',"干预成本（预算单位）":p["cost"]}
                    for p in plans]),hide_index=True,width="stretch")
    elif simulation.get("prices"):
        for label,data,color in (("政策组",simulation,"#43c5d0"),("无政策对照组",artifacts.get("control",{}).get("data",{}),"#a9b8cc")):
            if data.get("prices"):
                fig.add_trace(go.Scatter(x=list(range(len(data["prices"]))),y=data["prices"],name=label,line={"color":color,"width":2}))
        apply_dark_theme(fig,title="模拟价格 · 政策组与对照组" if "control" in artifacts else "模拟价格 · 政策组",height=height)
        fig.update_xaxes(title="模拟交易日");fig.update_yaxes(title="价格（起始值 100）")
        st.plotly_chart(fig,width="stretch",key=key)
        st.write(f'市场撮合仿真 · {simulation.get("market_participants","—")} 个规则交易主体，{simulation.get("trades","—")} 笔成交。')
    else:
        st.info("当前阶段还没有市场结果。")
        return
    st.write(outcome_summary(trace))


def render_evidence(trace, key):
    tabs=st.tabs(["实验结果","审查意见","任务与证据","工具调用"])
    with tabs[0]:
        render_market_result(trace,key+"market")
        with st.expander("报告结论与适用范围"):
            for claim in trace.get("report",{}).get("conclusions",[]):
                text=chinese(claim["conclusion"].split("，来源 ")[0])
                st.markdown("**"+text+"**")
                st.write("依据："+"、".join(chinese(k) for k in claim["evidence"])+f'；证据完整度 {claim["confidence"]:.0%}。')
                if claim["evidence"]==["simulation"]:
                    st.write("使用规则交易主体与固定初始订单深度，宏观条件为合成假设。" if "market_participants" in claim["facts"] else "基于前一日观测价格的条件回放，宏观条件为合成假设。")
                else: st.write(chinese(claim["caveat"]))
            st.json(trace.get("report",{}),expanded=False)
    with tabs[1]:
        if trace.get("critiques"):
            st.dataframe(pd.DataFrame([{"问题":critique_label(c),"责任环节":chinese(c["attribution"]),"处理建议":chinese(c["recommendation"])} for c in trace["critiques"]]),hide_index=True,width="stretch")
        else: st.write("本轮必需验证已通过审查。")
    with tabs[2]:
        expected=trace.get("task",{}).get("expected_outputs",[])
        st.dataframe(pd.DataFrame([{"验证项":EVIDENCE_NAMES.get(k,k),"结果":"已完成" if valid_evidence(k,trace.get("artifacts",{}).get(k,{})) else "待补齐"} for k in expected]),hide_index=True,width="stretch")
        with st.expander("计划与证据原文"):
            st.json(trace.get("plan",{}),expanded=False)
            st.json(trace.get("artifacts",{}),expanded=False)
    with tabs[3]:
        st.dataframe(pd.DataFrame([{"成员":ROLE_NAMES.get(t.get("agent"),t.get("agent")),"工具":chinese(t.get("tool",t.get("name",""))),"状态":chinese(t.get("status","")),"耗时（秒）":t.get("latency")} for t in trace.get("tool_calls",[])]),hide_index=True,width="stretch")
        with st.expander("工具调用原始记录"): st.json(trace.get("tool_calls",[]),expanded=False)
    st.download_button("下载本轮执行记录",json.dumps(trace,ensure_ascii=False,indent=2),"run_trace.json","application/json",key=key+"download")
