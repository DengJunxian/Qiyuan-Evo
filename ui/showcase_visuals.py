"""Interactive presentation of verified, recorded cases; never runs a simulation."""
from html import escape
import json

from core.competition_cockpit import ExperimentArchive
from core.team_evolution.evaluation import valid_evidence
from ui.presentation import EVIDENCE_NAMES, PLAN_NAMES, ROLE_NAMES, TOOL_NAMES, evidence_progress

CASE_LABELS = {"policy_report": "政策评估", "historical_analysis": "历史回放", "regulatory_planning": "干预比较"}
CASE_TITLES = {"policy_report": "一次政策评估，如何变得完整。", "historical_analysis": "让专业角色，校验历史的轨迹。", "regulatory_planning": "在多种选择中，看清干预的代价。"}
ROLE_DESCRIPTIONS = {"planner": "组织任务与核验计划", "analyst": "解析政策与作用渠道", "executor": "执行仿真或历史回放", "critic": "审查证据与报告一致性", "reporter": "汇总证据，形成报告", "verifier": "独立复跑，核对结果", "historian": "对照已观测的市场行情", "quant": "计算量化指标", "counterfactual": "比较同条件下的干预方案"}


def presentation_cases():
    """Optional read-only evidence: archive failure must not take down the gallery."""
    try:
        return [case for case in ExperimentArchive().comparisons()
                if case["baseline"]["task"]["task_type"] in CASE_LABELS]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        return []


def chart_data(case):
    artifacts = case["evolved"]["artifacts"]
    kind = case["baseline"]["task"]["task_type"]
    if kind == "historical_analysis":
        h = artifacts["historical_comparison"]["data"]
        return {"labels": h["dates"], "series": [
            {"name": "观测行情", "values": [n * 100 for n in h["observed_normalized"]]},
            {"name": "条件回放", "values": [n * 100 for n in h["simulated_normalized"]]}],
            "unit": "归一化指数 · 首日 = 100", "title": "历史与回放，同一张坐标。",
            "metric": f'{h["direction_consistency"]:.1%}', "metric_label": "方向一致率",
            "note": f'{h["dates"][0]}—{h["dates"][-1]} · 上证指数，{len(h["dates"])} 个交易日。条件回放使用前期观测价与合成宏观条件，不代表样本外预测。'}
    if kind == "regulatory_planning":
        c = artifacts["counterfactual_search"]["data"]
        series = [{"name": PLAN_NAMES.get(p["name"], p["name"]), "values": p["prices"]} for p in c["plans"]]
        equal_risk = len({plan["risk"]["max_drawdown"] for plan in c["plans"]}) == 1
        return {"labels": [f'第 {i} 步' for i in range(len(series[0]["values"]))], "series": series,
                "unit": "仿真价格", "title": "三条路径，放在一起比较。",
                "metric": f'{c["risk_reduction"] * 100:.2f}', "metric_label": "最大回撤降低 / 百分点",
                "note": f'同种子、同预算比较；本例推荐{PLAN_NAMES.get(c["recommended"], c["recommended"])}。目标为最大回撤 + 0.001 × 归一化成本。' + ('各方案的最大回撤相同。' if equal_risk else '')}
    simulation, control = artifacts["simulation"]["data"], artifacts["control"]["data"]
    return {"labels": [f'第 {i} 步' for i in range(len(simulation["prices"]))],
            "series": [{"name": "政策组", "values": simulation["prices"]}, {"name": "无政策对照", "values": control["prices"]}],
            "unit": "仿真价格", "title": "有了对照，变化才有参照。",
            "metric": f'{control["return_difference"] * 100:+.2f}', "metric_label": "期末收益差 / 百分点",
            "note": f'固定种子下的 {len(simulation["prices"])-1} 步仿真，初始价格 {simulation["prices"][0]:g}。' + ('本例两组期末收益相同，过程路径不同；' if control["return_difference"] == 0 else '') + '图中为仿真实验结果。'}


def _attr(value):
    return escape(json.dumps(value, ensure_ascii=False, allow_nan=False), quote=True)


def _graph(trace, previous):
    topology = trace["team_config"]["topology"]
    added = set(topology["active_agents"]) - set(previous["team_config"]["topology"]["active_agents"])
    nodes = []
    for role in topology["active_agents"]:
        events = [event for event in trace["agent_events"] if event["agent"] == role]
        completed = bool(events) and all(event["status"] == "completed" for event in events)
        tools = list(dict.fromkeys(tool for event in events for tool in event.get("tools", [])))
        detail = ROLE_DESCRIPTIONS.get(role, ROLE_NAMES.get(role, role))
        if tools:
            detail += " · 调用：" + "、".join(TOOL_NAMES.get(tool, tool) for tool in tools)
        label = ROLE_NAMES.get(role, role)
        nodes.append(f'<button type="button" class="qy-node{" is-added" if role in added else ""}" data-node="{escape(role)}" '
                     f'data-detail="{escape(detail, quote=True)}" aria-label="查看{escape(label)}职责" aria-pressed="false">'
                     f'<span class="qy-node-mark">{"＋" if role in added else "✓" if completed else "−"}</span><strong>{escape(label)}</strong>'
                     f'<small>{"新增角色" if role in added else "已完成分工" if completed else "未完成分工"}</small></button>')
    return (f'<div class="qy-team-map" data-edges="{_attr(topology["communication_edges"])}" '
            f'data-specialists="{len(topology["active_agents"]) - 5}">' + ''.join(nodes) + '</div>'
            '<p class="qy-node-detail" aria-live="polite">连线表示任务移交。选择角色，查看职责与工具。</p>')


def visual_story(cases, url_builder):
    if not cases:
        return "", ""
    # Keep the first recorded pair per task family, following the reference selection.
    unique = {}
    for case in cases:
        unique.setdefault(case["baseline"]["task"]["task_type"], case)
    tabs, stages, proofs = [], [], []
    for index, (kind, case) in enumerate(unique.items()):
        before, after = case["baseline"], case["evolved"]
        selected = index == 0
        hidden = '' if selected else ' hidden'
        tabs.append(f'<button type="button" id="qy-tab-{kind}" role="tab" data-case="{kind}" '
                    f'aria-selected="{str(selected).lower()}" aria-controls="qy-case-{kind}" tabindex="{0 if selected else -1}">{CASE_LABELS[kind]}</button>')
        modes = []
        for mode, trace in (("baseline", before), ("evolved", after)):
            count, total = evidence_progress(trace)
            checks = ''.join(f'<span class="qy-check {"is-complete" if valid_evidence(key, trace["artifacts"].get(key, {})) else "is-missing"}">'
                             f'<i aria-hidden="true">{"✓" if valid_evidence(key, trace["artifacts"].get(key, {})) else "−"}</i>'
                             f'{EVIDENCE_NAMES.get(key, key)}<span class="qy-sr">：{"已提供" if valid_evidence(key, trace["artifacts"].get(key, {})) else "待补充"}</span></span>'
                             for key in trace["task"]["expected_outputs"])
            modes.append(f'<div data-mode-panel="{mode}"{" hidden" if mode == "baseline" else ""}>' + _graph(trace, before)
                         + f'<div class="qy-proof-strip"><span>任务证据 <b>{count} / {total}</b></span><div>{checks}</div></div></div>')
        stages.append(f'<div id="qy-case-{kind}" class="qy-case-panel" role="tabpanel" aria-labelledby="qy-tab-{kind}" data-case-panel="{kind}"{hidden}>'
                      f'<div class="qy-case-heading"><div><h2>{CASE_TITLES[kind]}</h2><p>已完成实验 · 种子 {case["seed"]} · {CASE_LABELS[kind]}</p></div>'
                      '<div class="qy-segment" role="group" aria-label="团队版本"><button type="button" data-mode="baseline" aria-pressed="false">演进前</button>'
                      '<button type="button" data-mode="evolved" aria-pressed="true">演进后</button></div></div>' + ''.join(modes) + '</div>')
        data = chart_data(case)
        before_count, total = evidence_progress(before)
        after_count, _ = evidence_progress(after)
        added = set(after["team_config"]["topology"]["active_agents"]) - set(before["team_config"]["topology"]["active_agents"])
        addition = "、".join(ROLE_NAMES.get(role, role) for role in sorted(added))
        qualities = [f'{trace["evaluation"]["quality_score"] * 100:.1f}'.removesuffix('.0') for trace in (before, after)]
        url = escape(url_builder({"page": "演进实验", "baseline_run_id": before["run_id"], "view": "compare"}), quote=True)
        readout = ''.join(f'<span><i class="qy-line-key key-{i}"></i>{escape(s["name"])} <b data-series-value="{i}">{s["values"][-1]:.2f}</b></span>' for i, s in enumerate(data["series"]))
        # The data table is a usable native fallback and makes every plotted point inspectable.
        table = '<table><thead><tr><th>日期 / 步数</th>' + ''.join(f'<th>{escape(s["name"])}</th>' for s in data["series"]) + '</tr></thead><tbody>'
        for i, label in enumerate(data["labels"]):
            table += f'<tr><td>{escape(label)}</td>' + ''.join(f'<td>{s["values"][i]:.4f}</td>' for s in data["series"]) + '</tr>'
        table += '</tbody></table>'
        proofs.append(f'<div class="qy-proof-panel" data-proof-panel="{kind}"{hidden}>'
                      '<div class="qy-quality"><span class="qy-overline">规则质量评分</span>'
                      f'<div class="qy-score-pair"><div><small>演进前</small><strong>{qualities[0]}<em>%</em></strong></div><span aria-hidden="true">↗</span>'
                      f'<div><small>演进后</small><strong>{qualities[1]}<em>%</em></strong></div></div>'
                      f'<p>加入{escape(addition)}，将有效任务证据从 {before_count}/{total} 补齐至 {after_count}/{total}。</p>'
                      '<dl><div><dt>参与角色</dt>'
                      f'<dd>{before["evaluation"]["active_agent_count"]} → {after["evaluation"]["active_agent_count"]}</dd></div><div><dt>工具调用</dt>'
                      f'<dd>{before["evaluation"]["tool_calls"]} → {after["evaluation"]["tool_calls"]}</dd></div><div><dt>记录耗时</dt>'
                      f'<dd>{before["evaluation"]["latency"]:.2f} → {after["evaluation"]["latency"]:.2f} 秒</dd></div></dl>'
                      '<p class="qy-method-note">评分衡量本任务的完成度与证据质量。</p>'
                      f'<a class="qy-text-link" href="{url}" target="_self">查看完整实验 ↗</a></div>'
                      f'<div class="qy-market" data-chart="{_attr(data)}"><div class="qy-chart-heading"><h3>{data["title"]}</h3>'
                      f'<span>{data["unit"]}</span></div><div class="qy-chart-readout">{readout}</div>'
                      f'<div class="qy-chart-canvas" role="img" aria-label="{escape(data["title"])}，{escape(data["unit"])}；各点数值可通过下方滑块与数据表查看。"></div>'
                      f'<div class="qy-chart-scrub"><label for="qy-point-{kind}">逐点查看 <output data-chart-date>{escape(data["labels"][-1])}</output></label>'
                      f'<input id="qy-point-{kind}" type="range" min="0" max="{len(data["labels"])-1}" value="{len(data["labels"])-1}" step="1" aria-label="{CASE_LABELS[kind]}：查看数据点"></div>'
                      f'<div class="qy-market-conclusion"><strong>{data["metric"]}<small>{data["metric_label"]}</small></strong><p>{data["note"]}</p></div>'
                      f'<details class="qy-chart-table"><summary>查看图表数据</summary>{table}</details></div></div>')
    scene = ('<section class="qy-observatory" id="collaboration" aria-label="交互式协作演示"><div class="qy-case-tabs" role="tablist" aria-label="选择演示案例">'
             + ''.join(tabs) + '</div><div class="qy-stage">' + ''.join(stages) + '</div></section>')
    proof = ('<section class="qy-evidence-story" aria-labelledby="qy-evidence-title"><header class="qy-section-head"><div>'
             '<h2 id="qy-evidence-title">变化，交给证据来回答。</h2><p>从团队调整，到市场结果。沿着同一个案例，看见完整的前因后果。</p></div></header>'
             + ''.join(proofs) + '</section>')
    return scene, proof
