"""Competition-first Streamlit workspace using real service/archived evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from core.competition_cockpit import (DEFAULT_COMPETITION_DEMO, DEMO_SCALES, PROFILES, DemoJobs, ExperimentArchive, capabilities,
                                      generations, reference_directory)
from core.team_evolution.benchmark import benchmark_tasks
from core.team_evolution.config import baseline_config
from core.team_evolution.provenance import source_provenance
from ui.components.evolution import (ACTION_NAMES, EVIDENCE_NAMES, ROLE_NAMES, critique_label, events_from_trace, html, render_comparison,
                                     render_events, render_evidence, render_graph, render_timeline, render_market_result, section)
from ui.presentation import (PROFILE_NAMES, TASK_TITLES, chinese, evidence_progress, outcome_summary)

TEAM_PAGES = ["自演进驾驶舱", "协作过程", "演进实验", "三类任务", "技术与复现"]
TASK_LABELS = {"policy_report": "政策研判", "historical_analysis": "历史事件分析", "regulatory_planning": "监管优化"}
TASK_ENGLISH = {"policy_report": "REPORT GENERATION", "historical_analysis": "DATA ANALYSIS", "regulatory_planning": "TASK PLANNING"}


@st.cache_resource
def get_jobs():
    return DemoJobs()


def navigate(page):
    st.session_state.entry = page
    st.query_params["page"] = page
    st.session_state.ev_scroll_to_top = True


def load_context():
    requested = st.query_params.get("artifact_run", "")
    if "ev_archive_dir" not in st.session_state and re.fullmatch(r"run-[a-f0-9]{32}", requested):
        # Restore a selected demo artifact using an identifier, never a URL-supplied filesystem path.
        demo_root = Path(__file__).resolve().parents[1] / "outputs" / "competition_demos"
        saved = next(demo_root.glob(f"demo-*/comparisons/{requested}/before_after.json"), None)
        if saved:
            st.session_state.ev_archive_dir = str(saved.parents[2])
    archive = ExperimentArchive(st.session_state.get("ev_archive_dir", reference_directory()))
    comparisons = archive.comparisons()
    sources = {c["baseline"]["run_id"]: str(archive.root) for c in comparisons}
    if archive.root != reference_directory().resolve():
        reference = ExperimentArchive()
        saved = reference.comparisons()
        comparisons += [c for c in saved if c["baseline"]["run_id"] not in sources]
        sources.update({c["baseline"]["run_id"]: str(reference.root) for c in saved})
    st.session_state.ev_comparison_sources = sources
    result = st.session_state.get("ev_result", st.session_state.get("qiyuan_result"))
    if result is None and comparisons:
        family = st.session_state.get("ev_task_family", "policy_report")
        requested = st.query_params.get("artifact_run")
        result = next((c for c in comparisons if c["baseline"]["run_id"] == requested), None)
        if result is None:
            result = next((c for c in comparisons if c["baseline"]["task"]["task_type"] == family), comparisons[0])
    if result and "ev_scale" not in st.session_state:
        days = selected_trace(result)["task"]["constraints"]["simulation_days"]
        st.session_state.ev_scale = next((k for k,v in DEMO_SCALES.items() if v==days), "答辩演示")
    return archive, comparisons, result


def load_reference():
    st.query_params.pop("demo", None)
    st.query_params.pop("artifact_run", None)
    for key in ("ev_result", "qiyuan_result", "ev_job_id", "ev_action_error"):
        st.session_state.pop(key, None)
    st.session_state.ev_archive_dir = str(reference_directory())
    st.session_state.ev_origin = "已完成实验"


def selected_trace(result):
    return result.get("evolved", result) if result else None


def idle_trace():
    return {"team_config": baseline_config().to_dict(), "task_id": "等待任务", "run_id": "not-run",
            "agent_events": [], "messages": [], "artifacts": {}, "evaluation": {}, "tool_calls": []}


def begin_run(task_type=DEFAULT_COMPETITION_DEMO["task_type"], action="compare", *, fresh=False, saved_trace=None):
    try:
        output_dir = None if fresh or action=="reproduce" else st.session_state.get("ev_work_dir")
        job_id = get_jobs().start(task_type=task_type, profile=st.session_state.get("ev_profile", "REPRODUCIBLE DEMO"),
                                  scale=st.session_state.get("ev_scale", "答辩演示"), action=action,
                                  output_dir=output_dir, saved_trace=saved_trace)
        st.session_state.ev_job_id = job_id
        st.query_params["demo"] = job_id
        st.query_params.pop("artifact_run", None)
        st.session_state.ev_task_family = task_type
        st.session_state.ev_action_error = None
    except Exception as exc:
        st.session_state.ev_action_error = f"实验未启动：{type(exc).__name__} · {exc}"


def job_snapshot():
    job_id=st.session_state.get("ev_job_id", st.query_params.get("demo"))
    if not job_id:
        return None
    try:
        job = get_jobs().snapshot(job_id)
        if "ev_job_id" not in st.session_state:
            st.session_state.ev_job_id = job_id
            st.session_state.ev_restored = True
            st.session_state.ev_profile = job["profile"]
            st.session_state.ev_scale = job["scale"]
        return job
    except KeyError:
        st.session_state.pop("ev_job_id",None)
        st.query_params.pop("demo", None)
        st.session_state.ev_action_error = "实验入口已失效；可加载已完成参考实验或重新运行。"
        return None


def consume_job():
    job=job_snapshot()
    if job and job["status"]=="COMPLETED" and st.session_state.get("ev_delivered")!=job["job_id"]:
        result=job["result"]
        if job["action"]=="reproduce":
            st.session_state.ev_reproduction=result
        else:
            st.session_state.ev_result=result
            st.session_state.ev_archive_dir=job["output_dir"]
            st.session_state.ev_origin="恢复已完成实验" if st.session_state.pop("ev_restored",False) or job.get("recovered") else "本次执行已完成"
            st.session_state.ev_work_dir=job["output_dir"]
        st.session_state.ev_delivered=job["job_id"]
        return True
    return False


@st.fragment(run_every=1)
def live_job_panel(expanded=True):
    job=job_snapshot()
    if not job:
        return
    if consume_job():
        st.rerun()
    if job["status"]=="FAILED":
        st.error(f'实验未完成 · {job["error"]["type"]}。已完成的实验仍可查看。')
        with st.expander("失败原因与恢复"):
            st.write(job["error"]["message"])
            st.caption("失败记录保存失败：" + job["error"]["artifact_write_error"] if "artifact_write_error" in job["error"] else "失败记录已保存为 demo_failure.json。")
            st.caption("可切换离线备用模式重新运行，或加载已完成案例。")
            st.button("加载已完成参考实验", on_click=load_reference, key="ev_failure_reference")
        return
    if job["status"]=="COMPLETED":
        if job["action"]=="reproduce":
            (st.success if job["result"]["semantic_match"] else st.warning)("复跑完成，实验产物一致" if job["result"]["semantic_match"] else "复跑产物存在差异，已保留记录。")
        return
    observations=job["observations"]
    types={o["event"] for o in observations}
    completed = [True, any(o["event"]=="run_saved" and o.get("trace",{}).get("mode")=="baseline" for o in observations),
                 "baseline_evaluated" in types, "evolution_started" in types, "candidate_proposed" in types,
                 any(o["event"]=="run_saved" and o.get("trace",{}).get("mode")=="evolved" for o in observations),
                 "comparison_saved" in types, False]
    labels=["加载任务", "初始团队执行", "审查评估", "读取反馈", "调整配置", "改进后重跑", "配对比较", "保存产物"]
    st.html('<div class="ev-live-label"><span class="ev-dot"></span>实验进行中 · '+html(PROFILE_NAMES[job["profile"]])+
            ' · '+html(TASK_LABELS[job["task"]["task_type"]])+'</div>')
    if job["action"]=="compare":
        st.progress(sum(completed)/len(completed),text="正在完成两轮对照")
        st.html('<div class="ev-steps">'+''.join(f'<span class="{"done" if done else ""}">{i+1:02d} {html(label)}</span>' for i,(label,done) in enumerate(zip(labels,completed)))+'</div>')
    if expanded:
        traces=[o["trace"] for o in observations if o.get("trace") and o["trace"].get("mode")!="evolution"]
        trace=traces[-1] if traces else idle_trace()
        left,right=st.columns([1.65,1])
        with left:
            st.markdown("#### 当前团队")
            controller = next((o.get("controller") for o in reversed(observations) if o["event"]=="evolution_started"),None)
            if "candidate_proposed" in types and controller: controller=controller|{"status":"completed"}
            render_graph(trace,controller=controller)
        with right:
            st.markdown("#### 执行记录")
            render_events(trace,compact=True)
        proposed=next((o for o in reversed(observations) if o["event"]=="candidate_proposed"),None)
        if proposed:
            changes=proposed["history"]["actions"]
            st.caption("候选调整："+" · ".join(ACTION_NAMES.get(a["action_type"],a["action_type"]) for a in changes))


def provenance_strip(archive, result):
    trace=selected_trace(result)
    if not trace: return
    origin=st.session_state.get("ev_origin","已完成实验")
    started=trace.get("agent_events", [{}])[0].get("started_at", "")
    from datetime import datetime
    try: date=datetime.fromisoformat(started).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError: date="时间未记录"
    st.html(f'<div class="ev-provenance"><b>{html(origin)}</b><span>{html(date)}</span>'
            f'<span>{trace["task"]["constraints"]["simulation_days"]} 个模拟交易日</span></div>')


def cockpit(archive, comparisons, result):
    caps=capabilities()
    job=job_snapshot()
    running=bool(job and job["status"] in {"QUEUED","RUNNING"})
    trace=selected_trace(result) or idle_trace()
    comparison=result if result and "baseline" in result else None
    runtime=trace.get("runtime", {}) if result else archive.runtime(st.session_state.get("ev_profile","REPRODUCIBLE DEMO"))
    if running:
        runtime=next((o["trace"]["runtime"] for o in reversed(job["observations"])
                      if o.get("trace",{}).get("runtime",{}).get("backend")),
                     archive.runtime(job["profile"]))
    backend={"OPENJIUWEN":"openJiuwen 工作流", "LOCAL FALLBACK":"本地备用工作流"}.get(runtime.get("backend"),"运行环境待检查")
    st.html('<div class="ev-hero"><h1>自演进多智能体 · 金融政策数字风洞</h1>'
            '<p>团队完成政策任务，发现遗漏后调整分工、提示词和工具，再用同一任务复评。</p></div>')
    st.html('<div class="ev-capabilities">'+''.join(f'<span>{html(v)}</span>' for v in (
        f'{caps["role_count"]} 类基础角色', f'{caps["task_count"]} 类实验任务',
        f'{caps["mechanism_count"]} 类配置调整', backend))+'</div>')
    task_type=trace.get("task",{}).get("task_type",DEFAULT_COMPETITION_DEMO["task_type"])
    if running: task_type=job["task"]["task_type"]
    st.markdown("### "+TASK_TITLES.get(task_type,"当前实验"))
    if not running: provenance_strip(archive,result)
    buttons=st.columns([1.2,1.2,1.8])
    buttons[0].button("查看任务与审查记录",type="primary",width="stretch",on_click=navigate,args=("协作过程",),key="ev_to_collaboration",disabled=not result)
    buttons[1].button("重新运行此案例",width="stretch",key="ev_demo",disabled=running,
                      on_click=begin_run,kwargs={"fresh":True,"task_type":task_type})
    with buttons[2].popover("运行设置",width="stretch"):
        st.radio("运行方式",list(PROFILES),format_func=lambda x:PROFILE_NAMES[x],key="ev_profile",disabled=running)
        st.radio("实验规模",list(DEMO_SCALES),key="ev_scale",disabled=running,
                 format_func=lambda x:f"{x} · {DEMO_SCALES[x]} 天")
    if st.session_state.get("ev_action_error"): st.error(st.session_state.ev_action_error)
    if running:
        live_job_panel()
        return
    if job_snapshot() and job_snapshot()["status"] == "FAILED": live_job_panel(expanded=False)
    if comparison:
        a,b=comparison["baseline"],comparison["evolved"]
        ca,total=evidence_progress(a); cb,_=evidence_progress(b)
        st.html('<div class="ev-proof-strip"><div><small>初轮审查</small><b>'+html("；".join(critique_label(c) for c in a["critiques"]) or "未发现缺失项")+
                '</b></div><div><small>改进后验证</small><b>'+f'{ca}/{total} 项 → {cb}/{total} 项'+
                '</b></div><div><small>实际用时</small><b>'+f'{a["latency"]:.2f} 秒 → {b["latency"]:.2f} 秒</b></div></div>')
    left,right=st.columns([1.65,1],gap="large")
    with left:
        st.markdown("### 团队分工")
        history=comparison.get("evolution_history",[]) if comparison else []
        controller=history[-1]["controller_trace"] if history else None
        render_graph(trace,controller=controller)
    with right:
        st.markdown("### 本轮执行记录")
        render_events(trace,compact=True)
    section("", "团队做了哪些调整")
    render_timeline(comparison,compact=True)
    st.button("查看两轮对照与提示词变化",on_click=navigate,args=("演进实验",),width="stretch",key="ev_to_evolution")
    if result:
        section("", "任务产出")
        render_market_result(trace,"ev_home_result")
    else: st.info("暂无可复现实验结果。运行一个案例即可开始。")


def collaboration_page(archive, comparisons, result):
    section("", "任务执行与审查")
    if not result:
        st.info("暂无可复现实验结果")
        return
    st.write(TASK_TITLES.get(selected_trace(result)["task"]["task_type"],"当前实验"))
    options={"初始团队":result["baseline"],"改进后团队":result["evolved"]} if "baseline" in result else {"当前运行":result}
    if result.get("candidate"): options["候选配置复评"]=result["candidate"]
    choice=st.radio("执行阶段",list(options),horizontal=True,key="ev_trace_phase")
    trace=options[choice]
    provenance_strip(archive,trace)
    events=events_from_trace(trace)
    if trace.get("critiques"):
        st.info("审查发现：" + "；".join(critique_label(c) for c in trace["critiques"]))
    controls=st.columns(2)
    controls[0].button("下一步：查看团队如何调整",on_click=navigate,args=("演进实验",),type="primary",width="stretch")
    controls[1].button("按本轮配置复跑",key="ev_reproduce_trace",on_click=begin_run,
                       kwargs={"task_type":trace["task"]["task_type"],"action":"reproduce","saved_trace":trace},width="stretch")

    with st.container(border=True):
        render_graph(trace,inspect=True)
    if trace.get("errors"): st.warning(f'此运行保留 {len(trace["errors"])} 条执行错误。')
    tabs=st.tabs(["协作事件流","时间线回放","报告与证据"])
    with tabs[0]:
        kinds=["全部"]+sorted({e["kind"] for e in events})
        selected=st.selectbox("事件类型",kinds,key="ev_event_kind",format_func=lambda k:{"agent":"成员执行","delegation":"任务分派","message":"消息传递","review":"送交审查","result":"结果移交","feedback":"审查反馈"}.get(k,k))
        render_events(trace,rows=[e for e in events if selected=="全部" or e["kind"]==selected])
    with tabs[1]:
        if events:
            cursor=st.slider("查看第几条记录",1,len(events),len(events),key="ev_cursor_"+trace["run_id"])
            event=events[cursor-1]
            replay=deepcopy(trace)
            at=event["timestamp"]
            replay["agent_events"]=[e for e in trace["agent_events"] if e.get("started_at","")<=at]
            for item in replay["agent_events"]:
                if item.get("finished_at","")>at: item["status"]="running"
            replay["evaluation"]={} if cursor<len(events) else trace["evaluation"]
            render_graph(replay)
            render_events(trace,rows=[event])
    with tabs[2]: render_evidence(trace,"ev_trace_")


def memory_panel(archive, trace):
    rows=[r for r in archive.experiences() if r["task_type"]==trace["task"]["task_type"]]
    if not rows:
        st.info("还没有此类任务的经验记录。运行一次实验后会保存。")
        return
    references=trace.get("plan",{}).get("retrieved_experience_ids",[])
    st.dataframe(pd.DataFrame([{"记录":i+1,"验证结果":"通过" if r["final_score"]["task_success"] else "未完成",
        "成员数量":len(r["team_configuration"]["topology"]["active_agents"]),
        "首选工具":chinese(r["team_configuration"]["tool_policy"]["executor"]["preferred"]),
        "复用次数":r["times_reused"],"本轮采用":"是" if r["experience_id"] in references else "—",
        "问题":"；".join(critique_label(c) for c in r["critic_feedback"]) or "无遗漏"}
        for i,r in enumerate(rows)]),hide_index=True,width="stretch")
    default=next((i for i,r in enumerate(rows) if r["experience_id"] in references),0)
    selected=st.selectbox("查看经验记录",range(len(rows)),index=default,
                          format_func=lambda i:f'记录 {i+1} · 已复用 {rows[i]["times_reused"]} 次')
    row=rows[selected]
    used=row["experience_id"] in references
    if used:
        st.write(f'规划员读取了这条经验，采用 {len(row["team_configuration"]["topology"]["active_agents"])} 个成员的团队，'
                 f'优先使用{chinese(row["team_configuration"]["tool_policy"]["executor"]["preferred"])}。')
    else: st.write("这条经验未用于当前运行。")
    with st.expander("经验内容与引用记录"):
        st.json({"experience":row,"current_planner_references":references,
                 "current_team":trace["team_config"]["topology"]},expanded=False)


def evolution_page(archive, comparisons, result):
    section("", "从初轮审查到再次验证")
    provenance_strip(archive,result)
    if not result or "baseline" not in result:
        st.info("暂无可复现实验结果。请先运行两轮对照。")
        return
    st.write(TASK_TITLES[result["baseline"]["task"]["task_type"]])
    tabs=st.tabs(["调整记录","两轮对照","组织变化","历史经验"])
    with tabs[0]:
        baseline=result["baseline"]
        if baseline["critiques"]:
            st.markdown("**初轮遗漏了什么**")
            for critique in baseline["critiques"]:
                st.write(f'• {critique_label(critique)} → {chinese(critique["recommendation"])}')
        render_timeline(result)
        render_comparison(result,detail=False)
        extra=[h for h in archive.history() if h["evolution_id"] not in {x["evolution_id"] for x in result["evolution_history"]}
               and any(rid==result["baseline"]["run_id"] or rid in {t["run_id"] for c in comparisons if c["task_id"]==result["task_id"] for t in (c["baseline"],c["evolved"])} for rid in h["source_trace_ids"])]
        if extra:
            with st.expander(f"同任务的其他 {len(extra)} 条演进记录"):
                render_timeline(result,history=extra)
    with tabs[1]: render_comparison(result)
    with tabs[2]:
        rows=generations(comparisons+[result],result["baseline"]["task"])
        st.dataframe(pd.DataFrame([{"Generation":r["generation"],"Team Version":r["config"]["version"],
                                    "执行成员数":len(r["config"]["topology"]["active_agents"]),"规则评分":f'{r["quality"]:.1%}',
                                    "词元用量":r["trace"]["evaluation"]["token_usage"],"耗时（秒）":r["trace"]["latency"]} for r in rows]),hide_index=True,width="stretch")
        if rows:
            columns=st.columns(2)
            for index,column in enumerate(columns):
                with column:
                    selected=st.selectbox("比较起点" if index==0 else "比较终点",range(len(rows)),index=0 if index==0 else len(rows)-1,
                                           format_func=lambda i:f'第 {rows[i]["generation"]} 代 · 配置 {rows[i]["config"]["version"]}',key=f"ev_generation_{index}")
                    row=rows[selected]
                    render_graph(row["trace"])
                    with st.expander("提示词、工具配置与用量"):
                        st.json({"prompt_versions":{k:v["prompt_version"] for k,v in row["config"]["roles"].items()},
                                 "tool_policy":row["config"]["tool_policy"],"score":row["trace"]["evaluation"],"currency_cost":None},expanded=False)
    with tabs[3]: memory_panel(archive,result["evolved"])
    st.button("下一步：查看其他实验案例",on_click=navigate,args=("三类任务",),width="stretch")


def show_artifact(comparison, page):
    st.query_params.pop("demo", None)
    st.query_params["artifact_run"] = comparison["baseline"]["run_id"]
    st.session_state.pop("ev_job_id", None)
    st.session_state.ev_result=comparison
    st.session_state.ev_origin="已完成实验"
    st.session_state.ev_task_family=comparison["baseline"]["task"]["task_type"]
    source = st.session_state.get("ev_comparison_sources", {}).get(comparison["baseline"]["run_id"])
    if source:
        st.session_state.ev_archive_dir = source
    navigate(page)


def tasks_page(archive, comparisons, result):
    section("", "实验案例")
    columns=st.columns(3,gap="medium")
    for index,(task,column) in enumerate(zip(benchmark_tasks(),columns)):
        with column,st.container(border=True):
            st.html(f'<div class="ev-task-card"><h2>{html(TASK_LABELS[task.task_type])}</h2>'
                    f'<p>{html(task.objective)}</p></div>')
            available=[c for c in comparisons if c["baseline"]["task"]["task_type"]==task.task_type]
            if result and "baseline" in result and result["baseline"]["task"]["task_type"]==task.task_type: available=[result]+available
            comparison=available[0] if available else None
            if comparison:
                a,b=comparison["baseline"],comparison["evolved"]
                ca,total=evidence_progress(a); cb,_=evidence_progress(b)
                st.markdown(f'**完成验证 {ca}/{total} → {cb}/{total} 项**')
                st.html('<div class="ev-case-outcome">'+html(outcome_summary(b))+'</div>')
            else: st.info("暂无可复现实验结果")
            st.button("查看案例",key="ev_archive_"+task.task_type,width="stretch",disabled=not comparison,
                      on_click=show_artifact,args=(comparison,"协作过程"))
            with st.expander("重新运行"):
                st.button("运行两轮对照",key="ev_compare_"+task.task_type,width="stretch",on_click=begin_run,kwargs={"task_type":task.task_type})
                st.button("仅运行初始团队",key="ev_baseline_"+task.task_type,width="stretch",on_click=begin_run,kwargs={"task_type":task.task_type,"action":"baseline"})
                st.button("使用当前经验库运行",key="ev_evolved_"+task.task_type,width="stretch",on_click=begin_run,kwargs={"task_type":task.task_type,"action":"evolved"})
                st.button("按案例配置复跑",key="ev_reproduce_"+task.task_type,width="stretch",disabled=not comparison,on_click=begin_run,
                          kwargs={"task_type":task.task_type,"action":"reproduce","saved_trace":comparison["evolved"] if comparison else None})
    if comparisons:
        with st.expander("全部实验记录"):
            st.dataframe(pd.DataFrame([{"任务":TASK_LABELS[c["baseline"]["task"]["task_type"]],
                        "随机种子":c["seed"],"模拟天数":c["baseline"]["task"]["constraints"]["simulation_days"],
                        "初始评分":f'{c["baseline"]["evaluation"]["quality_score"]:.1%}',
                        "改进后评分":f'{c["evolved"]["evaluation"]["quality_score"]:.1%}',
                        "经验引用":len(c["evolved"]["plan"].get("retrieved_experience_ids",[]))}
                        for c in comparisons]),hide_index=True,width="stretch")
    st.button("继续探索金融政策风洞",on_click=navigate,args=("金融政策风洞",),width="stretch")


def technology_page(archive, comparisons, result):
    section("RUNTIME / REPRODUCIBILITY", "运行环境与实验资料", "区分已安装、可执行与本次实际完成的运行状态。")
    runtime=archive.runtime(st.session_state.get("ev_profile","REPRODUCIBLE DEMO"))
    trace=selected_trace(result)
    info=[("版本",runtime.get("package_version") or "未安装"),("所选执行后端",runtime["backend"]),
          ("环境状态",{"READY":"可用","FALLBACK":"本地备用"}.get(runtime.get("status"),"不可用")),("机制","工作流依赖调度" if runtime.get("sdk_imported") else "本地依赖调度")]
    st.html('<div class="ev-badges">'+''.join(f'<div><span>{html(k)}</span><strong>{html(v)}</strong></div>' for k,v in info)+'</div>')
    if trace:
        st.write(f'选中运行：{trace["runtime"]["backend"]} · 工作流完成：{"是" if trace["runtime"]["workflow_completed"] else "否"}')
        if result.get("evolution_history"):
            opt=result["evolution_history"][-1]["optimizer"]
            st.write("提示词优化：" + ("本地规则" if opt["backend"]=="LOCAL FALLBACK" else opt["backend"]))
    with st.expander("真实调用位置与运行详情"):
        st.code("openjiuwen.core.workflow.Workflow\nWorkflowComponent.invoke → Agent handler\nWorkflow.invoke(inputs, create_workflow_session(...))\nFeedbackPromptBuilder.build → optional cloud Prompt optimization",language="python")
        st.json({"selected_runtime":runtime,"recorded_runtime":trace.get("runtime") if trace else None},expanded=False)
    caps=capabilities()
    observed=bool(trace and trace.get("agent_events"))
    requirements=[("至少 3 类功能角色",f'{caps["role_count"]} 类基础角色，任务按需添加专家',"团队分工图",observed),
                  ("通信 / 分工 / 调度","任务分解、消息移交与依赖调度","协作记录",bool(trace and trace.get("messages"))),
                  ("提示词与工具调整","提示词版本管理与工具选择","调整记录",any(h.get("actions") for h in archive.history())),
                  ("数量 / 角色动态调整","增加角色、拆分任务与并行执行","两轮团队对照",any(c["baseline"]["team_config"]["topology"]!=c["evolved"]["team_config"]["topology"] for c in comparisons)),
                  ("基于历史表现改进","经验存储与任务规划检索","历史经验",any(c["evolved"]["plan"].get("retrieved_experience_ids") for c in comparisons)),
                  ("优化前后对比","相同任务、种子、数据和评价规则","两轮对照",bool(comparisons)),
                  ("3 类复杂任务",f'{caps["task_count"]} 类固定实验任务',"实验案例",len({c["task_id"] for c in comparisons})>=caps["task_count"]),
                  ("结构化与可解释","任务计划、审查意见与报告证据","任务与证据",bool(trace and trace.get("report")))]
    st.markdown("### 功能与运行记录")
    st.dataframe(pd.DataFrame([{"赛题要求":r,"系统实现":i,"现场证据":e,"当前归档": "有执行证据" if ok else "等待运行"} for r,i,e,ok in requirements]),hide_index=True,width="stretch")
    st.markdown("### 本地复现")
    if result and "baseline" in result:
        source = Path(st.session_state.get("ev_comparison_sources", {}).get(result["baseline"]["run_id"], archive.root))
        bundle = ExperimentArchive(source).read(str(Path("comparisons") / result["baseline"]["run_id"] / "competition_demo_summary.json"))
        if bundle:
            from core.team_evolution.evidence_bundle import bundle_zip
            try:
                data = bundle_zip(source, bundle)
                st.write("实验编号：" + bundle["experiment_id"])
                st.download_button("下载完整实验证据包", data, bundle["experiment_id"] + ".zip", "application/zip", key="ev_bundle_download")
            except (OSError, ValueError, KeyError):
                st.warning("该实验证据包校验未通过；可保留现有记录并重新运行实验。")
    st.button("加载已完成参考实验", on_click=load_reference, key="ev_load_reference")
    st.code(".venv/bin/python scripts/run_competition_demo.py --profile reproducible --task policy_report\n"
            ".venv/bin/python scripts/run_competition_benchmark.py --backend openjiuwen --rounds 3 --output outputs/competition_cockpit_reference\n"
            ".venv/bin/python -m pytest -q",language="bash")
    current=source_provenance()
    recorded=trace["runtime"].get("source_hash") if trace else None
    st.caption("源码版本匹配，可以按记录精确复跑。" if recorded==current["source_hash"] else "选中归档与当前源码版本不同；可查看归档，精确复跑需对应源码，或执行新实验。")
    with st.expander("版本与数据来源"):
        st.json({"artifact_directory":str(archive.root),"current_source_hash":current["source_hash"],"recorded_source_hash":recorded,
             "dataset_hash":trace["runtime"].get("dataset_hash") if trace else None},expanded=False)
    for name in ("summary.json","before_after.json","run_trace.json","evolution_history.json","team_topology.json","source_manifest.json"):
        value=archive.read(name)
        if value is not None:
            st.download_button("下载 "+name,json.dumps(value,ensure_ascii=False,indent=2),name,"application/json",key="ev_file_"+name)
    with st.expander("当前实现边界"):
        st.write("角色规划、评审和报告使用确定性规则。云端提示词优化需配置密钥，缺少密钥时使用本地规则。"
                 "尚不支持自动移除、合并或生成任意新角色。现有对照来自固定任务，尚未完成泛化与逐因素消融实验。")


def render_team_evolution(page="自演进驾驶舱"):
    consume_job()
    archive,comparisons,result=load_context()
    for issue in archive.issues:
        st.warning("部分归档不可用：" + issue)
    job = job_snapshot()
    if page!="自演进驾驶舱" and job and job["status"] != "COMPLETED":
        live_job_panel(expanded=False)
    if page=="技术与复现" and job and job["status"] == "COMPLETED" and job["action"] == "reproduce":
        (st.success if job["result"]["semantic_match"] else st.warning)("复跑完成，实验产物一致" if job["result"]["semantic_match"] else "复跑产物存在差异，已保留记录。")
        with st.expander("复跑证据"):
            st.write({"original": job["result"]["original_run_id"], "replay": job["result"]["rerun"]["run_id"]})
            st.download_button("下载复跑结果",json.dumps(job["result"],ensure_ascii=False,indent=2),"reproduction.json","application/json",key="ev_reproduction_download")
    if page!="自演进驾驶舱" and st.session_state.get("ev_action_error"): st.error(st.session_state.ev_action_error)
    renderer={"自演进驾驶舱":cockpit,"协作过程":collaboration_page,"演进实验":evolution_page,
              "三类任务":tasks_page,"技术与复现":technology_page}[page]
    renderer(archive,comparisons,result)
