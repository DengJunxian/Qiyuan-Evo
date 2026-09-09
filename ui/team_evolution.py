"""Competition-first Streamlit workspace using real service/archived evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import os
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

TEAM_PAGES = ["自演进驾驶舱", "协作过程", "演进实验", "三类任务", "技术与复现", "展示页面"]
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
    if "ev_profile" not in st.session_state:
        key=os.environ.get("QIYUAN_LLM_API_KEY", "").strip()
        st.session_state.ev_profile="ONLINE LIVE" if key and not key.startswith("your_") else "REPRODUCIBLE DEMO"
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


def runtime_record(result):
    """Label only the backend/model that produced the selected artifact."""
    trace=selected_trace(result)
    if not trace: return "等待实验", "尚无模型调用记录"
    runtime=trace.get("runtime", {})
    backend="openJiuwen 工作流" if runtime.get("workflow_completed") and runtime.get("backend")=="OPENJIUWEN" else "本地工作流"
    histories=result.get("evolution_history", [])
    optimizer=histories[-1].get("optimizer", {}) if histories else {}
    if optimizer.get("backend")=="OPENJIUWEN" and optimizer.get("provider_called"):
        model=optimizer.get("model", "已记录模型")
        label="DeepSeek V4.1 Flash" if model.startswith("deepseek-v4.1-flash") else model
        return backend, label+" · 提示词优化"
    return backend, "规则驱动提示词优化"


def run_controls(task_type, running):
    columns=st.columns([1.25,1,.75])
    columns[0].button("开始讲解：团队如何协作",type="primary",width="stretch",on_click=navigate,
                      args=("协作过程",),key="ev_to_collaboration")
    columns[1].button("重新运行此案例",width="stretch",key="ev_demo",disabled=running,
                     on_click=begin_run,kwargs={"fresh":True,"task_type":task_type})
    with columns[2].popover("运行设置",width="stretch"):
        st.radio("运行方式",list(PROFILES),format_func=lambda x:PROFILE_NAMES[x],key="ev_profile",disabled=running)
        st.radio("实验规模",list(DEMO_SCALES),key="ev_scale",disabled=running,
                 format_func=lambda x:f"{x} · {DEMO_SCALES[x]} 天")
    if st.session_state.get("ev_action_error"): st.error(st.session_state.ev_action_error)


def cockpit(archive, comparisons, result):
    caps=capabilities()
    job=job_snapshot()
    running=bool(job and job["status"] in {"QUEUED","RUNNING"})
    trace=selected_trace(result) or idle_trace()
    comparison=result if result and "baseline" in result else None
    task_type=job["task"]["task_type"] if running else trace.get("task",{}).get("task_type",DEFAULT_COMPETITION_DEMO["task_type"])
    backend,model=runtime_record(result)
    if running:
        backend="本地备用工作流" if job["profile"]=="OFFLINE FALLBACK" else "工作流执行中"
        model=PROFILE_NAMES[job["profile"]]
    st.html('<div class="demo-hero"><h1>一次任务，让团队学会更好的分工。</h1>'
            '<p>自演进多智能体金融政策数字风洞</p></div>')
    st.html('<div class="demo-runtime">'+''.join(f'<span>{html(v)}</span>' for v in (
        f'{caps["role_count"]} 类功能角色',f'{caps["task_count"]} 类可复现实验',backend,model))+'</div>')
    st.markdown("### "+TASK_TITLES.get(task_type,"当前实验"))
    if not running: provenance_strip(archive,result)
    run_controls(task_type,running)
    if running:
        live_job_panel()
        return
    if job and job["status"]=="FAILED": live_job_panel(expanded=False)
    if comparison:
        a,b=comparison["baseline"],comparison["evolved"]
        ca,total=evidence_progress(a); cb,_=evidence_progress(b)
        st.html('<div class="demo-scoreboard"><div><span>完成验证</span><strong>'+f'{ca}/{total} <i>→</i> {cb}/{total}'+'</strong></div>'
            '<div><span>规则评分</span><strong>'+f'{a["evaluation"]["quality_score"]:.0%} <i>→</i> {b["evaluation"]["quality_score"]:.0%}'+'</strong></div>'
            '<div><span>执行成员</span><strong>'+f'{a["evaluation"]["active_agent_count"]} <i>→</i> {b["evaluation"]["active_agent_count"]}'+'</strong></div></div>')
    left,right=st.columns([1.7,1],gap="large")
    with left:
        st.markdown("### 改进后的团队")
        histories=comparison.get("evolution_history",[]) if comparison else []
        render_graph(trace,controller=histories[-1]["controller_trace"] if histories else None,
                     previous=comparison["baseline"] if comparison else None)
    with right:
        st.markdown("### 从审查到改进")
        if comparison:
            failures=comparison["baseline"].get("critiques",[])
            st.html('<div class="demo-feedback"><b>初轮审查发现遗漏</b><p>'+html("；".join(critique_label(c) for c in failures) or "本轮未发现遗漏")+'</p></div>')
            render_timeline(comparison,compact=True)
            st.button("查看优化前后对比 →",on_click=navigate,args=("演进实验",),width="stretch",key="ev_to_evolution")
        else: st.info("暂无可复现实验结果。运行一个案例即可开始。")
    if comparison:
        with st.expander("实验条件与运行成本"):
            a,b=comparison["baseline"],comparison["evolved"]
            st.write(f'同任务、同数据、同评价规则；随机种子 {comparison["seed"]}。完成验证 {evidence_progress(a)[0]}/{evidence_progress(a)[1]} 项 → {evidence_progress(b)[0]}/{evidence_progress(b)[1]} 项。')
            st.write(f'初轮 {a["latency"]:.2f} 秒，改进后 {b["latency"]:.2f} 秒。额外核验的执行成本保留在对照记录中。')


def collaboration_page(archive, comparisons, result):
    section("", "团队如何协作，审查又发现了什么")
    if not result:
        st.info("暂无可复现实验结果")
        return
    st.write(TASK_TITLES.get(selected_trace(result)["task"]["task_type"],"当前实验"))
    options={"初始团队":result["baseline"],"改进后团队":result["evolved"]} if "baseline" in result else {"当前运行":result}
    if result.get("candidate"): options["候选配置复评"]=result["candidate"]
    choice=st.radio("执行阶段",list(options),horizontal=True,key="ev_trace_phase",label_visibility="collapsed")
    trace=options[choice]
    left,right=st.columns([1.65,1],gap="large")
    with left:
        render_graph(trace,inspect=True)
    with right:
        st.markdown("#### 审查结论")
        critiques=trace.get("critiques",[])
        if critiques:
            for c in critiques:
                st.html('<div class="demo-feedback"><b>'+html(critique_label(c))+'</b><p>'+html(chinese(c["recommendation"]))+'</p></div>')
        else: st.success("必需验证已完成，报告与证据一致。")
        count,total=evidence_progress(trace)
        st.html(f'<div class="demo-verdict"><span>验证完成度</span><strong>{count} / {total} 项</strong></div>')
        st.button("下一步：团队如何改进 →",on_click=navigate,args=("演进实验",),type="primary",width="stretch")
    if trace.get("errors"): st.warning(f'此运行保留 {len(trace["errors"])} 条执行错误。')
    st.markdown("### 分解任务与移交结果")
    events=events_from_trace(trace)
    handoffs=[e for e in events if e["kind"] in {"delegation","message","review","result"}]
    planning=next((e for e in events if e["kind"]=="agent" and e["actor"]==ROLE_NAMES["planner"]),None)
    render_events(trace,compact=True,rows=([planning] if planning else [])+handoffs[:4])
    with st.expander("展开任务分解、通信与完整轨迹"):
        plan=trace.get("plan",{})
        st.json(plan,expanded=False)
        render_events(trace)
        st.button("按本轮配置复跑",key="ev_reproduce_trace",on_click=begin_run,
                  kwargs={"task_type":trace["task"]["task_type"],"action":"reproduce","saved_trace":trace})
    with st.expander("结构化报告与金融结果"):
        render_evidence(trace,"ev_trace_")


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
    section("", "同一任务，团队改进了什么")
    if not result or "baseline" not in result:
        st.info("暂无可复现实验结果。请先运行两轮对照。")
        return
    st.write(TASK_TITLES[result["baseline"]["task"]["task_type"]])
    views={"compare":"前后对比","prompt":"提示词与工具","memory":"经验复用"}
    requested=st.query_params.get("view", "compare")
    if "ev_evolution_view" not in st.session_state: st.session_state.ev_evolution_view=requested if requested in views else "compare"
    choice=st.radio("演进证据",list(views),format_func=views.get,horizontal=True,key="ev_evolution_view",label_visibility="collapsed")
    st.query_params["view"]=choice
    a,b=result["baseline"],result["evolved"]
    if choice=="compare":
        render_comparison(result,detail=False)
        left,right=st.columns(2,gap="large")
        with left:
            st.markdown("#### 初始分工")
            render_graph(a)
        with right:
            st.markdown("#### 改进后分工")
            render_graph(b,previous=a)
        render_timeline(result,compact=True)
        with st.expander("完整指标、成本与评分细则"):
            render_comparison(result,show_config=False)
    elif choice=="prompt":
        from core.team_evolution.config import prompt_checks
        import difflib
        old,new=a["team_config"]["roles"]["planner"],b["team_config"]["roles"]["planner"]
        history=result.get("evolution_history",[])
        optimizer=history[-1].get("optimizer",{}) if history else {}
        st.html('<div class="demo-runtime"><span>'+html(runtime_record(result)[1])+'</span><span>规划员提示词 '+html(old["prompt_version"])+" → "+html(new["prompt_version"])+
                '</span><span>'+('复评后已采用' if history and history[-1].get("promotion",{}).get("accepted") else '沿用已有配置')+'</span></div>')
        st.markdown("#### 审查意见进入下一轮任务计划")
        columns=st.columns(2,gap="large")
        for col,label,role in zip(columns,("修改前","修改后"),(old,new)):
            with col:
                checks=prompt_checks(role["system_prompt"])
                st.html('<div class="demo-prompt"><b>'+label+' · '+html(role["prompt_version"])+
                    '</b><p>'+html(role["system_prompt"].split("<plan_contract>")[0].strip())+
                    '</p><div class="demo-contract">额外核验：'+html("、".join(chinese(k) for k in checks) or "未安排")+'</div></div>')
        st.markdown("#### 工具策略也随之调整")
        st.dataframe(pd.DataFrame([{"阶段":label,"首选工具":chinese(t["team_config"]["tool_policy"]["executor"]["preferred"]),
            "相同输入缓存":"启用" if t["team_config"]["tool_policy"]["executor"].get("cache") else "关闭",
            "实际工具调用":t["evaluation"]["tool_calls"]} for label,t in (("修改前",a),("修改后",b))]),hide_index=True,width="stretch")
        if optimizer.get("provider_called"):
            count=optimizer.get("token_usage")
            st.write(f'提示词优化用时 {optimizer.get("latency",0):.2f} 秒 · 模型用量 {count if count is not None else "未返回"} 词元。')
        with st.expander("提示词逐行差异与模型调用凭据"):
            st.code("\n".join(difflib.unified_diff(old["system_prompt"].splitlines(),new["system_prompt"].splitlines(),fromfile="修改前",tofile="修改后",lineterm="")),language="diff")
            st.json({k:optimizer.get(k) for k in ("api","model","timestamp","input_hash","request_messages_hash","response_hash","usage","backend")},expanded=False)
    else:
        refs=b.get("plan",{}).get("retrieved_experience_ids",[])
        st.html('<div class="demo-memory-flow"><div><span>上一轮</span><b>保存通过复评的配置</b></div><i>→</i>'
                f'<div><span>本轮检索</span><b>{len(refs)} 条相似任务经验</b></div><i>→</i>'
                f'<div><span>规划员采用</span><b>{len(b["team_config"]["topology"]["active_agents"])} 名成员 · 提示词 {html(b["team_config"]["roles"]["planner"]["prompt_version"])}</b></div></div>')
        st.markdown("### 这条经验如何影响了本轮分工")
        memory_panel(archive,b)
    st.button("下一步：验证三类任务 →",on_click=navigate,args=("三类任务",),width="stretch")


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
            kind={"policy_report":"报告生成","historical_analysis":"数据分析","regulatory_planning":"任务规划"}[task.task_type]
            st.html('<div class="demo-case-type">'+html(kind)+'</div>')
            st.html(f'<div class="ev-task-card"><h2>{html(TASK_LABELS[task.task_type])}</h2>'
                    f'<p>{html(task.objective)}</p></div>')
            available=[c for c in comparisons if c["baseline"]["task"]["task_type"]==task.task_type]
            if result and "baseline" in result and result["baseline"]["task"]["task_type"]==task.task_type: available=[result]+available
            comparison=available[0] if available else None
            if comparison:
                a,b=comparison["baseline"],comparison["evolved"]
                ca,total=evidence_progress(a); cb,_=evidence_progress(b)
                st.markdown(f'**完成验证 {ca}/{total} → {cb}/{total} 项**')
                st.write(f'执行成员 {a["evaluation"]["active_agent_count"]} → {b["evaluation"]["active_agent_count"]} 名')
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
            st.write("提示词优化：" + runtime_record(result)[1])
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
    with st.expander("单项实验文件下载"):
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
    from ui.demo_gallery import gallery_page
    renderer={"展示页面":gallery_page,"自演进驾驶舱":cockpit,"协作过程":collaboration_page,"演进实验":evolution_page,
              "三类任务":tasks_page,"技术与复现":technology_page}[page]
    renderer(archive,comparisons,result)
