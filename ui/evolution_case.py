"""A fictional, self-contained presentation case. Never writes to experiment storage."""
from html import escape
import json
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "DEMO-LIQ-01"
CHAPTERS = (
    ("任务定义", "一项政策，一次完整的团队进化。", "锁定研究目标、时间与预算，再让团队开始工作。", "01-mission.png"),
    ("初轮执行", "分析已经完成，验证还差一步。", "五名角色完成初轮协作，审查门禁拦住了证据不完整的报告。", "02-baseline.png"),
    ("反馈归因", "把四处缺口，定位到具体环节。", "反馈指向可修改的工作方式，每一项都对应明确的修复责任。", "03-diagnosis.png"),
    ("策略改写", "下一轮，先把检查写进计划。", "同时修改提示词约束与工具顺序，让修复可以被执行。", "04-strategy.png"),
    ("团队重组", "从五人串行，到八人协作。", "时间核验、情景设计和独立复核加入；可独立的准备工作并行推进。", "05-team.png"),
    ("候选复评", "改进是否成立，交给同条件复评。", "相同输入、种子与评分规则下，比较候选质量、证据与资源开销。", "06-evaluation.png"),
    ("版本发布", "通过门禁，成为下一轮的默认策略。", "发布经过复评的配置，并保留稳定版本与可触发的回退规则。", "07-release.png"),
    ("经验复用", "这一次的修复，成为下一次的起点。", "经验进入新任务的计划与角色配置，完成跨任务自演进闭环。", "08-memory.png"),
)
QUALITY = {"baseline": [60, 80, 40, 68], "evolved": [96, 96, 96, 80]}
CHECKS = ["政策来源", "双时间核验", "参数边界", "市场路径", "无干预对照", "风险计算", "独立复跑", "引用完整性", "预算约束", "结果归档"]
MISSING = {1, 4, 6, 7}
PRICES = {
    "不干预": [100, 99, 96, 93.2, 94, 95, 95.6, 96, 96.8, 97.2, 97.6],
    "信息澄清": [100, 99.5, 97, 94.1, 95.6, 96.8, 97.9, 98.4, 99.2, 99.8, 100.6],
    "流动性支持": [100, 99.7, 98, 95.7, 97.4, 98.6, 99.8, 100.9, 101.8, 102.5, 103.1],
}
ROLES = {"planner": "任务规划", "analyst": "政策分析", "executor": "仿真执行", "critic": "风险审查", "reporter": "报告生成",
         "temporal": "时间核验", "scenario": "情景设计", "verifier": "独立复核"}


def chapter_url(chapter, appearance="dark"):
    return "?" + urlencode({"page": "演进案例", "chapter": chapter, "appearance": appearance})


def quality_score(version):
    return sum(QUALITY[version]) / len(QUALITY[version])


def max_drawdown(prices):
    peak, drawdown = prices[0], 0
    for price in prices:
        peak = max(peak, price)
        drawdown = max(drawdown, 1 - price / peak)
    return drawdown


def _metric(label, value, note="", accent=False):
    return f'<div class="case-metric{" is-accent" if accent else ""}"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>'


def _checks(complete=False):
    return '<div class="case-checks">' + ''.join(
        f'<span class="{"missing" if i in MISSING and not complete else "passed"}"><i>{"−" if i in MISSING and not complete else "✓"}</i>{label}</span>'
        for i, label in enumerate(CHECKS)) + '</div>'


def _graph(evolved=False):
    if evolved:
        nodes = [("planner",9,45),("analyst",29,20),("temporal",50,20),("executor",70,45),("verifier",91,45),("critic",91,82),("reporter",70,82),("scenario",38,72)]
        edges = [("planner","analyst"),("analyst","temporal"),("temporal","executor"),("planner","scenario"),("scenario","executor"),("executor","verifier"),("verifier","critic"),("critic","reporter")]
    else:
        nodes = [(role,10+i*20,50) for i,role in enumerate(list(ROLES)[:5])]
        edges = list(zip(list(ROLES)[:4],list(ROLES)[1:5]))
    return ('<div class="case-graph-scroll"><div class="case-graph'+(' evolved' if evolved else '')+'" data-case-edges="'+escape(json.dumps(edges),quote=True)+'">'
            + ''.join(f'<div class="case-role {"new" if role in {"temporal","scenario","verifier"} else "blocked" if not evolved and role=="critic" else "waiting" if not evolved and role=="reporter" else ""}" '
                      f'data-role="{role}" style="left:{x}%;top:{y}%"><span>{"＋ 新增" if role in {"temporal","scenario","verifier"} else "待补证据" if not evolved and role=="critic" else "等待核验" if not evolved and role=="reporter" else "已完成"}</span>'
                      f'<strong>{ROLES[role]}</strong></div>' for role,x,y in nodes) + '</div></div>')


def _scene_bodies():
    return [
        # 01 — a complete brief, including the exact policy time distinction.
        '<div class="case-split"><article class="case-panel case-brief"><span class="case-label">流动性支持政策评估</span>'
        '<h3>先回答一个研究问题。</h3><blockquote>面对市场流动性收缩，哪一种干预方案能在预算约束内控制回撤，并形成完整、可复核的证据？</blockquote>'
        '<div class="case-policy-dates"><div><span>公告时间</span><b>10 月 12 日</b></div><i>→</i><div><span>生效时间</span><b>10 月 15 日</b></div><em>两类时间分别核验</em></div>'
        '<dl class="case-facts"><div><dt>研究窗口</dt><dd>生效后 10 个交易步</dd></div><div><dt>情景集合</dt><dd>不干预 / 信息澄清 / 流动性支持</dd></div>'
        '<div><dt>预算上限</dt><dd>1.0 归一化单位</dd></div><div><dt>固定条件</dt><dd>同一市场快照 · 随机种子 42</dd></div></dl></article>'
        '<aside class="case-panel case-outlook"><span class="case-label">完整案例预览 · 示例指标</span><h3>让改变，看得见。</h3>'
        '<div class="case-score"><span>初轮</span><strong>62<i>→</i><em>92</em></strong><small>四维等权评审分 / 100</small></div>'
        '<div class="case-mini-metrics">'+_metric("参与角色","5 → 8")+_metric("任务证据","6 → 10","共 10 项")+'</div>'
        '<p class="case-panel-note">交付政策结构、情景对照、风险诊断与可追溯报告；通过复评后，将策略保存并用于下一任务。</p></aside></div>'
        '<div class="case-takeaway"><b>本次进化的对象</b><span>提示词、工具顺序、角色分工与经验规则。市场输入与评价标准保持固定。</span></div>',
        # 02 — record-shaped demo content, visibly fictional on every screen.
        '<article class="case-panel"><div class="case-panel-head"><h3>初始团队 · v1.0</h3><span class="case-status warning">报告门禁阻断</span></div>'+_graph()+
        '<div class="case-inline-metrics">'+_metric("评审分","62 / 100")+_metric("任务证据","6 / 10")+_metric("工具调用","8 次")+_metric("执行耗时","78 秒")+'</div></article>'
        '<div class="case-bottom-grid"><article class="case-panel case-log"><h3>初轮执行摘要</h3>'
        '<div><time>00:00</time><b>规划员分解任务</b><span>生成政策分析与风险报告计划</span></div><div><time>00:12</time><b>完成政策解析</b><span>将公告日误作生效日</span></div>'
        '<div><time>00:56</time><b>返回单一政策情景</b><span>未安排无干预对照与独立复跑</span></div><div><time>01:18</time><b>评审提出补证要求</b><span>报告保持草稿，等待下一轮修复</span></div></article>'
        '<article class="case-panel"><h3>产物已形成，证据待补齐。</h3>'+_checks()+'</article></div>',
        # 03 — four actionable failures and their owners.
        '<div class="case-split diagnosis"><article class="case-panel"><div class="case-panel-head"><h3>审查发现 → 修复责任</h3><span class="case-status warning">4 项关键缺口</span></div>'
        '<div class="case-issue"><b>F01</b><div><h4>公告时间与生效时间混用</h4><p>政策结构中缺少 effective_at 字段。</p><span>提示词约束 → 时间核验员</span></div></div>'
        '<div class="case-issue"><b>F02</b><div><h4>缺少无干预参照</h4><p>单一政策路径无法回答情景间的差异。</p><span>工具计划 → 情景设计员</span></div></div>'
        '<div class="case-issue"><b>F03</b><div><h4>关键结果未独立复跑</h4><p>复现结论尚无第二次执行产物支撑。</p><span>执行拓扑 → 独立复核员</span></div></div>'
        '<div class="case-issue"><b>F04</b><div><h4>部分结论未绑定来源</h4><p>报告段落缺少对应产物与引用标识。</p><span>报告契约 → 报告生成员</span></div></div></article>'
        '<aside class="case-panel"><span class="case-label">演进控制器</span><h3>只修改与问题相关的配置。</h3>'
        '<div class="case-repair-plan"><div><b>P</b><span>双时间字段、来源与缺失值规则</span></div><div><b>U</b><span>增加对照、独立复跑与引用检查</span></div><div><b>G</b><span>加入三个专业角色，更新任务依赖</span></div><div><b>M</b><span>将通过验证的修复写入经验条目</span></div></div>'
        '<div class="case-callout"><b>验证条件固定</b><p>候选不能修改政策输入、市场参数、评分权重或工具权限。</p></div></aside></div>',
        # 04 — executable planning rules rather than superficial prompt rewriting.
        '<div class="case-prompt-grid"><article class="case-panel case-old"><span class="case-label">v1.0 · 初始指令</span><h3>完成一份政策分析报告。</h3>'
        '<div class="case-prompt">解析政策内容与方向。<br>运行市场仿真，分析风险。<br>汇总结果，输出研究报告。</div>'
        '<p class="case-panel-note">没有规定双时间字段、情景对照、复跑与引用检查的先后关系。</p></article>'
        '<article class="case-panel case-updated"><span class="case-label">候选 B · 新指令</span><h3>先满足研究契约，再生成结论。</h3>'
        '<ol class="case-prompt-rules"><li>分别提取公告日与生效日，每个字段绑定原文。</li><li>建立不干预、信息澄清与流动性支持三个情景。</li><li>对齐市场快照、随机种子与观察窗口。</li><li>独立复跑入选情景，核对计算产物。</li><li>逐条检查结论引用，缺少证据时阻断发布。</li></ol>'
        '<div class="case-status">修改依据：F01 · F02 · F03 · F04</div></article></div>'
        '<article class="case-panel case-tool-plan"><h3>新的工具顺序</h3><div><span>时间与来源核验</span><i>→</i><span>三情景计算</span><i>→</i><span>独立复跑</span><i>→</i><span>引用检查</span><i>→</i><span>报告汇总</span></div></article>',
        # 05 — explicit eight-role DAG, without pretending an actual runtime is active.
        '<article class="case-panel"><div class="case-panel-head"><div><h3>候选 B · 研究团队</h3><p>政策核验与情景准备分支并行，在市场计算前汇合。</p></div><span class="case-status">5 → 8 名角色</span></div>'+_graph(True)+'</article>'
        '<div class="case-specialists"><article><span>新增 · 时间核验</span><h3>把时间，放回正确的位置。</h3><p>区分公告与生效时间，校验政策窗口和字段来源。</p></article><article><span>新增 · 情景设计</span><h3>让结果拥有同条件参照。</h3><p>构造三类情景，统一初始状态、参数与随机种子。</p></article><article><span>新增 · 独立复核</span><h3>让关键计算，再经过一次验证。</h3><p>重新执行入选情景，核对产物，并向风险审查移交结果。</p></article></div>',
        # 06 — coherent synthetic price arrays and explicit review accounting.
        '<div class="case-split evaluation"><article class="case-panel"><span class="case-label">同条件候选比较 · 示例数值</span><h3>质量提升，也计入成本。</h3>'
        '<table class="case-table"><thead><tr><th>配置</th><th>评分</th><th>证据</th><th>结果</th></tr></thead><tbody>'
        '<tr><td>v1.0 初始团队</td><td>62</td><td>6/10</td><td>待修复</td></tr><tr><td>A · 仅改提示词</td><td>78</td><td>8/10</td><td>未通过</td></tr>'
        '<tr class="selected"><td>B · 完整策略更新</td><td>92</td><td>10/10</td><td>通过</td></tr><tr><td>C · 减少复核</td><td>84</td><td>9/10</td><td>未通过</td></tr></tbody></table>'
        '<div class="case-cost"><div><span>工具调用</span><b>8 → 13 次</b></div><div><span>执行耗时</span><b>78 → 96 秒</b></div></div>'
        '<p class="case-panel-note">评分由事实正确、计算一致、证据完整与资源效率四维等权平均；入选还需关键检查全部通过。</p></article>'
        '<article class="case-panel case-market"><div class="case-panel-head"><div><h3>三个情景，同一张坐标。</h3><p>合成情景价格 · 基期 = 100</p></div></div>'
        '<div class="case-chart-legend"><span>不干预</span><span>信息澄清</span><span>流动性支持</span></div>'
        '<div class="case-price-chart" role="img" aria-label="合成示例价格曲线，三情景最大回撤分别为6.8%、5.9%、4.3%" data-prices="'+escape(json.dumps(PRICES),quote=True)+'"></div>'
        '<div class="case-market-result"><div><span>不干预最大回撤</span><strong>6.8%</strong></div><i>→</i><div><span>流动性支持最大回撤</span><strong>4.3%</strong></div></div>'
        '<p class="case-panel-note">示例目标下选择流动性支持，成本 0.8 ≤ 预算 1.0；相对不干预的最大回撤降低 2.5 个百分点。</p></article></div>',
        # 07 — a fictional release with a stated rollback plan.
        '<div class="case-split"><article class="case-panel case-release"><span class="case-label">策略注册表 · 演示状态</span><h3>v1.1，成为新的默认版本。</h3>'
        '<div class="case-version-flow"><div><small>原策略</small><strong>v1.0</strong><span>保留快照</span></div><i>→</i><div class="active"><small>候选 B 已晋升</small><strong>v1.1</strong><span>下一任务默认加载</span></div></div>'
        '<div class="case-release-items"><span>提示词规则已固化</span><span>8 人团队已登记</span><span>工具检查已前置</span><span>经验条目已关联</span></div>'
        '<div class="case-callout"><b>运行中的任务继续使用原有快照。</b><p>新版本只影响后续任务，保证本次对照的输入与执行配置可追踪。</p></div></article>'
        '<article class="case-panel"><h3>发布门禁</h3><div class="case-gates"><div><b>✓</b><span>任务证据完整</span><strong>10 / 10</strong></div><div><b>✓</b><span>四维评审评分</span><strong>92 ≥ 85</strong></div><div><b>✓</b><span>关键阻断项</span><strong>0 项</strong></div><div><b>✓</b><span>干预预算约束</span><strong>0.8 ≤ 1.0</strong></div></div>'
        '<div class="case-rollback"><span>回退预案</span><h4>保留可用版本，守住验证要求。</h4><p>后续候选若跳过独立复核或改变评分规则，则阻断晋升；已发布版本出现回归时，撤回并恢复上一稳定快照。</p></div></article></div>',
        # 08 — the changed plan is the concrete evidence of cross-task reuse.
        '<div class="case-split"><article class="case-panel case-memory"><span class="case-label">经验条目 · POLICY-VERIFY-01</span><h3>保存的，是经过验证的工作方法。</h3>'
        '<dl class="case-facts"><div><dt>适用任务</dt><dd>含生效时滞的政策情景评估</dd></div><div><dt>触发问题</dt><dd>时间混用、对照缺失与复核不足</dd></div><div><dt>策略关联</dt><dd>v1.1 · 候选 B · 10 项证据检查</dd></div></dl>'
        '<div class="case-memory-rules"><div><b>01</b><span>先核验双时间与出处</span></div><div><b>02</b><span>同条件建立政策与参照情景</span></div><div><b>03</b><span>独立复跑，逐条绑定报告引用</span></div></div>'
        '<p class="case-panel-note">当数据范围、市场模型或任务契约变化时，经验需要重新验证。</p></article>'
        '<article class="case-panel case-next-task"><span class="case-label">下一任务 · 交易税费调整评估</span><h3>新任务，一开始就带着经验。</h3>'
        '<ol class="case-reuse-flow"><li><b>检索到相关修复</b><span>命中双时间、政策对照与独立核验规则</span></li><li><b>规划员采用 v1.1</b><span>加载 8 人团队与前置工具检查</span></li><li><b>检查进入首轮计划</b><span>从计划阶段安排对照、复跑与引用核验</span></li></ol>'
        '<div class="case-next-result"><span>计划已更新 · 待开始执行</span><strong>经验 → 配置 → 行为</strong><p>本次完整流程止于新任务计划生成；跨任务收益需再执行后评估。</p></div></article></div>'
        '<div class="case-takeaway"><b>自演进闭环完成</b><span>初轮执行 → 证据反馈 → 策略修改 → 独立复评 → 版本发布 → 下一任务复用。</span></div>',
    ]


def render_evolution_case():
    try:
        selected = max(1, min(8, int(st.query_params.get("chapter", 1))))
    except (ValueError, TypeError):
        selected = 1
    appearance = st.session_state.get("ui_theme", "dark")
    tabs, scenes = [], []
    for index, ((label, title, description, _), body) in enumerate(zip(CHAPTERS, _scene_bodies()), 1):
        title_parts = title.split("，", 1)
        display_title = (f'<span>{title_parts[0]}，</span><span>{title_parts[1]}</span>'
                         if len(title_parts) == 2 else title)
        tabs.append(f'<button type="button" role="tab" id="case-tab-{index}" data-chapter="{index}" aria-controls="case-scene-{index}" '
                    f'aria-selected="{str(index==selected).lower()}" tabindex="{0 if index==selected else -1}"><span>{index:02}</span>{label}</button>')
        scenes.append(f'<section class="qy-story-scene" id="case-scene-{index}" role="tabpanel" aria-labelledby="case-tab-{index}" data-scene="{index}" '
                      f'{"hidden" if index!=selected else ""}><header class="case-scene-heading"><div><span class="case-chapter-label">{index:02} / 08 · {label}</span>'
                      f'<h1>{display_title}</h1><p>{description}</p></div></header>{body}</section>')
    markup = ('<div class="qy-evolution-case" data-selected="'+str(selected)+'"><div class="case-context"><div><strong>流动性支持政策评估</strong><span>完整自演进案例</span></div>'
              '<span class="case-fiction-label">设计案例 · 示例数据</span></div><nav class="case-chapters" role="tablist" aria-label="演进案例章节">'+''.join(tabs)+'</nav>'
              +''.join(scenes)+'<footer class="case-footer"><span>DEMO-LIQ-01 · 流程演示，非实测结果</span><div>'
              f'<a href="?page=%E5%B1%95%E7%A4%BA%E9%A1%B5%E9%9D%A2&amp;appearance={appearance}#team-gallery" target="_self">返回图片演示 ↗</a>'
              '<button type="button" data-case-action="previous" aria-label="案例上一章">‹</button><output data-case-counter aria-live="polite">'+f'{selected:02} / 08'+'</output>'
              '<button type="button" data-case-action="next" aria-label="案例下一章">›</button></div></footer></div>')
    css = (ROOT / "theme" / "evolution_case.css").read_text(encoding="utf-8")
    script = (ROOT / "static" / "evolution_case.js").read_text(encoding="utf-8")
    st.html('<style>'+css+'</style>'+markup+'<script>'+script+'</script>', unsafe_allow_javascript=True)
