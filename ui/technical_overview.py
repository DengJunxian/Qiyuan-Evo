"""Report-based technical narrative, with implemented and proposed capabilities separated.

Source: 启元 Qiyuan-Evo 技术研究报告, 技术方案稿, 2026-09-11.
See docs/TECHNICAL_CONTENT.md for the report and implementation cross-checks.
"""
import streamlit as st


STRATEGIES = (
    ("P", "提示词与示例", "已接入提示词更新",
     "把审查反馈变成下一轮的工作约束。",
     "当前通过 FeedbackPromptBuilder 或本地规则更新规划提示词，并检查可执行契约。报告进一步提出按失败案例补充示例与字段检查规则。"),
    ("U", "工具调用策略", "已接入路径与缓存调整",
     "让工具选择与任务所需证据对应。",
     "当前支持仿真与历史回放路径选择、重试配置及相同输入的计算复用；独立复核重新执行。报告将来源覆盖、质量前置检查与停止条件纳入策略设计。"),
    ("G", "角色组合与任务拓扑", "已接入专家增补与并行",
     "让组织结构随证据缺口而改变。",
     "当前从角色库加入独立复核、历史分析、量化分析或方案比较角色，并更新任务依赖。报告提出进一步学习组织模板的选择；自动合并与裁剪角色属于扩展设计。"),
    ("M", "经验检索与上下文", "已接入经验保存与引用",
     "让上一轮的有效修复进入下一轮规划。",
     "当前保存任务经验、检索相关配置并记录规划员的引用。报告将工作记忆、案例经验、领域证据与程序性技能分层组织；关联检索与检索策略优化属于扩展设计。"),
    ("R", "模型路由与资源配置", "报告扩展设计",
     "按任务难度与风险分配计算资源。",
     "报告提出轻量模型处理简单抽取、较强模型处理复杂分析，并将预算与延迟纳入选择。当前提供后端与运行模式配置，尚未接入自动学习的模型路由或上下文多臂老虎机。"),
)


def technical_overview():
    strategies = ''.join(
        f'<details class="qy-strategy"{" open" if i == 0 else ""}>'
        f'<summary><span class="qy-strategy-letter">{letter}</span><span><strong>{title}</strong><small>{status}</small></span>'
        '<span class="qy-disclosure" aria-hidden="true">＋</span></summary>'
        f'<div><h3>{lead}</h3><p>{body}</p></div></details>'
        for i, (letter, title, status, lead, body) in enumerate(STRATEGIES)
    )
    st.html(
        '<div class="qy-tech-overview"><header class="qy-tech-hero">'
        '<div class="qy-eyebrow">技术架构 · Qiyuan Evo</div>'
        '<h1>协作持续进化。<br><span>每一步，都有据可循。</span></h1>'
        '<p>以 openJiuwen 组织研究任务，以 Civitas 计算市场响应。<br>把政策、行为、交易与证据，连接成一条完整的研究链路。</p>'
        '<nav class="qy-tech-jumps" aria-label="技术内容导航"><a href="#qy-architecture" target="_self">系统架构</a>'
        '<a href="#qy-strategies" target="_self">演进机制</a><a href="#qy-tech-runtime" target="_self">运行与复现 ↘</a></nav></header>'
        '<section id="qy-architecture" class="qy-tech-section" aria-labelledby="qy-architecture-title">'
        '<header><h2 id="qy-architecture-title">双平面，一底座。</h2><p>一条链路完成当前任务，另一条链路改进后续任务。两者共用可追踪的事实与计算依据。</p></header>'
        '<div class="qy-blueprint" aria-label="任务执行平面、系统演进平面与可信仿真证据底座的关系示意">'
        '<article class="qy-plane"><div class="qy-plane-label">当前任务</div><h3>任务执行平面</h3>'
        '<p>明确分工，沿依赖交付结构化成果。</p>'
        '<ol class="qy-plane-flow"><li><b>规划与分析</b><span>任务目标 · 政策结构</span></li>'
        '<li><b>仿真与审查</b><span>市场路径 · 风险证据</span></li>'
        '<li><b>核验与报告</b><span>一致性检查 · 结论引用</span></li></ol>'
        '<div class="qy-plane-foot">研究团队负责组织工作，市场主体负责响应情景。</div></article>'
        '<article class="qy-plane qy-plane-evolution"><div class="qy-plane-label">后续任务</div><h3>系统演进平面</h3>'
        '<p>从运行反馈，形成可验证的配置更新。</p>'
        '<ol class="qy-plane-flow"><li><b>轨迹与归因</b><span>执行记录 · 证据缺口</span></li>'
        '<li><b>候选与复评</b><span>局部修改 · 重新执行</span></li>'
        '<li><b>配置晋升与复用</b><span>通过门槛 · 下一轮加载</span></li></ol>'
        '<div class="qy-plane-foot">当前已验证候选晋升；完整发布与回滚体系为扩展设计。</div></article>'
        '<div class="qy-foundation"><div><span>共同依据</span><h3>可信仿真与证据底座</h3></div>'
        '<div class="qy-foundation-items"><span>政策与行情快照</span><span>异质市场主体</span><span>撮合与风险计算</span><span>产物与版本记录</span></div></div></div>'
        '<p class="qy-tech-note">架构按技术报告整理。研究策略与市场模型分开管理，协作方式的调整不应改变政策比较的研究条件。</p></section>'
        '<section id="qy-strategies" class="qy-tech-section" aria-labelledby="qy-strategies-title">'
        '<header><h2 id="qy-strategies-title">把经验，变成可执行的改变。</h2><p>报告将研究策略组织为五个部分。展开查看每一部分的工作方式与当前接入范围。</p></header>'
        '<div class="qy-tech-columns"><div class="qy-strategies">' + strategies + '</div>'
        '<aside class="qy-trust"><span class="qy-overline">可解释性</span><h3>结论背后的<br>三条证据链。</h3>'
        '<dl><div><dt>事实链</dt><dd>政策和数据来自哪里。<small>原文、来源、时间范围与假设。</small></dd></div>'
        '<div><dt>计算链</dt><dd>数值如何产生。<small>订单、市场状态、计算产物与指标。</small></dd></div>'
        '<div><dt>决策链</dt><dd>为何采用这套研究路径。<small>任务分工、修订依据与版本变化。</small></dd></div></dl>'
        '<p>模型提供解释与行动建议，计算模块生成数值，验证规则检查进入报告的证据。</p></aside></div></section>'
        '<section class="qy-tech-section qy-computation" aria-labelledby="qy-computation-title">'
        '<header><h2 id="qy-computation-title">市场结果，由计算产生。</h2><p>政策语义经过结构化处理，主体意图经过交易约束检查，再交由撮合内核更新市场状态。</p></header>'
        '<div class="qy-compute-flow"><div><span>政策文本</span><b>结构化政策包</b></div><i aria-hidden="true">→</i>'
        '<div><span>异质主体</span><b>合法订单集合</b></div><i aria-hidden="true">→</i>'
        '<div><span>撮合内核</span><b>价格与风险指标</b></div></div>'
        '<div class="qy-compute-notes"><p><b>政策中间表示</b>当前使用 PolicyPackage。报告提出的 PolicyIR 将事实字段、机制、情景参数与证据进一步统一封装。</p>'
        '<p><b>同条件的情景比较</b>对齐初始状态、数据、市场规则与随机条件，再比较政策结果。历史回放与合成情景保留各自的数据标识。</p></div></section>'
        '<section class="qy-tech-section" aria-labelledby="qy-efficiency-title"><header><h2 id="qy-efficiency-title">把计算，用在关键路径上。</h2>'
        '<p>报告从认知、调度与复用三个层面设计效率优化。当前页面展示实际耗时与调用记录。</p></header>'
        '<div class="qy-efficiency-list"><article><span>认知</span><div><h3>慢思考，快执行。</h3><p>以现有 FastAgentKernel 的分群决策为基础，设计由重大事件触发认知更新的快慢路径，保留主体差异。</p></div></article>'
        '<article><span>调度</span><div><h3>独立分支并行，交易依赖有序。</h3><p>已有任务图支持并行分析。报告进一步提出跨情景调度、资源背压与检查点恢复；单一市场内的状态更新遵循确定顺序。</p></div></article>'
        '<article><span>复用</span><div><h3>复用相同计算，保留独立核验。</h3><p>当前支持计算缓存，独立复跑绕过缓存。报告中的多层缓存、模型前缀复用与 GPU 内存优化属于后续系统设计。</p></div></article></div></section>'
        '<details class="qy-report-source"><summary>报告来源与框架接入说明</summary>'
        '<p>内容依据《启元 Qiyuan-Evo 技术研究报告》技术方案稿，资料核查日期为 2026 年 9 月 11 日；参照第三至八节及第十节整理，并与当前项目实现核对。</p>'
        '<p>当前框架接入为 Workflow、WorkflowComponent 与 FeedbackPromptBuilder。报告提及的 TeamRuntime、AgentCard、CommunicableAgent、InstructionOptimizer 和 SkillExperienceOptimizer 属于拟适配能力，不作为本页面已接入的组件展示。</p>'
        '<p>报告描述政策预评估、新闻风险诊断与监管规划三类业务方向。当前归档演示为政策研判、历史事件分析与监管优化，具体执行状态见下方记录。</p></details></div>'
    )
