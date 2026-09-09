# 启元 Qiyuan-Evo 改造审计与方案

本次修改基于用户提供的 Civitas-Economica 源码目录。原目录没有 Git 元数据，已建立本地原始快照用于变更审查；该快照不是 GitHub 的提交历史。

## 审计与复用边界

| 现有模块 | 审计结论 | 改造方式 |
|---|---|---|
| agents/manager_agent.py、roles/ | Manager 聚合新闻/量化/风险并输出交易意图，尚无通用 TaskPlan | 增加独立任务规划方法，保留交易接口 |
| agents/learning/、core/exchange/evolution.py | 策略先验、行情分型、交易策略基因和种群演化 | 留在 Layer B，不能计为团队演化 |
| agents/crews/investment_firm.py | 固定投研团队、投票和风险约束 | 保留其市场主体职责 |
| agents/debate_brain.py、reflection.py | 投资观点辩论、交易反思及日记；部分输出包含推理字段 | 保留市场路径；比赛暴露结构化审查与决策摘要 |
| agents/diagnostic/、report/ | 现有探针和工具循环，可扩展为证据审查与结构化报告 | 复用类，增加不依赖云端的接口 |
| core/policy_committee.py、policy/structured.py | LLM 委员会、规则式 PolicyPackage 均已存在 | 复用 PolicyPackage 作为执行事实和证据 |
| core/model_router.py | 模型回退、缓存和可观测性；没有团队晋升契约 | 保留；比赛单独标识编排后端与模型/优化器回退 |
| core/competition_demo.py、competition_compliance.py | 预录场景加载及材料扫描 | 保留历史功能；新增真实运行的 CompetitionService |
| engine/agent_scheduler.py、simulation_loop.py | 市场快慢 Agent 调度、宏观/社会耦合、撮合、策略生态 | 通过进程隔离的工具调用，固定随机种子 |
| core/calibration/replay_runner.py | 历史序列条件回放与误差评分；宏观面板含合成假设 | 复用回放计算，冻结观测行情并明确区分合成宏观假设 |
| ui/、app.py | Streamlit 现有政策/历史/研判页面；部分旧场景为预录数据 | 增加团队演进入口，仅消费 service 的运行产物 |
| scripts/、requirements、tests | 有展示脚本，无已提交测试目录，无 openJiuwen 依赖 | 增加固定基准、测试与 Python 3.12 验证环境 |

## 两层结构

Layer A：TaskSpec → Manager/Planner → Workflow DAG → 专业 Agent → Critic → Reporter → Evaluation → Evolution Controller → Experience Store → 下一次配置。

Layer B：Civitas PolicyPackage、MarketEnvironment、TraderAgent、订单簿撮合、ReplayRunner、RiskAnalyst、QuantAnalyst。Layer A 不生成市场价格，不改写 Layer B 的策略生态作为团队演化证据。

## 实现顺序和验收

1. 固定 JSON 数据契约、基准任务和观测行情快照；冻结评分规则。
2. 对已安装 openjiuwen 0.1.17.post1 源码核验 Workflow、Session、WorkflowComponent 和 FeedbackPromptBuilder。
3. 构建真实 Workflow DAG 与相同语义的本地回退；记录实际执行 Agent 和消息。
4. 根据 Critic 和历史工具绩效生成 Prompt、工具策略、拓扑候选；同任务同种子评估，严格质量改进才晋升，保存拒绝结果。
5. 经验按任务类型、数据、模型与约束兼容性检索。固定 baseline 不读经验。
6. 接入现有 Streamlit；运行三类基准、回归测试、真实 SDK 集成测试和 UI 测试。

Prompt 离线模式采用机器可读执行契约优化，不能声称调用过 LLM。云端配置完整时优先使用官方 FeedbackPromptBuilder 生成候选；候选仍须契约验证和实测晋升。质量、风险改善及成本均由运行产物计算，允许零/负增益。
