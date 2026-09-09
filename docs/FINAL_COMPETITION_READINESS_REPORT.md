# FINAL COMPETITION READINESS REPORT

本报告保留前一轮技术验收快照。后续前端中文化、默认案例与浏览器验证见 [展示动线与数据约定](PRODUCT_PRESENTATION.md)，其中记录最新参考数据和测试结果。
**结论：READY，适用于校赛的 REPRODUCIBLE DEMO / 原生 openJiuwen Workflow 演示。**

验收日期：2026-09-09。30 项核心要求均有源码与执行/界面证据；没有待修复的核心 FAIL。云端模型成功实连等扩展能力单独标为 PARTIAL 或未实现，见第 9 节。READY 表示当前研究原型可按本报告复现和答辩，不表示生产部署认证。

验收环境：Python 3.12.13、Streamlit 1.63.0、openjiuwen 0.1.17.post1。最终核心源码哈希：`baa97434b1e156d9a23457a2f7a864919d19450477f074839860fd0c89704185`。数据 SHA-256：`48abaf185050544cf93ef4adf7dcebab9225555fd4c8ac2be2b3b2b12481fb3c`。

## 1. Requirement Matrix

PASS 表示本项所述能力已被对应证据支持；PARTIAL 表示有实现但尚缺该项必需的验收证据；FAIL 表示核心能力缺失或链路不成立。

运行证据索引：

- **A**：政策报告，实验 `exp-499a56f05f254ebc921c9823211304a0`；Baseline `run-03c832c5ceb84493ba9a8066415a691a`，Evolved `run-ed925ef864ad40bb97385a8029ebed25`。
- **B**：历史分析，实验 `exp-e6b708fca3054e1da0fad6f0599fcac4`；Baseline `run-13d7ef0a85c34fc5a039fe1a4ec790cd`，Evolved `run-cfc8c9c6c15e4b05877db632d04be9bc`。
- **C**：监管规划，实验 `exp-208155351b3e4a419e8eb48628c1adf0`；Baseline `run-89a836f5230540dd9a6f3d8de0e7dca7`，Evolved `run-38ef29d2aadf4fc0b116c40d60e39220`。
- A/B/C 原始文件在 [`outputs/competition_readiness_final/`](../outputs/competition_readiness_final/)，每个运行对应 `runs/<run_id>/run_trace.json`；每个配对对应 `comparisons/<baseline_run_id>/competition_demo_summary.json`。
- **D（真实浏览器演示）**：实验 `exp-c0a04b0fd68445cba8904435e117fcd7`，目录 [`demo-df092d37ef454523bbfc029c81958c59`](../outputs/competition_demos/demo-df092d37ef454523bbfc029c81958c59/)；Baseline `run-297b9682209740829730a7244f16c1b9`，Evolved `run-4bdcbbffe764454b9ebda3cf3017f018`。
- **V**：[`evidence_verification.json`](../outputs/competition_readiness_final/evidence_verification.json)，16 项基于产物的交叉校验全部通过。
- **T**：45 项完整测试与最终 4 项界面/恢复回归；源码见 [`tests/`](../tests/)，具体运行日志见第 11 节。

| # | Requirement | 状态 | Code evidence | Runtime evidence | UI evidence |
|---|---|---|---|---|---|
| 1 | 至少 3 类功能 Agent | PASS | [baseline_config](../core/team_evolution/config.py#L30)；`config.baseline_config`、`TeamRuntime.dispatch` | A/B/C：5 类基础任务角色；另有真实 Controller，合计跨任务 10 类已执行角色 | 驾驶舱配置徽标、Team Graph |
| 2 | 每类 Agent 实际调用 | PASS | [dispatch](../core/team_evolution/runtime.py#L121)；`TeamRuntime.dispatch`；`CompetitionService._compare` 保存 Controller trace | V：启用节点集合等于完成节点集合；Controller 有独立 ExecutionTrace | 节点状态与协作事件 |
| 3 | Agent communication | PASS | [AgentMessage](../core/team_evolution/models.py#L88)；`AgentMessage`、`dispatch(incoming)`、Workflow 输入依赖 | A/B/C messages 覆盖实际拓扑边 | 协作过程的 delegation/message/review/result |
| 4 | Task decomposition | PASS | [plan_competition_task](../agents/manager_agent.py#L104)；`ManagerAgent.plan_competition_task` | TaskPlan 含 subtasks、assignments、dependency_graph | 协作过程 → 报告与证据 → 计划 |
| 5 | 动态 scheduling | PASS | [OpenJiuwenBackend](../core/team_evolution/backend.py#L49)；`EvolutionController.propose`、`OpenJiuwenBackend.execute`、有界 replan | B：serial → parallel；SDK 并行重叠与故障重规划测试通过 | B 的并行拓扑、调度策略与事件 |
| 6 | Prompt evolution | PASS | [PromptOptimizer](../core/team_evolution/backend.py#L109)；`PromptOptimizer.optimize`、Prompt Registry | A/B/C 初轮均记录 old/new/diff 和晋升结果；当前采用本地契约修复 | Evolution Timeline / Prompt Diff |
| 7 | Prompt 影响下一轮 | PASS | [execute_simulation](../core/team_evolution/runtime.py#L54)；`prompt_checks` 被 Planner 和 Runtime 读取 | Evolved Planner 的 prompt_hash 等于晋升文本哈希；同版本改内容行为测试通过 | v1/v2 对比、实际新增核验事件 |
| 8 | Tool / Skill evolution | PASS | [propose](../core/team_evolution/evolution.py#L18)；`tool_statistics`、`EvolutionController.propose` | 工具成功率、质量、时延、token 与失败次数进入策略 | Tool Strategy、Controller 统计 |
| 9 | Tool 影响下一轮选择 | PASS | [execute_simulation](../core/team_evolution/runtime.py#L54)；`execute_simulation` 按 preferred 选择真实函数 | B：market_simulation → historical_replay；新调用实际出现在 tool_calls | B 工具调用与策略前后对比 |
| 10 | Agent 数量或角色变化 | PASS | [propose](../core/team_evolution/evolution.py#L18)；`AgentRoleSpec` 注册专家并加入 active_agents | A/C：5 → 6；B：5 → 7；新角色完成执行 | 新增 verifier/historian/quant/counterfactual |
| 11 | Backend topology 真变化 | PASS | [OpenJiuwenBackend](../core/team_evolution/backend.py#L49)；新 TeamTopology 重建 Workflow 节点和连接 | V：真实配置变化、节点完成且消息沿新边传递 | Topology Diff、组织结构演化 |
| 12 | 从历史 performance 学习 | PASS | [_select_evolved](../core/competition_service.py#L92)；`ExperienceStore`、`_select_evolved`、tool statistics | A/B/C 的成功策略进入持久配置；后续轮次复用，未重复无收益晋升 | Experience Memory |
| 13 | 相似任务读取经验 | PASS | [retrieve](../core/team_evolution/store.py#L77)；`context_key` 与 `retrieve` | `test_similar_task_reads_successful_configuration_and_records_experience`：不同任务编号/目标表述读取此前成功配置并完成 | Planner 的经验编号与 Times Reused |
| 14 | 固定 baseline | PASS | [run_baseline](../core/competition_service.py#L110)；`run_baseline`、`baseline_config` | 同输入三轮 Baseline semantic_hash 稳定；经验读取为 0 | BASELINE / Static Agent Team |
| 15 | evolved | PASS | [run_evolved](../core/competition_service.py#L115)；`run_evolved`、持久化配置选择 | A/B/C 的下一轮从已晋升配置执行；全任务 evolved CLI 实跑 | EVOLVED / Team v2 |
| 16 | 同 task/seed/规则 | PASS | [validate_comparison](../core/competition_cockpit.py#L75)；`_compare` 共用 TaskSpec；`validate_comparison` | V 核对完整 TaskSpec、数据/源码哈希、模型及重新评分；篡改任务测试拒绝 | 来源条与配对条件 |
| 17 | 真实 before/after | PASS | [evaluate](../core/team_evolution/evaluation.py#L56)；`evaluate`、`evolution_gain`、加载时重算 | 9 组比较，分数由 trace 重算；篡改 gain 测试拒绝 | 分数、步骤、成本与持平项均实报 |
| 18 | 至少 3 类任务 | PASS | [benchmark_tasks](../core/team_evolution/benchmark.py#L28)；`benchmark.benchmark_tasks` | A/B/C 各 3 轮全部执行 | 三类任务卡片 |
| 19 | 共用 Agent Team infrastructure | PASS | [_run](../core/competition_service.py#L72)；`CompetitionService._run` → `TeamRuntime.run` | 三类都产生同 schema 的 trace，均完成原生 Workflow | 相同 Graph/Trace/Timeline/Comparison 组件 |
| 20 | 脚本复现 | PASS | [main](../scripts/run_competition_benchmark.py#L11)；`run_competition_benchmark.py`、`run_competition_demo.py` | all + 三个单任务参数，baseline/evolved/compare 均实跑；B 的 UI Reproduce 语义一致 | Reproduce 按钮、技术与复现命令 |
| 21 | 结构化输出 | PASS | [JSONModel](../core/team_evolution/models.py#L15)；`models.py` dataclass/JSONModel、schema/有限值检查 | Task/Plan/Message/Trace/Critique/Action/Snapshot 可 JSON 序列化 | 明细展开与 JSON/ZIP 下载 |
| 22 | evidence/confidence/critique | PASS | [synthesize_evidence](../agents/report/report_agent.py#L19)；`ReportAgent.synthesize_evidence`、`review_evidence` | V 核对每条 facts 等于所引 evidence；缺失与矛盾测试有效 | 报告与证据面板 |
| 23 | UI 展示协作过程 | PASS | [render_events](../ui/components/evolution.py#L203)；`render_graph`、`render_events`、observer | D 的 61 条后端观察；执行时 RUNNING/WAITING，结束为 COMPLETED | 实际浏览器检查 + AppTest |
| 24 | Evolution Timeline | PASS | [render_timeline](../ui/components/evolution.py#L313)；`render_timeline` | D 的真实 feedback/actions/评价及来源 trace | WHAT/WHY/EVIDENCE/EXPECTED/ACTUAL |
| 25 | Prompt Diff | PASS | [render_comparison](../ui/components/evolution.py#L252)；`difflib.unified_diff`、版本化 registry | D 的契约 checks 从空变为 control/replicate | 已实际展开检查 v1 → v2 |
| 26 | Topology Diff | PASS | [generations](../core/competition_cockpit.py#L186)；`generations`、真实 config 比较 | D 只呈现实际两代；拒绝晋升不创建新代 | 已检查左右拓扑与代际选择器 |
| 27 | UI before/after | PASS | [render_comparison](../ui/components/evolution.py#L252)；`render_comparison`、验证后的 service/artifact | D 70% → 100%，Steps 5 → 6、Latency 增加也显示 | 已检查评分组成、拓扑/Prompt/工具页签 |
| 28 | 真实 openJiuwen | PASS | [OpenJiuwenBackend](../core/team_evolution/backend.py#L49)；`backend.py` 导入并调用 Workflow/Component/session | A/B/C/D package=0.1.17.post1，invoked/completed=true；严格 SDK 测试不允许静默回退 | Runtime、记录后端、调用位置 |
| 29 | UI Runtime 与实际一致 | PASS | [technology_page](../ui/team_evolution.py#L407)；安装/READY 与 trace 执行状态分别读取 | Workflow 为 OPENJIUWEN；Prompt optimizer 为 LOCAL FALLBACK，不混标 | 技术页与演进页分别明确显示 |
| 30 | 密钥/网络失败安全降级 | PASS | [PromptOptimizer](../core/team_evolution/backend.py#L109)；`PromptOptimizer` timeout/exception、本地 backend、后台作业错误归档 | 缺密钥、SDK 缺失、云端 ConnectionError、工具失败、观察者故障测试通过 | 错误 trace 仍可呈现；明确 fallback，保留已完成结果 |

未发现以投资者 personality、随机参数、前端预设事件或虚假 SDK 包装冒充团队演进的链路。当前 Prompt 演进是可执行契约的配置级修复，工具与拓扑在下一轮确实被执行。

## 2. Architecture Summary

Layer A：TaskSpec → Manager/Planner → 原生 Workflow DAG → 专业 Agent/工具 → Critic → Reporter → 固定 Evaluation → Evolution Controller → 候选试跑 → 严格晋升/拒绝 → Experience Store → 下一轮。

Layer B：原 Civitas PolicyPackage、MarketEnvironment、A 股订单簿/撮合、ReplayRunner、QuantAnalyst、RiskAnalyst 和监管反事实环境。比赛服务通过专业工具调用 Layer B，价格与风险不由 Reporter 生成。

五个基础任务节点加一个独立演进控制器构成六类基础功能角色；专家按任务配置加入。沿用现有 Manager、Diagnostic 与 Report Agent，没有重建割裂 Demo。原市场主体 StrategyGenome / personality evolution 属于 Layer B；Team Evolution Plane 属于 Layer A，两者数据结构、控制器与验收指标分离。

## 3. openJiuwen Integration Evidence

[`backend.py`](../core/team_evolution/backend.py) 真实调用 `Workflow`、`WorkflowCard`、`WorkflowComponent`、`Start`、`End`、`create_workflow_session` 和 `Workflow.invoke`。每个任务角色对应 SDK component，输入 schema 携带上游结果，`wait_for_all=True` 使 Critic 等待并行依赖完成。

SDK 编排不需要 LLM 密钥。当前任务角色为确定性实现；因此“Workflow: OPENJIUWEN”和“Prompt optimizer: LOCAL FALLBACK”可以同时成立。严格 `--backend openjiuwen` 缺包/导入失败时报错；`auto` 才允许明确的本地回退。

云端优化入口直接使用官方 `FeedbackPromptBuilder.build`。已通过已安装源码、构造与受控成功/故障返回测试核验；真实云端成功请求尚未验收。优化记录包括 model、temperature、prompt_version、input hash、时间、backend、原始响应哈希及实际采用文本哈希，无密钥泄漏；不可获得的 token usage 记为 null。

## 4. Self-Evolution Evidence

固定默认场景为 `DEFAULT_COMPETITION_DEMO`：政策报告、seed 42、6 天。Baseline 是通用五角色流水线，缺少对照与独立复现。这是固定、显式的工程 failure case，由 `review_evidence` 判定；UI 不写死“失败事件”。

D 中发生三个真实变化：

1. Prompt v1 → v2：`checks: []` → `["control", "replicate"]`。
2. 工具策略：允许复用相同确定性调用；Executor 增加无政策对照，独立复现绕过缓存。
3. 拓扑：5 → 6 个任务节点；verifier 执行核验，Critic 等待其结果。

候选在同任务/seed/数据/模型/规则下实跑，质量严格改善且无审查退化后提交 SQLite。随后另一次 Evolved 运行从持久配置和经验读取，而非直接沿用内存候选。D 共 61 条观察事件，总流程 25.65 秒，包含候选试跑与写盘。

每类任务的 3 轮 benchmark 中，经验引用数量依次为 1、2、3；首轮晋升 v2，后续无新变化时不再晋升。所有三轮的对应 Baseline/Evolved 语义哈希稳定。工具统计与失败记录保留，基于历史上下文选择配置。

## 5. Three Benchmark Results

固定 seed=42，市场任务 12 天，历史数据为冻结的 18 个交易日观测窗口。各任务执行 3 轮，共 9 组对照；Evolved Task Success 为 9/9。以下为各任务第一轮实测值，其他轮质量相同。

| 任务 | Baseline Quality | Evolved Quality | Quality Gain | Agent 数量 | 工具调用 | Evolved Task Success |
|---|---:|---:|---:|---:|---:|---|
| 政策研判 / Report Generation | 70.00% | 100.00% | +30.00 pp | 5 → 6 | 4 → 6 | PASS |
| 历史分析 / Data Analysis | 55.00% | 95.50% | +40.50 pp | 5 → 7 | 4 → 7 | PASS |
| 监管规划 / Task Planning | 58.00% | 100.00% | +42.00 pp | 5 → 6 | 4 → 8 | PASS |

历史分析：归一化 RMSE=0.04021337，MAE=0.02985518，方向一致率=58.82%。这是有观测条件的历史回放，宏观变量为标注的合成假设，不能解释成未来市场预测。

监管规划：三方案为 no_action、clarification、liquidity_support；归一化成本分别为 0、0.2、0.8。当前固定样本下三个方案最大回撤相同，风险改善为 **0**，推荐 **no_action**。Quality=100% 表示方案/约束/证据任务完成，不表示干预取得金融收益。

## 6. Before/After Results

| 指标（第一轮） | 政策 A：B → E | 历史 B：B → E | 监管 C：B → E |
|---|---|---|---|
| Task Success | 未完成 → 通过 | 未完成 → 通过 | 未完成 → 通过 |
| Evidence Coverage | 100% → 100% | 100% → 100% | 100% → 100% |
| Critic Pass Rate | 66.67% → 100% | 57.14% → 100% | 66.67% → 100% |
| 独立复现 | 未执行 → 一致 | 未执行 → 一致 | 未执行 → 一致 |
| Steps | 5 → 6 | 5 → 7 | 5 → 6 |
| Retries | 0 → 0 | 0 → 0 | 0 → 0 |
| Token Usage | 0 → 0 | 0 → 0 | 0 → 0 |
| Latency（秒） | 5.287 → 12.632 | 4.488 → 0.014 | 5.085 → 18.513 |

Evidence Coverage 的分母是已输出结论；Baseline 输出较少但均绑定证据，因此覆盖率可为 100%，缺失输出由 completeness 与 Critic 检出。UI 没有伪造覆盖率提升。零 token 无法计算节省比例，`token_reduction=null`。增加核验会增加步骤与耗时；历史任务改用轻量 ReplayRunner，耗时下降源于工具选择变化。运行时延受机器负载影响，不作总体效率优势的确认性结论。

Quality 组成：完整度×Schema 30%、证据覆盖 20%、事实一致性 20%、任务规则质量 20%、独立复现 10%。成功还要求 Quality≥85%、必需证据齐全、无高严重度 Critique、无未恢复错误。采用程序校验与固定规则 Rubric；未调用 LLM Judge。候选训练成本单独保存，未藏入或省略为“免费演进”。

## 7. Frontend Demo Flow

一级导航：**自演进驾驶舱 → 协作过程 → 演进实验 → 三类任务 → 金融政策风洞 → 历史验证 → 技术与复现**。原成果展示、系统总览、政策实验、研判分析保留在业务能力入口。

建议答辩：

1. 首页讲清 Agent Team、三类演进、实际 SDK 与当前证据。点击“运行自演进演示”。
2. 观察真实执行状态；结束后点“查看完整证据链”，查看 Baseline 的两个缺失项。
3. 点“下一步”，展示 Prompt Diff、5 → 6 拓扑、真实 before/after 与评分组成。
4. 查看三任务结果与历史经验，再进入金融工具层说明应用价值。
5. 技术页核对 Runtime，下载同 experiment_id 的 Evidence Bundle。

实际浏览器已检查主页、Graph、事件、Timeline、Prompt Diff、Topology Diff、评分组成、代际、Experience Memory、任务卡、业务风洞、历史页与技术页。执行中/完成后刷新和服务重启恢复均已验收；历史任务 Reproduce 返回 semantic_match=true。

实时数据：后台观察者推送的运行状态、Agent/tool/message 事件与新完成 service 返回值。归档数据：初次加载参考 benchmark、历史时间线、代际与已完成比较，均明确标为 artifact 回放。经验复用次数从实际 TaskPlan 引用统计。

修复的现场问题：实验身份缺失、刷新丢失入口、协作阶段 Run ID 错位、页底跳转位置、评分组成不明显、部分表格单位不清、比较 gain 未重新校验。最终 UI 回归通过；没有残留的空图、乱码、异常大 JSON 或页面崩溃。

## 8. Reproduction Commands

从仓库目录执行，Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m pip check
python -m compileall -q agents core engine policy ui app.py scripts tests
python -m pytest -q

# 使用新目录得到独立的 Baseline → 候选 → Evolved 实验
python scripts/run_competition_benchmark.py --task all --seed 42 --mode compare --backend openjiuwen --rounds 3 --output outputs/my_acceptance
python scripts/verify_competition_evidence.py --input outputs/my_acceptance

# 同目录读取已晋升经验；Baseline 始终隔离经验
python scripts/run_competition_benchmark.py --task all --seed 42 --mode baseline --backend openjiuwen --output outputs/my_acceptance
python scripts/run_competition_benchmark.py --task all --seed 42 --mode evolved --backend openjiuwen --output outputs/my_acceptance

# 单任务接口
python scripts/run_competition_benchmark.py --task policy_report --seed 42 --mode baseline --output outputs/my_acceptance
python scripts/run_competition_benchmark.py --task historical_analysis --seed 42 --mode evolved --output outputs/my_acceptance
python scripts/run_competition_benchmark.py --task regulatory_planning --seed 42 --mode compare --output outputs/my_acceptance

# 与首页按钮相同的 6 天场景 / 显式离线模式
python scripts/run_competition_demo.py --profile reproducible
python scripts/run_competition_demo.py --profile offline --scale full
streamlit run app.py
```

可通过 `QIYUAN_REFERENCE_DIR=outputs/my_acceptance` 选择已完成结果。当前本机默认 `outputs/competition_cockpit_reference` 指向最终验收产物，旧阶段产物保留。outputs 被 Git 忽略，新的检出需先执行复现命令。

Evidence Bundle 复用已有 canonical trace 和配对拓扑，以清单内的 path/JSON pointer/SHA-256 定位以下逻辑产物：competition_demo_summary、baseline_trace、evolved_trace、evaluation、evolution_history、prompt_diff、topology_before、topology_after、benchmark_summary，另附 before_after 和 Controller trace。每个成员身份与内容哈希都在下载前校验。

## 9. Remaining Known Limitations

| 项目 | 状态 | 明确范围 |
|---|---|---|
| 云端 Prompt 优化成功实连 | PARTIAL | 官方接口已接入；缺钥与受控成功/故障测试完成，真实有效 Key 的成功请求未验收。默认演示走本地契约优化 + 真 SDK Workflow。 |
| 任意新角色生成、自动 retire/merge | 未实现 | 当前是已注册专家的 spawn/split/reroute/parallelize，未宣称自主生成新 Agent 代码。 |
| 持续在线学习、跨任务泛化与逐机制消融 | 未实现 | 当前为固定任务、历史配置检索和同任务联合候选评估；不宣称未知任务泛化或单项因果贡献。 |
| LLM qualitative judge | 未实现 | 当前评分为确定性检查与固定规则 Rubric，UI 已标明。 |
| 中断任务自动续算 | 未实现 | 服务重启恢复已保存结果；未完成任务标记 InterruptedRun，保留 trace 后重跑。 |
| 移动端/小屏布局完整验收 | PARTIAL | 实际检查为 1280×720 桌面视口；宽表允许内部横向滚动，小屏未完成全量验收。 |
| C++ 撮合扩展 | 本次未启用 | 使用原 Python 订单簿回退，未声称本次实跑使用 C++。 |
| 市场外推能力 | 未建立 | 6/12 天简化仿真与冻结历史窗口适用于方法演示，不是实时真实市场预测。监管零风险改善已保留。 |

这些边界不被计作已完成的增强功能。核心 30 项的 PASS 以表内限定实现为准。

## 10. Changed Files

本轮最终验收直接修改/新增：

| 文件 | 改动 |
|---|---|
| `core/team_evolution/models.py` | Trace/Comparison 增加 experiment_id |
| `core/competition_service.py` | 同一实验贯穿 Baseline、候选、Controller、Evolved，自动导出证据包 |
| `core/team_evolution/evidence_bundle.py` | 复用规范产物、身份/哈希校验、ZIP 导出 |
| `core/team_evolution/backend.py` | Prompt 调用的模型、版本、输入/响应哈希与时间元数据 |
| `core/team_evolution/evolution.py` | 向优化器传递真实 Prompt 版本 |
| `core/competition_cockpit.py` | 默认场景、比较 gain/身份验证、持久作业索引与中断恢复 |
| `scripts/run_competition_benchmark.py` | all + mode 路由、seed、参数校验与可靠退出码 |
| `scripts/run_competition_demo.py` | 复用 DEFAULT_COMPETITION_DEMO |
| `scripts/verify_competition_evidence.py` | 16 项产物交叉核验 CLI |
| `ui/team_evolution.py` | 刷新恢复、首页证据摘要、阶段正确 Run ID、单位、证据包下载 |
| `ui/components/evolution.py` | 透明评分组成与通过条件 |
| `app.py` | 页面入口恢复、跨页滚动位置修复 |
| `theme/competition_dark.css` | 首屏证据摘要布局 |
| `tests/test_competition_cockpit.py` | 相似任务的实际经验复用测试 |
| `tests/test_competition_readiness.py` | 默认演示、证据包、篡改、重启/刷新、LLM 元数据测试 |
| `README.md` | 以自演进系统为主线重构前半部分，补充角色/机制/复现/边界 |
| `docs/FINAL_COMPETITION_READINESS_REPORT.md` | 本报告 |

此前两阶段的复用与 UI 改造继续保留。完整相对原源码快照的 diff 列表和 patch 见 `outputs/git_diff_summary.txt`、`outputs/qiyuan_evo.patch`。没有提交或推送这些改动到 GitHub。

## 11. Tests Executed

| 检查 | 结果 | 证据 |
|---|---|---|
| Dependency check | PASS，无依赖冲突 | `.venv/bin/python -m pip check` |
| Compile | PASS | compileall：agents/core/engine/policy/ui/app/scripts/tests |
| Full unit/integration/UI suite | **45 passed，3 warnings，162.99s** | `tmp/pytest_final_acceptance.log` |
| 最后 UI/刷新/导航回归 | **4 passed，3 warnings，33.61s** | `tmp/pytest_ui_acceptance_pass.log` |
| 三任务 × 三轮 compare | PASS，9/9 Evolved success，OPENJIUWEN | `tmp/benchmark_readiness_final.log`、summary.json |
| all baseline / all evolved | PASS，三个任务均真实执行；Baseline 的未完成分数按设计保留 | `tmp/cli_all_baseline_final.log`、`tmp/cli_all_evolved_final.log` |
| 三个单任务参数 | PASS：policy baseline / historical evolved / regulatory compare | `tmp/cli_policy_baseline_final.log`、`tmp/cli_history_evolved_final.log`、`tmp/cli_regulatory_compare_final.log` |
| 产物交叉校验 | PASS，16/16 | `scripts/verify_competition_evidence.py`、evidence_verification.json |
| Streamlit import / server | PASS | `tmp/import_readiness_final.log`、`tmp/streamlit_final_acceptance.log` |
| 真实浏览器 | PASS：D 一键实跑、刷新、重启恢复、主要页面、历史任务 Reproduce | 本报告第 7 节与 D 的 event_stream/demo_manifest |
| Git diff whitespace | PASS | `git diff --check` |

3 条警告来自 SDK/Pydantic 配置与 dashscope 弃用提示，不是任务执行错误。导入裸 app 时 Streamlit 的无 ScriptRunContext 提示与 Python 撮合回退提示保留且已区分。

本轮验收捕获并修复过滚动辅助 iframe 的高度参数错误；最终回归已通过。READY 依据最终版本和对应产物，不依据修复前的中间日志。

**最终状态：READY。**
