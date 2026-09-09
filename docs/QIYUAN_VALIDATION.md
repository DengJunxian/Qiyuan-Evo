# 启元 Qiyuan-Evo 验收记录

> 本文件保留第一阶段的验收快照。当前驾驶舱、测试及源码身份以 [COCKPIT_VALIDATION.md](COCKPIT_VALIDATION.md) 为准。

验收日期：2026-09-09。Python 3.12.13，openjiuwen 0.1.17.post1，macOS arm64。

## 实际执行结果

通过 `python scripts/run_competition_benchmark.py --backend openjiuwen --rounds 3` 执行 3 类任务各 3 轮，共 9 组固定 baseline / evolved 对照。首轮分别晋升 v2，后两轮复用历史成功配置并拒绝无新增改进的晋升。全部 9 次 evolved 运行在冻结规则下成功。

| 任务 | Baseline 质量 | Evolved 质量 | 质量增量 | Agent 数 | 工具调用 | 基线/演进中位时延（秒） |
|---|---:|---:|---:|---|---|---|
| policy_report_v1 | 0.7000 | 1.0000 | +0.3000 | 5 → 6 | 4 → 6 | 4.461 / 12.558 |
| historical_analysis_v1 | 0.5500 | 0.9550 | +0.4050 | 5 → 7 | 4 → 7 | 4.438 / 0.014 |
| regulatory_planning_v1 | 0.5800 | 1.0000 | +0.4200 | 5 → 6 | 4 → 8 | 5.227 / 18.628 |

质量分衡量冻结的完成度、证据、数据一致性、任务指标和复现验证。通用 baseline 未执行额外对照与独立复现，不能将质量增量解读为金融预测准确率提升。政策与监管任务增加实验后，工具调用、步骤和时延上升；历史任务切换到适合该任务的条件回放工具后时延下降。时延是本机墙钟观测，运行期间存在其他测试负载。

历史任务实测：归一化 RMSE 0.040213，MAE 0.029855，方向一致率 58.82%。输入为 18 个交易日的冻结上证指数观测数据，宏观面板使用明确标识的 synthetic fixture。

监管任务推荐 `no_action`；实测最大回撤降低 0.000000，干预成本 0.00。三个候选方案在该固定场景下没有回撤改善，因此保留不干预建议，没有调整种子或更换评价指标。

所有任务角色未调用 LLM，token usage 为 0，token reduction 为 null。真实使用 openJiuwen 的部分是 Workflow 核心编排与并行节点执行。云端 Prompt 优化代码调用官方 FeedbackPromptBuilder，但本次无模型密钥，因此实际优化采用 LOCAL FALLBACK 执行契约修复。

## 可复现证据

- 数据 SHA256：`48abaf185050544cf93ef4adf7dcebab9225555fd4c8ac2be2b3b2b12481fb3c`。
- 源码/配置 SHA256：`371f021404da362a8cebf3fb5c8040850c43b9c13f7fbfebb460cb4d3732c8f7`；与当前代码重新计算值一致。
- `outputs/competition_benchmark/summary.json`：9 组评分与经验引用。
- `run_trace.json`：21 条任务 trace（9 baseline、3 candidate、9 evolved）。
- `evolution_trace.json`：9 条实际 Controller 执行记录；优化调用不与任务计费混淆。
- `before_after.json` / `team_topology.json`：相同任务、seed、数据、角色模型配置的比较。
- `evolution_history.json` / `prompt_registry.json`：旧/新 Prompt、diff、理由、评估和 accepted/rejected。
- `source_manifest.json`：源码文件级哈希。各 `runs/` 子目录保存独立仿真的 request/result/log。
- `outputs/source_audit.json`：相关 Python 源码的静态语法、定义与哈希清单。

复现要求语义产物和市场计算结果一致；UUID、时间戳和墙钟时延不要求字节一致。通过原记录复跑时检查源码哈希；新实验从新输出目录开始，已有目录会读取历史经验。

## 测试与界面

测试覆盖三类实跑、真实 SDK、DAG 依赖和并行、可序列化 trace、相同 seed、Prompt 实际影响规划、工具路由、拓扑变化、基线隔离、SQLite 晋升并发保护、快照不可变性、版本复跑、错误归因、重规划、无提升候选拒绝、缓存绕过、缺密钥和模型故障回退。

Streamlit AppTest 覆盖比赛对照结果、故障 trace 展示及原成果展示、系统总览、政策实验、历史验证和研判分析页面。另通过真实浏览器查看页面并触发对照运行。最终测试命令和数量在本文件末尾记录。

## 实现边界

- 云端 Prompt 优化尚未使用真实模型密钥完成在线验证；SDK 故障回退路径已测试。
- Planner/Critic/Reporter 是确定性计划、规则审查和证据合成；尚未实现自由自然语言的 LLM 动态规划与生成式评审。
- 自动拓扑演进已支持已注册专家的 spawn、拆分、parallelize 和 reroute；未实现自动 retire/merge 或任意新角色代码生成。
- Prompt、工具、拓扑联合候选能够改变下一次 AgentTeamConfig；未完成逐因素消融和未见任务泛化实验。
- 经验库存储与回用在本地 SQLite；未实现跨机器共享或分布式 Agent 集群。
- 历史模块属于含合成宏观假设的一步条件回放；本次未做长期预测或金融因果效应验证。
- 原有 C++ 撮合扩展未编译；比赛实际运行的是仓库原有 Python 订单簿与隔离 IPC 撮合。

## 文件变更清单

- `.env.example`
- `.gitattributes`
- `README.md`
- `agents/diagnostic/diagnostic_agent.py`
- `agents/manager_agent.py`
- `agents/report/report_agent.py`
- `agents/roles/news_analyst.py`
- `app.py`
- `benchmarks/competition/sse_2024_policy_window.csv`
- `benchmarks/competition/sse_2024_policy_window.manifest.json`
- `core/competition_service.py`
- `core/simulation_factory.py`
- `core/team_evolution/__init__.py`
- `core/team_evolution/backend.py`
- `core/team_evolution/benchmark.py`
- `core/team_evolution/config.py`
- `core/team_evolution/evaluation.py`
- `core/team_evolution/evolution.py`
- `core/team_evolution/models.py`
- `core/team_evolution/provenance.py`
- `core/team_evolution/runtime.py`
- `core/team_evolution/simulation_worker.py`
- `core/team_evolution/store.py`
- `core/team_evolution/tools.py`
- `docs/QIYUAN_IMPLEMENTATION_PLAN.md`
- `docs/QIYUAN_VALIDATION.md`
- `pytest.ini`
- `requirements-dev.txt`
- `requirements-lock.txt`
- `requirements.txt`
- `scripts/run_competition_benchmark.py`
- `tests/test_competition_ui.py`
- `tests/test_openjiuwen_integration.py`
- `tests/test_team_evolution.py`
- `ui/policy_lab.py`
- `ui/team_evolution.py`

## 最终检查记录

- `.venv/bin/python -m pytest -q`：28 passed，129.44 秒；3 条依赖库弃用警告，没有测试失败。日志：`tmp/pytest_verified.log`。
- 最后补充复跑错误的界面保护后，执行 `.venv/bin/python -m pytest -q tests/test_competition_ui.py`：2 passed，12.33 秒。涵盖工具故障和复跑异常下保留可用页面。日志：`tmp/pytest_ui_final.log`。
- `.venv/bin/python -m pip check`：No broken requirements found。
- `compileall`：agents、core、engine、policy、ui、scripts、app.py 编译检查通过。
- `git diff --check`：通过。
- 重启最终 Streamlit 代码后，在真实浏览器点击政策任务对照，显示 OPENJIUWEN / 0.1.17.post1 / Workflow 完成，评分 0.7 → 1.0，历史经验入口及实际 5 → 6 Agent 拓扑正常。
- Git diff 相对于收到的源码本地快照；原目录没有 Git 元数据，未将该快照冒充上游提交，未提交或推送改造代码到 GitHub。
