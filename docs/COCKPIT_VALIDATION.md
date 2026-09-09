# 自演进驾驶舱 · 校赛前端验收

验收日期：2026-09-09。Streamlit 1.63.0 / Python 3.12.13 / openjiuwen 0.1.17.post1。

## 新导航与讲解路线

自演进驾驶舱 → 协作过程 → 演进实验 → 三类任务 → 金融政策风洞 → 历史验证 → 技术与复现。

成果展示、系统总览、政策实验、研判分析保留在“金融政策风洞”和侧栏业务入口。监管优化与行为诊断仍在原研判分析中。

打开系统即显示明确标注的已完成实验。点击“查看完整证据链”进入 Baseline 协作和 Critic 反馈；再点击“演进实验”查看候选修改、评分、Prompt Diff、组织代际及经验引用。三个交互内可以查看完整的反馈—配置变更—重新评估证据。

## 实跑与数据来源

- 首页“运行自演进演示”实际点击验收：24.68 秒，固定 policy_report / seed=42 / 6 天市场仿真；后端 OPENJIUWEN，模型与 Prompt 优化明确为 LOCAL FALLBACK。
- 实测 Baseline / Evolved 质量 70.0% → 100.0%，Agent 5 → 6，真实产生 3 项 Prompt、工具和拓扑变更。步骤、调用与时延增加保留，未设置虚假节省。
- 演示产物：`outputs/competition_demos/demo-8a08bb5e61f04d959117ce1d39d3dc17`。包含 before_after.json、event_stream.json、demo_manifest.json、演进历史、经验库和 runs/。浏览器中观察到实际 RUNNING / WAITING 等节点状态、后端事件进度和完成后的自动结果展示。
- 正式参考实验：`outputs/competition_cockpit_reference/`，3 类任务各 3 轮，共 9 组真实 openJiuwen 配对比较。所有 evolved 任务通过冻结评估。
- Prompt/工具/拓扑是联合候选，质量增量不归因到某一项修改；不是金融预测准确率提升。监管任务保留零风险改善和 no_action 建议。
- 源码/配置 SHA256：`6e8d1d334abf5af417d42d4e2589645965184932ceb9ff723e89b4506bad1aea`，已与当前代码核对；数据 SHA256：`48abaf185050544cf93ef4adf7dcebab9225555fd4c8ac2be2b3b2b12481fb3c`。

| 组件 | 数据来源 | 显示语义 |
|---|---|---|
| 实时团队图、运行状态、事件进度 | CompetitionService 的不可变 ExecutionTrace 观测快照 | 后台正在执行；约 1 秒刷新 |
| 首页初始图、时间线、前后对照 | 经 TaskSpec / seed / 哈希配对与 trace 重新评分校验的真实 artifact | 已完成实验，标注原运行 ID、后端和规模 |
| 经验记忆、Times Reused | SQLite Experience Store 与真实 TaskPlan.retrieved_experience_ids | 可追溯引用，不模拟复用次数 |
| 代际与 Prompt / Tool Diff | 已观测的 AgentTeamConfig 和 EvolutionAction | 同任务输入下实际存在的版本；不捏造退役、合并或新代际 |
| 能力 Badge / Runtime | baseline_config、任务定义、能力映射与实际 SDK import/version | 已配置能力和可用状态；不把 READY 当作执行完成 |

## 演示与复现

```bash
.venv/bin/python -m streamlit run app.py
.venv/bin/python scripts/run_competition_demo.py --profile reproducible --task policy_report
.venv/bin/python scripts/run_competition_demo.py --profile offline --scale full
.venv/bin/python scripts/run_competition_benchmark.py --backend openjiuwen --rounds 3 --output outputs/competition_cockpit_reference
.venv/bin/python -m pytest -q
```

REPRODUCIBLE DEMO 使用冻结数据和真实可用 Workflow；ONLINE LIVE 请求云端 Prompt 优化并明确记录缺密钥/超时回退；OFFLINE FALLBACK 强制本地 DAG。完整实验为 12 天市场仿真，答辩演示为 6 天；历史观测窗口不截短。默认新演示使用独立目录，保留全部历史产物。

## 测试与浏览器验收

- 全量：39 passed，136.98 秒，3 条第三方 SDK 弃用警告；`tmp/pytest_cockpit_final.log`。
- 最后界面细节修改后重跑：3 passed，8.53 秒；`tmp/pytest_cockpit_ui_final.log`。
- 测试包括真实任务/SDK、观察者异常和快照不可变性、并行证据归属、原始配置复跑、坏产物/评分不符拒绝、所有新导航、原业务页面、空状态、工具故障、后台失败恢复和历史对照保留。
- 浏览器实际查看七个主页面，以及组织代际、Experience Memory、真实演示运行中和运行后状态。桌面检查约 1280×720。
- 已修复 SVG 被直接 HTML 过滤的问题，改用无 CDN 的本地 iframe；已压缩 Hero，使任务团队成为首屏视觉主体。
- `compileall`、`pip check`、`git diff --check` 通过。服务日志未发现 Traceback / Uncaught / ERROR。

## 当前界面边界

- 当前以桌面答辩为验收目标；手机及极窄窗口未做专项视觉验收，完整轨迹表可能需要横向滚动。
- 后台作业尚不支持服务进程重启后自动续跑；已保存实验可继续回放或重新执行。
- ONLINE LIVE 的真实云端 Prompt 优化未使用实际模型密钥验收；缺密钥和服务故障回退已测试。
- 原业务工具页保留其研究工作台布局与按需联网行为；比赛主按钮使用冻结数据、受控规模及明确模式。

## 本阶段文件清单

- `README.md`
- `app.py`
- `core/competition_cockpit.py`
- `core/competition_service.py`
- `core/team_evolution/runtime.py`
- `core/team_evolution/tools.py`
- `ui/team_evolution.py`
- `ui/components/evolution.py`
- `theme/competition_dark.css`
- `requirements.txt`
- `scripts/run_competition_demo.py`
- `tests/test_competition_cockpit.py`
- `tests/test_competition_ui.py`
- `docs/COCKPIT_DESIGN.md`
- `docs/COCKPIT_VALIDATION.md`
- `docs/QIYUAN_VALIDATION.md`
