# 展示动线与数据约定

本轮按 anti-defensive-writing-skill、humanizer-zh 和 product-design 整理前端；实验任务、种子、评价规则及市场内核未调整。

## 使用场景

评委第一次打开系统，需要看清成员分工、审查发现、配置改变和复评结果。原页面把评分与英文技术标签放在主视觉，具体任务和市场结果藏在详情里。调整后的首页直接打开一个已完成案例；用户可以立即检查记录，也可以新建一次运行。

操作对象是实验记录。“查看”读取已有结果；“重新运行”建立新的实验目录，保留以前的记录。默认案例使用 12 个模拟交易日，运行设置可切换到 6 日快速规模。相同配置复跑入口在协作记录中，固定记录中的任务、种子和团队配置。

## 页面职责

| 页面 | 用户要完成的事 | 下一步 |
|---|---|---|
| 驾驶舱 | 看具体任务、初轮缺项、完成验证数和耗时 | 查看任务与审查记录 |
| 协作记录 | 看谁执行、传递什么、审查发现什么 | 查看团队如何调整 |
| 演进对照 | 看提示词、工具和分工变化，核对两轮结果 | 查看其他实验案例 |
| 实验案例 | 比较政策研判、历史分析和监管优化 | 打开案例或运行实验 |
| 金融风洞 | 查看团队调用工具得到的市场结果 | 使用原研究工具 |
| 历史验证 | 默认查看历史观测与回放；新建标签保留原研究表单 | 技术资料 |
| 技术资料 | 核对运行后端、版本与数据来源 | 下载证据包或运行脚本 |

## 产品决策依据

- `rule/value-before-interruption`、`rule/smallest-intervention`：首次打开即读内置案例，不要求配置模型、联网或先点按钮。
- `rule/one-primary-action`：首页突出查看记录；运行方式和规模放进运行设置。实验卡保留查看入口，重跑选项折叠。
- `rule/inline-before-modal`：提示词全文、原始记录、评分细则放在就地展开区。
- `rule/control-matches-cardinality`：运行方式、实验规模与两轮阶段使用单选控件。
- `rule/preserve-mental-model`：保留原页面内部标识、深链接与研究工具入口；缩短显示名称。
- `rule/cover-reachable-states`：保留运行中、失败恢复、无数据、损坏归档、进程中断和刷新恢复处理。
- `rule/navigation-vs-action`：沿用 Streamlit 按钮与 URL 同步导航；完整浏览器历史语义尚未迁移为原生链接，避免破坏当前实验上下文。

## 数据真实性

默认案例位于 `benchmarks/competition/reference/`，由 `scripts/package_reference_experiments.py` 从真实实验归档导出。固定取每类任务按时间排列的第一个配对，不按评分筛选。源目录为 `outputs/presentation_reference/`；全部三个实验使用原生 openJiuwen 工作流、种子 42。

读取时校验文件 SHA-256，再从执行记录重新计算评分。原始运行编号、实验编号、价格、审查意见、提示词差异、实际用时和经验引用保持不变。历史经验是只读快照；新实验继续写入 outputs 下的独立经验库。

首页以通过验证项数呈现任务完成情况；规则评分保留在两轮对照。它不是行情预测准确率。政策案例期末收益差为零，监管案例风险改善为零，历史案例方向一致率约 58.8%，均在页面直接展示。没有添加价格噪声或修改负结果以美化曲线。

## 可达状态

已完成案例：本地载入并标注日期。运行中：消费真实观察事件。失败：保留已完成案例并提供重新运行入口。损坏：校验失败，不展示该归档分数。空目录：提示没有实验结果，提供运行入口。刷新：用页面与实验编号恢复；进程中断：恢复成失败状态，不冒充正在执行。

## 本轮验证

- 依赖检查无冲突；Python 编译与 git diff 空白检查通过。
- 三类任务全部使用 openJiuwen 重新执行，三组配对通过执行证据检查。
- 全套测试 49 项通过；运行态标签修正后的前端、默认数据及故障检查 8 项通过。
- 浏览器检查七个主页面、两轮指标、经验引用、历史曲线和运行状态。
- 网页完整重跑用时 31.32 秒。实验编号为 `exp-e8677a8249b6453289742933b92f9f72`；两轮市场产物与同种子参考案例逐项相同，新的耗时独立记录。
- 默认数据校验覆盖文件被改动、无联网运行环境、空目录与失败任务。

部分 Streamlit / Plotly 内置工具栏仍保留英文；原始 JSON 和源码标识按原样提供。移动端未做专项验收。原研究工作台保留了现有复杂表单，比赛主入口不再默认打开这些表单。

## 本轮修改范围

`app.py`、`ui/team_evolution.py`、`ui/components/evolution.py`、`ui/presentation.py`、`theme/competition_dark.css`；原业务页面中的说明文案位于 `ui/policy_lab.py`、`ui/backtest_panel.py`、`ui/dashboard.py`、`ui/demo_wind_tunnel.py`、`ui/regulator_optimization.py`。

默认数据读取与校验位于 `core/competition_cockpit.py`；新增 `scripts/package_reference_experiments.py` 和 `benchmarks/competition/reference/`。测试修改位于 `tests/test_competition_ui.py`、`tests/test_competition_cockpit.py`、`tests/test_presentation_reference.py`。使用说明同步至 README。
