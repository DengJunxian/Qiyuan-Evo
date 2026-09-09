# DeepSeek 实连与答辩演示记录

2026-09-09，三类固定任务在原生 openJiuwen 工作流中执行；提示词优化真实调用 DeepSeek V4.1 Flash，候选配置经同输入复评后用于下一轮。原始配对结果、独立核验、经验引用和模型调用记录均随仓库默认案例保存。

## 实际结果

| 任务 | 验证完成 | 规则评分 | 执行成员 | 优化用量 |
|---|---|---|---|---|
| policy_report | 3/5 → 5/5 | 70.0% → 100.0% | 5 → 6 | 1,028 词元 |
| historical_analysis | 3/6 → 6/6 | 55.0% → 95.5% | 5 → 7 | 997 词元 |
| regulatory_planning | 3/5 → 5/5 | 58.0% → 100.0% | 5 → 6 | 1,160 词元 |

规则评分衡量任务完成和证据质量。三种机制联合形成候选，以上对照不单独归因于模型或某一种机制。任务执行成本在“完整指标、成本与评分细则”中保留，云端优化与候选试跑成本单独记入训练开销。

## 模型调用

- 接口：`https://api.deepseek.com/v1`。
- 实际调用并返回的临时型号：`deepseek-v4.1-flash-expires-on-0910`。其可用期限由服务商控制；重跑时可在本地 `.env` 或 Streamlit Secrets 里填写仍可用的模型 ID。
- SDK：`openjiuwen 0.1.17.post1`，`FeedbackPromptBuilder.build`。
- 温度为 0；记录输入哈希、SDK 实际请求消息哈希、响应哈希、时间戳和 SDK 返回的真实词元用量。
- `LLM_INVOKE_INPUT / OUTPUT` 回调按 client_id 隔离，只导出摘要哈希和用量，不导出密钥或隐藏推理。
- LLM 参与提示词优化；规划、审查、报告和市场数值使用现有程序及金融工具。

## 首次运行的问题与修复

首次连续跑三类任务时，历史任务在 SDK 网络调用处返回 Connection error，按设计降级为本地契约修复。该次记录保留在本地 `outputs/deepseek_defense`，未标记为模型成功。关闭跨事件循环的共享 HTTP 连接后，从全新经验库重跑全部固定任务；三次模型调用均成功，候选全部通过复评。默认案例按任务顺序选取修复后第一组结果，没有按分数筛选。

## 金融事实

- 政策组与无政策组的期末收益差为 +0.00 个百分点。
- 回放与观测行情的方向一致率为 58.8%，归一化均方根误差为 0.0402。
- 比较 3 个方案后，推荐暂不干预；相对不干预的最大回撤降低 0.00 个百分点。

政策仿真与历史条件回放的边界、合成条件及数据来源仍在各实验报告中保留。零收益差和零风险改善没有改写为正效果。

## 复现

本地配置 `.env` 中的 `QIYUAN_LLM_API_KEY`、`QIYUAN_LLM_BASE_URL`、`QIYUAN_LLM_MODEL`，或在 Streamlit Secrets 中配置同名顶层字段。密钥不随仓库上传。

```bash
python scripts/run_competition_benchmark.py --task all --mode compare --seed 42 --rounds 1 --backend openjiuwen --cloud-optimizer --output outputs/new_deepseek_run
python scripts/verify_competition_evidence.py --input outputs/new_deepseek_run
streamlit run app.py --server.port 8502
# 在另一终端截取当前默认案例页面
python scripts/capture_competition_gallery.py --url http://localhost:8502 --output output/playwright/gallery
```

不带 `--cloud-optimizer` 可运行本地规则优化；直接打开应用不需要模型密钥，默认展示已归档实验。模型输出不保证逐字重复，已保存的 evolved 配置可在技术页按原配置复跑，确定性金融产物应一致。

## 校验

- 完整测试：52 项通过（含新 SDK 调用用量回归测试）；随后新增截图与导航测试，最终界面相关 10 项全部通过，共覆盖 54 项测试。
- 三类实连实验全部通过；证据核验脚本 16 项检查全部通过。
- 新版首页按钮另起一次联网演示并完成：实验 `exp-4bacb57ce74b456f95f743f157445cb6`，模型优化用量 1,606 词元，候选通过复评。
- 实际浏览器检查了总览、协作、三种演进视图、实验、截图翻页与技术页；1024 像素宽窗口未出现横向溢出。
- 浏览器直接打开主要页面，原图保存在 `static/competition_gallery`，每张图片绑定实验编号和 SHA-256。

## 实验编号

- `policy_report_v1`：`exp-5641d623ad504b1fa60a6a87e364fcae`。
- `historical_analysis_v1`：`exp-41feddc0162a434bb5ec33081f48a7d3`。
- `regulatory_planning_v1`：`exp-8806ebf3e5d14904bcf65401998065c6`。

源码哈希：`51927b4d74244226dd172d5932a667416a65defa1f8a0b25e9d315bd959cb30c`。
