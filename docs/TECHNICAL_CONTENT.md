# 技术页面内容来源

技术页面依据用户提供的《启元 Qiyuan-Evo 技术研究报告》技术方案稿更新。报告资料核查日期为 2026 年 9 月 11 日，原文件名为 `Qiyuan_Evo_技术研究报告 (1).docx`，SHA-256 为 `95f2ab4a68b08184c44e4419a068a9dcecf25270c986b8430785c53db9f0fd85`。原文件未修改，也未作为网站公开资源复制。

内容以报告第三至八节的架构、政策机制、协作、自演进和效率设计，以及第十节的可解释性为主。页面把技术方案与当前运行证据分开呈现：架构图为说明性示意，运行状态、版本、模型调用与证据下载继续读取现有服务和归档。

| 页面主题 | 报告依据 | 当前代码核对 |
| --- | --- | --- |
| 双平面与证据底座 | 第三节、第五节 | `core/competition_service.py`、`core/team_evolution/runtime.py`、`tools.py` |
| 提示词和工具策略 | 第六节 | `core/team_evolution/backend.py` 中的 FeedbackPromptBuilder；`evolution.py` 中的工具路径与缓存调整 |
| 专家增补与并行 | 第五、六节 | `evolution.py` 中的角色库增补和拓扑修改；`backend.py` 中的 Workflow 依赖调度 |
| 候选复评与晋升 | 第六节 | `core/competition_service.py` 中的候选执行与复评；`store.py` 中的配置晋升事务 |
| 经验保存与引用 | 第六节 | `store.py` 的检索及 `core/competition_service.py` 的规划配置采用 |
| 政策、市场与计算 | 第四节 | `policy/structured.py` 的 PolicyPackage，`agents/fast_agent_kernel.py` 的 FastAgentKernel |
| 三条可解释链路 | 第十节 | 当前的 artifact、runtime、messages、evolution_history 和证据包下载 |

PolicyIR、完整策略发布与回滚体系、自动合并与裁剪角色、上下文多臂老虎机、记忆检索策略优化、自动学习的模型路由、资源背压、检查点恢复和推理后端优化均按扩展设计表述。没有将报告介绍的框架开发分支组件或相关论文方法列为当前项目已集成的组件。

当前框架适配实际使用 Workflow、WorkflowComponent 和 FeedbackPromptBuilder。TeamRuntime、AgentCard、CommunicableAgent、InstructionOptimizer、SkillExperienceOptimizer 保留在报告接入说明中，并标为拟适配能力。

报告中的三类业务方向与当前基准案例分别说明，继续保留现有政策研判、历史事件分析与监管优化的归档名称、数值和链接。本次内容更新不新增实验，不修改评分或市场数据。
