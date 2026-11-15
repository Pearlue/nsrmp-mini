# 当前状态与证据

## 本次任务

用户要求：在本机 `projects/` 下建立贡献仓库，写完整开发计划和方案，并提交到远端。本次只编写规划文档，未实现 learned parser 或 REINFORCE。

## 仓库基准

- 上游：`kwesi07/nsrmp-mini`。
- 贡献 fork：`Pearlue/nsrmp-mini`，已验证其 parent 为上游仓库。
- 分支：`docs/issue-1-development-plan`；基于 upstream/main。
- 审核基线：`a099629d9d3de5ef0f53566f36938c7fb6709377`。
- 本地目录名：`nsrmp-mini-issue1`，位于用户的 `projects/` 下。
- 原因：检测时同名 `nsrmp-mini` 目录被另一项操作建立，故保留其状态，用独立 clone 隔离本轮工作。没有删除、reset、stash 或覆盖那个目录。
- 本轮不向上游 main 推送，也不创建实施 PR。

## 已完成并验证

- 已读取当前 Issue #1、README、核心代码、实验入口与相关测试。
- 已验证 GitHub 登录账户和 fork 关系，并克隆到新的独立本地目录。
- 已创建独立规划分支，配置 origin=fork、upstream=原仓库。
- 已创建 Python 3.12 venv，安装未更改的 requirements.txt。
- 已运行原有测试：`59 passed in 18.84s`。
- 已用现有规则 Parser 验证 DESIGN §2 的两个句子：均能解析，生成的 Move/AST 不同。
- 已编写总体方案、技术设计、文件级实施计划、test-first 验收矩阵及贡献指导。

## 实测环境

| 项目 | 实测值 |
|---|---|
| CPU | Apple M4 Pro |
| 统一内存 | 48 GiB |
| 架构 | arm64 |
| Python | 3.12.14 |
| PyTorch | 2.15.0 |
| NumPy | 2.5.4 |
| Matplotlib（安装记录） | 3.11.2 |
| pytest（安装记录） | 9.2.0 |

虚拟环境仅在本地，受 `.gitignore` 排除。没有为这次规划升级依赖或改写 requirements。

## 当前未完成（不得误报）

- 新的 learned_parser / learned_batch 模块、LearnedNSModel。
- REINFORCE、合法性 mask、surface-context lexer 的代码实现。
- 新增单元测试与数据泄漏防护测试。
- 三种子实验、学习曲线、模型性能比较与新模型运行时间。
- 新实验结果的 README Our results 段落。
- 上游 implementation PR 及 Issue 关闭。

## 规划期发现

1. 当前 `collate()` 会同时运行 parse 与 parse_roles；只替换模型读取字段不足以证明无 gold 泄漏。
2. `Example.n_steps` 和 `Batch.__len__` 分别依赖 program/programs，新训练路径需独立处理。
3. keyword-only 输入会丢失关系子句位置，存在相同输入对应不同 AST 的反例；建议保留原文 context。
4. self-inclusive reward mean 在单样本时 advantage 恒零；方案采用 RLOO / 历史 EMA。
5. 旧 phase-2 测试要求精确模型集合，新增模型必须 opt-in。
6. 旧 results 下的 checkpoint ignore 不是所有嵌套路径通用；未来保存新目录权重前需补充 ignore。

## 文档依据

- [Issue #1](https://github.com/kwesi07/nsrmp-mini/issues/1)，本次通过 GitHub CLI 读取，状态 OPEN。
- [固定基线的源码](https://github.com/kwesi07/nsrmp-mini/tree/a099629d9d3de5ef0f53566f36938c7fb6709377)。
- 技术 API 来源列于 [规划入口](README.md)。

以上测试结果只证明原有仓库在本机可运行，不证明尚未实现的设计已经训练成功。
