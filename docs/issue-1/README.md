# Issue #1：Learned Parser + REINFORCE 开发规划

状态：**规划完成，功能尚未实现**。本分支只添加文档与导航，不改训练代码、测试代码或已有实验结果。

| 项目 | 内容 |
| --- | --- |
| 上游 | `kwesi07/nsrmp-mini` |
| Fork | `Pearlue/nsrmp-mini` |
| 规划分支 | `docs/issue-1-learned-parser-plan` |
| 本地目录 | `~/projects/nsrmp-mini` |
| 基线提交 | `a099629d9d3de5ef0f53566f36938c7fb6709377` |
| 关联任务 | [Issue #1](https://github.com/kwesi07/nsrmp-mini/issues/1) |
| 目标机器 | Apple M4 Pro，48 GiB，CPU 优先 |
| 基线检查 | Python 3.12.14；59 个原有测试通过，19.12 秒；未重跑完整模型实验 |

## 阅读顺序

1. [完整开发方案](DEVELOPMENT_PLAN.md)：范围、模块职责、文件改动、数据接口、合法程序采样、梯度路径、风险、实验和交付。
2. [实施与验收清单](IMPLEMENTATION_CHECKLIST.md)：按依赖顺序推进，包含测试矩阵、未来命令和 PR 完成条件。
3. [基线检查记录](BASELINE_CHECK.md)：本次实际完成的环境准备和只读诊断，不把计划当成实现。

## 三个关键决定

- 一个开发者可以负责整个 Issue；实现按“数据边界 → Parser/Builder → 模型集成 → REINFORCE → 实验”的顺序推进。
- 保留现有默认实验路径，新增 learned-parser 显式选项和独立结果目录，防止现有模型与结果回归。
- 先实现 Issue 指定的关键词输入；同时记录其信息丢失风险。保留连接词的输入只能作为明确标注的额外变体，不能偷偷代替主实验。

## 本次不做的事情

不实现算法；不生成新的 IoU-M 或 Program Accuracy 数字；不替换原有结果；不向上游推送 `main`；不自动创建上游 PR 或发布评论。

## Summary for collaborators

This documentation-only branch proposes an end-to-end implementation of Issue #1: a constrained BiGRU role policy, deterministic role-to-DSL construction, REINFORCE training, leakage tests, and three-seed evaluation. It preserves the existing models and results. The upstream 59-test suite passes locally. A verified keyword-collision example and a leave-one-out reward baseline are included as explicit design considerations. No learned-parser implementation or new training result is claimed.
