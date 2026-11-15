# Issue #1：Learned Parser + REINFORCE 开发方案

**状态：规划完成；模型实现、learned-parser 测试和新实验尚未开始。**

目标是在现有二维合成任务中，把手写 Parser 替换为一个从最终场景损失学习角色分配的 Parser；复用 Concepts、Executor 和 ActionSimulator。不是完整复现 NSRMP，也不承诺一定优于已有模型。

## 阅读顺序

1. [DESIGN.md](DESIGN.md)：数据流、模块边界、张量接口、合法性采样、梯度与损失。
2. [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)：需要改动的文件、实施顺序、分阶段提交与失败处理。
3. [VALIDATION.md](VALIDATION.md)：test-first 验收矩阵、三种子实验、指标与数据泄漏检查。
4. [STATUS.md](STATUS.md)：这次实际做了什么、已验证的环境与测试、仍未执行的内容。

## 一张图看懂

```text
instruction text
      │
      ├── fixed split_steps
      └── fixed lexical scan：keywords + surface context
                           │
                 Embedding → BiGRU
                           │
                   每个 keyword 的 7 个 role logits
                           │
                 grammar-only mask + sampling
                           │
                   role trace → DSL builder
                           │
                     sampled Move programs
                           │
                  existing Executor + AS
                           │
                     predicted final boxes
                           │
                     final-scene loss
                           ├── ordinary backprop → Concepts / AS
                           └── detached reward → REINFORCE → Parser
```

## 两项需要明确说明的设计修正

**关键词去除过多信息。** 单纯去掉 `which is` 等语法信息，会让两句不同指令得到完全相同的关键词序列。推荐保留原文的上下文 token，仅对关键词预测角色；同时保留 `keywords_only` 作为严格对应 Issue 建议的对照。仍然不调用 `parse_roles`，也不提供 gold roles。具体反例与测试见 DESIGN §2。

**Baseline 不应包含当前样本自己的 reward。** 主方案对同一句话采样 K=4 个完整程序，使用其余 K−1 个样本的平均 reward 作 baseline。K=1 时使用上一批次以前的 EMA baseline。这样避免 `b=R.mean()` 的自包含问题，尤其是只有一个样本时 advantage 为零。细节见 DESIGN §6。

这两项是本贡献方案的建议，不是声称 Issue 原文或论文已经采用这些实现；实施 PR 应显式解释。

## 范围边界

| 做 | 不做 |
|---|---|
| 小型 BiGRU、7 角色预测、合法 DSL 采样 | YOLO/ResNet、真实图像输入 |
| 最终场景监督的 REINFORCE 联合训练 | 程序/角色标签监督训练 |
| 一步与现有多步任务兼容 | 学习 sentence splitter、任意深度程序 |
| 三个随机种子、program accuracy、运行时间 | 保证某个 IoU-M、论文完整复现 |
| CPU 优先、保留旧模型与原有测试 | 为此引入 PPO、Transformer、外部 RL 框架 |

## 来源与基准

- [Upstream Issue #1](https://github.com/kwesi07/nsrmp-mini/issues/1)：需求和验收标准。
- 核对代码基于 upstream `a099629d9d3de5ef0f53566f36938c7fb6709377`。
- [README](../../README.md)、[SPEC](../../SPEC.md) 与源码是当前简化实现的依据。
- [PyTorch GRU](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html)、[Categorical / REINFORCE](https://docs.pytorch.org/docs/2.15/distributions.html#score-function)、[detach](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.detach.html) 是接口依据。
- [Policy-gradient baselines](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients)：action-independent baseline 的理论条件。

不依赖论文小节编号作实现依据：Issue 与当前 README 的章节引用并不完全一致。本文仅把当前仓库与 Issue 的简化目标作为实施合同；不把固定 splitter 或此处的角色标注网络等同于论文完整的 Language Reasoner。
