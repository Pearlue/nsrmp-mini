# 验收与测试矩阵

> 下表是待实现测试，不表示测试已经存在。当前已执行的只有原有 59 项基线测试及规划期的只读诊断，见 [STATUS.md](STATUS.md)。

## 1. Test-first 核心矩阵

优先把最初的验收测试写入 Issue 指定的 `tests/test_learned_parser.py`。实验脚本级测试放 `tests/test_learned_experiments.py`；不得删除/放宽现有测试来制造“通过”。

| ID | 测试 | 输入/做法 | 必须满足 |
|---|---|---|---|
| L01 | 正确词法提取 | 四种 relation phrase、颜色/形状/object、重复词 | 顺序与 occurrence 保留、未知输入明确失败 |
| L02 | 输入歧义回归 | DESIGN §2 两个句子 | keywords 相同、规则 AST 不同；surface context 不同 |
| L03 | padding/length | 混合长度、缺失 step | pad 不参与 role、logp、entropy；backward GRU 不污染有效表示 |
| B01 | 正确 roles 重建 AST | simple/subj-relative/ref-relative/both-relative、四个 action | builder 输出与 parser.parse 精确相等 |
| B02 | wildcard | the cube、the red object | ANY 与 Filter 顺序正确，object 不提前丢失 |
| B03 | 重复词 | 相同颜色/形状出现多次 | 按 token position 分配，不按词面去重 |
| M01 | 类型与容量 mask | colour→action、第二个 action、同 slot 第二个 noun | 这些选择概率为零 |
| M02 | 全局可完成性 | 有关系无 anchor、剩余词不足以补齐 head/action | 导致死路的选择在采样前被屏蔽 |
| M03 | 完整合法性 | 所有支持模板重复采样，固定 seed | 不产生非法 DSL，不规则兜底 |
| M04 | 合法候选覆盖 | 小输入全枚举与 DP 对照 | 所有语法合法轨迹仍有可达路径 |
| M05 | 不暗藏语义答案 | 同句的 subject/ref 互换候选 | 语法合法但语义错误的候选没有被 mask 排除 |
| P01 | log-prob 正确 | 小手工 logits + 固定采样 trace | 与 masked 条件概率逐项求 log 后求和相等 |
| P02 | 梯度连通 | logp.backward、entropy.backward | 连接到 Parser；不经 `.item()`/重建 tensor 断图 |
| P03 | joint probability | 不同 trace 可能生成同一 AST | 明确 trace logp；不误当 program marginal |
| G01 | REINFORCE 方向 | 两选择固定 reward 的 toy | 正 advantage 推高对应概率，负 advantage 反向 |
| G02 | baseline 边界 | K=4 RLOO、K=1/B=1 EMA | 当前样本 reward 不进入自身 baseline；无除零 |
| G03 | detach 边界 | 仅 parser_loss backward | Concepts/AS 无梯度；Parser 有梯度 |
| G04 | task-loss 路径 | 固定 sampled program 的 task loss | Concepts/AS 有梯度；Parser 无穿过离散选择的直接梯度 |
| G05 | per-example reduction | 含 padding 的同一 batch | new_loss.mean() 与 old loss_fn 一致 |
| I01 | 多步执行 | 1、2、5 steps 与 padded steps | 下一步使用预测 boxes；缺失步不移动物体 |
| I02 | 受控 tiny training | 小型 policy + 确定性 executor/reward | 精确评估的期望 reward 提高，而不只是 policy loss 下降 |
| I03 | 真实联合 smoke | 小合成任务，不读 gold programs | 完成前后向、无 NaN、记录 reward 与 program accuracy |
| N01 | 无 gold 属性读取 | Example.program/subjects/refs/n_steps 属性设为抛错 | collate_learned + train_learned 能继续执行 |
| N02 | 无规则 Parser 依赖 | patch parse/parse_roles 以及消费模块已导入的别名为抛错 | 完整 learned 训练通路不触发它们 |
| N03 | 标注扰动不影响更新 | 固定训练 RNG，改变隐藏的 gold annotations | forward、reward、梯度、参数更新相同 |
| E01 | 评估指标 | 手工已知 AST/boxes | exact-match、IoU-M、simple/complex 分母正确 |
| E02 | 推理协议 | greedy 主表、seeded sampling 辅表 | 不用 target loss 在 K 个预测中选最优 |
| E03 | 随机基线 | 相同 mask 下均匀合法角色采样 | 实测 chance，不写成 1/7 |
| C01 | 默认兼容 | 原有 59 tests，不启用新开关 | 全部通过，旧模型集合和 metric key 不变 |
| C02 | 新实验集成 | opt-in + tiny 3 seeds + tmp_path | 新 row、accuracy、std、runtime 与配置全写出 |
| C03 | RNG 隔离 | 开/关额外评估与 logging 的对照 | 不改变后续训练采样与旧模型随机数行为 |
| C04 | 非覆盖结果 | 新目录与旧 results 对照 | 原有 tracked 数值、图、JSON 无变化 |

## 2. Gold 隔离测试的解释

允许在 builder 单元测试中提供正确 roles，验证确定性函数；允许在评估中用规则 Parser 作参考答案。这两种情况不是训练监督。

不允许在主训练中计算 gold-role 交叉熵、用规则程序选择候选、读取 gold subject 帮助 grounding，或先以正确程序预训练执行器再把该条件隐藏起来。

Poison test 必须覆盖数据构建、模型、loss 和 optimizer step 的**完整路径**。只在 forward 检查没有访问 `batch.programs` 不够，因为旧 `Batch.__len__` 和 `Example.n_steps` 已经间接依赖它。

训练不读取 gold program 指训练消费路径。合成数据生成器为了生成最终场景而内部使用程序是现有任务定义，不需要禁用或改造。

## 3. 如何避免不稳定测试

先用固定小 policy 和确定性 toy executor 验证更新方向，能枚举时直接计算期望 reward。将随机的长训练性能实验与快单元测试分开；固定数据 seed、初始化 seed 与采样 seed。不能用“某一次 reward 恰好变高”判定 REINFORCE 正确。

真实 tiny joint smoke 至少验证数值有限、无泄漏、梯度路径与日志；完整模型是否学会语义属于实验结果。若要加入严格提升阈值，必须先在多个固定 smoke seeds 上建立稳定证据，不能为碰巧通过而反复换 seed。

## 4. Program accuracy 定义

主表：**greedy whole-instruction exact-match**。每条 instruction 的全部 Move AST 与参考程序相等才记 1；对 B 条 instruction 平均。simple/complex 沿用仓库的 example-level 定义，多步中任一步复杂则该 example 属于 complex。

补充：按有效 step 的 exact-match，以及每条指令独立采样若干次得到的 sampled exact-match 均值。采样次数与 evaluation RNG 必须记录；不能混用 greedy 和 sampled 数字。

Builder 采用固定 canonical tuple 顺序。语义等价但 AST 不同可以另设诊断指标，但不能用它悄悄替换主表的 exact-match。

模型可能具有较高任务成功率、较低程序准确率；两项一起报告，不能从 IoU-M 单独推断 Parser 已学会正确语言结构。

## 5. 三种子实验协议

- 主训练 seeds 为 0、1、2，保持上游数据生成与规模；先固定配置再运行所有 seeds。
- 新增独立 validation seeds，例如 5000+seed；不得用 1000+seed 的 test 或泛化 test 调参。
- 同时记录 train/test split、generator 参数、模型/采样 RNG、软件版本、threads 与 commit。
- mean ± sample std 使用 `ddof=1`；单 seed 标注 std=N/A。
- 完整报告成功与失败 seed，NaN/发散不得从均值中静默删除。
- 同约束的 uniform-masked random policy 用于估计 chance；它未必在所有合法完整程序之间均匀，须使用准确名称。
- 新模型 K=4 增加计算成本，必须附时间与总执行次数；不宣称与 K=1 模型 compute-matched。
- 主表使用最后一个 epoch；若以后改用 checkpoint selection，只能预先指定 validation scene metric，不能以 program 标签挑 checkpoint。

## 6. 新表格式（模板，非结果）

| Model | IoU-M overall | simple | complex | Program accuracy | Source / protocol |
|---|---:|---:|---:|---:|---|
| Rule-parser NS | 0.776 ± 0.005 | 0.779 ± 0.006 | 0.774 ± 0.003 | by construction | 上游已有结果；非本轮重跑 |
| Parse-matched neural | 0.786 ± 0.011 | 0.788 ± 0.010 | 0.784 ± 0.012 | N/A | 上游已有结果 |
| Raw-sentence GRU | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.004 ± 0.001 | N/A | 上游已有结果 |
| Learned-parser NS, surface_context | not run | not run | not run | not run | 待实现，greedy 主评估 |
| Learned-parser NS, keywords_only | not run | not run | not run | not run | 可选词法信息消融 |

这些旧数值来自上游 README Table II analogue；不能把此模板当作已经完成的新实验。

## 7. Runtime 与产物审计

记录 data generation、training、evaluation 和 end-to-end wall time，精度至少到秒；注明是否包括环境安装、额外调参和绘图。记录主机 CPU、内存、PyTorch 版本、线程数和并发任务状况。

上游 README 的 8 分 6 秒只代表它已经报告的旧实验，不是新 REINFORCE 运行时间保证。新版本先用 smoke/profile 估计热点，再完成真实三种子计时；若不符合 laptop 运行目标，优先缓存词法/语法检查、批量执行、减少重复评估，不隐瞒减少的 epochs/data。

JSON、Markdown table、plot 必须由同一原始记录生成，校验 seed 数、指标范围与 N/A 标记。运行时间或曲线缺失时验收不完整。

## 8. Issue Definition of Done 对照

- [ ] 在贡献分支完成实施并向上游开 PR；不推 main。
- [ ] 新 test-first 文件覆盖所有 Issue 指定测试和上表关键边界。
- [ ] 原有 59 tests 保持通过；新 tests 的数量与结果真实记录。
- [ ] run_experiments 的 opt-in 入口包含 learned-parser model，3 seeds。
- [ ] 新表报告 overall/simple/complex IoU-M 和 program accuracy。
- [ ] README 以 Our results 说明成功、失败、改动和局限。
- [ ] 原有数值、测试和默认调用不回归。
- [ ] 完整运行时间与 laptop 环境已记录。
- [ ] 无 `.env`、凭证、venv、缓存或大权重进入提交。
- [ ] 若未超过 chance，完整记录已尝试配置与 accuracy 曲线。

这些框在规划阶段保持未勾选。完成文档提交不代表完成 Issue。
