# Issue #1 完整开发方案

> 状态：设计提案，不是已实现代码。基于上游提交 `a099629d9d3de5ef0f53566f36938c7fb6709377` 与 Issue #1。文中新增类、函数、参数和命令均为拟实现接口，除非明确标为现有。

## 1. 目标、范围与结论边界

### 1.1 要回答的问题

在不向训练提供正确程序的前提下，使用最终场景误差能否训练出一个有效的 Parser？把它接到现有 Concepts、Executor、ActionSimulator 后，IoU-M 与当前三个基线相比如何？

交付整个 Issue：可训练 Parser、合法 DSL 构建、REINFORCE、训练/评估接线、测试、三个随机种子、准确率曲线、运行时间与 README。性能不达预期也可以形成有效结果，但必须排除代码错误、标签泄漏与明显未完成调试。

### 1.2 明确不做

不训练 lexer 或 `then` splitter；不添加图像、检测器、真实机器人、复杂 NLP 框架或大型语言模型；不移植论文的整个层次化 Language Reasoner；不改动原有 baseline 架构来制造优势；不承诺 IoU-M 达到某个数值。

本任务使用合成 2D 特征与 gold boxes。即使成功，也只能说明该受约束设置中能从执行结果学习结构，不能声称复现论文，也不能把相对于一个训练失败的 GRU 的优势单独归因于符号推理。[S1][S2]

### 1.3 现有能力

| 组件 | 现状 | 本次实现策略 |
| --- | --- | --- |
| 数据生成、最终场景 | `data.py` 已有 | 不改生成分布；仍允许生成器内部用程序构造训练目标，但训练侧不读取其程序字段 |
| Parser | `parser.py` 的规则解析 | 作为旧模型与评估 oracle 保留，不用于新模型训练 |
| `split_steps` | 固定按 `then` 分句 | 复用，步骤数从文本计算 |
| Concepts / Executor / AS | 已有可微模块 | 复用原计算与 DSL 格式 |
| Loss / curriculum | 已有最终 boxes loss 与 simple-first | 保持原默认行为，给新训练路径增加逐样本 loss |
| Baselines / results | 已有 3 seeds 与 JSON/图表 | 保留；新结果独立存放 |

原始表中 IoU-M：规则 Parser NS 为 `0.776 ± 0.005`，parse-matched neural 为 `0.786 ± 0.011`，raw-sentence GRU 为 `0.005 ± 0.001`。它们是仓库已有结果，不是此次新跑出的数字。[S1]

## 2. 总体流程与开发边界

```text
训练输入：文本 + 初始物体特征/boxes + 真实最终 boxes
                         │
       固定 split_steps + 仅词法级关键词提取
                         │
            Embedding → BiGRU → 七角色 logits
                         │
        类型/容量/可完成性 mask → 逐词条件采样
                         │
          角色 trace → 确定性的 DSL Builder
                         │
       与现有格式相同的 tuple[Move, ...]
                         │
       Executor(Concepts) → ActionSimulator
                         │
                 预测最终 boxes
                         │
              逐样本最终场景 loss
              ┌──────────┴──────────┐
              │                     │
      普通反向传播到 VR/AS       detach 后变为 reward
                                    │
                         REINFORCE + entropy bonus
                                    │
                              更新 Parser 参数
```

Builder 和 mask 是人工定义的 DSL 语法约束；词语承担哪个语义角色由网络学习。不得通过“第一个名词短语一定是 subject”等规则，在 mask 中重新实现正确 Parser。

## 3. 预计改动文件

| 文件 | 操作 | 责任与兼容性要求 |
| --- | --- | --- |
| `nsrmp/learned_parser.py` | 新增 | 词法提取、输入类型、BiGRU policy、约束状态、采样、DSL Builder、policy trace |
| `nsrmp/batch.py` | 增量修改 | 增加独立 `LearnedBatch` / `collate_learned`；原 `Batch` / `collate` 不变 |
| `nsrmp/model.py` | 增量修改 | 新增 `LearnedNSModel` 与 policy-aware forward；旧 `NSModel` 不变 |
| `nsrmp/train.py` | 增量修改 | `loss_fn(..., reduction='none')`、独立 `train_learned`、逐句多采样 baseline、诊断日志；原 `train` 路径不改 |
| `nsrmp/metrics.py` | 增量修改 | 新增 learned-model 评估与程序准确率；保持旧 `evaluate` 的返回键与计算方式 |
| `run_experiments.py` | 增量修改 | 显式开启 learned row、独立结果、3 seeds、曲线、运行时；默认旧实验行为不变 |
| `tests/test_learned_parser.py` | 新增 | Issue 要求的核心构造、mask、reward、无 gold 泄漏测试 |
| `tests/test_learned_training.py` | 新增 | 梯度路径、loss reduction、baseline、padding、多步骤、可复现实验 |
| `tests/test_learned_experiments.py` | 新增 | 新 CLI/config、输出 schema、指标、3-seed 调度与不覆盖旧结果 |
| `README.md` | 增量修改 | 最终加入 `Our results`、命令、限制与运行时间；保留原有数字 |
| `results/learned_parser/<run-id>/` | 新增输出 | 新 JSON、表、曲线、配置、runtime、失败诊断；不覆盖原文件 |

原则上不改 `parser.py`、`data.py`、`executor.py`、`baseline.py`、旧测试或依赖。若遇到已证实的接口问题，单独提交最小兼容性修复并记录原因，不顺手重构。

## 4. 数据输入：从源头阻止答案泄漏

### 4.1 不能直接复用当前训练 Batch

现有 `collate` 会调用 `parse()` 与 `parse_roles()`；`Batch.__len__` 读取 `programs`；`Example.n_steps` 读取 `len(program)`；`NSModel.forward` 从 `batch.subjects.shape` 决定循环长度。仅在 forward 中不使用 `batch.programs`，不足以证明新路径不读 gold 程序。[S3][S4]

新增独立路径：

```python
# 拟议数据契约；不是现有 API
LearnedBatch:
    feats: Float[B, N, 32]
    boxes: Float[B, N, 4]
    mask: Float[B, N]
    final: Float[B, N, 4]          # 只用于 task loss
    keyword_ids: Long[B, S, L]
    keyword_lengths: Long[B, S]
    step_mask: Bool[B, S]
    n_steps: Long[B]              # len(split_steps(text))
    is_complex: Bool[B]           # 只用于给定 curriculum / 分组统计
    texts: tuple[str, ...]        # 评估追踪；不包含程序或角色标签
```

`__len__` 使用 `feats.shape[0]`；`select()` 仅索引上述字段。`collate_learned` 不读取 `Example.program`、`Example.n_steps`、`subjects` 或 `refs`。它只从已生成的普通输入和 `final_boxes` 构建数据。

`LearnedNSModel` 只把关键词张量传入 policy；`final`、难度标签、物体语义标签都不能成为 Parser 输入。执行器接收现有数值特征和 boxes，不读取生成器中的真实 colour/shape 字段。

### 4.2 评估标签单独保存

建立仅供评估使用的 `EvaluationTargets`，保存 gold subject、moved mask 和由 `parse(text)` 得到的参考程序。程序准确率只在 no-grad 评估阶段计算，不作为 reward、teacher forcing、mask 条件、预训练标签或超参数选择依据。

开发 unit test 可以使用人工角色和规则 Parser 验证 Builder；这不代表训练可以访问它们。配置选择依据验证集的场景 loss / IoU-M；程序准确率曲线是只读审计，不通过挑选最高 gold-program accuracy 的 checkpoint 变相监督模型。

## 5. Lexer 与一个必须提前记录的歧义

### 5.1 主实验输入

每步按出现顺序保留颜色、形状、`object` 以及规范化的关系词。关系短语采用最长词法匹配，例如 `to the right of → right`、`in front of → front`。词法类别来自现有常量，不调用 `_parse_np`、`parse_roles` 等结构解析逻辑。

重复词必须保留为不同 occurrence；不能转为 set。`object` 必须保留，占一个名词形状槽，构建时映射为 `ANY`。未知词/无法识别的输入应有明确错误路径，不能静默回退到规则 Parser。

### 5.2 已验证的信息丢失反例

```text
A: put the red cube which is behind the green dice to the right of the blue cube
B: put the red cube behind the green dice which is to the right of the blue cube

相同关键词：
[red, cube, behind, green, dice, right, blue, cube]

A 的 action = MoveRight；B 的 action = MoveBehind。
```

这两个文本均能被现有 Parser 解析，但正确 DSL 不同。本次用原始 Parser 进行了只读验证，详见基线记录。此例证明 keyword-only 表示并非一一对应原句结构；尚未测量随机生成数据中这种碰撞的频率。

因此不能预设 complex program accuracy 必然能达到 100%。先实现 Issue 指定的 `keywords_only`，在隔离的分析步骤统计词序列碰撞、不同参考 AST 数量和频率。`sum_x max_p count(x,p) / total` 仅是该样本上的确定性 keyword-only lookup 上界，不是总体泛化准确率的保证。

可选附加变体 `keyword_context`：为 BiGRU 保留原始连接词上下文或词法位置，不给 role 标签、不用它生成硬角色 mask。它是输入信息变化的独立消融，必须单列结果，不能悄悄替换 keywords-only 主实验。本 Issue 不要求扩展到 learned splitter。

## 6. BiGRU 角色 policy

### 6.1 初始配置

拟采用 embedding 维度 32、单层双向 GRU、每方向 hidden size 64、`batch_first=True`、dropout 0。每个词的 128 维双向输出接线性层得到 7 个 logits。它是小模型起点，不是已经验证的最佳设置。[S5]

短序列按长度 pack，或仅对真实步/真实 token 运行 GRU；零长度 padding step 不进入 pack。BiGRU 的反向状态不能被额外 padding 污染。

角色顺序固定为：

| role | 内容 |
| --- | --- |
| `subj_head` | 被移动物体的颜色/形状描述 |
| `subj_rel` | 用来定位 subject 的空间关系 |
| `subj_anchor` | subject 关系中的锚物体描述 |
| `ref_head` | 移动目标参照物的颜色/形状描述 |
| `ref_rel` | 用来定位 reference 的空间关系 |
| `ref_anchor` | reference 关系中的锚物体描述 |
| `action` | MoveLeft / Right / Front / Behind 对应关系词 |

### 6.2 返回接口

```python
PolicyTrace:
    programs: list[K][B][tuple_of_moves]
    trace_log_prob: Float[K, B]   # 保留梯度；真实词和真实步的 log-prob 求和
    entropy_bonus: Float[K, B]    # 保留梯度；真实决策的条件熵均值
    roles: Long[K, B, S, L]       # padding 为 -1，仅供诊断
```

训练用 `mode='sample'`；正式确定性评估用 `mode='greedy'`。每次采样使用实际 mask 后归一化的分布，不能按一个分布采样、用另一个分布计算 log-prob。

这里精确计算的是角色分配 trace 的概率；重复词可能令不同 trace 生成相同 AST。不能未经求和就宣称 trace log-prob 等于 AST 的边际 log-prob。REINFORCE 对采样 trace 优化执行 reward 仍成立，无需计算所有等价 AST 的概率。

## 7. 合法性 Mask 与 DSL Builder

### 7.1 合法性约束

四个 noun slots 为 subject head/anchor、reference head/anchor。颜色只能进入 noun slot，shape/`object` 只能进入 noun slot，关系词只能进入两个 relation slots 或 action。

每个 noun slot 最多一个颜色和一个 shape/object；head 必须有一个 shape/object，颜色可以缺失并填 ANY。anchor 仅在对应 relation 存在时激活，并必须有 shape/object。每个 relation 最多一个，action 恰好一个；每个关键词恰好使用一次。不允许递归 anchor：当前生成器最多是 head 的一层关系，但 subject 和 reference 可以同时带关系。[S3]

Mask 只检查类型、槽容量、依赖和剩余输入的可完成性。不根据 gold 程序、gold 物体、最终位置或 `parse_roles` 筛选；不根据场景中是否恰好有唯一物体来替 Parser 选择语义答案。

### 7.2 为什么不能独立采完再修补

每个词独立 softmax 采样，会出现两个 action、重复颜色、没有 reference 等情况。事后修补会改变实际采样分布，使记录的 log-prob 不再对应执行的选择；大量拒绝重采样也浪费计算。首版采用带状态的逐词约束采样。

```python
# 拟议伪代码
logits = role_head(bigru(embedding(keyword_ids)))
state = empty_slots()
trace_log_prob = logits.sum() * 0
entropy_terms = []
for i in real_token_positions:
    allowed = [False] * len(ROLES)
    for role_id, role in enumerate(ROLES):
        allowed[role_id] = type_and_capacity_ok(state, words[i], role) and \
            has_valid_completion(
                state.with_assignment(words[i], role),  # 新状态，不修改原状态
                words[i + 1:])
    if not any(allowed):
        raise InvalidGrammarState  # 不回退 gold、不静默丢样本
    dist = Categorical(logits=mask_with_negative_infinity(logits[i], allowed))
    chosen_id = dist.sample()     # 评估时为合法候选中的 argmax
    trace_log_prob += dist.log_prob(chosen_id)
    entropy_terms.append(dist.entropy())
    state = state.with_assignment(words[i], ROLES[int(chosen_id)])
assert is_complete(state)
program = role_to_program(state)
```

`has_valid_completion` 用 memoized DFS/DP，key 为当前位置、各槽占用与关系需求；可完成性只依赖剩余词类型和语法，不依赖网络参数。缓存布尔可行性而不是采样值或带梯度 logits。先做完整性测试，再优化，不穷举 `7^L` 的完整树。

记录每句候选规模、解码耗时、强制决策比例。simple 句应保留多个语义候选，至少包含 subject/reference 的错误交换；若 mask 总给出唯一正确程序，说明它可能把解析答案硬编码了。

### 7.3 Builder 与现有 DSL 严格兼容

```python
# 复用已有 noun_phrase_program；slot 中缺失颜色及 object 已规范化为 ANY
# 此代码是设计示意，subject_slots/reference_slots 来自采样后的槽状态

def build_np(colour, shape, relation=None, anchor_colour=ANY, anchor_shape=ANY):
    source = ('Scene',)
    if relation is not None:
        anchor = noun_phrase_program(anchor_colour, anchor_shape)
        source = ('Relate', anchor, relation)
    return noun_phrase_program(colour, shape, source=source)

subject = build_np(**subject_slots)
reference = build_np(**reference_slots)
move = ('Move', action_name, subject, reference)
```

每个 noun phrase 的 canonical 形式是 `Unique(Filter(Filter(source, colour), shape))`，包括 ANY。不要自行交换 Filter 顺序或省略 ANY，否则 exact program accuracy 会受到表示差异干扰。relation 的 anchor 同样包含 `Unique`，不能只输出一个 Filter。

## 8. LearnedNSModel 与现有接口兼容

新增模型复用同样的 Concepts、Executor、ActionSimulator，但 programs 来自 policy。每一步在上一步预测 boxes 上执行，最终 loss 只对整段指令的最终场景计算；不在后续步喂入真实中间场景。

```python
forward_with_policy(batch, n_samples=K, mode='sample') -> LearnedForward:
    predicted_boxes: Float[K, B, N, 4]
    subject_weights: Float[K, B, S, N]
    trace: PolicyTrace

forward(batch) -> (predicted_boxes[B, N, 4], subject_weights[B, S, N])
    # 用 K=1；eval 模式 greedy，保持二元返回兼容习惯
```

训练明确调用 `forward_with_policy`；不把旧 `NSModel.forward` 改成三元/四元返回。真实步数由新 Batch 的 step mask/n_steps 决定，不访问 `subjects.shape` 或 gold programs 的长度。缺失步给 Executor 传 `None`，不改变 boxes，也不贡献 log-prob/entropy。

K 个样本可以逐个执行，避免扩大显存；同一个 minibatch 的 K 次采样期间不能更新参数。之后统一计算 loss，一次 backward、一次 optimizer.step。首版可先以正确的 CPU 循环实现，再测量是否值得拼成 K×B 批次。

## 9. REINFORCE、baseline 与梯度隔离

### 9.1 Task loss 保持原定义

每个样本、每次采样的 loss：

`L[k,b] = sum_i mask[b,i] * (L1(pred[k,b,i], final[b,i]) + beta * (1 - IoU(pred[k,b,i], final[b,i])))`

`task_loss = L.mean()`。不除以物体数，不改原模型使用的 reduction。新增 `loss_fn(..., reduction='none')` 返回逐样本向量；默认 `reduction='mean'` 保持原 API 和数值路径。

### 9.2 推荐 K=4，逐句 leave-one-out baseline

```python
reward = -loss_per_sample.detach()       # [K, B]
baseline = (reward.sum(dim=0, keepdim=True) - reward) / (K - 1)
advantage = (reward - baseline).detach()
policy_loss = -(advantage * trace.trace_log_prob).mean()
total = task_loss + policy_loss - 0.01 * trace.entropy_bonus.mean()
optimizer.zero_grad()
total.backward()
clip_grad_norm_(all_parameters, max_norm=5.0)
optimizer.step()
```

同一句的 K 次采样独立且允许重复；不能为求多样性而按未建模的规则去重。只要执行损失按样本独立计算，leave-one-out baseline 就不使用当前采样自己的 reward，避免 self-inclusion 问题。K=1 调试模式使用更新前的历史 EMA baseline，初值 0；不得把 K=1 的 reward 自己当 baseline，否则 advantage 恒为 0。

Issue 的 `b=R.mean()` 可作为兼容参考实现，但需准确说明：当 batch mean 包含当前样本自身时，iid 情况下 score-function 梯度期望会缩小为原来的 `(B-1)/B`，不严格满足“完全不改变期望梯度”；同句 K 样本包含自身均值也有相同问题。它不等同于独立 baseline。本文选择 leave-one-out 是小范围的估计器修正，不改变学习目标。

log-prob 对完整 trace 求和，不能为了句长稳定随意取平均并继续宣称是同一完整程序概率。entropy bonus 使用每条 trace 的有效决策条件熵均值，以减少长度对强度的影响；这是实用探索正则，不宣称等于所有完整程序分布的精确熵梯度。

### 9.3 明确三条梯度路径

| 项 | Parser 参数 | Concepts / AS 参数 |
| --- | --- | --- |
| task_loss | 离散选择不能直接回传 | 正常反向传播 |
| policy_loss | 通过保留梯度的 trace log-prob | reward 已 detach，不经此项更新 |
| entropy bonus | 更新角色分布 | 不更新 |

程序对象是 Python tuple，离散采样本身无梯度；`Categorical.log_prob` 对 logits 保持可微，因此 REINFORCE 仍能训练 BiGRU。[S6] 用梯度测试验证，而不是只检查 `loss.backward()` 没报错。

## 10. Curriculum、稳定性与有限调试预算

主配置：沿用 60 epochs、batch 64、Adam、cosine decay、前 20 epochs simple-only；视觉/AS lr=3e-3，policy lr=3e-3，K=4，entropy coefficient=0.01，grad clip=5。设置是开发起点，不是已测试结论。

按顺序调试，不能所有变量一起改变：

1. 先验证 lexer/Builder/mask、reward 正负号、采样概率梯度、padding 和无泄漏。
2. 在简单任务运行小规模 sanity check，观察 task loss、reward、entropy、有效率、每个参数组梯度范数。
3. 保持无程序监督的同一设置，只比较 policy lr=1e-3 / 3e-3 / 1e-2；用固定验证场景指标选择。
4. 若仍早期锁死，单独尝试 entropy 0.03；记录 baseline/advantage 的尺度，不把高熵自动当作失败。
5. 若 simple 阶段已学会动作，尝试该阶段后冻结 AS。不要把从规则 Parser 监督管线得到的 AS checkpoint 偷渡到主结果；这种 warm-start 若用于诊断，必须独立标注。
6. 关键词碰撞不能靠更高 lr 修复。输入保留连接词只能作为独立变体，并报告表示发生了变化。

不使用 gold subject、gold roles 或正确程序预训练主实验。概念名称和动作 embedding 的语义可能与 Parser 联合发生置换，或者错误程序偶然产生相同最终场景，因此低 task loss 不必然意味着程序准确率高；必须并报两者。

## 11. 评估、三随机种子与公平性

### 11.1 数据与运行约定

主实验沿用 4000 train / 1000 test、3–5 objects、simple/complex 各半、60 epochs、seeds 0/1/2。保留原 test seed=`1000+seed` 和 generalisation seed=`2000+100*seed+axis_value`。独立验证集拟用 `50000+seed`，不能用最终 test 选超参数。

phase 1 必做；phase 2 必须完成接口、测试和正式 3-seed 评估以覆盖现有多步框架。对象数扩展使用 3–10，步数扩展训练 1–2 / 测试 1–5；不擅自改成论文的 7 步。噪声 sweep 不是 Issue 的必要新增实验。

训练 RNG 与评估 RNG 隔离；采样评估用独立 RNG context，不能消耗训练采样流。记录 seed、配置、软件版本、代码 commit、CPU 线程数与所有尝试。三 seed 报 mean ± sample std (`ddof=1`)，不只选最好 seed。

### 11.2 指标定义

- IoU-M overall / simple / complex 沿用现有 moved-object 汇总方式；不悄悄改为逐场景等权均值。
- 主 Program Accuracy 为整条指令的 exact canonical AST match，所有真实步都相同才算正确。另报逐步准确率，用于定位长指令失败。
- 主 task 指标使用 greedy constrained decoding；greedy 不保证全局 MAP。另报固定随机种子的 sampled program accuracy，满足 Issue 对 sampled programs 的观察需求。
- 程序有效率应为 100%（支持的模板范围内）；失败次数与无候选输入必须暴露，不能忽略后再计算 accuracy。
- chance 由同一 mask 下的随机合法角色策略实测，而不是 `1/7`；可行程序数因句子而异。
- 程序正确率、reward/task loss、entropy 均按 epoch 保存曲线。训练曲线来自训练指标；gold-program accuracy 只在隔离评估阶段观测。

### 11.3 报表安排

| Model | IoU-M overall | simple | complex | Program accuracy |
| --- | --- | --- | --- | --- |
| Rule-parser NS | 保留已发布结果 | 保留 | 保留 | given，不是 learned accuracy |
| Parse-matched neural | 保留已发布结果 | 保留 | 保留 | N/A，无显式输出程序 |
| Raw-sentence GRU | 保留已发布结果 | 保留 | 保留 | N/A，无显式输出程序 |
| Learned-parser NS | 待实际实验 | 待实际实验 | 待实际实验 | 待实际实验 |

表中 TBD 只是计划占位，不得发布成实验输出。可在独立新目录并排展示原始记录与新记录，注明源 commit；若重新跑旧基线，另存运行结果，不覆盖已发布历史数字。

旧 `tests/test_experiments.py` 对 phase 2 的 model keys 和旧指标 keys 有精确断言，因此采用显式 `--include-learned`，默认行为维持原样。新增指标仅放在 learned row / 新结果文件，不修改旧结果 schema。[S7]

## 12. Mac mini 执行策略与结果文件

CPU 优先。现有 Executor 中多处 `torch.zeros/tensor` 默认在 CPU，不能只把模型移到 MPS 就认为完成支持。48 GiB 内存足以准备这个小型实验，但学习是否收敛、额外解码开销和总 runtime 必须实际测量。

原 README 的约 8 分钟是原实现、原机器的历史记录，不是新增 REINFORCE 实验的承诺。K=4 不代表总时间必然精确变为 4 倍；需分别测数据生成、policy encoding、mask/DFS、Executor/AS、backward 和评估。

拟输出：

```text
results/learned_parser/<run-id>/
  config.json                 # 输入模式、seed、K、baseline、lr、线程数、版本
  phase1.json / phase2.json    # 每 seed、每 split 的真实结果
  results_phase1.md / results_phase2.md
  curves.json                 # 每 epoch 的 reward、loss、entropy、program accuracy
  program_accuracy.png
  learning_curves.png
  runtime.json                # 每 seed/阶段与总 wall-clock
  diagnostics.md              # 失败示例、尝试与限制
```

保留父目录原有 JSON/PNG/MD。新 run-id 目录若已存在，默认拒绝覆盖。权重如保存，应放在新目录并由适当 `.gitignore` 排除；当前 `results/*.pt` 未必覆盖任意新层级，实施时须核实再决定最小 ignore 修改。绝不提交 `.venv`、凭据或私人资料。

## 13. 开发阶段与依赖

| 阶段 | 交付 | 退出标准 |
| --- | --- | --- |
| P0 基线与边界 | 环境、原测试、接口与无泄漏测试骨架 | 原 59 测试通过；无程序输入契约明确 |
| P1 Lexer / Builder / mask | 关键词、角色槽、可完成性 DP、canonical DSL | 全模板正角色还原；非法角色被阻止；有合法多候选 |
| P2 BiGRU / trace | 网络、采样/greedy、log-prob、entropy | 梯度存在、概率正确、padding 不污染、可重现 |
| P3 LearnedNSModel / train | 多步执行、逐例 loss、LOO baseline、优化器 | 可控 reward 测试改善；两条梯度路径与无泄漏测试通过 |
| P4 实验与有限诊断 | phase1/phase2、3 seeds、曲线、runtime | 实验可重跑；所有失败与成功如实保存 |
| P5 文档与 PR | Our results、变更审查、上游 PR | 旧测试/数字不变；新测试通过；完整完成清单 |

Builder/mask 的 unit tests 可以与 policy 网络构建交替推进；系统训练效果必须在 P3 联合验证，不声称 mock 通过就代表 REINFORCE 收敛。

详尽测试矩阵、拟议命令与提交顺序见 [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)。不将任何日历期限当成保证收敛的承诺。

## 14. 失败处理与 PR 策略

遇到问题先区分：语法错误 / 梯度断开 / 输入不可辨识 / executor 不成熟 / 优化锁死 / 评估错误。每次只变一个因素，保存配置与曲线。小规模测试不能替代完整 3-seed 报告。

若经过上述有限诊断仍接近 chance，完成负结果报告：有效率、随机策略基线、准确率曲线、IoU-M、尝试列表和 runtime，不删失败 seed，不偷偷使用正确程序训练。这是 Issue 允许的科学结论，但工程验收仍要完成。

当前分支只推送规划。后续实现从上游基线或规划分支创建 `feat/issue-1-learned-parser`，保留计划历史；显式 stage 所需文件，分阶段 commit，推送个人 fork，再向 `kwesi07/nsrmp-mini:main` 提交 PR。不 force-push，不改上游 main；未完成时用 WIP commit 保存，而非破坏性清理。

## 15. 依据与来源

- [S1] [上游 README（基线快照）](https://github.com/kwesi07/nsrmp-mini/blob/a099629d9d3de5ef0f53566f36938c7fb6709377/README.md)：已有范围、结果、环境与限制。
- [S2] [Issue #1](https://github.com/kwesi07/nsrmp-mini/issues/1)：需求、建议设计、验收与协作规则。
- [S3] [data.py（基线快照）](https://github.com/kwesi07/nsrmp-mini/blob/a099629d9d3de5ef0f53566f36938c7fb6709377/nsrmp/data.py)：DSL、ANY、所有模板与生成器。
- [S4] [batch.py（基线快照）](https://github.com/kwesi07/nsrmp-mini/blob/a099629d9d3de5ef0f53566f36938c7fb6709377/nsrmp/batch.py)、[model.py](https://github.com/kwesi07/nsrmp-mini/blob/a099629d9d3de5ef0f53566f36938c7fb6709377/nsrmp/model.py)、[train.py](https://github.com/kwesi07/nsrmp-mini/blob/a099629d9d3de5ef0f53566f36938c7fb6709377/nsrmp/train.py)、[metrics.py](https://github.com/kwesi07/nsrmp-mini/blob/a099629d9d3de5ef0f53566f36938c7fb6709377/nsrmp/metrics.py)：当前接口与 loss。
- [S5] [PyTorch GRU 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)：双向输出与 batch-first 接口。以本机固定版本 `torch==2.15.0` 的实际行为为准。
- [S6] [PyTorch distributions 官方文档](https://docs.pytorch.org/docs/stable/distributions.html)：Categorical sample/log_prob 与 score-function REINFORCE。
- [S7] [tests/test_experiments.py（基线快照）](https://github.com/kwesi07/nsrmp-mini/blob/a099629d9d3de5ef0f53566f36938c7fb6709377/tests/test_experiments.py)：旧结果 schema 的回归边界。

关键词碰撞分析、leave-one-out 选择和新接口是本方案提出的设计/推导，不归因于原论文。本文没有重新核验论文全文，不沿用对话中互相不一致的章节编号，也不把 Issue 的简化架构当成原论文的原样实现。
