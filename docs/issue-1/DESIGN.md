# 技术设计：受约束角色采样的 Learned Parser

> 这是待实现方案。所有新类、函数、配置和命令都尚未存在；不得将伪代码视为已经运行的实现。基线源码：`a099629d9d3de5ef0f53566f36938c7fb6709377`。

## 1. 目标与现有接点

输入为 instruction、初始场景的 32D object features 和 4D boxes；输出为预测最终场景。训练目标仅使用最终场景 boxes，不使用正确 DSL、正确角色或正确 subject/reference 的标签。

当前 `NSModel.forward` 读取 `batch.programs`，而 `collate()` 通过 `parse()` 和 `parse_roles()` 制作这些字段。新的模型不能简单换一个 loss 后继续走这条数据通路。

复用：`Concepts`、`ActionSimulator`、`apply_move`、`Executor.run_step`、`data.noun_phrase_program`、原有 IoU 实现和数据生成器。默认不改变其数学行为。新建 Parser、无 gold 的 batch 通路、learned-model 训练及评估适配。

## 2. Lexer：关键词识别不等于语义解析

### 2.1 输入信息丢失的反例

以下两句都能被当前规则 Parser 接受，但程序不同：

```text
A: put the cube which is behind the green dice to the right of the red lego
B: put the cube behind the green dice which is to the right of the red lego
```

仅保留关键词时，两者都是：

```text
[cube, behind, green, dice, right, red, lego]
```

A 的 action 是 `MoveRight`，subject 是绿色骰子后面的方块；B 的 action 是 `MoveBehind`，reference 是红色 Lego 右边的绿色骰子。一个只观察上述关键词序列的 Parser 无法确定原句究竟是 A 还是 B。这个结论不依赖训练算法；相同输入不可能可靠区分两个不同标签。

### 2.2 推荐：上下文输入与角色输出分离

固定 lexer 执行大小写/标点规范化、已知词识别、关系短语匹配。输出：

- `context_tokens`：保留原句的可见语法信息，如 `put/the/which/is`，关系短语可折叠为单个 `left/right/front/behind` token。
- `keywords`：颜色、形状、`object`、关系短语，按出现顺序排列，保留每次出现的独立位置。
- `keyword_positions`：每个 keyword 在 context 中的索引，用于从 BiGRU 输出 gather 表示。

`which/is/the` 等 context-only token 不预测角色、不进入 DSL，也不计入 policy log-prob。它们是输入文本的一部分，不是标签。不要通过 `parse()`、`parse_roles()` 或规则名词短语树生成这些输入。

提供 `lexer_mode=surface_context`（推荐主设置）和 `keywords_only`（严格对应 Issue 建议的对照）。保留两种模式的差异说明，不能把后者的信息丢失归因于 REINFORCE 不够强。

`split_steps(text)` 仍固定。未知关键词、空步骤、不支持的词法输入须明确报错，不能静默调用规则 Parser 兜底。颜色和形状的固定词表是允许的先验。

## 3. 模块与数据接口

### 3.1 新训练数据通路：`nsrmp/learned_batch.py`

不改写旧 `Batch` 的字段语义。建议新增以下概念：

```python
ParserInputs:
    feats                 # float[B, N, 32]
    boxes                 # float[B, N, 4]，initial scene
    object_mask           # bool[B, N]
    context_ids           # long[B, S, T]
    context_lengths       # long[B, S]
    keyword_ids           # long[B, S, Kmax]
    keyword_positions     # long[B, S, Kmax]
    keyword_mask          # bool[B, S, Kmax]
    step_mask             # bool[B, S]
    n_steps               # long[B]，只从 split_steps(text) 得到

SceneTargets:
    final_boxes           # float[B, N, 4]

TrainingMetadata:
    is_complex            # bool[B]，只用于已有 curriculum 分组

EvaluationAnnotations:    # 单独构建，不进入训练模型输入
    reference_programs
    subjects
    moved_mask
    texts
```

`B` 是句子数，`S` 是最大步数，`T` 是上下文长度，`Kmax` 是最大关键词数；下文采样数另写 `K`，不要混淆。

`collate_learned(examples)` 只读取 text、feats、initial boxes、final_boxes 和 curriculum metadata。特别注意：当前 `Example.n_steps` 属性返回 `len(self.program)`；新通路必须改从文本分步，不能读取这个属性。`len(batch)` 从张量第 0 维获取，不能沿用旧 `Batch.__len__ = len(programs)`。

生成器内部使用程序构造 target 是合成数据生成的一部分；限制针对生成之后的模型输入与训练信号，不要求重写生成器。数据通路禁止读取 `Example.program/subjects/refs`。`is_complex` 仅能控制 curriculum，不可作为 Parser 的额外输入或 mask 的结构提示。

### 3.2 Parser：`nsrmp/learned_parser.py`

初始候选配置：embedding_dim=32、单层 bidirectional GRU、每方向 hidden_size=64、Linear(128,7)、无 dropout。属于可调的初始方案，不是已经验证的最优参数。

```python
encode(context_ids, lengths) -> context_hidden
keyword_hidden = gather(context_hidden, keyword_positions)
role_logits = linear(keyword_hidden)  # [B, S, Kmax, 7]
sample(role_logits, keywords, grammar_state) -> ParserSample
```

对不同长度 context 使用 packed sequence 或等价的正确长度处理，避免 backward GRU 把右侧 padding 当上下文。零长度 padded step 不送入 GRU。

### 3.3 七种角色

| role | 内容 | 约束 |
|---|---|---|
| subj_head | 被移动物体的颜色/形状 | 必须有一个 shape 或 object，颜色可省略 |
| subj_rel | 描述 subject 的关系 | 可选，最多一个 relation |
| subj_anchor | subject 关系的锚点物体 | 当且仅当 subj_rel 存在时启用 |
| ref_head | 移动目标参照物的颜色/形状 | 必须有一个 shape 或 object，颜色可省略 |
| ref_rel | 描述 reference 的关系 | 可选，最多一个 relation |
| ref_anchor | reference 关系的锚点物体 | 当且仅当 ref_rel 存在时启用 |
| action | 移动方向 | 恰好一个 relation keyword |

每个 keyword occurrence 恰好分配一个角色。相同词出现两次不能去重；`object` 占据 noun 位置、生成 shape=ANY，不能在分配角色前删掉。

## 4. 合法性 Mask 与 DSL Builder

### 4.1 Mask 仅约束语法

颜色/形状只能进入四个 noun slots；关系词只能进入 action/subj_rel/ref_rel。每个 noun slot 最多一个颜色和一个 noun；一个 noun 是 shape 或 object，二者互斥。subject/reference head 必须完整；anchor 与对应 relation 成对存在；action 恰好一个；不允许丢弃关键词。

不能写成“第一个名词一定是 subject”“最后一个关系一定是 action”，不能调用 `parse_roles` 获得正确结构，也不能查看场景答案决定哪些候选合法。语义错误但语法合法的程序必须仍可被采样；否则是把答案藏在 mask 中。

不要要求候选在真实场景中恰好选到一个对象。当前 Executor 是 soft execution；语法合法不意味着 grounding 正确。

### 4.2 逐词条件采样 + 完成可行性检查

为每个句子步骤维护状态：四个 noun slots 的颜色/名词占用、两个 relation slot 的占用、action 是否已经出现。尝试给当前 keyword 分配 role 后，检查剩余 keyword **类型** 是否还能补成完整合法程序。

使用记忆化 DFS / 小型动态规划回答 `can_complete(state, remaining_types)`。这是纯语法布尔函数，与神经参数、gold program、最终场景无关。不要枚举 `7^T` 个无约束分配后惩罚无效项；也不要设置候选数量上限后静默丢掉合法分支。

当前语法每个 head/anchor 至多一个颜色加一个 noun，每步最多 4 个 noun slots 加 3 个 relation/action 词，即最多 11 个关键词；必须用测试验证实际 lexer 长度与这个界限相容，未知扩展输入应显式报错。

```python
state = empty_state()
log_prob = differentiable_zero()
local_entropies = []
for t in valid_keyword_positions:
    valid_roles = [r for r in ROLES
                   if local_type_and_capacity_ok(state, keyword[t], r)
                   and can_complete(advance(state, keyword[t], r), remaining_types(t))]
    if not valid_roles:
        raise InvalidInstructionError(...)  # 不回退到规则 Parser
    dist = Categorical(logits=role_logits[t].masked_fill(not_valid, -inf))
    role = dist.sample()
    log_prob = log_prob + dist.log_prob(role)
    local_entropies.append(dist.entropy())
    state = advance(state, keyword[t], role)
    trace.append(role)
program = role_to_program(keywords, trace)
```

从可完成状态开始，每次只选择仍可完成的下一状态，才能保证最后得到合法程序。测试需同时证明 soundness（不放过非法项）和 coverage（合法候选不被错误屏蔽）。

由于 mask 依赖已采样角色，联合策略是条件概率乘积；必须累加**实际 masked 分布**的 log-prob，不是未 masked softmax 的概率。禁止 `.item()` / `detach()` 破坏这些 log-prob 的计算图。

重复词可能使不同角色轨迹构造出同一个 AST。因此接口中的 `log_prob` 严格指采样的 **role trace 的 log probability**；不声称它等于对所有轨迹求和后的 program marginal。对 `reward(builder(trace))` 使用轨迹 log-prob 的 REINFORCE 仍是正确目标。

### 4.3 Builder 是确定性转换

```python
def build_np(head, relation, anchor):
    source = ('Scene',)
    if relation is not None:
        anchor_prog = noun_phrase_program(anchor.colour_or_ANY,
                                          anchor.shape_or_ANY)
        source = ('Relate', anchor_prog, relation)
    return noun_phrase_program(head.colour_or_ANY, head.shape_or_ANY, source)

move = ('Move', ACTIONS[RELATIONS.index(action_word)],
        build_np(subj_head, subj_rel, subj_anchor),
        build_np(ref_head, ref_rel, ref_anchor))
```

保留 `noun_phrase_program` 的 Filter 顺序和 Unique 包装，便于与规则 Parser 的 tuple AST 精确比较。不要用英语字符串当作已完成 DSL。

### 4.4 推理

默认主结果用 masked greedy decoding；每步依然应用完成可行性检查。它是合法的逐决策贪心，不声称是全局最大概率程序。另报告固定 evaluation RNG 下的 sampled accuracy；禁止根据 gold final loss 选择 best-of-K 作为正式预测。

## 5. LearnedNSModel 与多步执行

在 `nsrmp/model.py` 新增独立 `LearnedNSModel(nn.Module)`；持有 Parser、Concepts、AS、Executor，但不调用依赖 `batch.programs` 和 `batch.subjects` 的旧 `NSModel.forward`。

接口建议：

```python
model.forward(inputs) -> (pred_boxes[B,N,4], subject_weights[B,S,N])
# 默认 greedy；保持普通推理的二元返回形态。

model.sample_forward(inputs, num_samples=K) -> LearnedOutput:
    pred_boxes       # [K, B, N, 4]
    subject_weights  # [K, B, S, N]
    programs         # K × B，每项 tuple[Move, ...]
    role_traces      # 同一采样对应的角色轨迹
    log_prob         # [K, B]；对有效词、有效 step 求和
    local_entropy    # [K, B]；对有效决策取均值
```

每个 sample 是一句话的完整程序序列，不是单独随机选择某个步骤。执行第 s 步时使用第 s−1 步**预测的** boxes；损失只计算最终状态。无 teacher forcing、无 gold intermediate state。

padded step 的 Move 为 None，subject weights 为 0，boxes 不变；不计入 log-prob、entropy 或 program-accuracy 分母。基于文本的 `step_mask` 决定执行步数。

先让 `[K,B]` 展平后调用已有分组 Executor；不修改 Executor 的核心算法。若纯 Python 构造程序成为性能瓶颈，先缓存纯词法结果/语法可行性，不提前引入 GPU/MPS 或重写 VR。

## 6. Loss、Baseline 与梯度

### 6.1 每样本场景损失

新增 `per_example_loss(pred, gold, mask, beta=1)`：

```text
L[k,b] = Σ_i mask[b,i] * ( ||pred[k,b,i]-gold[b,i]||_1
                         + beta * (1-IoU(pred[k,b,i],gold[b,i])) )
```

旧 `loss_fn` 仍返回这个量的 batch mean，默认数值语义不变；不要改成按物体数平均而改变已有实验。新训练的 `task_loss=L.mean()` 对 `[K,B]` 平均。

### 6.2 主方案：同句独立采样 + leave-one-out baseline

```python
R = -L.detach()                                  # [K, B]
if K > 1:
    baseline = (R.sum(dim=0, keepdim=True) - R) / (K - 1)
else:
    baseline = previous_ema_value                # 当前 batch 采样前的值，初值 0
advantage = (R - baseline).detach()
parser_loss = -(advantage * output.log_prob).mean()
entropy_bonus = output.local_entropy.mean()
total = task_loss + parser_loss - 0.01 * entropy_bonus
optimizer.zero_grad()
total.backward()
clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
# K=1 的 EMA 在本次更新之后再更新，不能提前包含本样本。
```

K 个采样要使用独立随机抽样；不在每个 sample 前重置同一 seed。重复采到同一程序是允许的，不静默去重/重加权。

Issue 的 self-inclusive batch mean 可以运行，但不满足“baseline 与当前 sampled action 独立”的简单无偏条件。在独立样本的简化条件下，会将期望策略梯度缩小为 `(M−1)/M`；M=1 时为零。K>1 时用 leave-one-out 避免这一点。此处是对 Issue 建议的明确修正，不是更换 REINFORCE。

`local_entropy` 指在实际访问的各 masked 分布上的局部熵正则；因为后续 mask 依赖采样历史，不能未经推导声称这是完整联合程序熵梯度的精确无偏估计。使用它作为明确记录的探索正则即可。

### 6.3 参数更新边界

- `task_loss`：通过已采样的固定程序更新 Concepts / AS；不能通过离散采样直接更新 Parser。
- `parser_loss`：通过仍保留计算图的 log-prob 更新 Parser；reward detach 防止它通过场景损失反向影响 Concepts / AS。
- entropy：更新 Parser；不改变 VR/AS。
- 一个 optimizer 可使用两组参数；初始 lr 均为 3e-3，后续只在验证集上比较 Parser lr 变化。
- 不添加程序交叉熵、角色标签 loss 或 gold-subject loss；诊断型 mock reward 只能存在于单元测试，不能进入正式数据训练。

## 7. 训练策略与诊断

基础配置与上游数据规模对齐：4000 train / 1000 test、60 epochs、batch 64、seeds 0/1/2、Adam + cosine、前 20 epochs simple-only。新模型 K=4，greedy 主评估。它执行次数更多，必须记录 compute/runtime，不能把相同 epoch 当成相同计算预算。

先跑简单小样本，分别验证 Parser 梯度和执行模块梯度，再进入完整联合训练。可选的后续尝试包括 Parser 较高学习率、不同 entropy 系数、simple 阶段之后冻结 AS；冻结只冻结本次无程序监督训练得到的参数，不能用 rule-parser NS 的权重预热主结果。

每 epoch 记录 task loss、reward、advantage 方差、parser gradient norm、局部 entropy、有效程序比例、validation IoU-M 和 program accuracy 曲线。Program accuracy 是审计指标，不参与 reward、反向传播、自动调参或 checkpoint 选择；主 checkpoint 默认最后一个 epoch。

不同模块共同学习时，低 reward 可能来自执行器尚未学会，而不是程序错误。语义错误程序也可能被可学习执行器补偿，因此 task success 不等于 program accuracy。

## 8. 兼容性与实验入口

当前 `tests/test_experiments.py` 对默认 phase-2 的模型集合做精确相等检查。不能无条件在 `_models()` 中添加新行；增加 opt-in 配置和 CLI（见实施计划），旧调用默认行为不变。

旧 `metrics.evaluate` 保持四个原有 key。新增 `evaluate_learned` 或独立适配器；program 指标单独存储，不给所有旧行硬塞新 key。纯神经 baseline 的 program accuracy 标为 N/A，不是 0；规则 Parser 的 100% 是规则构造对照而非 learned competence。

新结果写入独立目录；原有 `results/phase1.json`、`phase2.json`、noise 输出和图保持原样。跨机器重新训练的数值不保证逐 bit 一致，任何重跑结果需独立标记环境，不能偷偷覆盖历史记录。

## 9. 研究结论边界

即使 learned-parser NS 优于 raw-sentence GRU，也只说明这个受约束、固定词法/分步、合成场景设置下的新系统表现更好；不能单独归因于 symbolic execution，也不能宣称论文结论已被复现。Keyword-only 与 surface-context 必须分别标注；若使用不同计算预算或调参预算，也要说明。

实现成功的标准首先是机制和证据正确；经过公平尝试仍训练失败，也应提交完整曲线和诊断，而不是更换为使用 gold parser 的训练后继续称为 learned parser。
