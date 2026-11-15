# 实施与验收清单

> 本清单是后续开发任务，不是已完成实现。当前只有“规划基线”部分完成。总体设计与所有接口定义见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)。

## 0. 规划基线：本次已完成

- [x] 核对上游 Issue #1、代码与测试接口。
- [x] Fork 至 `Pearlue/nsrmp-mini`，克隆到 `~/projects/nsrmp-mini`。
- [x] 创建独立规划分支 `docs/issue-1-learned-parser-plan`。
- [x] 用 `uv` 建立 Python 3.12 `.venv`，安装上游固定依赖。
- [x] 原有 59 tests 通过；见 [BASELINE_CHECK.md](BASELINE_CHECK.md)。
- [x] 写出开发方案、模块接口、测试矩阵与交付清单。
- [x] 只读验证关键词碰撞反例；不新增算法实现。

## 1. Test-first 推进顺序

每个阶段先新增会失败的目标测试，再实现最小功能，然后跑新测试和原回归测试。不要为了勾选清单跳过失败测试或降低旧断言。

### P0：训练数据边界

- [ ] 建立独立 `LearnedBatch`、`collate_learned`，不改旧 `Batch/collate` 默认路径。
- [ ] `n_steps` 从 `split_steps(text)` 计算，不访问 `Example.n_steps`。
- [ ] 用只含文本、数值特征、boxes、最终场景、curriculum 难度的 minimal fixture 进行训练。
- [ ] 把 fixture 的 `program/subjects/refs` 设为一访问就抛异常；确认 collate 和训练仍能运行。
- [ ] 将 `parse/parse_roles` 及其已导入别名 monkeypatch 为抛异常，完整执行新 collate 和训练路径。
- [ ] 验证 evaluation targets 只在训练之外构建，不被 policy 或 mask 访问。

### P1：词法、Builder 与约束

- [ ] 实现最长关系短语匹配，保留 token 次序、重复 occurrence 与 `object`。
- [ ] 手工角色能还原 simple、subject-relative、reference-relative、both-relative、wildcard 和 multi-step 所有模板。
- [ ] Builder 使用现有 `noun_phrase_program`，保持 ANY、Filter 顺序和 anchor Unique。
- [ ] 类型 mask 禁止 colour→action、relation→noun slot 等错误。
- [ ] 容量 mask 禁止两个 action、重复 noun shape/colour。
- [ ] 可完成性 mask 禁止遗留无 action、缺 head、缺 anchor 的死路。
- [ ] 不使用正确角色、gold 程序、最终场景或 gold 对象唯一性作 mask 条件。
- [ ] simple 输入仍可生成 subject/reference 交换等有效但错误的程序，证明 mask 不是规则答案。
- [ ] 所有支持模板的采样有效率 100%；不能静默跳过无候选样本。

### P2：BiGRU 与 policy trace

- [ ] 用真实长度处理 padding 与零长度 step，输出 `[B,S,L,7]` logits 或等价展平形式。
- [ ] 确认 logits、sampled trace log-prob、entropy 都有正确的梯度关系。
- [ ] 手算小例子的 masked 概率与 `sum(log p(selected_role))`，和实现一致。
- [ ] K 次样本为独立 draws，允许重复；不做未建模的去重或 best-of-K 筛选。
- [ ] greedy 在固定输入/参数下确定；random sample 在固定 seed 下可复现。
- [ ] 增加 padding 不改变有效句子的 policy 输出。
- [ ] `trace_log_prob` 对真实 tokens 和 steps 求和；entropy 的长度归一化按方案固定。

### P3：执行器与 REINFORCE

- [ ] 新增 LearnedNSModel，保留旧 NSModel 的返回值与行为。
- [ ] `forward_with_policy` 返回 K 次预测和对应 trace；每个 loss 必须对应同一次采样。
- [ ] `loss_fn(reduction='none').mean()` 与原默认 loss 数值和梯度一致。
- [ ] K=4 leave-one-out baseline 实现正确；K=1 使用更新前 EMA，不出现恒零 advantage。
- [ ] task-only backward：Concepts/AS 有梯度，Parser 不经离散选择收到 task 梯度。
- [ ] policy-only backward：Parser 有梯度，Concepts/AS 不经 reward 收到梯度。
- [ ] entropy-only backward：只影响 Parser。
- [ ] 单次更新只有一个 backward 和一个 optimizer.step；K 次采样期间不改参数。
- [ ] 多步执行只使用上一部预测场景；补齐步为 no-op，不读取真实中间场景。
- [ ] 实现固定随机种子的 tiny reward improvement 测试，并记录前后 expected reward。

### P4：实验与诊断

- [ ] 新实验显式启用，不改变默认旧模型集合和旧指标 schema。
- [ ] phase 1、phase 2 的 tiny integration test 都通过。
- [ ] 实现 deterministic program accuracy、逐步 accuracy、固定 RNG 的 sampled accuracy。
- [ ] 实测同一 mask 下的随机策略 chance；不能用 1/7 代替。
- [ ] 固定独立验证集，只用场景指标选择有限超参数尝试；final test 只报告。
- [ ] 完成 seeds 0/1/2，保存全部种子，不择优丢弃失败运行。
- [ ] 输出 IoU-M overall/simple/complex、program accuracy、曲线与实际 runtime。
- [ ] 测量关键词碰撞，仅作为独立分析；若加 marker/context 变体则单独命名和报告。
- [ ] 新结果目录不覆盖 `results/phase1.json`、`phase2.json`、已有 markdown 或图。

### P5：报告与 PR

- [ ] README 增加 `Our results`，标明固定 lexer/splitter 和简化环境。
- [ ] 如不收敛，写明尝试、chance 比较、准确率曲线、有效率与运行时间。
- [ ] 原 59 tests 及所有新增测试通过；不得删除、skip 或改松原测试。
- [ ] 检查 `.env`、keys、权重、`.venv` 等没有进入暂存区。
- [ ] 确认只向个人 fork 的 feature branch 推送，无 force-push/main push。
- [ ] 建立上游 PR，说明整体改动、验证命令、结果限制与 Issue #1 对应项。

## 2. 核心测试矩阵

| 测试主题 | 放置位置 | 如何验收 |
| --- | --- | --- |
| 全模板 round trip | `test_learned_parser.py` | 人工正确 roles → Builder → 与 oracle AST 相等；包括 both-relative |
| 合法性 mask | `test_learned_parser.py` | 逐条非法类型/容量/依赖测试；剩余可完成性边界；无全 -inf |
| 语法不等于答案 | `test_learned_parser.py` | 错误但可执行的 subject/reference 交换仍有正概率 |
| 关键词碰撞 | `test_learned_parser.py` | 两文本具有相同关键词、不同 gold AST；不能误写成必须恢复两者的成功测试 |
| log-prob 与 entropy | `test_learned_parser.py` | 与小规模手算分布一致；mask 后再归一化，无 padding 项 |
| reward 提升 | `test_learned_parser.py` | 固定小环境/冻结可控 executor，以真实目标场景计算 reward；tiny policy run 的期望 reward 提高 |
| joint smoke | `test_learned_training.py` | 从随机模型初始化跑少量 minibatches，loss/grad 有限；不把稳定达到目标 IoU 当 unit-test 前提 |
| 无 gold 泄漏 | 两个核心测试文件 | 新 collate、train 全链路中禁用 oracle，gold 属性访问抛异常；不是仅 grep 源码 |
| 梯度隔离 | `test_learned_training.py` | 三个 loss 分别 backward 检查对应参数组；detach 生效 |
| loss 回归 | `test_learned_training.py` | 新 reduction 的均值/梯度与原定义相等 |
| LOO / K=1 | `test_learned_training.py` | 固定 reward 张量手算；K=1 不除零，EMA 更新时序正确 |
| 变长/多步骤 | `test_learned_training.py` | 混合步数、padding no-op、序列 log-prob、预测场景逐步传递 |
| 评估确定性 | `test_learned_experiments.py` | greedy 重复一致；采样评估不改变训练 RNG 状态 |
| 新输出 schema | `test_learned_experiments.py` | 包含 learned row、3 个 seeds、程序指标、runtime；旧 rows 键不改 |
| 原项目回归 | 现有 `tests/` | 原 59 tests 全部继续通过，旧结果文件 diff 为空 |

Reward improvement 测试应先隔离估计器正确性：可用二选一 toy policy 和固定执行器，但 reward 必须来自执行后的最终场景误差，而非拿 `program == gold_program` 作 reward。此 fixture 的成功不等于真实 joint learning 收敛。真实训练的结果由 P4 报告，不通过不断更换 seed 让一个 flaky unit test 偶然通过。

## 3. 命令：现有可用与未来新增分开

### 3.1 当前已经存在的命令

```bash
cd ~/projects/nsrmp-mini
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest -q

# 旧模型的完整实验；当前任务没有执行这一条。
# 使用独立目录，避免覆盖仓库已有结果。
.venv/bin/python run_experiments.py --phase all --out results/upstream-validation/run-01
```

已有 `.venv` 时直接复用，不反复重建。当前上游依赖已固定，首版不额外引入 transformers、RL 框架或大型下载。

### 3.2 拟新增参数：目前不能直接运行

`--include-learned` 默认为 false，开启后在原实验集合中加入新模型。拟新增 `--samples-per-sentence`、`--parser-lr`、`--entropy-coef`、`--parser-input`；其 config 默认值必须与方案一致。原 `--phase`、`--out`、`--epochs`、`--n-train`、`--seeds` 保留原语义。

```bash
# 以下仅是实现后的验收命令草案；当前分支尚不支持这些新增参数。
.venv/bin/pytest -q tests/test_learned_parser.py tests/test_learned_training.py
.venv/bin/pytest -q

.venv/bin/python run_experiments.py \
  --phase all --include-learned \
  --seeds 3 --epochs 60 --n-train 4000 \
  --samples-per-sentence 4 --parser-lr 0.003 --entropy-coef 0.01 \
  --parser-input keywords_only \
  --out results/learned_parser/run-01
```

首轮功能开发使用 tiny Config 的新 pytest integration tests，不要每改一处就重跑完整实验。正式配置锁定后，再运行三种子数据。

## 4. 后续建议提交顺序

1. `test: define learned-parser input and DSL contracts`：先落测试与输入边界。
2. `feat: add constrained BiGRU role parser`：Builder、mask、trace、网络。
3. `feat: integrate learned-program execution and REINFORCE`：模型、逐例 loss、baseline、梯度测试。
4. `feat: add learned-parser evaluation and experiment outputs`：指标、schema、曲线、runtime。
5. `docs: report learned-parser results and limitations`：真实结果、所有尝试、README。

开发中的未完成状态用明确标注的 WIP commit 保存。每次 stage 显式文件列表，查看 `git diff --cached`；不要以 `git stash` 或破坏性 reset 隐藏别人的改动。

## 5. 最终验收与科学结果的区别

工程完成：实现、有效性、无泄漏、兼容性、可重跑实验、文档和 PR 都已交付。

学习成功：场景 reward 与程序准确率经过实际实验改善，且不是 mask 硬编码答案或监督泄漏造成。

工程可以完成而学习效果仍不佳；后者必须如实报告，但不能用“允许负结果”掩盖接口未接完、梯度错误或实验没跑。
