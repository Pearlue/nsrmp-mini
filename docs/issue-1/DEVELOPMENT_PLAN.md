# 开发计划与交付顺序

> 当前仅编写方案；下列实现项全部待执行。技术细节以 [DESIGN.md](DESIGN.md) 为准，验收以 [VALIDATION.md](VALIDATION.md) 为准。

## 1. 工作范围

由一个贡献者完整负责 Issue #1：lexer/Parser、合法采样与 DSL builder、LearnedNSModel、REINFORCE、测试、实验和文档。无需等待另一位贡献者先完成 Parser；内部仍保留清晰接口，以便单独测试。

本机开发使用 Python 3.12 venv、PyTorch 和 pytest，CPU 为正式首选。现有 Executor 多处直接构造 CPU tensor，因此不把 MPS 迁移与此 Issue 绑定。48GB 内存不是当前小模型的主要约束；优先观察 CPU 时间、合法性搜索与程序执行开销。

以下为实施阶段而非异步执行承诺；每阶段通过门槛后再进入下一阶段。

## 2. 文件级改动清单

| 文件 | 操作 | 内容 | 保留约束 |
|---|---|---|---|
| `nsrmp/learned_parser.py` | 新建 | 词法扫描、两种 context 模式、BiGRU、角色 head、完成可行性 mask、采样/greedy、DSL builder | 不调用规则语义解析；不依赖场景答案 |
| `nsrmp/learned_batch.py` | 新建 | ParserInputs、targets/annotations 分离、collate/select、text-derived step count | 不沿用依赖 programs 的旧 len 和 n_steps |
| `nsrmp/model.py` | 扩展 | 独立 LearnedNSModel、sample_forward、多步执行 | 不改变 Concepts/AS/旧 NSModel 的数学语义 |
| `nsrmp/train.py` | 扩展 | per_example_loss、train_learned、K samples、RLOO/EMA、entropy、分组 lr、日志 | 旧 train 默认行为及 loss_fn 返回值不变 |
| `nsrmp/metrics.py` | 扩展 | learned-model evaluation adapter、AST exact-match、greedy/sample 分开 | 旧 evaluate 仍返回原有四个指标 |
| `run_experiments.py` | 扩展 | 显式 opt-in 新模型、独立输出、3 seeds、曲线和 runtime | 默认模型集合、noise 实验和老结果不变 |
| `tests/test_learned_parser.py` | 新建，最先写 | lexer/builder/mask/梯度/无泄漏/小训练 | 不删改原有 59 项测试的判定 |
| `tests/test_learned_experiments.py` | 新建 | CLI/config、3 seed、JSON/table、兼容与随机数隔离 | 小数据、临时输出、不污染 results |
| `.gitignore` | 按需扩展 | 防止未来嵌套结果目录的 checkpoints、`.env` 被提交 | 当前 `results/*.pt` 不覆盖所有嵌套路径 |
| `README.md` | 扩展 | 真实的 Our results、新设置、失败尝试、runtime | 原有表格不覆盖、不写 reproduced |
| `results/learned_parser/<run-id>/` | 实验后新建 | JSON、表、曲线、配置、环境、计时、日志 | 目前不预造结果、不填预测分数 |
| `docs/issue-1/` | 持续更新 | 设计决定、实施状态、验收记录 | 明确计划和已完成事实 |

原则上不改 `nsrmp/parser.py`、`nsrmp/data.py`、`nsrmp/executor.py`、`nsrmp/baseline.py`。发现必须修改时，单列原因并添加回归测试，不借此重写整个仓库。

## 3. P0：锁定基线与输入合同

### 工作

1. 记录上游 commit、环境、旧测试结果；保留当前 results 原样。
2. 编写两个不同句子得到相同 keywords 的回归 fixture，验证各自的 AST 确实不同。
3. 确认 `surface_context` 主设置与 `keywords_only` 对照，并在将来的 PR 中突出这一偏离 Issue 的理由。
4. 写出张量 shapes、ParserSample/ModelOutput 字段，以及无 gold 数据访问边界。
5. 核对默认实验集合不能变化：phase-2 现有测试要求精确的三个模型 key。

### 门槛

- 原有 59 个测试通过。
- 输入信息丢失问题写进测试和文档，不用“训练再久一点”掩盖。
- 新数据通路不会调用 `Example.n_steps` 或 legacy `collate()`。

建议提交：`test: specify learned parser inputs and leakage boundaries`。

## 4. P1：先完成确定性工具与约束测试

### 工作

1. 为 lexer、role-to-program、mask 编写失败测试，再实现纯函数。
2. 覆盖 simple、relative subject、relative reference、both-relative、ANY、重复关键词、四种 action、多步骤。
3. 以类型/slot 状态与记忆化完成检查实现语法 mask。
4. 对很短序列穷举合法分配，验证 mask 没有放过非法项、也没有误删合法项。
5. 验证同一句话仍允许语义错误但语法合法的 subject/reference 互换；防止约束编码答案。

### 门槛

- 给定正确 roles 时，builder 对所有模板返回与规则 Parser 完全一致的 AST。
- 经过 mask 的完整采样在支持的输入域内 100% 语法合法。
- 不靠拒绝采样或“发现不合法就调用规则 Parser”维持成功率。

建议提交：`feat: add lexical inputs and grammar-constrained DSL builder`。

## 5. P2：实现概率 Parser 与无 gold batch

### 工作

1. 实现 embedding + BiGRU + 7-role classifier；packed lengths 正确处理 padding。
2. 根据 context 的 keyword_positions 提取每个 keyword 的上下文表示。
3. 训练用条件采样，推理用合法性约束下 greedy；返回 role_trace、program、log_prob、local_entropy。
4. 实现独立 ParserInputs 与 collate/select，训练 targets 和评估 annotations 分离。
5. 完成 K/B/S 维度测试和随机数可重复性测试。

### 门槛

- log_prob 与手算的 masked 条件概率之和相符，保留 autograd graph。
- 没有 padding/缺失 step 的梯度贡献。
- 给相同句子重复采样时不因重置 seed 而永远返回同一结果。
- 改动 gold program/subjects/refs 或令其读取时报错，仍可构建训练 batch。

建议提交：`feat: add BiGRU role policy and gold-free training batches`。

## 6. P3：接入模型与 REINFORCE

### 工作

1. 新建 LearnedNSModel，复用 Concepts、AS、Executor；不复用旧 NSModel 的 forward 实现。
2. 对 K 份完整程序执行现有多步场景更新，输出 `[K,B,N,4]` 最终 boxes。
3. 提供逐样本 loss，保留旧 loss_fn 的平均行为。
4. 添加 train_learned：reward detach、RLOO baseline、K=1 EMA fallback、entropy、梯度裁剪、optimizer 与 cosine。
5. 用固定 toy executor 验证 Parser loss 的方向，再用真实模型验证两条梯度路径。
6. 确保 eval/logging 不改变训练随机数状态；使用独立 RNG context，禁止 evaluation 中调用全局 seed 后不恢复。

### 门槛

- parser-only loss 更新 Parser，不更新 Concepts/AS；task-only loss 更新执行模块，不通过离散选择更新 Parser。
- tiny controlled run 的**期望 reward**提高；这不是仅看 parser_loss 变小。
- K=1、B=1 不因 self-mean baseline 导致策略梯度永久为零。
- 现有 59 项测试保持通过；无 gold-program/role/subject 泄漏。

建议提交：`feat: train sampled programs with REINFORCE and final-scene loss`。

## 7. P4：先诊断，再运行三种子实验

### 第一轮 smoke

- 256 个 simple train examples，小型独立 validation set，单 seed，少量 epochs。
- 查看 reward、entropy、合法性比例、各参数组梯度、program accuracy；不要求达到论文或规则模型的数值。
- 用日志区分 Parser 不学、执行器不学、两者共同塌陷；不能只用“loss 在下降”判定成功。

### 完整单步实验

- 固定 4000 train / 1000 test / 60 epochs / seeds 0,1,2。
- 使用与原表相同的数据生成设置；另建 validation，不用 test 挑超参数。
- 新增 learned-parser row，并列原有 rule NS、parse-matched neural、raw-sentence GRU。
- 先报告 overall/simple/complex IoU-M 与 greedy exact program accuracy，再附 sampled accuracy、chance baseline 和 runtime。

### 多步兼容与泛化

- 沿用 phase-2：训练 1–2 steps，测试 1–5 steps；step splitter 仍给定。
- 验证中间状态来自模型预测；报告整条指令 exact match 和按 step accuracy。
- 对象数 3–10 的评估沿用现有实验组织，避免改变旧图的原始数值。
- Feature-noise 扫描不是完成本 Issue 的强制新增实验，不额外扩大范围。

建议提交：`feat: add opt-in learned parser experiments and accuracy reporting`。

## 8. P5：有界调试与失败报告

不无限扩大搜索或偷偷加入程序标签。先排除实现错误，再在固定验证集上逐项比较：

| 尝试 | 与基础设置的变化 | 用途 |
|---|---|---|
| A0 | 基础：K=4、entropy=0.01、同 lr、curriculum | 主起点 |
| A1 | Parser lr 提高，其他不变 | 区分策略学习过慢 |
| A2 | entropy 调整，其他不变 | 识别过早塌陷/持续过度探索 |
| A3 | simple 阶段后冻结本次训练的 AS | 观察 reward 非平稳性 |
| A4 | K=1 + EMA，对照 K=4 + RLOO | 观察采样方差与成本 |
| A5 | keywords_only 对照 surface_context | 量化输入信息丢失，不混入主结果 |

初始先在单个开发 seed 上诊断，选定配置后完整跑 3 seeds。选择依据为 validation scene loss/IoU-M，program labels 不进入自动配置搜索或 checkpoint 选择。所有尝试保留配置和结果，不能只保留成功 seed。

如果仍接近随机：记录每种尝试、合法采样率、reward/entropy/program-accuracy 曲线、与同约束随机策略的比较，以及可能的输入歧义或参数补偿。正确的失败实验可以交付；有 bug 的训练结果不能用“负结果”豁免修复。

## 9. P6：复核、报告、PR

1. 从干净环境复跑完整 tests，保存实际测试数与时间。
2. 检查 diff：原有源码行为和结果未被意外改变，未提交 secrets/venv/checkpoints。
3. README 新增 `Our results`；新表中的未运行项写 N/A 或 not run，不填估计值。
4. 核对 JSON → table → plot 一致；3 seeds 使用 sample std (`ddof=1`)。
5. 记录数据生成、训练、评估与总 wall time；主机/版本/threads 一并记录。
6. push contributor fork 的 feature branch；完成实施后才向上游开 PR，不直接推 `main`。
7. PR 用 `Refs kwesi07/nsrmp-mini#1` 描述关系；只有完整满足验收时才提出关闭 Issue。

建议提交：`docs: report learned parser results and limitations`。

## 10. 当前可执行与拟新增命令

### 已存在，可直接执行

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest -q
# 下面是旧实验；会训练旧模型，不会实现或测试 learned parser。
.venv/bin/python run_experiments.py --phase all --out results/baseline-rerun
```

### 设计中的接口，实施后才可执行

```bash
# 新 CLI 参数尚未实现。
.venv/bin/python run_experiments.py \
  --phase 1 --include-learned-parser --parser-samples 4 \
  --lexer-mode surface_context --seeds 3 \
  --out results/learned_parser/main-run

.venv/bin/python run_experiments.py \
  --phase 2 --include-learned-parser --parser-samples 4 \
  --lexer-mode surface_context --seeds 3 \
  --out results/learned_parser/multistep-run
```

`Config` 拟增加 `include_learned_parser=False`、`parser_samples=4`、`lexer_mode='surface_context'` 及独立 Parser 配置。明确提供 learned-only 调试选择，避免每次调 Parser 都重新跑全部旧模型；该选择不得改变默认行为。

## 11. 建议实验产物

```text
results/learned_parser/<run-id>/
  config.json                # 包括 baseline commit、train/val/test seeds、所有超参数
  environment.json           # Python/依赖版本、CPU、threads、git commit
  phase1.json / phase2.json   # 逐 seed 的指标；仅包含实际运行的 phase
  results.md                 # 新旧行的来源标注，N/A 与 sample std
  learning_curves.json       # 每 epoch 的 reward、entropy、validation 指标
  program_accuracy.png       # greedy / sampled / chance 明确区分
  runtime.json               # data/train/eval/total 秒数
  run.log                    # 命令、失败信息、诊断
```

默认不保存或提交权重。需要本地 checkpoint 时先设置嵌套路径 ignore，并在 metadata 中记录重建方式。日志不包含账号 token 或机器私密目录内容。
