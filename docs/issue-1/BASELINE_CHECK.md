# 规划阶段基线检查

记录日期：2026-11-14（America/Los_Angeles）。这是本次实际执行的检查，不是拟议实现的测试结果。

## 1. 仓库与权限

- 上游：`kwesi07/nsrmp-mini`，默认分支 `main`。
- 上游权限：可读，没有直接 push 权限。
- 个人 fork：`Pearlue/nsrmp-mini`；GitHub API 已确认其 parent 为 `kwesi07/nsrmp-mini`。
- 本地：`~/projects/nsrmp-mini`，独立 `.git`，不是 submodule。
- `origin` 指向个人 fork，`upstream` 指向原仓库。
- 规划分支：`docs/issue-1-learned-parser-plan`。
- 代码基线：`a099629d9d3de5ef0f53566f36938c7fb6709377`。
- Issue #1 状态核验：OPEN；建议设计与验收要求已读取。

没有修改 Git 用户身份；沿用本机已配置的提交身份。没有读取或提交任何 token、`.env` 或密钥。

## 2. 本机环境

| 项目 | 实际检查值 |
| --- | --- |
| 架构 | arm64 |
| CPU | Apple M4 Pro |
| 内存 | 51,539,607,552 bytes = 48 GiB |
| Python | CPython 3.12.14 |
| 虚拟环境 | 仓库内 `.venv`，由 `uv venv --python 3.12 .venv` 创建 |
| torch | 2.15.0 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| pytest | 9.2.0 |

依赖来自上游 `requirements.txt`，未调整版本。`.venv` 已被现有 `.gitignore` 排除。

## 3. 实际运行命令与结果

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest -q
```

输出摘要：

```text
...........................................................              [100%]
59 passed in 19.12s
```

这验证了原项目在本机环境中的基线测试可运行。该时间是测试套件用时，不是完整模型训练用时，也不是未来 REINFORCE 的 runtime。

没有运行 `run_experiments.py --phase all`；没有生成新的 IoU-M、program accuracy 或训练曲线。README 中原实验数字仅作为历史结果保留。

## 4. 已验证的关键词碰撞

使用未修改的 `nsrmp.parser.parse`，对两个语法合法输入进行只读诊断：

```text
A = put the red cube which is behind the green dice to the right of the blue cube
B = put the red cube behind the green dice which is to the right of the blue cube
```

仅保留颜色、形状、关系词与 `object` 后，实际输出：

```text
same_keywords: True
keywords: ['red', 'cube', 'behind', 'green', 'dice', 'right', 'blue', 'cube']
different_programs: True
actions: ['MoveRight', 'MoveBehind']
```

因此严格 keyword-only 输入可能丢失决定结构的连接词信息。本诊断没有测量这种碰撞在训练/测试生成分布中的发生率，也没有训练模型。后续实现应把它作为需要测量的表示限制，而不是预先断言模型一定无法学习任何有用结构。

## 5. 回归约束的实际核对

`tests/test_experiments.py` 对旧 phase 2 的模型集合要求恰好为 `ours/baseline/parsed_baseline`，对旧指标集合要求恰好为 `iou/iou_m/subject_acc/placement_acc`。因此方案采用显式开启 learned 模型，而不改变默认行为或强改旧测试。

`batch.py`、`data.py`、`model.py` 中现有 `programs`、`parse_roles`、`Example.n_steps` 和 `subjects.shape` 依赖已检查；方案使用独立 LearnedBatch 避免把它们带入新训练路径。

## 6. 本次文件范围

只添加根 `AGENTS.md`、`docs/issue-1/` 规划材料，并在根 README 增加规划入口。不修改 `.py` 实现、原测试、requirements、SPEC 或已有 results。

`~/projects/AGENTS.md` 按工作区规则增加新仓库索引；它位于项目集合根，那里没有 Git，不属于本仓库提交，也不会上传个人工作区索引。

本次没有实现 Learned Parser 或 REINFORCE，没有完成 Issue #1，也没有向上游发布 PR 或评论。远端规划分支的提交与推送结果以最终 Git 检查为准。
