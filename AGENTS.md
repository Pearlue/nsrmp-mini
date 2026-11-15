# nsrmp-mini 工作约定

## 导航与当前状态

- 原项目说明：[README.md](README.md)；历史设计：[SPEC.md](SPEC.md)。
- Issue #1 规划入口：[docs/issue-1/README.md](docs/issue-1/README.md)。
- 实现方案：[DEVELOPMENT_PLAN.md](docs/issue-1/DEVELOPMENT_PLAN.md)。
- 测试与阶段验收：[IMPLEMENTATION_CHECKLIST.md](docs/issue-1/IMPLEMENTATION_CHECKLIST.md)。
- 本机基线证据：[BASELINE_CHECK.md](docs/issue-1/BASELINE_CHECK.md)。
- 本次交付仅为开发规划；没有实现 Learned Parser、REINFORCE 或新的模型实验。后续实现须按用户的新任务开展，不能把未完成项写成已完成。

## 工作范围

Issue #1 的目标是在给定 lexer 和 step splitter 的简化环境中，从最终场景损失学习程序。复用 Concepts、Executor 和 ActionSimulator；不扩展到真实图像检测、机器人控制或完整论文复现。

## 工程规则

1. Python 3.12，使用仓库内 `.venv` 与 `uv`，保持 `requirements.txt` 的现有固定版本；pytest test-first。
2. 在独立分支工作，推送到自己的 fork；不要向上游或 fork 的 `main` 直接推送。后续实现完成再向上游发 PR。本次规划任务不自动发 PR 或 issue 评论。
3. 不使用 `git stash`、`git reset --hard`、`git checkout --` 覆盖别人的工作；保留无关改动，必要时采用明确列出文件的 WIP commit。不 force-push。
4. 不提交 `.env`、凭据、密钥、虚拟环境、模型权重或私人课程材料。提交前审查显式文件清单，不使用无差别 `git add .`。
5. 保持现有模型、59 个基线测试及已发布结果不变；新增结果写入独立目录。不得以放宽旧测试来迁就新实现。
6. Learned Parser 的训练输入、reward、mask、采样和选模型过程不得使用 gold programs、`parse_roles`、gold subject/reference labels。规则解析仅用于测试 oracle 与隔离的最终评估；给定的 curriculum 难度标签可用于选训练样本。
7. 区分语法有效性与语义正确性；mask 不得把固定模板的正确角色硬编码出来。记录关键词信息丢失、reward/程序准确率不一致、训练不收敛等结果。
8. 文档与结果明确写 `Our results`；不得声称复现原论文结果。规划中的 API/CLI 必须标注尚未实现。
9. CPU 是首版执行目标。不要仅调用 `.to('mps')` 就声称支持 MPS；现有 Executor 含 CPU 张量创建，需要单独设备一致性工作。
10. 每个阶段同步计划的完成状态、实际验证命令和输出；只有观测到成功才声称测试、推送或实验完成。
