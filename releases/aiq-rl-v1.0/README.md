# AIQ-RL v1.0

这是 Qwen3.5-0.8B 的 6k 强化学习主线使用的完整题目与输入包。它独立编号，正式 ArchitectureIQ benchmark 的版本仍为 v1.5。

| 来源 | 训练题 | 留出题 |
|---|---:|---:|
| 旧选择题复用 `mcq_replay` | 1,760 | 0 |
| 第二版排序题 `ranking_v2` | 1,939 | 60 |
| 条件反转题 `dataflip_v2` | 1,434 | 80 |
| 第三版排序题 `ranking_v3` | 900 | 100 |
| 架构选择题 `arch170` | 120 | 50 |
| 合计 | 6,153 | 290 |

训练集包含 3,314 道选择题和 2,839 道排序题。留出集包含 130 道选择题和 160 道排序题。80 道条件反转留出题来自 40 对题，每对整体进入同一划分。

## 文件与训练输入

| 文件 | 内容 |
|---|---|
| `train.jsonl` / `heldout.jsonl` | 原始冻结题目，含 prompt、标准答案、题号、来源、任务、分组和元数据。原始字节保持不变。 |
| `train_think.jsonl` / `heldout_think.jsonl` | 实际训练与评测使用的原生 thinking 输入。使用这两个文件复现实验。 |
| `arch170_canonical.jsonl` | 从旧题池按原始行提取的 170 道架构题，用于复现统一答案格式。 |
| `prep_rows.py` | 冻结转换脚本，去掉 system 消息，统一架构题答案格式。 |
| `POOL_MANIFEST.json` | 原始题池 manifest，包含来源、切分、泄漏检查与历史待补项。历史路径是来源记录，不是运行依赖。 |
| `MANIFEST.json` / `SHA256SUMS` | 本次 release 的版本、文件身份、计数与校验值。 |
| `LICENSE` / `LICENSE.converter` | 题目采用 ArchitectureIQ MIT 许可。转换脚本保留 PoorRL Apache-2.0 许可。 |

每行是一个 JSON 对象。原始 `prompt` 是聊天消息列表，`answer` 是选项字母或五个字母的排序链。`question_id` 是冻结题号。`source`、`task`、`family`、`type` 和 `group` 保留题源及划分身份。转换后只保留训练器需要的字段。

在本目录校验与重建：

```sh
shasum -a 256 -c SHA256SUMS
python3 prep_rows.py convert train.jsonl /tmp/aiq-rl-v1.0-train-think.jsonl --arch170-canonical arch170_canonical.jsonl
python3 prep_rows.py convert heldout.jsonl /tmp/aiq-rl-v1.0-heldout-think.jsonl --arch170-canonical arch170_canonical.jsonl
cmp train_think.jsonl /tmp/aiq-rl-v1.0-train-think.jsonl
cmp heldout_think.jsonl /tmp/aiq-rl-v1.0-heldout-think.jsonl
```

训练输入 MD5 `669b570dd041fc5eb06d84e121f41c37` 匹配冻结训练记录。留出输入 MD5 `6c922effa8e2538888251540e155cffd` 与冻结评测输入逐字节一致。MD5 仅用于匹配历史实验，文件完整性使用 SHA256。

## 证据边界

标准答案来自原题池的执行结果。本次 release 保留既有题目与标签，没有重新执行所有候选训练。

冻结 manifest 的题号、分组、实例、数据集、监督微调留出泄漏与反转对切分检查均为零重叠。`dataflip_v2` 在当时尚未完成正式全源 release 与完整复审，原 manifest 的此项限定仍然适用。部分来源使用不同质量门槛，不能把整个混合题池当作正式 v1.5 benchmark 的同等认证版本。

这 290 道留出题已用于多次训练阶段比较。用于新的模型选择或宣称泛化提升时，需要另外冻结独立确认集。题池包含 `xor_classification`，本次忠实复现历史实验，不等于后续推荐题池。

本包完整覆盖这条 6k RL 主线的题目和实际输入。它不包含其他历史 RL 题池、监督微调教师回答、模型权重、全部 rollout 或候选训练的原始测量文件。

## 获取与使用

两个仓库提供完全相同的 `aiq-rl-v1.0` 数据包：

- [ArchitectureIQ 官方 release](https://github.com/renrua52/ArchitectureIQ/releases/tag/aiq-rl-v1.0)
- [poor-rl release](https://github.com/guoshaoyang-pku/poor-rl/releases/tag/aiq-rl-v1.0)

[6k 主线结果](https://guoshaoyang-pku.github.io/blogs/aiq-rl-pool-b-2026-10-06.html)保留曲线、配对统计和真实回答。[PoorRL H200 配方](https://github.com/guoshaoyang-pku/poor-rl/blob/main/docs/RECIPE_H200.md)说明训练入口。

引用数据时请标明 **AIQ-RL v1.0 / pool B**，并引用 ArchitectureIQ 的论文或官方仓库。
