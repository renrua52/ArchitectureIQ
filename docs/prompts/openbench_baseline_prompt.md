# Prompt：开源 Benchmark → 数据集 → Baseline → 扰动候选（新 session）

你是 ArchitectureIQ 的成员。请在这个代码库（`/Users/guoshaoyang/Desktop/workdir/ArchitectureIQ`，分支 `shaoyang/local-agent-dev`）里完成一次"开源数据 → 数据集实例 → 好 baseline → 扰动候选"的完整闭环。先读 `AGENTS.md`（尤其第 1 节核心不变量和第 5 节复用规则），再动手。

## 目标

1. **取数据**：从开源 benchmark / Kaggle / HuggingFace 等选择一个**规模小、训练快（单候选 ≤ 2 分钟 CPU）、可复现**的有监督任务（如一个表格/分类/回归数据集），下载并封装成本仓库的 problem 实例：
   - 按列式存储放 `backend/data/problems/{problem_id}/`，写入 `problem_spec.json`（含 `selection_metric`、`files`、`dataset_id`/`problem_id`）+ `README.md`；
   - 写 `synthesize.py`，保证 `import → synthesize()` 可复现地生成 `train.pt` / `test.pt`（固定 seed，写进 spec）；
   - 数据集来源、许可证、下载/生成脚本、seed、预处理要全部记录在 problem 的 README 里，不允许"隐式在线依赖"（离线可重建）。
2. **Baseline**：用本仓库 pipeline（`write_candidate` → `run_ground_truth`，不要手写训练循环）在候选池里跑出**一个或几个好的 baseline setting**：
   - baseline 定义 = 在当前数据集上表现好（按 `selection_metric` 排名靠前）且训练预算合理（`total_samples_seen` 适中）的 setting；
   - 保存 `results/summary.json`（n_seeds ≥ 5）和 `curves.npz`；把 baseline 的 candidate_id 写进 problem README。
3. **扰动候选（autoresearch 风格的比较集）**：围绕 baseline 生成一批扰动 candidates，**每次只改 1–2 个维度**（模型 depth/width/残差、optimizer/lr/weight_decay、loss、batch_size），并遵守硬约束：
   - **参数量 ≤ baseline 最大参数的 1.1 倍**；
   - **训练量 `total_samples_seen` ≤ baseline 的 1.1 倍**（即不许"更大模型 + 更久训练"来赢，只允许架构/优化器选择差异）；
   - 候选配置是闭集：可调参数及其取值必须来自你写在 problem README 里的表格，模型输出只能是该闭集内的组合。
4. **产出与验证**：
   - 每个候选走 `spec → 渲染 .py → 执行 → GT` 的完整路径（`candidate_spec.json` + `model.py`/`train.py` + `results/summary.json`），禁止任何旁路；
   - 给出一张 baseline vs 扰动候选的对比表（mean ± std、seed 级 win-rate、ratio），并指出哪些改动"明显有益/明显有害/无差异"；
   - 如果发现某个扰动超过 1.1× 约束，标红并说明你如何拒绝/修正。

## 注意

- 先做一个小规模 smoke（1 个数据集 + ≤ 5 个候选）验证整条链路再扩展；
- 不要提交 `backend/data/`（gitignored）；可以提交 `docs/`、problem README 之外的代码改动；
- 完成后报告：数据集来源、baseline 的 candidate_id 和 loss、扰动候选清单与对比表、违反 1.1× 约束的情况。
