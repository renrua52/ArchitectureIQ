# V1 收尾：重出 1000 题题库 — 计划书（草案 v0.2）

> 状态：**草案，大部分决策已确认**。剩余待定事项见文末第 5 节。
> 工作笔记（事实备查）：`docs/plans/v1-final-working-notes.md`

## 1. 目标

为 ArchitectureIQ V1 收尾，重新生成一版 **1000 道题**的题库（BakeFile/题库集合），旧版 1000 题改名归档保留。

## 2. 保持不变的部分（已确认 Q1-b）

- 生成管线不变：dataset instance → candidate sets + GT → questions（`spec → code → run → GT` 核心不变量不动）。
- 六大题目 bucket 划分不变：univariate / multivariate / bigram / xor / spiral / general_tabular，各 ≈1/6（166–167）。
- 题型划分与比例不变：`architecture_only` 400 / `optimizer_only` 300 / `mixed` 300（不使用 `loss_only`）。
- 3 个选项；gap_k=5.0；param_ratio_cap=1.5；max_failed_seeds=1；allow_reuse_fallback=true；seed=0。
- 组题工具沿用 `tools/benchmark_v1_build.py`（72 层配额计划：bucket × 题型 × gap × param）。
- 凡本计划未提及的环节，一律沿用旧版配置与流程。

## 3. 计划变更（已确认）

### 3.1 删除 KAN 架构（Q2-a：彻底删除）

- 从代码中删除 KAN：`models/kan.py`、registry 注册、`profile.py` 的 `kan` 属性、`prompts/formatters.py` 的 `format_kan_nl` 及 dispatch、inspector 镜像、`tools/` 下 kan 专属脚本、`tests/test_kan.py` 及其他测试中的 kan 引用。
- 从 `profiles/v1*.yaml` 的 pools 与网格中删除 kan。
- 日后如需恢复，从 git 历史中找回。
- v2.x profiles 中的 kan 字段一并清理（已确认 Q7），保持全仓一致。

### 3.2 MLP 架构精简（Q3-b）

- 去掉"激活函数"可变维度：所有 MLP 固定 **ReLU**（`mlp.activations` 网格移除；`leaky_relu_slope` 字段处置见 Q8）。
- `layer_norm`：改为**全局单 bool，candidate set 生成时全 set 采样一次**（已确认 Q3-impl 方案 1）。set 内所有候选共享同一开关 → 题内架构间天然一致；跨 set 保留 True/False 多样性。
- 其余可变组件保留不变：depth [1..6]、width [16..256]、residual [false,true]。
- `leaky_relu_slope` spec 字段保留占位（兼容旧 spec 加载，Q8）。

### 3.3 预算档位调整（Q1-b 补充）

- 删除 1024 档，保留 **2048 / 4096 / 8192 / 16384 四档**，各 ≈250 题。
- profile 的 budgets 池同步删除 1024。

### 3.4 profile 策略（Q4：按用户方案执行）

- 开发期间新建 `profiles/v1.1.yaml`（基于 v1.yaml 修改），旧 `v1.yaml` 保留不动以便对照。
- 发布（导出 BakeFile）前将 `v1.1.yaml` 重命名为 `v1.yaml`，旧文件归档（命名见 Q9）；manifest 中 profile_hash 会相应变化，可追溯。

### 3.5 旧题库去向（Q5-b：改名归档）

- `benchmarks/v1_llm/` → `benchmarks/v1_llm_legacy/`（已确认 Q9），新题库写到 `benchmarks/v1_llm/`。
- `benchmark_releases/v1_llm_bundle/` → `benchmark_releases/v1_llm_bundle_legacy/`；旧 profile 归档为 `profiles/v1_legacy.yaml`。

## 4. 执行步骤

0. ~~GT 可行性检验（Q6 前置）~~ **已完成**：结论 Q6 可行，全流程预计 2–6h（工作笔记 J 节）；生产 GT 必须并行（串行约 50h），驱动复用探针队列机制。
1. ~~按 3.1/3.2 修改代码与 `profiles/v1.1.yaml`~~ **已完成**（变更清单与两处自主判断见工作笔记 K 节）。
2. ~~更新受影响的测试与 parity 测试~~ **已完成**：`pytest tests/` 252 全过；另修复一个预先存在的过时 hash 断言（见工作笔记 K 节）。
3. ~~生成新 dataset instances 与候选池，跑 GT~~ **已完成**：43 datasets / 517 sets（含 top-up 173 个 n=30 大 set）/ 9934 候选，两轮 GT 共 ~3.1h 零报错（工作笔记 L/M/N 节；9 档宽度 + n=30 决策见 N 节）。
4. ~~运行 `benchmark_v1_build.py` 生成新 1000 题集合与 manifest~~ **已完成**：1000/1000，配额与旧版精确一致，relaxed=171（工作笔记 N 节）。
5. 归档旧题库与 bundle（3.5）；导出 BakeFile / 新 v1_llm_bundle。**部分完成**：旧题库已归档 `benchmarks/v1_llm_legacy/`；BakeFile 已导出 `frontend/quiz/public/data/questions.json` 并通过 schema 校验。剩余：旧 bundle（`/root/v1bundle`）归档、新 bundle 打包（如需）。
6. 发布前 profile 重命名（3.4）；更新 README / 文档中涉及 KAN、MLP 可变组件、预算档位的描述。

## 5. 待定事项

| # | 问题 | 状态 |
|---|------|------|
| Q1 | 配额沿用，预算删 1024 留 4 档 | **已确认** |
| Q2 | KAN 彻底删除（代码+测试），git 可恢复 | **已确认** |
| Q3 | MLP 固定 ReLU；layer_norm 题内不变；其余保留 | **已确认** |
| Q3-impl | layer_norm 全局单 bool、set 级采样一次（方案 1） | **已确认** |
| Q4 | 开发用 v1.1.yaml，发布前重命名为 v1.yaml | **已确认** |
| Q5 | 旧题库与 bundle 改名归档 | **已确认** |
| Q6 | GT 在本服务器约 1 天跑完 | **待可行性检验**（步骤 0） |
| Q7 | v2.x profiles 中 kan 字段一并清理 | **已确认** |
| Q8 | `leaky_relu_slope` 保留占位 | **已确认** |
| Q9 | 归档命名：`v1_llm_legacy/`、`v1_llm_bundle_legacy/`、`v1_legacy.yaml` | **已确认** |
| Q10 | 新版 supply 计划：**加大供给**——每 set 候选从 ~10 增至 12–16（dataset ~43 / set ~355 不变），压降 relaxed 题数；GT 约 2–3h，机时可行（实测见工作笔记 J 节） | **已确认** |
| Q11 | n_seeds 沿用 10 | **已确认** |

**全部设计决策已对齐。** 可行性检验结论：Q6 可行（全流程预计 2–6h，远低于 24h 预算）。生产 GT 必须并行跑（串行约 50h），驱动复用探针队列机制；探针脚本 `tools/gt_feasibility_probe.py` 留在本地、**不进 git**。
