# V1 收尾工作笔记（仅工作用，持续更新）

记录讨论中确认的事实、代码现状与决策依据。非用户交付物。

## A. 旧版 1000 题题库（事实）

- 位置：`benchmarks/v1_llm/`，含 `manifest.json`、`collection.json`、`questions/`（正好 1000 个 `q_xxxxxx` 文件夹）、`llm_results_matrix.json` 等 LLM 评测结果。
- 生成来源：manifest 记录 `created_from` 指向一个旧 worktree 的 `data/`，profile=`v1`，profile_hash=`90421abe32ec88c7`。
- 生成配置（manifest.config）：num_questions=1000，num_choices=3，seed=0，gap_k=5.0，param_ratio_cap=1.5，max_failed_seeds=1，allow_reuse_fallback=true。
- 配额统计（manifest.stats）：
  - 6 bucket 各 166–167 题：univariate 167 / multivariate 167 / bigram 167 / xor 167 / spiral 166 / general_tabular 166。
  - 题型：architecture_only 400 / optimizer_only 300 / mixed 300（无 loss_only）。
  - 预算档：1024:199 / 2048:198 / 4096:200 / 8192:205 / 16384:198。
  - gap_constrained True 700 / False 300；param_similar True 802 / False 198。
  - relaxed_questions=129（配额不足时放宽兜底）。
- manifest.generation_notes 中有 `kan_decision: "kept"` 及当时保留 KAN 的证据（A100 上 KAN GT 中位 46.7s vs MLP 22.5s，约 2 倍，可接受）。
- 组题脚本：`tools/benchmark_v1_build.py`（480 行），72 层配额 = 6 bucket × 3 题型 × gap(2) × param(2)。`QUESTION_TYPES = ("architecture_only", "optimizer_only", "mixed")`。
- 另有发布物：`benchmark_releases/v1_llm_bundle/`（manifest + README，供 GitHub Release 下载）；`benchmark_releases/question_packs/` 里有 v2.5 的 100 题小包（gru/xor），与 V1 1000 题不是同一批。

## B. KAN 接触点清单（删除时的影响面）

代码：
- `src/architecture_iq/models/kan.py` — KanModelFamily 本体。
- `src/architecture_iq/registry.py` — `_register_all()` 中 import + register。
- `src/architecture_iq/profile.py:105` — `Profile.kan` 属性。
- families 的 `compatible_model_types()` 返回 `["mlp", "kan"]`：
  - `families/univariate_regression/family.py:149`
  - `families/multivariate_regression/family.py:154`
  - `families/synthetic_tabular_classification/__init__.py:381`
- `prompts/formatters.py:52` `format_kan_nl` + `:107` dispatch。
- inspector 镜像 `tools/question_inspector/prompt_format.py`（parity 测试 `tests/test_prompt_format_parity.py`）。
- `tools/question_inspector/app.py`、`custom_settings.py`。

profile：
- `profiles/v1.yaml`：`pools.model_types` 中的 `kan` + 整段 `kan:` 网格（含 archetypes）。v2.x profiles 也可能含 kan（待查）。

工具/测试：
- kan 专属工具：`tools/generate_kan_mlp_demo.py`、`generate_kan_mlp_classification_calibration.py`、`generate_kan_mlp_multivariate_calibration.py`、`kan_mlp_benchmark_report.py`。
- 测试：`tests/test_kan.py`，以及引用 kan 的 `test_new_families.py`、`test_profile_model_gating.py`、`test_question_generation.py`、`test_v1_unified_pool.py`、`test_synthetic_tabular_classification.py` 等。

文档：AGENTS.md / README / docs/plans/kan/ 目录。

## C. MLP 可变组件现状（profiles/v1.yaml + models/mlp.py）

| 组件 | 现状 | 来源 |
|------|------|------|
| depth | [1,2,3,4,5,6] | profile `mlp.depth` |
| width | [16,32,64,128,256] | profile `mlp.width` |
| residual | [false,true]（全局一个 bool） | profile `mlp.residual` |
| activations | 每层独立采样自 [relu, leaky_relu, gelu, silu] | profile `mlp.activations` |
| layer_norm | 每层独立 `rng.choice([True, False])` —— **硬编码在 `MlpModelFamily.sample_spec()`，不在 profile** | 代码 |
| leaky_relu_slope | 固定 0.01（spec 中带字段） | profile `mlp.leaky_relu_slope` |

- spec 字段：`type/depth/width/residual/layer_norm(每层)/activations(每层)/leaky_relu_slope/input_dim/output_dim`。
- 渲染 `render_model_py` 生成独立 `_activation()` 映射（relu/leaky_relu/gelu/silu）。
- 注意向后兼容：`_sync_candidate_files()` 会用新 renderer 重渲染旧 candidate；若删 spec 字段需考虑旧 spec 校验/加载（`validate` 检查 depth==len(activations)==len(layer_norm)）。

## D. 已确认的决策

- **D1（Q1-b）：** 配额完全沿用旧版（6 bucket ≈ 各 1/6、题型 400/300/300、3 choices、gap_k=5.0、param_ratio_cap=1.5、max_failed_seeds=1、allow_reuse_fallback、seed=0），唯一改动：预算删 1024 档，保留 2048/4096/8192/16384 四档，各 ≈250 题。
- **D2（Q2-a）：** KAN 彻底删除（代码 + 测试 + profile + 工具），日后需要从 git 历史恢复。
- **D3（Q3-b）：** MLP 激活函数固定 ReLU；layer_norm 在同一题目的各 choice 间保持不变；depth/width/residual 网格保留不变。
- **D4（Q4）：** 开发期间新建 `profiles/v1.1.yaml`，发布前重命名为 `v1.yaml`，旧 v1.yaml 归档。
- **D5（Q5-b）：** 旧 1000 题（`benchmarks/v1_llm/`）与 `benchmark_releases/v1_llm_bundle/` 改名归档保留，不删除。
- **D6（Q6）：** GT 目标在本服务器约 1 天跑完；先做可行性采样计时再开工。
- **D7（Q3-impl）：** layer_norm 采用方案 1——全局单 bool，candidate set 生成时全 set 采样一次，set 内所有候选共享。
- **D8（Q7）：** v2.x profiles 中的 kan 字段一并清理。
- **D9（Q8）：** `leaky_relu_slope` spec 字段保留占位，兼容旧 spec 加载。
- **D10（Q9）：** 归档命名 `benchmarks/v1_llm_legacy/`、`benchmark_releases/v1_llm_bundle_legacy/`、`profiles/v1_legacy.yaml`。
- **D11（Q11）：** n_seeds 沿用 10。
- **D12：** 执行顺序：① GT 可行性检验 → ② 改代码（删 KAN、MLP 精简、v1.1.yaml）→ ③ 重建数据集/候选池/出题。

## E. 环境备忘

- 本机无 `rg`，用 `grep`/`find`。
- **解释器（已修正）：** 必须用 `/root/miniconda3/envs/aiq/bin/python`（Python 3.10.18，torch 2.6.0+cpu）。base 环境 `/root/miniconda3/bin/python` 是 3.8，跑不动现有代码（`zip(strict=True)` 需要 3.10+）。
  **注意：** aiq 环境里有指向兄弟 worktree（`/root/autodl-tmp/guoshaoyang/ArchitectureIQ`）的 editable 安装，直接 `import architecture_iq` 会串到那边；必须 `sys.path.insert(0, 本仓 src)` 后再 import（探针脚本已内置）。已验证这样 import 解析到本仓。
- 当前分支：`feat/tcc-quiz-backend-dev`；工作树根：`/root/autodl-tmp/tcc/ArchitectureIQ`。
- **GPU：本服务器无可用 GPU**（nvidia-smi 二进制存在但无驱动）。32 核 CPU，377GB RAM。→ GT 跑 CPU；实测表明 tiny 模型在 CPU 上很快（见 I 节结果）。

## F. Q3-impl 技术约束：layer_norm"题内架构间不变"

事实：
- 组题组合只在**单个 candidate set 内部**枚举（`benchmark_v1_build.py` 的 `index_set_entries` 用 `combinations(pool, 3)`，题目从不跨 set 混候选）。
- layer_norm 现状：**每层独立 `rng.choice([True, False])`，硬编码在 `MlpModelFamily.sample_spec()`**，不在 profile。depth=6 时有 64 种 pattern。
- 若保持现状、仅在组题时筛"layer_norm 向量一致"的三元组：同 depth 且同 pattern 的三候选极稀疏，基本上组不出题。**不可行。**

可选实现方案：
- **方案 1（已确认采用）：layer_norm 改为全局单 bool，且在 candidate set 生成时全 set 采样一次。** set 内所有候选共享同一 layer_norm 开关 → 题内架构间天然一致；跨 set 仍有 True/False 多样性，benchmark 级别保留该因素的数据。改动：spec 字段 `layer_norm: list[bool]` → `bool`（或保留 list 但全层同值），`sample_spec` 接受 set 级注入值，validate/render 微调。
- 方案 2：保留每层 list，set 内共享一条"最大深度模板"，候选按自己 depth 取前缀。保留层级差异但语义别扭、实现绕。
- 方案 4（最简但丢维度）：layer_norm 全 False 固定，等同删除该可变组件。

## G. Q4 方案评估（Verdent 意见：合适，可执行）

- 优点：开发期间新旧 profile 并存，可随时对照/回滚；发布物干净（只有一个 v1.yaml，与题库名 v1_llm 对应）。
- 注意事项：
  1. `benchmark_v1_build.py --profiles` 默认 `"v1"`，发布前重命名后默认行为即指向新配置，开发期间需显式传 `--profiles v1.1`。
  2. profile_hash 会变，新 manifest 与旧 manifest 的 hash 不同，可追溯，无冲突。
  3. 重命名时机建议在导出 BakeFile 之前完成，保证发布物内部一致。
  4. 旧 v1.yaml 的归档命名与旧题库归档命名一起定（Q9）。

## H. 旧版组题 supply 统计（从 benchmarks/v1_llm/questions/*/question.json 实测）

- 1000 题共使用 **2671 个唯一候选**（3000 个 choice 槽位，复用约 329 槽，与 relaxed=129 + 部分复用一致）。
- **355 个 candidate set**、**43 个 dataset instance**（univariate 8 / multivariate 7 / bigram 8 / synthetic_tabular_classification 20）。
- synthetic_tabular_classification 一家覆盖 xor/spiral/general_tabular 三个 bucket 共 499 题。
- 平均每 set 约 7.5 个候选被用上进题（实际 set 内生成候选更多，未通过质量/显著性过滤的未入题）。
- 推论：新版 GT 总量 ≈ 3000+ 候选 × n_seeds=10 次训练；删 1024 档省约 20% 预算。可行性检验需实测 CPU 单候选 GT 时间后外推。

## I. GT 可行性检验方案（待用户同意后执行）

**目标：** ① 实测本服务器（32 核 CPU，无 GPU）单候选 GT 墙钟时间；② 调优并行配置使吞吐最大化；③ 外推全量时间，定 Q10 supply 计划。

**已有并行能力（代码现状，无需改动）：**
- seed 级并行：`ground_truth/runner.py` 支持环境变量 `ARCHITECTURE_IQ_SEED_WORKERS`（候选内 n_seeds 用 ProcessPoolExecutor 并行）+ `ARCHITECTURE_IQ_SEED_TORCH_THREADS`（每 worker 线程数，默认 1）。仅 CPU 模式生效。
- 候选级并行：探针脚本自行多进程跑多个候选。
- 总并发约束：`并行候选数 × seed_workers × torch_threads ≤ 32`（留 1–2 核余量，按 30 算）。

**采样矩阵（Phase A 延迟基线）：** 4 family（univariate / multivariate / bigram / synthetic_tabular_classification）× 2 模型规模（depth1-width16 / depth6-width256）× 2 预算档（2048 / 16384）= 16 个代表性候选。慢格子可用 n_seeds=2 计时再 ×5 外推（seed 间近似线性），单格子设超时保护（如 20 min，记录为下限）。synthetic_tabular_classification 本地无实例，先补建 1 个。

**并行调优（Phase B 吞吐实测）：** 用代表性混合负载测 2–3 种配置，比较 candidates/hour：
- C1：30 候选并行 × 1 seed-worker × 1 线程
- C2：8 候选并行 × 4 seed-workers × 1 线程（≈32 核）
- C3（备选）：15 候选 × 2 seed-workers × 1 线程

**外推与产出（Phase C）：** 按旧版构成（2671 入题候选 + 未入题损耗，删 1024 档后预算分布）估算总机时 → 给出 24h 内可行的 supply 计划（dataset 数 / set 数 / 每 set 候选数，即 Q10）→ 结果写回本节与计划书。

**实施物：** 探针脚本 `tools/gt_feasibility_probe.py`（走标准 `write_candidate` + `run_ground_truth` 管线，产物放任务专属临时目录，不改任何 pipeline 代码）。

**注意：** profile `budgets` 池为 [1024, 2048, 5120, 10240, 20480, 40960]，与旧组题的 5 档（1024–16384）不同——预算档由 generate-candidates 的 `--budget` 决定，探针直接在 spec 里写目标预算。

## J. 可行性检验结果（2026-08-23 实测，aiq 环境 py3.10/torch2.6-cpu）

**Phase A 延迟基线**（n_seeds=2，单线程，墙钟秒；×5 ≈ 生产 10-seed 单候选耗时）：

| family | 模型 | b=2048 | b=16384 |
|--------|------|-------:|--------:|
| univariate | mlp_small | 0.56 | 0.96 |
| univariate | mlp_large | 1.16 | 5.64 |
| multivariate | mlp_small | 0.56 | 0.97 |
| multivariate | mlp_large | 1.17 | 5.40 |
| stabcls | mlp_small | 0.58 | 1.24 |
| stabcls | mlp_large | 3.03 | 20.13 |
| bigram | transformer_small | 1.14 | 5.75 |
| bigram | transformer_large | 6.13 | **46.51（最坏格）** |
| bigram | gru_large | 1.96 | 11.83 |

最坏格（transformer_large @16384）生产耗时 ≈ 232s/候选。全部 18 格零失败零超时。

**Phase B 吞吐**（40 个生产分布混合候选，n_seeds=10，零失败）：

| 配置 | 并发 | seed_workers | 墙钟 | candidates/hour |
|------|-----:|-------------:|-----:|----------------:|
| C1 | 30 | 1 | 84.1s | 1712 |
| C2 | 8 | 4 | 56.1s | 2567 |
| **C3（推荐）** | **15** | **2** | **52.1s** | **2765** |

**Phase C 外推（1000 题全量）：**
- 规模假设：~3500–4000 候选（旧版 355 set × ~10；2671 入题）。
- GT 机时：4000 ÷ 2765/h ≈ **1.5 小时**（C3 配置）；打 3 倍余量也仅 ~4.5h。
- 对比：串行跑（现有 CLI 默认）≈ 45 core-s/候选 × 4000 ≈ 50 CPU-h，32 核串行等效 ~50h → **生产必须并行跑 GT**，驱动可直接复用探针的 `_run_queue` 队列机制。
- 数据集生成 / 候选生成 / 组题 / 导出均为分钟级。
- **结论：1 天预算非常充裕，预计全流程 2–6h。** Q6 可行。
- Q10 建议：维持旧版规模（43 dataset / ~355 set / 每 set ~10 候选）；机时富余，可考虑每 set 增至 12–16 候选以降低 relaxed 题数（旧版 129）。
- 探针产物：`tools/gt_feasibility_probe.py`；数据 `data/gt_probe/`（gitignored）；stabcls 测试实例 `stabcls_8f41be`。

## K. 代码修改完成记录（2026-08-23，计划书步骤 1–2）

**MLP 精简（D3/D7）：**
- `models/mlp.py`：`sample_spec` 固定 ReLU（每层 list 形状保留，兼容旧 spec）；layer_norm 改全局单 bool，接受 `shared["layer_norm"]` 注入，standalone 采样回退 rng。`leaky_relu_slope` 占位保留（D9）。
- `models/base.py`：`sample_spec` 抽象签名加 `shared` 参数；`gru_lm.py`/`transformer_lm.py` 同步签名（忽略该参数）。
- `candidates/generator.py`：`sample_model` 透传 `shared`。
- `candidates/sets.py`：`sample_candidate_set_pool` 每 set 采样一次 layer_norm（仅在会采样 model 时），经独立 `model_shared` 注入两个 `sample_model` 调用点；不进 manifest `fixed_shared`（值可从 set 内任一候选 spec 读回，rng seed 可复现）。

**KAN 删除（D2/D8）：**
- 删 `models/kan.py`；registry、`profile.py` 的 `kan` 属性、formatters + inspector 镜像（`prompt_format.py`）的 `format_kan_nl` 及 dispatch、三个 family 的 `compatible_model_types()`（→ `["mlp"]`）。
- inspector：`app.py` 删 KAN 编辑器（import、`_kan_defaults`、`_kan_activation_options`、`_render_kan_setting_fields`、dispatch 分支）；`custom_settings.py` 删两个 kan 分支。
- 删 4 个 kan 专属工具 + `tests/test_kan.py` + 3 个 kan 工具测试。
- **超出原清单的判断（已向用户报告）：** 另删 `tools/build_xor_sampled_pool.py`、`tools/materialize_xor_candidate_matrix.py` 及各自测试——它们是 v2.x MLP/KAN 对比 pilot 工具，运行时依赖 kan 注册与 v2.x kan 网格，D2+D8 后必然损坏。保留 `tools/audit_xor_pair_capacity.py` + 测试（纯读已有 artifact 的审计逻辑，不需要 kan 注册，对 legacy 数据仍可用）。
- `benchmark_v1_build.py`：`BUDGET_TIERS` 删 1024 档（D1）；manifest `generation_notes` 改为 `kan_decision: "removed"`。
- v2.x profiles 清理（D8）：v2 / v2.1 / v2.2 / v2.3-gru-pilot / v2.3-xor-pilot / v2.4-xor-review / v2.5-xor-holdout / v2.5-xor-screen 删 pools 的 kan、`kan:` 网格块、引用 kan 的 model_gates（family 默认已是 mlp-only，gate 冗余故整体删除）。**`v1.yaml` 按 D4 保持不动**（发布前才归档为 v1_legacy.yaml）。
- 测试更新：`test_new_families` / `test_v1_unified_pool` / `test_synthetic_tabular_classification` 期望改 mlp-only；`test_prompt_format_parity` 删 kan 项；`test_question_inspector` 删 3 个 kan 编辑器测试；`test_question_generation` 的 manifest 标签 kan→gru_lm；`test_profile_model_gating.py` 整体重写为 v2.x mlp-only 契约。

**v1.1.yaml（D4）：** 基于 v1.yaml；两处自主判断已报告：① budgets 池直接设为四档 `[2048, 4096, 8192, 16384]`（与 benchmark tiers 对齐，interactive/audit 工具一致；旧池的 5120/10240/20480/40960 不属于 benchmark 档位）；② mlp 段保留 `activations: [relu]` 单选项网格（`interactive.py` 会枚举该键，删除会 KeyError；采样代码不读它）。

**更正与补充（2026-08-23，budget 池考证）：** 上一轮我误称旧池含 9 个值（含 4096/8192/16384）——实际旧 v1.yaml 池自初始提交起就是 6 值 `[1024, 2048, 5120, 10240, 20480, 40960]`（约束：能被 batch_size 16/32/64 整除），与 git 已提交版本一致。消费方只有 `interactive.py`（预算选项）和 `question_audit_lib.py`（allowed_budgets 校验）；CLI `generate-candidates` 必须显式传 `--budget`，**旧 1000 题的供给驱动绕开了池、直接按五档 tier（1024/2048/4096/8192/16384）传预算**——4096/8192/16384 甚至不在池里，证明池从未约束 benchmark 生成。旧 manifest（`/root/ArchitectureIQ/benchmarks/v1_llm/manifest.json`）stats 只有这五档，5120/10240/20480/40960 从未进入任何题目，属于初始设计遗留的探索性选项。v1.1 把池设为四档是"让池匹配实际用法"，不改变 benchmark 行为。

**git 历史已找到：** `/root/ArchitectureIQ`（用户确认可跨 worktree 查看）是健康克隆，`shaoyang/local-agent-dev` + `origin/quiz-backend`（= 本 worktree 的 a763818）历史完整，`kan.py` 在历史中（c8e582d、8f60bf4），D2 的"从 git 恢复 KAN"有保障。丢失的 checkpoint 提交 74cbbd3 在该克隆中也不存在（纯本地 checkpoint，未推送），无实际损失。旧 1000 题 manifest 也在该克隆：`benchmarks/v1_llm/manifest.json`，profile_hash 90421abe 与本 worktree 的 v1.yaml 一致，再次佐证 hash 测试修正正确。本 worktree 的 git 修复方案（分支指回 a763818）仍待用户拍板。

**测试修正（预先存在，与本次改动无关）：** `test_execution_device.py` 期望 v1 profile_hash `164f68c29f6730dc`，但本 worktree 的 v1.yaml 实际 hash 为 `90421abe32ec88c7`——与旧 1000 题 manifest 记录一致，证明盘上 v1.yaml 才是生成旧题库的版本，测试期望已过时，已改为实际 hash。

**验证：** `pytest tests/` 252 全过（含 parity）；冒烟：v1.1 下 set 内 layer_norm 全一致、activations 全 relu、跨 set True/False 多样性保留、write→GT 链路跑通（mean_test_mse 正常）。

**环境警告：本 worktree git 历史损坏。** HEAD（`feat/tcc-quiz-backend-dev`）指向的 commit object 缺失，`git status/log/diff` 均不可用；仅剩 `origin/quiz-backend` 分支可达。D2"日后从 git 历史恢复 KAN"依赖远端或其他 worktree 仍有历史——恢复能力未验证。本次改动无法 git diff，变更清单以本节为准。

**留到发布前（计划书步骤 6）：** README/AGENTS.md/docs 中 KAN、MLP 可变组件、预算档位的描述更新；v1.1.yaml→v1.yaml 重命名；旧题库/bundle/profile 归档（D5/D10）。

## L. 供给生成完成记录（2026-08-23，计划书步骤 3 前半）

**git 修复（用户批准）：** `git update-ref refs/heads/feat/tcc-quiz-backend-dev a763818`（只动分支指针，工作区/索引未动），git 全部功能恢复；status 显示改动与 K 节清单一致。尚未提交（用户决定先不提交）。

**新增两个工具（待提交）：**
- `tools/benchmark_v1_supply.py`：数据集 + set 骨架生成（无 GT）。只走 canonical 管线（`create_dataset`/`sample_candidate_set_pool`/`write_candidate`/`write_set_manifest`），幂等可重跑。
- `tools/benchmark_v1_gt.py`：并行 GT 驱动（C3 配置：15 candidate workers × 2 seed workers × 1 torch thread）。扫描缺 `results/summary.json` 的候选，resume-safe，状态写 `data/benchmark_v1_gt_status.json`。注意 summary 里 `failed_seeds` 是 int。

**供给方案（已执行）：** 43 datasets（uni/mvar/bigram/xor/spiral 各 7，general_tabular 8 = smooth_additive 3 + sparse_interaction 3 + piecewise_boundary 2；stabcls 用 `rule_family` family_option 定向生成，bucket 判定见 `benchmark_v1_build.dataset_bucket`）；每 dataset 8 sets（344 总），vary 配比按旧题型配额（arch 400/opt 300/mixed 300）定为 3×{model} + 2×{optimizer} + 2×{model,optimizer,loss} + 1×{optimizer,loss}；预算四档轮转各 86 sets；每 set 12–16 候选。

**执行结果审计（全过）：** 43 datasets / 344 sets / **4802 candidates**；tier 各 86 sets；每 set 候选数 12–16 均匀；kan=0；MLP 激活全 relu；**set 内 layer_norm 全一致**（候选内与跨候选均验证），跨 set True/False = 129/159（288 个含 MLP 的 set）；budget = steps × batch_size 全一致。旧 8 个 dataset 已移至 `data/datasets_legacy_v1/`（reversible）。

**GT 预计：** 4802 ÷ 2765/h ≈ **1.7h**（C3 实测吞吐）。启动命令：
`python tools/benchmark_v1_gt.py --profile v1.1 --workers 15 --seed-workers 2`
组题命令（GT 完成后）：
`python tools/benchmark_v1_build.py --data-root data --profiles v1.1`（已确认 `--data-root`/`--profiles` 参数存在；glob 路径与现布局匹配）。

## M. GT 完成 + 组题缺口诊断（2026-08-23）

**GT 结果：** 4802 候选，**1.21h** 跑完（峰值 ~3983/h）：ok=4498，excluded=242（failed_seeds≥3），failed_seeds≤2 有 62，**error=0**。状态在 `data/benchmark_v1_gt_status.json`。

**组题（步骤 4）首次运行：743/1000，缺 257。** 注意 `benchmark_v1_build.py` 没有 sys.path 置顶，需 `PYTHONPATH=src` 跑（否则落到 guoshaoyang worktree 的 editable 安装）。运行前已把旧 build 产物归档到 `benchmarks/v1_llm_legacy/`（D5 的一部分）。

**缺口全部集中在 param_similar=True（参数量比 ≤1.5）的 strata**：
- arch+ps：uni 9 条 / mvar 8 / bigram 97 / xor 6 / spiral 11 / general 11（每 bucket 需 ~54 题）
- 另 xor mixed+ps 缺 19
- 根因：**删 KAN 后 model 池只剩 MLP（stabcls/mvar/uni）或 transformer+gru（bigram）**。MLP 宽度 16→256 翻倍带来 ≥4× 参数跳变，depth 1→2 参数近似翻倍；ps_ok 三元组只剩"同宽度 + 相邻深度（{3,4}/{4,5}/{5,6}）± residual"这种窄带。arch 参数比中位数 6.8（bigram）到 61（uni），p10 都有 2.3–7.5。旧版靠 KAN 的不同参数量级填满这些 strata——这是删 KAN 的直接后果（用户要求有问题就提）。
- 备选修复方案：
  1. **（推荐）补 narrow-band arch 供给**：扩展 `mlp.sample_spec` 的 shared 支持 pin width + 限制 depth 集合，补生成"同宽度、相邻深度、residual 两态"的小 set（每 set 6–8 候选），每 bucket ~20–40 set ≈ 1200–1600 候选，GT ~30–40min。保持配额与题目语义不变。
  2. 放宽 param_ratio_cap：不够——即使 cap=3，uni/mvar 的 ps 供给仍差一个量级；cap 提到 ~8 则"param_similar"名存实亡。
  3. 削减 ps 配额、重新分配：偏离旧版比例（用户要求对齐旧版），不推荐。
- 当前 `benchmarks/v1_llm/` 只有 assembly_shortfall.json（build 未写题目）；bake 未做，等配额填满。

## N. 参数量相近缺口修复：9 档宽度 + n=30 大 set（2026-08-23，用户拍板）

**讨论结论（模拟验证，tools 模拟脚本未留存）：**
- 加中间宽度档（24/48/96/192）本身不提升 ps 三元组总量（参数 ∝ width²，相邻档比 1.5 → 参数比 2.25 > cap 1.5），n=14 时 3.5→4.0 条/set。
- **扩大 set 才是数量杠杆**：三元组 C(n,3) 立方增长、GT 线性。n=30 时每 set 原始 ps 三元组 36→42 条（5档→9档），每候选产出 ~6 倍于 n=14。
- **两者结合的化学反应**：9 档 + n=30 时跨宽度簇占比 49%→**88%**——用户要的"宽度 vs 深度取舍"题成为 ps 题主力。
- bigram（tf/gru 池）同理解决：参数量同样由二次项主导，n=30 模拟 ~132 条原始 ps 三元组/set，投影 gap∧ps ≈300 条 vs 配额 38，**不需要动 d_model 网格**。
- distinct cell 天花板：MLP 9档 99 个 (d,w,res) 组合，bigram 58 个，n=30 放得下。

**已确认的决策：** ① 宽度 9 档 `[16,24,32,48,64,96,128,192,256]` 为 v1.1 正式池定义；② 旧 129 个小 arch set 保留在池；③ n=30 固定。

**执行：**
1. `v1.1.yaml` mlp.width 改 9 档（已做，带注释）。
2. `benchmark_v1_supply.py --topup`：每 dataset +3 arch set（vary={model}，n=30，全部 bucket）+ stabcls 每 dataset +2 mixed set（n=30）。幂等（按已有大 set 计数跳过）。
3. top-up 结果：**173 set / 5190 候选**（arch 129 + mixed 44——44 而非 42 是因为 22 个 stabcls dataset × 2），预算四档分布 38/39/46/50，9 档宽度分布均匀（每档 482–538），layer_norm set 内一致性审计通过。
4. GT round 2 启动（C3 配置），预计 ~1.3h。
5. 之后 rebuild + bake；若仍有缺口则 top-up 循环（脚本幂等）。

**GT round 2 结果：** 5132 候选，1.86h，ok=4919 / excluded=159 / error=0。速率前慢后快（pending 扫描按字母序 bigram 先行，transformer 最贵 ~232s/候选；MLP 段飙到 2000+/h），非故障。

**cross_profile_reuse 事件：** 宽度改 9 档后 profile_hash 变化（H1=fe2eca16 旧 4802 候选 → H2=fdad4cc5 新 5190 候选），build 拒绝混合池。因旧 5 档是 9 档子集（旧候选是新网格的合法样本），在 v1.1.yaml 加 `cross_profile_reuse` allowlist 放行两个 hash 的全模型类型。现 profile_hash=5463a577（发布前重命名还会再变）；manifest 的 candidate_profile_provenance 记录两个来源，可审计。

**build 成功：1000/1000**，配额与旧版精确一致（bucket 166–167、arch/opt/mixed 400/300/300、gap 700/300、ps 802/198）。relaxed=171（旧版 129，供给近翻倍但宽松题略增，可接受；若在意可对缺口 stratum 再 top-up）。清理了首次失败 build 残留的 2 个游离 question 目录（q_c03bef、q_c4032b）。

**bake 完成：** `export_quiz_static.py --data-root benchmarks/v1_llm`（需先在 benchmarks/v1_llm 建 `datasets -> ../../data/datasets` 软链，exporter 按 data-root 解析数据集）→ `frontend/quiz/public/data/questions.json`（1000 题，`validate_quiz_bake.py` 通过）。旧 46 题演示 bake 备份为 `questions.json.bak`。

**目标验收：** arch+参数量相近 322 题中，MLP 跨宽度 222 / 同宽度 46（跨宽度占 MLP 题 83%），191 题用到新宽度档；bigram 54 题为 tf/gru 组合。用户目标 2（宽度 vs 深度取舍题）达成。

**产物位置：** 题库 `benchmarks/v1_llm/`（gitignored）；旧版 `benchmarks/v1_llm_legacy/`（git 内原有产物，已 mv 归档）；前端 bake `frontend/quiz/public/data/questions.json`（git 跟踪，已更新待提交）。

**剩余（发布前，计划书步骤 6）：** README/AGENTS.md/docs 更新（KAN 删除、MLP 9 档、预算四档、新工具链 benchmark_v1_supply/benchmark_v1_gt）；`v1.1.yaml`→`v1.yaml` + 旧 v1.yaml→`v1_legacy.yaml`；`/root/v1bundle` 归档（D10）；git 提交（时机用户定）。

**题面修饰修复（2026-08-23，用户抽查后）：** ① `models/mlp.py` 渲染模板 MLPBlock 行缩进 8→12 空格，与 `nn.Linear` 行对齐（纯空白，GT 不受影响；build 时 `_sync_candidate_files` 按 spec 重渲染磁盘代码）。② `formatters.py` + inspector 镜像 `prompt_format.py`："1 hidden layers" 单复数修正（depth==1 时 "hidden layer"）。修复后已重跑 build（1000/1000，配额不变）+ bake（validate 通过）。parity 测试 18 过。抽查 4 题（q_f8db31/q_fe3196/q_1b82b0/q_b1704f）GT 答案全部核对无误。
