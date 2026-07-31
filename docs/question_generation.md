# 出题逻辑文档 · ArchitectureIQ Question Generation

本文件说明 ArchitectureIQ 的**出题（question generation）完整逻辑**：从数据集实例到候选，再到题目与 prompt 的每一步、每个可调旋钮、以及生成每种题型的精确命令。它是对 [AGENTS.md](../AGENTS.md) 的补充——AGENTS.md 讲“不变量与架构”，本文讲“怎么出题、旋钮在哪、命令怎么敲”。运营视角的状态总账与后续计划见 [STATUS_AND_PLAN.md](./STATUS_AND_PLAN.md)。

> **一句话原则**：好的 ArchitectureIQ 题，不是 target 看上去多复杂，而是人能从**生成机制**提出明确的 architecture hypothesis，且这个 hypothesis 能被**同一套 generated-code GT pipeline** 稳定验证。

> **有效性、离线惊讶代理和用户惊讶是不同轴**：
> - **有效性 (validity)**：答案由数据决定而非噪声（GT 稳定、win_rate 高、方差不重叠）。
> - **`intrinsic_surprise_proxy`**：用反启发式、blind-solver 失败和存量 GT 测量“事前多难叫准”。这是难度的离线冷启动 proxy，不是用户标签。
> - **`observed_surprise_rate`**：用户在答案/真实排名揭晓后显式选择“出乎意料 / 符合预期”才能得到；本地 reaction 采集已实现，权威聚合/report 尚未实现。
> - **`predicted_personal_surprise`**：未来推荐器结合前两层和 session 上下文的预测；当前尚未实现。
>
> 离线出题的北极星仍是：**GT 极稳定 × 盲解-启发式-集成 正确率趋近随机**。一道题若 10 seed 都同一赢家、但没人能纯推理事先叫准方向——这正是“actually run it 才知道”的杀手锏。当前难度 proxy 可用 `tools/difficulty/score_questions.py` **测量**，不是主观断言，也不是已观测到的用户惊讶。

> **惊讶 ≠ 答错 ≠ 点赞**。答错可能是疏忽，惊讶可能是困惑，点赞/继续作答是另一种产品信号。三者必须分开采集和报告，不得相互推断。

---

## 0. 三层产物与三个命令

出题是一条三阶段流水线，每阶段一个 CLI 命令，产物逐层包含：

```
profile → pools → 采样
   │
   ├─(1) create-dataset ────► dataset instance   data/datasets/{family}/{dataset_id}/
   │                             dataset_spec.json + synthesize.py + train.pt/test.pt(+额外文件)
   │
   ├─(2) generate-candidates ─► candidate set     .../candidates/set_{budget}_{m}_{o}_{l}_{hash}/
   │                             每个 c_{hash}/: candidate_spec.json + model/loss/optimizer/train.py
   │                             + results/summary.json + results/curves.npz   ← 真实执行得到的 GT
   │
   └─(3) generate-question ───► question run       .../questions/run_{n}q_{c}c_{hash}/
                                 每个 q_{hash}/: question.json + prompt.txt
```

**核心不变量**（详见 AGENTS.md §1）：GT 永远来自**执行生成的代码**，不是平行逻辑。`spec JSON → 渲染 .py → import & run → metrics`。出题时只从**已存的 GT**里挑选、排序、洗牌，绝不重算指标。

**关键区分**：
- **target**：`synthesize.py` 里合成出的数学函数 / 概率过程（题面里会展示它的代码）。
- **GT（ground truth）**：真实执行 candidate 的 `train.py`、跑 `n_seeds` 个种子后的测试指标排名。**GT 不是符号回归求出来的公式**——它是训练结果。

---

## 1. Stage 1 — 创建数据集实例 `create-dataset`

**入口**：`architecture_iq.datasets.create_dataset`；每个 family 的 `DatasetFamily.create_instance()`。

### 命令

```bash
AIQ=".venv/bin/architecture-iq"

# 一元回归（symbolic expression of x on [0,1]）
$AIQ create-dataset --family univariate_regression --seed 71

# 多元回归（指定输入维度 n，n ∈ profile 的 input_dims 池 = [2,3,4,5,8]）
$AIQ create-dataset --family multivariate_regression --seed 81 --input-dim 4

# bigram 语言模型（一阶 Markov，vocab=32, context=16）
$AIQ create-dataset --family bigram_lm --seed 91

# 交互式（逐项询问 family 与 seed）
$AIQ create-dataset -i

# 从 profile 池里随机挑 family
$AIQ create-dataset --random-family --seed 5
```

### 旋钮

| 旋钮 | 作用 | 备注 |
|------|------|------|
| `--family` | 选择数据集 family | 必须在 `profiles/v1.yaml → pools.dataset_families` 中 |
| `--seed` | 实例种子（默认 0） | 内部派生所有子种子流；**同 family+同 seed(+同 input-dim) → 同实例**（内容寻址 id） |
| `--input-dim` | 仅 multivariate：输入维度 n | 必须 ∈ `dataset_configs.multivariate_regression.input_dims` |
| `--random-family` | 随机挑 family | 与 `--family` 互斥 |
| `-i` | 交互式 | 不能与其它参数混用 |

### 产物

`data/datasets/{family}/{dataset_id}/`：
- `dataset_spec.json`——冻结的合成参数 + `selection_metric` + 内容寻址 `dataset_id`
- `synthesize.py`——把冻结参数嵌进模板；**执行它**得到数据张量
- `train.pt` / `test.pt`——物化的训练/测试张量（bigram 另有 `transition.npz`）

> **控制 target 质量的地方就在这里。** 当前一元 sampler（`families/univariate_regression/sampler.py`）只检查 AST 里有没有非线性节点，不检查该节点是否真依赖 `x`、是否发生抵消，也没有曲率/频谱/有效复杂度约束。所以会出现 `tanh(2*cos(2π*-1.5)) + x`（其实就是 `x + 常数`，线性 R²=1）这类退化 target。**出题前应人工或用脚本扫一遍新实例的 target**（见 §6 质量清单）。

### 各 family 当前生成方式（实质）

| Family | 生成方式 | 实质 | selection_metric |
|--------|----------|------|------------------|
| `univariate_regression` | `[0,1]` 上采样深度 ≤3 的表达式树；256 train / 256 test；无噪声 | 随机符号函数 | `test_mse` |
| `multivariate_regression` | 每维一个非线性项 + 1–2 个交互项；维度从 `[2,3,4,5,8]` 选 | 以 additive 为主的符号函数 | `test_mse` |
| `bigram_lm` | 随机 32×32 转移矩阵，一阶 Markov 采样长度 16 的窗口 | 只依赖前一个 token 的 lookup-table 学习 | `test_ce` |

---

## 2. Stage 2 — 生成候选集 + GT `generate-candidates`

**入口**：`architecture_iq.candidates.sets.generate_candidate_set`。
采样：`candidates/generator.py`（`sample_candidate`, `sample_model/optimizer/loss`）。
写文件：`write_candidate()`。GT：`ground_truth/runner.py → run_ground_truth()`。

### 命令

```bash
DS="data/datasets/multivariate_regression/mvar_9faf9d"

# architecture-only：只变 model（optimizer/loss/batch 固定共享）
$AIQ generate-candidates "$DS" --budget 81920 --count 28 --vary model --seed 10002

# optimizer-only
$AIQ generate-candidates "$DS" --budget 40960 --count 24 --vary optimizer --seed 10006

# loss-only（注意：见下方“可组合上限”）
$AIQ generate-candidates "$DS" --budget 40960 --count 7 --vary loss --seed 10007

# mixed：同时变多个轴
$AIQ generate-candidates "$DS" --budget 81920 --count 28 --vary model --vary optimizer --vary loss --seed 10009

# 交互式（可手工固定 optimizer / loss / batch_size 的具体值）
$AIQ generate-candidates -i
```

### 旋钮

| 旋钮 | 作用 | 备注 |
|------|------|------|
| `--budget` | `total_samples_seen`（= steps × batch_size） | 决定 set 内所有候选共享的训练预算；`batch_size` 仍可在网格内变 |
| `--count` | 生成候选数 | 见下“可组合上限” |
| `--vary {model,optimizer,loss}` | 哪些轴变化（可重复） | 决定 set 文件夹名里的 `var`/`fix` 与将来的题型 |
| `--seed` | 采样种子 | 决定采样到哪些具体配置 |
| `-i` | 交互式 | 可 pin optimizer/loss/batch 的确切值；非交互模式这些 invariant 轴会随机固定 |

### 生成的采样池（`profiles/v1.yaml`）

- **model_types**：`mlp`（回归）、`transformer_lm`（bigram）——每个 family 通过 `compatible_model_types()` 只用其一。
  - MLP 网格：depth `[1..6]`、width `[16,32,64,128,256]`、residual `[F,T]`、activation `[relu,leaky_relu,gelu,silu]`。
  - transformer_lm 网格：d_model `[32,64,128]`、num_layers `[1,2,3]`、num_heads `[2,4]`、d_ff `[64,128,256]`。
- **optimizers**：`SGD, Adam, AdamW, RMSprop, Adagrad`；lr `[1e-4..1e-2]`、weight_decay `[0,1e-5,1e-4,1e-3]`、SGD momentum `[0,0.9]`、batch_size `[16,32,64]`。
- **losses**（per family）：回归 = `mse, mse_l1, mse_l2`；bigram = `cross_entropy, cross_entropy_l1, cross_entropy_l2`。带 `_l1/_l2` 的会额外采样 `lambda ∈ [1e-4,1e-3,1e-2]`。

### ⚠ 可组合上限（`--count` 不能超过某轴的唯一配置数）

`sample_candidate_set_pool` 用 `count × 20` 次尝试去凑**唯一**候选，凑不够就 `RuntimeError: Could not sample N unique candidates`。当只变一个低基数轴时会撞上限：

| 只变 | 唯一配置数上限 | 建议 count |
|------|----------------|-----------|
| `loss`（回归或 bigram） | **7** = `mse` + `{l1,l2}×3 λ` | ≤ 7 |
| `optimizer` | 很大（5 类型 × lr × wd × …） | 可 20–40 |
| `model`（MLP） | 很大（6×5×2×4…） | 可 20–40 |
| `mixed` | 各轴乘积，很大 | 可 20–40 |

> loss-only 题因此天然只能有 ≤7 个 choice 池——够出 2–4 选题，但一个 set 里能通过显著性的**不同**子集有限。

### GT 执行（`run_ground_truth`）

对每个候选：
1. `_sync_candidate_files()` 从 `candidate_spec.json` 重渲染 4 个 `.py`（保证代码=spec）。
2. 跑 `n_seeds=10` 个种子（seed 0..9），每步在**整个** test split 上评估，得到 learning curve。
3. 写 `summary.json`：`mean_{metric}`、`std_{metric}`、每 seed 结果、`failed_seeds`、`excluded`。
4. 写 `curves.npz`：逐步指标。

**失败语义**：单 seed 若 loss/指标非有限或 final > `fail_threshold` 则标记 `failed`。`failed_seeds ≥ max_failed_seeds(=3)` → 整个候选 `excluded=true`。

> **本仓库已加固**（本分支）：`eligible_candidate_paths` 现在排除**任何 `failed_seeds > 0`** 的候选（不只是 `excluded`）。这样避免“正确答案有 seed 发散到 1e31、丢弃后只平均剩余 9 个仍判赢”与题面“10 seeds mean”矛盾的旧 bug。

### 并行大批量生成（本仓库工具）

单个候选（tiny 模型）在**单线程**下最快：多线程反而变慢。因此用**多进程、每进程 1 线程**吃满所有核：

```bash
# tools/batch_generate/parallel_sets.py 读取一个 JSON plan，
# 先顺序写好所有 set 的骨架（仅采样+写 .py），再把 GT 循环 fan-out 到进程池。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python tools/batch_generate/parallel_sets.py \
  --plan tools/batch_generate/batch_plan.json --workers 9
```

plan 格式见 `tools/batch_generate/batch_plan.json`：每个 set 一项 `{label, dataset_path, budget, count, vary[], seed}`。该工具**不重实现任何生成逻辑**，只并行化 `generate_candidate_set` 里顺序跑的 GT 循环，复用 `sample_candidate_set_pool / write_candidate / run_ground_truth`。

---

## 3. Stage 3 — 组装题目 + 渲染 prompt `generate-question`

**入口**：`architecture_iq.questions.generator.generate_questions`。prompt：`prompts/renderer.py`。

### 命令

```bash
DS="data/datasets/multivariate_regression/mvar_9faf9d"
SET="$DS/candidates/set_81920_var_fix_fix_XXXXXX"   # 用实际路径

# 从单个 set 出题（同预算）
$AIQ generate-question "$DS" "$SET" --num-questions 6 --num-choices 4 --seed 9012

# 跨预算题：传多个不同 --budget 的 set
$AIQ generate-question "$DS" "$SET_82k" "$SET_41k" --num-questions 6 --num-choices 4 --seed 9013

# 交互式
$AIQ generate-question -i
```

### 旋钮

| 旋钮 | 作用 |
|------|------|
| `--num-questions` | 要生成的题数（需存在这么多**不同**的通过显著性的子集，否则报错） |
| `--num-choices` | 每题 choice 数（≥2，默认 `profile.num_choices`） |
| `--seed` | 决定子集洗牌与 choice 字母洗牌 |

### 组装逻辑（`generate_questions`）

1. `load_candidate_pool_from_sets`：并集多个 set 的候选，过滤掉 `excluded` 或 `failed_seeds>0`。
2. `find_significant_subsets`：找出所有/若干个通过**显著性**的 `num_choices` 子集。
   - 组合数 ≤ `max_exhaustive_combinations`(500k) 时**穷举**，否则随机采样 `max_attempts` 次。
   - 每个子集先过 `choices_compatible`（必须有轴变化；单轴题要求 batch_size 不变），再过 `validate_significance`。
3. `_pick_distinct_subsets`：按候选集合去重，选出 `num_questions` 个不同子集。
4. 每个子集 → `build_question_record`：
   - 推断 `type`（`infer_question_type`）与 `invariant/varying_axes`。
   - 校验每个候选 `steps × batch_size == total_samples_seen`。
   - `correct_letter` 指向 GT 赢家（`selection_metric` 最优者），然后**洗牌 choice 顺序**（赢家位置随机）。
5. 写 `question.json` + `run.json`；`write_prompt` 渲染 `prompt.txt`。

### 惊讶值选题接口（Inspector `Next` 已接线，出题排序尚未接线）

当前 `find_significant_subsets` 将通过的子集洗牌，`_pick_distinct_subsets` 再取前 `num_questions` 个；这是可复现的随机策展，**不是质量排序**。Inspector 已用 `surprise_catalog.py` + `surprise_recommender.py` 把 manifest-only 冷启动 Beta 和 ε-greedy 接入 `Next`，但仍未改变离线出题器。自然的出题扩展点仍是在两者之间增加纯读取的 `rank_significant_subsets`/stratified sampler：

1. `choices_compatible` + `validate_significance` 仍是不可降级的**有效性硬门**。反馈再高也不能救回不显著、有 failed seed 或未确认的题。
2. 将 `tools/difficulty/score_questions.py` 的读取逻辑抽成可复用的 subset scorer，先修复 metric direction、并列候选、文件/字母顺序偏置，再产生版本化 `intrinsic_surprise_proxy`。
3. 预选可按 proxy 排名或分层抽样，同时限制同 dataset instance/family/type 的过度重复；不得为追求高分而修改候选 GT、winner 或 prompt。
4. 用户产生的 `observed_surprise_rate` 和策略产生的 `predicted_personal_surprise` 属于 release 外的动态数据。它们只能决定已审核题目的展示顺序，**不写入 `question.json`、GT、candidate summary 或 `prompt.txt`**。

### 显著性判据（`significance/validator.py`，来自 profile）

一个子集要成题，赢家必须同时满足：
- **gap**：`|runner_up.mean − winner.mean| ≥ gap_min`（默认 0.05）。
- **win_rate**：赢家在逐 seed 比较中夺冠的比例 `≥ win_rate_min`（默认 0.7）。
- **non-overlap**（可选，默认开）：`winner.mean + winner.std < runner_up.mean − runner_up.std`。
- 池里不能有 `excluded` 候选，mean 必须有限。

### 题型（`type`）如何决定

由 choice 间**实际变化的轴**推断（`infer_question_type`）：

| varying（限 model/opt/loss） | type |
|------------------------------|------|
| 只 `model` | `architecture_only` |
| 只 `optimizer` | `optimizer_only` |
| 只 `loss` | `loss_only` |
| 其它/多个 | `mixed` |

> 注意：题型看的是**子集内实际变化**，不是 set 的 `--vary`。一个 `--vary model` 的 set 里，若被选中的子集恰好 batch_size 也不同，会被 `choices_compatible` 挡掉（单轴题要求 batch_size 不变）。

### prompt 内容（`prompts/renderer.py`）

`prompt.txt` 逐段拼装：header → Dataset（family 模板 + `synthesize.py` 的 `target`/`synthesize` 摘录）→ 训练协议 → Sample budget（共享或 per-choice）→ Evaluation metric → 每个 Choice 的 model/optimizer/loss 自然语言 + **on-disk 代码摘录**。

**绝不包含**：最终 metric、learning curve、seed 统计——只给结构信息，逼模型用 architecture reasoning 作答。渲染前会 `_sync_candidate_files()` 保证摘录的代码=被执行的代码。

---

## 4. 预算与公平对比契约

| 规则 | 强制点 |
|------|--------|
| 所有 choice 同一份物化数据 | 一题一个 `dataset_id` |
| 每 choice 预算显式 | `training_steps × batch_size == total_samples_seen`，写进 spec 和 prompt |
| 单 set → 共享预算 | 一个 set 内所有候选同 `total_samples_seen` |
| 跨 set → 允许跨预算 | 并集多个不同 `--budget` 的 set；`_budget_field` 标 `budget.mixed`，prompt 给 per-choice schedule |
| 按 `selection_metric` 排名 | 显著性 validator + `correct_letter` |
| 无指标泄漏 | renderer 排除 GT |
| 可复现 | 内容寻址 id + GT 记录环境元数据 |

---

## 5. 当前题库盘点（截至本批次）

### 已有 family（3 个注册插件）
`univariate_regression`、`multivariate_regression`、`bigram_lm`（见 `registry.py`）。这是 **3 个 family，不是 3 道题**。

### 生成每种题型的最短路径

| 想要 | 命令要点 |
|------|----------|
| architecture-only | set `--vary model`；题自然是 `architecture_only` |
| optimizer-only | set `--vary optimizer` |
| loss-only | set `--vary loss`（count ≤ 7） |
| mixed | set `--vary model --vary optimizer --vary loss` |
| 跨预算 | 生成多个不同 `--budget` 的 set，`generate-question` 时一起传入 |

### 本 session 新增题库（2026-07，大预算批次）

在 8 个**全新实例**上生成了 **42 道新题**（每实例限量，避免过拟合固定 test）：

| 题型 | 数量 | 实例 / 预算 |
|------|------|-------------|
| `architecture_only` | 20 | sym_c804cc(82k)、mvar_9faf9d d4(82k)、mvar_978f4c d8(82k)、bg_0fff4b(82k)——各 5 题 |
| `optimizer_only` | 10 | sym_316367(41k)、mvar_befdab d5(41k)——各 5 题 |
| `mixed` | 12 | mvar_e3e90e(41k, 5c)×8、sym_411dbf x³(82k, 3c)×4 |

批次命令记录在 `tools/batch_generate/batch_plan.json`；GT 状态在 `tools/batch_generate/_batch1_index.json`（226 候选：222 ok / 4 excluded / 0 failed-seed）。

### 关键经验：预算越大越能测归纳偏置，但过简 target 会“全部收敛”

本批次量化验证了两条：

1. **大预算显著削弱 capacity shortcut**。旧主集 60 题“永远选参数量最大”命中 **66.7%**；本批次 arch-only 题 max-param 命中降到 **35.0%**（deepest 40%、widest 25%），全题型综合 **23.8%**。原因：更大预算让结构合适的小网追平大网，从而奖励 architecture reasoning 而非单纯容量。

2. **loss-only 在当前 target 上无法出题**。mse / mse_l1 / mse_l2（小 λ）对 test 指标的影响 **gap ≈ 0.00000**——最好与最差候选 test_mse 差 < 2e-5，通不过 `gap_min=0.05`。这是真实 benchmark 结论：**在这些 target 上，loss 选择基本不影响泛化**。要出 loss 题，需要能放大 loss 差异的 target（如带 outlier 的 `contaminated_regression` 配 MSE/Huber/MAE，见 §7）。

3. **过简 target 的 mixed 题会顶部打平**。x³ 这类目标在 82k 预算下多数候选收敛到 test_mse≈0（28 个里 19 个 <0.001），5-choice 找不到干净的 #1-vs-#2 gap；降到 3-choice（好/中/差各一）才成题。**启示**：mixed/arch 题要么用结构更丰富的 target（多交互项，如 mvar_e3e90e），要么故意保留可区分的难度梯度。

---

### Subagent 盲评结果（13 道代表题，2026-07）

用 subagent 对 13 道代表题做了**盲解 + 质量评审**（脚本见 `tools/batch_generate/`，结果 `_eval_results.json`）：每题一个 solver 只看 prompt（无指标）作答，再一个 judge 对照正确答案评质量。

| 指标 | 结果 | 解读 |
|------|------|------|
| 盲解正确率 | **7/13 (54%)** | 高于随机（4-choice=25%），但远非满分——题有真实难度，不是送分 |
| 平均质量分 | **3.77 / 5** | judge 认为整体“良好偏上” |
| determinable | **12 yes / 1 partly / 0 no** | 几乎所有题的答案**可由 architecture reasoning 推出**，不是靠噪声/运气——直接回应了“会不会像符号回归只有平凡或混沌”的担忧 |
| high shortcut-risk | **4/13** | 主要集中在 optimizer 题（见下） |

按题型：
- **architecture_only（8 题）**：avg quality **4.0/5**，8/8 determinable，仅 1 题 high-shortcut。盲 solver 只 4/8 对——多次掉进“选最宽/最大网”的陷阱而正确答案是结构更合适的中等网（judge 明确称赞这些是“defeats the biggest-net heuristic”的好题）。**这是本批次质量最高的一类**，正中 ArchitectureIQ 靶心。
- **optimizer_only（2 题）**：solver 2/2 全对，但 avg quality **3.0/5**、2/2 high-shortcut——因为正确答案总是唯一的 Adam / 唯一调好 lr 的那个，“永远选 Adam / 选最大 lr”就能蒙对。**改进方向**：加入调坏的 Adam 或调好的 SGD/Adagrad，逼真正的 optimizer-dynamics 推理。
- **mixed（3 题）**：avg quality **3.7/5**。难 target（mvar_e3e90e，多交互项）的两道 5-choice 是好题（solver 被“选 Adam”骗错，正确答案靠 residual+depth 的组合能力）；易 target（x³）那道靠 optimizer 区分，shortcut 中等。

**结论**：architecture_only + 大预算是本 pipeline 目前**最能测出 architecture intuition** 的题型；optimizer_only 需要更对抗的 distractor 设计；loss_only 在现有 target 上无法成题。judge 的逐题 critique 是设计下一批 distractor 的现成素材。

---

## 6. 出题质量清单（每次出新题前过一遍）

这些是审计出的真实弱点，出题前务必检查：

1. **target 有效复杂度**：扫新实例的 `target`——是否退化成线性/常数（线性 R² 接近 1）？是否 `sin(sin(sin(x)))` 这类失控？理想是**有明确 architecture hypothesis** 的结构。
2. **capacity shortcut**：出完题后，跑一遍“永远选参数量最大/最深/最宽/最大 FLOPs”的 baseline。若某 shortcut 命中率远高于随机基线，说明题在奖励容量而非 architecture reasoning——换更大预算（让小网追平）或让候选参数量匹配。
3. **failed-seed 语义**：确认没有 `failed_seeds>0` 的候选进正确答案（本分支已在 `eligible_candidate_paths` 加固）。
4. **单轴纯净**：architecture-only 题里 optimizer/loss/batch_size 必须真的不变（`choices_compatible` 会挡，但要确认 set 是 `--vary model` 生成的）。
5. **答案分布**：检查 `correct_letter` 在题库里的分布，避免位置偏置（洗牌应已保证，但要验）。
6. **预算够大**：小预算题容易被“容量=赢”主导；大预算更能测归纳偏置与 sample efficiency。
7. **每实例限量**：同一实例最多贡献 1–3 题，避免过拟合固定 test split（当前 test split 同时用于 learning curve、候选比较、子集筛选，实为 validation set）。
8. **离线 proxy 只是冷启动**：检查 `intrinsic_surprise_proxy` 的 metric direction、tie 处理和 blind-solver 来源；不把它称为用户 `observed_surprise_rate`。
9. **发布边界**：本地 gitignored `data/` 盘点的 261 题不等于当前 attested 60 题 release。公开/online 选题只能从 runtime manifest 验证过的当前 release 中返回。
10. **动态分数不泄漏**：`observed_surprise_rate`/`predicted_personal_surprise` 不进入 `question.json`、GT 或 prompt；答题前 UI 也不显示 winner、hardness 或惊讶分。

---

## 7. 更符合 ArchitectureIQ 的题目方向（roadmap）

核心不是“把公式弄复杂”，而是：**生成机制有明确、可控、可见的结构；不同 architecture 对该结构有不同归纳偏置；答案仍由 generated-code GT 稳定确认。**

| 优先级 | 建议 Family | 测什么 | 候选 architecture |
|--------|-------------|--------|-------------------|
| P0 | `spectral_regression` | 低频/高频表示、spectral bias | ReLU MLP、SIREN、Fourier-feature MLP |
| P0 | `structured_memory_lm` | receptive field、循环记忆、全局注意力 | TCN、GRU、local/global Transformer |
| P1 | `translated_motifs` | locality、weight sharing、平移等变 | CNN、flat MLP、small ViT |
| P1 | `anova_interaction_regression` | additive vs 高阶交互 | NAM、CrossNet、普通 MLP |
| P2 | `set_relations` | permutation invariance、关系建模 | DeepSets、Set Transformer、flat MLP |
| 独立 mechanics | `conditioned_/contaminated_regression` | optimizer conditioning、outlier 下 loss 选择 | 固定模型，只变 optimizer 或 MSE/Huber/MAE |

**典型题范例**（表达能力档）：序列长 64，`y_t = XOR(x_{t-3}, x_{t-24})`；A=causal TCN receptive field 9、B=TCN receptive field 33、C=local-attention window 16。hypothesis 很清楚：A、C 看不到 lag 24，B 能。更难版本让三者都可表达，再用有限数据+预算测 sample efficiency。

**难度三档**：
1. **表达能力**：某些 architecture 结构上看不到所需信息。
2. **归纳偏置**：都能表示，但参数量匹配、数据有限。
3. **学习动力学**：短/长预算下 winner crossover，单独计分为 learning mechanics。

---

## 7.5 难度工程：label noise、v2 profile、难度过滤器

这一节记录如何把题目从“中等难度”推向“专家也叫不准”，以及配套工具。核心洞察：**当前无噪声光滑 target 封顶了难度**——无噪声下容量够大+训得够久 test error 就 →0，于是 bias-variance tradeoff 不存在，正则化、Adam-vs-SGD 泛化差异、double descent 这些真前沿机制全部失效，只剩“容量”和“收敛速度”可考，而这两者最容易被启发式蒙对。

### 7.5.1 Label noise（最大杠杆）

给**训练标签**加高斯噪声，**测试集保持=真函数**。一步解锁多个机制：double descent（容量非单调）、Adam-vs-SGD 泛化 gap、正则化价值、并救活 loss-only、干掉“选 Adam”捷径。

```bash
# 只回归族支持；test 永远是真函数，train 标签带噪
architecture-iq create-dataset --profile v2 --family multivariate_regression \
  --seed 311 --input-dim 4 --noise-std 0.15
architecture-iq create-dataset --profile v2 --family univariate_regression \
  --seed 301 --noise-std 0.25
```

实现要点（遵守 spec→code→run→GT 不变量）：
- `--noise-std` 冻结进 `params.noise = {enabled, type:"gaussian_label", std, seed, applies_to:"train_only"}`，进内容寻址 id ⇒ **带噪实例 id ≠ 干净实例 id**（同 seed 同表达式也不同）。
- `synthesize.py` 模板里只对 `train_y` 加噪（`label_noise_seed` 派生自 instance_seed），`test_y` 保持 `target(test_x)`。噪声代码可见 ⇒ 自动进 prompt 摘录。
- prompt 诚实：dataset `.md` 不再硬说 “noiseless”；`format_regression_protocol` 动态输出 “Training labels carry additive Gaussian noise (std=…); the held-out test labels are the exact target function”。
- GT 路径**完全不变**——只 `load_tensors` 读已物化张量。
- 回归测试：`tests/test_new_families.py::test_label_noise_train_only_and_reproducible`（验证 train 有噪、test=真函数、id 随噪变化）。

### 7.5.2 v2 profile（edge-of-stability / 更宽网格）

`profiles/v2.yaml` = v1 + **上探的 lr 网格**（`… 1e-2, 3e-2, 1e-1`）+ 更宽 weight_decay/lambda + 更大 budget（到 163840）。用于 optimizer 题探 edge-of-stability，以及 noise 批次。**v1 保持不动**，保证已生成 v1 artifacts 可复现。

**经验教训**：architecture-only / loss-only / mixed 用 **v1 的安全 lr**（这些题里 lr 是固定共享的，若固定到 v2 的高 lr，所有深网会一起发散、大量 excluded）。**只有 optimizer-only 用 v2**（edge-of-stability 正是要点）。批量生成时按此拆分 profile。

### 7.5.3 难度过滤器（`intrinsic_surprise_proxy` 雏形）

`tools/difficulty/score_questions.py`（零训练，纯读 artifacts）对每题打三个正交分：
1. **validity**：win_rate、gap、非重叠。
2. **anti-heuristic**：赢家是否**违背**“选最大参数/最深/最宽/最激进 optimizer”——违背越多越难（只统计该轴真有区分度的启发式）。
3. **blind difficulty proxy**：这些启发式的**多数投票**是否会选错（`ensemble_heuristic_wrong`）。

```bash
python tools/difficulty/score_questions.py --top 25   # 输出 _scores.json + 最难题排行
```

用它重挖已有候选、榨出最难题，并作为“够不够难”的**客观门槛**（而非主观断言）。最终应配“专家惊讶过滤器”：只保留 **GT 稳定 + 赢家反直觉 + 盲解集成会错** 三者同时成立的题。

这个工具当前使用少量手工启发式和手工组合权重，因此严格名称应是 `intrinsic_surprise_proxy`，而不是“用户惊讶值”。它不能从用户答错推断 surprise，也不能从高 proxy 推断 like/继续作答。进入自动策展前，必须先修复高分更好/低分更好的 metric direction，并用 tie set 代替任意 `argmax` 先到先得。

### 7.5.4 本 session 噪声批次的实测发现

在 5 个带噪实例上生成 **254 候选 → 34 道新题**（arch 12 / mixed 15 / optimizer 7），加上难度过滤器覆盖全库 **261 题、130 道击败启发式集成**。关键实证：

| 机制 | 结果 | 结论 |
|------|------|------|
| **double descent / 反容量** | 3/3 带噪 arch 集里，**赢家都不是最大网**；多元集 test_mse 随容量非单调（如 mvar_866b4e：d5 w256 赢过 d6 w256） | 大预算+噪声成功杀死 capacity 捷径 |
| **Adam-vs-SGD 泛化** | `uni_d6bbf5` 带噪 optimizer 集：**Adagrad/SGD 占据前 4，Adam/AdamW 掉到第 9-14 名** | “永远选 Adam”在这里给出**接近最差**的答案——专家级陷阱 |
| **正则化价值（loss-only）** | 带噪后 `mse`（无正则）变**最差**，正则 loss 更好——**方向对了**；但 gap 仅 0.017 < gap_min 0.05 | 信号复活但**未过阈**；需更强噪声 + 更难 target，或对 loss 题单设更低 gap_min |
| **target 难度依赖** | 一元带噪题几乎全部无法成题（target 太易，含噪仍 ~0.002-0.006 打平）；多元带噪题成题良好 | **噪声只在够难/够高维的 target 上生效**；出难题优先多元、高维、多交互 |

**下一步（提高难度产出率）**：(a) loss 题需更大 noise-std（0.3-0.5）+ 高维 target 才能让正则化过阈；(b) SGD-beats-Adam 现象在一元上 gap 太小，应搬到多元；(c) 可为 double descent 单独做“定宽扫描”set（固定除 width 外全部，密集扫 width），配噪声看非单调峰。

### 7.5.5 盲评实证：噪声题真的更难

对 10 道最难带噪题做 **3 独立 solver 盲解 + judge**（`tools/batch_generate/_eval2_noisy_hardness.json`）：

| 指标 | 干净批次 | 带噪批次 | 结论 |
|------|----------|----------|------|
| per-solver 正确率 | 54% | **46%**（随机 25%） | 噪声让真 reasoning agent 更容易错 |
| architecture-only solver 正确 | 4/8 | **0/9** | 每个专家在所有带噪 arch 题上都被骗 |
| “有捷径且捷径给出错误答案”的题 | 部分 | **10/10** | judge 逐题确认 solver 正是掉进这些捷径 |

被证伪的专家捷径（judge 原话）：“选最小网抗噪”→错、“选最大/最有表达力网”→错、“选最正则化”→错、“选自适应 optimizer/最高 lr”→错。这正是用户要的**“连专家都无法可靠事先预测、必须真跑才知道”**的机制：GT 稳定可复现，但纯推理叫不准方向。注意 judge determinable 只 4/10 = yes——说明这些题**难到有后见之明也未必能一眼判定**，难度确实上了一个台阶。

## 7.6 用户惊讶与推荐闭环（reaction/策略核心本地已实现，服务闭环待接）

### 反馈与曝光数据

- 本地 Inspector 已在用户 commit answer 并 reveal 真实排名后，用结构化 `question_reaction_submitted` 记录 `{reaction: "surprise", value: true|false, timing: "after_reveal"}`；skip 不产生评价。事件绑定 session/attempt/question/version/可选 release，使用稳定 event ID，并复用 append-only outbox、幂等上传和冲突隔离。Edge 与 `19000` migration 已实现，仍待 hosted 部署验收。
- surprise 与 like 使用独立 reaction 类型/统计；answer correctness 只能作为分析上下文，不能自动充当任何 reaction。
- 如要评估“是否更愿意做下去”或对推荐策略做无偏比较，必须另记 `question_presented`，包含 policy version、decision ID、source/position 和 selection probability/propensity。只看 answer 或 reaction 会丢失未作答曝光分母。

### 统计与策略

1. 对每个 `(release_id, question_id, question_version)` 按唯一 attempt/reaction 统计 yes/no，用 family/type 层级的 Beta-Binomial 先验得到平滑后验 `observed_surprise_rate`；保留 raw duplicate/conflict 用于质量审计。
2. `surprise_catalog.py` 与 `surprise_recommender.py` 已实现 manifest-only、tie-aware 冷启动、特征缺失重归一化、Beta 后验及默认 20% exploration，并接入本地 `Next`。`question_presented` 同步记录 policy/decision/mode/source/position/精确 mixture propensity。后续再接 hosted 聚合与版本化 Thompson/contextual serving，生成真正的 `predicted_personal_surprise`。
3. 有效性始终是硬门；策略只从当前 runtime-attested release 的未作答题中选择，保留 15%–20% exploration 与 family/type/dataset 多样性，服务不可用时回退已有顺序 `Next`。
4. A/B 主指标可以是预先声明的 `observed_surprise_rate`，但 answer/continue rate、accuracy、family/type 覆盖和独立 like 必须作 guardrail。无充足样本、区间/后验证据或 propensity 日志时，不宣称策略提升。

### 产物边界

当前 `docs/STATUS_AND_PLAN.md` 的 261 题是本地 `data/` 盘点，而当前 attested bundle 是 60 题。只有后者能进入现行线上候选池；新题必须先通过 canonical publisher/manifest/registry 发布。动态 surprise 聚合和策略状态应保存在外部持久化服务或与 release 绑定的独立 policy snapshot，不得回写 canonical 题目、candidate 或 GT artifact。

---

## 8. 常见错误（出题相关）

1. 手改 `question.json`——显著性和字母洗牌是自动的，改了会不一致；应重跑 CLI。
2. loss-only set `--count` > 7 → `Could not sample N unique candidates`。
3. 只重跑 `generate-question`（不新建 candidate set）——只是重新组合 choices/letters，不产生新训练结果或新知识点。
4. 假设 `data/` 在 git 里——它是 gitignored；测试用 fixture 或 temp dir。
5. 跨预算题前未确认 `budget.mixed` 与 per-choice schedule 正确渲染。

---

*生成/更新此文档时，请同步核对 `profiles/v1.yaml`（池与网格）、`registry.py`（family 清单）、`significance/validator.py`（判据）是否与正文一致——本文只描述稳定逻辑，具体池内容以这些源为准。*
