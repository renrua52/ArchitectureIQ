# V2.2 Gap Calibration Report

状态：校准建议，不修改冻结的 `v1`、`v2` 或 `v2.1` profile。

## 1. 结论先行

当前 question artifacts 不支持继续使用一个跨 metric 的 universal `gap_min=0.05` 作为 V2.2 的完整显著性定义。原始 gap 的单位随 selection metric 和 family 改变：同样的 `0.05` 对 bigram cross-entropy、回归 MSE、分类 cross-entropy 的相对含义不同。

建议 V2.2 保留 raw gap 作为审计字段，但将正式 gate 拆成：

1. family × selection_metric × question_type 的 relative gap；
2. 基于 seed-level final metric 的 pooled standardized effect size；
3. 现有 win-rate；
4. 可选的 seed-delta 分位数鲁棒性门槛。

分位数应描述每题 seed-level winner-vs-runner-up 差值，不应把当前 artifact 样本分位数直接冻结成 profile 常数。当前 evidence 只足以形成 pilot thresholds，不能宣称已经完成 V2.2 冻结校准。

## 2. Evidence scope and method

扫描范围：

- `data/**/questions/**/question.json`
- `outputs/**/question.json`
- 题目引用的 candidate `candidate_spec.json` 与 `results/summary.json`

去重规则：按 `question_id` 去重；`data/` 版本优先于 `outputs/` 镜像。结果为 59 个 unique question IDs、79 个 question paths。profile 分布为：v1 36、v2 18、v2.1 5。

当前可观测组合只有：

| Profile | Family | Metric | Question type | Unique questions |
|---|---|---|---|---:|
| v1 | `bigram_lm` | `test_ce` | `mixed` | 4 |
| v1 | `multivariate_regression` | `test_mse` | `mixed` | 4 |
| v1 | `univariate_regression` | `test_mse` | `mixed` | 28 |
| v2 | `synthetic_tabular_classification` | `test_ce` | `architecture_only` | 18 |
| v2.1 | `synthetic_tabular_classification` | `test_ce` | `architecture_only` | 5 |

没有足够的 optimizer-only 或 loss-only question 样本，不能为这些类型单独估计规则。分类 V2.1 只有 5 题，KAN 相关阈值仍应视为诊断性建议。

## 3. Observed question gap distribution

下面的 quantiles 是已有题目 `question.json.significance.gap` 的 raw absolute mean gap：

| Profile / family / metric / type | n | min | q10 | q25 | median | q75 | q90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 / bigram / `test_ce` / mixed | 4 | 0.0563 | 0.0841 | 0.1257 | 0.1659 | 0.2318 | 0.3199 | 0.3786 |
| v1 / multivariate / `test_mse` / mixed | 4 | 0.1360 | 0.1402 | 0.1465 | 0.1716 | 0.2329 | 0.3042 | 0.3517 |
| v1 / univariate / `test_mse` / mixed | 28 | 0.0536 | 0.0625 | 0.0746 | 0.1242 | 0.2047 | 0.3270 | 0.4513 |
| v2 / classification / `test_ce` / architecture-only | 18 | 0.0598 | 0.0816 | 0.0989 | 0.1630 | 0.3394 | 0.4891 | 0.4937 |
| v2.1 / classification / `test_ce` / architecture-only | 5 | 0.0554 | 0.0592 | 0.0648 | 0.1002 | 0.1053 | 0.1240 | 0.1365 |

所有这些题目都已经通过当前 `gap_min=0.05`，所以这些数据描述的是当前生成 gate 的 accepted tail，而不是完整候选池的自然分布。不能据此把 q10 或 median 当作无偏效果分布。

## 4. Raw, relative, and standardized views

对每题按 selection metric 排序，取 winner 与 runner-up：

- `raw_gap = abs(mean_runner - mean_winner)`
- `relative_gap = raw_gap / max((abs(mean_runner) + abs(mean_winner)) / 2, epsilon)`
- `pooled_effect = raw_gap / sqrt((std_winner^2 + std_runner^2) / 2)`

`std_*` 是跨 seed 的 final metric 标准差，不是标准误；若 V2.2 改用标准误，必须显式记录这一语义变化。

| Group | relative gap: min / median | pooled effect: min / median |
|---|---:|---:|
| v1 / bigram `test_ce` | 0.0175 / 0.0482 | 8.63 / 11.01 |
| v1 / multivariate `test_mse` | 0.1526 / 0.1940 | 3.62 / 4.80 |
| v1 / univariate `test_mse` | 0.2617 / 1.0689 | 1.92 / 2.91 |
| v2 / classification `test_ce` | 0.4568 / 0.7472 | 1.90 / 8.22 |
| v2.1 / classification `test_ce` | 0.2426 / 0.2876 | 2.18 / 2.66 |

这直接说明 universal raw gap 会错配尺度：bigram 的 raw gap 可以很小但 standardized effect 很大；univariate MSE 的 mean scale 跨题变化大，relative gap 比 raw gap 更稳定；V2.1 分类 KAN/MLP 的 relative gap 明显低于 v2 MLP-only 分类，不能复用 v2 的相对阈值。

当前每题 winner 的 seed-delta q25（定义为 runner final metric - winner final metric，越大越好）均为正：

| Group | n | q25 最小值 | q25 中位数 |
|---|---:|---:|---:|
| v1 / bigram | 4 | 0.0523 | 0.1555 |
| v1 / multivariate | 4 | 0.0984 | 0.1520 |
| v1 / univariate | 28 | 0.0234 | 0.1035 |
| v2 / classification | 18 | 0.0359 | 0.1356 |
| v2.1 / classification | 5 | 0.0423 | 0.0816 |

这支持把 seed-delta quantile 作为连续的 robustness diagnostic；但 n=4 或 n=5 的组仍远不足以冻结严格 quantile 常数。

## 5. V2.2 pilot profile shape

下面是建议的 profile 形状，不是本轮直接写入的冻结 profile：

```yaml
significance:
  protocol: v2.2_effect_aware
  default:
    win_rate_min: 0.70
    effect_size_min: 2.0
    seed_delta_quantile:
      quantile: 0.25
      min: 0.0
  rules:
    bigram_lm:
      test_ce:
        mixed:
          relative_gap_min: 0.02
    multivariate_regression:
      test_mse:
        mixed:
          relative_gap_min: 0.15
    univariate_regression:
      test_mse:
        mixed:
          relative_gap_min: 0.25
    synthetic_tabular_classification:
      test_ce:
        architecture_only:
          relative_gap_min: 0.20
```

这些值的含义是“覆盖当前已接受题目、同时停止 universal raw gap 的 pilot 起点”：

- bigram 的 observed relative-gap minimum 为 0.0175，因此 0.02 是接近现有 accepted tail 的轻量起点；
- multivariate 的 minimum 为 0.1526，因此 0.15 只应作为暂定值；
- univariate 的 minimum 为 0.2617，因此 0.25 是保守起点；
- V2.1 classification 的 minimum 为 0.2426，但只有 5 题，0.20 仅用于避免把当前小样本全部拒绝，不能当作已校准下限；
- `effect_size_min=2.0` 与当前 non-overlap heuristic 在两个 std 近似相等时大致对应，但现有 univariate 最小 observed effect 为 1.92，若执行 2.0 会拒绝至少一条现有题，须先作 policy 决策。

建议第一轮 V2.2 校准同时报告 gate 前后保留率，不要仅报告通过题的分布。若必须保持现有题的 accepted set，可先用 `effect_size_min=1.75`，并把 2.0 作为 strict diagnostic gate；这需要新的 calibration evidence 才能决定。

## 6. Existing fields vs required code extensions

当前已有、可直接复用的字段：

| 位置 | 当前字段 | 语义 |
|---|---|---|
| profile significance | `gap_min`, `win_rate_min`, `use_non_overlap` | 当前 universal raw gap、seed win-rate、均值±std 不重叠 |
| `question.json` | `significance.gap`, `win_rate`, `metric`, `passed` | 当前 gate 的记录结果 |
| candidate summary | `mean_<metric>`, `std_<metric>`, `seed_results[*].final_<metric>`, `failed` | 足够重算 relative gap、pooled effect、seed delta |
| validator result | `gap`, `win_rate`, `winner_index`, `reason` | 当前 validator 输出 |

需要代码扩展的字段/行为：

1. `SignificanceResult` 增加 `raw_gap`（或保持 `gap` 作为 raw alias）、`relative_gap`、`effect_size`、`seed_delta_quantile` 和对应 quantile；
2. validator 按 `family + metric + question_type` 读取 profile rule，而非只读一个 `gap_min`；
3. 对 higher-is-better metric 使用方向归一化的 delta，不能在 validator 外假定全部 metric 都是 minimization；
4. `question.json.significance` 写入这些派生值和实际使用的 rule/protocol version，便于审计复算；
5. audit tools 独立重算并比较新字段；历史题缺字段时继续走 legacy compatibility path；
6. tests 覆盖 zero/near-zero denominator、不同 metric 方向、失败 seed、3-choice winner/runner-up、以及 profile 没有 family-specific rule 的 fallback。

`Profile` 当前用 `raw` 保存完整 YAML，新增 unknown significance keys 不会破坏加载；但在 validator 尚未扩展前，这些 keys 只会被忽略，不能宣称规则已经生效。

## 7. Decision on each proposed quantity

- raw gap：保留，作为可读性和回溯字段；不再作为跨 family universal gate。
- relative gap：写入 V2.2 profile，按 family × metric × type 配置；使用对称 mean denominator 和 epsilon。
- effect size：写入 V2.2 profile，建议 pooled across-seed std；默认门槛需在 calibration rerun 后冻结。
- win-rate：保留现有字段与 `0.70` 默认，作为离散 robustness gate。
- quantile：只配置 seed-delta quantile 的 `q` 与最小值；不要把当前 question population 的 quantile 直接写成 profile threshold。

## 8. Evidence and limitations

主要代码证据：

- `src/architecture_iq/significance/validator.py`：当前只计算 raw gap、win-rate、non-overlap，并假定调用方传入 `higher_is_better`；
- `profiles/v1.yaml`、`profiles/v2.yaml`：当前只声明 `gap_min=0.05`、`win_rate_min=0.7`、`use_non_overlap=true`；
- `tests/test_significance.py`：只覆盖 pass 与 raw gap failure；
- `tools/question_audit_lib.py`：当前独立复算的也是 `gap` 与 `win_rate`；
- `outputs/question_expansion_phase/review_collection_24.json`：当前 24-question review collection 的顺序/范围；
- `outputs/demo_release_integration/gate12_*_question_input_audit.json`：包含 V1/V2.1 candidate provenance、summary metric 与 profile hash，但不是完整 calibration report。

限制：当前 outputs 含 data/outputs 镜像，已按 question ID 去重；许多 calibration 工具输出没有统一的 persisted calibration manifest。因此本报告是 artifact-based pilot calibration，不等同于完整候选池的 acceptance-rate study。
