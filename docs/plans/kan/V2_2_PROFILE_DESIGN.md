# V2.2 KAN Profile Design

状态：已实现 v2.2 profile，尚未开始大规模训练；pilot metadata 不改变 frozen Gate 语义。

本文记录 v2.2 profile 的候选语义和审计门槛。当前 v2.yaml 与 v2.1.yaml 不修改；v2 hash 为 3993a8aef680d37c，v2.1 hash 为 525167b7afdb6bf8，v2.2 hash 为 cdc6e6d564b13cc6。v2.2 的 KanModelFamily.sample_spec() 已支持按 input_dim 从显式 archetypes 选择。

## 1. 目标和边界

v2.2 只扩展 synthetic tabular classification 的 KAN architecture pool：

- 保留 efficient_spline_v1、output_dim=2、cross_entropy、batch_size=32 和 total_samples_seen=8192；
- 保留 v2.1 的 profile-aware gate：classification 允许 mlp 与 kan，其它旧 profile 语义不改变；
- 每个 input_dim（4、8、16）显式提供 12 个 KAN architecture specs，共 36 个可审计 specs；
- architecture-only 题目必须在同一个 dataset、budget、optimizer、loss 下比较候选；
- 不把 KAN 类型或参数量变成唯一的答案线索。

## 2. 实现状态与冻结边界

当前 registry/validator 可以验证 KAN spec；v2.2 sampler 在存在 archetypes 时按 input_dim 选择：

    profile.kan.depth × width × grid_size × spline_order × base_activation

v2.2 已加入按 input_dim 绑定的 architecture list；没有对应 archetype 的旧 profile/family 继续使用原有标量 pool。显式 pair 仍需通过 calibration 报告审计，不能仅凭 profile 视为已验证。

安全顺序应是：

1. 显式 classification-KAN architecture sampler 已实现；
2. 用下面的 36 个 specs 做离线 validator 和 GT smoke；
3. profiles/v2.2.yaml 已建立，当前 hash 为 cdc6e6d564b13cc6；
4. 不回写 v2/v2.1 candidate、question 或 collection artifacts。

## 3. 推荐的 12 个 architecture archetypes

每个 archetype 表示 (depth, width, grid_size, spline_order, base_activation)。input_dim 和 output_dim=2 由 dataset/profile 注入。

| id | depth | width | grid | order | activation | matched partner |
|---|---:|---:|---:|---:|---|---|
| A1 | 1 | 4 | 5 | 2 | silu | A2 |
| A2 | 2 | 4 | 3 | 2 | relu | A1 |
| A3 | 1 | 4 | 5 | 3 | silu | A4 |
| A4 | 2 | 4 | 3 | 3 | gelu | A3 |
| A5 | 1 | 8 | 3 | 2 | tanh | A6 |
| A6 | 3 | 4 | 7 | 2 | silu | A5 |
| A7 | 1 | 8 | 5 | 3 | relu | A8 |
| A8 | 2 | 8 | 3 | 2 | gelu | A7 |
| A9 | 1 | 12 | 3 | 2 | silu | A10 |
| A10 | 3 | 8 | 3 | 2 | tanh | A9 |
| A11 | 1 | 16 | 3 | 2 | gelu | A12 |
| A12 | 2 | 12 | 3 | 2 | relu | A11 |

All activations are already accepted by src/architecture_iq/models/kan.py. The six matched pairs deliberately vary depth, width, grid, order, and activation while keeping trainable parameter counts close at the same input dimension. This makes a parameter-only heuristic insufficient as the intended design.

The explicit pool has 12 × 3 = 36 classification KAN specs. The unconstrained envelope would have 3 depths × 4 widths × 3 grids × 2 orders × 4 activations × 3 input_dims = 864 combinations; that larger number is not a release target because it is not stratified or audited.

## 4. Parameter-count calculation

For this KAN implementation, each layer has a base branch plus spline basis branch. Let:

    B = 1 + grid_size + spline_order
    P(I,d,w,g,k) = B × (I×w + d×w² + 2×w)

The final 2×w term is the output layer (output_dim=2); there are no trainable spline biases. This formula agrees with trainable_parameter_count() and KanModelFamily.build_module().

For the 12-spec pool:

| input_dim | min params | max params | mean params |
|---:|---:|---:|---:|
| 4 | 320 | 2160 | 989.3 |
| 8 | 432 | 2496 | 1194.7 |
| 16 | 624 | 3264 | 1605.3 |

Across all 36 specs the range is 320–3264 and the mean is 1263.1. Matched-pair relative parameter differences are at most about 12.8% across the three input dimensions; at input_dim=8 they are 0–3.7%. The pool therefore contains both low/high capacity variation and near-equal-capacity architecture variation.

## 5. Combination and training cost

The formal classification protocol remains:

    train_size=1024
    test_size=2048
    batch_size=32
    total_samples_seen=8192
    training_steps=256 per seed
    n_seeds=10

One candidate therefore performs 2,560 optimizer steps and evaluates 81,920 training rows across all seeds. The 36-spec pool requires 92,160 optimizer steps and about 2.95 million training-row evaluations before any MLP control candidates are added.

A local CPU smoke benchmark in this worktree measured two budget-1024 MLP candidates at about 15.4 seconds serial total for 10 seeds. Scaling only by the 8x sample budget gives roughly 61 seconds per similarly sized candidate at budget 8192. KAN spline cost scales with the layer basis/parameter count, so a practical planning envelope is approximately 1–4 minutes per candidate on this CPU, or roughly 1–2 hours for 36 KAN specs. This is an estimate, not a completion claim; first run 2 seeds on all archetypes, then 10 seeds only for stable pairs.

Recommended execution stages:

1. 36 specs × 2 seeds: validator, render/import, forward/backward, GT smoke;
2. select at least 10 specs per input dimension with finite metrics and no failed seeds;
3. 10-seed calibration for selected specs and matched MLP controls;
4. only then generate new v2.2 diagnostic questions.

## 6. Required validation gates

Before freezing v2.2:

- v2 and v2.1 load with their current hashes;
- v2.2 has hash cdc6e6d564b13cc6 and a freeze note;
- every explicit spec passes KAN model validation, renders, imports, and produces logits of shape (batch, 2);
- every spec records trainable_parameter_count from the model, not a hand-written estimate;
- exactly 12 architecture specs are available for each input dimension 4/8/16;
- six matched pairs are present per input dimension and their parameter ratios are reported;
- candidate sets keep dataset, budget, batch size, optimizer, and loss fixed;
- Gate12 checks profile/hash, family compatibility, finite GT metrics, failed seeds, and summary provenance;
- architecture-only reports include fixed-parameter and faster-runtime baselines, not only a KAN-vs-MLP label;
- v2.2 candidates/questions use new output directories and never overwrite v2/v2.1 artifacts.
- calibration_metadata.protocol_version is v2.2_effect_aware_pilot; its rules are pilot-only and the active Gate still uses frozen gap_min=0.05.

The release should be blocked if the explicit list is ignored by the sampler, if any input dimension has fewer than 10 valid specs, or if all significant question outcomes can be predicted by parameter count alone.
