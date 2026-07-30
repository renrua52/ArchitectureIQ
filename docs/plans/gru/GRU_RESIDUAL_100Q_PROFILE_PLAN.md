# GRU 层残差 profile：100 题可审计 review pool 计划

## 目标与边界

建立一个新的 profile（暂定名 `v2.5-gru-residual-architecture-pilot`），在不修改冻结的 `v2.4-gru-architecture-pilot` 或其既有 GT artifact 的前提下，验证默认启用 **逐层 GRU residual** 的架构题。该 run 是供 Inspector 人工审查的 review/blind pool，不是 canonical blind evaluation。

目标产物为一个 100 题的二选一 `architecture_only` 题集：每题均为一条 GRU 轨迹与一条 Transformer 轨迹的比较，且通过独立审计。

## 冻结的比较契约

| 项目 | 新 profile 的规则 |
|---|---|
| 数据族 | `bigram_lm` |
| 可变化项 | 仅完整 model spec（架构） |
| 题型 / choices | `architecture_only` / 2 |
| 优化器 | Adam，`lr=1e-3`、`weight_decay=0`、`betas=(0.9, 0.999)` |
| loss / batch / device | cross-entropy / 32 / CPU；同题固定 |
| 训练预算与 GT | 继承 v2.4 的 5120 samples、10 seeds（0--9）与健康性门槛 |
| GRU 架构语义 | 默认 `layer_residual: true`；每一层为 `h = h + GRU_layer(h)`；无额外 norm、dropout 或投影参数 |
| Transformer | 架构与训练协议不变；可复用已完成且可审计的 v2.4 Transformer GT |
| 复用策略 | `blind_pair_unique`：不同 GRU--Transformer pair；候选可跨题复用，不设每候选次数上限 |
| 题目显著性 | `gap_min=0`；仍要求赢家显著性（见“待确认”） |
| 赢家类型平衡 | 最终 100 题中任一 winner type（GRU 或 Transformer）不得超过 70 题 |

`gap_min=0` 只移除绝对均值差的筛选；它不意味着接受随机或不稳定的赢家。审计依然必须逐题重算并确认所有胜利显著性条件。

## Profile 与模型实现

1. 新增 `profiles/v2.5-gru-residual-architecture-pilot.yaml`，从 v2.4 复制非架构语义，并设：
   - `gru_lm.layer_residual: true`；
   - `significance.gap_min: 0`；
   - 其他 profile 参数不因本次工作而隐式变化。
2. 在 `gru_lm` model spec 增加布尔字段 `layer_residual`。
   - 旧 artifact 缺省该字段时按 `false` 解释，保证 v2.4 的 `candidate_spec.json` 和复现语义不变；
   - 新 profile 采样的每个 GRU spec 必须显式写入 `true`；
   - `false` 保留目前的单个多层 `nn.GRU` 实现；`true` 使用逐个一层 `nn.GRU` 的 `ModuleList`，每层后做残差相加；不增加 LayerNorm、dropout 或额外可训练参数。
3. `render_model_py()` 与实际模型实现必须完全同义；为两种 `layer_residual` 值增加结构/前向回归测试，并保持无残差的历史测试通过。
4. 两条题面格式路径都显式展示：

   `- Layer residual connections: enabled; after each GRU layer, h = h + GRU_layer(h).`

   即包内 `src/architecture_iq/prompts/formatters.py` 和 Inspector 的 `tools/question_inspector/prompt_format.py` 都要更新并做 parity test。Transformer 题面不虚构该 GRU 属性。

## 轨迹来源与采样

1. **沿用**现有 `bigram_lm` dataset instance（`bg_fdc03b`）：它是 16 条可复用 v2.4 Transformer GT 的共同物化数据，且新 v2.5 residual GRU 也必须在同一 `dataset_id` 上训练，才能满足同题共享数据的不变量。profile 是新的，但 dataset instance 不是；dataset 语义和训练协议均未改变。
2. 从已完成、健康且审计通过的 v2.4 Transformer GT 中，以固定 RNG seed **随机抽取 16 条 Transformer trajectories**。
   - Transformer 代码和训练协议未变，因此其数值 GT 可跨 profile 复用；
   - 每条引入的旧轨迹保留原 `candidate_spec.json`、候选 ID、source profile/hash 与原 GT，不复制/改写成 v2.5 candidate；
   - v2.5 candidate-set/run manifest 新增 `historical_transformer_provenance`，逐条记录 source dataset、candidate、profile/hash、compatibility assertion 与选择 RNG seed；审计器据此以受控的历史来源验证它们。
3. 在该既有 dataset instance 下，按 v2.5 profile 随机采样 **16 个不重复的 residual GRU configurations**（沿用 v2.4 的 13 widths × 8 depths 采样空间），每个完整跑 10 seeds 的真实 `train.py` GT；不得复用没有 residual 的旧 GRU 轨迹。
4. 所有新 GRU 的 candidate/set manifest 记录 `layer_residual: true`、profile hash、训练脚本 hash 与并行设置。CPU seed 并行继续采用一核一 seed 的受控配置，前提是运行前 smoke 验证维持既有确定性与性能门槛。
5. Gate 1/2：审计 16 条新 GRU、16 条历史 Transformer 及其 bridge provenance；任何 profile/dataset/训练身份不兼容、失败 seed 超阈值、或 residual 字段不正确的候选均不进入 pair pool。

## 100 题组装与赢家平衡

1. 在通过 Gate 1/2 的两类候选间枚举所有唯一 GRU--Transformer pair，以新 profile 的显著性函数重算：`gap_min=0`，其余显著性门槛见待确认项。只保留显著、跨类型且兼容的 pair。
2. 使用固定的 run RNG seed 随机打乱 eligible pairs，再构成 100 个 pair-unique 题目；每题仅包含一 GRU 和一 Transformer。run 明确声明：
   - `candidate_reuse_policy: blind_pair_unique`；
   - `candidate_reuse_allowed: true`；
   - `pair_reuse_policy: unique`；
   - `canonical_blind_evaluation: false`；
   - `run_purpose: review_blind_pool`。
3. 统计 100 题独立重算得到的 `winner_model_type`。若任一类型多于 70：
   - 从该多余赢家类型的已选题中，用 run RNG 确定性地随机删除到至多 70；
   - 从尚未选择、赢家为另一类型、显著且 pair-unique 的 eligible pool 随机补入，直到重新达到 100；
   - 每轮填补后重新检查 pair 唯一、题内兼容性、显著性和赢家比例。
4. 如果另一赢家类型的 eligible unique-pair 容量不足以使两类各至少 30 题，**停止而不静默放宽 70% 限制**。报告两类 eligible/selected 数量、受限原因和可选处理：增加 residual GRU candidates、增加可用 Transformer trajectories、或由你显式修改平衡规则。

在 16×16 的候选矩阵中，pair 唯一的理论上限是 256，足以容纳 100 题；但显著性和赢家平衡可能降低实际容量，因此 100 题不是在训练开始前可保证的结果。

## Gate 3/4 审计与交付

审计器对该 run 必须逐题并独立检查：

- 候选 provenance（新 residual GRU 或受控的历史 Transformer bridge）与 profile/dataset 兼容性；
- 每题是 GRU-vs-Transformer，模型类型不相同；
- 同一 pair 在整个 run 中不重复；候选跨题复用不作为失败条件；
- 仅模型规格变化，数据、优化器、loss、batch、预算、device 保持一致；
- 新 profile 的 `gap_min=0` 与已确认的胜利显著性规则均通过；
- `correct_letter`、winner、记录的 gap/win rate 与独立重算一致；
- 100 题恰好完成，`winner_model_type` 的频数均不大于 70；
- public prompt 不泄漏私有 GT，且每道 residual GRU 题面清楚写明逐层 residual。

交付内容：v2.5 profile、实现/回归测试、候选与 run manifests、Gate 1/2/3/4 JSON/Markdown 报告、100 题目录和 Inspector collection。Inspector 可直接显示这个 review pool 的答案/指标，因为本题集不用于人与 agent 的正式分数比较。

## 执行顺序与人工插手点

1. **确认语义**（下方两个待确认点）；冻结 profile 名称、显著性规则和 bridge provenance schema。
2. 实现 profile/model/prompt/audit/selection 改动，做 unit + residual smoke。
3. 新建 dataset，随机选择 16 条 Transformer，训练 16 条 residual GRU，做 Gate 1/2。
4. 预审 pair 容量与赢家类型分布；若少于 100 或任一类别无法达到 30，暂停给出证据，等你决定扩充候选还是调整规则。
5. 生成、平衡、审计 100 题，创建 Inspector collection；交付前可先抽取 5--6 题供你盲审题面。

## 待你确认的语义（推荐默认）

1. **“仍要求胜利显著性”具体保留哪些条件？** 推荐：仅把 `gap_min` 从 0.05 改为 0，继续要求 `win_rate_min=0.70` 与 `use_non_overlap=true`。这样不再要求效果量下限，但仍排除 seed 层面不稳定、均值区间重叠的赢家。若你的意思是只保留 `win_rate_min`、取消 non-overlap，需要在 profile 和审计报告中明确为另一种契约。
2. **16 个新 residual GRU 的采样空间是否保持 v2.4 的 104 种（13 widths × 8 depths）？** 推荐先保持，便于将 residual 作为唯一架构语义改变；若希望进一步减少“深层仍难训”的风险，可在新 profile 中改写 depth distribution/配置池，但那会同时改变 residual 与采样分布，因果解释会更弱。

已按你的描述默认：`blind_pair_unique`（pair 不重复、候选可无限跨题复用），并允许 Inspector 显示 review pool 的答案；这两点不再作为阻塞项。
