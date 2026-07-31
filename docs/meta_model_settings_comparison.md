# Meta Model Setting 对比(60题回归 vs 9000题回归)

日期:2026-07-24
范围:`data/meta_model_studies/` 下的 setting-to-loss 回归/分类研究 + `artifacts/wide_v2_full30_meta_model_final_summary.json`。

---

## 0. 结论

| | 60题回归(`setting_to_loss_60q_id_v1`) | 9000题回归(`wide_v2_full30_global_conditioned_fixed`) |
|---|---|---|
| **头部结果** | 外部盲测 54/60 = **90.0%** | macro 显著三选一准确率 **82.11%** |
| **性质** | 3环境内 ID split,窄样本 | 30环境池化 ID split,已过显著性过滤 |
| **两者关系** | 不可直接比较(任务窄度/口径/样本量均不同) | — |

---

## 1. 已剔除的设定

| 被剔除的设定 | 结论 |
|---|---|
| 旧版 63.50% / 66.85% / 68.08% | 已作废,被 82.11% 取代 |
| `wide_v2_full30_family_logo_recovery` | log R² 为负(-0.35~-0.37),失败案例 |
| `winner_clf_fast` / `winner_clf_trees`(74.85%) | 聚合口径(按15个dataset)与其他 setting(按27个environment)不一致,不可比 |
| `winner_clf_global_trees`(56~62%) | 数字可疑,不采纳 |
| `winner_clf`、`winner_clf_trees2` | 无产出 |
| `setting_to_loss_wide_v2_b1_*`、`_completed_*`、`wide_v2_logo_smoke17_*` | 阶段性 pilot,非最终结果 |

---

## 2. Setting:60题回归(`setting_to_loss_60q_id_v1`)

### 2.1 数据与切分

| 项目 | 内容 |
|---|---|
| 覆盖范围 | 3 个 environment(3 family 各1个固定 dataset instance + budget + batch size + loss) |
| train | 每个 environment 内 900 行 |
| test(内部验证) | 每个 environment 内 100 行,同一环境 |
| test(外部盲测) | 冻结60题,同环境、候选组合不重叠 |
| train/test 关系 | 环境内 ID split |

### 2.2 模型结构

- 特征工程(`tools/meta_model_study/features.py::FeatureEncoder`,`feature_set="full"`):优化器 type×log10(lr) 交互项、weight_decay/momentum/betas、budget 的 log(total_samples_seen)/log(batch_size)、loss_id、模型架构展开特征(MLP:depth/log2(width)/激活函数分布;Transformer:d_model/num_heads/num_layers等)+ 原始 setting 全字段展开;体量特征仅 `derived.log_total_params`;单环境故 `dataset_conditioning="unaware"`;Pipeline 内接 `DictVectorizer → VarianceThreshold(0) → StandardScaler`。
- 最优回归器:`XGBRegressor`(以 bigram 环境为例:`n_estimators=800, max_depth=6, learning_rate=0.1, colsample_bytree=0.6, min_child_weight=1, reg_alpha=0.01, reg_lambda=0.1, subsample=1.0`),预测目标 `log(mean_loss)`。
- 各 family 最优模型:Bigram→ExtraTrees,Multivariate/Univariate→XGBoost(逐环境单独调参,不共享模型)。

### 2.3 结果

| 指标 | 数值 |
|---|---|
| 单环境5折CV(bigram) | log R²=0.968,mae_log=0.0049 |
| 外部冻结60题盲测 | 54/60 = **90.0%**(cv_champion/集成/XGBoost并列) |
| 结构化OOD侧测(leave-one-optimizer-out等) | log R² 转负(如-0.20) |

---

## 3. Setting:9000题回归(`wide_v2_full30_*`)

3 family × 10 dataset instance = 30 environment,10,000行(9000 train + 1000 validation,5 seed)。

### 3.1 Setting A — 分数据集独立训练(`wide_v2_full30_dataset_pooled_baseline`)

| 项目 | 内容 |
|---|---|
| train | 每个 dataset 各自训一个模型,约600行/dataset |
| test | 各自dataset验证行,15个dataset汇总1000行 |
| train/test 关系 | 同dataset内 ID split |
| 模型结构 | `FeatureEncoder(dataset_conditioning="unaware")`;最优 `ExtraTreesRegressor`(如 bg_5d89d5:`n_estimators=500, max_depth=16, max_features=1.0, min_samples_leaf=1`) |
| 各方法结果(macro显著三选一) | XGBoost 78.33% > ExtraTrees 77.78% > RandomForest 75.35% > compact_polynomial_ridge 60.63% > full_ols/full_ridge ~54-55% > MLP 52.44% > compact_ols/ridge ~45% > params_ols 31.14% |

### 3.2 Setting B — 全局共享模型(`wide_v2_full30_global_conditioned_fixed`,headline)

| 项目 | 内容 |
|---|---|
| train | 全部9000行,1个共享模型 |
| test | 全部1000行,覆盖27个environment(152,592三元组,显著性过滤后87,837个即57.6%参与计分) |
| train/test 关系 | 跨dataset共享参数的池化 ID split(非LOGO,train已含test所在的同批30个dataset instance) |
| 模型结构 | `FeatureEncoder(dataset_conditioning="id")` + compact/raw full特征,122维 → `Pipeline(features=FeatureEncoder, model=ExtraTreesRegressor(n_estimators=300, max_depth=16, max_features=0.7, min_samples_leaf=5, random_state=20260714))`,预测`log(mean_loss)` |
| 训练内5折CV | log R²=0.687 |
| 最终结果 | **82.11%**(RandomForest_fixed 78.60% / XGBoost_fixed 70.80% / MLP_fixed 60.31% / full_ridge_fixed 55.27% / compact_ridge_fixed 43.55%) |

### 3.3 对照:真正跨数据集泛化(`grouped_dataset_logo`)

| 项目 | 内容 |
|---|---|
| train/test 关系 | LOGO(leave-one-dataset-out),15折 |
| 结果 | log R² 0.365~0.474(最好为`unaware+with_params`) |

---

## 4. 预测目标(loss)本身的尺度:均值/标准差 vs 模型预测误差

建模目标统一是 `log(mean_loss)`(自然对数),不是原始 loss。原始 loss 跨 family/环境差 10+ 个数量级(如 `wide_v2_full30_gt_snapshot_bounds.json` 中 multivariate 环境的 `observed_environment_regret_upper` 从 0.44 到 4.1e11 不等),所以只在 log 尺度上报告均值/标准差有意义;raw 尺度仅在单一环境内给出示意值。

**目标(log loss)均值/标准差(即 `constant_mean` 常数预测值 = train 均值;标准差取 `constant_mean` 的 RMSE,因其残差偏置≈0)** vs **最优模型的预测误差标准差(RMSE)**:

| Setting | 环境 | 目标均值(log loss) | 目标标准差(log loss) | 目标均值(raw loss,示意) | 最优模型 | 预测误差 RMSE(log) | 预测误差 MAE(log) | R² |
|---|---|---|---|---|---|---|---|---|
| 60题(单环境) | bigram_bg_0021c1(CE) | 1.223 | 0.0417(train 5折CV) | ≈3.40 | ExtraTrees | 0.0068(val) | 0.0028 | 0.975 |
| 60题(单环境) | multivariate_mvar_c59a30(MSE) | -0.367 | 0.698(train 5折CV) | ≈0.69 | XGBoost | 0.282(train CV) | 0.193 | 0.836 |
| 60题(单环境) | univariate_sym_62678b(MSE) | -1.653 | 1.006(train 5折CV) | ≈0.19 | XGBoost | 0.585(train CV) | 0.359 | 0.662 |
| 9000题(全局池化,headline) | 30个环境池化 | -0.109 | 3.533(锁定validation) | 不适用(跨family不可比) | ExtraTrees(fixed) | 1.875(锁定validation) | 0.633 | 0.718 |

**结论:**

1. 60题设定里,每个环境内部的 loss 本身波动就很小(尤其 bigram,标准差仅0.04),模型几乎是在做"内插",预测误差远小于目标标准差,R²自然很高(0.66~0.98)。
2. 9000题全局池化设定里,把30个环境（3种loss尺度、不同family）的 log loss 硬拼在一起,目标标准差被拉大到3.53,模型（ExtraTrees）能把预测误差压到1.875,即解释了约72%的方差(R²=0.718),但绝对误差规模明显大于60题单环境设定。
3. 两个协议的 R² 和三选一准确率不能跨设定直接比较,因为目标标准差(任务难度的分母)本身相差近百倍(0.04~1.0 vs 3.53)。

---

## 5. LLM 评测(已补充)

### 5.1 原始 60 题盲测(3 family × 20 题)

| 模型 | 60题正确率 | bigram_lm | multivariate | univariate | 协议 |
|---|---|---|---|---|---|
| Claude Opus 4.8 high | **34/60 = 56.7%** | 13/20 | 13/20 | 8/20 | 单 prompt 60 题,无工具调用 |
| GPT-5.5 high | 27/60 = 45.0% | 10/20 | 10/20 | 7/20 | 同上 |
| GPT-5.6-SOL blind(旧) | 25/60 = 41.7% | — | — | — | 同 prompt |
| 共享 ExtraTrees(meta-model) | 43/60 = 71.7% | — | — | — | 30 环境宽协议训练,60 题为外部锁定题 |
| 随机 | 20/60 = 33.3% | — | — | — | — |

### 5.2 新 ranking 题协议

除 60 题外,还构造了 per-dataset 5 选排序题:

- **bg0021c1**: 10 题,候选来自 `data/datasets/bigram_lm/bg_0021c1/candidates/set_5120_var_var_fix_f94f2e`,60 个 eligible candidates,每题 6 个 calibration + 5 个 target 排序。
- **sym62678b**: 8 题,候选来自对应 univariate 候选集。
- 评估方式:LLM 根据 prompt + 曲线图片对 T1–T5 排序,与 `true_order` 比较 inversion 计数(0 为全对)。
- 产物路径:
  - `artifacts/ranking_questions/ranking_bg0021c1_pairwise_sig5_q10/`
  - `artifacts/ranking_questions/ranking_sym62678b_pairwise_sig5_q8/`

当前尚无 LLM 批量结果,后续跑完后补到本节。

### 5.3 与 meta-model 的关系

- 在 60 题上,**Claude Opus 4.8 (56.7%) 显著优于 GPT-5.5 (45.0%)**,但两者都远低于共享 ExtraTrees 的 71.7%。
- LLM 在 univariate_regression 上最弱(40–35%),仅略高于随机 33.3%;在 bigram_lm 和 multivariate 上可达 50–65%。
- LLM 与 meta-model 的差距说明:当前 prompt 信息不足以让 LLM 稳定复现 GT 排名,而 tabular meta-model 能从 9k 条结构化记录中学到更稳定的 setting→loss 映射。

---

## 6. 总结论

1. 当前唯一可信的头部数字:**82.11%**(Setting B,ID split,ExtraTrees,全局共享)。
2. 60题的90.0%与9000题的82.11%不可直接比较。
3. 跨数据集真实泛化(LOGO)明显弱于ID split:log R² 0.37~0.47 vs 0.68~0.73;family级LOGO为负R²。
4. 两代研究最优模型均为树模型(XGBoost/ExtraTrees);线性方法(OLS/Ridge)全面偏弱(31%~61%)。
5. 纯参数量启发式(`params_ols`)仅31.14%,远低于完整特征的树模型。
6. 82.11%应视为"数据集内选型能力"上限,不代表对全新数据集的泛化能力。

---

## 6. LLM 评测(待补充)

> 占位:另一 session 正在进行的 LLM 评测结果(含新增 setting 的测试)完成后补充至此,与上述 meta model 结果并列对比。
