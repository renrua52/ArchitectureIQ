# ArchitectureIQ 后端存储设计（Backend Storage）

> 状态：**已实现**（2026-07-31，分支 `shaoyang/local-agent-dev`）。本文件是 `backend/` 物理布局与数据 schema 的权威定义。
> 关联：`plan-v2.md`（管线设计）、`PROTOCOLS.md`（评测协议）、`AGENTS.md`（不变量）。
> 存储 API 实现：`src/architecture_iq/storage/`；迁移工具：`tools/storage/migrate_data_layout.py`。

---

## 1. 目标

- 题目实例与评测实例完全分离：题目实例只描述 **数据集 + 一组 config JSON**；组合出题、筛选、评分全部在评测端。
- 列式存储：`problems / trainers / candidates / results` 作为平铺列，题目编号在各列内部，便于整体提取"所有题目 / 所有 configs / 所有曲线"，各列体积可以悬殊。
- generator（生成套件）与存储系统解耦：generator 只通过 storage API 写；评测端只通过 storage API 读。发布时只需要 `data + eval`，generator 不是发布物。
- 所有代码设计都包含在题目代码里：candidate config 是**闭合 JSON**，只描述 `model / optimizer / loss / budget`，不携带新逻辑。

## 2. 目录结构

```text
backend/
├── data/                                # 题目实例仓库（列式存储，gitignored）
│   ├── problems/{problem_id}/           # 数据集 + 介绍文档 + 生成代码 + 物化张量
│   ├── trainers/{trainer_id}/           # 训练脚本（train.py 模板，按 family 独立）
│   ├── candidates/{problem_id}/         # 该 problem 的 config JSON 闭集
│   └── results/{problem_id}/{candidate_id}/   # summary.json + curves.npz（+ 可选 ckpt）
├── generator/                           # （规划）生成套件，与存储解耦，发布不需要
└── eval/                                # （规划）评测端：组合题 + LLM 评测 + 后续 meta-model
```

### 2.1 `data/problems/{problem_id}/`

| 文件 | 说明 |
|------|------|
| `dataset_spec.json` | problem spec（原 `dataset_spec.json`，含 family/params/selection_metric/significance/files） |
| `README.md` | 介绍文档（自动生成：family、metric、significance、数据来源） |
| `synthesize.py` | 数据生成代码（执行它得到物化张量） |
| `train.pt` / `test.pt` / 其它物化文件 | `dataset_spec.json["files"]` 里列出的所有文件 |

### 2.2 `data/trainers/{trainer_id}/`

| 文件 | 说明 |
|------|------|
| `trainer_spec.json` | `{schema_version, trainer_id, family, version, content_sha256, source}` |
| `train.py` | family 训练循环模板（当前只有 `bigram_lm` 与 regression 两套） |

`trainer_id = "{family}_v1"`。训练脚本按 family 独立；渲染 model/loss/optimizer 的规则属于 generator 侧（解耦）。

### 2.3 `data/candidates/{problem_id}/{candidate_id}.json`

一个 config JSON（schema_version 2.0），**闭合**：

```json
{
  "schema_version": "2.0",
  "problem_id": "mvar_866b4e",
  "candidate_id": "c_37c2e2",
  "family": "multivariate_regression",
  "budget": {"training_steps": 5120, "batch_size": 16, "total_samples_seen": 81920},
  "model": {"type": "mlp", "depth": 4, "width": 32},
  "optimizer": {"type": "SGD", "lr": 0.001, "weight_decay": 1e-05, "momentum": 0},
  "loss": {"loss_id": "mse"}
}
```

- 不存任何 `.py` 文件；代码在评测/运行期由（problem 代码规则 + config）确定性渲染，保证 prompt 展示的代码 = 执行的代码。
- demo settings = config 子集（后续评测设计时加 `role: demo|eval` 标记，供 few-shot）。

### 2.4 `data/results/{problem_id}/{candidate_id}/`

| 文件 | 说明 |
|------|------|
| `summary.json` | GT（与 candidates 一一对应；n_seeds / mean / std / seed_results） |
| `curves.npz` | 逐 step 曲线 |
| `ckpt/`（可选） | 未来需要时的检查点 |

## 3. ID 规则

- `problem_id` = 原 `dataset_id`（content-addressed，family 前缀保留：`mvar_*` / `bg_*` / `sym_*`）。
- `candidate_id` = 原 `c_{hash}`，不变。
- `trainer_id` = `{family}_v1`（内容 sha256 防漂移）。
- 所有引用一律用 ID，不存路径（旧 `question.json` 里的 `candidate_path` 在迁移后废弃）。

## 4. 不变量（延续 AGENTS.md）

1. spec → code → run → GT 一致：GT 只来自执行生成代码。
2. 同一个 problem 的所有 candidate 共享同一物化数据（公平比较）。
3. candidate config 是闭合 JSON（不引入新逻辑）。
4. 题目非重复（non-repeating candidates）、anti-shortcut gates 在评测端执行。
5. `backend/data/` gitignored（同旧 `data/`）；发布用 attested bundle。

## 5. 旧布局 → 新布局映射

| 旧（`data/datasets/{family}/{dataset_id}/`） | 新 |
|---|---|
| `dataset_spec.json` + `synthesize.py` + `train.pt/test.pt` + 其它 `files` | `data/problems/{problem_id}/` |
| `candidates/set_*/c_{hash}/candidate_spec.json` | `data/candidates/{problem_id}/{candidate_id}.json`（config v2） |
| `candidates/set_*/c_{hash}/results/{summary.json,curves.npz}` | `data/results/{problem_id}/{candidate_id}/` |
| `candidates/set_*/c_{hash}/train.py`（family 模板） | `data/trainers/{family}_v1/train.py` |
| `questions/run_*/q_*/` | 移出题目实例 → 评测端（`eval/`，后续实现） |

## 6. 迁移工具

```bash
# 默认 copy 模式，写入 backend/data/
python tools/storage/migrate_data_layout.py

# 只预览
python tools/storage/migrate_data_layout.py --mode dry-run

# 只迁移指定 family / 前 N 个 dataset
python tools/storage/migrate_data_layout.py --families multivariate_regression
python tools/storage/migrate_data_layout.py --limit 3

# 迁移后删除旧目录（先 copy 验证后再用）
python tools/storage/migrate_data_layout.py --mode move
```

幂等：重复运行会覆盖同名输出，可安全重跑。

## 7. 实施状态

- [x] 设计文档（本文件）
- [x] storage API：`src/architecture_iq/storage/schema.py` + `repository.py`
- [x] 迁移脚本 + 测试
- [x] 存量 `data/datasets/*` → `backend/data/`（copy 模式，旧数据保留待验证）
- [ ] generator 迁移到 storage API（解耦改造，另行安排）
- [ ] `backend/generator/`、`backend/eval/` 落地（generator 套件独立；eval 由评测设计驱动）
