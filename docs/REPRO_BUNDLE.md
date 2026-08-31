# 可复现代码包（repro bundle）

网站顶栏的 **↓ Code** 按钮把当前题目打包成一个 zip，用户解压后
`python reproduce.py` 就能在自己机器上把 Ground Truth 重跑出来。

设计要点：**不改 bake、不改 schema、不改 `tools/export_quiz_static.py`。**
复现需要的文件早就内联在 BakeFile 里了，前端只是把它们重新打包。

| bake 字段 | 打进包里的位置 |
|---|---|
| `detail.dataset.files` | `dataset/dataset_spec.json`、`dataset/synthesize.py` |
| `detail.choices[].files` | `choices/<L>/{candidate_spec,model,loss,optimizer,train}` |
| `reveal.files[letter]` | `choices/<L>/reference/summary.json`（**仅答题后**） |
| `detail.prompt` | `prompt.txt` |

## 包结构

```
architectureiq_q_502033.zip
└── q_502033/
    ├── README.md            # 静态文件, 来自 frontend/quiz/repro/README.md
    ├── reproduce.py         # 静态文件, 来自 frontend/quiz/repro/reproduce.py
    ├── question.json        # 生成: metric / seeds / 每个选项的 budget
    ├── prompt.txt
    ├── dataset/{dataset_spec.json, synthesize.py}
    └── choices/<A|B|C>/
        ├── {candidate_spec.json, model.py, loss.py, optimizer.py, train.py}
        └── reference/summary.json     # 仅答题后
```

包里**不放 `.pt`**：`synthesize()` 的输出与磁盘上 materialize 出来的张量逐位相同
（这是 pipeline 的核心 invariant），所以数据从源码重新生成即可，包体积约 50 KB。

## 门控（gating）

| 时机 | 内容 |
|---|---|
| 答题前 | 全部代码 + `prompt.txt`。**没有** `reference/`，`question.json` 里**没有** `correct_letter` / `ranked`，`answered: false` |
| 答题后 | 追加 `choices/*/reference/summary.json`，`question.json` 补上 `correct_letter` / `ranked`，`answered: true` |

两个版本的代码文件逐字节相同 —— 只有结果被扣住。门控判定复用页面里的
`answered`，与文件查看弹窗（`main.tsx` 的 `InfoModal`）用的是同一个开关，
所以"弹窗里能看到的"和"zip 里有的"永远一致。

## 两份实现，一个契约

| 实现 | 位置 | 用途 |
|---|---|---|
| 浏览器 | `frontend/quiz/src/bundle.ts` + `frontend/quiz/src/zip.ts` | 线上的下载按钮；零依赖 zip（STORE，不压缩） |
| 命令行 | `tools/export_repro_bundle.py` | 从任意 bake 导出，只读 bake、**不 import `architecture_iq`** |

`reproduce.py` 与 `README.md` 是 `frontend/quiz/repro/` 下的**真实文件**：TS 侧用
Vite 的 `?raw` 导入，Python 侧直接读同一个文件，所以不存在"TS 里那份 Python 脚本
过期了"的问题。

`tests/test_repro_bundle.py::test_browser_and_cli_builders_agree` 通过
`frontend/quiz/scripts/dump-bundle.mjs`（esbuild 把 `bundle.ts` 打成 Node 可跑的
模块）逐文件比较两侧输出。除 `.json` 外必须逐字节相同；`.json` 只要求解析后相等，
因为两个 JSON writer 对同一个数的写法可以不同（`3e-05` vs `0.00003`）。

```bash
.venv/bin/python tools/export_repro_bundle.py --list
.venv/bin/python tools/export_repro_bundle.py --question q_502033 --out /tmp/q --answered
cd /tmp/q/q_502033 && python reproduce.py --seeds 1
```

## `reproduce.py` 的语义

严格镜像 `src/architecture_iq/ground_truth/runner.py` 与
`src/architecture_iq/runtime/loader.py`，但不 import 仓库代码（只要 torch + numpy）：

1. 加载 `dataset/synthesize.py`，调 `synthesize()` 得到固定的 train/test 划分。
2. 每个选项：把 `choices/<L>/` 插到 `sys.path[0]`，**先清掉缓存的
   `model`/`optimizer`/`loss`/`train` 模块**，再按路径加载 `train.py`
   —— 不清的话选项 B 会拿到选项 A 的 `model`，这也是 `loader.py` 里
   `_clear_cached_sibling_modules` 存在的原因。
3. `seed = base_seed + i`，逐 seed 调
   `train_and_eval(..., steps=, batch_size=, seed=, fail_threshold=, device=)`，
   取 `final_{selection_metric}`。
4. 聚合 `np.mean` / `np.std`（`ddof=0`），失败 seed 剔除；全失败则 `inf`。
   均值最低者为答案。

## 精确性

同一 torch 构建 + 同一线程数下，重跑与记录值**逐位相同**（实测 `rel_err=0`）。
线程数不同会有约 `1e-7` 的相对偏差 —— float32 归约不满足结合律，线程数决定了
部分和的相加顺序。

`summary.json` 的 `environment.torch_threads_per_seed` 记录了当初的线程数，
打包时会写进 `question.json`（这是环境信息、不是答案，所以答题前也给），
`reproduce.py` 默认据此 `torch.set_num_threads(...)`。因此默认用相对容差
（`--tol`，默认 `1e-4`）判定，而不是要求相等；超出容差则打印全部 mismatch 并以
非 0 退出码结束。

## 已知边界

- 旧 bake 若没有 `files` 字段，按钮会 disabled 并在 `title` 里说明
  （`canBuildBundle()`）。
- 只做单题下载，不做"打包全部题目"。
- zip 不压缩：几十 KB 的纯文本，省掉一个前端依赖更划算。

---

## 验证记录（2026-08-31，当前 30 题 bake）

### 1. 包里的代码 = 生成器合成、且真正跑出 GT 的那一份

- **bake ↔ 磁盘逐字节一致**：把 30 题的包全部导出，与
  `data/datasets/{family}/{dataset_id}/**` 对比 —— 390 个 `.py`、210 个 JSON
  （`candidate_spec.json` / `dataset_spec.json` / `results/summary.json`）
  全部逐字节相同，0 缺失、0 不一致；每个 `summary.json["execution"] == "candidate_py_files"`。
- **spec → 代码可重新渲染出同样的字节**：这批候选由 v1.4 生成器
  （commit `55a634d`，分支 `fix/v12-template-sampler-clean`）产出。用该 commit 的
  `src/` 对 90 个候选的 `candidate_spec.json` 重跑
  `ground_truth/runner._sync_candidate_files()`，360 个 `.py` 与磁盘/包中的
  **完全无差异**。即包里的代码就是 `write_candidate()` 由冻结 spec 合成的结果。
- **注意（分支差异，非包的问题）**：前端分支 `quiz-frontend` 不包含 `55a634d`，
  它的 `models/mlp.py` 仍要求旧的 per-layer `activations` 字段，而 v1.4 的 MLP spec
  用单个 `activation`。因此**在 quiz-frontend 上**对这批候选调用
  `run_ground_truth()` / 渲染 prompt 会抛 `KeyError: 'activations'`。
  重跑 GT 或重新出题必须在 v1.4 生成器分支上做。下载包不受影响：它自带全部代码，
  不 import `architecture_iq`。

### 2. `reproduce.py` 的数值已实测对上

| 范围 | 结果 |
|---|---|
| 30 题 × 3 选项 × seed 0（共 90 次训练，走打好的包） | 90/90 与 `summary.json` 记录值一致，其中 **89 个 `rel_err=0`（逐位相同）**，0 mismatch，退出码全为 0 |
| q_b11c3d（bigram_lm）全量：3 选项 × 10 seeds | 30 个逐 seed 值 + 3 个 mean **全部 `rel_err=0`**，胜者 C == `correct_letter` |
| 用的包 | 上面那次是**浏览器里点按钮真下载下来的 zip**（非 CLI 导出），48413 B / 21 entries（答题前）、54194 B / 24 entries（答完） |
| pytest | `tests/test_repro_bundle.py` 7 passed，其中 `test_reproduce_py_reproduces_recorded_ground_truth` 真跑 `reproduce.py` 并断言与记录值相对误差 < 1e-3 |

逐位相同的前提是相同 torch 构建 + 相同线程数；包里的 `question.json` 会带上
`summary.json` 记录的 `torch_threads_per_seed`，`reproduce.py` 默认按它
`torch.set_num_threads(...)`。换线程数时相对偏差约 1e-7，仍远小于默认容差 1e-4
（上面 90 次里唯一一个非逐位的就是这种量级）。
