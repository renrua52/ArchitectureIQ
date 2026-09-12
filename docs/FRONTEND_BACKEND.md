# 前端 / 后端怎么分工

用大白话说：

1. **后端**跑题目生产流水线（pipeline）：造数据、训候选、筛出 `q_xxxxxx` 这类题目，再**导出成一份 bake 文件**（JSON 题包）。
2. **bake 必须符合**仓库里的约定：`contracts/quiz_bake.schema.json`（可用 `tools/validate_quiz_bake.py` 检查）。小样例：`contracts/examples/mini_bake.json`。
3. **前端**只负责**展示 bake**：读 JSON，画出题目、选项、揭晓等。不要去读原始的 `q_xxxxxx` 文件夹，也不要跑训练流水线。
4. 旧的 Streamlit 页面先别加新功能；产品界面在 `frontend/quiz/`。

## 在哪个分支开发

从现在起统一在 `main` 开发。开始工作前先同步：

```bash
git fetch origin --prune
git switch main
git pull --ff-only origin main
```

前端主要改 `frontend/quiz/`；后端主要改 `src/architecture_iq/`、
`profiles/`、`tools/export_quiz_static.py` 和 `examples/`。前后端共同依赖的
约定放在 `contracts/`，修改 schema 前要同时运行 BakeFile 校验和前端构建。

## 本地怎么跑

后端导出并检查：

```bash
.venv/bin/python tools/export_quiz_static.py
.venv/bin/python tools/validate_quiz_bake.py
```

前端：

```bash
cd frontend/quiz
# 可选：用小题包
# cp ../../contracts/examples/mini_bake.json public/data/questions.json
npm install && npm run dev
```
