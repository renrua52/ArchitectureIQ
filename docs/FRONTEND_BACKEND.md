# 前端 / 后端怎么分工

用大白话说：

1. **后端**跑题目生产流水线（pipeline）：造数据、训候选、筛出 `q_xxxxxx` 这类题目，再**导出成一份 bake 文件**（JSON 题包）。
2. **bake 必须符合**仓库里的约定：`contracts/quiz_bake.schema.json`（可用 `tools/validate_quiz_bake.py` 检查）。小样例：`contracts/examples/mini_bake.json`。
3. **前端**只负责**展示 bake**：读 JSON，画出题目、选项、揭晓等。不要去读原始的 `q_xxxxxx` 文件夹，也不要跑训练流水线。
4. 旧的 Streamlit 页面先别加新功能；产品界面在 `frontend/quiz/`。

## 在哪个分支开发

从 `main` 已开好两条分支（请拉最新再开发）：

| 谁 | 分支 | 主要改这些目录 |
|----|------|----------------|
| 前端同学 | `quiz-frontend` | `frontend/quiz/` |
| 后端同学 | `quiz-backend` | `src/architecture_iq/`、`profiles/`、`tools/export_quiz_static.py`、`examples/` 等 |

两边都可能改到的约定放在 `contracts/`：改 schema 前先说一声，前后端都看过再合。

> 注：不能叫分支 `frontend`，因为仓库里已有 `frontend/vanilla`，Git 不允许再建同名层级。

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
