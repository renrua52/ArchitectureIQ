# 发布候选整理：跟踪与可移植性基线

适用工作区：`ArchitectureIQ-main-integration`。本文件只定义发布候选的 Git 与产物边界；不迁移或删除其它 worktree。

## 1. Git 跟踪规则

| 类别 | 位置 | 规则 |
|---|---|---|
| 发布代码与配置 | `src/`、`tools/`、`profiles/`、`tests/`、`frontend/quiz/` | 跟踪 |
| 正式前端数据 | `frontend/quiz/public/data/questions.json` | 暂时跟踪；它是当前静态 demo 的交付输入 |
| 发布证明 | `outputs/demo_release_integration/RELEASE_FREEZE_MANIFEST.json`、最终报告 | 跟踪；只存轻量 manifest 和说明 |
| 训练数据和 GT | `data/`、`llm_runs/` | 忽略；由外部数据包或本机生成提供 |
| 运行与测试临时物 | `.pytest-audit*/`、`.pytest-tmp-*/`、`.tmp/`、`outputs/` 其余内容 | 忽略 |
| 人工审计要求 | `../reports/AUDIT_REQUIREMENTS_2026-07-23.md` | 应作为需求文档跟踪，不可忽略 |

`AGENTS.md` 的 worktree 边界修改同样属于应审阅、应跟踪的工作，不应以 ignore 掩盖。

## 2. 可移植 release manifest

`tools/freeze_demo_manifest.py` 以 `demo_release_freeze_v2` 和 `path_base=repository_root` 标识路径语义，并且必须只写仓库相对、POSIX 格式路径，例如：

```text
outputs/demo_release_integration/demo_release_collection_v2.json
profiles/v2.2.yaml
frontend/quiz/public/data/questions.json
```

collection 重建以 UTF-8/LF 写入，令 SHA-256 跨平台稳定；manifest 保留该 SHA-256，并链接 tracked `release_specs/demo_085aa570328f.json`；因而即使 collection/data 作为外部产物交付，也能在目标机器验证得到的是同一份内容。任何位于仓库外的输入都应被生成器拒绝。

## 3. 当前发布边界与未决事项

冻结 collection 和原始 `data/` 都是本地忽略产物；因此仅把路径由绝对改为相对，并不能让全新 clone 自动重建 46 题 demo。合入 `main` 前必须二选一：

1. 交付一个带 hash 的外部 release-data bundle，并在发布说明中写明取得和验证步骤；或
2. 将最小的、可审计的 collection manifest 纳入 Git，并提供从外部 data root 生成 BakeFile 的命令。

本阶段不将大规模训练 artifact、静态导出或性能目录加入 Git。当前交付 bundle 是 `outputs/demo_release_integration/demo_085aa570328f_data_root.zip`；其 SHA-256 记录在 freeze manifest，解压后的目录直接作为 `--data-root`。

## 4. 每次发布前的检查

```powershell
git status --short
git check-ignore -v .pytest-audit
python tools/build_demo_release_collection.py --spec release_specs/demo_085aa570328f.json --data-root <external-data-root> --output outputs/demo_release_integration/demo_release_collection_v2.json
python tools/freeze_demo_manifest.py --collection outputs/demo_release_integration/demo_release_collection_v2.json --release-spec release_specs/demo_085aa570328f.json --data-bundle outputs/demo_release_integration/demo_085aa570328f_data_root.zip --output outputs/demo_release_integration/RELEASE_FREEZE_MANIFEST.json
python -m pytest tests/test_freeze_demo_manifest.py -q --basetemp .pytest-tmp-release-manifest
```

通过条件：没有意外的未跟踪运行产物；manifest 中没有机器绝对路径；所有 manifest hash 与当前输入一致。
