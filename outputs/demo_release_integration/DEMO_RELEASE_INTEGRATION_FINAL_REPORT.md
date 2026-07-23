# Demo integration internal-audit report（2026-07-21）

本报告取代早期的 32 题 pilot 报告；39 题版本仍保留为 v1 历史冻结记录。当前候选 collection 为 `demo_085aa570328f`。

## 当前内部审计候选

- 46 道题，92 个 candidate；collection 内全局 candidate-disjoint（重复 0）
- 题目顺序由 `demo_release_collection_v2.json` 固定
- 题目均带 profile、profile_hash、track、source_run
- 静态 Inspector 导出：46/46，0 failures
- React BakeFile：46 题，`ordered=true`

| track | profile | 题数 | Gate 3/4 |
|---|---|---:|---:|
| classification_v2_architecture | v2 | 12 | 12/12 |
| v1_optimizer | v1 | 10 | 10/10 |
| v1_mixed | v1 | 12 | 12/12 |
| classification_kan_v2.1_diagnostic | v2.1 | 5 | 5/5 |
| classification_kan_v2.2_expanded | v2.2 | 7 | 7/7 |

## 本轮 React 前端新增能力

- 参数量：choice card 直接显示 `Trainable Parameter Count`；历史 `candidate_spec.json` 缺失时显示 `—`，KAN 既有模型字段保持兼容。
- 说明：reveal 前显示按 family 生成的 task description，并可展开完整 benchmark instructions；dataset/choice info modal 可查看 candidate spec、model、train、loss、optimizer 等说明文件。
- 评论与审核：reveal 后支持 confidence 1–5、keep/revise/reject 和自由文本 comment；反馈通过 `audit_feedback` telemetry 和 session export 保存。
- 分类可视化：synthetic classification 提供规则感知二维 feature projection，class-colored train points、outlined test points、经验 `P(class 1)` 和 feature-pair 说明；bigram heatmap、回归 scatter、learning curves 保持 family-aware 渲染。
## 计划里程碑审计

### M1：远端 main 基线 — 完成

- 本地缓存 `origin/main = 4c254fceaa453b7e780c00c7dd84b7a7fef16c52`
- telemetry 分支缓存 `85d174d341a6e4ba3fcf07520d1a9970955fe3bc`
- GitHub 页面已在线确认 `frontend/quiz`、`services`、`supabase/functions/telemetry`
- Git CLI 在线刷新仍受 Schannel `SEC_E_NO_CREDENTIALS` 阻断，因此 hash 明确标记为“本地已有 ref”

### M2：collection/API 骨架 — 完成

- 新增 `tools/export_quiz_collection.py`，将 collection manifest 转为 React BakeFile
- React quiz 支持 ordered Next、question menu、track/profile/hash provenance
- 新增 `services/quiz_api`：collections/questions/answer endpoints
- GET question 不返回 `reveal`；POST answer 后才返回 reveal
- `tools/quiz_api_smoke.py` 通过
- localhost 浏览器 smoke 已验证：46 题、题目菜单顺序、末题 Next disabled、v2.2 hash 显示正确

### M3：性能基线 — CPU 完成，GPU 不可运行

- torch `2.12.1+cpu`，CUDA unavailable
- CPU smoke：串行 15.355 s，并行 wall 9.919 s，speedup 1.548×
- multiprocessing Pipe 被 managed Windows 环境禁止，自动使用 ThreadPool fallback
- serial/parallel summary fingerprints 全部一致
- 正式生成继续使用 CPU；GPU 加速比记录为 unavailable，而不是伪造结果

### M4：第一批题包 — 按实际容量完成

- architecture 和 mixed 均达到 10 题目标
- optimizer 当前只有 3 题：已有候选池存在 403 个显著 pair，但只能形成 3 个稳定全局不复用 pair
- v2.1 KAN 保持冻结并保留 5 题；v2.2 新建扩展 profile 后新增 7 题
- v2.2 KAN pool：24/24 Gate 1/2 pass，7/7 Gate 3/4 pass
- 未降低 significance gate，也未修改 v1/v2/v2.1

### M5：完整 pilot — 本轮容量截止，缺口已验证

- loss-only：状态明确为 `deferred`，本轮不生成正式题目；1024 和 2048 两个预算各有 7 个 Gate 1/2 pass candidate，但显著 pair 均为 0。它是 coverage gap，不是已完成题包，也不写成 `excluded` 题目记录。
- architecture_easy / architecture_hard：本轮不再扩容，当前题数均为 0；后续若继续扩展，应新建候选/profile 轨道，不回写冻结 profile
- 人工盲审和 Luna blind/post-reveal audit 尚未完成；Luna provider 额度不足时不伪造审计记录
- 因此当前是“内部审计 pilot 候选”，不是 70 题完整 pilot

### M6：内部审计候选 — 代码/静态/API 完成，人工审计待执行

- React `npm ci --ignore-scripts` 成功
- React `tsc -b` 成功
- React `npm run build` 成功（Vite 6.4.3）
- telemetry API validation smoke 通过
- collection API smoke 通过
- Streamlit Inspector、旧 static exporter 和 React surface 保持并存
- 本轮范围为本机可信研究人员审计；为保持静态页面可直接答题，BakeFile 仍包含 reveal，但答题前 UI 不渲染答案。
- 面向外部部署的 BakeFile 分离与 API-only reveal 暂列为后续加固项，不作为本轮本地审计阻塞。


## 2026-07-22 独立验收补记

独立 collection/前端顺序审计命令：

```text
python tools/audit_demo_collection.py --collection outputs/demo_release_integration/demo_release_collection_v1.json --output outputs/demo_release_integration/demo_collection_audit.json
```

结果：`collection_id=demo_d151466cdd4b`，`question_count=39`，`candidate_count=78`，`frontend_order_match=true`，`valid=true`。

本次 focused pytest 重跑结果为 `2 passed`；另有 4 个测试在 Windows 临时目录 setup 阶段因 `WinError 5`（temp ACL）未进入断言，记录为环境权限问题，不将其误报为题目逻辑失败。

### 2026-07-22 46 题收尾验收

- `demo_release_collection_v2.json`：46 questions、92 candidates。
- question type：`architecture_only=24`、`mixed=12`、`optimizer_only=10`；没有 `architecture_easy` 或 `architecture_hard` 题。
- 独立 collection audit：`frontend_order_match=true`、`candidate_reuse_count=0`、`valid=true`。
- 默认 React BakeFile 已切换为 v2；旧 39 题 BakeFile 保留为 `questions_v1.json`。
- `RELEASE_FREEZE_MANIFEST.json` 已更新为 v2；旧 39 题 manifest 保留为 `RELEASE_FREEZE_MANIFEST_v1.json`。
## 当前冻结入口

- collection：[demo_release_collection_v2.json](demo_release_collection_v2.json)（v1 历史版本仍保留）
- React source：[frontend/quiz](../../frontend/quiz)
- React production output：[frontend/quiz/dist](../../frontend/quiz/dist)
- 静态 Inspector：[static_demo_release](static_demo_release)
- 容量证据：[capacity_report_release_final3](capacity_report_release_final3)

## 后续唯一必要动作

1. 用户按 collection 顺序盲做，记录 answer、confidence、用时和 keep/revise/reject。
2. 人工结果冻结后再运行 Luna blind 与 post-reveal audit。
3. 若要达到 70 题上限，应新建后续 candidate/profile 扩容，不回写冻结 profile，也不降低 gate。
