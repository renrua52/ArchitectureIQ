# ArchitectureIQ 产品开发记录

最后更新：2026-07-12

当前里程碑：M1 — 可回流的内部答题 MVP

## 记录边界

这里维护做题网站的动态优先级、产品决策、验收标准和已交付变化。稳定的
benchmark 流水线合同见 [`AGENTS.md`](../AGENTS.md)，设计原理见
[`plan-v2.md`](../plan-v2.md)，已经可用的功能和命令见
[`README.md`](../README.md)。

状态统一使用：`讨论中`、`待开发`、`进行中`、`阻塞`、`已完成`、`暂不做`。
Backlog 条目只更新状态，不重复创建；完成后移入“已交付”。

除非条目明确给出 hosted 验收证据，本文的“已交付”只表示当前本地工作树已实现并
通过相应本地检查，不表示已经推送、部署或在真实网络服务上验收。当前所有新增的
回流、Reports 与 runtime release attestation 能力都尚未完成真实 hosted 验收。
本轮 Surprise/Recommendation/Reports 收口后，全仓为
`1078 passed in 152.47s`；41 个相关 Python 文件 Ruff、format 和全工作树
`git diff --check` 均通过。全仓 Ruff 审计仍有与本轮无关的既有债，未擅自修改共享
工作树中的其他开发内容。
`report-app` preflight 的精确输入 fingerprint 为
`4d3c6bfe6649c96334605c83856e691f167de0a1e60be7845857394475cd85f5`：全部业务和
安全合同 PASS，但 Git tracked/clean/match-HEAD 三项按真实 dirty/untracked 工作树
FAIL，因此 `static_overall=FAIL`、`overall=FAIL`；hosted acceptance 仍为
UNVERIFIED，`deploy_ready=false`。本地合同检查不替代真实 PostgreSQL 或 hosted 验收。

## 当前线上与仓库的关系

- 线上站点：<https://architecture-iq.streamlit.app/>。
- Git remote 是 `renrua52/ArchitectureIQ`；仓库声明的 Community Cloud 入口是
  `tools/question_inspector/app.py`。远端 `main` 与本地分支基点当前同为
  `034aaad7bde1cfb6dd0dd73dcaea22f5c15be0d0`，但回流、Reports、manifest 和
  registry 都仍是未提交/dirty 工作树内容，因此不可能已经来自远端 `main`。
- 2026-07-12 最新匿名浏览器复查可直接载入站点，页面标题为
  `ArchitectureIQ Question Inspector`，显示 60 道题、Question/Prompt 页签和
  Add custom setting；页面没有本地新增的 `Upload pending session events` 或
  `Upload comment`。Streamlit creator 链接与 Git remote owner 都是 `renrua52`，
  因此可以确认线上是该项目的旧版 inspector；但页面未暴露 runtime Git SHA，仍不能
  仅凭 UI 断言精确 source commit。
- 具体 repo、branch、entry、deploy ID 和 source commit 仍只保存在 Streamlit Cloud
  控制台，仓库没有 `.streamlit` 或部署 workflow；在管理员补充控制台证据前不猜测。
- `data/` 是本地生成目录且被 gitignore。本地新生成的题不会自动上线。当前工作树的
  no-argument inspector 直接读取受版本控制的 `examples/quiz_demo/bundle/`，不再复制
  到可写 `data/` 后读取，并在展示题目前执行完整 release attestation；当前访问门控后的
  实际线上 revision 尚未得到登录态证明。
- 当前题库包含 3 个 family、每个 20 题，共 60 题。当前 bundle 的
  `quiz_manifest.json` 记录了 60 个 question version、3 个 source run 和全部
  1483 个 artifact（4,039,296 bytes）哈希；manifest 文件 SHA-256 是
  `9fa3c9e28aa81dffd7ea751be40245d1f62f01c252b91024e62de0d8bb230005`，其内容寻址标识是
  `release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f`。
- `docs/STATUS_AND_PLAN.md` 盘点的 261 道题位于 gitignored 本地 `data/`，
  **不等于当前 attested 60 题 release**。任何线上推荐都只能从 runtime
  manifest 验证过的当前 release 选题；本地题先经 canonical
  publisher/manifest/registry 发布后才能进入该池。
- 每道题依赖完整 dataset、candidate set、GT、prompt、run 和 question
  artifact，不能只上传一个 `question.json`。

### 维护者发布新题

1. 使用标准 CLI 完成 `create-dataset` → `generate-candidates` →
   `generate-question`，GT 必须来自执行生成代码；不要手改 `question.json`。
2. 本地用 inspector 检查新 question 或 question run。
3. 先 dry-run。参数路径相对于默认 `--data-root data`，因此要从
   `datasets/...` 开始：

   ```bash
   .venv/bin/python tools/publish_quiz_bundle.py --dry-run \
     datasets/univariate_regression/sym_XXXXXX/questions/run_5q_3c_XXXXXX
   ```

4. 确认投影的 question 列表、artifact 数和 `release_id` 后，去掉
   `--dry-run` 发布到默认目标 `examples/quiz_demo/bundle/`：

   ```bash
   .venv/bin/python tools/publish_quiz_bundle.py \
     datasets/univariate_regression/sym_XXXXXX/questions/run_5q_3c_XXXXXX
   ```

   也可以把单个 `q_XXXXXX/` 作为 source；此时 `run.json` 保持原样，
   manifest 会用 `source_runs[].partial: true` 和 `selected_question_ids` 记录
   实际发布的题。
5. 从完整 attested bundle 生成服务端 registry JSON 与只含三组 `INSERT` 的 data
   migration。输出必须在 bundle 外；新 release 使用新的文件名与 migration timestamp：

   ```bash
   .venv/bin/python tools/export_feedback_registry.py \
     --bundle examples/quiz_demo/bundle \
     --json-output supabase/registries/release_<64hex>.json \
     --sql-output supabase/migrations/<timestamp>_feedback_question_registry_release_<prefix>.sql
   ```

   Exporter 会同时执行 runtime artifact attestation 与 publisher GT validation；不能
   从单个 `question.json` 或未验证 manifest 直接生成答案库。`registry_id` 绑定权威
   question/choice 内容，manifest SHA 只作为 provenance。
6. 检查 bundle、manifest、registry JSON/data migration 的 diff，运行发布与 registry
   测试，并用 `--check` 做字节级复验，再一起提交：

   ```bash
   .venv/bin/pytest -q tests/test_quiz_bundle_publish.py tests/test_feedback_registry.py
   git diff --stat -- examples/quiz_demo/bundle
   ```

7. 首次部署 STATS-003 时，数据库先应用 `14000` registry schema migration，
   再应用该 release 的
   timestamped insert-only data migration（当前 release 为 `14500`），再用
   `15000` 切换 authoritative aggregate Reports，最后用 `16000` 加入
   authoritative Answers/Proposals 和双 revision status，再用 `17000` 加入六业务页
   单 SQL/MVCC snapshot，并部署匹配的
   migration/client/app；不把 registry 写权限授予
   持有 service-role key 的 feedback Edge。随后推送到 Streamlit Cloud 实际绑定的
   branch，等待重新部署，必要时在
   Cloud 控制台 reboot。发布器只更新 Git 中的 bundle，不会自动上传到网络或
   触发 Streamlit 部署。已完成该切换的数据库，后续 release 只应用新的
   reviewed data migration，不重放 `14000/15000/16000/17000`。这些 migration 目前
   只是本地部署输入，尚未在 hosted PostgreSQL 上应用或验收。

发布器仅接受 canonical pipeline 已生成的 artifact。它会拒绝缺文件、路径越界、
symlink、无效引用、GT 已排除的 choice、重复 question ID 和同路径不同内容。
目标 bundle 是只追加的：不支持替换已发布的 question ID；修改题目时应从标准
pipeline 生成新题并以新 ID 发布。公共网站仍不接受任意 `.py` 或未审核题包。

### 发布 smoke validation

QPUB-003 已接入 dry-run、正式发布、manifest refresh 和直接 manifest build。
它不重跑训练，而是在发布前检查现有 canonical artifact 的内部一致性：

- candidate budget 均为正整数且满足
  `training_steps * batch_size == total_samples_seen`；set 的 count/budget 与完整
  candidate 内容一致；question 的单一或 mixed budget 与 choices 一致；
- `dataset_spec`、question evaluation/significance 和 summary 使用同一 selection
  metric；summary 的 candidate ID、`execution="candidate_py_files"`、seed 配置与
  顺序、失败数、excluded 状态和动态 mean/std/final metric 字段完整一致；
- `correct_letter` 必须等于按 summary mean 和 metric 方向得到的唯一 winner；
  significance gap 必须等于 winner 与 runner-up 的 mean 差，win rate 必须是
  `[0, 1]` 内的有限数；
- prompt 会扫描 GT 结构标记以及常见的显式答案、结果标题和 mean/std/final
  数值泄漏；非 `test_mse` 题目若仍宣称选择 “best test MSE” 也会拒绝。
- choice 的 GT 必须 `failed_seeds == 0` 且 `excluded == false`；不允许用部分成功
  seed 的均值冒充题面声明的完整 seed 统计。
- 所有被发布器读取的 JSON 必须是严格 RFC 8259 JSON；`NaN`、`Infinity` 和
  `-Infinity` 会 fail closed。完全失败候选的 aggregate mean/std 使用 JSON `null`。

历史 metric `test_mse`、`test_ce` 允许省略方向，均按 lower-is-better 解释。
新增 metric 必须同时在 `dataset_spec.json` 和
`question.json.evaluation.higher_is_better` 声明相同的 boolean；只声明一侧或两侧
冲突都会拒绝发布。prompt 扫描是 smoke/启发式检查，模板或 metric 文案变化后仍需
维护者人工检查，不能把测试通过理解为已经证明无泄漏。

### Manifest 与 release

- `quiz_manifest.json` 是当前 bundle 快照的索引，不是另一份题库。它记录
  source run、question ID 及 version、family/dataset/path、数量，以及每个发布
  artifact 的 path、size 和 SHA-256。
- `question_version` 是 canonical `question.json` 的稳定哈希，与回流事件使用
  同一算法，使历史答案可对应到作答当时的题目版本。
- `release_id` 是整个 manifest 内容核心的 SHA-256。只要题目、引用关系或
  任一 artifact 内容/路径改变，release 就改变。它是逻辑上的内容快照，
  不是单独目录、Git tag 或部署动作；旧 release 由 Git 历史保留。
- 可选 `generated_at` 只是描述元数据，不参与 `release_id` 计算。当前
  manifest 也不写 Git commit；部署溯源以提交 bundle/manifest 的 Git commit 为准。

### Runtime Release Attestation

当前工作树的默认 bundled root 在建立题池前会重新读取文件字节并完成整包证明：严格
校验 manifest schema，重算 release core，核对 source runs、questions、counts、固定
entry path，以及排序后一一对应的物理 artifact inventory；每个 artifact 都核对路径、
类型、size 和 SHA-256，拒绝 symlink/special file，并严格解析每个 `question.json` 后
重算 question version。该过程不把 manifest 或 artifact 的 mtime 当缓存凭据，因此
same-size/same-mtime 篡改仍会被发现。

默认 `examples/quiz_demo/bundle/` 缺少或含无效 manifest 时 fail closed，不会退回旧题池
或把内容误标成当前 release。只有维护者显式选择的非默认本地开发 root 可以在完全没有
manifest 时以 unversioned 模式浏览；一旦该 root 存在 manifest，也必须通过同样校验。
UI 显示完整 release ID、manifest SHA-256、已验证 artifact 数、固定 Streamlit entry，
以及从实际 checkout 读取、再与 allowlisted 环境声明交叉核对的完整 40-hex Git SHA；
checkout 不可用、任一已配置声明格式错误、声明冲突或 Git 结果无效时显示 `N/A`，不猜测
branch、deploy ID 或 deploy time。以上均是本地实现证据；当前匿名站点仍是
缺少这些 release/feedback 控件的旧 UI，尚未完成新版本部署与 runtime 验收。

### 部署后追溯账本

部署状态不能由正在被部署的 commit 自己完整声明：provider deploy ID 和真实上线时间
只能在部署发生后获得，而把它们再写回同一 commit 会产生新的 source commit。因此
`tools/deployment_ledger.py` 使用部署后的 retrospective JSONL 审计链，不把
`deployments/ledger.jsonl` 或 post-deploy evidence 纳入该部署的输入 fingerprint，也不
创建“当前部署”占位记录。缺少账本等价于尚无已记录部署，不等价于失败或已经上线。

单个 deployment key 按
`candidate_attested → deployment_declared → postgres_accepted / roundtrip_accepted /
source_mapping_attested → activated → superseded | rolled_back` 演进。三类 evidence 可按
任意顺序追加，但必须绑定同一 release、source commit、manifest/registry、provider
deploy、backend project 与 ingest/report origin 组成的 deployment context，并禁止跨
deployment 重用 evidence hash、run/request/export identity。candidate 会从声明的 Git
commit blob 重算 rollout fingerprint，而不是信任 dirty working tree。

云平台映射仍有明确的信任边界：本地 JSON 不能凭空成为 provider proof。
`source_mapping_attested` 只接受 hash-bound 的原始 control-plane capture，并要求独立
review；最终状态使用 `ACTIVATED_REVIEWED`，不称为 provider-verified。账本 hash chain
可发现中间记录的修改、重排或删除，但无法单独发现尾部截断；最后一条 record hash
还必须由 Git history 或外部审计记录固定。当前仓库没有真实 staging/provider evidence，
因此没有创建 ledger event，也没有任何 confirmed deployment。

### Release 追溯记录

| Release | Source commit / 状态 | 追溯结论 |
|---|---|---|
| `release_bec00e86071f939e1153b7b5402961388bedf37483ce2e41b6505add58792831` | `e29d6d78432a2df9ce0beaaa92abbd6d767bd77b` | 审计恢复出的旧 release → source commit 映射；该映射未写入 manifest，是需要用部署记录/tag/映射表固化的追溯债。最新页面只证明线上仍是旧版 60 题 UI，不能据此证明其精确 release/source commit。 |
| `release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f` | 当前新 bundle，尚未部署 | 已通过独立 canonical bundle/manifest 审计；source commit 要等正式提交后记录，不能用旧 commit 代替。 |

## 产品决策

| ID | 日期 | 决策 | 原因与影响 |
|---|---|---|---|
| DEC-001 | 2026-07-11 | 区分“发布题目”和“上传回流数据” | 前者是维护者审核后的题库发布；后者是答题者主动提交自己的足迹或评论。UI、权限、存储和安全模型完全分开。 |
| DEC-002 | 2026-07-11 | 批量足迹和单题评论共用版本化、append-only 事件模型 | 评论是单个事件，整段足迹是事件数组；复用幂等、鉴权、限流和统计入口。 |
| DEC-003 | 2026-07-11 | 回流数据不写入题目 artifact，也不依赖 Streamlit 本地磁盘 | 题目和 GT 保持只读；Community Cloud 本地文件会随重启丢失，正式回流必须进入外部持久化服务。 |
| DEC-004 | 2026-07-11 | 当前按内部开发者工具设计，不建设复杂隐私/consent 流程 | 站点已有内部 access；仍不上传 secrets、本地路径、异常堆栈或任意代码，以保证数据可用性和系统安全。 |
| DEC-005 | 2026-07-11 | 参考 Kahoot 的报告模型，自建 ArchitectureIQ 网站 | 保留 dataset、代码、learning curve 和 propose setting 等品牌能力；当前是异步自助答题，不实现 Kahoot 的 PIN 房间、倒计时、排行榜或 WebSocket。 |
| DEC-006 | 2026-07-11 | GitHub 负责代码和题库分发，独立网络服务负责用户数据 | Git 仓库和 Streamlit 本地磁盘都不适合作为并发写入的数据接收端；首选 Supabase Edge Function + Postgres，也保留兼容其他 HTTP endpoint 的客户端。 |
| DEC-007 | 2026-07-12 | 题库 release 使用 bundle 内容寻址，发布器只追加 canonical artifact | `release_id` 由 question/version、run、counts 和全部 artifact 哈希决定；`generated_at` 和 Git commit 不参与内容标识。不允许覆盖旧 question ID，避免历史答案被静默重解释。 |
| DEC-008 | 2026-07-12 | Reports 使用独立且平台级受限的私有应用 | report token 只认证 Streamlit 到 Edge Function，不能识别网页访问者；Reports app 没有内建登录，因此匿名可访问的 hosted URL 不通过上线验收。 |
| DEC-009 | 2026-07-12 | 发布题目对 seed 完整性和 JSON 合法性 fail closed | 任一 choice 有 failed seed 即拒绝；完全失败 aggregate 用 `null`，通用 JSON writer 禁止非有限常量。发布包必须来自标准 GT/renderer/publisher，不手改 artifact。 |
| DEC-010 | 2026-07-12 | Bundle 只接受 canonical artifact allowlist | question 只发布 `question.json`/rendered prompt，candidate 只发布规范声明的代码/spec/results，set/run/dataset 同理；`custom_settings`、`__pycache__` 等用户或运行时文件不得进入 release，manifest build 遇到额外文件 fail closed。 |
| DEC-011 | 2026-07-12 | 服务端按 7 字段 logical event 区分幂等重试和 event-ID 内容冲突 | `schema_version/event_id/event_type/session_id/question_id/question_version/payload` 使用 PostgreSQL JSONB equality；`occurred_at`、`sequence` 和传输元数据不参与身份。相同内容重试返回 200；不同内容复用同一 ID 返回 409，first-write-wins 且整批原子不写。 |
| DEC-012 | 2026-07-12 | 现有回流服务采用四阶段、可指纹追溯的兼容升级 | `expand → ingest-cutover → lockdown-report → report-app` 逐段推进；后阶段累积检查前阶段输入。`lockdown-report` 按 `12500 → 13000 → 13500 → 14000 → 14500 → 15000 → 16000 → 17000 → 18000 → 19000 → 20000` 应用；`19000` 必须先于 presentation/reaction-emitting Inspector 暴露，`20000` 再添加 surprise Reports。`13000` 的 ingestion schema 与 client、`14000–20000` 的 registry/report/filter/reaction SQL 与严格 client/app 都要协调。不能对运行中旧 writer/client 无协调执行全部 pending migration；本地 preflight 永远不冒充 hosted 验收。 |
| DEC-013 | 2026-07-12 | Upload 必须同时配置 endpoint 与 Bearer token，并只接受完整严格回执 | 缺任一配置时仍记录和下载足迹，但不触网也不宣称成功。只有四个非负整数 counter、总数等于本批事件数、conflict/rejected 均为零，且 header/body 为同一个 canonical RFC UUID 时才把事件标为 acknowledged；generic、partial、malformed 或不匹配的 2xx 保持 pending。 |
| DEC-014 | 2026-07-12 | 批量 Upload 只发送 pending 事件并遵守 receiver 分批上限；409 逐事件隔离 | 每批最多 500 events / 1 MiB。整批 409 后逐事件重试，确认发生 logical-content conflict 的 ID 进入 quarantine，其他 withheld/new 事件继续按原 ID 上传；网络或可重试错误继续 pending，避免一个坏 ID 永久阻塞后续足迹。 |
| DEC-015 | 2026-07-12 | 默认题库必须通过 Runtime Release Attestation | 默认 bundle 的 manifest 缺失、无效或任一 artifact 不匹配都 fail closed；只有显式非默认本地 root 在完全没有 manifest 时允许 unversioned 开发浏览，不能附带 release ID。 |
| DEC-016 | 2026-07-12 | 回流事件只接受 Python、JavaScript 与 PostgreSQL 可无损共享的 JSON 子集 | recursive integer-valued JSON number 限于 ±(2^53−1)，字符串拒绝未配对 Unicode surrogate；identifier/comment 长度统一按 Unicode code point。Python 在触网前拒绝，Edge 在调用 RPC 前再次拒绝，避免 200 receipt 对应的 payload 已被 JavaScript 静默舍入。 |
| DEC-017 | 2026-07-12 | 服务端题库 registry 上线前，不得把客户端自报正确率/维度称为权威；切换后绝不 fallback | Event payload 的 `is_correct` 和维度只作 raw audit。Authoritative Reports 根据 registered release membership 与所选 choice 派生；无 registry 或无法匹配时显示 N/A/quality exclusion，而不是回退客户端值。 |
| DEC-018 | 2026-07-12 | Registry attribution 在 report time 动态解析，不写回 raw event | 旧事件可能先于 release registry 到达；动态精确 join 让后续登记自动覆盖历史数据，同时保持 `feedback_events` 与 logical-event 幂等合同不变。缺 release 不猜测，同一题可属于多个 release；registry 只能 owner-reviewed insert，service-role 只读。 |
| DEC-019 | 2026-07-12 | REPORT-002 明细与 aggregate 共用 registry authority，并用独立 detail revision fail closed | Answers/Proposals 只读取 matched projection，不从 payload 回退维度/正确性；`16000` 与两明细 RPC 同事务重建七列 status。Verifier 直接检查该 status；UI 由 snapshot 内嵌同一 `registry_v1/detail_v1` authority facts，不再单独预取 status。`18000` 再为六个业务 RPC 和 snapshot 统一追加服务端权威 session/attempt drilldown。 |
| DEC-020 | 2026-07-12 | 六个业务报告必须来自单 SQL statement/MVCC snapshot | `17000` 通过一次 App GET → 一次 Edge/PostgREST RPC 返回 `business_snapshot_v1`、server `snapshot_at`、内嵌 authority/counts 与严格 `pages_json` text。完整校验后才同时替换六页；失败保留上一完整快照。Ingestion observability 与 Registry quality 仍为独立、非原子质量请求，不冒充同一 cutoff。 |
| DEC-021 | 2026-07-12 | 原子业务快照按完整行稳定前缀做字节预算，不裁字段 | 六页保留精确 `total` 和请求 `limit`，但只有排序前 N 行进入 JSON/字节计算，再按每页预算截取完整前缀；`pages_json` 最终不超过 4 MiB，Edge 原样转发已验证的 PostgREST snapshot JSON 以保留 bigint，Python 仍有 10 MiB 外层上限。截页显示 `shown of total` 并禁用 CSV；exact total 仍明确需要 O(N) 数据库扫描。 |
| DEC-022 | 2026-07-12 | Hosted 验收拆成数据库证据和应用路径证据 | 先用 staging owner/admin DSN 运行 rollback-only PostgreSQL verifier，核对 migration catalog、应用函数属性/ACL、forced RLS、table grants、精确 trigger mask、全部稳定命名约束，并从 60/180 个 hosted registry rows 重算 `registry_id`；再运行会留下永久事件的 Edge write/read roundtrip。前者不证明 Edge/restart/concurrency/load，后者不替代 catalog/RLS 证据，两份输出都必须绑定同一 Git SHA/preflight fingerprint。 |
| DEC-023 | 2026-07-12 | 部署账本是部署后的回顾性审计，不是应用自报的当前状态 | deploy ID/time 在部署后才存在，写回同一 commit 会产生自引用悖论；因此只把 ledger parser/tool/docs 纳入 preflight，实际 JSONL/evidence 不参与部署输入 fingerprint。没有真实事件时不创建 placeholder。 |
| DEC-024 | 2026-07-12 | 激活只表示共同 deployment context 下的证据已由维护者复核 | PostgreSQL、hosted roundtrip、source mapping 必须绑定同一 release/source/provider/backend/origin context，且 evidence identity 不得跨部署复用。可手写 JSON 永远不冒充 provider 密码学证明，状态明确为 `ACTIVATED_REVIEWED`；hash-chain 末端另需 Git 或外部 head pin。 |
| DEC-025 | 2026-07-12 | Session/attempt 是六业务页的全局权威筛选，不扩展辅助视图语义 | `18000` 在保留旧 positional 参数前缀的前提下，把 `session_id`/`attempt_id` 追加到六业务 RPC 与 snapshot；两个值同时存在时按 AND 精确筛选。Ingestion、Registry quality、authority status 与 exact-event resolution 不接受 identity filter，避免把全局/单事件辅助证据错误归因到一次 attempt。 |
| DEC-026 | 2026-07-12 | 惊讶是 reveal 后的结构化 reaction，与正确性、comment 和 like 分离 | 统一使用 `intrinsic_surprise_proxy`、`observed_surprise_rate`、`predicted_personal_surprise` 三层术语。**惊讶 ≠ 答错 ≠ 点赞**；answer correctness 只作分析上下文，自由文本 comment 不作结构化标签。用户只能在 commit answer 并看到真实结果后选“有惊讶 / 没有惊讶”；like/继续行为如需优化则独立采集。 |
| DEC-027 | 2026-07-12 | 题目 validity 是不可降级的硬门，动态惊讶分数不进入 canonical release | 只有 attested release 中 significance/seed 合同通过的题才能参与排序；用户反馈再高也不能救回无效题。`observed_surprise_rate` 与 `predicted_personal_surprise` 保存在外部 store/版本化 policy snapshot，只改变展示顺序；不回写 `question.json`、GT、candidate summary 或 prompt，避免内容身份漂移和指标泄漏。 |
| DEC-028 | 2026-07-12 | 推荐只在 runtime-attested 当前 release 内运行，且每次决策必须可评估 | 当前本地策略记录 `question_presented`、policy version、decision ID、source/position 和 selection probability/propensity，不用 answer 冒充曝光。候选排除本 attempt 已答题、保留探索与 family 多样性；catalog/策略不可用时 fail safe 到原顺序 `Next`。无 hosted persistence、propensity 分析和预先声明的 A/B 证据不宣称提升。 |
| DEC-029 | 2026-07-12 | 第一版 Next 使用私有离线 cold-start catalog，不扫描开发题库 | Catalog 只遍历 attested manifest 的 60 个 question dirs，严格读取已存 question/spec/summary，不重训、不写回 artifact。Validity 是硬门；参数量/depth/width/optimizer shortcut 并列均分、all-equal 不计。当前 Next 使用该 prior 与本 attempt 本地曝光，不读取 SURPRISE-002 或声称跨 session 个性化。 |
| DEC-030 | 2026-07-12 | SURPRISE-002 只统计第一条有效 post-answer reaction，并保持独立质量守恒 | 同 session/attempt/release/question/version 必须先有权威 answer；第一条合法 reaction 进入 yes/no，后续进入 duplicate，registry mismatch、非法 payload、缺 prior answer 进入互斥 orphan。Question 与 quality 是两个独立 SQL/RPC 请求，不冒充 `business_snapshot_v1` 的同一 MVCC cutoff。 |

## 参考 Kahoot 后的产品模型

Kahoot 把“题目本身”和“一次实际作答”分开，并围绕 game session 提供
Summary、Participants、Questions、Feedback 和 spreadsheet/API 报告。ArchitectureIQ
采用相同的核心分层，但不复制目前不需要的实时课堂功能：

参考：Kahoot 官方的 [quiz reports](https://support.kahoot.com/hc/en-us/articles/360035063054-Kahoot-quiz-reports)
与 [reports API guide](https://support.kahoot.com/hc/en-us/articles/11735948502931)。

| Kahoot 概念 | ArchitectureIQ 对应 | 当前用途 |
|---|---|---|
| Kahoot + version | 题库 `release_id` + `question_version` | 题目修改后，历史答案仍能关联到当时版本 |
| Game session | `session_id` / 一次做题 session | 批量上传和统计的基本单位 |
| Participant | 当前内部开发者 session；以后可选账号 | 区分一次参与，无需先建设用户系统 |
| Answer | `answer_submitted` event | 保存选择的 letter/candidate 和时间 |
| Open response / feedback | `comment_submitted` event | 当前题单条消息回流 |
| Post-answer reaction | `question_reaction_submitted` event（本地已实现） | reveal 后独立记录 surprise yes/no；不从答错或 comment 推断 |
| Question exposure | `question_presented` event（本地已实现） | 记录 policy/decision/propensity，为 continuation 和 A/B 提供分母 |
| Report API | 私有回流 API + 统计页 | 按 session、题目、family、类型聚合 |

推荐的最小线上结构：

```text
GitHub (代码 + curated 题库 bundle)
        │ deploy
        ▼
Streamlit Cloud (做题 UI)
        │ HTTPS: 单事件 / 整段 session
        ▼
Ingestion endpoint (鉴权、校验、幂等)
        │ atomic RPC: exact retry / content conflict / all-or-none insert
        ▼
Postgres/Supabase ──► 私有 Reports 页面 / SQL / CSV
        ▲
        │ owner-reviewed registry data migration（题库答案/维度，只读给 service role）
        └── attested Git bundle
```

Streamlit 继续负责“分发”。需要新增的网络服务不是另一套前端平台，而是一个小型
写入 API 和可查询数据库。当前推荐 Supabase，是因为一张 JSONB 事件表即可启动，
后续还能增加 SQL 聚合、Edge Function、内部鉴权和 dashboard。

本地部署合同中，`14000` 定义私有 release/question/choice registry 和动态权威
投影，`14500` 登记当前 60 题 release，`15000` 在不改四个业务 RPC schema 的
前提下把它们切换到该投影，`16000` 新增 Answers/Proposals 两个权威
明细 RPC，并把 status 收口为 `registry_v1/detail_v1` 七列合同；`17000` 再增加
`business_snapshot_v1`，用一条 SQL statement 返回六个业务页；`18000` 增加
session/attempt 筛选；`19000` 严格接收 presentation/reaction；`20000` 添加两个
SURPRISE-002 RPC。独立内部入口
`tools/feedback_reports/app.py` 的本地实现已经具备 Summary、
Sessions、Questions、Answers、Proposals、Comments 六个业务 tab，以及
**Ingestion observability**、**Registry quality**、**Surprise**、**Data quality**
共十个 tab，
并通过专用
read token 调用受保护的 `feedback-report` Edge
Function。业务视图先按 client `release_id` claim + 顶层 question ID/version 精确关联
server registry，再只用权威 release/family/dataset/type 和派生 correctness 聚合；不匹配
raw events 不进入业务筛选或正确率。ingestion 页签只发送 UTC `from`/`to`，按服务端
请求 `started_at` 筛选；registry quality 也只用 event-time `from`/`to`，避免用未知
事件的自报内容筛选。受保护 client 另允许仅 ingestion summary 使用严格 request UUID、
仅 verifier/operator exact-event view 使用 event ID，UI 都不暴露。`18000` 已在本地为
六业务 RPC 与 snapshot 追加全局 `session_id`/`attempt_id` 精确筛选；旧 positional 前缀
保持兼容。内容或 identity 筛选生效时不会请求两个全局质量视图；authority status
不接受任何筛选，exact-event 只接受 event ID。Surprise question/quality 使用同一组
八个 identity/time filter，但各自独立请求，也不属于六视图业务快照；其失败不会覆盖
六视图业务快照。截断页不能导出
CSV。该链路尚未部署到 hosted Supabase 或私有 Streamlit，因此
REPORT-001/REPORT-002/STATS-001 仍是进行中，不能据此声称线上数据已持久化或可查询。

当前默认 bundle 已通过 runtime attestation，含 60 个题目、1483 个 artifact；私有
`surprise_catalog` 只遍历这 60 个 manifest question dirs，逐题严格读取
`question.json`、choice `candidate_spec.json` 与 `results/summary.json`。它以
correct/significance/failed/excluded 为 validity 硬门，用 tie-aware 参数量、depth、width、
optimizer shortcuts 构造 cold-start Beta prior；不扫描本地 261 题全集、不执行训练，也不
把答案、GT 或分数写回 artifact。Inspector 的 `Next` 已接默认 20% ε-greedy：exploit
最高 prior，explore 当前 attempt 最低曝光池，排除当前/已答题并尽量切换 family；异常时
回退顺序 Next。每次发布题导航在本地 trace/browser outbox 记录
`question_presented` 的 decision/policy/mode/propensity/source/position。当前还不读取
SURPRISE-002 的远端后验，因此不称为跨 session 个性化推荐。

`20000` 的本地 SURPRISE-002 question RPC 只统计同一精确 identity 下、权威 answer
之后的第一条合法 reaction，返回 answered attempts、yes/no/rating、coverage、
`observed_surprise_rate` 与 Beta(1,1) `posterior_mean`；quality RPC 将所有 raw reaction
守恒拆成 valid/orphan/duplicate，并把 orphan 拆为 registry-unmatched、invalid payload、
missing prior answer。两者和本地 Surprise tab/严格 client/Edge allowlist 已接线，但未在
真实 hosted PostgreSQL/Supabase/私有 Streamlit 验收。

REPORT-002 的 Answers 与 Proposals 只返回 registry-matched events，共享六业务页的
权威 release/family/type/question 与 event-time 筛选。Answer 行同时显示 client choice/
correctness、canonical candidate、server-derived result 与 mismatch；Proposal 行覆盖
proposed/rejected setting，并对 setting/inheritance JSON、nullable int32 seeds 和 error type
做严格 schema/安全展示。两页都遵守 1,000-row 上限与 complete-page-only CSV。UI 在
一次 refresh 中只发送一个 business-snapshot GET；Edge 只调用一次 PostgREST，`18000`
重建 `17000` snapshot 并把 identity filters 透传到单 SQL statement/MVCC snapshot 内的 Summary/Sessions/Questions/Answers/Proposals/
Comments。返回行带 server `snapshot_at`，并内嵌 `registry_v1`/`detail_v1` authority facts、
registry counts 和严格 `pages_json` text。Client 校验完整六页、共同 limit/zero offset 与
跨页计数守恒后，App 才原子替换六页和 metadata；失败刷新继续显示上一完整快照，
不会混用新旧页。Ingestion observability 与 Registry quality 仍各自独立请求，不共享该
MVCC cutoff；内容或 identity 筛选时跳过。该设计只完成本地实现/合同检查，未在真实 PostgreSQL
执行或验收。

业务 Summary 的端到端 ingestion failure rate 仍必须显示 N/A。OBS-001B/STATS-002B
已通过独立 RPC 和第五页签展示 `included_in_rate = true` 的已持久化 outcome 子集，
包括 recorded request failure rate，以及 verified idempotent、legacy unclassified
duplicate 和真实 content conflict 的拆分；零分母显示 N/A，且
`end_to_end_coverage_available` 固定为 false。401/405/缺配置只进安全日志，未到 Edge
的请求以及 outcome write 在 timeout/schema/HTTP/全库故障下的丢失也不在可靠覆盖内，
因此仍需 hosted Postgres、RLS/grant 与覆盖率验收。

当前 ingestion summary 的 SQL/Python 精确合同为 22 列，其中包含与所筛选 outcome
request UUID 关联的 `conflict_audit_event_count`。部署前可用
`tools/feedback_rollout_preflight.py` 对四个阶段做累积、只读静态检查，并记录 Git SHA
和所枚举 rollout/compatibility 输入的 SHA-256。既有 migration 进入 fingerprint 是为
了可复现检查，不表示要对现有数据库重放。工具不接受 credential、URL 或 project ref，
也不访问 Supabase；
即使静态合同全部通过，仍固定报告 `hosted_verified=false`、
`deploy_ready=false`。当前共享工作树的部署输入尚未全部纳入干净提交，因此检查结果
应为 FAIL，而不是可部署。

精确 hosted proof 的本地编排会先对完整 bundle 执行 runtime/registry 双重
attestation，选取实际 release 中的题目，并在任何永久 POST 前要求随机 event ID
的 exact event-resolution 返回 `not_found`。它还会在写入前严格验证 `16000`
重建的七列 `feedback_report_authority_status()`：`registry_v1`/
`business_reports_authoritative=true`、至少覆盖本地 attested bundle 的 registry
counts，以及 `detail_v1`/`detail_reports_authoritative=true`。Answers/Proposals 还必须
对随机不存在 question 返回 complete empty page，在不依赖历史行的情况下证明
`16000` RPC/Edge allowlist 可达；另一个随机不存在 question 必须在任何写入前通过
`17000` 返回 authority-attested `business_snapshot_v1`：Summary 为严格全零单行，其余
五页 complete empty，server `snapshot_at`、内嵌 revisions/counts 均有效。通过后结果
记录 `business_snapshot_verified=true`。默认流程随后覆盖单 comment
fresh/resume、同 session answer+proposal+comment 三事件首写 `3/0/0/0` 与原样重放
`0/3/0/0`，以及 changed-text/same-ID `409` first-write-wins。探针故意伪造
client family/type/正确性，exact resolution 必须返回 registry 中的 canonical
release/question/version/family/dataset/type 与派生 correctness，并标记 client mismatch。它还会在
每个明细事件的 `[occurred_at, occurred_at + 1 ms)` 窄窗口中逐页扫描，按 event ID
要求唯一 Answers/Proposals 行，并精确核对 answer authority/mismatch 与 proposal
setting/inheritance/seeds。legacy fixture 仍保留隔离的两事件路径，但不是当前 CLI hosted proof。
成功批次可见后，verifier 还要求真实 session+attempt 正向命中精确六页，并分别用错误
session/真实 attempt、真实 session/错误 attempt 得到全空六页；只有三项都通过才输出
`session_attempt_filters_verified=true`。

显式 `--include-mixed-batch-probe` 另验证被测 `new + conflict` 顺序的整批
all-or-none；withheld 新 event 的 exact resolution 仍必须为 `not_found`。这不证明
`duplicate + conflict`、任意并发或端到端 coverage，而且会永久留下 outcome/audit
footprint，应优先在 staging 运行并保存所有 request UUID。exact-event
证据由 `14000` 提供，单独不证明业务 RPC 已切换；七列 authority-status 同时
绑定 `15000` aggregate 与 `16000` detail cutover，空六页 snapshot 负对照证明受保护
`17000/18000` 路由，真实明细行及 identity 正负对照校验则证明 REPORT-002 语义。整套 verifier 目前仅
完成本地实现/合同测试，尚未对真实 hosted Supabase 运行，因此两类证据都
不能标记为 hosted PASS。

## Backlog

| ID | 模块 | 优先级 | 状态 | 用户结果 | 最小验收标准 | 依赖 |
|---|---|---:|---|---|---|---|
| TRACE-003 | 全部足迹上传 | P0 | 进行中 | 用户点击 Upload 提交当前 session 中仍 pending 的全部足迹 | 本地已完成 endpoint + Bearer 必需、严格四 counter/UUID 回执、500 events / 1 MiB 分批、pending-only outbox 和 409 逐事件 quarantine；仍需部署 Supabase 后验证真实持久化、重复上传、冲突隔离与重启后的统计 | STORE-001 |
| COMMENT-001 | 单题评论 | P0 | 进行中 | 用户可只对当前题提交一条消息 | 单事件 UI/回执/失败转 batch 已完成；等待正式 endpoint | STORE-001 |
| SURPRISE-001 | 单题惊讶 reaction | P0 | 进行中 | 用户在揭晓答案后点击“出乎意料 / 符合预期” | 本地已完成 reveal-only UI、严格 boolean `question_reaction_submitted`、按 session/attempt/release/question-version 稳定 event ID、trace/browser outbox/即时或批量上传、Edge 校验、`19000` forward migration 和合同测试；仍待真实 hosted apply、duplicate/conflict roundtrip 与跨重启验收 | TRACE-003、STORE-001 |
| SURPRISE-002 | 权威惊讶统计 | P1 | 进行中 | 维护者看到每题 yes/no/rating/coverage 和平滑后验，并审计 reaction 数据质量 | 本地 `20000`、13-view Edge/严格 client 与 Surprise tab 已完成：只聚合 registry-matched 且有 earlier answer 的第一条 exact-attempt reaction，返回 counts、`observed_surprise_rate`、Beta(1,1) posterior；quality 守恒拆分 raw/valid/orphan/duplicate。两个 RPC 独立于 business snapshot。仍待真实 hosted apply、行语义/ACL/RLS/筛选验收 | SURPRISE-001、STATS-003、REPORT-001 |
| STORE-001 | 回流接收服务 | P0 | 进行中 | 上传后数据跨 Streamlit 重启可靠保留 | SQL/Edge Function/鉴权/大小与批次限制、原子 RPC 与内容冲突拒绝已实现；本地 registry-aware verifier 默认覆盖正常 answer+proposal+comment 三事件首传/原样 duplicate、single conflict，并可显式覆盖 `new + conflict` mixed batch。等待真实 hosted 验收 | 维护者提供 Supabase project |
| DEPLOY-001 | 回流/Reports 分阶段上线 | P0 | 进行中 | 内部站点的 Upload、单题 comment 和私有统计使用同一可追溯 hosted store | 四阶段 cumulative preflight 已纳入完整 schema/registry/report/consumer 和 rollback-only PostgreSQL staging verifier。后者可直接验 migration/function/ACL/RLS/grant/trigger/constraint、从 hosted 60/180 rows 重算 registry hash 并执行无提交反例；仍待形成干净 commit、获得 staging admin DSN 并保存真实 PASS JSON，再运行 authority/exact-event/detail/snapshot 与多事件/duplicate/conflict endpoint proof | 维护者提供 Supabase project、OPEN-003 |
| STATS-001 | 私有统计后台 | P1 | 进行中 | 维护者可按 release/family/type/question 看使用情况与 Answers/Proposals 明细，并查看独立的接收链路观测与数据质量提示 | 本地六业务页只统计/展示 registry-matched events；正确率、release/family/dataset/type 均由服务端 registry 派生。Registry quality 保留 unknown/mismatch，绝不回退客户端值。待 hosted 数据验收；业务 Summary 的端到端 ingestion failure rate 继续 N/A | STORE-001、REPORT-001、REPORT-002、OBS-001、STATS-003 |
| STATS-003 | 服务端 release/question registry | P0 | 进行中 | 正确率和 release/family/type 由发布题库事实派生，不信任客户端自报 | 本地已完成双重 attestation exporter、60/180 registry、`14000–16000` authority、registry quality/exact-event，以及会从 hosted question/choice scalar rows 重建 canonical core 并核对 `registry_db3f1a...` 的 staging verifier。仍需真实 PostgreSQL apply/PASS JSON、并发和 endpoint verifier 验收，完成前不标“已完成” | QPUB-001、STORE-001 |
| REPORT-001 | Kahoot 式内部报告 | P1 | 进行中 | 开发者看到六个业务页、Ingestion observability、Registry quality、Surprise 和 Data quality 共十个页签 | 本地 13 个受保护 RPC/endpoint/严格 client/UI、安全 CSV、registry/outcome/surprise 质量信号和 exact request/event/snapshot verifier 已实现；待部署 hosted Supabase 与启用平台级维护者访问控制的私有 Streamlit，并用真实回流验收；匿名 Reports URL 不得上线 | STORE-001、维护者提供 Supabase project |
| REPORT-002 | Answers / Proposals 足迹明细 | P1 | 进行中 | 维护者可查看权威 answer 与实际 proposed/rejected setting 明细，并按 session/attempt drilldown | 本地已完成 `16000` 分页明细与 `18000` 六业务 RPC/snapshot 全局 session/attempt 筛选、严格 client/UI、安全 JSON/CSV；辅助视图在 identity filter 下明确不可用。仍待 hosted 正向命中、错误 session/attempt 空结果与 `session_attempt_filters_verified=true` 验收 | STATS-003、REPORT-001 |
| STATS-004 | 原子统计快照 / watermark | P1 | 进行中 | Summary 与五个业务明细页来自同一数据库快照 | 本地 `17000` 已实现一次 App GET → 一次 Edge/PostgREST → 单 SQL/MVCC snapshot；六页采用完整行 byte-bounded 稳定前缀、精确 totals、4 MiB `pages_json` cap，宽行只在 rank/limit gate 后转 JSON，外层 bigint 不经 JS Number；失败保留上一完整快照。仍需真实 PostgreSQL apply/catalog/ACL、并发写入/负载与 hosted verifier 的 empty-snapshot negative control/`business_snapshot_verified` 验收；两个 quality RPC 明确不在同一快照内 | REPORT-001 |
| OBS-001 | Ingestion request 可观测性 | P1 | 进行中 | 维护者能查看已持久化请求子集的成功、幂等重试、内容冲突、拒绝和服务失败比例及覆盖边界 | 私有 append-only outcome 事实层、service-role-only 聚合 RPC、server-time 口径、独立 Ingestion observability 页签和仅 verifier/operator 使用的 UUID 精确筛选已完成；待部署 hosted Postgres migration，并验证 forced RLS/grants、精确正负对照、真实 outcome coverage 与故障缺口。端到端 ingestion failure rate 继续 N/A | STORE-001、REPORT-001 |
| QPUB-008 | Release/source commit 追溯 | P1 | 进行中 | 每个实际部署的 release 都能定位 source commit、provider deploy、backend 与验收记录 | 本地已完成 retrospective append-only ledger、commit-blob fingerprint、runtime checkout SHA 和三类 hosted evidence 的共同 deployment context；状态只称 `ACTIVATED_REVIEWED`。当前无 staging/Cloud 凭据，未创建任何真实 ledger event；仍待干净 commit、Cloud control-plane capture、PostgreSQL PASS 与 hosted roundtrip PASS | QPUB-001、DEPLOY-001、OPEN-003 |
| PRIV-001 | 对外开放前的隐私策略 | P2 | 暂不做 | 如果未来开放公网，再补充采集说明与保留/删除策略 | 当前内部 access 不阻塞 M1 | 对外开放计划 |
| QPUB-004 | 外部贡献题目审核 | P2 | 待开发 | 外部贡献者可提交候选题包供维护者审核 | 隔离解包、schema/路径校验、人工审核，canonical pipeline 重渲染/必要时重跑 | QPUB-002、QPUB-003 |
| COMMENT-002 | 评论运营 | P2 | 待开发 | 评论可标注、处理和导出 | 状态/标签、脱敏、删除请求与审计 | COMMENT-001、STORE-001 |
| STATS-002 | 数据质量监控 | P2 | 进行中 | 维护者能发现基础回流质量问题，后续扩展到异常流量 | 已对 registry 缺失/未知 release/成员不匹配、invalid letter/candidate mismatch、client context/correctness disagreement、unmatched comment/proposal，以及 ingestion failure/idempotent/conflict/coverage 分级提示；六业务页已有本地原子 snapshot，但两个 quality RPC 保持独立。仍需 hosted 数据验收与有明确分母/窗口的异常频率阈值 | STORE-001、OBS-001、STATS-004 |
| RECO-001 | 离线惊讶策展 / session 选下一题 | P1 | 进行中 | `Next` 优先展示更可能产生有价值惊讶且不重复的题 | 本地已接完整 60 题 attested catalog 与 `Next`：严格 artifact-only validity/scorer、tie-aware cold start、Beta prior、completed/current 过滤、family 多样性、默认 20% ε-greedy、最低本地曝光探索、精确 mixture propensity、`question_presented` decision/exposure trace，以及异常顺序 fallback。仍待 hosted presentation persistence/propensity 报告；当前不读取 SURPRISE-002、不跨 session 个性化 | QPUB-001、SURPRISE-001 |
| RECO-002 | 个性化惊讶预测与实验 | P2 | 讨论中 | 根据当前 session 表现、兴趣与题型疲劳估计 `predicted_personal_surprise` | 每次曝光记 policy version/decision ID/source/position/selection propensity；分层后验/Thompson Sampling 不跨 release 混数据；A/B 主指标预先定义为 `observed_surprise_rate`，answer/continue rate、accuracy、family/type 覆盖和独立 like 作 guardrail；无充足样本/区间或 propensity 不宣称提升 | RECO-001、SURPRISE-002 |
| MAINT-001 | 全仓 lint 清理 | P2 | 待开发 | 当前共享工作树重新达到全仓 Ruff clean | 当前 feedback/Reports/preflight 目标范围 Ruff 与 format 已通过；本轮没有把共享工作树其他模块的既有债冒充为已清理，也未覆盖对应未提交改动 | 确认其他模块改动归属 |

### 惊讶值与推荐验收合同（SURPRISE/RECO，本地已接线、hosted 待验收）

1. **三层可解释**：离线 `intrinsic_surprise_proxy`、用户 `observed_surprise_rate` 和策略 `predicted_personal_surprise` 分字段、分版本、分报告；不用任一层冒充另一层。
2. **reaction 完整链路**：reveal 前禁用惊讶按钮；reveal 后 true/false 都通过 Python、Edge、forward SQL migration 的严格 schema，进 append-only trace/store；重试、browser restore、download recovery、single/batch upload、duplicate 和 conflict 都有正反测试。
3. **权威归因与守恒**：聚合使用 registry 的 release/question/version/family/type，不信任 payload 自报；rating = yes + no，response 分母口径明确，所有现有全局筛选下仍守恒；unknown/mismatch 进 quality 不进权威比例。
4. **推荐池安全**：只返回 runtime-attested 当前 release 题，排除本 attempt 已答、无效或未登记题；本地 261 题不能绕过 60 题 manifest 边界。tie 不依赖字母/文件路径，metric direction 正确，同策略快照与 seed 可复现。
5. **无泄漏**：动态分数、用户比例和策略状态不写入 `question.json`、GT、candidate summary 或 prompt；答题前 UI 不显示 winner/hardness/surprise 信号。
6. **可评估与降级**：每次策略曝光有 policy/decision/propensity，保留预声明 exploration；读取或推荐服务失败不阻断做题，且明确回退 `Next`。只有 A/B 样本量、区间/后验与 guardrail 同时达标才宣称提升。

### 本地交付说明：下载足迹恢复上传（TRACE-005）

**Download session JSON** 现在可以作为浏览器 session 丢失后的独立 recovery outbox
上传，不会合并进当前 live trace。文件对象先用 size 过 10 MiB gate；超限时不调用
`getvalue()`，任何网络请求前还必须通过 strict UTF-8/JSON、exact wire schema、RFC 3339
timestamp、event count、trace ID、session ID 与逐事件 schema 校验。

文件本身不携带、也不能恢复旧浏览器的 acknowledged/quarantine 状态。canonical 完整
文件内容生成稳定 `recovery_id`；当前浏览器 session 在该 ID 下保存 pending、acknowledged
和 quarantined 进度，因此重复选择同一内容时只发送 pending。即使另一个文件复用同一
event ID，只要 logical payload 不同就会生成不同 `recovery_id` 并实际请求权威服务端，
由其返回 duplicate 或 409 content conflict，而不是被第一份文件的本地状态错误抑制。

发送复用 500-event / 1-MiB 分批、Bearer、严格四 counter/UUID 回执和 409 逐事件隔离；
只有确认冲突的事件进入 quarantine，可重试失败继续 pending。相关 recovery、feedback、
outbox 测试共 167 项通过，Streamlit AppTest widget smoke 通过。该条目只完成本地交付，
尚未在当前可访问但仍为旧 UI 的站点证明已部署，也没有真实 hosted Supabase 验收证据。

### 当前 release 修复证据（已完成）

旧 release
`release_bec00e86071f939e1153b7b5402961388bedf37483ce2e41b6505add58792831`
的四类数据质量问题已通过 canonical pipeline 修复，未手改 question/GT 结论：

- 使用现有 stored GT 通过标准 `generate-question` 生成新的 multivariate run
  `run_20q_3c_3a8a23`；60 题的 180 个 choices 全部满足
  `failed_seeds == 0 && excluded == false`。
- 8 个 fully-failed candidate 重新执行标准 `run_ground_truth()`；aggregate
  mean/std 以 `null` 表示。当前 bundle 的 454 个 JSON 文件均可由拒绝非标准常量的
  严格 parser 读取。
- fresh-target publisher 只复制三个 run 实际引用的 3 个 candidate set；旧的
  32-candidate orphan set 不再进入 bundle。进一步的 canonical allowlist 审计排除了
  source 中的 `custom_settings` 和 `__pycache__`；artifact 数从 1708 降为 1483。
- bigram/univariate prompt 均由当前标准 renderer 重渲染；20 个 bigram prompt
  不再宣称 “best test MSE”。
- 新 manifest 保持 60 questions / 3 source runs / 3 datasets / 192 candidates，
  内容寻址 release 为
  `release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f`。

## 已交付

| 日期 | ID | 交付内容 | 版本 |
|---|---|---|---|
| 2026-07-11 | BASE-001 | 60 题发布包；Next/Random；提交后显示指标与会话分数 | `034aaad` 之前 |
| 2026-07-11 | BASE-002 | 自定义 setting 可继承 choice，经标准 `write_candidate` → `run_ground_truth` 路径训练并显示临时曲线 | `034aaad` |
| 2026-07-11 | TRACE-001 | 版本化 append-only event/trace schema；稳定 question version、session sequence 和 event-id 幂等；成功/失败/校验拒绝分开统计 | 当前工作树 |
| 2026-07-11 | TRACE-002 | 答案、有效/被拒 setting、训练成功/失败与 comment 均进入 session trace；Reset 开启新 attempt 而不删除旧足迹 | 当前工作树 |
| 2026-07-11 | TRACE-004 | 侧边栏下载与上传使用完全相同的 JSON envelope | 当前工作树 |
| 2026-07-12 | TRACE-005 | 下载的 session JSON 可经 10 MiB pre-read gate、strict UTF-8/JSON、exact wire schema、RFC 3339、trace/event identity 校验进入独立 recovery outbox；canonical 完整内容 `recovery_id` 在当前浏览器保存 pending/ack/quarantine，同内容重试只发 pending，不同 payload 即使复用 event ID 也交给服务端裁决。复用 500 events / 1 MiB、严格回执与 409 isolation；相关 recovery/feedback/outbox 167 tests 和 AppTest widget smoke 通过 | 当前工作树；未部署、未 hosted 验收 |
| 2026-07-11 | STORE-000 | 通用 Bearer HTTP client；单事件与整段 trace 的真实本地端到端测试通过 | 当前工作树 |
| 2026-07-12 | TRACE-003A | Inspector 只有在 endpoint 与 Bearer token 同时存在时才允许上传；成功必须返回 accepted/duplicate/conflict/rejected 四个完整非负整数 counter、匹配本批总数、零 conflict/rejected，以及 header/body 完全相同的 canonical request UUID。其他 2xx 和失败均保持 pending | 当前工作树；未 hosted 验收 |
| 2026-07-12 | TRACE-003B | Session Upload 只发送未 acknowledged、未 quarantined 的 pending events，按 receiver 的 500 events / 1 MiB 上限分批；整批 409 后逐事件隔离，只 quarantine 已确认 conflict 的 ID，其他事件继续上传或保持 pending | 当前工作树；未 hosted 验收 |
| 2026-07-11 | STORE-001A | Supabase append-only 表、Edge Function、RLS、幂等写入和 4 个 REPORT-001 SQL views 已具备可部署脚手架 | 当前工作树 |
| 2026-07-12 | STORE-001B | Node harness 直接执行真实 `feedback-ingest` Edge Function，并用持久化 mock RPC 验证首传/完全相同重试、content conflict 409、new+matching+conflict 整批不写，以及 commit 后响应丢失再重试；first-write-wins 且最终 event_id 集合无丢失无覆盖 | 当前工作树 |
| 2026-07-12 | STORE-001C | opt-in hosted roundtrip verifier 复用生产 client/schema，显式确认永久写入；首次 POST 前向 stderr 输出并 flush run ID，严格校验 header/body canonical request UUID 和 fresh/resume receipt。早期 synthetic 四业务视图路径保留为 legacy 合同测试；当前 CLI 默认的 registry-aware 验证见 STATS-003C | 当前工作树；未 hosted 运行 |
| 2026-07-12 | STORE-001D | hosted verifier 默认追加真实 changed-text/same-ID 探针：严格要求结构化 409 `EVENT_ID_CONFLICT`、`0/0/1/1` 和不同于正常 POST 的 canonical request UUID；随后 exact event resolution 必须仍是原 registry-matched comment。显式 `--skip-conflict-probe` 才可跳过并返回 `conflict_verified=false`；fresh/resume 默认都会留下永久 conflict footprint | 当前工作树；未 hosted 运行 |
| 2026-07-12 | STORE-001E | hosted verifier 新增显式 `--include-mixed-batch-probe`：在正常及 single-conflict proof 后提交 `[changed-text conflict, deterministic new comment]`，严格要求第三个两两不同 request UUID、结构化 409 `0/0/1/2`、exact outcome 的 conflict/audit `1/1`、classified `2` 和 reuse rate `0.5`；原 event 的 resolution 不变，withheld 新 event 仍必须 `not_found`，证明被测 `new + conflict` batch 的 first-write-wins 与 all-or-none。该 flag 与 skip 互斥，fresh/resume 都会留下第二笔永久 conflict outcome/audit；尚未真实 hosted 运行，也不外推到 `duplicate + conflict` 或并发 | 当前工作树 |
| 2026-07-12 | STORE-001F | Python/Edge 增加跨运行时无损 JSON 合同：recursive integer-valued number 超出 ±(2^53−1) 时分别在触网/RPC 前 fail closed；未配对 Unicode surrogate 同样拒绝；identifier/comment 上限在两端统一按 Unicode code point，使 emoji 边界与 PostgreSQL `length` 一致。Python 与真实 Edge Node harness 覆盖正负安全整数边界、嵌套越界、emoji 边界和 lone surrogate | 当前工作树；未 hosted 验收 |
| 2026-07-12 | STORE-001G | registry-aware hosted verifier 默认正常 trace 为同 session 的 answer+proposal+comment 三事件：首传必须精确 `3/0/0/0`，原样 replay 必须 `0/3/0/0`；两次 canonical request UUID 与各自 ingestion outcome 独立核对，exact resolution 与 Answers/Proposals 明细行必须返回 registry-derived context/correctness/setting facts。resume 区分本次首写与已有 duplicate，显式 skip 不冒充通过；legacy fixture 仍保留隔离两事件路径 | 当前工作树；未 hosted 验收 |
| 2026-07-12 | OBS-001A | 新增独立 private/append-only ingestion outcome 表；Edge 对鉴权成功 POST 记录 success（含 duplicate-only）、client rejection、service failure 和 nullable storage state，且 outcome 写入 fail-open、不改变原 receipt。401/405/缺配置仅安全日志；不存 token/body/IP/comment/setting；failure rate 继续 N/A，等待 hosted coverage 与独立聚合/RPC | 当前工作树 |
| 2026-07-12 | OBS-001B | 新增 private/service-role-only `feedback_report_ingestion_summary`：只聚合 `included_in_rate=true`，按服务端 `started_at` 的 `[from,to)` 返回始终一行；request failure rate 为 `(client rejection + service failure) / recorded requests`，duplicate event-ID rate 为 `duplicate / (accepted + duplicate)`，零分母为 N/A。Edge/client 已支持该 RPC，Reports 新增独立 Ingestion observability 页签；内容筛选时跳过查询，观测错误不覆盖业务快照。`end_to_end_coverage_available=false`，业务 Summary failure rate 继续 N/A | 当前工作树 |
| 2026-07-12 | OBS-001C | ingestion summary 新增仅该 view 可用的严格 UUID `request_id` 精确筛选，Reports UI 不暴露；hosted verifier 在永久写入前先以随机合法 UUID 验证全零/null，再用 receipt UUID 要求唯一 success outcome 与 accepted/duplicate counters/rates 一致，并用另一合法不存在 UUID 做写后负对照。该 proof 仍只覆盖 persisted subset，尚未完成真实 hosted 验收 | 当前工作树 |
| 2026-07-12 | OBS-001D | ingestion summary 新增 `conflict_audit_event_count`，按已筛选 outcome request ID 关联 private sidecar，并由严格 client 要求与 `conflicting_event_count` 相等；默认 hosted conflict probe 因而能同时验证 exact 409 outcome、sidecar correlation 和业务数据未覆盖。真实 hosted SQL/RLS/并发仍待验收 | 当前工作树 |
| 2026-07-12 | STATS-002A | Reports 新增不发起额外请求的 Data quality 页签：unknown answer correctness 与 client rejection 为 warning，service failure 为 error，event-ID 复用为 info（可能是正常幂等重试，不证明 payload 相同或冲突）；由于 coverage 不完整，且 ingestion/registry quality 仍与六业务页使用独立快照，页面不显示绿色整体健康，也不推断 unknown release | 当前工作树 |
| 2026-07-12 | STATS-002B | 新增 service-role-only 原子 ingest RPC：以 PostgreSQL JSONB equality 比较 7 字段 logical event；exact replay 返回 200，content conflict 返回 409 `EVENT_ID_CONFLICT`，并在 private append-only sidecar 留下无 payload/hash 的审计记录。任一冲突使同批新事件全部不写；Reports 将 schema 1.1 的 verified idempotent、legacy unclassified duplicate 与真实 conflict 分开统计，reuse rate 以全部 `classified_event_count` 为分母，包含冲突批次中 new-but-withheld 的事件；Inspector 对冲突给出不可盲重试提示。仅完成本地合同，尚未 hosted PostgreSQL/并发验收 | 当前工作树 |
| 2026-07-12 | OPS-001A | 只读 `feedback_rollout_preflight.py` 按四个 rollout phase 累积检查完整 migration inventory/order、Edge、生产 clients、hosted verifier、Inspector、release attestation、registry exporter/JSON/data migration、`15000` aggregate、`16000` detail/双 revision status、`17000` atomic snapshot 与 Reports app，以及 Git tracked/clean/HEAD 字节状态和精确输入 fingerprint。三态输出不冒充 hosted 验收且固定 `deploy_ready=false`；当前未提交输入会正确 fail closed | 当前工作树 |
| 2026-07-12 | OPS-001B | 新增显式 `feedback_postgres_acceptance.py`：DSN 仅来自环境变量且必须 `--confirm-staging`/非生产标签；惰性 psycopg 可选依赖；真实 catalog 当前核对含两个 surprise RPC 的 15 个应用函数、六表 ACL/FORCE RLS/no-policy、精确 `tgtype` 和全部稳定命名约束；从 hosted 180 choice rows 重建 60 题 canonical registry 并校验内容 hash/dual revisions；statement-level append-only 与 deferred registry 反例全在 savepoint 内，最终显式 rollback/close、绝不 commit。修正三份待部署 SQL 和 verifier 中把 PostgreSQL `COALESCE` 伪装成 `pg_catalog` 普通函数的不可执行写法，并以 migration/preflight tamper tests 防回归。工具已纳入 report-app fingerprint；尚未获得 staging DSN，不能记 hosted PASS | 当前工作树；真实 staging 未运行 |
| 2026-07-12 | QPUB-008A | 匿名浏览器重新核对线上 `/`：页面实际可载入 60 题旧版 Question Inspector，保留 Add custom setting，但没有 session upload 或单题 comment；Streamlit creator 与 Git remote owner 同为 `renrua52`。这证明项目关系和线上/本地功能差异，不冒充精确 branch/source SHA 证据 | 线上只读页面 + 当前仓库；Cloud 控制台映射仍待管理员确认 |
| 2026-07-12 | QPUB-008B | 新增 retrospective deployment ledger：candidate 从声明 commit blobs 重算 rollout fingerprint，三类 evidence 共享 context 并防跨 deployment 重用，source mapping 绑定独立 raw control-plane capture 与不同 reviewer，状态明确为 `READY_FOR_REVIEWED_ACTIVATION` / `ACTIVATED_REVIEWED`。Canonical JSONL hash chain、显式确认/加锁/原子 append、terminal/replacement 与外部 head-pin 边界均已本地实现；未生成任何虚假当前部署记录 | 当前工作树；无真实 staging/provider evidence |
| 2026-07-12 | QPUB-008C | Inspector runtime SHA 现在必须来自实际 checkout，任一 allowlisted env SHA 只作一致性核对；最小 Git 环境清除重定向、禁用 replacement objects，checkout 缺失、声明畸形或冲突均显示 `N/A`。Ledger tool/docs 和 runtime identity 合同已纳入 `report-app` fingerprint，实际 ledger/evidence 保持 post-deploy 排除 | 当前工作树；当前线上旧 revision 尚无该控件 |
| 2026-07-12 | UX-001 | **My session** tab 已汇总整个浏览器 session 的答案/正确率、proposed/rejected settings、custom run 成功/失败和 comments；答案、setting、comment 使用安全字段表格；支持完整 JSON 下载，并区分 pending 与 endpoint 已确认的事件。页面状态仍只属于当前浏览器 session，可靠跨 session 持久化依赖 STORE-001 endpoint | 当前工作树 |
| 2026-07-12 | REPORT-001A | 本地可部署的 raw-event parameterized RPC、专用 Bearer `feedback-report` Edge Function、严格 Python client、独立四-tab Streamlit UI、UTC 筛选、精确 page cap 与安全 CSV 已实现；模拟 endpoint 的真实浏览器 smoke 已通过，尚未 hosted deployment | 当前工作树 |
| 2026-07-12 | REPORT-001B | 新增 `13500` additive raw-view hardening：session question count 按 `(question_id, question_version)`；question attempt count 按 `(session_id, coalesced attempt_id)`；raw accuracy 只用 boolean known answers，并在旧列尾部追加 known/incorrect/unknown。proposal 的 `n_seeds/base_seed` 保持 nullable integer，但字符串、小数和 int32 越界值安全返回 `NULL`，不再使整 view cast 失败。Preflight 将旧列 prefix、proposal exact shape、ACL marker 与 migration inventory 纳入 `lockdown-report` | 当前工作树；未 PostgreSQL/hosted catalog 与反例验收 |
| 2026-07-12 | STATS-003A | 新增 fully-attested feedback registry exporter：同时执行 runtime 全 artifact attestation 与 publisher GT validation，生成 bundle 外 deterministic JSON 和仅含三组显式 `INSERT` 的 SQL；当前 release 固化 60 questions / 180 choices / `registry_db3f1a...`。Registry ID 不受 manifest `generated_at` 影响，manifest SHA 仅作 provenance，`--check` 字节级复验 | 当前工作树；未 hosted apply |
| 2026-07-12 | STATS-003B | 本地 `14000/14500/15000/16000` 部署输入定义 owner-reviewed、service-role 只读、FORCE RLS、append-only release/question/choice registry；deferred FK/constraint trigger 设计用于校验 correct choice、数量完整及跨 release 同 question-version 一致。动态 view 按 release claim + question ID + version 精确解析历史 raw event，不猜 release、不改原事件；当前六业务 RPC 只使用 matched authority，unknown/mismatch 单列质量 RPC | 当前工作树；未真实 PostgreSQL/catalog/约束反例/并发验收 |
| 2026-07-12 | STATS-003C | hosted verifier 的本地编排从完整 attested bundle 选择真实 registry 成员，先校验 `registry_v1/detail_v1` 七列 status、两个 detail RPC empty negative control 与 `17000` 六页 empty snapshot negative control，再用故意错误的 client family/type/is_correct 验证 exact-event 仍返回 canonical registry facts；默认 answer+proposal+comment 三事件首写/重放还会精确核对 Answers/Proposals 行，并覆盖 single conflict 与可选 mixed withheld | 当前工作树；未真实 hosted 运行，不标记 authority/detail/snapshot/exact-event 为 hosted PASS |
| 2026-07-12 | STATS-003D | Registry 安全收口增加 child-row deferred inventory 复验和同 `question_version` advisory lock；preflight 从完整 1483-artifact attested bundle 重建 registry，并逐字节核对 JSON/SQL，防止答案库成对漂移。Report identifier 统一按 Unicode code point 限长，legacy 缺失 selected letter 可进入 exact audit；Reports UI 由 `17000` snapshot 内嵌并强制验证 `registry_v1` 和 `detail_v1` 双 revision facts，不再单独请求 status | 当前工作树；真实 PostgreSQL 触发器/RLS/并发与 hosted cutover 仍待验收 |
| 2026-07-12 | REPORT-001C | Reports 将 KPI 改为 Authoritative accuracy，新增 Registry quality 页签与严格 client schema；缺/未知 release、question membership、invalid letter、candidate/context/correctness mismatch、unmatched comment/proposal 均可见。业务筛选与正确率不再读取 payload 自报维度或 `is_correct`；exact-event RPC 为 hosted verifier 提供不受历史聚合污染的单 event 证据 | 当前工作树；未 hosted 验收 |
| 2026-07-12 | REPORT-002A | `16000` 新增 registry-matched Answers/Proposals 分页 RPC；与 `17000` 合并后形成 11-RPC Edge/Python allowlist、严格 answer/proposal/snapshot schema、七列 dual-revision status、六业务页/九页签 UI、安全 JSON 展示与 complete-page-only CSV。明细行保留 session/attempt identity，当前筛选仅为 release/family/type/question/date | 当前工作树；未 hosted RPC/行语义验收 |
| 2026-07-12 | REPORT-002B | 新增 forward migration `18000`：保留七个历史 positional 参数前缀并追加 `session_id`/`attempt_id`，六业务 RPC 均按服务端权威 identity 筛选，snapshot 逐页透传；严格 Edge/client/UI 已接线。Hosted verifier 本地编排要求真实 pair 正向命中及错误 session/attempt 两个空快照，证据字段为 `session_attempt_filters_verified=true` | 全仓 `934 passed`，目标 Ruff/format 与 `git diff --check` 通过；`contract.report.session_attempt_filters` 及其余业务/安全合同 PASS，最终本地输入 fingerprint 为 `b5919224c6b56306d1125d534621ec6f819ec2f3342305f339f6c86eff067186`。当前工作树仍 untracked/dirty，且未部署、未执行真实 PostgreSQL/hosted 验收 |
| 2026-07-12 | STATS-004A | `17000` 新增 service-role-only `feedback_report_business_snapshot`；App 一次 GET、Edge 一次 PostgREST、单 SQL statement/MVCC snapshot 返回 `business_snapshot_v1`、server `snapshot_at`、内嵌 registry/detail authority/counts 和 `pages_json` text。严格 client 校验六页与跨页计数后原子替换 UI；失败 refresh 保留上一完整快照。Hosted verifier 在首个写入前要求随机不存在 question 的全空六页负对照，并输出 `business_snapshot_verified` | 当前工作树；本地实现/合同检查完成，未真实 PostgreSQL 或 hosted 验收 |
| 2026-07-12 | STATS-004B | `17000` 增加六页完整行 byte budgets 与 4 MiB 总 cap；六个 bounded MATERIALIZED staging CTE 先保留稳定排序前 `p_limit` 行和 exact total，再执行 JSON/UTF-8 字节计算，避免所有未返回宽行都被序列化。无效/null limit 在业务 RPC 前短路；Edge 校验后原样嵌入 PostgREST snapshot JSON，保留超过 JS safe integer 的外层 bigint；客户端用 `rows < total` 禁止部分 CSV | 当前工作树；backend 18、报告 focused 375、全仓 811 均通过，仍未真实 PostgreSQL/PostgREST 负载或 hosted 验收 |
| 2026-07-12 | QPUB-001 | `quiz_manifest.json` 已覆盖稳定 release ID、question versions、source/partial runs、counts 和全部 artifact SHA-256 inventory | 当前工作树 |
| 2026-07-12 | QPUB-002 | `tools/publish_quiz_bundle.py` 可 dry-run 或发布整个 run/单题；拒绝重复 ID、缺失/越界/冲突 artifact 和无效 GT choice | 当前工作树 |
| 2026-07-12 | QPUB-003 | 发布 smoke validation 已覆盖 metric 方向与全链一致性、candidate/set/question budget、summary/seed identity、存储 GT winner/correct answer/significance，以及 prompt GT 泄漏启发式扫描 | 当前工作树 |
| 2026-07-12 | QPUB-005 | Inspector 默认直接读取 bundled release 并执行完整 runtime attestation：重算 release/question version，核对 source runs、questions、counts 和 1483 个 artifact 的 path/type/size/SHA-256，拒绝 symlink/special file，且不信任 mtime cache；默认 manifest 缺失/无效或任一内容不匹配均 fail closed。只有显式非默认本地 root 在完全没有 manifest 时可 unversioned 浏览；UI 显示 release、manifest SHA、固定 entry 与严格环境 Git SHA/N/A | 当前工作树；未部署、未 hosted runtime 验收 |
| 2026-07-12 | QPUB-006 | 发布器对 choice partial seed failure、非标准 JSON 数值和非-MSE 题目的 stale MSE prompt fail closed | 当前工作树 |
| 2026-07-12 | QPUB-007 | Publisher 使用 canonical file allowlist；不复制 `custom_settings`/`__pycache__`，direct manifest build 对额外物理 artifact fail closed | 当前工作树 |
| 2026-07-12 | QPUB-009 | Publisher 与 runtime release attestation 对任意层级 duplicate JSON object key fail closed；publisher/feedback question version 同步拒绝超出 JavaScript safe-integer 范围的 integer-valued number 和未配对 Unicode surrogate，避免同一 release 在不同 runtime 得到不同解释 | 当前工作树 |
| 2026-07-12 | QPUB-010 | 题目发布流程增加 registry export/data migration：不能网页直传单个 question JSON；维护者必须从完整 canonical bundle 生成新 release registry，review 后随 schema/report migration 部署。Ingest Edge 的 service-role 不获 registry INSERT 权限，避免接收端凭据能改答案库 | 当前工作树 |
| 2026-07-12 | QREL-001～004 | 新 60 题 release 已消除 partial-seed choice、bare `Infinity`、orphan candidate set 和 bigram stale MSE 文案 | `release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f` |
| 2026-07-12 | VERIFY-001 | REPORT-002/`16000`、STATS-004/`17000`、三事件 detail-row、empty snapshot、byte budgets、bigint 原样转发和 verifier 写前顺序已纳入本地合同；报告 focused `375 passed`，全仓 `811 passed`，fingerprint=`615c0f8f42afd95749f543f209f1a8565ea1e060d9b246935de5734ecc1c1c7c`。全部业务合同 PASS；Git 三项 FAIL、hosted UNVERIFIED、`deploy_ready=false` 均符合当前未提交未部署事实 | 当前工作树；真实 PostgreSQL/catalog/RLS/并发/负载与 hosted 验收仍待办 |
| 2026-07-12 | VERIFY-002 | OPS-001B direct PostgreSQL acceptance、完整 constraint/trigger catalog、registry content re-hash、COALESCE 可执行性回归和 preflight tamper 已纳入；focused `300 passed`、全仓 `867 passed`，目标 Ruff/format/diff-check 通过，fingerprint=`72571ecf8a602604e242cab533f58bf61743c930a3cb055b6467a5931484cf73`。无 DSN 时 CLI 在触网前输出脱敏 JSON 并 exit 2。业务/静态合同 PASS；Git 三项仍 FAIL、hosted UNVERIFIED、`deploy_ready=false`。全仓 Ruff 的 19 条其他模块既有债未纳入本轮修复 | 当前工作树；真实 staging DSN/catalog PASS、Edge roundtrip、重启/并发/负载仍待办 |
| 2026-07-12 | VERIFY-003 | QPUB-008 retrospective ledger、shared deployment context、raw provider capture/double review、evidence reuse/terminal/causal guards、commit-blob fingerprint 与 runtime checkout SHA/no-replace 均纳入 `report-app` 合同和 tamper tests。Focused `337 passed`、全仓 `923 passed`，10 个目标 Python 文件 Ruff/format 与全工作树 diff-check 通过；fingerprint=`1be695fb9c954c269efa1c2778ee823ec8b5b4b683e02c06299a2483315035be`。全部业务/安全合同 PASS；Git 三项按 dirty/untracked 事实 FAIL，hosted UNVERIFIED、`deploy_ready=false` | 当前工作树；未创建 ledger event，真实 staging/provider evidence 仍待办 |
| 2026-07-12 | SURPRISE-001A | Inspector 在答案/真实排名揭晓后采集一次“出乎意料 / 符合预期”；`question_reaction_submitted` 使用严格 reaction/value/timing/attempt schema 和稳定 event ID，同值重放幂等、反值显式冲突。My session、download/recovery、browser outbox、pending/ack/quarantine 均复用现有链路；Edge 与 `19000` migration 同步扩 enum/payload constraint，preflight/rollback-only acceptance 已登记 | 本地 focused reaction/feedback/recommender 251 项与 migration/preflight/acceptance 159 项通过；未部署、未 hosted roundtrip |
| 2026-07-12 | RECO-001A | 新增无网络/Streamlit 依赖的推荐核心：`.5/.3/.2` available-weighted cold start、Beta 强度 4、显式 reaction 后验、valid/blocked/completed 硬过滤、family 多样性、稳定 tie-break、最低曝光探索及默认 20% ε-greedy；公开结果只含 question identity、mode、propensity，不泄漏答案/GT/私有 prior | 初始核心切片通过 `tests/test_surprise_recommender.py` 42 项；manifest/catalog/Next 接线由后续 RECO-001B 完成 |
| 2026-07-12 | RECO-001B | 新增只读私有 catalog：只遍历当前 attested 60 题，严格读取 question/spec/summary，以 validity 硬门和 tie-aware 参数量/depth/width/optimizer shortcuts 构造 cold start；`Next` 已接 80/20 ε-greedy、完成题过滤、family 多样性、最低本地曝光探索和顺序 fail-safe。Initial/Next/Random/picker 均为发布题记录 `question_presented` decision/policy/mode/propensity/source/position | Catalog/recommender/manifest focused 88 项通过；当前 catalog 60/60 valid。本地实现，不读取远端 reaction posterior，未 hosted 持久化或 A/B 验收 |
| 2026-07-12 | SURPRISE-002A | 新增 additive forward `20000`：service-role-only per-question first-valid-post-answer ratings 与 reaction-quality conservation 两 RPC；Edge/client 严格扩为 13 views，Reports 增加 Surprise tab。Question 页显示 answer/rating/yes/no/coverage/observed rate/Beta(1,1) posterior；Quality 页显示 raw/valid/orphan/duplicate 及互斥 orphan breakdown | SQL/Edge/client/UI/preflight/acceptance 均为当前本地实现；两个 RPC 是独立 statement，不属于六业务页 snapshot。未真实 PostgreSQL apply、未 hosted Reports 验收 |
| 2026-07-12 | SURPRISE-002B | 修正 Surprise question 分页合同：删除 RPC 内部 `p_limit` 截断，统一由 PostgREST `limit/offset + count=exact` 产生稳定分页和可信 total，避免第二页为空及完整 CSV 误判；严格 client 保留 bigint 原始 JSON、15/10 列 schema、比例/时间/守恒 fail-closed。`question_presented` 另补 browser outbox、下载恢复、ack/quarantine 六字段不丢失回归 | 全仓 `1078 passed`；41 个相关 Python 文件 Ruff/format、全工作树 diff-check 通过。`report-app` 所有业务/安全合同 PASS；Git 三项因 dirty/untracked 如实 FAIL，hosted 仍 UNVERIFIED |
| 2026-07-12 | VERIFY-004 | 真实本地浏览器复验 reaction/recommendation/report 闭环：第二道已揭晓题的“符合预期”被锁定并进入 pending outbox；Surprise-aware `Next` 从 bigram 切换到 multivariate，session exposure 从 3 增为 4；独立 Reports Surprise tab 显示 yes/no、coverage、observed rate、Beta posterior 与 raw/valid/orphan/duplicate 守恒，刷新成功 | 使用本地严格 mock report endpoint 验 UI，不冒充 PostgreSQL/Supabase hosted 证据；测试 tabs、Inspector、Reports 和 mock server 均已关闭 |

## 讨论记录

| 日期 | 主题 | 结论 | 下一步 |
|---|---|---|---|
| 2026-07-11 | 线上站点与本地仓库关系 | 应用代码和 60 题发布包来自仓库；`data/` 下的新题只在本地可见 | 新 release 和 smoke validation 已完成；后续确认 Cloud branch、推送并验证线上 manifest |
| 2026-07-12 | 线上只读复查 | 最新匿名浏览器可载入 60 题旧版 inspector：有 Question/Prompt 和 Add custom setting，没有 session upload/comment；creator 与 remote owner 同为 `renrua52`。远端 main=`034aaad...` 且不含当前新功能；页面未暴露精确 release/source SHA | 管理员从 Cloud 控制台记录 repo/branch/entry/deploy ID；部署后核对 runtime SHA、release、registry、Upload 与私有 Reports，不从题数猜精确版本 |
| 2026-07-11 | 用户数据回流 | 需要两种入口：整段足迹批量 Upload，以及当前题评论单条 Upload | 先完成事件 schema、session 采集和下载，再接外部持久化 endpoint |
| 2026-07-11 | 是否直接使用 Kahoot | 不迁移；借鉴其 quiz-version / game-session / question-report 分层，自建更适合 ArchitectureIQ 的交互和统计 | 实现内部 session event API 与 Reports 页面，不做实时房间 |
| 2026-07-11 | 回流 MVP 第一轮实现 | 单 comment 和整 session 已共用同一 endpoint；事件先留在 session，再上传，失败不会丢；Supabase 作为正式接收端 | 创建 Supabase project、部署 migration/function、配置 Streamlit secrets |
| 2026-07-12 | 我的足迹页 | 当前浏览器 session 已有可审阅、可下载、可重试上传的统一视图；acknowledged 只表示收到 endpoint 完整回执，不等同于浏览器本地状态具备持久性 | 部署并验证 STORE-001，之后再把 My session 扩展为跨 session 的私有报告入口 |
| 2026-07-12 | 下载足迹恢复上传 | 本地 UI、严格文件 gate/parser、canonical-content recovery state、pending-only 分批与 409 isolation 已完成；恢复 outbox 不合并 live trace，也不从文件继承旧浏览器 acknowledged/quarantine | 部署后用真实 hosted receiver 验证 accepted/duplicate/409 quarantine；此前不声称线上可用 |
| 2026-07-12 | 内部 Reports 第一轮 | 独立 app、raw-event prefilter RPC、read endpoint/token、四个业务 tab 和 CSV 的本地闭环已完成；accepted-event store 不能推导 ingestion failure rate；read token 不替代网页访问者鉴权 | 第五个 ingestion observability tab 已在第二阶段补齐；下一步部署 hosted Supabase，并把 Reports 作为启用平台级维护者访问控制的私有 Streamlit 验收 |
| 2026-07-12 | Ingestion outcome 第一至三阶段 | 独立 sanitized outcome 事实表覆盖鉴权成功 POST 的成功、客户端拒绝与服务失败；duplicate-only 归成功，storage 不确定性不再伪称 rejected。独立聚合 RPC 和第五页签仅报告已记录子集；receipt UUID 精确正例和不存在 UUID 负对照用于 hosted proof，不冒充端到端覆盖 | 部署并验证 hosted Postgres migration、forced RLS/grants 与 outcome coverage。401/405/未到 Edge/全库不可达不混入 recorded rate，业务 Summary 的端到端 rate 继续 N/A |
| 2026-07-12 | 数据质量监控第一阶段 | 先复用 Summary/Ingestion 已有事实做分级提示，不为展示层新增 RPC；当时 duplicate 只证明 event ID 被复用，不能冒充 payload equality/conflict；coverage 不完整，且两个 quality RPC 仍不与六页 business snapshot 原子一致，因此不显示绿色整体健康，也不从筛选无结果推断 unknown release | 内容冲突分类已在下一阶段改用数据库 JSONB equality；六业务页本地原子 snapshot 已由 STATS-004 补齐，仍需 hosted 验收和有明确分母/窗口的异常频率阈值 |
| 2026-07-12 | Event-ID 内容冲突 | 不使用跨语言自制 fingerprint；数据库在 advisory lock 内用 7 字段 JSONB equality 权威分类。普通重试仍成功，冲突整批 409/first-write-wins，并把 legacy outcome 明确保留为 unclassified，避免把历史 duplicate 伪装成已验证安全重试 | 本地 hosted-verifier 编排默认验证 single event，显式 `--include-mixed-batch-probe` 验证被测 `new + conflict` all-or-none；仍需分阶段部署并在真实 PostgreSQL 运行两种 probe、验证 migration/ACL/并发锁，另补 `duplicate + conflict` 与并发场景 |
| 2026-07-12 | Hosted mixed-batch 探针范围 | mixed probe 不设为默认，避免每次 smoke 都多制造一笔故意的 rejection/audit；需要 all-or-none 证据时显式 opt in，且不能与 recovery-only skip 同用。即使 resume，也会产生新的 single 及 mixed conflict footprint；outcome fail-open 超时不代表 audit 未提交 | 优先在 staging 运行并保存三个两两不同 request UUID；不得把本地编排或一次 `new + conflict` 结果写成 hosted 全并发/端到端验收 |
| 2026-07-12 | 服务端权威题库 | 原始 event 继续无条件 append-only 接收；`14000` 定义 registry/动态投影，owner-reviewed `14500` 单独登记 release/question/choice，`15000` 使四个 aggregate RPC 只聚合 matched authority，`16000` 加入两个 authoritative detail RPC。无匹配事件保留 raw、进入质量统计但不进业务维度/正确率 | 本地部署输入和 `registry_v1/detail_v1` 预检已具备，但 hosted verifier 尚未真实运行。在 staging 验证 deferred inventory/correct-choice/跨-release 一致性、service-role 只读、旧 unknown 登记后自动 matched、伪造 client facts 仍返回 registry authority，并保存 dual-status、exact-event 与真实 Answers/Proposals 行证据 |
| 2026-07-12 | REPORT-002 明细 | 本地 Answers/Proposals 明细已与 aggregate 共用 registry-matched projection；`18000` 在不破坏旧 positional caller 的前提下为六业务页和 snapshot 增加 session/attempt 全局筛选 | 在 staging 以三事件探针验证 exact detail、真实 session+attempt 正向命中及两个错误 identity 空快照，保存 `session_attempt_filters_verified=true`；在此之前不声称线上可用 |
| 2026-07-12 | 惊讶值与推荐 | 采用 `intrinsic_surprise_proxy` / `observed_surprise_rate` / `predicted_personal_surprise` 三层模型；validity 为硬门，惊讶、答错、点赞分开，动态分数不进 canonical artifact。SURPRISE-001 reaction、60 题私有 catalog、`Next` ε-greedy、`question_presented` 本地足迹及 SURPRISE-002 SQL/Reports 已完成本地接线 | 下一步在 staging/hosted 验证 `19000/20000`、presentation/reaction 持久化及 surprise 行语义；再补 propensity/exposure 报告和远端后验输入。数据与评估证据充足后才进入 RECO-002 个性化/A-B，当前不声称线上已有推荐 |
| 2026-07-12 | 现网回流升级顺序 | 旧 direct-insert receiver 不能直接跨过 writer lockdown；先扩展到 `12000`，再切 RPC-aware ingest，维护 Reports 窗口内按 `12500/13000/13500/14000/14500/15000/16000/17000/18000/19000/20000` 应用。Presentation/reaction forward `19000` 必须先于接收新事件的 Edge/Inspector revision，`20000` 必须先于匹配的 Surprise Reports；fresh stack 仍按 timestamp 全序应用。每一步用同一 Git SHA/输入 fingerprint 关联审查与部署 | 当前本地 preflight/acceptance 已登记 `19000/20000`，但未形成干净部署 revision，也没有真实 hosted evidence；下一步在 staging 验证 catalog/constraint/RPC、presentation、reaction true/false、duplicate/conflict、surprise aggregates/quality 与现有报告不回退 |
| 2026-07-12 | Release → commit → deploy 追溯 | deploy ID/time 只能在部署后获得，不能写进产生该部署的同一 source commit。采用 post-deploy JSONL ledger；源码/文档进入 preflight，记录/evidence 不进入部署输入。Provider capture 与 endpoint context 仍是双人复核的运营证据，不冒充平台签名 | 先形成干净部署 commit；随后保存 Cloud raw export、PostgreSQL PASS、authoritative roundtrip PASS，按状态机追加并把最终 record hash 固定在 Git/外部审计系统 |
| 2026-07-12 | 当前 release 数据审计 | 旧 release 的 partial failures、非严格 JSON、orphan set 和 bigram metric 文案均已通过标准 GT/generator/renderer/fresh publisher 修复 | 推送并部署新 release 后做线上 manifest/题数/抽题 smoke；旧答案继续由旧 Git history + release ID 解释 |

## 开放问题

| ID | 问题 | 需要决定 | 截止条件 |
|---|---|---|---|
| OPEN-001 | 是否采用推荐的 Supabase 项目，还是已有内部数据库/API？ | 维护者提供 endpoint/token；客户端保持通用 HTTP contract | 启用线上 Upload 前 |
| OPEN-002 | 是否需要按开发者身份跨 session 统计？ | 当前 session id 足够；需要时接 Streamlit 登录身份或内部 user id | REPORT-001 设计时 |
| OPEN-003 | Streamlit Cloud 当前绑定的精确 branch/source commit 是什么？ | 已从 creator、remote、页面标题和旧版 UI 确认项目关系；仍由站点管理员在 Cloud 控制台确认 branch/source commit 并写回本文 | 自动发布流程上线前 |
