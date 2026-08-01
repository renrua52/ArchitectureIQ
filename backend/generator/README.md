# generator（规划）

生成套件将与存储系统完全解耦：只通过 `src/architecture_iq/storage/` 的 repository API 写入
`backend/data/`，不感知评测端格式；发布时不需要本套件。当前生成代码仍位于 `src/architecture_iq/`
与 `tools/`，后续迁入本目录。
