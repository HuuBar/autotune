# 实跑 FAQ 增补 · T1/T2 实测已知问题（W-C2 增补稿）

**体例**：同《03_实跑支持手册.md》§6 FAQ（症状 / 已知原因 / 处置）。本稿为增补条目，
评审后可整体并入 03 手册 §6 表格（替换表中"（由 W-C2 补充）"占位行）。

| 症状 | 已知原因 | 处置 |
|---|---|---|
| 任何命令启动即报 yaml 解析错：`expected <block end>, but found '<scalar>'`（config.yaml line 69 附近） | **config.yaml 既有未闭合引号**：main 分支 HEAD 的 `llm.deepseek-v4.api_key: "` 引号未闭合（HEAD 缺陷，非本批引入），导致整个 config.yaml 无法被 yaml.safe_load 解析。**本分支已修**（W-B 提交 5631a16：`api_key: "` → `api_key: ""`） | 确认在 feat/kg-wc（或含 W-B 成果的分支）上实测；若必须在 main 上跑，手工把该行补成 `api_key: ""` 即可，一行改动 |
| `pytest tests/ -q` 报 `tests/unit/tools/code_test.py` 收集失败（`ModuleNotFoundError: No module named 'langchain'`） | worktree/pip 环境不稳定时包装进了别的解释器（如 pip 与 python3 不对应），当前环境缺 langchain；code_test.py 顶部 import tools.* 需要 langchain。**main 分支既有现象，与本批无关** | 判定方法：① 错误仅出现在 code_test.py 的 collection 阶段，且 `tests/knowledge_pipeline/`、`tests/unit/agent/` 不受影响；② `python3 -c "import langchain"` 同样失败；③ 切到 main 分支同样复现。三条皆中即与本批无关，用 `pip install -r requirements.txt`（确认 `which pip` 与 `which python3` 同源）补齐后重跑；验收回归命令只圈 `tests/knowledge_pipeline/ tests/unit/agent/` 两个目录 |
| `kg_report_evidence` 的返回文本是"E9 重排后的假说全景"，而不是设计书 §4.2 表格里的"Top3 推荐" | **D2 拍板口径（2026-07-23）**：视图与工具输出定位为**信息供给而非路线推荐**——禁用"推荐/建议"措辞，E9 排序只作参考信息呈现，行动判断归 Planner。§4.2 表中"返回重排后的 Top3 推荐"是拍板前措辞，实现按拍板后口径返回全景文本（W-B 单测已固化"无推荐措辞"断言） | 不是 bug，无需处置；若实测发现模型因此不会用排序信息，按设计书 §9 开放问题 1 的迭代路径处理（先归因模型能力，再考虑轻约束） |
| `kg_report_evidence` 返回 `❌ …node_id …不在图内…未落盘` / `kg_decision` 返回 `❌ …reason 必须非空…未落盘` | **特性不是 bug**：非法 node_id 拒绝落盘是防"编造节点污染账本"的硬规则（§4.2）；reason 强制非空是 D2 保留的唯一硬规则（决策落盘纪律，对应 Plan1"放线"事故教训） | 无需处置。若 Planner 频繁编造 node_id：视图"假说全景"与 `kg_next()` 均列出全部合法 id，属模型没读图信息的信号（同开放问题 1 处置路径）；若空 reason 被拒后模型补理由重试成功，正是设计意图 |
