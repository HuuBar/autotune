# 回传报告（T2 知识图谱 Middleware ｜ W-A+W-B+W-C）

## 0. 执行概要
- 时间：2026-07-23 ｜ 分支：feat/kg-middleware（基于 feat/knowledge-pipeline @ ef898da）｜ Python 3.12
- 设计依据：《01_设计书_T2知识图谱Middleware.md》（§4 契约零偏离）；交接包 00 README 纪律全遵守
- 实施：W-A/W-B/W-C 三个工作包顺序施工（各自独立 worktree/分支）+ 独立 reviewer 终审（PASS_WITH_NOTES，1 minor 已修）

## 1. 工作包状态
| 工作包 | 状态 | 内容 | 测试 |
|---|---|---|---|
| W-A | ✅ | OpenAICompatClient 瞬态重试（5xx/连接/超时 ≤2 次指数退避 1s/4s；4xx 与内容类不重试）+ `python -m knowledge_pipeline.compiler` CLI（--live/--cache）+ py>=3.10 声明 | 离线验收命令 exit 0、graph 过 M1、两产物齐全 |
| W-B | ✅ | `agent/kg_middleware.py`（视图注入 + 三工具 + E9 重排）+ config 节 + 启动接线 + Planner 挂载 + kg 导出 | 28 单测（a~f 组全覆盖） |
| W-C | ✅ | V2/V3/V6/V7 集成测试（真实构建链 + mock LLM）+ 实跑检查单 + FAQ 增补 | 17 集成测试；pin 版本（langchain-core 1.2.28/deepagents 0.5.2）复跑通过 |
| 合计 | ✅ | **252 pytest 全绿**（207 既有 + 45 新增，无回归） | 终审复核一致 |

## 2. 契约符合性（设计书 §4 逐项）
- 三工具签名逐字符合 §4.2；非法 node_id 拒绝不落盘；kg_decision 空 reason 拒绝
- 视图：四类内容 + 纪律行；≤2000 字符按 E9 排序截断；**遵循 D2 拍板——无"推荐/建议执行"措辞**（有固化断言测试）
- state.json 六键（§4.4）；账本复用 M3 记录 schema 与 `Ledger.validate_record`，后验更新唯一调用 `ledger.update_posterior`（无公式复刻）
- E9：priority = posterior × Δmid/(recon+run)；refuted→兄弟 boost；blocked→requires 传递冻结；uncertain ×0.5 降权；同分 id 升序；confirmed 0.5→0.95 锚点
- 改动边界：主系统仅 agent.py / factory.py / config.yaml 三处 + kg_middleware.py 新增；prompt/*.md、llm/、tools/、utils/ 零触碰
- **关闭态零行为变化**：enabled 缺省/false 时 middleware 列表、工具集、system_message、日志逐字节一致（V2 测试走真实 build+invoke 链验证）

## 3. 验收标准核对（设计书 §7）
| 项 | 状态 | 证据等级 |
|---|---|---|
| V1 live 编译 | 待服务器实测（CLI 已就绪，离线缓存模式已验证 exit 0 + 过校验） | [live] 待取 |
| V2 关闭兼容 | ✅ pytest（真实链） | [mock] |
| V3 开启启动+视图注入 | ✅ pytest | [mock] |
| V4 真实运行中 Planner 自发调 kg_report_evidence | 待 live | [live] 待取 |
| V5 真实运行中重排/boost 可见 | 待 live | [live] 待取 |
| V6 双策略降级 | ✅ pytest | [mock] |
| V7 导出齐全+schema 合法 | ✅ pytest | [mock] |

**真实回路条款声明**：本批所有"跑通"均为 [mock] 级（机制语义固化）；效果类结论（编排 vs 无编排）本批零证据，不做任何声明；V1/V4/V5 只能由服务器 live 取证，操作步骤见 `docs/knowledge-pipeline/实跑检查单_T1T2.md`。

## 4. 新发现与提请裁决
1. **config.yaml HEAD 既有缺陷**：`api_key: "` 未闭合引号导致主系统配置 YAML 解析失败（base commit 即坏）。已修为 `""`（commit 5631a16），超出"新增节"范围但属必要修复——提请确认。
2. **§4.2"返回 Top3 推荐"与 D2 拍板冲突**：实现返回完整假说全景且无推荐措辞（遵循拍板后口径，FAQ 已登记）。**提请算法团队正式确认此为终稿口径，并把 §4.2 表格措辞对齐。**
3. **kg_decision 分支→假说 boost 的判定**：branch 文本正则命中假说 id（启发式，工程默认值 4）——设计书语义空白，可能漏报/误报；如需结构化，建议 schema 给 decision.branches 加结构化 effects（与上一批新发现-8 同源）。
4. **验收进展口径**（工程默认值 3）：以 acceptance 断言机读结果为准，run 型动作 confirmed 后重估。请算法过目。
5. **环境版本敏感性**：langchain-core 1.5.x 与 pin 的 1.2.28 存在 API 漂移（12 个测试在 1.5.x 下失败，pin 版本全绿）。服务器务必 `pip install -r requirements.txt` 按 pin 安装。
6. E9 参数（uncertain_dampen=0.5、边际口径）沿用工程默认值，待算法给正式规则。

## 5. 协议偏差声明
- 本批严格按设计书 §4 实现，无擅自偏离；两处超出（config 引号修复、FAQ/检查单新增文档）均已登记。
- 沙箱环境不稳定（pip 依赖多次被清空、git lock 残留），均已在施工中恢复；最终状态 252 测试全绿实证。

## 6. 复现
```bash
git checkout feat/kg-middleware
pip install -r requirements.txt && pip install pytest
python -m pytest tests/knowledge_pipeline/ tests/unit/agent/ -q   # 252 passed
python -m knowledge_pipeline.compiler compile --cards docs/knowledge-pipeline/自动化ML优化战术知识卡_v1.md \
  --brief knowledge_pipeline/fixtures/tbox_brief.yaml --out compile_out   # exit 0
# 服务器 live 验收：docs/knowledge-pipeline/实跑检查单_T1T2.md（V1~V7 逐条）
```
