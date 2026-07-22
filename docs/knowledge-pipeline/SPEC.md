# SPEC — knowledge_pipeline（知识工程流水线前端）接口契约

> 本文件是 M1~M6 实现的**唯一接口事实源**。语义来源：`01_知识工程架构设计_v1.md`、`03_架构全景_时间线推演与量化评测.md`、`02_示例编排图_tbox.yaml`（黄金夹具）。
> 纪律：本 SPEC 只固化"实现契约"，不改动上游语义；语义空白处以【工程默认值】标注，并须列入回传报告「新发现」。
> 工程基线：Python ≥3.10；核心逻辑仅依赖标准库 + PyYAML；测试用 pytest；LLM 调用走可替换客户端接口，端点由配置/环境变量注入，不写死。

## 0. 包结构（新增，不改动 autotune 既有后端代码）

```
knowledge_pipeline/
  __init__.py            # 导出版本号与公共类型
  schema.py              # M1：图文档加载/规范化（YAML→dict + 结构常量）
  validator.py           # M1：确定性校验器
  assertions.py          # M1：验收断言解析器（机器可判）
  simulator/
    __init__.py
    mock_executor.py     # M2：mock 执行器（预制产物、可配成功/失败序列）
    scheduler.py         # M2：dry-run 调度器（E1~E8）
    trace.py             # M2：执行轨迹日志（JSONL）
  ledger.py              # M3：假说账本 + 后验更新 + 领域聚合
  compiler/
    __init__.py
    llm_client.py        # M4：OpenAI 兼容客户端协议 + 缓存客户端 + 伪造客户端
    prompts.py           # M4：编译提示词模板
    compiler.py          # M4：编排编译器（选卡论证 → 成图 → 校验门）
  bench/
    __init__.py
    scripts.py           # M6：事故剧本库 S1~S6
    baseline_agent.py    # M6：无编排对照（确定性启发式基线）
    metrics.py           # M6：预登记指标计算
  e2e/
    __init__.py
    run_pipeline.py      # M5：一键端到端（编译→校验→模拟→回写→聚合）
  fixtures/
    tbox_golden.yaml            # 黄金夹具（已归档）
    bad_graphs/bad_01..06.yaml  # ≥6 个坏图
    tbox_brief.yaml             # TBox 任务书 + 地形描述（M4 输入）
tests/knowledge_pipeline/
  test_validator.py test_simulator.py test_ledger.py test_compiler.py
  test_e2e.py test_bench.py
```

## 1. 编排图 schema（M1）

顶层键：`version`(str) / `meta` / `goal` / `nodes` / `edges` / `decisions` / `writeback`。

- `meta`: `{task: str, created_by: str, source_cards: list[str]}`。
- `goal`: `{statement: str, acceptance: list[str]}`。acceptance 每条必须被 assertions 解析（见 §2）。
- `nodes[]` 公共字段：`id`(str, 全图唯一) / `layer` / `title`(str)。
  - `layer: domain` → `rationale: str`。
  - `layer: hypothesis` → `domain`(引用 domain 节点 id) / `mechanism: str` / `prior: float∈[0,1]` / `expected: {metric, delta, confidence}` / `cost: {recon_steps: int, run_minutes: number}`。
  - `layer: action` → `belongs_to`(引用 hypothesis 或 domain 节点 id) / `type: recon|edit|run` / `do: str` / `tool_hint: str` / `produces: str`（产物名）/ `verify: str`（成功判据，非空）/ `retry_budget: int≥0`。
  - 可选 `priority: float`（编译器标注；缺省时调度器按 §4-E2 计算）。
- `edges[]`: `{from, to, type: requires|informs|alternatives|no_parallel, reason?: str}`。from/to 必须是已声明的节点 id 或 decision id。
- `decisions[]`: `{id, after: <action id>, read: <产物名>, branches: [{if: str, then: str}...]}`，分支可用 `else` 键替代 `if` 表示兜底。
- `writeback`: `{ledger: str(路径), record_schema: str, after_domain_review: str}`。

L0 目标层以 `goal` 顶层字段表达（与黄金夹具一致），不占 nodes。

## 2. 验收断言解析（M1, assertions.py）

语法：`<metric_path> <op> <number>`；op ∈ `>=, <=, ==, >, <`；metric_path 允许点号、中文、字母、数字、下划线（如 `per_app.王者荣耀.F1`、`per_app.bilibili.F1_delta`）。
接口：`parse_assertion(s: str) -> Assertion(metric, op, value)`，解析失败抛 `AssertionParseError`；`evaluate(assertion, facts: dict) -> bool`（facts 支持点号路径取值，`_delta` 后缀表示相对基线差值——取 `facts[metric_base + "_delta"]` 或调用方提供）。

## 3. 校验器（M1, validator.py）——确定性检查清单

接口：`validate_graph(doc: dict) -> list[ValidationError]`；`validate_file(path) -> list[ValidationError]`；`ValidationError` 含 `code, message, path`（人类可读）。**全部检查通过 = 返回空列表**。

必查项（对应交接验收）：
1. V-SCHEMA：顶层键齐全、类型正确；节点/边/决策的必填字段齐全。
2. V-UNIQ：节点 id 唯一；decision id 与节点 id 不冲突。
3. V-DAG：全部边（requires/informs/alternatives）构成的图上**无环**（no_parallel 是无向约束，不参与有向环检测）。
4. V-REQUIRES-EXISTS：每条 requires/informs 边的 from/to 指向存在节点；requires 不得指向 decision（decision 由 `after` 触发）。
5. V-ACTION-VERIFY：每个 action 有非空 `verify` 且有 `retry_budget`。
6. V-ALT-GROUP：alternatives 边成组完整——同一 alternatives 关系涉及 ≥2 个节点，且同组节点属于同一 hypothesis 的下游分支或同一 domain；组内节点不得再有 requires 相互依赖（互斥分支不能互为前置）。
7. V-ACCEPTANCE：goal.acceptance 每条可被 §2 解析。
8. V-DECISION：每个 decision 的 `after` 指向存在 action、`read` 等于该 action 的 `produces`；分支条件完备——**每个分支有 (if|else) 与 then，且（存在 else 兜底 或 分支数 ≥2）**【工程默认值：完备性按结构判定，语义穷尽性交由编译器自检与人工评审，列入新发现】。
9. V-LAYER-REF：hypothesis.domain 指向 domain 层节点；action.belongs_to 指向存在的 hypothesis 或 domain 节点。
10. V-PRIOR：hypothesis.prior ∈ [0,1]。

坏图夹具 ≥6 个，每个触发不同检查（环、requires 指向不存在、action 缺 verify、alternatives 组残缺、acceptance 不可解析、decision 分支不完备/指向不存在），测试断言全部被拦截且 message 可读。

## 4. dry-run 调度模拟器（M2）

### 4.1 MockExecutor
```python
class MockExecutor:
    def __init__(self, playbook: dict): ...
    # playbook: {action_id: {"artifacts": [...候选产物序列...],
    #                        "fail_sequence": [bool...],   # 每次尝试是否 verify 失败
    #                        "facts": {...}}}              # 环境事实（供决策与验收评估）
    def execute(self, action: Node, params: dict) -> ExecutionResult
    # ExecutionResult = {ok: bool, artifact: dict|None, verify_passed: bool, log: str}
    def resolve_branch(self, decision: Decision, artifacts: dict) -> tuple[int, str]
    # 按 playbook 中决策事实选择分支索引，返回 (branch_index, 理由)；理由强制落盘
```
- 同一 action 多次尝试按 fail_sequence 依次消费；序列用尽后取最后一项。

### 4.2 调度器（scheduler.py）——E1~E8 实现契约
节点状态机：`pending → ready → running → confirmed | refuted | blocked`（blocked 为终态且升级，**禁止静默跳过**）。
逐 tick 循环（v1 串行）：
- **E1 解锁**：action 的全部 requires 源节点状态 == confirmed 且产物已落盘 → ready。
- **E2 选序**：ready 集合中按 priority 降序；priority 缺省 = `parse_delta_mid(expected.delta) / (cost.recon_steps + cost.run_minutes)`【工程默认值：delta 区间取中值】。
- **E3 决策点**：decision 的 after 动作完成 → 读产物 → resolve_branch → 命中分支的 then 动作提为最高优先；未命中任何 if 且无 else → 记 `decision_deadend` 事件并按 else 语义跳过（判 blocked 的替代不发生）。【判定结果 + 理由写入决策日志】
- **E4 no_parallel**：v1 串行不共调度；调度器在选中某节点时检查其 no_parallel 对端是否在 running/已选中批次，若是则推迟并在轨迹记 `no_parallel_hold`（含 reason）。
- **E5 每动作必验**：执行后按 playbook 判定 verify；失败 → 消耗 retry_budget 重试；耗尽 → blocked + `escalate` 事件：冻结其下游（下游置 blocked_pending，不参与解锁）、继续其他 domain、终报单列「未决阻塞」。
- **E6 informs 回写**：informs 源 confirmed 后、目标解锁前，将源产物注入目标 params，轨迹记 `param_update`。
- **E7 早停**：连续 k（默认 3，可配）个假说验证**无验收进展**（acceptance 评估无新通过项）→ `early_stop` 并报告。
- **E8 写回前置**：节点 confirmed/refuted 后先完成账本写回（调用 M3 Ledger.append），写回成功才允许下游解锁；写回失败视为该节点 blocked。
- 假说级状态聚合：hypothesis 由其下属全部 action 终态推出（全部 confirmed→confirmed；任一 blocked→blocked；按 playbook 实测收益判定 confirmed/refuted/uncertain——见 §5 状态词表）。
- 运行结束（图耗尽或早停）输出 `Trace`（JSONL 事件流）+ 终报 dict（验收断言核对结果、blocked 清单、早停原因）。
接口：`Simulator(graph: dict, executor: MockExecutor, ledger: Ledger, config: SimConfig).run() -> SimResult`；`SimResult = {trace_path, final_report, hypothesis_outcomes}`。

### 4.3 轨迹（trace.py）
JSONL，每事件一行：`{tick, event, node_id?, detail}`；事件词表：`unlock/select/execute/verify_fail/retry/blocked/escalate/decision/param_update/no_parallel_hold/writeback/early_stop/acceptance_check/run_end`。

## 5. 假说账本（M3, ledger.py）

- 存储：`hypotheses.jsonl`，append-only；记录字段（对齐 writeback.record_schema 扩展）：
  `{node_id, status, evidence, metrics, posterior, notes, ts}`，其中 status ∈ `confirmed|refuted|uncertain|blocked`；metrics 为 dict（数据不是文本）；ts 为 ISO8601。
- 接口：
  ```python
  class Ledger:
      def __init__(self, path): ...
      def append(self, record: dict) -> None        # 缺必填字段/非法 status → ValueError（非法写回拒绝）
      def records(self) -> list[dict]
  def update_posterior(prior: float, status: str) -> float
  def domain_report(records: list[dict], graph: dict) -> dict  # 按领域聚合收益与状态计数，给下轮 prior 调整建议
  ```
- 后验更新【工程默认值，列入新发现】：confirmed → `p + (1-p)*0.9`；refuted → `p*(1-0.9)`；uncertain → 不变（notes 记"不计入战果"）；blocked → 不变。
  （锚点：03 文档 H1 prior 0.5 confirmed → 0.95，与系数 0.9 一致。）
- `domain_report` 输出：`{domain_id: {title, confirmed, refuted, uncertain, blocked, total_gain, prior_suggestion}}`；prior_suggestion ∈ `up|down|hold`（confirmed≥1 且有正收益→up；refuted≥1 或 uncertain≥1 且无正收益→down；未探索→hold）。

## 6. 编排编译器（M4, compiler/）

- 输入：`cards_path`（知识卡 md）、`brief_path`（任务书+地形 YAML）、`config: {endpoint, model, api_key_env, cache_path?}`。
- 流程（两步 LLM 调用 + 确定性校验门）：
  1. **选卡**：LLM 输出采用/拒绝论证表（每张卡 `{card_id, decision: adopt|reject, rationale}`）。
  2. **成图**：LLM 输出编排图 YAML。
  3. **校验门**：产出图过 `validator.validate_graph`，不过 → 把错误列表回灌 LLM 重试（默认 ≤3 次），仍不过 → 编译失败（**不绕过校验**）。
  4. 编译器自检：每个假说有 mechanism+prior；领域层 ≥3 个正交方向（数量检查 + rationale 非空）。
- `llm_client.py`：
  ```python
  class LLMClient(Protocol):
      def complete(self, system: str, user: str) -> str: ...
  class OpenAICompatClient  # base_url/model 由构造参数或 KP_LLM_BASE_URL/KP_LLM_API_KEY 注入
  class CachedClient        # 从 cache_path 读预录响应（按 prompt 哈希寻址），e2e 离线用
  class FakeClient          # 测试用，按注册表返回
  ```
- 接口：`compile_graph(cards_path, brief_path, client: LLMClient, validator=validate_graph, max_retries=3) -> CompileResult`；`CompileResult = {graph: dict, graph_yaml: str, card_decisions: list[dict], attempts: int}`。
- 自检要求（验收）：产出图领域数 ≥3；每个假说 mechanism 与 prior 非空；card_decisions 覆盖卡库全部卡片（adopt 或 reject 均须有 rationale）。

## 7. 端到端（M5, e2e/run_pipeline.py）

一键脚本 `scripts/run_e2e.sh`（或 `python -m knowledge_pipeline.e2e.run_pipeline`）：
编译（默认 CachedClient + 预录 TBox 编译输出，可用 `--live` 走真实端点）→ 校验 → 模拟执行（内置 TBox 事故剧本 playbook，复现 03 文档时间线：断链修复、重复列、J2 走"阈值+K折升级"分支、H3 边际提升复跑判 uncertain、验收通过）→ 回写账本 → 领域聚合报告。
产物目录 `e2e_out/`：`graph.yaml / trace.jsonl / decisions.log / hypotheses.jsonl / domain_report.json / acceptance_report.json`；退出码 0 = 全链通过。

## 8. 量化评测台（M6, bench/）

- `scripts.py`：事故剧本库 S1~S6（断链/重复列/脆弱JSON/小样本阈值陷阱/边际提升噪声/沙箱依赖缺失），每剧本 `{id, trigger, env_response, expected_response}`，对双方完全一致、可重复。
- `baseline_agent.py`：无编排对照——确定性启发式自由探索 agent（仅有任务书；按固定策略侦察-尝试-评估，无前置解锁/无决策点/无早停规则外的停止）。【工程默认值：离线 L1 用启发式基线替代 LLM ReAct，列入新发现】
- `metrics.py` 预登记指标：到达验收总步数 / 信息缺失型事故数 / 探索方向覆盖数 / 决策点分支正确率 / S4 陷阱识别 / S5 噪声当战果次数 / 盲抄次数。按剧本独立计分再聚合。
- 输出 `bench_report.json`：每剧本 × 双方 × 指标 + 汇总。

## 9. 提交与质量纪律

- 分支：`feat/knowledge-pipeline`；子代理在各自 worktree 分支（feat/kp-m1 …）提交，主代理合并。
- 每模块独立 commit：`feat(M1): ...` / `test(M2): ...`（Conventional Commits 带模块号）。
- 每模块交付前自跑 `pytest tests/knowledge_pipeline/test_<module>.py` 全绿。
- 任何与 01/03 文档冲突或 SPEC 未覆盖的语义空白 → 写入回传报告「新发现」，不擅自改设计。
