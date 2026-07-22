"""dry-run 调度模拟器：按执行语义 E1~E8 走编排图，输出完整执行轨迹。

语义来源：SPEC §4.2（E1~E8 实现契约）、01 文档 §2.3（节点状态机）与 §4
（执行语义）、03 文档第二部分（逐拍时间线，黄金剧本的复现基准）。

状态机（01 文档 §2.3 + SPEC §4.2 扩展）：
``pending → ready → running → confirmed | refuted | blocked``，
扩展两个工程状态：

- ``blocked_pending``：E5 升级时被冻结的下游（不参与解锁，终报单列）；
- ``skipped``：E3 决策分支显式跳过的节点（判定理由落盘，区别于"静默跳过"）。

工程默认值（SPEC 语义空白处，均列入交付报告「新发现」）：

- E2 缺省 priority 公式中的 delta 区间取中值（SPEC 已标注）；
  ``parse_delta_mid`` 对非数值 delta（如 "阈值方差↓"）返回 0.0。
- E3 分支 then 为自然语言，"then 动作提为最高优先"的机器映射由
  ``executor.branch_effects`` 提供；缺失时退化为 then/else 文本内已知
  action id 扫描。
- E4 在 v1 串行下操作化为：选中节点 X 时，其 no_parallel 对端若仍在
  本 tick 的 ready 候选集中（若在并行世界线里本会同批调度），则对端记
  ``no_parallel_hold`` 并推迟；对端 running 的情形同规则（面向未来）。
- E7 "无验收进展"判定口径：假说裁决时刻重估全部验收断言，相对上一裁决
  时刻无"新通过"断言即记一次无进展；baseline（开跑时）已通过的断言不计。
- E8 账本写回以"假说终局裁决"为记账粒度（03 文档第三部分口径：账本按
  假说记账）；写回与裁决在同一处理步内原子完成，下游解锁发生在后续 tick
  的 E1，故"写回成功才允许下游解锁"自然有序；写回失败 → 假说判 blocked
  并冻结其动作下游。
- 假说终态聚合（SPEC §4.2 一句话规则的展开）：任一动作 blocked → blocked；
  全部非跳过动作 confirmed 后，按 run 产物 ``measured_gain`` 判定——
  末次收益 ≥ 边际阈值 → confirmed；≤ 0 → refuted；(0, 阈值) 且复跑用尽
  → uncertain（"不计入战果"）；无实测收益口径 → confirmed（机制存在性
  假说由侦察/修复成功直接确认，如 H1/H6）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..assertions import Assertion, evaluate, parse_assertion
from ..schema import Decision, Node, iter_decisions, iter_edges, iter_nodes
from ..validator import validate_graph
from .mock_executor import ExecutionResult, MockExecutor
from .trace import Trace

__all__ = [
    "LedgerLike",
    "SimConfig",
    "SimResult",
    "Simulator",
    "parse_delta_mid",
]

# ---- 节点状态（01 文档 §2.3 + 工程扩展） -----------------------------------
PENDING = "pending"
READY = "ready"
RUNNING = "running"
CONFIRMED = "confirmed"
REFUTED = "refuted"
BLOCKED = "blocked"
#: E5：被冻结的下游（SPEC §4.2："下游置 blocked_pending，不参与解锁"）
BLOCKED_PENDING = "blocked_pending"
#: E3：决策分支显式跳过（理由落盘；区别于被禁止的"静默跳过"）
SKIPPED = "skipped"

#: 假说终态词表（SPEC §5 status ∈ confirmed|refuted|uncertain|blocked）
HYPOTHESIS_OUTCOMES = ("confirmed", "refuted", "uncertain", "blocked")


_DELTA_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


class LedgerLike(Protocol):
    """账本结构化协议（SPEC §5 签名子集；M3 ledger.py 由另一工程师并行开发）。

    模拟器只依赖 ``append``：假说裁决时写回一条
    ``{node_id, status, evidence, metrics, posterior, notes, ts}``。
    非法写回（缺字段/非法 status）由 M3 实现拒收（ValueError），模拟器把
    任何 append 异常按 E8 处理（该假说视为 blocked）。
    """

    def append(self, record: dict) -> None: ...


@dataclass(frozen=True)
class SimConfig:
    """模拟器配置。

    - ``k``：E7 早停阈值——连续 k 个假说验证无验收进展则停止（01 文档 §4，
      默认 3，防单轮噪声触发）。
    - ``max_ticks``：防死循环保险（工程默认值）。
    - ``trace_path``：JSONL 轨迹输出路径；``None`` 表示只留内存轨迹。
    """

    k: int = 3
    max_ticks: int = 1000
    trace_path: str | Path | None = None


@dataclass
class SimResult:
    """运行结果（SPEC §4.2：``{trace_path, final_report, hypothesis_outcomes}``）。

    额外携带 ``events``（内存轨迹副本）便于测试与下游分析。
    """

    trace_path: str | None
    final_report: dict[str, Any]
    hypothesis_outcomes: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "final_report": self.final_report,
            "hypothesis_outcomes": self.hypothesis_outcomes,
        }


def parse_delta_mid(delta: Any) -> float:
    """解析 expected.delta 的中值（E2 缺省 priority 的分子）。

    - ``"+0.05~0.20"`` → 0.125（区间取中值，SPEC §4.2-E2 工程默认值）；
    - ``"-10%"`` → 0.1（取效应幅度）；
    - 单个数值 → 其绝对值；
    - 非数值描述（如 ``"阈值方差↓"``）→ 0.0（无法量化，列「新发现」）。
    """
    if isinstance(delta, bool):
        return 0.0
    if isinstance(delta, (int, float)):
        return abs(float(delta))
    text = str(delta)
    nums = [float(m) for m in _DELTA_NUM_RE.findall(text)]
    if not nums:
        return 0.0
    if "%" in text:
        nums = [n / 100.0 for n in nums]
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2.0
    return abs(nums[0])


def _posterior(prior: float, status: str) -> float:
    """后验更新——复用 M3 ``ledger.update_posterior``（单一事实源，防口径漂移）。

    confirmed → p + (1-p)*0.9（锚点：03 文档 H1 0.5→0.95）；
    refuted → p*0.1；uncertain / blocked → 不变。
    """
    from ..ledger import update_posterior

    return update_posterior(prior, status)


class Simulator:
    """dry-run 调度模拟器（SPEC §4.2 接口：``Simulator(...).run() -> SimResult``）。

    构造时先过 M1 校验器（编译门的延续：非法图不进入执行语义）。
    """

    def __init__(
        self,
        graph: dict,
        executor: MockExecutor,
        ledger: LedgerLike,
        config: SimConfig | None = None,
    ):
        errors = validate_graph(graph)
        if errors:
            summary = "; ".join(str(e) for e in errors[:5])
            raise ValueError(f"编排图未通过 M1 校验，拒绝模拟（前 5 条）: {summary}")
        self.graph = graph
        self.executor = executor
        self.ledger = ledger
        self.config = config or SimConfig()

        # ---- 索引 --------------------------------------------------------
        self.nodes: dict[str, Node] = {}
        self.actions: dict[str, Node] = {}
        self.hypotheses: dict[str, Node] = {}
        for node in iter_nodes(graph):
            self.nodes[node.id] = node
            if node.layer == "action":
                self.actions[node.id] = node
            elif node.layer == "hypothesis":
                self.hypotheses[node.id] = node

        self.decisions: dict[str, Decision] = {
            d.id: d for d in iter_decisions(graph)
        }
        self.decisions_by_after: dict[str, list[Decision]] = {}
        for d in self.decisions.values():
            self.decisions_by_after.setdefault(d.after, []).append(d)

        self.requires_in: dict[str, list[str]] = {}
        self.requires_out: dict[str, list[str]] = {}
        self.informs_in: dict[str, list[str]] = {}
        self.no_parallel_peers: dict[str, dict[str, str | None]] = {}
        for edge in iter_edges(graph):
            if edge.type == "requires":
                self.requires_in.setdefault(edge.to_id, []).append(edge.from_id)
                self.requires_out.setdefault(edge.from_id, []).append(edge.to_id)
            elif edge.type == "informs":
                self.informs_in.setdefault(edge.to_id, []).append(edge.from_id)
            elif edge.type == "no_parallel":
                self.no_parallel_peers.setdefault(edge.from_id, {})[edge.to_id] = edge.reason
                self.no_parallel_peers.setdefault(edge.to_id, {})[edge.from_id] = edge.reason

        # hypothesis → 下属 action
        self.hyp_actions: dict[str, list[str]] = {h: [] for h in self.hypotheses}
        for action in self.actions.values():
            owner = action.get("belongs_to")
            if owner in self.hyp_actions:
                self.hyp_actions[owner].append(action.id)

        # 验收断言（goal.acceptance 已过 V-ACCEPTANCE，解析必然成功）
        self.acceptance: list[Assertion] = [
            parse_assertion(s) for s in (graph.get("goal", {}).get("acceptance") or [])
        ]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _priority(self, node: Node) -> float:
        """E2 选序优先级：显式 ``priority`` 优先；缺省 = 预期收益中值/验证成本。

        收益/成本取自所属假说（SPEC §4.2-E2）；直属于 domain 的动作无
        假说标注，工程默认值取 0.0。
        """
        explicit = node.get("priority")
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            return float(explicit)
        owner = self.hypotheses.get(str(node.get("belongs_to", "")))
        if owner is None:
            return 0.0
        expected = owner.get("expected") or {}
        cost = owner.get("cost") or {}
        gain = parse_delta_mid(expected.get("delta"))
        total_cost = float(cost.get("recon_steps", 0)) + float(cost.get("run_minutes", 0))
        if total_cost <= 0:
            return 0.0
        return gain / total_cost

    def _hypothesis_of(self, action_id: str) -> str | None:
        owner = str(self.actions[action_id].get("belongs_to", ""))
        return owner if owner in self.hypotheses else None

    def _requires_descendants(self, start: str) -> list[str]:
        """requires 边的传递下游（仅 action 节点；decision 由 after 触发，自然死亡）。"""
        seen: list[str] = []
        stack = list(self.requires_out.get(start, []))
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.append(nid)
            stack.extend(self.requires_out.get(nid, []))
        return [n for n in seen if n in self.actions]

    def _acceptance_status(self) -> tuple[set[str], list[dict[str, Any]]]:
        """对当前环境事实评估全部验收断言，返回 (通过集合, 逐条明细)。"""
        facts = self.executor.facts
        passed: set[str] = set()
        detail: list[dict[str, Any]] = []
        for assertion in self.acceptance:
            entry: dict[str, Any] = {"assertion": str(assertion)}
            try:
                ok = evaluate(assertion, facts)
                entry["passed"] = ok
                entry["actual"] = _safe_lookup(facts, assertion.metric)
            except (KeyError, TypeError) as exc:
                ok = False
                entry["passed"] = False
                entry["error"] = str(exc)
            if ok:
                passed.add(str(assertion))
            detail.append(entry)
        return passed, detail

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> SimResult:
        """逐 tick 走图（v1 串行），返回 :class:`SimResult`。"""
        cfg = self.config
        trace = Trace(cfg.trace_path)
        state: dict[str, str] = {nid: PENDING for nid in self.nodes}
        artifacts: dict[str, dict[str, Any]] = {}
        params: dict[str, dict[str, Any]] = {nid: {} for nid in self.actions}
        informed: set[tuple[str, str]] = set()  # (source, target) 已注入
        boosted: set[str] = set()
        fired_decisions: set[str] = set()
        retries_left: dict[str, int] = {
            nid: int(a.get("retry_budget", 0)) for nid, a in self.actions.items()
        }
        reruns_used: dict[str, int] = {}
        gain_history: dict[str, list[float]] = {}  # action → 历次实测收益
        hyp_verdicts: dict[str, str] = {}
        hyp_notes: dict[str, str] = {}
        early_stop_info: dict[str, Any] | None = None
        end_reason = "graph_exhausted"

        rules = getattr(self.executor, "outcome_rules", None) or {}
        marginal_threshold = float(rules.get("marginal_gain_threshold", 0.05))
        marginal_reruns = int(rules.get("marginal_reruns", 1))

        # E7 baseline：开跑时已通过的断言不算"进展"
        passed, _ = self._acceptance_status()
        streak = 0
        tick = 0

        def emit(event: str, node_id: str | None = None, **detail: Any) -> None:
            trace.emit(tick, event, node_id, detail)

        def inject_informs(target: str) -> None:
            """E6：把已 confirmed 的 informs 源产物注入目标 params（解锁/执行前）。"""
            for source in self.informs_in.get(target, []):
                if state.get(source) == CONFIRMED and (source, target) not in informed:
                    artifact = artifacts.get(source) or {}
                    name = str(self.actions[source].get("produces", source))
                    params[target][name] = artifact
                    informed.add((source, target))
                    emit("param_update", target, source=source, artifact=name)

        def freeze_downstream(node_id: str, cause: str) -> list[str]:
            """E5/E8：冻结 requires 传递下游 → blocked_pending（不静默跳过）。"""
            frozen: list[str] = []
            for desc in self._requires_descendants(node_id):
                if state[desc] in (PENDING, READY):
                    state[desc] = BLOCKED_PENDING
                    frozen.append(desc)
                    emit(
                        "blocked",
                        desc,
                        state=BLOCKED_PENDING,
                        cause=cause,
                        reason=f"上游 {node_id} {cause}，冻结不参与解锁",
                    )
            return frozen

        def verdict_hypothesis(hyp: str, verdict: str, notes: str) -> None:
            """假说终局裁决 + E8 账本写回 + E7 早停计数。"""
            nonlocal streak, passed, early_stop_info
            if hyp in hyp_verdicts:
                return
            hyp_verdicts[hyp] = verdict
            hyp_notes[hyp] = notes
            state[hyp] = verdict
            gains = [
                g
                for aid in self.hyp_actions.get(hyp, [])
                for g in gain_history.get(aid, [])
            ]
            # E8：先写回账本；写回失败视为该节点 blocked
            record = {
                "node_id": hyp,
                "status": verdict,
                "evidence": [
                    f"{aid}:{self.actions[aid].get('produces', aid)}"
                    for aid in self.hyp_actions.get(hyp, [])
                    if state.get(aid) == CONFIRMED
                ],
                "metrics": {
                    # 规范键名 "gain"（SPEC §5 裁决：账本收益键统一，与
                    # ledger.domain_report 默认 gain_key 对齐）；playbook 产物层的
                    # "measured_gain" 是环境事实键，写回时归一化为 "gain"。
                    "gain": gains[-1] if gains else None,
                    "gains": gains,
                    "expected_metric": (self.hypotheses[hyp].get("expected") or {}).get(
                        "metric"
                    ),
                },
                "posterior": _posterior(
                    float(self.hypotheses[hyp].get("prior", 0.5)), verdict
                ),
                "notes": notes,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            try:
                self.ledger.append(record)
            except Exception as exc:  # noqa: BLE001 — E8：任何写回失败都按 blocked 处理
                hyp_verdicts[hyp] = BLOCKED
                hyp_notes[hyp] = f"账本写回失败: {exc}"
                state[hyp] = BLOCKED
                frozen: list[str] = []
                for aid in self.hyp_actions.get(hyp, []):
                    frozen.extend(freeze_downstream(aid, "writeback_failed"))
                emit(
                    "escalate",
                    hyp,
                    cause="writeback_failed",
                    error=str(exc),
                    frozen=sorted(set(frozen)),
                    reason="E8：账本写回失败，假说视为 blocked 并冻结下游",
                )
                return
            emit("writeback", hyp, record=record)

            # E7：验收进展判定（每次假说裁决后重估）
            new_passed, detail = self._acceptance_status()
            progress = sorted(new_passed - passed)
            emit(
                "acceptance_check",
                None,
                newly_passed=progress,
                passed_count=len(new_passed),
                total=len(self.acceptance),
                streak_before=streak,
                final=False,
            )
            if progress:
                streak = 0
                passed = new_passed
            else:
                streak += 1
            if streak >= cfg.k and early_stop_info is None:
                early_stop_info = {
                    "k": cfg.k,
                    "streak": streak,
                    "reason": f"连续 {streak} 个假说验证无验收进展（k={cfg.k}）",
                    "at_hypothesis": hyp,
                }
                emit("early_stop", None, **early_stop_info)

        def execute_with_retries(node: Node, rerun: bool = False) -> ExecutionResult | None:
            """执行 + E5 重试/阻塞升级。返回 None 表示节点已 blocked。"""
            attempt = 0
            result = self.executor.execute(node, params[node.id])
            attempt += 1
            emit(
                "execute",
                node.id,
                attempt=attempt,
                rerun=rerun,
                verify_passed=result.verify_passed,
                log=result.log,
            )
            while not result.verify_passed:
                emit(
                    "verify_fail",
                    node.id,
                    attempt=attempt,
                    retries_left=retries_left[node.id],
                    log=result.log,
                )
                if retries_left[node.id] <= 0:
                    # E5：重试预算耗尽 → blocked + 升级（禁止静默跳过）
                    state[node.id] = BLOCKED
                    emit(
                        "blocked",
                        node.id,
                        state=BLOCKED,
                        attempts=attempt,
                        reason="verify 持续失败，retry_budget 耗尽",
                    )
                    frozen = freeze_downstream(node.id, "blocked")
                    emit(
                        "escalate",
                        node.id,
                        cause="retry_exhausted",
                        frozen=frozen,
                        reason="阻塞升级：冻结下游、继续其他 domain、终报单列未决阻塞",
                    )
                    hyp = self._hypothesis_of(node.id)
                    if hyp is not None:
                        verdict_hypothesis(hyp, BLOCKED, f"动作 {node.id} 重试预算耗尽")
                    return None
                retries_left[node.id] -= 1
                emit(
                    "retry",
                    node.id,
                    attempt=attempt + 1,
                    retries_left=retries_left[node.id],
                )
                result = self.executor.execute(node, params[node.id])
                attempt += 1
                emit(
                    "execute",
                    node.id,
                    attempt=attempt,
                    rerun=rerun,
                    verify_passed=result.verify_passed,
                    log=result.log,
                )
            return result

        def fire_decision(decision: Decision) -> None:
            """E3：决策点判定（条件判定结果与理由强制落盘）。"""
            fired_decisions.add(decision.id)
            branch_index, reason = self.executor.resolve_branch(decision, artifacts)
            branches = decision.branches
            if branch_index < 0 or branch_index >= len(branches):
                # 无规则命中：图内有 else → 走 else 兜底；无 else → decision_deadend
                else_idx = next(
                    (i for i, b in enumerate(branches) if "else" in b), None
                )
                if else_idx is None:
                    emit(
                        "decision_deadend",
                        decision.id,
                        after=decision.after,
                        reason=reason,
                        note="未命中任何 if 且无 else 兜底，按 E3 跳过（不发生判 blocked 的替代）",
                    )
                    return
                branch_index = else_idx
                reason = f"{reason}；走图内 else 兜底分支"
            branch = branches[branch_index]
            effects = {"boost": [], "skip": []}
            branch_effects = getattr(self.executor, "branch_effects", None)
            if callable(branch_effects):
                effects = branch_effects(decision.id, branch_index)
            else:  # 退化：then/else 文本扫描已知 action id（列「新发现」）
                text = str(branch.get("then", branch.get("else", "")))
                effects = {
                    "boost": [a for a in self.actions if re.search(rf"\b{a}\b", text)],
                    "skip": [],
                }
            condition = branch.get("if", "else")
            action_desc = branch.get("then", branch.get("else", ""))
            boosted.update(b for b in effects["boost"] if b in self.actions)
            for skip_id in effects["skip"]:
                if skip_id in self.actions and state[skip_id] in (PENDING, READY):
                    state[skip_id] = SKIPPED
            emit(
                "decision",
                decision.id,
                after=decision.after,
                read=decision.read,
                branch_index=branch_index,
                condition=condition,
                action=action_desc,
                reason=reason,  # 判定理由强制落盘（01 文档 §2.4）
                boost=list(effects["boost"]),
                skip=list(effects["skip"]),
            )

        def maybe_verdict(hyp: str) -> None:
            """假说级状态聚合（SPEC §4.2；展开规则见模块 docstring）。"""
            if hyp in hyp_verdicts:
                return
            acts = [a for a in self.hyp_actions.get(hyp, []) if state[a] != SKIPPED]
            if not acts:
                return
            if any(state[a] == BLOCKED for a in acts):
                blocked_at = next(a for a in acts if state[a] == BLOCKED)
                verdict_hypothesis(hyp, BLOCKED, f"动作 {blocked_at} blocked")
                return
            if not all(state[a] == CONFIRMED for a in acts):
                return
            gains = [
                g
                for aid in acts
                for g in gain_history.get(aid, [])
            ]
            if not gains:
                verdict_hypothesis(
                    hyp, CONFIRMED, "无实测收益口径，全部动作 confirmed（机制性确认）"
                )
                return
            final_gain = gains[-1]
            if final_gain >= marginal_threshold:
                verdict_hypothesis(
                    hyp, CONFIRMED, f"实测收益 {final_gain:+.2f} ≥ 边际阈值 {marginal_threshold}"
                )
            elif final_gain <= 0:
                verdict_hypothesis(hyp, REFUTED, f"实测收益 {final_gain:+.2f} ≤ 0")
            else:
                verdict_hypothesis(
                    hyp,
                    "uncertain",
                    f"边际提升（实测 {final_gain:+.2f} < 阈值 {marginal_threshold}，"
                    f"复跑 {reruns_used.get(hyp, 0)} 次后仍未达阈），不计入战果",
                )

        # ---- 逐 tick 循环（v1 串行：每 tick 执行一个节点） ----------------
        while True:
            active = [
                nid for nid in self.actions if state[nid] in (PENDING, READY)
            ]
            if not active:
                break
            tick += 1
            if tick > cfg.max_ticks:
                end_reason = "max_ticks_exceeded"
                break

            # E1 解锁：全部 requires 源 confirmed（产物已落盘）→ ready
            for nid in list(active):
                if state[nid] != PENDING:
                    continue
                sources = self.requires_in.get(nid, [])
                if all(state.get(s) == CONFIRMED for s in sources):
                    inject_informs(nid)  # E6：目标解锁前注入已就位的 informs 产物
                    state[nid] = READY
                    emit("unlock", nid, requires=sources)

            # E6：目标已 ready 而 informs 源在其后 confirmed 的情形，执行前补注入
            for nid in active:
                if state[nid] == READY:
                    inject_informs(nid)

            ready = sorted(
                (nid for nid in self.actions if state[nid] == READY),
                key=lambda n: (
                    0 if n in boosted else 1,  # E3：命中分支的 then 动作提为最高优先
                    -self._priority(self.actions[n]),  # E2：priority 降序
                    n,  # 确定性 tie-break（工程默认值：id 升序）
                ),
            )
            if not ready:
                # 无 ready 但仍有 pending（前置永不可达，如上游被分支跳过）→ 停滞收尾
                end_reason = "stalled"
                break

            chosen = ready[0]
            # E4：no_parallel 对端在 ready 候选集（同批）或 running → 推迟并记录
            for other in ready[1:]:
                reason = self.no_parallel_peers.get(chosen, {}).get(other)
                if reason is not None:
                    emit(
                        "no_parallel_hold",
                        other,
                        held_by=chosen,
                        reason=reason or "no_parallel 约束",
                    )
            for peer, reason in self.no_parallel_peers.get(chosen, {}).items():
                if state.get(peer) == RUNNING:  # 面向未来的并行情形
                    emit("no_parallel_hold", chosen, held_by=peer, reason=reason or "")

            # E2 select → running → 执行
            state[chosen] = RUNNING
            emit(
                "select",
                chosen,
                priority=round(self._priority(self.actions[chosen]), 6),
                boosted=chosen in boosted,
            )
            node = self.actions[chosen]
            result = execute_with_retries(node)
            if result is None:
                continue  # 已 blocked + 升级

            artifacts[chosen] = result.artifact or {}
            state[chosen] = CONFIRMED
            boosted.discard(chosen)
            gain = (result.artifact or {}).get("measured_gain")
            if isinstance(gain, (int, float)) and not isinstance(gain, bool):
                gain_history.setdefault(chosen, []).append(float(gain))

            # 边际提升复跑（03 文档 T+9~10：+0.03 → 复跑 +0.01 → uncertain）
            hyp = self._hypothesis_of(chosen)
            while (
                hyp is not None
                and isinstance(gain, (int, float))
                and not isinstance(gain, bool)
                and 0 < float(gain) < marginal_threshold
                and reruns_used.get(hyp, 0) < marginal_reruns
                and node.get("type") == "run"
            ):
                reruns_used[hyp] = reruns_used.get(hyp, 0) + 1
                rerun_result = execute_with_retries(node, rerun=True)
                if rerun_result is None:
                    break
                artifacts[chosen] = rerun_result.artifact or {}
                gain = (rerun_result.artifact or {}).get("measured_gain")
                if isinstance(gain, (int, float)) and not isinstance(gain, bool):
                    gain_history.setdefault(chosen, []).append(float(gain))

            if state[chosen] != CONFIRMED:
                continue  # 复跑中耗尽重试预算而 blocked

            # E3：decision 的 after 动作完成 → 读产物 → 判定分支
            for decision in self.decisions_by_after.get(chosen, []):
                if decision.id in fired_decisions:
                    continue
                if all(
                    state.get(s) == CONFIRMED
                    for s in self.requires_in.get(decision.id, [])
                ):
                    fire_decision(decision)

            # 假说终态聚合（含 E8 写回、E7 早停计数）
            if hyp is not None:
                maybe_verdict(hyp)
            if early_stop_info is not None:
                end_reason = "early_stop"
                break

        # ---- 收尾：终局验收核对 + 终报 ------------------------------------
        final_passed, acceptance_detail = self._acceptance_status()
        emit(
            "acceptance_check",
            None,
            newly_passed=[],
            passed_count=len(final_passed),
            total=len(self.acceptance),
            final=True,
            results=acceptance_detail,
        )
        blocked_nodes = sorted(n for n in self.actions if state[n] == BLOCKED)
        blocked_pending = sorted(
            n for n in self.actions if state[n] == BLOCKED_PENDING
        )
        skipped = sorted(n for n in self.actions if state[n] == SKIPPED)
        pending_unresolved = sorted(n for n in self.actions if state[n] == PENDING)
        emit(
            "run_end",
            None,
            reason=end_reason,
            ticks=tick,
            acceptance_passed=f"{len(final_passed)}/{len(self.acceptance)}",
            unresolved_blockers=blocked_nodes,
        )
        trace.close()

        hypothesis_outcomes = {
            hyp: {
                "status": hyp_verdicts.get(hyp, "unverified"),
                "notes": hyp_notes.get(hyp, ""),
                "actions": {a: state[a] for a in self.hyp_actions.get(hyp, [])},
            }
            for hyp in self.hypotheses
        }
        final_report = {
            "goal": self.graph.get("goal", {}).get("statement", ""),
            "end_reason": end_reason,
            "ticks": tick,
            "acceptance": acceptance_detail,
            "acceptance_passed": {
                "passed": len(final_passed),
                "total": len(self.acceptance),
            },
            # 未决阻塞（03 文档失败路径：终报单列，交人裁决）
            "unresolved_blockers": blocked_nodes,
            "blocked_pending": blocked_pending,
            "skipped": skipped,
            "pending_unresolved": pending_unresolved,
            "early_stop": early_stop_info,
            "unverified_hypotheses": sorted(
                h for h in self.hypotheses if h not in hyp_verdicts
            ),
        }
        return SimResult(
            trace_path=str(trace.path) if trace.path is not None else None,
            final_report=final_report,
            hypothesis_outcomes=hypothesis_outcomes,
            events=trace.events(),
        )


def _safe_lookup(facts: dict[str, Any], metric: str) -> Any:
    """点号路径取值（缺失返回 None，仅供报告展示）。"""
    if metric in facts:
        return facts[metric]
    cur: Any = facts
    for part in metric.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur
