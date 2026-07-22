"""M2 dry-run 调度模拟器测试。

覆盖（语义来源：SPEC §4 / 01 文档 §2.3、§4 / 03 文档第二部分）：

- E1~E8 逐条测试（测试名标注 E 编号）；
- 黄金夹具 + tbox_playbook 全链演示：首批解锁顺序、J1/J2 分支（改 playbook
  事实可切换）、no_parallel_hold、失败→重试→blocked→escalate（下游冻结、
  终报含未决阻塞、无静默跳过）、早停（k=3）、验收 5 条全过；
- mock 执行器可配置成功/失败序列、trace JSONL 词表、账本协议 stub。

账本依赖：SPEC §5 的 Ledger 由 M3 并行开发；此处用内存 stub 实现
``LedgerLike`` 协议（``append(record: dict)``），不 import ledger.py。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from knowledge_pipeline import load_graph
from knowledge_pipeline.simulator import (
    EVENT_VOCAB,
    MockExecutor,
    PlaybookError,
    SimConfig,
    Simulator,
    Trace,
    TraceEventError,
    load_playbook,
    parse_delta_mid,
    read_trace,
)

FIXTURES = Path(__file__).resolve().parents[2] / "knowledge_pipeline" / "fixtures"
GOLDEN = FIXTURES / "tbox_golden.yaml"
PLAYBOOK = FIXTURES / "tbox_playbook.yaml"
BAD_CYCLE = FIXTURES / "bad_graphs" / "bad_01_cycle.yaml"


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------


class InMemoryLedger:
    """LedgerLike 协议内存 stub（SPEC §5 append 签名）。"""

    def __init__(self, fail: bool = False):
        self.records: list[dict] = []
        self.fail = fail

    def append(self, record: dict) -> None:
        if self.fail:
            raise RuntimeError("ledger unavailable")
        self.records.append(record)


def make_action(aid, belongs_to, atype="recon", produces=None, retry_budget=1,
                priority=None):
    node = {
        "id": aid,
        "layer": "action",
        "belongs_to": belongs_to,
        "type": atype,
        "do": f"do {aid}",
        "tool_hint": "mock",
        "produces": produces or f"out_{aid.lower()}",
        "verify": "mock 判定",
        "retry_budget": retry_budget,
    }
    if priority is not None:
        node["priority"] = priority
    return node


def make_hypothesis(hid, domain="D1", prior=0.5, delta="+0.0~0.2",
                    recon_steps=1, run_minutes=4):
    return {
        "id": hid,
        "layer": "hypothesis",
        "domain": domain,
        "mechanism": f"mechanism of {hid}",
        "prior": prior,
        "expected": {"metric": "m.F1", "delta": delta, "confidence": "low"},
        "cost": {"recon_steps": recon_steps, "run_minutes": run_minutes},
    }


def make_graph(nodes, edges=None, decisions=None, acceptance=None):
    """构造可通过 M1 校验的最小图。"""
    return {
        "version": "1.0",
        "meta": {"task": "t", "created_by": "test", "source_cards": []},
        "goal": {
            "statement": "test goal",
            "acceptance": acceptance if acceptance is not None else ["m.F1 >= 0.9"],
        },
        "nodes": nodes,
        "edges": edges or [],
        "decisions": decisions or [],
        "writeback": {
            "ledger": "hypotheses.jsonl",
            "record_schema": "{node_id, status, evidence, metrics, posterior, notes, ts}",
            "after_domain_review": "x",
        },
    }


def domain_node(did="D1"):
    return {"id": did, "layer": "domain", "rationale": f"rationale {did}"}


def run_sim(graph, playbook, tmp_path, k=3, ledger=None):
    trace_path = tmp_path / "trace.jsonl"
    sim = Simulator(
        graph,
        MockExecutor(playbook),
        ledger if ledger is not None else InMemoryLedger(),
        SimConfig(k=k, trace_path=trace_path),
    )
    return sim.run(), trace_path


def events_of(result, event):
    return [e for e in result.events if e["event"] == event]


def golden_playbook():
    return load_playbook(PLAYBOOK)


def run_golden(tmp_path, playbook=None, ledger=None, k=3):
    graph = load_graph(GOLDEN)
    pb = playbook if playbook is not None else golden_playbook()
    return run_sim(graph, pb, tmp_path, k=k, ledger=ledger)


# --------------------------------------------------------------------------
# mock 执行器与基础设施
# --------------------------------------------------------------------------


class TestParseDeltaMid:
    """E2 缺省 priority 的 delta 中值解析（SPEC §4.2-E2 工程默认值）。"""

    def test_range_mid(self):
        assert parse_delta_mid("+0.05~0.20") == pytest.approx(0.125)

    def test_percent_magnitude(self):
        assert parse_delta_mid("-10%") == pytest.approx(0.10)

    def test_single_number(self):
        assert parse_delta_mid("+0.3") == pytest.approx(0.3)

    def test_non_numeric_returns_zero(self):
        assert parse_delta_mid("阈值方差↓") == 0.0


class TestTrace:
    """SPEC §4.3：JSONL 轨迹，每事件一行 {tick, event, node_id?, detail}。"""

    def test_jsonl_format_and_vocab(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with Trace(path) as trace:
            trace.emit(1, "unlock", "A1", {"requires": []})
            trace.emit(2, "run_end", None, {"reason": "x"})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first == {
            "tick": 1,
            "event": "unlock",
            "node_id": "A1",
            "detail": {"requires": []},
        }
        second = json.loads(lines[1])
        assert "node_id" not in second  # node_id 可选
        assert {json.loads(l)["event"] for l in lines} <= set(EVENT_VOCAB)

    def test_unknown_event_rejected(self, tmp_path):
        with pytest.raises(TraceEventError):
            Trace(tmp_path / "t.jsonl").emit(1, "not_an_event")


class TestMockExecutor:
    """SPEC §4.1：可配置成功/失败序列、候选产物序列、决策分支判定。"""

    def test_fail_sequence_consumed_per_attempt_then_last(self):
        from knowledge_pipeline import to_node

        action = to_node(make_action("A1", "H1", produces="rep"))
        ex = MockExecutor({"actions": {"A1": {"fail_sequence": [True, False, True]}}})
        r1 = ex.execute(action, {})
        r2 = ex.execute(action, {})
        r3 = ex.execute(action, {})
        r4 = ex.execute(action, {})  # 序列用尽 → 取最后一项（True）
        assert (r1.verify_passed, r2.verify_passed) == (False, True)
        assert (r3.verify_passed, r4.verify_passed) == (False, False)

    def test_artifacts_consumed_per_success_and_facts_update_merged(self):
        from knowledge_pipeline import to_node

        action = to_node(make_action("A1", "H1", produces="rep"))
        ex = MockExecutor({
            "facts": {"m": {"F1": 0.5}},
            "actions": {"A1": {"artifacts": [
                {"v": 1},
                {"v": 2, "facts_update": {"m": {"F1": 0.7}}},
            ]}},
        })
        r1 = ex.execute(action, {})
        r2 = ex.execute(action, {})
        r3 = ex.execute(action, {})  # 用尽 → 最后一项
        assert (r1.artifact["v"], r2.artifact["v"], r3.artifact["v"]) == (1, 2, 2)
        assert "facts_update" not in r2.artifact  # facts_update 不进产物本体
        assert ex.facts["m"]["F1"] == 0.7

    def test_resolve_branch_no_rule_no_default_returns_minus_one(self):
        from knowledge_pipeline import to_decision

        decision = to_decision({
            "id": "J1", "after": "A1", "read": "rep",
            "branches": [
                {"if": "条件甲", "then": "做 X"},
                {"if": "条件乙", "then": "做 Y"},
            ],
        })
        ex = MockExecutor({"decisions": {"J1": {"rules": [
            {"when": [{"path": "x", "op": "==", "value": 1}], "branch": 0, "reason": "r"},
        ]}}})
        idx, reason = ex.resolve_branch(decision, {"A1": {"x": 2}})
        assert idx == -1
        assert reason

    def test_missing_action_config_raises(self):
        from knowledge_pipeline import to_node

        ex = MockExecutor({"actions": {}})
        with pytest.raises(PlaybookError):
            ex.execute(to_node(make_action("ZZ", "H1")), {})


# --------------------------------------------------------------------------
# E1 解锁
# --------------------------------------------------------------------------


class TestE1Unlock:
    """E1（01 文档 §4）：全部 requires 前置 confirmed 且产物就位 → ready。"""

    def test_e1_requires_gates_unlock_until_source_confirmed(self, tmp_path):
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_action("A", "H1", produces="ra"),
            make_action("B", "H1", produces="rb"),
        ]
        edges = [{"from": "A", "to": "B", "type": "requires"}]
        graph = make_graph(nodes, edges)
        playbook = {"actions": {"A": {"artifacts": [{"ok": 1}]},
                                "B": {"artifacts": [{"ok": 1}]}}}
        result, _ = run_sim(graph, playbook, tmp_path)

        unlocks = events_of(result, "unlock")
        # B 在 A confirmed 之前不得解锁：A 的 unlock 在 tick 1，B 的更晚
        assert unlocks[0]["node_id"] == "A" and unlocks[0]["tick"] == 1
        b_unlock = next(e for e in unlocks if e["node_id"] == "B")
        a_exec = next(e for e in events_of(result, "execute") if e["node_id"] == "A")
        assert b_unlock["tick"] > a_exec["tick"]
        # 执行顺序 A 先于 B
        exec_order = [e["node_id"] for e in events_of(result, "execute")]
        assert exec_order.index("A") < exec_order.index("B")


# --------------------------------------------------------------------------
# E2 选序
# --------------------------------------------------------------------------


class TestE2Priority:
    """E2（SPEC §4.2）：ready 集合按 priority 降序；缺省 = 预期收益中值/验证成本。"""

    def test_e2_default_priority_from_expected_over_cost(self, tmp_path):
        # HX: 中值 0.2 / (1+3) = 0.05；HY: 中值 0.05 / (1+1) = 0.025 → X 先
        nodes = [
            domain_node(),
            make_hypothesis("HX", delta="+0.0~0.4", recon_steps=1, run_minutes=3),
            make_hypothesis("HY", delta="+0.0~0.1", recon_steps=1, run_minutes=1),
            make_action("X", "HX", produces="rx"),
            make_action("Y", "HY", produces="ry"),
        ]
        graph = make_graph(nodes)
        playbook = {"actions": {"X": {"artifacts": [{}]}, "Y": {"artifacts": [{}]}}}
        result, _ = run_sim(graph, playbook, tmp_path)
        selects = [e["node_id"] for e in events_of(result, "select")]
        assert selects[:2] == ["X", "Y"]

    def test_e2_explicit_priority_overrides_default(self, tmp_path):
        nodes = [
            domain_node(),
            make_hypothesis("HX", delta="+0.0~0.4", recon_steps=1, run_minutes=3),
            make_hypothesis("HY", delta="+0.0~0.1", recon_steps=1, run_minutes=1),
            make_action("X", "HX", produces="rx"),
            make_action("Y", "HY", produces="ry", priority=99.0),
        ]
        graph = make_graph(nodes)
        playbook = {"actions": {"X": {"artifacts": [{}]}, "Y": {"artifacts": [{}]}}}
        result, _ = run_sim(graph, playbook, tmp_path)
        selects = [e["node_id"] for e in events_of(result, "select")]
        assert selects[0] == "Y"  # 显式 priority 99 压过缺省公式


# --------------------------------------------------------------------------
# E3 决策点
# --------------------------------------------------------------------------


def _decision_graph_and_playbook():
    nodes = [
        domain_node(),
        make_hypothesis("H1"),
        make_hypothesis("H2"),
        make_action("S", "H1", produces="report", priority=50.0),
        make_action("T", "H1", atype="edit", produces="fix"),
        make_action("Y", "H2", produces="ry", priority=100.0),
    ]
    edges = [{"from": "S", "to": "T", "type": "requires"}]
    decisions = [{
        "id": "J",
        "after": "S",
        "read": "report",
        "branches": [
            {"if": "命中条件", "then": "优先执行 T"},
            {"else": "跳过 T"},
        ],
    }]
    graph = make_graph(nodes, edges, decisions)
    return graph


class TestE3Decision:
    """E3（01 文档 §2.4、§4）：读产物评估条件走分支，判定结果与理由落盘。"""

    def test_e3_branch_hit_boosts_then_action_and_logs_reason(self, tmp_path):
        graph = _decision_graph_and_playbook()
        playbook = {
            "actions": {"S": {"artifacts": [{"hit": 1}]},
                        "T": {"artifacts": [{}]},
                        "Y": {"artifacts": [{}]}},
            "decisions": {"J": {
                "rules": [{"when": [{"path": "hit", "op": ">=", "value": 1}],
                           "branch": 0, "reason": "report.hit={hit} ≥ 1 → 修复分支"}],
                "default": {"branch": 1, "reason": "未命中 → 跳过"},
                "effects": {"0": {"boost": ["T"]}, "1": {"skip": ["T"]}},
            }},
        }
        result, _ = run_sim(graph, playbook, tmp_path)
        dec = events_of(result, "decision")
        assert len(dec) == 1
        d = dec[0]
        assert d["node_id"] == "J"
        assert d["detail"]["branch_index"] == 0
        assert d["detail"]["boost"] == ["T"]
        assert "修复分支" in d["detail"]["reason"]  # 理由强制落盘
        # Y（priority 100）先跑；S 完成后 T 被 boost：T 是下一个执行的
        selects = [e["node_id"] for e in events_of(result, "select")]
        assert selects == ["Y", "S", "T"]
        t_select = events_of(result, "select")[2]
        assert t_select["detail"]["boosted"] is True

    def test_e3_no_match_no_else_records_decision_deadend(self, tmp_path):
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_action("S", "H1", produces="report"),
        ]
        decisions = [{
            "id": "J",
            "after": "S",
            "read": "report",
            "branches": [  # 两个 if、无 else（V-DECISION 合法形态）
                {"if": "条件甲", "then": "做 X"},
                {"if": "条件乙", "then": "做 Y"},
            ],
        }]
        graph = make_graph(nodes, decisions=decisions)
        playbook = {
            "actions": {"S": {"artifacts": [{"v": 0}]}},
            "decisions": {"J": {"rules": [
                {"when": [{"path": "v", "op": ">=", "value": 100}],
                 "branch": 0, "reason": "不会命中"},
            ]}},  # 无 default → resolve_branch 返回 -1
        }
        result, _ = run_sim(graph, playbook, tmp_path)
        deadends = events_of(result, "decision_deadend")
        assert len(deadends) == 1
        assert deadends[0]["node_id"] == "J"
        assert result.final_report["end_reason"] == "graph_exhausted"


# --------------------------------------------------------------------------
# E4 no_parallel
# --------------------------------------------------------------------------


class TestE4NoParallel:
    """E4（01 文档 §4）：互斥节点不可并行；v1 串行下对端在候选集则推迟并记录。"""

    def test_e4_no_parallel_hold_recorded_with_reason(self, tmp_path):
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_action("P", "H1", atype="run", produces="rp", priority=10.0),
            make_action("Q", "H1", atype="run", produces="rq", priority=5.0),
        ]
        edges = [{"from": "P", "to": "Q", "type": "no_parallel",
                  "reason": "全量训练资源互斥"}]
        graph = make_graph(nodes, edges)
        playbook = {"actions": {"P": {"artifacts": [{"measured_gain": 0.2}]},
                                "Q": {"artifacts": [{"measured_gain": 0.2}]}}}
        result, _ = run_sim(graph, playbook, tmp_path)
        holds = events_of(result, "no_parallel_hold")
        assert len(holds) == 1
        assert holds[0]["node_id"] == "Q"  # 低优先级对端被推迟
        assert holds[0]["detail"]["held_by"] == "P"
        assert "资源互斥" in holds[0]["detail"]["reason"]
        # 两者仍串行各跑一拍（v1 串行：hold 之后对端下一 tick 正常执行）
        selects = [e["node_id"] for e in events_of(result, "select")]
        assert selects == ["P", "Q"]


# --------------------------------------------------------------------------
# E5 每动作必验：失败 → 重试 → blocked → 升级
# --------------------------------------------------------------------------


class TestE5VerifyRetryBlocked:
    """E5（01 文档 §2.3、§4）：verify 失败重试，耗尽 → blocked + 升级，禁止静默跳过。"""

    def _fail_graph(self):
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_hypothesis("H2"),
            make_action("A", "H1", produces="ra", retry_budget=1),
            make_action("B", "H1", produces="rb"),
            make_action("C", "H1", produces="rc"),
            make_action("D", "H2", produces="rd"),
        ]
        edges = [
            {"from": "A", "to": "B", "type": "requires"},
            {"from": "B", "to": "C", "type": "requires"},
        ]
        return make_graph(nodes, edges)

    def test_e5_retry_exhausted_blocks_and_escalates_freezing_downstream(
        self, tmp_path
    ):
        ledger = InMemoryLedger()
        playbook = {"actions": {
            "A": {"fail_sequence": [True, True], "artifacts": [{}]},  # 1+1 次全败
            "B": {"artifacts": [{}]},
            "C": {"artifacts": [{}]},
            "D": {"artifacts": [{"measured_gain": 0.2}]},
        }}
        result, _ = run_sim(self._fail_graph(), playbook, tmp_path, ledger=ledger)

        # 失败 → 重试 → blocked 事件序列完整（A：1 次初始 + 1 次重试 = 2 次尝试）
        a_events = [e["event"] for e in result.events if e.get("node_id") == "A"]
        assert a_events == ["unlock", "select", "execute", "verify_fail",
                            "retry", "execute", "verify_fail", "blocked", "escalate"]
        # 下游 B、C 被冻结为 blocked_pending，且从未执行（无静默跳过：状态显式）
        assert result.final_report["blocked_pending"] == ["B", "C"]
        assert "B" not in [e["node_id"] for e in events_of(result, "execute")]
        assert "C" not in [e["node_id"] for e in events_of(result, "execute")]
        escalate = events_of(result, "escalate")[0]
        assert escalate["detail"]["frozen"] == ["B", "C"]
        # 终报单列「未决阻塞」
        assert result.final_report["unresolved_blockers"] == ["A"]
        # 其他 domain 继续推进：D 正常执行
        assert "D" in [e["node_id"] for e in events_of(result, "execute")]
        # 假说裁决：H1 blocked（账本记 status=blocked），H2 confirmed
        assert result.hypothesis_outcomes["H1"]["status"] == "blocked"
        assert result.hypothesis_outcomes["H2"]["status"] == "confirmed"
        assert ledger.records[0]["status"] == "blocked"

    def test_e5_retry_succeeds_within_budget(self, tmp_path):
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_action("A", "H1", produces="ra", retry_budget=2),
        ]
        playbook = {"actions": {"A": {"fail_sequence": [True, False],
                                      "artifacts": [{"ok": 1}]}}}
        result, _ = run_sim(make_graph(nodes), playbook, tmp_path)
        a_events = [e["event"] for e in result.events if e.get("node_id") == "A"]
        assert a_events == ["unlock", "select", "execute", "verify_fail",
                            "retry", "execute"]
        assert result.hypothesis_outcomes["H1"]["status"] == "confirmed"
        assert not events_of(result, "blocked")


# --------------------------------------------------------------------------
# E6 informs 参数回写
# --------------------------------------------------------------------------


class TestE6Informs:
    """E6（01 文档 §4）：informs 源完成后，目标参数先按产物更新再解锁/执行。"""

    def test_e6_source_artifact_injected_into_target_params(self, tmp_path):
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_action("S", "H1", produces="profile", priority=10.0),
            make_action("R", "H1", produces="rr", priority=5.0),
            make_action("T", "H1", atype="edit", produces="rt", priority=1.0),
        ]
        edges = [
            {"from": "R", "to": "T", "type": "requires"},
            {"from": "S", "to": "T", "type": "informs"},
        ]
        graph = make_graph(nodes, edges)
        profile_artifact = {"dominant_error": "FP", "val_samples": 320}
        playbook = {"actions": {
            "S": {"artifacts": [profile_artifact]},
            "R": {"artifacts": [{}]},
            "T": {"artifacts": [{}]},
        }}
        executor = MockExecutor(playbook)
        sim = Simulator(graph, executor, InMemoryLedger(),
                        SimConfig(trace_path=tmp_path / "t.jsonl"))
        result = sim.run()

        updates = events_of(result, "param_update")
        assert len(updates) == 1
        assert updates[0]["node_id"] == "T"
        assert updates[0]["detail"]["source"] == "S"
        assert updates[0]["detail"]["artifact"] == "profile"
        # 注入发生在 T 的 unlock/执行之前
        t_unlock_tick = next(e for e in events_of(result, "unlock")
                             if e["node_id"] == "T")["tick"]
        assert updates[0]["tick"] <= t_unlock_tick
        # 执行器实际收到的 params 含源产物
        received = executor.received_params["T"][0]
        assert received["profile"] == profile_artifact


# --------------------------------------------------------------------------
# E7 早停
# --------------------------------------------------------------------------


class TestE7EarlyStop:
    """E7（01 文档 §4）：连续 k 个假说验证无验收进展 → 停止并报告。"""

    def test_e7_early_stop_after_k_consecutive_no_progress(self, tmp_path):
        nodes = [domain_node()]
        playbook_actions = {}
        for i in range(1, 5):  # H1..H4 各一个 run 动作，均无验收进展
            hid, aid = f"H{i}", f"A{i}"
            nodes.append(make_hypothesis(hid))
            nodes.append(make_action(aid, hid, atype="run", produces=f"r{i}"))
            playbook_actions[aid] = {"artifacts": [{"measured_gain": 0.1}]}
        # 验收指标在 facts 中缺失 → 永远不通过 → 每次裁决都无"新通过"
        graph = make_graph(nodes, acceptance=["m.F1 >= 0.99"])
        playbook = {"actions": playbook_actions, "facts": {}}
        result, _ = run_sim(graph, playbook, tmp_path, k=3)

        stops = events_of(result, "early_stop")
        assert len(stops) == 1
        assert stops[0]["detail"]["k"] == 3
        assert stops[0]["detail"]["streak"] == 3
        assert result.final_report["end_reason"] == "early_stop"
        assert result.final_report["early_stop"]["at_hypothesis"] == "H3"
        # 第 4 个假说的动作不再执行（早停即停）
        executed = [e["node_id"] for e in events_of(result, "execute")]
        assert executed == ["A1", "A2", "A3"]
        assert "A4" not in executed

    def test_e7_progress_resets_streak(self, tmp_path):
        # H1 无进展（streak=1）→ H2 带来验收进展（重置）→ 不早停
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_hypothesis("H2"),
            make_action("A1", "H1", atype="run", produces="r1"),
            make_action("A2", "H2", atype="run", produces="r2"),
        ]
        graph = make_graph(nodes, acceptance=["m.F1 >= 0.9"])
        playbook = {
            "facts": {"m": {"F1": 0.5}},
            "actions": {
                "A1": {"artifacts": [{"measured_gain": 0.1}]},
                "A2": {"artifacts": [{"measured_gain": 0.1,
                                      "facts_update": {"m": {"F1": 0.95}}}]},
            },
        }
        result, _ = run_sim(graph, playbook, tmp_path, k=3)
        assert not events_of(result, "early_stop")
        checks = events_of(result, "acceptance_check")
        assert any(c["detail"]["newly_passed"] for c in checks)


# --------------------------------------------------------------------------
# E8 写回前置
# --------------------------------------------------------------------------


class TestE8Writeback:
    """E8（01 文档 §4）：写回成功才允许下游解锁；写回失败视为 blocked。"""

    def _chain_graph(self):
        nodes = [
            domain_node(),
            make_hypothesis("H1"),
            make_hypothesis("H2"),
            make_action("A1", "H1", atype="run", produces="r1"),
            make_action("B1", "H2", atype="run", produces="r2"),
        ]
        edges = [{"from": "A1", "to": "B1", "type": "requires"}]
        return make_graph(nodes, edges)

    def test_e8_writeback_precedes_downstream_unlock(self, tmp_path):
        ledger = InMemoryLedger()
        playbook = {"actions": {
            "A1": {"artifacts": [{"measured_gain": 0.2}]},
            "B1": {"artifacts": [{"measured_gain": 0.2}]},
        }}
        result, _ = run_sim(self._chain_graph(), playbook, tmp_path, ledger=ledger)
        seq = [(e["event"], e.get("node_id")) for e in result.events]
        wb_idx = seq.index(("writeback", "H1"))
        unlock_b1 = seq.index(("unlock", "B1"))
        assert wb_idx < unlock_b1  # 假说裁决写回先于下游解锁
        assert [r["node_id"] for r in ledger.records] == ["H1", "H2"]
        # 记录字段符合 SPEC §5 record_schema
        record = ledger.records[0]
        assert set(record) >= {"node_id", "status", "evidence", "metrics",
                               "posterior", "notes", "ts"}
        assert record["status"] == "confirmed"
        assert record["posterior"] == pytest.approx(0.5 + 0.5 * 0.9)

    def test_e8_writeback_failure_marks_blocked_and_freezes_downstream(
        self, tmp_path
    ):
        ledger = InMemoryLedger(fail=True)
        playbook = {"actions": {
            "A1": {"artifacts": [{"measured_gain": 0.2}]},
            "B1": {"artifacts": [{"measured_gain": 0.2}]},
        }}
        result, _ = run_sim(self._chain_graph(), playbook, tmp_path, ledger=ledger)
        # 写回失败 → 假说 blocked + escalate，下游 B1 冻结、永不解锁
        assert result.hypothesis_outcomes["H1"]["status"] == "blocked"
        escalates = events_of(result, "escalate")
        assert escalates and escalates[0]["detail"]["cause"] == "writeback_failed"
        assert "B1" not in [e["node_id"] for e in events_of(result, "unlock")]
        assert "B1" in result.final_report["blocked_pending"]
        assert not events_of(result, "writeback")


# --------------------------------------------------------------------------
# 黄金夹具 + tbox_playbook 全链演示（03 文档第二部分时间线）
# --------------------------------------------------------------------------


class TestGoldenTimeline:
    """黄金图 + TBox 事故剧本：复现 03 文档逐拍剧本。"""

    def test_first_batch_ready_unlock_order(self, tmp_path):
        result, _ = run_golden(tmp_path)
        tick1_unlocks = [e["node_id"] for e in result.events
                         if e["event"] == "unlock" and e["tick"] == 1]
        assert set(tick1_unlocks) == {"A1", "A3", "A4"}  # T+0：无前置 → ready
        # 按收益/成本选 A1（最便宜且杠杆最大）
        first_select = events_of(result, "select")[0]
        assert first_select["node_id"] == "A1"

    def test_j1_fix_branch_and_a2_retry_then_confirmed(self, tmp_path):
        ledger = InMemoryLedger()
        result, _ = run_golden(tmp_path, ledger=ledger)
        # T+2：J1 走"修复"分支，理由落盘
        j1 = next(e for e in events_of(result, "decision") if e["node_id"] == "J1")
        assert j1["detail"]["branch_index"] == 0
        assert j1["detail"]["boost"] == ["A2"]
        assert "未接线" in j1["detail"]["reason"]
        # T+3~4：A2 第一次冒烟失败 → retry → 通过
        a2 = [e["event"] for e in result.events if e.get("node_id") == "A2"]
        assert a2[:6] == ["unlock", "select", "execute", "verify_fail",
                          "retry", "execute"]
        # H1 confirmed，posterior 0.5→0.95，账本第 1 条
        h1 = ledger.records[0]
        assert h1["node_id"] == "H1" and h1["status"] == "confirmed"
        assert h1["posterior"] == pytest.approx(0.95)

    def test_a4_duplicate_columns_then_a3_error_profile(self, tmp_path):
        result, _ = run_golden(tmp_path)
        executed = [e["node_id"] for e in events_of(result, "execute")]
        assert "A4" in executed and "A3" in executed
        # A5/A7 在 A4 confirmed 后才解锁（requires A4）
        a4_exec_tick = next(e for e in events_of(result, "execute")
                            if e["node_id"] == "A4")["tick"]
        for aid in ("A5", "A7"):
            unlock_tick = next(e for e in events_of(result, "unlock")
                               if e["node_id"] == aid)["tick"]
            assert unlock_tick > a4_exec_tick

    def test_j2_threshold_with_kfold_branch_and_h2_gain(self, tmp_path):
        ledger = InMemoryLedger()
        result, _ = run_golden(tmp_path, ledger=ledger)
        # T+7：FP 主导 且 320<500 → "先 A5 但 A6 升级 K 折"分支（H6 激活）
        j2 = next(e for e in events_of(result, "decision") if e["node_id"] == "J2")
        assert j2["detail"]["branch_index"] == 1
        assert j2["detail"]["boost"] == ["A5", "A6"]
        assert "320" in j2["detail"]["reason"]
        # T+8：A6 实测 +0.12 → H2 confirmed，账本落实测收益
        h2 = next(r for r in ledger.records if r["node_id"] == "H2")
        assert h2["status"] == "confirmed"
        assert h2["metrics"]["gain"] == pytest.approx(0.12)
        assert result.hypothesis_outcomes["H6"]["status"] == "confirmed"

    def test_no_parallel_hold_appears_in_golden_run(self, tmp_path):
        result, _ = run_golden(tmp_path)
        holds = events_of(result, "no_parallel_hold")
        assert len(holds) == 1
        assert holds[0]["node_id"] == "A7"
        assert holds[0]["detail"]["held_by"] == "A5"
        assert "单原子" in holds[0]["detail"]["reason"]

    def test_h3_marginal_gain_rerun_then_uncertain(self, tmp_path):
        ledger = InMemoryLedger()
        result, _ = run_golden(tmp_path, ledger=ledger)
        # T+9~10：A8 实测 +0.03 → 边际提升复跑 → 复跑 +0.01
        a8_execs = [e for e in events_of(result, "execute")
                    if e["node_id"] == "A8"]
        assert len(a8_execs) == 2
        assert a8_execs[1]["detail"]["rerun"] is True
        # 判 uncertain，账本记"不计入战果"
        h3 = next(r for r in ledger.records if r["node_id"] == "H3")
        assert h3["status"] == "uncertain"
        assert "不计入战果" in h3["notes"]
        assert h3["metrics"]["gains"] == [pytest.approx(0.03), pytest.approx(0.01)]
        assert h3["posterior"] == pytest.approx(0.6)  # uncertain → 后验不变

    def test_acceptance_all_five_pass_and_unverified_hypotheses(self, tmp_path):
        result, _ = run_golden(tmp_path)
        report = result.final_report
        assert report["acceptance_passed"] == {"passed": 5, "total": 5}
        assert all(item["passed"] for item in report["acceptance"])
        assert report["early_stop"] is None
        assert report["end_reason"] == "graph_exhausted"
        assert report["unverified_hypotheses"] == ["H4", "H5"]  # 03：未验×2
        assert report["unresolved_blockers"] == []

    def test_ledger_record_order_and_statuses(self, tmp_path):
        ledger = InMemoryLedger()
        run_golden(tmp_path, ledger=ledger)
        assert [(r["node_id"], r["status"]) for r in ledger.records] == [
            ("H1", "confirmed"),
            ("H6", "confirmed"),
            ("H2", "confirmed"),
            ("H3", "uncertain"),
        ]
        for record in ledger.records:
            assert set(record) >= {"node_id", "status", "evidence", "metrics",
                                   "posterior", "notes", "ts"}

    def test_trace_jsonl_on_disk_valid(self, tmp_path):
        result, trace_path = run_golden(tmp_path)
        events = read_trace(trace_path)
        assert events == result.events
        assert all(e["event"] in EVENT_VOCAB for e in events)
        ticks = [e["tick"] for e in events]
        assert ticks == sorted(ticks)
        assert events[-1]["event"] == "run_end"


class TestGoldenBranchSwitching:
    """改 playbook 事实可切换决策分支（决策由事实驱动，非写死）。"""

    def test_j2_switches_branch_when_val_samples_600(self, tmp_path):
        pb = copy.deepcopy(golden_playbook())
        pb["actions"]["A3"]["artifacts"][0]["val_samples"]["游戏"] = 600
        result, _ = run_golden(tmp_path, playbook=pb)
        j2 = next(e for e in events_of(result, "decision") if e["node_id"] == "J2")
        assert j2["detail"]["branch_index"] == 0  # ≥500 → 直接"先 A5"
        assert j2["detail"]["boost"] == ["A5"]
        assert "600" in j2["detail"]["reason"]

    def test_j2_switches_to_weight_first_when_fn_dominant(self, tmp_path):
        pb = copy.deepcopy(golden_playbook())
        pb["actions"]["A3"]["artifacts"][0]["dominant_error"] = "FN"
        result, _ = run_golden(tmp_path, playbook=pb)
        j2 = next(e for e in events_of(result, "decision") if e["node_id"] == "J2")
        assert j2["detail"]["branch_index"] == 2
        assert j2["detail"]["boost"] == ["A7"]

    def test_j1_else_branch_skips_a2_when_nothing_unwired(self, tmp_path):
        pb = copy.deepcopy(golden_playbook())
        pb["actions"]["A1"]["artifacts"][0]["unwired_count"] = 0
        result, _ = run_golden(tmp_path, playbook=pb)
        j1 = next(e for e in events_of(result, "decision") if e["node_id"] == "J1")
        assert j1["detail"]["branch_index"] == 1  # else 兜底
        assert j1["detail"]["skip"] == ["A2"]
        # A2 被显式跳过（落盘），从未执行；H1 仍 confirmed
        assert "A2" in result.final_report["skipped"]
        assert "A2" not in [e["node_id"] for e in events_of(result, "execute")]
        assert result.hypothesis_outcomes["H1"]["status"] == "confirmed"


class TestGoldenFailureScenario:
    """03 文档失败路径：A2 重试耗尽 → blocked → 升级（不静默跳过）。"""

    def test_a2_permanent_failure_escalates_and_other_domains_continue(
        self, tmp_path
    ):
        ledger = InMemoryLedger()
        pb = copy.deepcopy(golden_playbook())
        pb["actions"]["A2"]["fail_sequence"] = [True, True, True]  # 预算 2 → 3 次全败
        result, _ = run_golden(tmp_path, playbook=pb, ledger=ledger)

        a2 = [e["event"] for e in result.events if e.get("node_id") == "A2"]
        assert a2 == ["unlock", "select", "execute", "verify_fail",
                      "retry", "execute", "verify_fail",
                      "retry", "execute", "verify_fail", "blocked", "escalate"]
        # 终报单列「未决阻塞」；H1 判 blocked 并写回账本
        assert result.final_report["unresolved_blockers"] == ["A2"]
        assert result.hypothesis_outcomes["H1"]["status"] == "blocked"
        h1 = next(r for r in ledger.records if r["node_id"] == "H1")
        assert h1["status"] == "blocked"
        # 其他 domain 继续：A3/A4/A5/A6 正常执行，J2 照常判定
        executed = {e["node_id"] for e in events_of(result, "execute")}
        assert {"A3", "A4", "A5", "A6"} <= executed
        assert events_of(result, "decision")


# --------------------------------------------------------------------------
# 校验门
# --------------------------------------------------------------------------


def test_simulator_rejects_invalid_graph(tmp_path):
    """模拟器构造时过 M1 校验门：非法图拒绝执行（01 文档 §3 校验门延续）。"""
    bad = load_graph(BAD_CYCLE)
    with pytest.raises(ValueError, match="校验"):
        Simulator(bad, MockExecutor({"actions": {}}), InMemoryLedger(),
                  SimConfig(trace_path=tmp_path / "t.jsonl"))
