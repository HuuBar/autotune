"""M6 量化评测台测试。

覆盖（语义来源：SPEC §8 / 03 文档第四部分）：

a) 剧本库完整性：6 个剧本、SPEC §8 字段齐全、图过 M1 校验、playbook 覆盖、
   双方事实一致；
b) 每个剧本双方各跑两遍，轨迹确定性可重复（canonical 后逐事件一致）；
c) 指标计算正确：构造已知轨迹断言七项指标数值（双侧）；
d) 聚合报告结构正确（bench_report.json）；
e) S4/S5 对照：有编排侧陷阱被识别 / 噪声不当战果，优于无编排侧。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_pipeline.bench import (
    METRIC_NAMES,
    SCENARIO_IDS,
    BaselineAgent,
    canonical_events,
    compute_metrics,
    load_scripts,
    run_bench,
)
from knowledge_pipeline.bench.runner import run_orchestrated
from knowledge_pipeline.validator import validate_graph

SCENARIOS = load_scripts()

# --------------------------------------------------------------------------
# a) 剧本库完整性
# --------------------------------------------------------------------------


def test_library_has_exactly_s1_to_s6():
    assert tuple(sorted(SCENARIOS)) == SCENARIO_IDS


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_scenario_required_fields(sid):
    sc = SCENARIOS[sid]
    # SPEC §8：每剧本 {id, title, trigger, env_response, expected_response}
    for field_name in ("id", "title", "trigger", "env_response", "expected_response"):
        assert getattr(sc, field_name), f"{sid} 缺字段 {field_name}"
    assert sc.id == sid
    # 有编排图过 M1 校验；playbook 覆盖全部 action/decision
    assert validate_graph(sc.graph) == []
    action_ids = {n["id"] for n in sc.graph["nodes"] if n["layer"] == "action"}
    assert action_ids <= set(sc.playbook["actions"])
    decision_ids = {d["id"] for d in sc.graph["decisions"]}
    assert decision_ids <= set(sc.playbook["decisions"])
    # 双方完全一致的事实（03 文档：剧本对双方完全一致）
    assert sc.playbook["facts"] == sc.facts
    assert [str(s) for s in sc.brief["acceptance"]] == [
        str(s) for s in sc.graph["goal"]["acceptance"]
    ]
    # 指标判据覆盖全部决策点
    assert set(sc.expected_decisions) == decision_ids


# --------------------------------------------------------------------------
# b) 确定性可重复：同输入两跑结果一致
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_orchestrated_trace_deterministic(sid, tmp_path):
    sc = SCENARIOS[sid]
    events_1, _ = run_orchestrated(sc, tmp_path / "run1")
    events_2, _ = run_orchestrated(sc, tmp_path / "run2")
    assert canonical_events(events_1) == canonical_events(events_2)


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_baseline_trace_deterministic(sid):
    sc = SCENARIOS[sid]
    result_1 = BaselineAgent(sc).run()
    result_2 = BaselineAgent(sc).run()
    assert result_1.events == result_2.events
    assert result_1.facts == result_2.facts
    assert result_1.stop_reason == result_2.stop_reason


# --------------------------------------------------------------------------
# c) 指标计算正确（构造已知轨迹断言数值）
# --------------------------------------------------------------------------


def _mini_scenario(**overrides):
    """最小 duck-type 剧本（仅含指标计算所需属性）。"""
    graph = {
        "meta": {"source_cards": ["C9", "C2"]},
        "goal": {"statement": "t", "acceptance": ["overall.F1 >= 0.9"]},
        "nodes": [
            {"id": "D1", "layer": "domain", "rationale": "r"},
            {
                "id": "H1",
                "layer": "hypothesis",
                "domain": "D1",
                "mechanism": "m",
                "prior": 0.5,
                "expected": {"metric": "overall.F1", "delta": "+0.1", "confidence": "low"},
                "cost": {"recon_steps": 1, "run_minutes": 1},
            },
            {
                "id": "A1",
                "layer": "action",
                "belongs_to": "H1",
                "type": "recon",
                "do": "d",
                "tool_hint": "t",
                "produces": "p1",
                "verify": "v",
                "retry_budget": 1,
            },
            {
                "id": "A2",
                "layer": "action",
                "belongs_to": "H1",
                "type": "run",
                "do": "d",
                "tool_hint": "t",
                "produces": "p2",
                "verify": "v",
                "retry_budget": 1,
            },
        ],
        "edges": [{"from": "A1", "to": "A2", "type": "requires"}],
        "decisions": [
            {"id": "J1", "after": "A1", "read": "p1", "branches": [{"if": "x", "then": "y"}]}
        ],
        "writeback": {"ledger": "l", "record_schema": "r", "after_domain_review": "a"},
        "version": "1.0",
    }
    baseline_env = {
        "probes": {"p1": {"direction": "方向乙", "reveals": "r"}},
        "interventions": {
            "c1": {
                "direction": "方向甲",
                "needs_probes": ["p1"],
                "gains": [0.03],
                "fits_scenario": False,
                "fit_reason": "不适用",
            }
        },
    }
    sc = SimpleNamespace(
        graph=graph,
        baseline_env=baseline_env,
        expected_decisions={"J1": 0},
        expected_first_choice={"choice": "c1", "protocol_upgrade": True},
        marginal_threshold=0.05,
        trap={
            "orchestrated_signals": [
                {"event": "decision", "node_id": "J1", "detail": {"branch_index": 0}}
            ],
            "baseline_signals": [{"event": "challenge"}],
        },
        inapplicable_cards=("C9",),
    )
    for key, value in overrides.items():
        setattr(sc, key, value)
    return sc


def test_metrics_orchestrated_known_trace():
    sc = _mini_scenario()
    events = [
        {"tick": 1, "event": "execute", "node_id": "A2", "detail": {"verify_passed": True}},
        # A2 首做时 A1 未 confirmed → 信息缺失事故 1
        {"tick": 2, "event": "execute", "node_id": "A1", "detail": {"verify_passed": True}},
        {"tick": 2, "event": "decision", "node_id": "J1",
         "detail": {"branch_index": 1}},  # 期望 0 → 分支错误
        {"tick": 3, "event": "writeback", "node_id": "H1",
         "detail": {"record": {"status": "confirmed",
                               "metrics": {"gain": 0.03}}}},  # 噪声当战果
        {"tick": 3, "event": "acceptance_check",
         "detail": {"passed_count": 1, "total": 1}},
    ]
    m = compute_metrics(events, "orchestrated", sc)
    assert m == {
        "steps_to_acceptance": 2,       # tick<=3 的 execute 数
        "info_gap_incidents": 1,
        "direction_coverage": 1,        # A1/A2 → H1 → D1
        "decision_branch_accuracy": 0.0,
        "s4_trap_identified": 0.0,      # J1 走了 branch 1，信号（branch 0）未命中
        "s5_noise_as_win": 1,           # 0 < 0.03 < 0.05 的 confirmed
        "blind_card_applications": 1,   # source_cards ∩ inapplicable = {C9}
    }


def test_metrics_orchestrated_clean_trace():
    sc = _mini_scenario()
    events = [
        {"tick": 1, "event": "execute", "node_id": "A1", "detail": {"verify_passed": True}},
        {"tick": 1, "event": "decision", "node_id": "J1", "detail": {"branch_index": 0}},
        {"tick": 2, "event": "execute", "node_id": "A2", "detail": {"verify_passed": True}},
        {"tick": 2, "event": "writeback", "node_id": "H1",
         "detail": {"record": {"status": "uncertain",
                               "metrics": {"gain": 0.03}}}},
        {"tick": 2, "event": "acceptance_check",
         "detail": {"passed_count": 0, "total": 1}},
    ]
    m = compute_metrics(events, "orchestrated", sc)
    assert m["steps_to_acceptance"] is None  # 未到达验收
    assert m["info_gap_incidents"] == 0
    assert m["decision_branch_accuracy"] == 1.0
    assert m["s4_trap_identified"] == 1.0
    assert m["s5_noise_as_win"] == 0  # uncertain 不计入战果


def test_metrics_baseline_known_trace():
    sc = _mini_scenario()
    events = [
        {"step": 0, "event": "decide",
         "detail": {"choice": "c1", "protocol_upgrade": False}},  # 期望 upgrade=True → 错误
        {"step": 1, "event": "edit", "detail": {"card": "c1", "direction": "方向甲"}},
        {"step": 2, "event": "run_fail", "detail": {"card": "c1", "failure": "f"}},
        {"step": 3, "event": "probe",
         "detail": {"probe": "p1", "direction": "方向乙"}},  # 先决策后补查
        {"step": 4, "event": "run",
         "detail": {"card": "c1", "gain": 0.03, "direction": "方向甲"}},
        {"step": 4, "event": "claim_win", "detail": {"card": "c1", "gain": 0.03}},
        {"step": 4, "event": "acceptance_check",
         "detail": {"passed_count": 1, "total": 1}},
    ]
    m = compute_metrics(events, "baseline", sc)
    assert m == {
        "steps_to_acceptance": 4,       # step<=4 的动作事件数
        "info_gap_incidents": 1,        # p1 在首次编辑之后才补查
        "direction_coverage": 2,        # 方向甲 + 方向乙
        "decision_branch_accuracy": 0.0,
        "s4_trap_identified": 0.0,      # 无 challenge 事件
        "s5_noise_as_win": 1,           # 0 < 0.03 < 0.05 的 claim_win
        "blind_card_applications": 1,   # c1 fits_scenario=false
    }


def test_metrics_no_trap_scenario_is_none():
    sc = _mini_scenario(trap=None)
    m = compute_metrics([], "orchestrated", sc)
    assert m["s4_trap_identified"] is None
    assert m["decision_branch_accuracy"] is None  # 无 decision 事件


def test_canonical_events_strips_timestamps():
    events = [
        {"tick": 1, "event": "writeback",
         "detail": {"record": {"node_id": "H1", "ts": "2025-01-01T00:00:00"}}}
    ]
    canon = canonical_events(events)
    assert "ts" not in canon[0]["detail"]["record"]
    assert events[0]["detail"]["record"]["ts"]  # 原轨迹不被修改


# --------------------------------------------------------------------------
# d) + e) 聚合报告 与 S4/S5 对照（共享一次全量运行）
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    return run_bench(out)


def test_report_structure(report):
    assert report["meta"]["scenarios"] == list(SCENARIO_IDS)
    assert report["metrics"] == list(METRIC_NAMES)
    for sid in SCENARIO_IDS:
        entry = report["scenarios"][sid]
        for side in ("orchestrated", "baseline"):
            assert set(METRIC_NAMES) <= set(entry[side]), f"{sid}/{side} 指标不全"
            assert "acceptance_passed" in entry[side]
            assert "trace_path" in entry[side]
    # 聚合：七项 × 双侧，含 value/n/values
    for metric in METRIC_NAMES:
        for side in ("orchestrated", "baseline"):
            agg = report["aggregate"][metric][side]
            assert set(agg) == {"value", "n", "values"}
            assert agg["n"] <= len(SCENARIO_IDS)
    assert set(report["preregistered_checks"]) == {
        "s4_trap_identified",
        "s5_noise_as_win",
    }


def test_report_json_written(tmp_path):
    out = tmp_path / "bench"
    run_bench(out)
    data = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    assert data["meta"]["scenarios"] == list(SCENARIO_IDS)
    assert (out / "S1" / "trace.jsonl").is_file()
    assert (out / "S1" / "baseline_trace.jsonl").is_file()
    assert (out / "S1" / "hypotheses.jsonl").is_file()


# 逐剧本 × 双侧 × 七项指标的冻结值（回归基线；改动剧本数据需同步更新）
EXPECTED_METRICS = {
    "S1": {
        "orchestrated": {"steps_to_acceptance": 7, "info_gap_incidents": 0,
                         "direction_coverage": 3, "decision_branch_accuracy": 1.0,
                         "s4_trap_identified": None, "s5_noise_as_win": 0,
                         "blind_card_applications": 0},
        "baseline": {"steps_to_acceptance": 12, "info_gap_incidents": 2,
                     "direction_coverage": 3, "decision_branch_accuracy": 1.0,
                     "s4_trap_identified": None, "s5_noise_as_win": 1,
                     "blind_card_applications": 1},
    },
    "S2": {
        "orchestrated": {"steps_to_acceptance": 5, "info_gap_incidents": 0,
                         "direction_coverage": 3, "decision_branch_accuracy": 1.0,
                         "s4_trap_identified": None, "s5_noise_as_win": 0,
                         "blind_card_applications": 0},
        "baseline": {"steps_to_acceptance": 5, "info_gap_incidents": 1,
                     "direction_coverage": 2, "decision_branch_accuracy": 1.0,
                     "s4_trap_identified": None, "s5_noise_as_win": 0,
                     "blind_card_applications": 0},
    },
    "S3": {
        "orchestrated": {"steps_to_acceptance": 6, "info_gap_incidents": 0,
                         "direction_coverage": 3, "decision_branch_accuracy": 1.0,
                         "s4_trap_identified": None, "s5_noise_as_win": 0,
                         "blind_card_applications": 0},
        "baseline": {"steps_to_acceptance": 6, "info_gap_incidents": 1,
                     "direction_coverage": 2, "decision_branch_accuracy": 1.0,
                     "s4_trap_identified": None, "s5_noise_as_win": 0,
                     "blind_card_applications": 0},
    },
    "S4": {
        "orchestrated": {"steps_to_acceptance": 7, "info_gap_incidents": 0,
                         "direction_coverage": 3, "decision_branch_accuracy": 1.0,
                         "s4_trap_identified": 1.0, "s5_noise_as_win": 0,
                         "blind_card_applications": 0},
        "baseline": {"steps_to_acceptance": 3, "info_gap_incidents": 0,
                     "direction_coverage": 1, "decision_branch_accuracy": 0.0,
                     "s4_trap_identified": 0.0, "s5_noise_as_win": 0,
                     "blind_card_applications": 0},
    },
    "S5": {
        "orchestrated": {"steps_to_acceptance": None, "info_gap_incidents": 0,
                         "direction_coverage": 3, "decision_branch_accuracy": 1.0,
                         "s4_trap_identified": None, "s5_noise_as_win": 0,
                         "blind_card_applications": 0},
        "baseline": {"steps_to_acceptance": None, "info_gap_incidents": 2,
                     "direction_coverage": 3, "decision_branch_accuracy": 1.0,
                     "s4_trap_identified": None, "s5_noise_as_win": 2,
                     "blind_card_applications": 2},
    },
    "S6": {
        "orchestrated": {"steps_to_acceptance": 5, "info_gap_incidents": 0,
                         "direction_coverage": 2, "decision_branch_accuracy": 1.0,
                         "s4_trap_identified": None, "s5_noise_as_win": 0,
                         "blind_card_applications": 0},
        "baseline": {"steps_to_acceptance": 5, "info_gap_incidents": 1,
                     "direction_coverage": 2, "decision_branch_accuracy": 1.0,
                     "s4_trap_identified": None, "s5_noise_as_win": 0,
                     "blind_card_applications": 0},
    },
}


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_frozen_metrics(report, sid):
    for side in ("orchestrated", "baseline"):
        actual = {m: report["scenarios"][sid][side][m] for m in METRIC_NAMES}
        assert actual == EXPECTED_METRICS[sid][side], f"{sid}/{side}"


def test_orchestrated_no_process_incidents_anywhere(report):
    """有编排侧过程纪律：全部剧本信息缺失事故/噪声战果/盲抄均为 0。"""
    for sid in SCENARIO_IDS:
        orch = report["scenarios"][sid]["orchestrated"]
        assert orch["info_gap_incidents"] == 0, sid
        assert orch["s5_noise_as_win"] == 0, sid
        assert orch["blind_card_applications"] == 0, sid
        assert orch["decision_branch_accuracy"] == 1.0, sid


# --------------------------------------------------------------------------
# e) S4/S5 对照：有编排侧优于无编排侧
# --------------------------------------------------------------------------


def test_s4_trap_identified_contrast(report):
    s4 = report["scenarios"]["S4"]
    assert s4["orchestrated"]["s4_trap_identified"] == 1.0
    assert s4["baseline"]["s4_trap_identified"] == 0.0
    # 附带：baseline 在小样本剧本上决策路线错误（未升级验收协议）且过早收工
    assert s4["baseline"]["decision_branch_accuracy"] == 0.0
    assert s4["baseline"]["direction_coverage"] < s4["orchestrated"]["direction_coverage"]


def test_s5_noise_as_win_contrast(report):
    s5 = report["scenarios"]["S5"]
    assert s5["orchestrated"]["s5_noise_as_win"] == 0
    assert s5["baseline"]["s5_noise_as_win"] >= 1
    # 有编排侧验收未全过（拒绝用噪声凑验收）
    assert s5["orchestrated"]["acceptance_passed"]["passed"] < (
        s5["orchestrated"]["acceptance_passed"]["total"]
    )


def test_preregistered_checks_hold(report):
    checks = report["preregistered_checks"]
    assert checks["s4_trap_identified"]["holds"] is True
    assert checks["s5_noise_as_win"]["holds"] is True
