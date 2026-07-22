"""M1 校验器测试：黄金夹具通过、坏图逐条拦截、错误信息可读。

语义来源：SPEC §3（V-* 检查清单）。坏图夹具每个触发不同检查，
测试断言错误码命中且 message 含相关节点/边 id（人类可读）。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from knowledge_pipeline import (
    ValidationError,
    load_graph,
    validate_file,
    validate_graph,
)

FIXTURES = Path(__file__).resolve().parents[2] / "knowledge_pipeline" / "fixtures"
GOLDEN = FIXTURES / "tbox_golden.yaml"
BAD_DIR = FIXTURES / "bad_graphs"


# --------------------------------------------------------------------------
# 黄金夹具
# --------------------------------------------------------------------------


def test_golden_fixture_passes_validate_file() -> None:
    assert validate_file(GOLDEN) == []


def test_golden_fixture_passes_validate_graph() -> None:
    doc = load_graph(GOLDEN)
    assert validate_graph(doc) == []


# --------------------------------------------------------------------------
# 坏图：每个触发不同检查，message 非空且含相关 id
# --------------------------------------------------------------------------

# (文件名, 期望错误码, message 中必须出现的子串)
BAD_CASES = [
    ("bad_01_cycle.yaml", "V-DAG", ["A1", "A2"]),
    ("bad_02_requires_missing.yaml", "V-REQUIRES-EXISTS", ["A_GHOST"]),
    ("bad_03_action_no_verify.yaml", "V-ACTION-VERIFY", ["A2", "verify"]),
    ("bad_04_alt_internal_requires.yaml", "V-ALT-GROUP", ["A2", "A3"]),
    ("bad_05_acceptance.yaml", "V-ACCEPTANCE", ["王者荣耀"]),
    ("bad_06_decision_incomplete.yaml", "V-DECISION", ["J1", "else"]),
    ("bad_07_alt_oneside.yaml", "V-ALT-GROUP", ["A2"]),
    ("bad_08_decision_read_mismatch.yaml", "V-DECISION", ["J1", "error_profile", "wiring_report"]),
]


@pytest.mark.parametrize("filename, code, needles", BAD_CASES)
def test_bad_graph_rejected_with_readable_message(
    filename: str, code: str, needles: list[str]
) -> None:
    errors = validate_file(BAD_DIR / filename)
    assert errors, f"{filename} 应被拦截"
    hits = [e for e in errors if e.code == code]
    assert hits, f"{filename} 应触发 {code}，实际: {[str(e) for e in errors]}"
    for e in hits:
        assert e.message.strip(), "错误信息必须非空（人类可读）"
        assert e.path.strip(), "错误路径必须非空"
    joined = "\n".join(e.message for e in hits)
    for needle in needles:
        assert needle in joined, f"{code} 错误信息应包含 {needle!r}: {joined}"


def test_bad_fixtures_cover_at_least_six_distinct_checks() -> None:
    files = sorted(BAD_DIR.glob("bad_*.yaml"))
    assert len(files) >= 6, "坏图夹具数量需 >= 6"
    codes = {e.code for f in files for e in validate_file(f)}
    assert len(codes) >= 6, f"坏图应覆盖 >=6 种检查，实际: {codes}"


# --------------------------------------------------------------------------
# 各检查项的定向单测（在最小合法图上注入单点缺陷）
# --------------------------------------------------------------------------


def _minimal_graph() -> dict:
    return {
        "version": "1.0",
        "meta": {"task": "t", "created_by": "test", "source_cards": []},
        "goal": {"statement": "s", "acceptance": ["overall.F1 >= 0.9"]},
        "nodes": [
            {"id": "D1", "layer": "domain", "title": "领域", "rationale": "r"},
            {
                "id": "H1",
                "layer": "hypothesis",
                "domain": "D1",
                "title": "假说",
                "mechanism": "m",
                "prior": 0.5,
                "expected": {"metric": "overall.F1", "delta": "+0.05", "confidence": "low"},
                "cost": {"recon_steps": 1, "run_minutes": 5},
            },
            {
                "id": "A1",
                "layer": "action",
                "belongs_to": "H1",
                "type": "recon",
                "do": "d",
                "tool_hint": "t",
                "produces": "rep",
                "verify": "v",
                "retry_budget": 1,
            },
        ],
        "edges": [],
        "decisions": [],
        "writeback": {
            "ledger": "hypotheses.jsonl",
            "record_schema": "{node_id, status}",
            "after_domain_review": "r",
        },
    }


def test_minimal_graph_is_valid() -> None:
    assert validate_graph(_minimal_graph()) == []


def _codes(errors: list[ValidationError]) -> set[str]:
    return {e.code for e in errors}


def test_v_schema_missing_top_level_key() -> None:
    doc = _minimal_graph()
    del doc["writeback"]
    errors = validate_graph(doc)
    assert "V-SCHEMA" in _codes(errors)
    assert any("writeback" in e.message for e in errors)


def test_v_uniq_duplicate_node_id() -> None:
    doc = _minimal_graph()
    doc["nodes"].append(copy.deepcopy(doc["nodes"][0]))
    errors = validate_graph(doc)
    assert "V-UNIQ" in _codes(errors)
    assert any("D1" in e.message for e in errors if e.code == "V-UNIQ")


def test_v_uniq_decision_id_collides_with_node() -> None:
    doc = _minimal_graph()
    doc["decisions"].append(
        {
            "id": "A1",
            "after": "A1",
            "read": "rep",
            "branches": [{"if": "x", "then": "y"}, {"else": "z"}],
        }
    )
    errors = validate_graph(doc)
    assert "V-UNIQ" in _codes(errors)


def test_v_dag_cycle_via_informs() -> None:
    doc = _minimal_graph()
    doc["nodes"].append(
        {
            "id": "A2",
            "layer": "action",
            "belongs_to": "H1",
            "type": "run",
            "do": "d",
            "tool_hint": "t",
            "produces": "out",
            "verify": "v",
            "retry_budget": 0,
        }
    )
    doc["edges"] = [
        {"from": "A1", "to": "A2", "type": "informs"},
        {"from": "A2", "to": "A1", "type": "requires"},
    ]
    errors = validate_graph(doc)
    assert "V-DAG" in _codes(errors)


def test_v_dag_ignores_no_parallel() -> None:
    """no_parallel 是无向约束，互相声明不构成环（SPEC §3 V-DAG）。"""
    doc = _minimal_graph()
    doc["nodes"].append(
        {
            "id": "A2",
            "layer": "action",
            "belongs_to": "H1",
            "type": "run",
            "do": "d",
            "tool_hint": "t",
            "produces": "out",
            "verify": "v",
            "retry_budget": 0,
        }
    )
    doc["edges"] = [
        {"from": "A1", "to": "A2", "type": "no_parallel"},
        {"from": "A2", "to": "A1", "type": "no_parallel"},
    ]
    assert "V-DAG" not in _codes(validate_graph(doc))


def test_v_requires_from_decision_forbidden() -> None:
    """requires 来自 decision 被禁止；指向 decision 合法（黄金夹具 A3->J2 形态）。"""
    doc = _minimal_graph()
    doc["nodes"].append(
        {
            "id": "A2",
            "layer": "action",
            "belongs_to": "H1",
            "type": "run",
            "do": "d",
            "tool_hint": "t",
            "produces": "out",
            "verify": "v",
            "retry_budget": 0,
        }
    )
    doc["decisions"].append(
        {
            "id": "J1",
            "after": "A1",
            "read": "rep",
            "branches": [{"if": "x", "then": "y"}, {"else": "z"}],
        }
    )
    doc["edges"] = [{"from": "J1", "to": "A2", "type": "requires"}]
    errors = validate_graph(doc)
    assert "V-REQUIRES-EXISTS" in _codes(errors)
    assert any("J1" in e.message for e in errors if e.code == "V-REQUIRES-EXISTS")

    doc["edges"] = [{"from": "A1", "to": "J1", "type": "requires"}]  # 指向 decision：合法
    assert "V-REQUIRES-EXISTS" not in _codes(validate_graph(doc))


def test_v_action_verify_empty_string() -> None:
    doc = _minimal_graph()
    doc["nodes"][2]["verify"] = "   "
    errors = validate_graph(doc)
    assert "V-ACTION-VERIFY" in _codes(errors)


def test_v_action_missing_retry_budget() -> None:
    doc = _minimal_graph()
    del doc["nodes"][2]["retry_budget"]
    errors = validate_graph(doc)
    assert "V-ACTION-VERIFY" in _codes(errors)


def test_v_alt_group_cross_domain() -> None:
    """alternatives 组成员解析到不同 domain → 组残缺。"""
    doc = _minimal_graph()
    doc["nodes"] += [
        {"id": "D2", "layer": "domain", "title": "另一领域", "rationale": "r"},
        {
            "id": "H2",
            "layer": "hypothesis",
            "domain": "D2",
            "title": "假说2",
            "mechanism": "m",
            "prior": 0.5,
            "expected": {"metric": "m", "delta": "+1", "confidence": "low"},
            "cost": {"recon_steps": 1, "run_minutes": 1},
        },
        {
            "id": "A2",
            "layer": "action",
            "belongs_to": "H2",
            "type": "edit",
            "do": "d",
            "tool_hint": "t",
            "produces": "x",
            "verify": "v",
            "retry_budget": 0,
        },
    ]
    doc["edges"] = [{"from": "A1", "to": "A2", "type": "alternatives"}]
    errors = validate_graph(doc)
    assert "V-ALT-GROUP" in _codes(errors)


def test_v_alt_group_same_hypothesis_ok() -> None:
    """同一 hypothesis 下游分支的 alternatives 组合法。"""
    doc = _minimal_graph()
    doc["nodes"].append(
        {
            "id": "A2",
            "layer": "action",
            "belongs_to": "H1",
            "type": "edit",
            "do": "d",
            "tool_hint": "t",
            "produces": "x",
            "verify": "v",
            "retry_budget": 0,
        }
    )
    doc["edges"] = [{"from": "A1", "to": "A2", "type": "alternatives"}]
    assert "V-ALT-GROUP" not in _codes(validate_graph(doc))


def test_v_decision_after_must_be_action() -> None:
    doc = _minimal_graph()
    doc["decisions"].append(
        {
            "id": "J1",
            "after": "H1",  # 不是 action
            "read": "rep",
            "branches": [{"if": "x", "then": "y"}, {"else": "z"}],
        }
    )
    errors = validate_graph(doc)
    assert "V-DECISION" in _codes(errors)
    assert any("H1" in e.message for e in errors if e.code == "V-DECISION")


def test_v_layer_ref_wrong_layer() -> None:
    doc = _minimal_graph()
    doc["nodes"][1]["domain"] = "A1"  # hypothesis.domain 指向 action
    errors = validate_graph(doc)
    assert "V-LAYER-REF" in _codes(errors)


def test_v_layer_ref_belongs_to_missing() -> None:
    doc = _minimal_graph()
    doc["nodes"][2]["belongs_to"] = "H_GHOST"
    errors = validate_graph(doc)
    assert "V-LAYER-REF" in _codes(errors)
    assert any("H_GHOST" in e.message for e in errors if e.code == "V-LAYER-REF")


@pytest.mark.parametrize("prior", [-0.1, 1.5, "high"])
def test_v_prior_out_of_range(prior: object) -> None:
    doc = _minimal_graph()
    doc["nodes"][1]["prior"] = prior
    errors = validate_graph(doc)
    assert "V-PRIOR" in _codes(errors)


def test_validate_file_missing_path_returns_error() -> None:
    errors = validate_file(BAD_DIR / "no_such_file.yaml")
    assert len(errors) == 1
    assert errors[0].code == "V-SCHEMA"
    assert errors[0].message.strip()
