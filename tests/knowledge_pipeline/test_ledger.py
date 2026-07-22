"""M3 假说账本测试：append-only 账本、后验更新、领域聚合报告。

语义来源：SPEC §5；01 文档 §5（指标是数据不是文本；append-only）；
03 文档第三部分（H1 confirmed posterior 0.5→0.95、H2 confirmed 实测
+0.12、H3 边际判 uncertain"不计入战果"、H4/H5 未验；领域聚合：D1 上调、
D3 未探索保留）。

裁决备注（列入回传报告「新发现」）：03 文档称账本"追加 5 条（…未验×2）"，
但 SPEC §5 状态词表不含"未验"——本测试按"未验假说不落账本"实现：
场景 e) 覆盖 5 个假说（H1~H5），账本记录 3 条（H1/H2/H3），H4/H5 未验
即无记录，domain_report 经图映射识别 D3 为未探索（hold）。
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from knowledge_pipeline import load_graph
from knowledge_pipeline.ledger import (
    STATUSES,
    Ledger,
    domain_report,
    update_posterior,
)

FIXTURES = Path(__file__).resolve().parents[2] / "knowledge_pipeline" / "fixtures"
GOLDEN = FIXTURES / "tbox_golden.yaml"


def _iso(minute: int = 0) -> str:
    return datetime(2025, 1, 1, 12, minute, tzinfo=timezone.utc).isoformat()


def make_record(node_id: str, status: str, **overrides) -> dict:
    """构造一条合法账本记录（字段对齐 writeback.record_schema）。"""
    rec = {
        "node_id": node_id,
        "status": status,
        "evidence": f"trace/{node_id}.log",
        "metrics": {},
        "posterior": 0.5,
        "notes": "",
        "ts": _iso(),
    }
    rec.update(overrides)
    return rec


def make_03_scenario_records() -> list[dict]:
    """03 文档第三部分的账本场景：H1/H2 confirmed、H3 uncertain；H4/H5 未验（无记录）。"""
    return [
        make_record(
            "H1", "confirmed",
            evidence="trace/wiring_report + fix_commit",
            metrics={"gain": 0.28},
            posterior=update_posterior(0.5, "confirmed"),  # 0.5 → 0.95
            notes="断链修复，零成本释放收益",
            ts=_iso(4),
        ),
        make_record(
            "H2", "confirmed",
            evidence="trace/run_result_h2",
            metrics={"gain": 0.12},
            posterior=update_posterior(0.7, "confirmed"),
            notes="阈值路线实测 +0.12",
            ts=_iso(8),
        ),
        make_record(
            "H3", "uncertain",
            evidence="trace/run_result_h3",
            metrics={"gain": 0.03, "rerun_gain": 0.01},
            posterior=update_posterior(0.6, "uncertain"),  # 不变
            notes="边际提升复跑仍边际，判 uncertain，不计入战果",
            ts=_iso(10),
        ),
        # H4/H5 未验：不落账本（见模块 docstring 裁决备注）
    ]


# --------------------------------------------------------------------------
# a) 合法记录追加、读回一致
# --------------------------------------------------------------------------


def test_append_and_read_back(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "hypotheses.jsonl")
    recs = make_03_scenario_records()
    for r in recs:
        ledger.append(r)
    assert ledger.records() == recs  # 读回与写入逐字段一致


def test_records_empty_when_file_missing(tmp_path: Path) -> None:
    assert Ledger(tmp_path / "nope.jsonl").records() == []


def test_append_writes_jsonl_one_line_per_record(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.jsonl"
    ledger = Ledger(path)
    ledger.append(make_record("H1", "confirmed"))
    ledger.append(make_record("H2", "refuted"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["node_id"] == "H1"
    assert json.loads(lines[1])["node_id"] == "H2"


# --------------------------------------------------------------------------
# b) 非法写回：缺字段 / status 非法 / metrics 非 dict → ValueError 且不落盘
# --------------------------------------------------------------------------


def _illegal_records() -> list[dict]:
    missing_field = make_record("H1", "confirmed")
    del missing_field["metrics"]
    bad_status = make_record("H1", "verified")  # 不在词表内
    metrics_not_dict = make_record("H1", "confirmed", metrics="F1 +0.12")  # 文本而非数据
    missing_posterior = make_record("H1", "confirmed")
    del missing_posterior["posterior"]
    bad_posterior = make_record("H1", "confirmed", posterior=1.5)
    return [missing_field, bad_status, metrics_not_dict, missing_posterior, bad_posterior]


@pytest.mark.parametrize("bad", _illegal_records(), ids=["缺字段", "status非法", "metrics非dict", "缺posterior", "posterior越界"])
def test_illegal_write_rejected_and_file_untouched(tmp_path: Path, bad: dict) -> None:
    path = tmp_path / "hypotheses.jsonl"
    ledger = Ledger(path)
    with pytest.raises(ValueError):
        ledger.append(bad)
    assert not path.exists()  # 校验先于打开文件：非法写回不触碰磁盘


def test_illegal_write_does_not_corrupt_existing_ledger(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.jsonl"
    ledger = Ledger(path)
    good = make_record("H1", "confirmed")
    ledger.append(good)
    before = path.read_text(encoding="utf-8")
    for bad in _illegal_records():
        with pytest.raises(ValueError):
            ledger.append(bad)
    assert path.read_text(encoding="utf-8") == before  # 既有内容零改动
    assert ledger.records() == [good]


def test_status_vocabulary_is_spec_frozen() -> None:
    assert set(STATUSES) == {"confirmed", "refuted", "uncertain", "blocked"}


# --------------------------------------------------------------------------
# c) append-only 语义：重复 append 只增不改
# --------------------------------------------------------------------------


def test_append_only_semantics(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.jsonl"
    ledger = Ledger(path)
    rec1 = make_record("H1", "uncertain", ts=_iso(1))
    rec2 = make_record("H1", "confirmed", metrics={"gain": 0.28}, ts=_iso(2))  # 同一假说再次写回
    ledger.append(rec1)
    after_first = path.read_text(encoding="utf-8")
    ledger.append(rec2)
    after_second = path.read_text(encoding="utf-8")
    assert after_second.startswith(after_first)  # 旧行原样保留，只追加不改写
    assert ledger.records() == [rec1, rec2]  # 顺序即写入顺序


# --------------------------------------------------------------------------
# d) 后验更新：各分支 + 0.5→0.95 锚点
# --------------------------------------------------------------------------


def test_update_posterior_anchor_03_doc() -> None:
    """03 文档锚点：H1 prior 0.5 confirmed → 0.95。"""
    assert update_posterior(0.5, "confirmed") == pytest.approx(0.95)


def test_update_posterior_confirmed_branch() -> None:
    assert update_posterior(0.7, "confirmed") == pytest.approx(0.7 + 0.3 * 0.9)  # 0.97
    assert update_posterior(0.0, "confirmed") == pytest.approx(0.9)
    assert update_posterior(1.0, "confirmed") == pytest.approx(1.0)


def test_update_posterior_refuted_branch() -> None:
    assert update_posterior(0.7, "refuted") == pytest.approx(0.07)
    assert update_posterior(0.0, "refuted") == pytest.approx(0.0)


def test_update_posterior_uncertain_and_blocked_unchanged() -> None:
    for status in ("uncertain", "blocked"):
        assert update_posterior(0.6, status) == pytest.approx(0.6)


def test_update_posterior_rejects_illegal_input() -> None:
    with pytest.raises(ValueError):
        update_posterior(0.5, "verified")
    with pytest.raises(ValueError):
        update_posterior(1.2, "confirmed")
    with pytest.raises(ValueError):
        update_posterior(-0.1, "confirmed")


# --------------------------------------------------------------------------
# e) 黄金夹具 + 03 场景：domain_report 领域聚合
# --------------------------------------------------------------------------


def test_domain_report_03_scenario() -> None:
    graph = load_graph(GOLDEN)
    records = make_03_scenario_records()
    report = domain_report(records, graph)

    # 覆盖图内全部领域（含未探索）
    assert set(report) == {"D1", "D2", "D3", "D4"}

    # D1 接线完整性：H1 confirmed +0.28 → 确认高收益 → prior 上调（03：接线域 up）
    d1 = report["D1"]
    assert d1["title"] == "接线完整性"
    assert (d1["confirmed"], d1["refuted"], d1["uncertain"], d1["blocked"]) == (1, 0, 0, 0)
    assert d1["total_gain"] == pytest.approx(0.28)
    assert d1["prior_suggestion"] == "up"

    # D2 类别不平衡处理：H2 confirmed +0.12、H3 uncertain 不计入战果
    d2 = report["D2"]
    assert (d2["confirmed"], d2["refuted"], d2["uncertain"], d2["blocked"]) == (1, 0, 1, 0)
    assert d2["total_gain"] == pytest.approx(0.12)  # H3 的 0.03/0.01 不计入
    assert d2["prior_suggestion"] == "up"  # SPEC §5：confirmed≥1 且有正收益 → up

    # D3 特征空间：H5 未验 → 未探索 → 保留（03：特征域 hold）
    d3 = report["D3"]
    assert (d3["confirmed"], d3["refuted"], d3["uncertain"], d3["blocked"]) == (0, 0, 0, 0)
    assert d3["total_gain"] == 0.0
    assert d3["prior_suggestion"] == "hold"

    # D4 评估稳健性：图内无账本记录 → 未探索 hold
    assert report["D4"]["prior_suggestion"] == "hold"


def test_domain_report_action_records_map_to_parent_domain() -> None:
    """action 节点记录沿 belongs_to 链映射回领域（A2→H1→D1）。"""
    graph = load_graph(GOLDEN)
    records = [make_record("A2", "confirmed", metrics={"gain": 0.1})]
    report = domain_report(records, graph)
    assert report["D1"]["confirmed"] == 1
    assert report["D1"]["total_gain"] == pytest.approx(0.1)


def test_domain_report_unknown_node_ignored() -> None:
    graph = load_graph(GOLDEN)
    records = [make_record("H99", "confirmed", metrics={"gain": 9.9})]
    report = domain_report(records, graph)
    assert all(b["confirmed"] == 0 and b["total_gain"] == 0.0 for b in report.values())


def test_domain_report_prior_suggestion_rules() -> None:
    """SPEC §5 规则矩阵：up / down / hold 各分支。"""
    graph = load_graph(GOLDEN)
    base = make_03_scenario_records()  # D1 up、D2 up、D3/D4 hold
    report = domain_report(base, graph)
    assert report["D1"]["prior_suggestion"] == "up"

    # 仅 uncertain 且无正收益 → down（03：加权域边际→降权 的规则形态）
    only_uncertain = [make_record("H5", "uncertain", metrics={"gain": 0.03})]
    assert domain_report(only_uncertain, graph)["D3"]["prior_suggestion"] == "down"

    # 仅 refuted → down
    only_refuted = [make_record("H5", "refuted", metrics={"gain": -0.02})]
    assert domain_report(only_refuted, graph)["D3"]["prior_suggestion"] == "down"

    # confirmed 但无正收益（gain 非数值或 ≤0）→ hold（不 up 不 down）
    confirmed_no_gain = [make_record("H5", "confirmed", metrics={})]
    assert domain_report(confirmed_no_gain, graph)["D3"]["prior_suggestion"] == "hold"

    # 仅 blocked → hold
    only_blocked = [make_record("H5", "blocked")]
    assert domain_report(only_blocked, graph)["D3"]["prior_suggestion"] == "hold"


def test_domain_report_gain_key_override() -> None:
    """total_gain 口径：调用方可指定 metric 键（工程默认值 gain_key='gain'）。"""
    graph = load_graph(GOLDEN)
    records = [make_record("H1", "confirmed", metrics={"f1_delta": 0.28})]
    assert domain_report(records, graph)["D1"]["total_gain"] == 0.0  # 默认键无命中
    assert domain_report(records, graph, gain_key="f1_delta")["D1"]["total_gain"] == pytest.approx(0.28)


def test_domain_report_does_not_mutate_inputs() -> None:
    graph = load_graph(GOLDEN)
    records = make_03_scenario_records()
    snapshot = copy.deepcopy(records)
    domain_report(records, graph)
    assert records == snapshot
