"""M5 端到端测试：一键流水线（编译 → 校验 → 模拟 → 回写 → 聚合）。

验收口径（00_README §3-M5 + SPEC §7）：

a) 默认离线链（CachedClient + 预录 TBox 编译输出）跑通，退出码 0，
   产物六件套齐全且 JSON/YAML 可解析；
b) 产物内容正确性：trace 含 decision/writeback/acceptance_check 事件；
   hypotheses.jsonl 每行过 M3 Ledger 校验；domain_report 覆盖 4 领域；
   acceptance_report 5 条全过且与 tbox_playbook 事实一致；
c) 编译缓存缺失或损坏时非 0 退出且 stderr 报错可读；
d) 与 M2 一致性：e2e 的假说裁决与 M2 黄金全链演示一致
   （H1 confirmed 0.95 / H2 confirmed / H3 uncertain / H6 confirmed）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from knowledge_pipeline.e2e.run_pipeline import ARTIFACTS, DECISIONS_LOG
from knowledge_pipeline.ledger import Ledger
from knowledge_pipeline.schema import load_graph
from knowledge_pipeline.simulator import (
    MockExecutor,
    SimConfig,
    Simulator,
    load_playbook,
    read_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "knowledge_pipeline" / "fixtures"
LLM_CACHE = FIXTURES / "llm_cache"
GOLDEN = FIXTURES / "tbox_golden.yaml"
PLAYBOOK = FIXTURES / "tbox_playbook.yaml"


def _run_cli(*argv: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """以子进程跑 ``python -m knowledge_pipeline.e2e.run_pipeline``。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "knowledge_pipeline.e2e.run_pipeline", *argv],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def e2e_run(tmp_path_factory):
    """模块级：默认离线链子进程跑一次，全组用例共享产物。"""
    out = tmp_path_factory.mktemp("e2e_out")
    proc = _run_cli("--out", str(out))
    return proc, out


class TestRerunIdempotency:
    """终审 Major-1 回归：同一 --out 目录重跑不得污染账本与聚合报告。"""

    def test_rerun_same_out_dir_does_not_duplicate_ledger(self, tmp_path):
        out = tmp_path / "rerun"
        first = _run_cli("--out", str(out))
        assert first.returncode == 0, first.stderr
        baseline = (out / "hypotheses.jsonl").read_text(encoding="utf-8")
        second = _run_cli("--out", str(out))
        assert second.returncode == 0, second.stderr
        rerun = (out / "hypotheses.jsonl").read_text(encoding="utf-8")
        assert len([l for l in rerun.splitlines() if l.strip()]) == len(
            [l for l in baseline.splitlines() if l.strip()]
        ), "重跑后账本条目数必须不变（重跑清旧账纪律）"
        report = json.loads((out / "domain_report.json").read_text(encoding="utf-8"))
        assert report["D2"]["confirmed"] == 1, "重跑后领域聚合不得翻倍"


# --------------------------------------------------------------------------
# a) 默认离线链：退出码 0 + 六件套齐全可解析
# --------------------------------------------------------------------------


class TestOfflineChainArtifacts:
    def test_exit_code_zero(self, e2e_run):
        proc, _ = e2e_run
        assert proc.returncode == 0, (
            f"退出码非 0: {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        assert proc.stderr == ""  # 全链通过时 stderr 干净
        # 终报 stdout 打印分支与理由（决策日志可溯源）
        assert "J1" in proc.stdout and "理由" in proc.stdout
        assert "J2" in proc.stdout and "320" in proc.stdout

    def test_six_artifacts_present_and_parseable(self, e2e_run):
        _, out = e2e_run
        for name in ARTIFACTS:
            assert (out / name).is_file(), f"缺产物 {name}"
        # YAML 可解析且是干净图（过 M1 校验的编译产物）
        graph = yaml.safe_load((out / "graph.yaml").read_text(encoding="utf-8"))
        assert isinstance(graph, dict) and graph.get("nodes") and graph.get("edges")
        # JSON 产物可解析
        for name in ("card_decisions.json", "domain_report.json", "acceptance_report.json"):
            payload = json.loads((out / name).read_text(encoding="utf-8"))
            assert payload, f"{name} 为空"
        # JSONL 产物逐行可解析
        for name in ("trace.jsonl", "hypotheses.jsonl"):
            lines = (out / name).read_text(encoding="utf-8").splitlines()
            assert lines, f"{name} 为空"
            for line in lines:
                json.loads(line)
        # SPEC §7 口径的决策日志（附加产物）
        log_text = (out / DECISIONS_LOG).read_text(encoding="utf-8")
        assert "J1" in log_text and "J2" in log_text and "理由" in log_text

    def test_graph_yaml_matches_compiled_golden_shape(self, e2e_run):
        """编译产物与黄金图同构（缓存重放的 M4 契约）。"""
        _, out = e2e_run
        graph = yaml.safe_load((out / "graph.yaml").read_text(encoding="utf-8"))
        golden = load_graph(GOLDEN)
        assert {n["id"] for n in graph["nodes"]} == {n["id"] for n in golden["nodes"]}
        assert {d["id"] for d in graph["decisions"]} == {d["id"] for d in golden["decisions"]}

    def test_card_decisions_cover_full_card_library(self, e2e_run):
        _, out = e2e_run
        decisions = json.loads((out / "card_decisions.json").read_text(encoding="utf-8"))
        assert len(decisions) == 29  # 卡库全部卡片，adopt/reject 均有 rationale
        assert all(d["decision"] in ("adopt", "reject") and d["rationale"] for d in decisions)


# --------------------------------------------------------------------------
# b) 产物内容正确性
# --------------------------------------------------------------------------


class TestArtifactContents:
    def test_trace_contains_required_events(self, e2e_run):
        _, out = e2e_run
        events = read_trace(out / "trace.jsonl")
        kinds = {e["event"] for e in events}
        assert {"decision", "writeback", "acceptance_check"} <= kinds
        # decision 事件可溯源：分支索引 + 条件 + 理由齐备
        decisions = [e for e in events if e["event"] == "decision"]
        assert {e["node_id"] for e in decisions} == {"J1", "J2"}
        for e in decisions:
            assert isinstance(e["detail"]["branch_index"], int)
            assert e["detail"]["reason"]
        # 复现 03 时间线分支：J1 走"修复"（0），J2 走"阈值+K折升级"（1）
        j1 = next(e for e in decisions if e["node_id"] == "J1")
        j2 = next(e for e in decisions if e["node_id"] == "J2")
        assert j1["detail"]["branch_index"] == 0
        assert j2["detail"]["branch_index"] == 1
        assert "320" in j2["detail"]["reason"]
        # 终局验收核对事件 5/5
        final_check = [e for e in events if e["event"] == "acceptance_check"
                       and e["detail"].get("final")]
        assert final_check and final_check[-1]["detail"]["passed_count"] == 5
        assert events[-1]["event"] == "run_end"

    def test_hypotheses_jsonl_passes_ledger_validation(self, e2e_run):
        _, out = e2e_run
        lines = (out / "hypotheses.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4  # H1/H6/H2/H3 落账；H4/H5 未验不落账
        for line in lines:
            Ledger.validate_record(json.loads(line))  # 非法写回应被拒，这里全合法

    def test_domain_report_covers_four_domains(self, e2e_run):
        _, out = e2e_run
        report = json.loads((out / "domain_report.json").read_text(encoding="utf-8"))
        assert set(report) == {"D1", "D2", "D3", "D4"}
        for bucket in report.values():
            assert {"title", "confirmed", "refuted", "uncertain", "blocked",
                    "total_gain", "prior_suggestion"} <= set(bucket)
        # D2：H2 confirmed +0.12（计入战果）、H3 uncertain（不计入）→ up
        d2 = report["D2"]
        assert (d2["confirmed"], d2["uncertain"]) == (1, 1)
        assert d2["total_gain"] == pytest.approx(0.12)
        assert d2["prior_suggestion"] == "up"
        # D3 特征空间未探索 → hold
        assert report["D3"]["prior_suggestion"] == "hold"

    def test_acceptance_report_five_pass_matching_playbook_facts(self, e2e_run):
        _, out = e2e_run
        acc = json.loads((out / "acceptance_report.json").read_text(encoding="utf-8"))
        assert acc["total"] == 5 and acc["passed"] == 5 and acc["all_passed"] is True
        by_assertion = {r["assertion"]: r for r in acc["results"]}
        assert all(r["passed"] for r in by_assertion.values())
        # 实际值与 tbox_playbook.yaml 事实演化一致
        assert by_assertion["per_app.王者荣耀.F1 >= 0.75"]["actual"] == pytest.approx(0.79)
        assert by_assertion["overall.F1 >= 0.935"]["actual"] == pytest.approx(0.937)
        assert by_assertion["overall.Recall >= 0.97"]["actual"] == pytest.approx(0.971)
        assert by_assertion["per_app.bilibili.F1_delta >= -0.01"]["actual"] == pytest.approx(-0.004)
        assert by_assertion["per_app.抖音.F1_delta >= -0.01"]["actual"] == pytest.approx(-0.003)


# --------------------------------------------------------------------------
# c) 编译缓存缺失 / 损坏 → 非 0 退出 + 可读报错
# --------------------------------------------------------------------------


class TestCacheFailures:
    def test_cache_dir_missing(self, tmp_path):
        proc = _run_cli("--out", str(tmp_path / "out"),
                        "--cache", str(tmp_path / "no_such_cache"))
        assert proc.returncode != 0
        assert "编译" in proc.stderr and "缓存" in proc.stderr

    def test_cache_dir_empty(self, tmp_path):
        empty = tmp_path / "empty_cache"
        empty.mkdir()
        proc = _run_cli("--out", str(tmp_path / "out"), "--cache", str(empty))
        assert proc.returncode != 0
        assert "缓存未命中" in proc.stderr or "cache" in proc.stderr

    def test_cache_entries_corrupted(self, tmp_path):
        corrupt = tmp_path / "corrupt_cache"
        shutil.copytree(LLM_CACHE, corrupt)
        for txt in corrupt.glob("*.txt"):
            txt.write_text("这不是 JSON 也不是 YAML 的损坏内容", encoding="utf-8")
        proc = _run_cli("--out", str(tmp_path / "out"), "--cache", str(corrupt))
        assert proc.returncode != 0
        assert "编译" in proc.stderr  # 阶段名 + 人类可读原因
        assert proc.stderr.strip(), "stderr 必须给出可读原因"


# --------------------------------------------------------------------------
# d) 与 M2 黄金全链演示一致
# --------------------------------------------------------------------------


class TestConsistencyWithM2:
    EXPECTED = {  # M2 TestGoldenTimeline 断言口径
        "H1": ("confirmed", 0.95),   # posterior 0.5→0.95
        "H2": ("confirmed", 0.97),   # 0.7+(1-0.7)*0.9
        "H3": ("uncertain", 0.6),    # uncertain → 后验不变
        "H6": ("confirmed", 0.96),   # 0.6+(1-0.6)*0.9
    }

    def test_ledger_outcomes_match_m2_golden_demo(self, e2e_run):
        _, out = e2e_run
        records = {
            r["node_id"]: r
            for r in (json.loads(line) for line in
                      (out / "hypotheses.jsonl").read_text(encoding="utf-8").splitlines())
        }
        assert set(records) == {"H1", "H2", "H3", "H6"}  # H4/H5 未验不落账
        for hyp, (status, posterior) in self.EXPECTED.items():
            assert records[hyp]["status"] == status, f"{hyp} 状态与 M2 演示不一致"
            assert records[hyp]["posterior"] == pytest.approx(posterior)
        assert records["H2"]["metrics"]["gain"] == pytest.approx(0.12)
        assert "不计入战果" in records["H3"]["notes"]

    def test_same_as_direct_m2_golden_run(self, e2e_run, tmp_path):
        """直接以 M2 接口跑黄金图+playbook，与 e2e 产物逐假说比对。"""
        _, out = e2e_run
        ledger = Ledger(tmp_path / "m2_hypotheses.jsonl")
        sim = Simulator(load_graph(GOLDEN), MockExecutor(load_playbook(PLAYBOOK)),
                        ledger, SimConfig())
        result = sim.run()
        e2e_records = {
            r["node_id"]: r
            for r in (json.loads(line) for line in
                      (out / "hypotheses.jsonl").read_text(encoding="utf-8").splitlines())
        }
        m2_records = {r["node_id"]: r for r in ledger.records()}
        assert set(e2e_records) == set(m2_records)
        for hyp, rec in m2_records.items():
            assert e2e_records[hyp]["status"] == rec["status"]
            assert e2e_records[hyp]["posterior"] == pytest.approx(rec["posterior"])
            assert e2e_records[hyp]["metrics"]["gains"] == rec["metrics"]["gains"]
        # 假说级结果与 M2 SimResult.hypothesis_outcomes 一致
        for hyp, (status, _) in self.EXPECTED.items():
            assert result.hypothesis_outcomes[hyp]["status"] == status
