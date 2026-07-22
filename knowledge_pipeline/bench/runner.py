"""M6：评测台运行器 —— 跑全部剧本，聚合成 ``bench_report.json``。

语义来源：SPEC §8（输出 bench_report.json：每剧本 × 双方 × 指标 + 汇总）、
03 文档第四部分（同一剧本下 (a) 按图执行 vs (b) 自由探索，唯一变量是编排；
每剧本独立计分再聚合）。

用法：

.. code-block:: bash

    python -m knowledge_pipeline.bench.runner --out bench_out

每剧本产物（``<out>/<剧本id>/``）：``trace.jsonl``（有编排侧 M2 Trace）、
``baseline_trace.jsonl``（无编排侧轨迹）、``hypotheses.jsonl``（M3 账本）。

工程默认值（列入交付报告「新发现」）：

- 有编排侧 SimConfig 取 ``k=5``（E7 早停阈值）：bench 目标是走完全图以观察
  各剧本终局，而非尽早收工；k=3 会在 S5 等"验收长期无进展"剧本中提前
  截断边际提升复跑剧情。
- 每剧本运行前清理同名旧账本/轨迹，保证重跑可重复（回归测试套件纪律）。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ..ledger import Ledger
from ..simulator import MockExecutor, SimConfig, Simulator
from .baseline_agent import BaselineAgent, BaselineResult
from .metrics import METRIC_NAMES, aggregate, compute_metrics
from .scripts import Scenario, load_scripts

__all__ = ["SIDES", "run_scenario", "run_bench", "main"]

SIDES: tuple[str, ...] = ("orchestrated", "baseline")

#: 有编排侧早停阈值（见模块 docstring 工程默认值）
_BENCH_EARLY_STOP_K = 5


def run_orchestrated(scenario: Scenario, out_dir: Path) -> tuple[list[dict], dict]:
    """有编排侧：M2 Simulator 按图执行，返回 (事件流, 终报)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    ledger_path = out_dir / "hypotheses.jsonl"
    for p in (trace_path, ledger_path):
        if p.exists():
            p.unlink()  # 重跑清旧账（可重复性纪律）
    sim = Simulator(
        scenario.graph,
        MockExecutor(copy.deepcopy(scenario.playbook)),  # 运行中就地演化 facts，防污染剧本对象
        Ledger(ledger_path),
        SimConfig(k=_BENCH_EARLY_STOP_K, trace_path=trace_path),
    )
    result = sim.run()
    return result.events, result.final_report


def run_baseline_side(scenario: Scenario, out_dir: Path) -> BaselineResult:
    """无编排侧：确定性启发式基线自由探索。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = BaselineAgent(scenario).run()
    trace_path = out_dir / "baseline_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as fh:
        for event in result.events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return result


def run_scenario(scenario: Scenario, out_dir: Path) -> dict[str, Any]:
    """跑单个剧本的 A/B 对照，返回双侧指标 + 摘要。"""
    sc_dir = out_dir / scenario.id
    orch_events, orch_report = run_orchestrated(scenario, sc_dir)
    baseline = run_baseline_side(scenario, sc_dir)
    return {
        "title": scenario.title,
        "trigger": scenario.trigger,
        "env_response": scenario.env_response,
        "expected_response": scenario.expected_response,
        "orchestrated": {
            **compute_metrics(orch_events, "orchestrated", scenario),
            "end_reason": orch_report.get("end_reason"),
            "acceptance_passed": orch_report.get("acceptance_passed"),
            "trace_path": str((sc_dir / "trace.jsonl")),
        },
        "baseline": {
            **compute_metrics(baseline.events, "baseline", scenario),
            "stop_reason": baseline.stop_reason,
            "acceptance_passed": dict(baseline.acceptance_passed),
            "trace_path": str(sc_dir / "baseline_trace.jsonl"),
        },
    }


def _preregistered_checks(per_scenario: dict[str, Any]) -> dict[str, Any]:
    """预登记对照检查（03 文档指标表中的硬目标）：S4 陷阱识别、S5 噪声。"""
    s4 = per_scenario.get("S4", {})
    s5 = per_scenario.get("S5", {})
    checks = {
        "s4_trap_identified": {
            "orchestrated": s4.get("orchestrated", {}).get("s4_trap_identified"),
            "baseline": s4.get("baseline", {}).get("s4_trap_identified"),
        },
        "s5_noise_as_win": {
            "orchestrated": s5.get("orchestrated", {}).get("s5_noise_as_win"),
            "baseline": s5.get("baseline", {}).get("s5_noise_as_win"),
        },
    }
    checks["s4_trap_identified"]["holds"] = (
        checks["s4_trap_identified"]["orchestrated"] == 1.0
        and checks["s4_trap_identified"]["baseline"] == 0.0
    )
    checks["s5_noise_as_win"]["holds"] = (
        checks["s5_noise_as_win"]["orchestrated"] == 0
        and (checks["s5_noise_as_win"]["baseline"] or 0) >= 1
    )
    return checks


def run_bench(
    out_dir: str | Path = "bench_out",
    scripts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """跑全部剧本并写出 ``bench_report.json``，返回报告 dict。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = load_scripts(scripts_dir)

    per_scenario: dict[str, Any] = {}
    for sid in sorted(scenarios):
        per_scenario[sid] = run_scenario(scenarios[sid], out)

    metrics_only = {
        sid: {side: {m: entry[side][m] for m in METRIC_NAMES} for side in SIDES}
        for sid, entry in per_scenario.items()
    }
    report = {
        "meta": {
            "module": "M6 量化评测台",
            "scenarios": sorted(scenarios),
            "sides": list(SIDES),
            "early_stop_k": _BENCH_EARLY_STOP_K,
            "deterministic": True,
        },
        "metrics": list(METRIC_NAMES),
        "scenarios": per_scenario,
        "aggregate": aggregate(metrics_only),
        "preregistered_checks": _preregistered_checks(per_scenario),
    }
    report_path = out / "bench_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M6 量化评测台：S1~S6 A/B 对照")
    parser.add_argument("--out", default="bench_out", help="产物目录（默认 bench_out）")
    parser.add_argument("--scripts", default=None, help="剧本库目录（默认 fixtures/bench）")
    args = parser.parse_args(argv)
    report = run_bench(args.out, args.scripts)
    checks = report["preregistered_checks"]
    print(f"bench_report: {Path(args.out) / 'bench_report.json'}")
    for name, check in checks.items():
        print(
            f"  {name}: orchestrated={check['orchestrated']} "
            f"baseline={check['baseline']} holds={check['holds']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
