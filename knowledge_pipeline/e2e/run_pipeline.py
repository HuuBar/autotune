"""M5：端到端一键流水线——编译 → 校验 → 模拟执行 → 回写 → 聚合报告。

用法：

.. code-block:: bash

    python -m knowledge_pipeline.e2e.run_pipeline [--out e2e_out/] [--live]
    # 或
    bash scripts/run_e2e.sh [--out e2e_out/] [--live]

默认离线链（复现 03 文档第二部分 TBox 逐拍时间线）：

1. **编译**（M4）：:class:`CachedClient` 重放 ``fixtures/llm_cache`` 预录响应，
   对 ``fixtures/tbox_brief.yaml`` × 知识卡库跑 ``compile_graph``
   （选卡论证 → 成图 → M1 确定性校验门，不绕过校验）；
   ``--live`` 切换为 :class:`OpenAICompatClient`，端点由 ``KP_LLM_BASE_URL`` /
   ``KP_LLM_MODEL`` / ``KP_LLM_API_KEY`` 环境变量注入。
2. **校验**（M1）：落盘后的 ``graph.yaml`` 再过一次 ``validate_graph``
   （编译门的延续，保证产物即校验通过态）。
3. **模拟执行**（M2）：``Simulator + MockExecutor`` 按
   ``fixtures/tbox_playbook.yaml`` 演出事故剧本（断链修复、重复列、
   J2 走"阈值+K折升级"分支、H3 边际提升复跑判 uncertain），
   轨迹写 ``trace.jsonl``，账本写产物目录 ``hypotheses.jsonl``（E8 回写）。
4. **聚合**（M3）：``domain_report`` 按领域聚合收益与 prior 调整建议，
   落 ``domain_report.json``；终局验收断言逐条核对落 ``acceptance_report.json``。

产物（``--out`` 目录，默认 ``e2e_out/``；前六件为交付契约六件套）：

- ``graph.yaml``            编译产出的干净 YAML（过 M1 校验）
- ``card_decisions.json``   选卡论证表（每张卡 adopt/reject + rationale）
- ``trace.jsonl``           执行轨迹（decision 事件含分支与理由，决策可溯源）
- ``hypotheses.jsonl``      假说账本（append-only，每条过 M3 校验）
- ``domain_report.json``    领域聚合报告（收益 + prior_suggestion）
- ``acceptance_report.json`` 验收断言逐条核对结果
- ``decisions.log``         决策日志（SPEC §7 口径；与 trace 中 decision 事件同源）

退出码：全链通过（产物齐全、验收断言核对完成、无未决阻塞）= 0；
任何阶段失败 / 验收断言未全过 / 存在未决阻塞 = 非 0，stderr 给人类可读原因。

工程默认值（SPEC §7 未覆盖处，列入回传报告「新发现」）：

- SPEC §7 产物清单中的 ``decisions.log`` 与任务书六件套中的
  ``card_decisions.json`` 语义不同（前者是决策分支日志，后者是选卡论证表），
  本实现**两份都产**，不牺牲任一契约。
- "验收断言核对完成"的通过口径：全部断言 passed 且无未决阻塞才记退出码 0；
  链跑完但验收未全过时仍写齐产物，以退出码 1 + stderr 说明收场。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ..compiler import (
    CachedClient,
    CompileError,
    OpenAICompatClient,
    compile_graph,
)
from ..compiler.llm_client import LLMResponseError
from ..ledger import Ledger, domain_report
from ..simulator import (
    MockExecutor,
    SimConfig,
    Simulator,
    load_playbook,
)
from ..validator import validate_graph

__all__ = [
    "ARTIFACTS",
    "DEFAULT_OUT",
    "StageError",
    "build_parser",
    "main",
    "run_pipeline",
]

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # knowledge_pipeline/
_REPO_ROOT = _PACKAGE_ROOT.parent

#: 默认输入（离线链）：卡库 / 任务书 / 预录缓存 / 事故剧本
DEFAULT_CARDS = _REPO_ROOT / "docs" / "knowledge-pipeline" / "自动化ML优化战术知识卡_v1.md"
DEFAULT_BRIEF = _PACKAGE_ROOT / "fixtures" / "tbox_brief.yaml"
DEFAULT_CACHE = _PACKAGE_ROOT / "fixtures" / "llm_cache"
DEFAULT_PLAYBOOK = _PACKAGE_ROOT / "fixtures" / "tbox_playbook.yaml"

#: 默认产物目录（SPEC §7）
DEFAULT_OUT = "e2e_out"

#: 交付契约六件套（--out 目录内）
ARTIFACTS: tuple[str, ...] = (
    "graph.yaml",
    "card_decisions.json",
    "trace.jsonl",
    "hypotheses.jsonl",
    "domain_report.json",
    "acceptance_report.json",
)

#: SPEC §7 口径的附加产物（决策日志，与 trace 的 decision 事件同源）
DECISIONS_LOG = "decisions.log"


class StageError(RuntimeError):
    """流水线某一阶段失败（``stage`` 为阶段名，消息为人类可读原因）。"""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="M5 端到端一键流水线：编译 → 校验 → 模拟执行 → 回写 → 聚合报告",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"产物目录（默认 {DEFAULT_OUT}/）",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="走真实 LLM 端点（OpenAICompatClient，KP_LLM_BASE_URL/KP_LLM_MODEL/"
        "KP_LLM_API_KEY 环境变量注入）；默认离线重放预录缓存",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="离线模式 LLM 预录缓存目录（默认 fixtures/llm_cache）",
    )
    parser.add_argument("--cards", default=str(DEFAULT_CARDS), help="知识卡库 Markdown")
    parser.add_argument("--brief", default=str(DEFAULT_BRIEF), help="任务书+地形 YAML")
    parser.add_argument(
        "--playbook", default=str(DEFAULT_PLAYBOOK), help="模拟执行事故剧本 YAML"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="编译器每步 LLM 最大尝试次数（默认 3）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="E7 早停阈值：连续 k 个假说验证无验收进展（默认 3）",
    )
    return parser


# --------------------------------------------------------------------------
# 各阶段
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _stage_compile(args: argparse.Namespace):
    """阶段 1：编译（M4；离线 CachedClient / --live OpenAICompatClient）。"""
    cards, brief = Path(args.cards), Path(args.brief)
    for p, label in ((cards, "知识卡库"), (brief, "任务书")):
        if not p.is_file():
            raise StageError("编译", f"{label}文件不存在: {p}")
    if args.live:
        try:
            client = OpenAICompatClient()  # 端点由 KP_LLM_* 环境变量注入
        except ValueError as exc:
            raise StageError("编译", f"live 模式端点配置缺失: {exc}") from exc
        mode = f"live（{client.endpoint}，model={client.model}）"
    else:
        try:
            client = CachedClient(args.cache)
        except ValueError as exc:
            raise StageError("编译", f"离线缓存不可用: {exc}") from exc
        mode = f"offline（缓存 {args.cache}）"
    try:
        result = compile_graph(cards, brief, client, max_retries=args.max_retries)
    except KeyError as exc:
        # CachedClient 缓存未命中（含缓存损坏/条目缺失）
        raise StageError(
            "编译",
            f"LLM 预录缓存未命中（缓存缺失或损坏）: {exc}",
        ) from exc
    except LLMResponseError as exc:
        raise StageError("编译", f"LLM 端点调用失败: {exc}") from exc
    except CompileError as exc:
        raise StageError("编译", str(exc)) from exc
    print(f"[编译] 模式={mode}；选卡 {len(result.card_decisions)} 张"
          f"（adopt {sum(1 for d in result.card_decisions if d['decision'] == 'adopt')} / "
          f"reject {sum(1 for d in result.card_decisions if d['decision'] == 'reject')}），"
          f"选卡调用 {result.select_attempts} 次、成图调用 {result.attempts} 次，"
          f"已过 M1 校验门 + 编译器自检")
    return result


def _stage_validate(graph: dict) -> None:
    """阶段 2：校验（M1；落盘图再过确定性校验门，不绕过）。"""
    errors = validate_graph(graph)
    if errors:
        detail = "\n".join(f"  - {e}" for e in errors[:20])
        raise StageError("校验", f"编译产物未通过 M1 校验（{len(errors)} 条）:\n{detail}")
    print(f"[校验] graph.yaml 通过 M1 确定性校验（V-* 全绿，"
          f"{len(graph.get('nodes') or [])} 节点 / "
          f"{len(graph.get('edges') or [])} 边 / "
          f"{len(graph.get('decisions') or [])} 决策点）")


def _stage_simulate(args: argparse.Namespace, graph: dict, out: Path):
    """阶段 3：模拟执行 + 回写（M2 Simulator × MockExecutor × M3 Ledger）。"""
    playbook_path = Path(args.playbook)
    if not playbook_path.is_file():
        raise StageError("模拟执行", f"playbook 文件不存在: {playbook_path}")
    try:
        playbook = load_playbook(playbook_path)
    except Exception as exc:
        raise StageError("模拟执行", f"playbook 加载失败: {exc}") from exc
    ledger_path = out / "hypotheses.jsonl"
    trace_path = out / "trace.jsonl"
    for p in (ledger_path, trace_path):
        if p.exists():
            p.unlink()  # 重跑清旧账（append-only 账本的可重复性纪律，同 bench runner）
    ledger = Ledger(ledger_path)
    sim = Simulator(
        graph,
        MockExecutor(playbook),
        ledger,
        SimConfig(k=args.k, trace_path=trace_path),
    )
    try:
        result = sim.run()
    except Exception as exc:
        raise StageError("模拟执行", f"调度器运行失败: {exc}") from exc
    print(f"[模拟] {result.final_report['ticks']} ticks 跑完"
          f"（结束原因: {result.final_report['end_reason']}）；"
          f"账本写回 {len(ledger.records())} 条假说裁决")
    return result, ledger


def _stage_aggregate(graph: dict, ledger: Ledger, sim_result, out: Path) -> dict:
    """阶段 4：聚合（M3 domain_report + 验收断言逐条核对）。"""
    # 账本收益键已统一为 "gain"（SPEC §5 裁决回写）；显式传参仅为可读性。
    report = domain_report(ledger.records(), graph, gain_key="gain")
    _write_json(out / "domain_report.json", report)

    final = sim_result.final_report
    acceptance = {
        "goal": final.get("goal", ""),
        "passed": final["acceptance_passed"]["passed"],
        "total": final["acceptance_passed"]["total"],
        "all_passed": final["acceptance_passed"]["passed"]
        == final["acceptance_passed"]["total"],
        "results": final["acceptance"],
    }
    _write_json(out / "acceptance_report.json", acceptance)
    print(f"[聚合] domain_report 覆盖 {len(report)} 个领域；"
          f"验收断言 {acceptance['passed']}/{acceptance['total']} 通过")
    return {"domain_report": report, "acceptance_report": acceptance}


# --------------------------------------------------------------------------
# 终报（stdout：分支与理由 / 假说裁决 / 验收 / 领域建议）
# --------------------------------------------------------------------------


def _print_final_report(sim_result, aggregates: dict) -> None:
    events = sim_result.events
    print("\n===== 决策日志（分支与理由，溯源 trace.jsonl 的 decision 事件） =====")
    for e in events:
        if e["event"] == "decision":
            d = e["detail"]
            print(f"  tick={e['tick']} {e['node_id']}: 命中分支[{d['branch_index']}]"
                  f"「{d['condition']}」→ {d['action']}")
            print(f"      理由: {d['reason']}")
            if d.get("boost"):
                print(f"      boost: {d['boost']}")
            if d.get("skip"):
                print(f"      skip: {d['skip']}")
        elif e["event"] == "decision_deadend":
            print(f"  tick={e['tick']} {e['node_id']}: 死端（{e['detail'].get('reason')}）")

    print("\n===== 假说裁决（账本 hypotheses.jsonl） =====")
    for hyp, outcome in sim_result.hypothesis_outcomes.items():
        print(f"  {hyp}: {outcome['status']} — {outcome['notes']}")

    acc = aggregates["acceptance_report"]
    print(f"\n===== 验收断言核对（{acc['passed']}/{acc['total']}） =====")
    for item in acc["results"]:
        mark = "PASS" if item.get("passed") else "FAIL"
        actual = item.get("actual", item.get("error", "?"))
        print(f"  [{mark}] {item['assertion']}（实际: {actual}）")

    print("\n===== 领域聚合与下轮 prior 建议（domain_report.json） =====")
    for domain_id, bucket in aggregates["domain_report"].items():
        print(f"  {domain_id}（{bucket['title']}）: confirmed={bucket['confirmed']} "
              f"refuted={bucket['refuted']} uncertain={bucket['uncertain']} "
              f"blocked={bucket['blocked']} total_gain={bucket['total_gain']:+.2f} "
              f"→ prior {bucket['prior_suggestion']}")

    final = sim_result.final_report
    if final["unverified_hypotheses"]:
        print(f"\n未验证假说（不落账本）: {final['unverified_hypotheses']}")
    if final["unresolved_blockers"]:
        print(f"未决阻塞（交人裁决）: {final['unresolved_blockers']}")


def _write_decisions_log(sim_result, out: Path) -> None:
    """决策日志（SPEC §7 产物口径）：与 trace 的 decision 事件同源转写。"""
    lines: list[str] = []
    for e in sim_result.events:
        if e["event"] != "decision":
            continue
        d = e["detail"]
        lines.append(
            f"tick={e['tick']} {e['node_id']}: branch[{d['branch_index']}]"
            f" if={d['condition']} then={d['action']}"
            f" | boost={d.get('boost')} skip={d.get('skip')}"
            f" | 理由: {d['reason']}"
        )
    (out / DECISIONS_LOG).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """跑全链，返回聚合摘要；阶段失败抛 :class:`StageError`。"""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 阶段 1：编译（M4，含 M1 校验门）
    compiled = _stage_compile(args)
    (out / "graph.yaml").write_text(compiled.graph_yaml + "\n", encoding="utf-8")
    _write_json(out / "card_decisions.json", compiled.card_decisions)

    # 阶段 2：校验（M1，落盘图复查）
    _stage_validate(compiled.graph)

    # 阶段 3：模拟执行 + E8 账本回写（M2 × M3）
    sim_result, ledger = _stage_simulate(args, compiled.graph, out)
    _write_decisions_log(sim_result, out)

    # 阶段 4：聚合报告（M3 + 验收核对）
    aggregates = _stage_aggregate(compiled.graph, ledger, sim_result, out)
    _print_final_report(sim_result, aggregates)
    print(f"\n产物目录 {out}/ 已写齐: {', '.join(ARTIFACTS)} + {DECISIONS_LOG}")
    return {
        "out": str(out),
        "hypothesis_outcomes": sim_result.hypothesis_outcomes,
        "final_report": sim_result.final_report,
        **aggregates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_pipeline(args)
    except StageError as exc:
        print(f"流水线失败（阶段：{exc.stage}）\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — 兜底：任何未分类失败都要可读、非 0
        print(f"流水线失败（未分类异常）: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # 收口判定：验收断言核对完成且全过、无未决阻塞 → 0
    acc = summary["acceptance_report"]
    final = summary["final_report"]
    if not acc["all_passed"]:
        failed = [r["assertion"] for r in acc["results"] if not r.get("passed")]
        print(f"流水线完成但验收断言未全过（{acc['passed']}/{acc['total']}）: "
              f"{failed}", file=sys.stderr)
        return 1
    if final["unresolved_blockers"]:
        print(f"流水线完成但存在未决阻塞: {final['unresolved_blockers']}",
              file=sys.stderr)
        return 1
    print("端到端全链通过：编译 → 校验 → 模拟 → 回写 → 聚合，验收全过，退出码 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
