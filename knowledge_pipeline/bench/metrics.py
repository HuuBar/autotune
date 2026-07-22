"""M6：预登记指标计算（SPEC §8 七项；03 文档第四部分指标表与归因原则）。

输入：双方执行轨迹（有编排侧 = M2 Trace 事件流；无编排侧 = baseline 轨迹）。
输出：每剧本 × 双方 × 七项指标。

七项预登记指标（语义来源：03 文档第四部分预登记指标表）：

1. ``steps_to_acceptance`` 到达验收总步数
   —— 首次验收全过之前的动作执行步数（有编排侧 = execute 事件数，
   无编排侧 = probe/edit/run/run_fail 事件数）；未到达为 ``None``。
2. ``info_gap_incidents`` 信息缺失型事故数（先决策后补查）
   —— 动作首做时其信息依赖（有编排侧 = requires/informs 源动作；
   无编排侧 = needs_probes ∪ preconditions 探针）尚未就位的 (动作, 依赖) 对数。
3. ``direction_coverage`` 探索方向覆盖数
   —— 实际触达的正交方向数（有编排侧 = 已执行动作所属 domain；
   无编排侧 = probe/edit/run 事件的 direction）。
4. ``decision_branch_accuracy`` 决策点分支正确率
   —— 有编排侧 = decision 事件分支索引与剧本 expected_decisions 的一致率；
   无编排侧 = decide 事件与 expected_first_choice 的一致率（含是否升级协议）。
5. ``s4_trap_identified`` S4 陷阱识别（阈值满分被追问、验收协议升级）
   —— 剧本声明 trap 信号时计算：全部信号命中 → 1.0，否则 0.0；
   剧本无 trap 节 → ``None``（不参与聚合）。
6. ``s5_noise_as_win`` 噪声当战果次数
   —— 有编排侧 = 账本写回中 status==confirmed 且 0 < measured_gain < 边际阈值
   的记录数；无编排侧 = 0 < gain < 边际阈值的 claim_win 事件数。
7. ``blind_card_applications`` 盲抄次数（与剧本不符的卡片套用）
   —— 有编排侧 = 图 meta.source_cards ∩ 剧本 inapplicable_cards；
   无编排侧 = 首次套用 fits_scenario=false 卡片的次数。

归因原则（03 文档第四部分）：过程指标为主判据；每剧本独立计分再聚合
（:func:`aggregate` 防单剧本运气）；剧本对双方完全一致（:func:`canonical_events`
提供可重复的轨迹比较口径）。

工程默认值（列入交付报告「新发现」）：

- 步数口径用"动作执行次数"而非 tick——编排侧重试/复跑发生在同一 tick 内，
  用 tick 会系统性偏袒有编排侧。
- ``canonical_events`` 递归剔除 ``ts`` 键（账本写回记录含时间戳），使
  "同输入两跑结果一致"成为可判定的回归断言。
"""

from __future__ import annotations

import copy
from typing import Any

from ..schema import iter_edges, iter_nodes

__all__ = [
    "METRIC_NAMES",
    "aggregate",
    "canonical_events",
    "compute_metrics",
]

METRIC_NAMES: tuple[str, ...] = (
    "steps_to_acceptance",
    "info_gap_incidents",
    "direction_coverage",
    "decision_branch_accuracy",
    "s4_trap_identified",
    "s5_noise_as_win",
    "blind_card_applications",
)

#: 聚合口径：计数类求和，比率/步数类求均值（None 不参与）
_SUM_METRICS = frozenset(
    {"info_gap_incidents", "s5_noise_as_win", "blind_card_applications"}
)

_BASELINE_ACTION_EVENTS = frozenset({"probe", "edit", "run", "run_fail"})


def canonical_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """轨迹规范化：递归剔除 ``ts`` 键（账本时间戳），返回深拷贝。

    用于"同输入两跑结果一致"的确定性回归断言（03 文档：可重复性是核心
    价值）。
    """

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k != "ts"}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    return _strip(copy.deepcopy(events))


# ----------------------------------------------------------------------
# 共用片段
# ----------------------------------------------------------------------


def _first_full_pass(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """首个"验收全过"的 acceptance_check 事件。"""
    for e in events:
        if e.get("event") != "acceptance_check":
            continue
        detail = e.get("detail") or {}
        total = detail.get("total", 0)
        if total and detail.get("passed_count") == total:
            return e
    return None


def _signal_match(signal: dict[str, Any], event: dict[str, Any]) -> bool:
    """trap 信号匹配：event 名一致 + node_id（可选）+ detail 子集。"""
    if event.get("event") != signal.get("event"):
        return False
    if signal.get("node_id") is not None and event.get("node_id") != signal["node_id"]:
        return False
    detail = event.get("detail") or {}
    return all(detail.get(k) == v for k, v in (signal.get("detail") or {}).items())


def _trap_score(events: list[dict[str, Any]], signals: list[dict[str, Any]]) -> float:
    matched = all(any(_signal_match(sig, e) for e in events) for sig in signals)
    return 1.0 if matched else 0.0


# ----------------------------------------------------------------------
# 有编排侧（M2 Trace 事件流）
# ----------------------------------------------------------------------


def _orchestrated_metrics(events: list[dict[str, Any]], scenario: Any) -> dict[str, Any]:
    graph = scenario.graph

    # ---- 图索引：动作 → (requires/informs 源动作, 所属 domain)
    action_owner: dict[str, str] = {}
    hyp_domain: dict[str, str] = {}
    for node in iter_nodes(graph):
        if node.layer == "action":
            action_owner[node.id] = str(node.get("belongs_to", ""))
        elif node.layer == "hypothesis":
            hyp_domain[node.id] = str(node.get("domain", ""))
    deps: dict[str, set[str]] = {}
    for edge in iter_edges(graph):
        if (
            edge.type in ("requires", "informs")
            and edge.to_id in action_owner
            and edge.from_id in action_owner
        ):
            deps.setdefault(edge.to_id, set()).add(edge.from_id)

    executes = [e for e in events if e.get("event") == "execute"]

    # 1. 到达验收总步数（execute 事件数口径）
    acc = _first_full_pass(events)
    steps: int | None = None
    if acc is not None:
        steps = sum(1 for e in executes if e.get("tick", 0) <= acc.get("tick", 0))

    # 2. 信息缺失型事故（首做时 requires/informs 源动作未 confirmed）
    done: set[str] = set()
    seen: set[str] = set()
    info_gaps = 0
    for e in executes:
        nid = e.get("node_id")
        if nid not in seen:
            seen.add(nid)
            info_gaps += sum(1 for s in deps.get(nid, ()) if s not in done)
        if (e.get("detail") or {}).get("verify_passed"):
            done.add(nid)

    # 3. 探索方向覆盖数（已执行动作所属 domain）
    directions = {
        hyp_domain.get(action_owner.get(nid, ""), action_owner.get(nid, ""))
        for nid in seen
    }
    directions.discard("")

    # 4. 决策点分支正确率
    dec_events = [e for e in events if e.get("event") == "decision"]
    branch_acc: float | None = None
    if dec_events:
        correct = sum(
            1
            for e in dec_events
            if scenario.expected_decisions.get(str(e.get("node_id")))
            == (e.get("detail") or {}).get("branch_index")
        )
        branch_acc = correct / len(dec_events)

    # 5. S4 陷阱识别
    trap: float | None = None
    if scenario.trap:
        trap = _trap_score(events, scenario.trap.get("orchestrated_signals") or [])

    # 6. 噪声当战果（confirmed 但实测收益落在边际区间）
    threshold = float(scenario.marginal_threshold)
    noise = 0
    for e in events:
        if e.get("event") != "writeback":
            continue
        record = (e.get("detail") or {}).get("record") or {}
        if record.get("status") != "confirmed":
            continue
        gain = (record.get("metrics") or {}).get("measured_gain")
        if isinstance(gain, (int, float)) and not isinstance(gain, bool) and 0 < gain < threshold:
            noise += 1

    # 7. 盲抄（图 source_cards 与剧本不适用卡的交集）
    source_cards = set((graph.get("meta") or {}).get("source_cards") or [])
    blind = len(source_cards & set(scenario.inapplicable_cards))

    return {
        "steps_to_acceptance": steps,
        "info_gap_incidents": info_gaps,
        "direction_coverage": len(directions),
        "decision_branch_accuracy": branch_acc,
        "s4_trap_identified": trap,
        "s5_noise_as_win": noise,
        "blind_card_applications": blind,
    }


# ----------------------------------------------------------------------
# 无编排侧（baseline 轨迹）
# ----------------------------------------------------------------------


def _baseline_metrics(events: list[dict[str, Any]], scenario: Any) -> dict[str, Any]:
    interventions = (scenario.baseline_env or {}).get("interventions") or {}

    # 1. 到达验收总步数（动作事件数口径）
    acc = _first_full_pass(events)
    steps: int | None = None
    if acc is not None:
        acc_step = acc.get("step", 0)
        steps = sum(
            1
            for e in events
            if e.get("event") in _BASELINE_ACTION_EVENTS and e.get("step", 0) <= acc_step
        )

    # 首做编辑步 / 探针步
    first_edit: dict[str, int] = {}
    probe_step: dict[str, int] = {}
    directions: set[str] = set()
    for e in events:
        step = e.get("step", 0)
        detail = e.get("detail") or {}
        if e.get("event") == "edit":
            first_edit.setdefault(str(detail.get("card")), step)
            if detail.get("direction"):
                directions.add(str(detail["direction"]))
        elif e.get("event") == "probe":
            probe_step.setdefault(str(detail.get("probe")), step)
            if detail.get("direction"):
                directions.add(str(detail["direction"]))
        elif e.get("event") == "run" and detail.get("direction"):
            directions.add(str(detail["direction"]))

    # 2. 信息缺失型事故（先决策后补查：探针在首次编辑之后或从未做）
    info_gaps = 0
    for card, edit_step in first_edit.items():
        iv = interventions.get(card) or {}
        required = set(iv.get("needs_probes") or [])
        required |= {pc.get("probe") for pc in (iv.get("preconditions") or [])}
        for probe_id in required:
            pstep = probe_step.get(str(probe_id))
            if pstep is None or pstep > edit_step:
                info_gaps += 1

    # 4. 决策点分支正确率（开局固定路线 vs 剧本期望）
    decides = [e for e in events if e.get("event") == "decide"]
    branch_acc: float | None = None
    expected = scenario.expected_first_choice
    if decides and expected:
        correct = sum(
            1
            for e in decides
            if (e.get("detail") or {}).get("choice") == expected.get("choice")
            and bool((e.get("detail") or {}).get("protocol_upgrade"))
            == bool(expected.get("protocol_upgrade"))
        )
        branch_acc = correct / len(decides)

    # 5. S4 陷阱识别（baseline 策略无追问机制 → 信号不命中即 0.0）
    trap: float | None = None
    if scenario.trap:
        trap = _trap_score(events, scenario.trap.get("baseline_signals") or [])

    # 6. 噪声当战果（正收益即战果中的边际收益）
    threshold = float(scenario.marginal_threshold)
    noise = sum(
        1
        for e in events
        if e.get("event") == "claim_win"
        and 0 < float((e.get("detail") or {}).get("gain", 0.0)) < threshold
    )

    # 7. 盲抄（首次套用 fits_scenario=false 的卡片）
    blind = sum(
        1
        for card in first_edit
        if (interventions.get(card) or {}).get("fits_scenario") is False
    )

    return {
        "steps_to_acceptance": steps,
        "info_gap_incidents": info_gaps,
        "direction_coverage": len(directions),
        "decision_branch_accuracy": branch_acc,
        "s4_trap_identified": trap,
        "s5_noise_as_win": noise,
        "blind_card_applications": blind,
    }


def compute_metrics(
    events: list[dict[str, Any]], side: str, scenario: Any
) -> dict[str, Any]:
    """计算一侧轨迹的七项预登记指标。

    ``side`` ∈ ``{"orchestrated", "baseline"}``；``scenario`` 为
    :class:`knowledge_pipeline.bench.scripts.Scenario`（鸭子类型）。
    """
    if side == "orchestrated":
        return _orchestrated_metrics(events, scenario)
    if side == "baseline":
        return _baseline_metrics(events, scenario)
    raise ValueError(f"未知侧别 {side!r}（允许: orchestrated / baseline）")


def aggregate(per_scenario: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """跨剧本聚合（03 文档：每剧本独立计分再聚合，防单剧本运气）。

    ``per_scenario`` = ``{剧本 id: {侧别: 指标 dict}}``；输出
    ``{指标: {侧别: {"value": 聚合值, "n": 计入剧本数, "values": {剧本: 值}}}}``。
    计数类求和，其余求均值；``None``（不适用/未到达）不参与。
    """
    sides = ("orchestrated", "baseline")
    out: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        out[metric] = {}
        for side in sides:
            values = {
                sid: sides_metrics[side][metric]
                for sid, sides_metrics in per_scenario.items()
                if sides_metrics.get(side, {}).get(metric) is not None
            }
            if not values:
                agg: float | None = None
            elif metric in _SUM_METRICS:
                agg = sum(values.values())
            else:
                agg = sum(values.values()) / len(values)
            out[metric][side] = {"value": agg, "n": len(values), "values": values}
    return out
