"""M6：无编排对照 —— 确定性启发式自由探索 agent。

语义来源：SPEC §8（baseline_agent.py：无编排对照——确定性启发式基线，
仅有任务书；按固定策略侦察-尝试-评估，无前置解锁/无决策点/无早停规则外的
停止）与 03 文档第四部分（对照设定：同一剧本下自由 ReAct 的 agent 仅有
任务书，唯一变量是编排）。

【工程默认值，列入交付报告「新发现」】SPEC §8 已标注：离线 L1 用启发式
基线替代 LLM ReAct。本 agent 是"自由 ReAct"的离线确定性替身，固定策略：

1. **开局侦察只看误差结构**（``INITIAL_PROBE``）——自由 agent 拿到任务书
   的典型第一反应；没有"先把未知显式化"的前置解锁纪律。
2. **固定优先级逐卡套用**（``CARDS``，顺序即优先级）——无决策点：唯一的
   路线选择是开局一次 ``decide``（固定选阈值卡，且不追问样本量/不升级
   验收协议），之后照单全收。
3. **无单原子约束**：改动只增不回滚，run 事件携带 ``applied_so_far``
   （≥2 时 ``attribution_ambiguous=True``，改动累积导致无法归因）。
4. **无早停规则外的停止**：验收全过即收工（S4 过早收工：满分不追问）；
   卡片用尽即止。
5. **正收益即战果**：任何 gain > 0 记 ``claim_win``——无边际阈值、无复跑、
   无 uncertain 判态（S5 把 +0.03/+0.01 噪声当战果）。
6. **先决策后补查**：前提未满足时 run 确定性失败（环境响应），然后才去
   补查对应探针（``run_fail`` → ``probe``），必要时返工重改
   （``repair: re_edit``）——信息缺失型事故的复现机制。

轨迹格式（与 M2 Trace 对称但独立，SPEC §8：无编排侧=baseline 自己的
轨迹记录）：每事件 ``{step, event, detail}``；``step`` 只对动作事件
（probe/edit/run/run_fail）递增，判定/结论事件附着于当前 step。
事件词表见 :data:`EVENT_VOCAB`。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..assertions import evaluate, parse_assertion

__all__ = [
    "CARDS",
    "EVENT_VOCAB",
    "INITIAL_PROBE",
    "BaselineAgent",
    "BaselineResult",
    "run_baseline",
]

#: 固定卡片库（套用顺序即优先级）。自由 agent 跨剧本携带同一套卡——
#: 是否适用由环境判定（fits_scenario），agent 不自知（盲抄的复现机制）。
CARDS: tuple[str, ...] = (
    "tune_threshold",   # 阈值搜索卡
    "sample_weights",   # 分层样本加权卡
    "prune_features",   # 特征剔除卡
    "fix_wiring",       # 接线修复卡
)

#: 开局固定侦察点（误差结构）
INITIAL_PROBE = "error_profile"

#: 轨迹事件词表
EVENT_VOCAB: tuple[str, ...] = (
    "probe",            # 侦察（含先决策后补查）
    "decide",           # 开局固定路线选择（无决策点：全程唯一一次）
    "edit",             # 套用卡片做改动（repair=True 表示补查后返工）
    "run",              # 运行评估（detail.gain 为实测收益）
    "run_fail",         # 前提未满足的确定性失败（环境响应）
    "claim_win",        # 正收益即战果（无边际阈值/无复跑）
    "no_gain",          # 收益 <= 0
    "challenge",        # 追问验证分数/升级验收协议（本基线策略永不触发；
                        #  作为 S4 陷阱识别的判定信号保留在词表中）
    "acceptance_check", # 验收断言核对（每卡之后 + 终局）
    "stop",             # 停止（acceptance_passed / cards_exhausted）
)

#: 动作事件（step 递增的单位）
_ACTION_EVENTS = frozenset({"probe", "edit", "run", "run_fail"})


@dataclass
class BaselineResult:
    """baseline 运行结果：轨迹 + 终局事实 + 验收核对。"""

    events: list[dict[str, Any]]
    facts: dict[str, Any]
    acceptance_passed: dict[str, int]  # {"passed": k, "total": n}
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": copy.deepcopy(self.events),
            "facts": copy.deepcopy(self.facts),
            "acceptance_passed": dict(self.acceptance_passed),
            "stop_reason": self.stop_reason,
        }


class BaselineAgent:
    """确定性启发式自由探索 agent（输入只有任务书 + 环境事实）。

    ``scenario`` 为 :class:`knowledge_pipeline.bench.scripts.Scenario`
    （鸭子类型：需要 ``brief`` / ``facts`` / ``baseline_env`` 三个属性）。
    同一输入重跑必然产生同一轨迹（无任何随机源）。
    """

    def __init__(self, scenario: Any):
        self.scenario = scenario

    def run(self) -> BaselineResult:
        env = self.scenario.baseline_env
        probes: dict[str, Any] = env.get("probes") or {}
        interventions: dict[str, Any] = env.get("interventions") or {}
        facts = copy.deepcopy(self.scenario.facts)
        acceptance = [parse_assertion(str(s)) for s in self.scenario.brief["acceptance"]]

        events: list[dict[str, Any]] = []
        step = 0
        probed: set[str] = set()
        applied: list[str] = []
        gains_used: dict[str, int] = {}

        def emit(event: str, **detail: Any) -> None:
            nonlocal step
            if event in _ACTION_EVENTS:
                step += 1
            if event not in EVENT_VOCAB:
                raise ValueError(f"非法 baseline 事件 {event!r}")
            events.append({"step": step, "event": event, "detail": detail})

        def do_probe(probe_id: str) -> None:
            cfg = probes[probe_id]
            emit(
                "probe",
                probe=probe_id,
                direction=cfg.get("direction"),
                reveals=cfg.get("reveals"),
            )
            probed.add(probe_id)

        def acceptance_status() -> list[str]:
            passed: list[str] = []
            for assertion in acceptance:
                try:
                    if evaluate(assertion, facts):
                        passed.append(str(assertion))
                except (KeyError, TypeError):
                    continue
            return passed

        def acc_check(final: bool = False) -> list[str]:
            passed = acceptance_status()
            emit(
                "acceptance_check",
                passed_count=len(passed),
                total=len(acceptance),
                final=final,
            )
            return passed

        def deep_merge(base: dict, update: dict) -> None:
            for key, value in update.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = copy.deepcopy(value)

        # ---- 1. 开局侦察：固定只看误差结构 -------------------------------
        do_probe(INITIAL_PROBE)
        # ---- 2. 固定路线选择（无决策点：不追问样本量、不升级验收协议） -----
        emit(
            "decide",
            choice=CARDS[0],
            protocol_upgrade=False,
            basis="任务书误差结构 → 阈值卡（固定优先级；不追问样本量/不升级协议）",
        )
        # ---- 3. 逐卡套用（改动累积、无早停规则外的停止） ------------------
        for card in CARDS:
            if len(acceptance_status()) == len(acceptance):
                break
            iv = interventions[card]
            applied.append(card)
            emit(
                "edit",
                card=card,
                direction=iv.get("direction"),
                repair=False,
                applied_so_far=list(applied),
            )
            guard = 0
            while True:
                guard += 1
                if guard > 50:  # 防死循环保险（探针有限，正常不会触发）
                    raise RuntimeError(f"baseline 卡片 {card} 前提循环异常")
                unmet = next(
                    (
                        pc
                        for pc in (iv.get("preconditions") or [])
                        if pc.get("probe") not in probed
                    ),
                    None,
                )
                if unmet is not None:
                    # 先决策后补查：改动已做、运行失败，才去补查
                    emit(
                        "run_fail",
                        card=card,
                        failure=unmet.get("failure"),
                        missing_probe=unmet.get("probe"),
                    )
                    do_probe(unmet["probe"])
                    if unmet.get("repair") == "re_edit":
                        emit(
                            "edit",
                            card=card,
                            direction=iv.get("direction"),
                            repair=True,
                            applied_so_far=list(applied),
                        )
                    continue
                idx = gains_used.get(card, 0)
                gains_used[card] = idx + 1
                gains = [float(g) for g in (iv.get("gains") or [0.0])]
                gain = gains[min(idx, len(gains) - 1)]
                facts_update = iv.get("facts_update")
                if isinstance(facts_update, dict):
                    deep_merge(facts, facts_update)
                run_detail: dict[str, Any] = {
                    "card": card,
                    "direction": iv.get("direction"),
                    "gain": gain,
                    "applied_so_far": list(applied),
                    "attribution_ambiguous": len(applied) > 1,
                }
                if iv.get("val_score") is not None:
                    run_detail["val_score"] = iv["val_score"]
                emit("run", **run_detail)
                if gain > 0:
                    # 正收益即战果：无边际阈值、无复跑、无 uncertain 判态
                    emit("claim_win", card=card, gain=gain)
                else:
                    emit("no_gain", card=card, gain=gain)
                break
            acc_check()

        # ---- 4. 终局 ------------------------------------------------------
        passed = acc_check(final=True)
        stop_reason = (
            "acceptance_passed" if len(passed) == len(acceptance) else "cards_exhausted"
        )
        emit("stop", reason=stop_reason)
        return BaselineResult(
            events=events,
            facts=facts,
            acceptance_passed={"passed": len(passed), "total": len(acceptance)},
            stop_reason=stop_reason,
        )


def run_baseline(scenario: Any) -> BaselineResult:
    """便捷入口：``BaselineAgent(scenario).run()``。"""
    return BaselineAgent(scenario).run()
