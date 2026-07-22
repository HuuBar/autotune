"""Mock 执行器：按 playbook 返回预制产物，支持可配置成功/失败序列与决策分支切换。

语义来源：SPEC §4.1（MockExecutor 接口契约）、01 文档 §7（L1 离线级 dry-run）、
03 文档第二部分（mock 环境内置事故剧本，逐拍时间线）。

Playbook schema（本模块自定义，YAML 文档，黄金实例见
``knowledge_pipeline/fixtures/tbox_playbook.yaml``）：

.. code-block:: yaml

    meta: {id: str, description: str}          # 剧本标识（可选）
    facts: {...}                               # 全局环境事实（验收断言评估的基线；
                                               #  支持嵌套 dict，点号路径取值）
    actions:                                   # 每个 action 节点一节
      <action_id>:
        artifacts: [...]                       # 候选产物序列：每次"成功"执行消费一个，
                                               #  用尽后取最后一项（SPEC §4.1）。
                                               #  产物内可带 ``facts_update`` 键：
                                               #  成功时深合并入全局 facts（模拟环境
                                               #  因该动作而演化），该键不进产物本体。
        fail_sequence: [bool, ...]             # 每次尝试是否 verify 失败（True=失败），
                                               #  序列用尽后取最后一项（SPEC §4.1）。
    decisions:                                 # 每个 decision 一节（E3 的机器判定依据；
                                               #  模拟器自身不做自然语言理解）
      <decision_id>:
        rules:                                 # 有序规则，首条命中生效
          - when: [{path, op, value}, ...]     # 全部条件成立（条件见下）
            branch: int                        # decision.branches 的分支索引
            reason: str                        # 判定理由（强制落盘；支持 {点号路径} 插值）
        default: {branch, reason}              # 可选；缺省时无命中 → 返回 -1
        effects:                               # 分支效果（SPEC §4.1 未覆盖的工程默认值，
          "0": {boost: [node_id...]}           #  见 branch_effects；then 为自然语言，
          "1": {skip: [node_id...]}            #  分支→节点的机器映射由 playbook 供给）
    outcome_rules:                             # 假说终态判定的 mock 口径（SPEC §4.2
      marginal_gain_threshold: 0.05            #  假说级状态聚合 + 03 文档"边际提升复跑"）
      marginal_reruns: 1

条件（``when`` 条目）：``path`` 为点号路径，在「决策 after 动作产物 ∪ 全局 facts」
（产物优先）上取值；``op`` ∈ ``==, !=, >=, <=, >, <, in, contains, exists,
not_exists``；路径缺失时除 ``not_exists`` 外一律判不成立。
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..schema import Decision, Node

__all__ = [
    "ExecutionResult",
    "MockExecutor",
    "PlaybookError",
    "load_playbook",
]

#: 条件操作符
_COND_OPS = ("==", "!=", ">=", "<=", ">", "<", "in", "contains", "exists", "not_exists")

#: 产物内保留键：成功时深合并入全局 facts，不进入产物本体
_FACTS_UPDATE_KEY = "facts_update"

#: 理由插值：{点号路径}
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


class PlaybookError(ValueError):
    """playbook 结构错误（如缺少某 action / decision 的配置）。"""


@dataclass(frozen=True)
class ExecutionResult:
    """单次执行结果（SPEC §4.1：``{ok, artifact, verify_passed, log}``）。"""

    ok: bool
    artifact: dict[str, Any] | None
    verify_passed: bool
    log: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "artifact": copy.deepcopy(self.artifact),
            "verify_passed": self.verify_passed,
            "log": self.log,
        }


def load_playbook(path: str | Path) -> dict[str, Any]:
    """从 YAML 文件加载 playbook 为 dict。失败抛 :class:`PlaybookError`。"""
    p = Path(path)
    if not p.is_file():
        raise PlaybookError(f"playbook 文件不存在: {p}")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlaybookError(f"playbook YAML 解析失败: {p}: {exc}") from exc
    if not isinstance(doc, dict):
        raise PlaybookError(f"playbook 顶层必须是映射(mapping): {p}")
    return doc


def _lookup(ctx: Mapping[str, Any], path: str) -> Any:
    """点号路径取值（嵌套映射逐段下钻，优先精确匹配扁平键）。缺失抛 KeyError。"""
    if path in ctx:
        return ctx[path]
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """深合并 update 到 base（就地修改 base 并返回）。"""
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _match_condition(cond: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    """求值单条 ``{path, op, value}`` 条件（语义见模块 docstring）。"""
    path = str(cond.get("path", ""))
    op = str(cond.get("op", "=="))
    if op not in _COND_OPS:
        raise PlaybookError(f"非法条件操作符 {op!r}（允许: {_COND_OPS}）")
    try:
        actual = _lookup(ctx, path)
    except KeyError:
        return op == "not_exists"
    if op == "exists":
        return True
    if op == "not_exists":
        return False
    expected = cond.get("value")
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "contains":
        return expected in actual
    # 比较类操作符要求数值可比
    try:
        if op == ">=":
            return actual >= expected
        if op == "<=":
            return actual <= expected
        if op == ">":
            return actual > expected
        if op == "<":
            return actual < expected
    except TypeError:
        return False
    raise PlaybookError(f"未覆盖的条件操作符 {op!r}")  # pragma: no cover


def _format_reason(template: str, ctx: Mapping[str, Any]) -> str:
    """理由插值：``{点号路径}`` 替换为上下文取值；缺失路径保留原样。"""

    def _sub(m: re.Match[str]) -> str:
        try:
            return str(_lookup(ctx, m.group(1)))
        except KeyError:
            return m.group(0)

    return _PLACEHOLDER_RE.sub(_sub, template)


@dataclass
class MockExecutor:
    """按 playbook 演出的 mock 执行器（SPEC §4.1）。

    - 同一 action 多次尝试按 ``fail_sequence`` 依次消费（索引=第几次尝试），
      用尽后取最后一项；成功执行按 ``artifacts`` 依次消费（索引=第几次成功），
      用尽后取最后一项。
    - ``received_params``：测试钩子，记录每次执行收到的 params 快照
      （E6 参数回写的可观测点）。
    """

    playbook: dict[str, Any]
    received_params: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.playbook, dict):
            raise PlaybookError("playbook 必须是 dict（可用 load_playbook 加载）")
        self._facts: dict[str, Any] = copy.deepcopy(self.playbook.get("facts") or {})
        self._attempts: dict[str, int] = defaultdict(int)
        self._successes: dict[str, int] = defaultdict(int)
        rules = self.playbook.get("outcome_rules") or {}
        # 假说终态判定口径（SPEC §4.2 假说级状态聚合的工程默认值）
        self.outcome_rules: dict[str, Any] = {
            "marginal_gain_threshold": 0.05,
            "marginal_reruns": 1,
            **rules,
        }

    # -- 环境事实 ----------------------------------------------------------

    @property
    def facts(self) -> dict[str, Any]:
        """当前环境事实（全局 facts 深合并各成功产物的 facts_update 后的快照）。"""
        return copy.deepcopy(self._facts)

    # -- SPEC §4.1 接口 -----------------------------------------------------

    def execute(self, action: Node, params: dict) -> ExecutionResult:
        """执行一个 action 节点的一次尝试。

        ``fail_sequence`` 判定本次 verify 是否失败；成功时按成功次数从
        ``artifacts`` 取预制产物，并把产物内的 ``facts_update`` 深合并入环境事实。
        """
        cfg = (self.playbook.get("actions") or {}).get(action.id)
        if cfg is None:
            raise PlaybookError(f"playbook.actions 缺少动作 {action.id!r} 的配置")
        self.received_params[action.id].append(copy.deepcopy(dict(params or {})))

        attempt = self._attempts[action.id]
        self._attempts[action.id] += 1
        fail_seq = list(cfg.get("fail_sequence") or [])
        failed = bool(fail_seq[min(attempt, len(fail_seq) - 1)]) if fail_seq else False
        if failed:
            return ExecutionResult(
                ok=False,
                artifact=None,
                verify_passed=False,
                log=(
                    f"{action.id} 第 {attempt + 1} 次尝试 verify 失败"
                    f"（playbook fail_sequence 第 {min(attempt, len(fail_seq) - 1) + 1} 项）"
                ),
            )

        arts = list(cfg.get("artifacts") or [{}])
        success_idx = self._successes[action.id]
        self._successes[action.id] += 1
        artifact = copy.deepcopy(arts[min(success_idx, len(arts) - 1)])
        facts_update = artifact.pop(_FACTS_UPDATE_KEY, None)
        if isinstance(facts_update, Mapping):
            _deep_merge(self._facts, facts_update)
        return ExecutionResult(
            ok=True,
            artifact=artifact,
            verify_passed=True,
            log=(
                f"{action.id} 第 {attempt + 1} 次尝试通过，产出 "
                f"{action.get('produces', '?')}（候选产物第 "
                f"{min(success_idx, len(arts) - 1) + 1}/{len(arts)} 项）"
            ),
        )

    def resolve_branch(self, decision: Decision, artifacts: dict) -> tuple[int, str]:
        """按 playbook 决策规则选择分支，返回 ``(分支索引, 理由)``。

        判定上下文 = after 动作产物 ∪ 全局 facts（产物优先）。规则按序匹配，
        首条命中生效；无命中时取 ``default``；二者皆无返回 ``(-1, 理由)``，
        由调度器按 E3 走图内 else 兜底或记 ``decision_deadend``。
        理由强制落盘（支持 ``{点号路径}`` 插值，把实测值写进决策日志）。
        """
        cfg = (self.playbook.get("decisions") or {}).get(decision.id)
        if cfg is None:
            raise PlaybookError(f"playbook.decisions 缺少决策 {decision.id!r} 的配置")
        artifact = artifacts.get(decision.after) or {}
        ctx = copy.deepcopy(self._facts)
        _deep_merge(ctx, artifact)

        for rule in cfg.get("rules") or []:
            when = rule.get("when") or []
            if all(_match_condition(c, ctx) for c in when):
                return int(rule["branch"]), _format_reason(str(rule.get("reason", "")), ctx)
        default = cfg.get("default")
        if default is not None:
            return int(default["branch"]), _format_reason(
                str(default.get("reason", "")), ctx
            )
        return -1, "无规则命中且 playbook 未配置 default"

    def branch_effects(self, decision_id: str, branch_index: int) -> dict[str, list[str]]:
        """分支效果：``{"boost": [...], "skip": [...]}``（SPEC §4.1 之外的扩展）。

        语义空白说明（列入交付报告「新发现」）：图 schema 中分支 ``then`` 是自然
        语言描述，E3 要求"命中分支的 then 动作提为最高优先"，但 then→节点 的机器
        映射没有 schema 承载。工程默认值：由 playbook 的 ``decisions.<id>.effects``
        按分支索引显式给出；调度器在本方法缺失时退化为 then/else 文本中的已知
        action id 扫描。
        """
        cfg = (self.playbook.get("decisions") or {}).get(decision_id) or {}
        effects = (cfg.get("effects") or {}).get(str(branch_index)) or {}
        return {
            "boost": list(effects.get("boost") or []),
            "skip": list(effects.get("skip") or []),
        }
