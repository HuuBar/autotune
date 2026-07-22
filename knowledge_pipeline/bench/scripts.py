"""M6：事故剧本库 S1~S6 —— 加载、解析与完整性校验。

语义来源：SPEC §8（bench/scripts.py 接口契约）、03 文档第四部分（L1 离线
dry-run A/B：每个剧本 = 触发条件 + 环境响应 + 期望应对；剧本对双方完全一致，
可重复性是本测试台的核心价值——它就是 harness 的回归测试套件）。

剧本数据存放于 ``knowledge_pipeline/fixtures/bench/*.yaml``，每剧本一份。
剧本 YAML 结构（本模块契约）：

.. code-block:: yaml

    id: S1                     # 必填；S1~S6
    title: str                 # 必填
    trigger: str               # 必填：触发条件
    env_response: str          # 必填：环境响应
    expected_response: str     # 必填：期望应对
    brief:                     # 任务书（双方唯一共同输入之一）
      statement: str
      acceptance: [str, ...]   # 必须能被 M1 assertions 解析；且与有编排图
                               # goal.acceptance 一致（剧本事实一致性守卫）
    facts: {...}               # 初始环境事实（双方唯一事实源；注入 playbook.facts
                               # 与 baseline 初始 facts，保证双方面对同一世界）
    graph: "../tbox_golden.yaml"   # 有编排图：相对本文件的路径，或内联 mapping
    playbook_extends: "../tbox_playbook.yaml"  # 可选：playbook 基座（深合并）
    playbook: {...}            # 基座之上的覆盖（深合并）；facts 由本剧本 facts 合入
    expected_decisions: {J1: 0, J2: 1}  # 指标 4 判据：各决策点的正确分支索引
                                        # （必须覆盖图内全部 decision）
    expected_first_choice: {choice: tune_threshold, protocol_upgrade: false}
                               # 指标 4 baseline 判据：开局固定路线中的正确选择
    marginal_threshold: 0.05   # 指标 6 噪声判阈（缺省 0.05，对齐 playbook
                               # outcome_rules.marginal_gain_threshold）
    trap:                      # 可选；存在时计算指标 5（S4 陷阱识别）
      orchestrated_signals: [{event: decision, node_id: J2, detail: {branch_index: 1}}]
      baseline_signals: [{event: challenge}]
    inapplicable_cards: []     # 指标 7 有编排侧判据：与本剧本不符的知识卡 id
    baseline_env:              # 无编排环境（确定性响应表；与 playbook 同源事实）
      probes:
        <probe_id>: {direction: str, reveals: str}
      interventions:
        <card_id>:             # 必须覆盖 baseline_agent.CARDS 全部卡片
          direction: str
          needs_probes: [probe_id, ...]   # 信息依赖（指标 2 判据）
          preconditions: [{probe: id, failure: str, repair: re_edit?}]
          gains: [float, ...]             # 第 i 次成功 run 的收益（用尽取末项）
          facts_update: {...}             # 成功 run 后深合并入 facts
          val_score: float                # 可选：run 事件上报的验证集分数（S4 满分）
          fits_scenario: bool             # 指标 7 判据（环境事实，agent 不可见）
          fit_reason: str

工程默认值（SPEC §8 语义空白处，列入交付报告「新发现」）：

- 剧本的有编排图允许多剧本共享（S1~S5 复用黄金夹具图 + 各剧本 playbook
  覆盖；S6 内联专用图）——SPEC 只要求"该剧本的有编排图/playbook"，未要求
  每剧本独占一份图。
- ``playbook_extends`` 深合并：dict 递归合并，其余类型（list/标量）整体替换。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..assertions import AssertionParseError, parse_assertion
from ..schema import iter_decisions, iter_nodes, load_graph
from ..validator import validate_graph
from .baseline_agent import CARDS

__all__ = [
    "SCENARIO_IDS",
    "REQUIRED_FIELDS",
    "ScriptError",
    "Scenario",
    "default_scripts_dir",
    "load_scenario",
    "load_scripts",
]

#: 剧本库固定成员（03 文档第四部分：S1 断链｜S2 重复列名｜S3 脆弱 JSON｜
#: S4 小样本阈值陷阱｜S5 边际提升噪声｜S6 沙箱依赖缺失）
SCENARIO_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

#: SPEC §8 + 任务书要求的每剧本必填字段
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "trigger",
    "env_response",
    "expected_response",
)


class ScriptError(ValueError):
    """剧本文件结构/一致性错误（剧本库是回归测试套件，装载即全量校验）。"""


def default_scripts_dir() -> Path:
    """剧本库缺省目录：``knowledge_pipeline/fixtures/bench/``。"""
    return Path(__file__).resolve().parents[1] / "fixtures" / "bench"


def _deep_merge(base: Any, override: Any) -> Any:
    """深合并 override 到 base 的副本（dict 递归，其余类型整体替换）。"""
    if isinstance(base, dict) and isinstance(override, dict):
        out = copy.deepcopy(base)
        for key, value in override.items():
            out[key] = _deep_merge(out[key], value) if key in out else copy.deepcopy(value)
        return out
    return copy.deepcopy(override)


@dataclass(frozen=True)
class Scenario:
    """一个事故剧本（对双方完全一致的事实包 + 双侧执行素材）。

    - 有编排侧：``graph`` + ``playbook``（直接喂给 M2 Simulator/MockExecutor）；
    - 无编排侧：``brief`` + ``facts`` + ``baseline_env``（喂给 BaselineAgent）；
    - 判据：``expected_decisions`` / ``expected_first_choice`` / ``trap`` /
      ``inapplicable_cards`` / ``marginal_threshold``（指标 4/5/6/7）。
    """

    id: str
    title: str
    trigger: str
    env_response: str
    expected_response: str
    brief: dict[str, Any]
    facts: dict[str, Any]
    graph: dict[str, Any]
    playbook: dict[str, Any]
    baseline_env: dict[str, Any]
    expected_decisions: dict[str, int] = field(default_factory=dict)
    expected_first_choice: dict[str, Any] | None = None
    marginal_threshold: float = 0.05
    trap: dict[str, Any] | None = None
    inapplicable_cards: tuple[str, ...] = ()
    path: str = ""


def _err(path: Path, msg: str) -> ScriptError:
    return ScriptError(f"剧本 {path.name}: {msg}")


def _resolve_yaml(base_dir: Path, ref: str) -> dict[str, Any]:
    p = (base_dir / ref).resolve()
    if not p.is_file():
        raise ScriptError(f"引用文件不存在: {ref}（解析为 {p}）")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScriptError(f"YAML 解析失败: {p}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ScriptError(f"YAML 顶层必须是 mapping: {p}")
    return doc


def load_scenario(path: str | Path) -> Scenario:
    """加载并全量校验一个剧本文件；任何不一致抛 :class:`ScriptError`。"""
    p = Path(path)
    if not p.is_file():
        raise ScriptError(f"剧本文件不存在: {p}")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScriptError(f"剧本 YAML 解析失败: {p}: {exc}") from exc
    if not isinstance(doc, dict):
        raise _err(p, "顶层必须是 mapping")

    # ---- 必填字段（SPEC §8：{id, title, trigger, env_response, expected_response}）
    for key in REQUIRED_FIELDS:
        if not isinstance(doc.get(key), str) or not str(doc.get(key)).strip():
            raise _err(p, f"必填字段 {key!r} 缺失或为空")

    base_dir = p.parent

    # ---- 有编排图（路径引用或内联 mapping），必须过 M1 校验器
    graph_ref = doc.get("graph")
    if isinstance(graph_ref, str):
        graph = load_graph((base_dir / graph_ref).resolve())
    elif isinstance(graph_ref, dict):
        graph = copy.deepcopy(graph_ref)
    else:
        raise _err(p, "graph 必须是相对路径或内联 mapping")
    graph_errors = validate_graph(graph)
    if graph_errors:
        summary = "; ".join(str(e) for e in graph_errors[:5])
        raise _err(p, f"有编排图未通过 M1 校验（前 5 条）: {summary}")

    # ---- 任务书：acceptance 可解析 且 与图 goal.acceptance 一致（双方同事实守卫）
    brief = doc.get("brief")
    if not isinstance(brief, dict) or not isinstance(brief.get("acceptance"), list):
        raise _err(p, "brief.acceptance 必须是非空列表")
    for s in brief["acceptance"]:
        try:
            parse_assertion(str(s))
        except AssertionParseError as exc:
            raise _err(p, f"brief.acceptance 断言不可解析: {s!r}: {exc}") from exc
    goal_acceptance = [str(s) for s in (graph.get("goal", {}).get("acceptance") or [])]
    if [str(s) for s in brief["acceptance"]] != goal_acceptance:
        raise _err(
            p,
            "brief.acceptance 必须与有编排图 goal.acceptance 完全一致"
            f"（剧本对双方同事实）；brief={brief['acceptance']} graph={goal_acceptance}",
        )

    # ---- playbook：extends 基座 + 覆盖深合并 + facts 注入（双方唯一事实源）
    playbook: dict[str, Any] = {}
    extends = doc.get("playbook_extends")
    if extends is not None:
        playbook = _resolve_yaml(base_dir, str(extends))
    override = doc.get("playbook") or {}
    if not isinstance(override, dict):
        raise _err(p, "playbook 必须是 mapping")
    playbook = _deep_merge(playbook, override)
    facts = doc.get("facts") or {}
    if not isinstance(facts, dict):
        raise _err(p, "facts 必须是 mapping")
    playbook["facts"] = _deep_merge(playbook.get("facts") or {}, facts)
    facts = copy.deepcopy(playbook["facts"])  # 合入后的最终事实为双方共用

    # playbook 必须覆盖图内全部 action / decision（装载期拦截，防运行期缺配）
    action_ids = {n.id for n in iter_nodes(graph) if n.layer == "action"}
    decision_ids = {d.id for d in iter_decisions(graph)}
    pb_actions = set((playbook.get("actions") or {}).keys())
    pb_decisions = set((playbook.get("decisions") or {}).keys())
    missing_actions = sorted(action_ids - pb_actions)
    missing_decisions = sorted(decision_ids - pb_decisions)
    if missing_actions:
        raise _err(p, f"playbook.actions 缺少动作配置: {missing_actions}")
    if missing_decisions:
        raise _err(p, f"playbook.decisions 缺少决策配置: {missing_decisions}")

    # ---- 指标判据
    expected_decisions = doc.get("expected_decisions") or {}
    if set(expected_decisions) != decision_ids:
        raise _err(
            p,
            "expected_decisions 必须覆盖图内全部 decision "
            f"（图内 {sorted(decision_ids)}，剧本给出 {sorted(expected_decisions)}）",
        )
    expected_first_choice = doc.get("expected_first_choice")
    if expected_first_choice is not None and not isinstance(expected_first_choice, dict):
        raise _err(p, "expected_first_choice 必须是 mapping")
    marginal_threshold = float(doc.get("marginal_threshold", 0.05))
    trap = doc.get("trap")
    if trap is not None:
        if not isinstance(trap, dict) or "orchestrated_signals" not in trap or "baseline_signals" not in trap:
            raise _err(p, "trap 必须含 orchestrated_signals 与 baseline_signals")
    inapplicable_cards = tuple(str(c) for c in (doc.get("inapplicable_cards") or []))

    # ---- 无编排环境：干预必须覆盖全部固定卡片，前提引用的探针必须已声明
    baseline_env = doc.get("baseline_env")
    if not isinstance(baseline_env, dict):
        raise _err(p, "baseline_env 缺失")
    probes = baseline_env.get("probes") or {}
    interventions = baseline_env.get("interventions") or {}
    if not probes or not isinstance(probes, dict):
        raise _err(p, "baseline_env.probes 必须是非空 mapping")
    missing_cards = [c for c in CARDS if c not in interventions]
    if missing_cards:
        raise _err(p, f"baseline_env.interventions 缺少固定卡片配置: {missing_cards}")
    for card_id, iv in interventions.items():
        if not isinstance(iv, dict):
            raise _err(p, f"interventions.{card_id} 必须是 mapping")
        if "fits_scenario" not in iv:
            raise _err(p, f"interventions.{card_id} 缺 fits_scenario（指标 7 判据）")
        for probe_id in iv.get("needs_probes") or []:
            if probe_id not in probes:
                raise _err(p, f"interventions.{card_id}.needs_probes 引用未声明探针 {probe_id!r}")
        for pc in iv.get("preconditions") or []:
            if pc.get("probe") not in probes:
                raise _err(p, f"interventions.{card_id}.preconditions 引用未声明探针 {pc.get('probe')!r}")
            if "failure" not in pc:
                raise _err(p, f"interventions.{card_id}.preconditions 缺 failure 描述")

    return Scenario(
        id=str(doc["id"]),
        title=str(doc["title"]),
        trigger=str(doc["trigger"]),
        env_response=str(doc["env_response"]),
        expected_response=str(doc["expected_response"]),
        brief=dict(brief),
        facts=facts,
        graph=graph,
        playbook=playbook,
        baseline_env={
            "probes": copy.deepcopy(probes),
            "interventions": copy.deepcopy(interventions),
        },
        expected_decisions={str(k): int(v) for k, v in expected_decisions.items()},
        expected_first_choice=copy.deepcopy(expected_first_choice),
        marginal_threshold=marginal_threshold,
        trap=copy.deepcopy(trap),
        inapplicable_cards=inapplicable_cards,
        path=str(p),
    )


def load_scripts(scripts_dir: str | Path | None = None) -> dict[str, Scenario]:
    """加载整个剧本库（缺省 ``fixtures/bench/``），返回 ``{id: Scenario}``。

    完整性守卫（03 文档第四部分"剧本对双方完全一致、可重复"）：
    剧本集合必须恰好是 S1~S6 各一份，否则抛 :class:`ScriptError`。
    """
    d = Path(scripts_dir) if scripts_dir is not None else default_scripts_dir()
    if not d.is_dir():
        raise ScriptError(f"剧本库目录不存在: {d}")
    scenarios: dict[str, Scenario] = {}
    for path in sorted(d.glob("*.yaml")):
        sc = load_scenario(path)
        if sc.id in scenarios:
            raise ScriptError(f"剧本 id 重复: {sc.id}（{path.name}）")
        scenarios[sc.id] = sc
    missing = [sid for sid in SCENARIO_IDS if sid not in scenarios]
    extra = sorted(sid for sid in scenarios if sid not in SCENARIO_IDS)
    if missing or extra:
        raise ScriptError(f"剧本库必须恰好含 S1~S6；缺失 {missing}，多出 {extra}")
    return scenarios
