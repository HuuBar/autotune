"""编排图确定性校验器。

语义来源：SPEC §3（V-* 检查清单）与 01 文档 §3（编译器硬性校验职责）。
接口契约：``validate_graph(doc) -> list[ValidationError]``，全部通过返回空列表。

实现判定说明（SPEC 语义空白处的工程默认值，详见交付报告「新发现」）：

* V-REQUIRES-EXISTS：SPEC 原文“requires 不得指向 decision”，但黄金夹具存在
  ``requires: A3 -> J2``（决策点以产物为前置输入）。按黄金夹具裁决：允许
  requires **指向** decision（决策点等待输入产物），禁止 requires **来自**
  decision（decision 由 ``after`` 触发，不能作为前置产物源）。
* V-ALT-GROUP：alternatives 组 = 无向连通分量。单边残缺 = 自环（from==to，
  组内节点 <2）或端点未声明；同组节点须解析到同一 domain（belongs_to 为
  domain 时取其自身，为 hypothesis 时取其 domain）；组内互依赖按 requires
  可达性（传递闭包）判定，而非仅直接边。
* V-DECISION：分支完备性按结构判定——if 分支须有非空 if 与 then；else 兜底
  分支的 ``else`` 值本身即分支动作描述，不再要求 then（黄金夹具 J1 为此形态，
  SPEC §3 原文“每个分支有 (if|else) 与 then”按黄金夹具裁决放宽）；且
  （存在 else 兜底 或 分支数 >= 2）。语义穷尽性交由编译器自检与人工评审。
* V-SCHEMA：SPEC §1 将 ``title`` 列为节点公共字段，但黄金夹具的 action 节点
  普遍省略 title。按黄金夹具裁决：title 不作硬性必填，仅在出现时校验为字符串。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .assertions import AssertionParseError, parse_assertion
from .schema import (
    ACTION_TYPES,
    DECISION_REQUIRED_FIELDS,
    DIRECTED_EDGE_TYPES,
    EDGE_REQUIRED_FIELDS,
    EDGE_TYPES,
    GOAL_REQUIRED_FIELDS,
    HYPOTHESIS_COST_FIELDS,
    HYPOTHESIS_EXPECTED_FIELDS,
    LAYERS,
    META_REQUIRED_FIELDS,
    NODE_REQUIRED_FIELDS,
    TOP_LEVEL_KEYS,
    WRITEBACK_REQUIRED_FIELDS,
    GraphLoadError,
    load_graph,
)

__all__ = ["ValidationError", "validate_graph", "validate_file"]


@dataclass(frozen=True)
class ValidationError:
    """单条校验错误。``code`` 为 V-* 检查码，``path`` 定位文档内位置。"""

    code: str
    message: str
    path: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.path}: {self.message}"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _non_empty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


# --------------------------------------------------------------------------
# 各检查项（均为纯函数，输入 doc 与公共索引，输出错误列表）
# --------------------------------------------------------------------------


def _raw_items(doc: Mapping[str, Any], key: str) -> list[Any]:
    items = doc.get(key)
    return items if isinstance(items, list) else []


def _check_schema(doc: Mapping[str, Any]) -> list[ValidationError]:
    """V-SCHEMA：顶层键齐全、类型正确；节点/边/决策必填字段齐全。"""
    errs: list[ValidationError] = []

    def err(msg: str, path: str) -> None:
        errs.append(ValidationError("V-SCHEMA", msg, path))

    for key in TOP_LEVEL_KEYS:
        if key not in doc:
            err(f"缺少顶层键 {key!r}", key)

    version = doc.get("version")
    if "version" in doc and not isinstance(version, str):
        err(f"version 必须是字符串，得到 {type(version).__name__}", "version")

    meta = doc.get("meta")
    if "meta" in doc:
        if not isinstance(meta, Mapping):
            err("meta 必须是映射", "meta")
        else:
            for f in META_REQUIRED_FIELDS:
                if f not in meta:
                    err(f"meta 缺少必填字段 {f!r}", f"meta.{f}")
            if "source_cards" in meta and not isinstance(meta["source_cards"], list):
                err("meta.source_cards 必须是列表", "meta.source_cards")

    goal = doc.get("goal")
    if "goal" in doc:
        if not isinstance(goal, Mapping):
            err("goal 必须是映射", "goal")
        else:
            for f in GOAL_REQUIRED_FIELDS:
                if f not in goal:
                    err(f"goal 缺少必填字段 {f!r}", f"goal.{f}")
            acc = goal.get("acceptance")
            if "acceptance" in goal:
                if not isinstance(acc, list) or not all(isinstance(a, str) for a in acc):
                    err("goal.acceptance 必须是字符串列表", "goal.acceptance")

    for key in ("nodes", "edges", "decisions"):
        if key in doc and not isinstance(doc[key], list):
            err(f"{key} 必须是列表", key)

    for i, raw in enumerate(_raw_items(doc, "nodes")):
        path = f"nodes[{i}]"
        if not isinstance(raw, Mapping):
            err("节点必须是映射", path)
            continue
        nid = raw.get("id")
        if _non_empty_str(nid):
            path = f"nodes[{nid}]"
        else:
            err("节点缺少非空字符串 id", path)
        layer = raw.get("layer")
        if layer not in LAYERS:
            err(f"节点 {nid!r} layer 必须是 {LAYERS} 之一，得到 {layer!r}", f"{path}.layer")
            continue  # 层未知则无法判定该层必填字段
        # title 在黄金夹具的 action 节点上普遍缺省：不作硬性必填，仅在出现时校验类型
        if "title" in raw and not isinstance(raw.get("title"), str):
            err(f"节点 {nid!r} title 必须是字符串", f"{path}.title")
        for f in NODE_REQUIRED_FIELDS[layer]:
            if f not in raw or raw.get(f) is None:
                err(f"{layer} 节点 {nid!r} 缺少必填字段 {f!r}", f"{path}.{f}")
        if layer == "hypothesis":
            expected = raw.get("expected")
            if isinstance(expected, Mapping):
                for f in HYPOTHESIS_EXPECTED_FIELDS:
                    if f not in expected:
                        err(f"hypothesis {nid!r} expected 缺少 {f!r}", f"{path}.expected.{f}")
            elif "expected" in raw:
                err(f"hypothesis {nid!r} expected 必须是映射", f"{path}.expected")
            cost = raw.get("cost")
            if isinstance(cost, Mapping):
                for f in HYPOTHESIS_COST_FIELDS:
                    if f not in cost:
                        err(f"hypothesis {nid!r} cost 缺少 {f!r}", f"{path}.cost.{f}")
                if "recon_steps" in cost and not isinstance(cost["recon_steps"], int):
                    err(f"hypothesis {nid!r} cost.recon_steps 必须是整数", f"{path}.cost.recon_steps")
                if "run_minutes" in cost and not _is_number(cost["run_minutes"]):
                    err(f"hypothesis {nid!r} cost.run_minutes 必须是数值", f"{path}.cost.run_minutes")
            elif "cost" in raw:
                err(f"hypothesis {nid!r} cost 必须是映射", f"{path}.cost")
        if layer == "action":
            atype = raw.get("type")
            if "type" in raw and atype not in ACTION_TYPES:
                err(f"action {nid!r} type 必须是 {ACTION_TYPES} 之一，得到 {atype!r}", f"{path}.type")
            rb = raw.get("retry_budget")
            if "retry_budget" in raw and (
                not isinstance(rb, int) or isinstance(rb, bool) or rb < 0
            ):
                err(f"action {nid!r} retry_budget 必须是 >=0 的整数，得到 {rb!r}", f"{path}.retry_budget")

    for i, raw in enumerate(_raw_items(doc, "edges")):
        path = f"edges[{i}]"
        if not isinstance(raw, Mapping):
            err("边必须是映射", path)
            continue
        for f in EDGE_REQUIRED_FIELDS:
            if f not in raw or raw.get(f) is None:
                err(f"边缺少必填字段 {f!r}", f"{path}.{f}")
        etype = raw.get("type")
        if "type" in raw and etype not in EDGE_TYPES:
            err(f"边 type 必须是 {EDGE_TYPES} 之一，得到 {etype!r}", f"{path}.type")

    for i, raw in enumerate(_raw_items(doc, "decisions")):
        path = f"decisions[{i}]"
        if not isinstance(raw, Mapping):
            err("决策必须是映射", path)
            continue
        did = raw.get("id")
        if _non_empty_str(did):
            path = f"decisions[{did}]"
        for f in DECISION_REQUIRED_FIELDS:
            if f not in raw or raw.get(f) is None:
                err(f"决策缺少必填字段 {f!r}", f"{path}.{f}")
        if "branches" in raw and not isinstance(raw["branches"], list):
            err("决策 branches 必须是列表", f"{path}.branches")

    wb = doc.get("writeback")
    if "writeback" in doc:
        if not isinstance(wb, Mapping):
            err("writeback 必须是映射", "writeback")
        else:
            for f in WRITEBACK_REQUIRED_FIELDS:
                if f not in wb:
                    err(f"writeback 缺少必填字段 {f!r}", f"writeback.{f}")
    return errs


def _index_nodes(doc: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[ValidationError]]:
    """节点索引 + V-UNIQ（节点 id 唯一）。"""
    errs: list[ValidationError] = []
    nodes: dict[str, Mapping[str, Any]] = {}
    for i, raw in enumerate(_raw_items(doc, "nodes")):
        if not isinstance(raw, Mapping):
            continue
        nid = raw.get("id")
        if not _non_empty_str(nid):
            continue
        if nid in nodes:
            errs.append(
                ValidationError("V-UNIQ", f"节点 id {nid!r} 重复定义", f"nodes[{nid}]")
            )
        else:
            nodes[nid] = raw
    return nodes, errs


def _index_decisions(
    doc: Mapping[str, Any], nodes: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], list[ValidationError]]:
    """决策索引 + V-UNIQ（decision id 唯一且不与节点 id 冲突）。"""
    errs: list[ValidationError] = []
    decisions: dict[str, Mapping[str, Any]] = {}
    for raw in _raw_items(doc, "decisions"):
        if not isinstance(raw, Mapping):
            continue
        did = raw.get("id")
        if not _non_empty_str(did):
            continue
        if did in nodes:
            errs.append(
                ValidationError(
                    "V-UNIQ", f"decision id {did!r} 与节点 id 冲突", f"decisions[{did}]"
                )
            )
        elif did in decisions:
            errs.append(
                ValidationError("V-UNIQ", f"decision id {did!r} 重复定义", f"decisions[{did}]")
            )
        else:
            decisions[did] = raw
    return decisions, errs


def _check_dag(
    doc: Mapping[str, Any],
    nodes: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> list[ValidationError]:
    """V-DAG：requires/informs/alternatives 构成的有向图无环。

    no_parallel 是无向资源约束，不参与（SPEC §3）。DFS 三色标记找环。
    """
    adjacency: dict[str, list[tuple[str, str]]] = {}
    known = set(nodes) | set(decisions)
    for i, raw in enumerate(_raw_items(doc, "edges")):
        if not isinstance(raw, Mapping):
            continue
        etype, frm, to = raw.get("type"), raw.get("from"), raw.get("to")
        if etype not in DIRECTED_EDGE_TYPES:
            continue
        if not (_non_empty_str(frm) and _non_empty_str(to)):
            continue
        if frm not in known or to not in known:
            continue  # 悬空端点由 V-REQUIRES-EXISTS / V-ALT-GROUP 报告
        adjacency.setdefault(frm, []).append((to, etype))

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in known}
    cycle: list[str] | None = None

    def dfs(v: str, stack: list[str]) -> None:
        nonlocal cycle
        color[v] = GRAY
        stack.append(v)
        for nxt, _ in adjacency.get(v, []):
            if cycle is not None:
                return
            if color.get(nxt) == GRAY:
                start = stack.index(nxt)
                cycle = stack[start:] + [nxt]
                return
            if color.get(nxt) == WHITE:
                dfs(nxt, stack)
        stack.pop()
        color[v] = BLACK

    for v in sorted(known):
        if cycle is not None:
            break
        if color[v] == WHITE:
            dfs(v, [])
    if cycle is None:
        return []
    return [
        ValidationError(
            "V-DAG",
            "编排图存在有向环（requires/informs/alternatives）: " + " -> ".join(cycle),
            "edges",
        )
    ]


def _check_requires_exists(
    doc: Mapping[str, Any],
    nodes: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> list[ValidationError]:
    """V-REQUIRES-EXISTS：requires/informs 端点指向已声明节点/decision。

    工程默认值（见模块 docstring）：requires 允许指向 decision（等待输入产物），
    禁止 requires 来自 decision（decision 由 after 触发，不是产物源）。
    """
    errs: list[ValidationError] = []
    known = set(nodes) | set(decisions)
    for i, raw in enumerate(_raw_items(doc, "edges")):
        if not isinstance(raw, Mapping):
            continue
        etype, frm, to = raw.get("type"), raw.get("from"), raw.get("to")
        if etype not in ("requires", "informs"):
            continue
        path = f"edges[{i}]({frm}->{to}, {etype})"
        for label, endpoint in (("from", frm), ("to", to)):
            if _non_empty_str(endpoint) and endpoint not in known:
                errs.append(
                    ValidationError(
                        "V-REQUIRES-EXISTS",
                        f"{etype} 边的 {label} 端 {endpoint!r} 未声明（既非节点也非 decision）",
                        path,
                    )
                )
        if etype == "requires" and frm in decisions:
            errs.append(
                ValidationError(
                    "V-REQUIRES-EXISTS",
                    f"requires 边不得来自 decision {frm!r}（decision 由 after 触发，不能作为前置产物源）",
                    path,
                )
            )
        if etype == "requires" and frm == to and _non_empty_str(frm):
            errs.append(
                ValidationError("V-REQUIRES-EXISTS", f"requires 边自环 {frm!r}", path)
            )
    return errs


def _check_action_verify(nodes: Mapping[str, Mapping[str, Any]]) -> list[ValidationError]:
    """V-ACTION-VERIFY：每个 action 有非空 verify 且有 retry_budget。"""
    errs: list[ValidationError] = []
    for nid, raw in nodes.items():
        if raw.get("layer") != "action":
            continue
        if not _non_empty_str(raw.get("verify")):
            errs.append(
                ValidationError(
                    "V-ACTION-VERIFY",
                    f"action {nid!r} 缺少非空 verify（成功判据）",
                    f"nodes[{nid}].verify",
                )
            )
        if "retry_budget" not in raw or raw.get("retry_budget") is None:
            errs.append(
                ValidationError(
                    "V-ACTION-VERIFY",
                    f"action {nid!r} 缺少 retry_budget（重试预算）",
                    f"nodes[{nid}].retry_budget",
                )
            )
    return errs


def _alt_groups(
    doc: Mapping[str, Any], known: set[str]
) -> tuple[list[set[str]], list[ValidationError]]:
    """alternatives 无向连通分量分组 + 单边残缺检查。"""
    errs: list[ValidationError] = []
    adjacency: dict[str, set[str]] = {}
    for i, raw in enumerate(_raw_items(doc, "edges")):
        if not isinstance(raw, Mapping) or raw.get("type") != "alternatives":
            continue
        frm, to = raw.get("from"), raw.get("to")
        path = f"edges[{i}]({frm}->{to}, alternatives)"
        if not (_non_empty_str(frm) and _non_empty_str(to)):
            errs.append(
                ValidationError("V-ALT-GROUP", "alternatives 边缺少 from/to 端点", path)
            )
            continue
        if frm not in known or to not in known:
            missing = frm if frm not in known else to
            errs.append(
                ValidationError(
                    "V-ALT-GROUP",
                    f"alternatives 组残缺（单边）：端点 {missing!r} 未声明，备选关系涉及 <2 个有效节点",
                    path,
                )
            )
            continue
        if frm == to:
            errs.append(
                ValidationError(
                    "V-ALT-GROUP",
                    f"alternatives 组残缺（单边）：节点 {frm!r} 与自身互斥，备选组需 >=2 个节点",
                    path,
                )
            )
            continue
        adjacency.setdefault(frm, set()).add(to)
        adjacency.setdefault(to, set()).add(frm)

    groups: list[set[str]] = []
    seen: set[str] = set()
    for start in adjacency:
        if start in seen:
            continue
        group: set[str] = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v in group:
                continue
            group.add(v)
            stack.extend(adjacency[v] - group)
        seen |= group
        groups.append(group)
    return groups, errs


def _requires_reachability(
    doc: Mapping[str, Any], known: set[str]
) -> dict[str, set[str]]:
    """requires 边的传递闭包（用于组内互依赖判定）。"""
    adjacency: dict[str, set[str]] = {}
    for raw in _raw_items(doc, "edges"):
        if not isinstance(raw, Mapping) or raw.get("type") != "requires":
            continue
        frm, to = raw.get("from"), raw.get("to")
        if frm in known and to in known:
            adjacency.setdefault(frm, set()).add(to)
    reach: dict[str, set[str]] = {v: set() for v in known}

    def visit(start: str, cur: str) -> None:
        for nxt in adjacency.get(cur, ()):
            if nxt not in reach[start]:
                reach[start].add(nxt)
                visit(start, nxt)

    for v in known:
        visit(v, v)
    return reach


def _check_alt_group(
    doc: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Any],
) -> list[ValidationError]:
    """V-ALT-GROUP：alternatives 组完整（SPEC §3 第 6 条）。

    - 同一 alternatives 关系涉及 >=2 个节点（单边/自环即残缺）；
    - 同组节点解析到同一 domain（同一 hypothesis 下游分支或同一 domain）；
    - 组内节点不得有 requires 相互依赖（按可达性，含传递依赖）。
    """
    known = set(nodes) | set(decisions)
    groups, errs = _alt_groups(doc, known)
    if not groups:
        return errs
    reach = _requires_reachability(doc, known)

    def domain_of(nid: str) -> str | None:
        raw = nodes.get(nid)
        if raw is None:
            return None
        layer = raw.get("layer")
        if layer == "domain":
            return nid
        if layer == "hypothesis":
            return raw.get("domain") if isinstance(raw.get("domain"), str) else None
        if layer == "action":
            parent = raw.get("belongs_to")
            if not isinstance(parent, str):
                return None
            if parent in nodes and nodes[parent].get("layer") == "domain":
                return parent
            if parent in nodes and nodes[parent].get("layer") == "hypothesis":
                d = nodes[parent].get("domain")
                return d if isinstance(d, str) else None
        return None

    for group in groups:
        members = sorted(group)
        domains = {m: domain_of(m) for m in members}
        known_domains = {d for d in domains.values() if d is not None}
        if len(known_domains) > 1:
            detail = ", ".join(f"{m}(domain={domains[m]})" for m in members)
            errs.append(
                ValidationError(
                    "V-ALT-GROUP",
                    f"alternatives 组成员须属于同一 hypothesis 下游分支或同一 domain，实际跨域: {detail}",
                    "edges(alternatives)",
                )
            )
        for a in members:
            for b in members:
                if a != b and b in reach.get(a, ()):
                    errs.append(
                        ValidationError(
                            "V-ALT-GROUP",
                            f"alternatives 组内存在 requires 依赖 {a!r} -> {b!r}（互斥分支不能互为前置）",
                            "edges(alternatives)",
                        )
                    )
    return errs


def _check_acceptance(doc: Mapping[str, Any]) -> list[ValidationError]:
    """V-ACCEPTANCE：goal.acceptance 每条可被 §2 解析。"""
    errs: list[ValidationError] = []
    goal = doc.get("goal")
    if not isinstance(goal, Mapping):
        return errs
    acc = goal.get("acceptance")
    if not isinstance(acc, list):
        return errs
    for i, item in enumerate(acc):
        if not isinstance(item, str):
            continue  # 类型错误已由 V-SCHEMA 报告
        try:
            parse_assertion(item)
        except AssertionParseError as exc:
            errs.append(
                ValidationError("V-ACCEPTANCE", f"验收断言不可解析: {exc}", f"goal.acceptance[{i}]")
            )
    return errs


def _check_decisions(
    doc: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[ValidationError]:
    """V-DECISION：after 指向存在 action、read 等于其 produces、分支结构完备。

    完备性工程默认值（SPEC §3 第 8 条）：每个分支有 (if|else) 与 then，且
    （存在 else 兜底 或 分支数 >= 2）。
    """
    errs: list[ValidationError] = []
    for did, raw in decisions.items():
        path = f"decisions[{did}]"
        after = raw.get("after")
        after_node: Mapping[str, Any] | None = None
        if not _non_empty_str(after) or after not in nodes:
            errs.append(
                ValidationError(
                    "V-DECISION", f"decision {did!r} 的 after 指向不存在的节点 {after!r}", f"{path}.after"
                )
            )
        else:
            after_node = nodes[after]
            if after_node.get("layer") != "action":
                errs.append(
                    ValidationError(
                        "V-DECISION",
                        f"decision {did!r} 的 after 必须指向 action 节点，实际 {after!r} 为 {after_node.get('layer')!r}",
                        f"{path}.after",
                    )
                )
        read = raw.get("read")
        if after_node is not None and after_node.get("layer") == "action":
            produces = after_node.get("produces")
            if _non_empty_str(read) and read != produces:
                errs.append(
                    ValidationError(
                        "V-DECISION",
                        f"decision {did!r} 的 read={read!r} 与 after 动作 {after!r} 的 produces={produces!r} 不符",
                        f"{path}.read",
                    )
                )
        branches = raw.get("branches")
        if not isinstance(branches, list):
            continue  # 类型错误已由 V-SCHEMA 报告
        has_else = False
        for j, br in enumerate(branches):
            bpath = f"{path}.branches[{j}]"
            if not isinstance(br, Mapping):
                errs.append(ValidationError("V-DECISION", f"decision {did!r} 分支必须是映射", bpath))
                continue
            has_cond = _non_empty_str(br.get("if"))
            has_else_branch = _non_empty_str(br.get("else"))
            if not (has_cond or has_else_branch):
                errs.append(
                    ValidationError(
                        "V-DECISION", f"decision {did!r} 分支缺少非空 if/else 条件", bpath
                    )
                )
            if has_else_branch:
                has_else = True
            # if 分支必须有 then；else 分支的 else 值本身即动作描述（黄金夹具 J1 形态）
            if has_cond and not _non_empty_str(br.get("then")):
                errs.append(
                    ValidationError(
                        "V-DECISION", f"decision {did!r} 的 if 分支缺少非空 then", bpath
                    )
                )
        if branches and not has_else and len(branches) < 2:
            errs.append(
                ValidationError(
                    "V-DECISION",
                    f"decision {did!r} 分支不完备：仅 {len(branches)} 个 if 分支且无 else 兜底"
                    "（要求：存在 else 或分支数 >= 2）",
                    f"{path}.branches",
                )
            )
    return errs


def _check_layer_ref(nodes: Mapping[str, Mapping[str, Any]]) -> list[ValidationError]:
    """V-LAYER-REF：hypothesis.domain 指向 domain 层节点；
    action.belongs_to 指向存在的 hypothesis 或 domain 节点。"""
    errs: list[ValidationError] = []
    for nid, raw in nodes.items():
        layer = raw.get("layer")
        if layer == "hypothesis":
            domain = raw.get("domain")
            if _non_empty_str(domain):
                target = nodes.get(domain)
                if target is None:
                    errs.append(
                        ValidationError(
                            "V-LAYER-REF",
                            f"hypothesis {nid!r} 的 domain 引用不存在的节点 {domain!r}",
                            f"nodes[{nid}].domain",
                        )
                    )
                elif target.get("layer") != "domain":
                    errs.append(
                        ValidationError(
                            "V-LAYER-REF",
                            f"hypothesis {nid!r} 的 domain 必须指向 domain 层节点，实际 {domain!r} 为 {target.get('layer')!r}",
                            f"nodes[{nid}].domain",
                        )
                    )
        elif layer == "action":
            parent = raw.get("belongs_to")
            if _non_empty_str(parent):
                target = nodes.get(parent)
                if target is None:
                    errs.append(
                        ValidationError(
                            "V-LAYER-REF",
                            f"action {nid!r} 的 belongs_to 引用不存在的节点 {parent!r}",
                            f"nodes[{nid}].belongs_to",
                        )
                    )
                elif target.get("layer") not in ("hypothesis", "domain"):
                    errs.append(
                        ValidationError(
                            "V-LAYER-REF",
                            f"action {nid!r} 的 belongs_to 必须指向 hypothesis 或 domain 节点，实际 {parent!r} 为 {target.get('layer')!r}",
                            f"nodes[{nid}].belongs_to",
                        )
                    )
    return errs


def _check_prior(nodes: Mapping[str, Mapping[str, Any]]) -> list[ValidationError]:
    """V-PRIOR：hypothesis.prior ∈ [0,1]。"""
    errs: list[ValidationError] = []
    for nid, raw in nodes.items():
        if raw.get("layer") != "hypothesis" or "prior" not in raw:
            continue
        prior = raw.get("prior")
        if not _is_number(prior) or not (0.0 <= float(prior) <= 1.0):
            errs.append(
                ValidationError(
                    "V-PRIOR",
                    f"hypothesis {nid!r} 的 prior 必须在 [0,1] 区间，得到 {prior!r}",
                    f"nodes[{nid}].prior",
                )
            )
    return errs


# --------------------------------------------------------------------------
# 公共接口
# --------------------------------------------------------------------------


def validate_graph(doc: Mapping[str, Any]) -> list[ValidationError]:
    """对编排图文档执行 SPEC §3 全部确定性检查。全部通过返回空列表。"""
    if not isinstance(doc, Mapping):
        return [
            ValidationError(
                "V-SCHEMA", f"图文档顶层必须是映射，得到 {type(doc).__name__}", "<root>"
            )
        ]

    errors: list[ValidationError] = []
    errors += _check_schema(doc)

    nodes, errs = _index_nodes(doc)
    errors += errs
    decisions, errs = _index_decisions(doc, nodes)
    errors += errs

    errors += _check_dag(doc, nodes, decisions)
    errors += _check_requires_exists(doc, nodes, decisions)
    errors += _check_action_verify(nodes)
    errors += _check_alt_group(doc, nodes, decisions)
    errors += _check_acceptance(doc)
    errors += _check_decisions(doc, nodes, decisions)
    errors += _check_layer_ref(nodes)
    errors += _check_prior(nodes)
    return errors


def validate_file(path: str | Path) -> list[ValidationError]:
    """加载 YAML 图文件并校验。加载失败以 V-SCHEMA 错误返回（不抛异常）。"""
    try:
        doc = load_graph(path)
    except GraphLoadError as exc:
        return [ValidationError("V-SCHEMA", str(exc), str(path))]
    return validate_graph(doc)


def assert_valid(doc: Mapping[str, Any]) -> None:
    """便捷断言：存在错误时抛 :class:`ValueError`（汇总全部错误）。"""
    errors = validate_graph(doc)
    if errors:
        raise ValueError(
            "编排图校验失败:\n" + "\n".join(f"  - {e}" for e in errors)
        )
