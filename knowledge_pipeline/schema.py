"""编排图 schema：YAML 加载、结构常量与轻量节点/边/决策封装。

语义来源：SPEC §1（编排图 schema）与 01 文档 §2（四层结构 L0~L3、边类型）。
L0 目标层以顶层 ``goal`` 字段表达，不占 nodes（SPEC §1 末段）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

SCHEMA_VERSION = "1.0"

# ---- 结构常量（SPEC §1） -------------------------------------------------

#: 顶层键（V-SCHEMA 必查齐全）
TOP_LEVEL_KEYS: tuple[str, ...] = (
    "version",
    "meta",
    "goal",
    "nodes",
    "edges",
    "decisions",
    "writeback",
)

#: 节点层（01 文档 §2.1：L1 领域 / L2 假说 / L3 动作；L0 为顶层 goal）
LAYERS: tuple[str, ...] = ("domain", "hypothesis", "action")

#: 边类型（01 文档 §2.2）
EDGE_TYPES: tuple[str, ...] = ("requires", "informs", "alternatives", "no_parallel")

#: 参与有向无环检测的边类型；no_parallel 是无向资源约束，不参与（SPEC §3 V-DAG）
DIRECTED_EDGE_TYPES: tuple[str, ...] = ("requires", "informs", "alternatives")

#: 动作类型（SPEC §1 nodes[].type）
ACTION_TYPES: tuple[str, ...] = ("recon", "edit", "run")

#: 各层节点除公共字段（id/layer/title）外的必填字段
NODE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "domain": ("rationale",),
    "hypothesis": ("domain", "mechanism", "prior", "expected", "cost"),
    "action": (
        "belongs_to",
        "type",
        "do",
        "tool_hint",
        "produces",
        "verify",
        "retry_budget",
    ),
}

#: 其他结构必填字段
META_REQUIRED_FIELDS: tuple[str, ...] = ("task", "created_by", "source_cards")
GOAL_REQUIRED_FIELDS: tuple[str, ...] = ("statement", "acceptance")
EDGE_REQUIRED_FIELDS: tuple[str, ...] = ("from", "to", "type")
DECISION_REQUIRED_FIELDS: tuple[str, ...] = ("id", "after", "read", "branches")
WRITEBACK_REQUIRED_FIELDS: tuple[str, ...] = (
    "ledger",
    "record_schema",
    "after_domain_review",
)
HYPOTHESIS_EXPECTED_FIELDS: tuple[str, ...] = ("metric", "delta", "confidence")
HYPOTHESIS_COST_FIELDS: tuple[str, ...] = ("recon_steps", "run_minutes")


# ---- 轻量封装 -------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """编排图节点。``data`` 保留原始字段字典，便于 M2/M3 透传未知字段。"""

    id: str
    layer: str
    title: str
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


@dataclass(frozen=True)
class Edge:
    """编排图边。``from_id``/``to_id`` 为节点 id 或 decision id（SPEC §1）。"""

    from_id: str
    to_id: str
    type: str
    reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """决策点：``after`` 动作完成后读 ``read`` 产物，按 ``branches`` 选择分支。

    分支项结构为 ``{"if": str, "then": str}`` 或 ``{"else": str, "then": str}``
    （SPEC §1；01 文档 §2.4）。
    """

    id: str
    after: str
    read: str
    branches: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


# ---- 加载与规范化 ---------------------------------------------------------


class GraphLoadError(ValueError):
    """图文档加载失败（文件缺失 / YAML 语法错误 / 顶层不是映射）。"""


def load_graph(path: str | Path) -> dict[str, Any]:
    """从 YAML 文件加载编排图文档为 dict。失败抛 :class:`GraphLoadError`。"""
    p = Path(path)
    if not p.is_file():
        raise GraphLoadError(f"图文件不存在: {p}")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GraphLoadError(f"YAML 解析失败: {p}: {exc}") from exc
    if not isinstance(doc, dict):
        raise GraphLoadError(f"图文档顶层必须是映射(mapping): {p}")
    return doc


def to_node(raw: dict[str, Any]) -> Node:
    """原始节点 dict → :class:`Node`。调用前应先过校验器。"""
    return Node(
        id=str(raw.get("id", "")),
        layer=str(raw.get("layer", "")),
        title=str(raw.get("title", "")),
        data=dict(raw),
    )


def to_edge(raw: dict[str, Any]) -> Edge:
    """原始边 dict → :class:`Edge`。"""
    return Edge(
        from_id=str(raw.get("from", "")),
        to_id=str(raw.get("to", "")),
        type=str(raw.get("type", "")),
        reason=raw.get("reason"),
        data=dict(raw),
    )


def to_decision(raw: dict[str, Any]) -> Decision:
    """原始决策 dict → :class:`Decision`。"""
    branches = raw.get("branches") or []
    return Decision(
        id=str(raw.get("id", "")),
        after=str(raw.get("after", "")),
        read=str(raw.get("read", "")),
        branches=list(branches),
        data=dict(raw),
    )


def iter_nodes(doc: dict[str, Any]) -> Iterator[Node]:
    """遍历文档中的节点（跳过非映射项，便于校验器容错复用）。"""
    for raw in doc.get("nodes") or []:
        if isinstance(raw, dict):
            yield to_node(raw)


def iter_edges(doc: dict[str, Any]) -> Iterator[Edge]:
    for raw in doc.get("edges") or []:
        if isinstance(raw, dict):
            yield to_edge(raw)


def iter_decisions(doc: dict[str, Any]) -> Iterator[Decision]:
    for raw in doc.get("decisions") or []:
        if isinstance(raw, dict):
            yield to_decision(raw)


def node_index(doc: dict[str, Any]) -> dict[str, Node]:
    """节点 id → Node 索引（重复 id 时保留首个，交由校验器报告）。"""
    index: dict[str, Node] = {}
    for node in iter_nodes(doc):
        index.setdefault(node.id, node)
    return index
