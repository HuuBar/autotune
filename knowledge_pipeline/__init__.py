"""knowledge_pipeline：知识工程流水线前端（执行前编排 → 执行 → 执行后回写）。

M1 模块：编排图 schema + 确定性校验器 + 验收断言解析器。
接口契约见 docs/knowledge-pipeline/SPEC.md §0~§3。
"""

from .assertions import Assertion, AssertionParseError, evaluate, parse_assertion
from .schema import (
    ACTION_TYPES,
    DIRECTED_EDGE_TYPES,
    EDGE_TYPES,
    LAYERS,
    SCHEMA_VERSION,
    TOP_LEVEL_KEYS,
    Decision,
    Edge,
    GraphLoadError,
    Node,
    iter_decisions,
    iter_edges,
    iter_nodes,
    load_graph,
    node_index,
    to_decision,
    to_edge,
    to_node,
)
from .validator import ValidationError, assert_valid, validate_file, validate_graph

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # schema
    "SCHEMA_VERSION",
    "TOP_LEVEL_KEYS",
    "LAYERS",
    "EDGE_TYPES",
    "DIRECTED_EDGE_TYPES",
    "ACTION_TYPES",
    "Node",
    "Edge",
    "Decision",
    "GraphLoadError",
    "load_graph",
    "to_node",
    "to_edge",
    "to_decision",
    "iter_nodes",
    "iter_edges",
    "iter_decisions",
    "node_index",
    # assertions
    "Assertion",
    "AssertionParseError",
    "parse_assertion",
    "evaluate",
    # validator
    "ValidationError",
    "validate_graph",
    "validate_file",
    "assert_valid",
]
