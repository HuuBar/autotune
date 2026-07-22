"""M4 编排编译器（选卡论证 → 成图 → 确定性校验门）。

接口契约：docs/knowledge-pipeline/SPEC.md §6。
"""

from .compiler import (
    CompileError,
    CompileResult,
    compile_graph,
    load_card_ids,
    self_check_card_decisions,
    self_check_graph,
)
from .llm_client import (
    CachedClient,
    FakeClient,
    LLMClient,
    LLMResponseError,
    OpenAICompatClient,
)
from .prompts import (
    CARD_SELECTION_SYSTEM,
    GRAPH_SYSTEM,
    card_selection_user,
    graph_user,
)

__all__ = [
    "compile_graph",
    "CompileResult",
    "CompileError",
    "load_card_ids",
    "self_check_graph",
    "self_check_card_decisions",
    "LLMClient",
    "LLMResponseError",
    "OpenAICompatClient",
    "CachedClient",
    "FakeClient",
    "CARD_SELECTION_SYSTEM",
    "GRAPH_SYSTEM",
    "card_selection_user",
    "graph_user",
]
