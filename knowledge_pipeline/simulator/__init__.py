"""M2：dry-run 调度模拟器（mock 执行器 + E1~E8 调度器 + JSONL 轨迹）。

接口契约：docs/knowledge-pipeline/SPEC.md §4。
"""

from .mock_executor import (
    ExecutionResult,
    MockExecutor,
    PlaybookError,
    load_playbook,
)
from .scheduler import (
    LedgerLike,
    SimConfig,
    SimResult,
    Simulator,
    parse_delta_mid,
)
from .trace import EVENT_VOCAB, Trace, TraceEventError, read_trace

__all__ = [
    "EVENT_VOCAB",
    "ExecutionResult",
    "LedgerLike",
    "MockExecutor",
    "PlaybookError",
    "SimConfig",
    "SimResult",
    "Simulator",
    "Trace",
    "TraceEventError",
    "load_playbook",
    "parse_delta_mid",
    "read_trace",
]
