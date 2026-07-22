"""执行轨迹日志（JSONL）。

语义来源：SPEC §4.3。每事件一行：``{tick, event, node_id?, detail}``。
事件词表见 :data:`EVENT_VOCAB`；其中 ``decision_deadend`` 在 SPEC §4.2-E3
正文中被显式要求（"记 ``decision_deadend`` 事件"），虽未列入 §4.3 词表，
按 E3 裁决纳入（列入交付报告「新发现」）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["EVENT_VOCAB", "Trace", "TraceEventError", "read_trace"]

#: 事件词表（SPEC §4.3 + §4.2-E3 的 decision_deadend）
EVENT_VOCAB: tuple[str, ...] = (
    "unlock",          # E1：节点解锁 pending → ready
    "select",          # E2：本 tick 选中执行的节点
    "execute",         # 一次执行尝试（含边际提升复跑，detail.rerun 标注）
    "verify_fail",     # E5：verify 未通过
    "retry",           # E5：消耗 retry_budget 重试
    "blocked",         # 节点进入 blocked / blocked_pending（终态，禁止静默跳过）
    "escalate",        # E5/E8：阻塞升级，冻结下游
    "decision",        # E3：决策点判定（分支索引 + 理由强制落盘）
    "decision_deadend",  # E3：无分支命中且无 else 兜底
    "param_update",    # E6：informs 源产物注入目标 params
    "no_parallel_hold",  # E4：对端在 running/同批候选，推迟并记录原因
    "writeback",       # E8：假说裁决写回账本成功
    "early_stop",      # E7：连续 k 个假说验证无验收进展
    "acceptance_check",  # 验收断言核对（每次假说裁决后 + 终局）
    "run_end",         # 运行结束（图耗尽 / 早停 / 超 tick 上限）
)


class TraceEventError(ValueError):
    """事件名不在词表内（防止轨迹事件随意造词，保证下游可解析）。"""


class Trace:
    """JSONL 轨迹写出器。

    ``path`` 为 ``None`` 时只保留内存副本（不落盘），便于纯内存测试。
    每个事件同时追加到内存列表，:meth:`events` 返回完整副本。
    """

    def __init__(self, path: str | Path | None = None):
        self.path: Path | None = Path(path) if path is not None else None
        self._events: list[dict[str, Any]] = []
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    def emit(
        self,
        tick: int,
        event: str,
        node_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """追加一条事件。``event`` 必须在 :data:`EVENT_VOCAB` 内。"""
        if event not in EVENT_VOCAB:
            raise TraceEventError(f"非法轨迹事件 {event!r}（词表见 EVENT_VOCAB）")
        record: dict[str, Any] = {"tick": tick, "event": event}
        if node_id is not None:
            record["node_id"] = node_id
        record["detail"] = dict(detail or {})
        self._events.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        return record

    def events(self) -> list[dict[str, Any]]:
        """返回已发事件的完整副本（按发生顺序）。"""
        return [dict(e) for e in self._events]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 轨迹文件为事件列表（测试与下游分析用）。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
