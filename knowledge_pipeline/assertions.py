"""验收断言解析与求值（机器可判）。

语义来源：SPEC §2。语法：``<metric_path> <op> <number>``；
op ∈ ``>=, <=, ==, >, <``；metric_path 允许点号、中文、字母、数字、下划线
（如 ``per_app.王者荣耀.F1``、``per_app.bilibili.F1_delta``）。

``_delta`` 后缀约定（SPEC §2）：表示相对基线的差值指标，取值时按完整
metric_path 在 facts 中查找（即 ``facts[metric_base + "_delta"]``，由调用方
提供差值事实），解析器本身不做差值计算。
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

__all__ = ["Assertion", "AssertionParseError", "parse_assertion", "evaluate"]

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}

#: 完整断言：`<metric> <op> <number>`（允许任意空白；数值支持符号/小数/科学计数）
_ASSERTION_RE = re.compile(
    r"^\s*(?P<metric>[^\s<>=!]+?)\s*(?P<op>>=|<=|==|>|<)\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)

#: metric_path 合法字符：字母/数字/下划线/点号/中文等非 ASCII 文字（re.UNICODE 下 \w 覆盖中文）
_METRIC_RE = re.compile(r"^\w[\w.]*$", re.UNICODE)


class AssertionParseError(ValueError):
    """断言解析失败。``assertion`` 属性保留原始输入串。"""

    def __init__(self, assertion: str, reason: str):
        self.assertion = assertion
        self.reason = reason
        super().__init__(f"无法解析验收断言 {assertion!r}: {reason}")


@dataclass(frozen=True)
class Assertion:
    """机器可判的验收断言：``metric op value``（SPEC §2）。"""

    metric: str
    op: str
    value: float

    def __str__(self) -> str:  # 便于日志/报告回显
        return f"{self.metric} {self.op} {self.value:g}"


def parse_assertion(s: str) -> Assertion:
    """解析 ``<metric_path> <op> <number>`` 形式的验收断言。

    解析失败抛 :class:`AssertionParseError`。
    """
    if not isinstance(s, str) or not s.strip():
        raise AssertionParseError(str(s), "断言为空或不是字符串")
    m = _ASSERTION_RE.match(s)
    if m is None:
        raise AssertionParseError(
            s, "不符合语法 '<metric_path> <op> <number>'（op ∈ >=, <=, ==, >, <）"
        )
    metric = m.group("metric")
    if not _METRIC_RE.match(metric) or ".." in metric or metric.endswith("."):
        raise AssertionParseError(
            s, f"非法 metric_path {metric!r}（仅允许点号、中文、字母、数字、下划线）"
        )
    return Assertion(metric=metric, op=m.group("op"), value=float(m.group("value")))


def _lookup(facts: Mapping[str, Any], metric: str) -> Any:
    """按点号路径在 facts 中取值。

    优先精确匹配扁平键（``facts["per_app.bilibili.F1_delta"]``），
    否则逐段下钻嵌套映射。未找到抛 :class:`KeyError`。
    """
    if metric in facts:
        return facts[metric]
    parts = metric.split(".")
    cur: Any = facts
    for part in parts:
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"facts 中缺少指标 {metric!r}（在段 {part!r} 处断链）")
    return cur


def evaluate(assertion: Assertion, facts: Mapping[str, Any]) -> bool:
    """对 facts 求值断言。facts 支持点号路径取值（嵌套 dict 或扁平带点键）。

    ``_delta`` 后缀指标直接按完整路径取值（差值事实由调用方提供，SPEC §2）。
    指标缺失抛 :class:`KeyError`；取值非数值抛 :class:`TypeError`。
    """
    if not isinstance(assertion, Assertion):
        raise TypeError(f"assertion 必须是 Assertion，得到 {type(assertion).__name__}")
    actual = _lookup(facts, assertion.metric)
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise TypeError(f"指标 {assertion.metric!r} 的值不是数值: {actual!r}")
    return _OPS[assertion.op](float(actual), assertion.value)
