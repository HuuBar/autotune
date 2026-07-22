"""M1 验收断言解析与求值测试。

语义来源：SPEC §2。语法 ``<metric_path> <op> <number>``；
metric_path 允许点号、中文、字母、数字、下划线；``_delta`` 后缀表示
相对基线差值，取值按完整路径在 facts 中查找。
"""

from __future__ import annotations

import pytest

from knowledge_pipeline import (
    Assertion,
    AssertionParseError,
    evaluate,
    load_graph,
    parse_assertion,
)
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[2] / "knowledge_pipeline" / "fixtures" / "tbox_golden.yaml"


# --------------------------------------------------------------------------
# 解析正例
# --------------------------------------------------------------------------


def test_golden_acceptance_all_parseable() -> None:
    """黄金夹具 goal.acceptance 全部可解析（SPEC §2 + V-ACCEPTANCE 基准）。"""
    doc = load_graph(GOLDEN)
    for line in doc["goal"]["acceptance"]:
        a = parse_assertion(line)
        assert a.metric and a.op in {">=", "<=", "==", ">", "<"}
        assert isinstance(a.value, float)


@pytest.mark.parametrize(
    "text, metric, op, value",
    [
        ("overall.F1 >= 0.935", "overall.F1", ">=", 0.935),
        ("per_app.王者荣耀.F1 >= 0.75", "per_app.王者荣耀.F1", ">=", 0.75),
        ("per_app.bilibili.F1_delta >= -0.01", "per_app.bilibili.F1_delta", ">=", -0.01),
        ("train_time <= 60", "train_time", "<=", 60.0),
        ("acc == 1.0", "acc", "==", 1.0),
        ("loss < 0.5", "loss", "<", 0.5),
        ("gain > 0", "gain", ">", 0.0),
        ("score >= +1.5e-3", "score", ">=", 0.0015),
        ("metric_path >=.5", "metric_path", ">=", 0.5),
        ("  a.b_c >=   2  ", "a.b_c", ">=", 2.0),
    ],
)
def test_parse_valid(text: str, metric: str, op: str, value: float) -> None:
    a = parse_assertion(text)
    assert (a.metric, a.op) == (metric, op)
    assert a.value == pytest.approx(value)


def test_assertion_str_roundtrip() -> None:
    a = parse_assertion("overall.F1 >= 0.935")
    assert str(a) == "overall.F1 >= 0.935"


# --------------------------------------------------------------------------
# 解析反例
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",                     # 空串
        "   ",                  # 全空白
        "王者荣耀 F1 明显更好",   # 无操作符
        "overall.F1 =>= 0.9",   # 非法操作符
        "overall.F1 = 0.9",     # 单等号不在 op 词表
        "overall.F1 >= ",       # 缺数值
        ">= 0.9",               # 缺 metric
        "overall.F1 >= high",   # 数值不可解析
        "overall/F1 >= 0.9",    # metric 含非法字符
        "overall..F1 >= 0.9",   # metric 含连续点号
        "overall.F1 >= 0.9 extra",  # 尾部多余 token
    ],
)
def test_parse_invalid(text: str) -> None:
    with pytest.raises(AssertionParseError):
        parse_assertion(text)


def test_parse_error_message_contains_input() -> None:
    with pytest.raises(AssertionParseError) as excinfo:
        parse_assertion("王者荣耀 F1 明显更好")
    assert "王者荣耀 F1 明显更好" in str(excinfo.value)


# --------------------------------------------------------------------------
# 求值
# --------------------------------------------------------------------------


FACTS_NESTED = {
    "overall": {"F1": 0.94, "Recall": 0.975},
    "per_app": {
        "王者荣耀": {"F1": 0.76},
        "bilibili": {"F1_delta": -0.005},
        "抖音": {"F1_delta": -0.02},
    },
}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("overall.F1 >= 0.935", True),
        ("overall.F1 >= 0.95", False),
        ("overall.Recall >= 0.97", True),
        ("per_app.王者荣耀.F1 >= 0.75", True),
        ("per_app.bilibili.F1_delta >= -0.01", True),   # _delta 差值指标
        ("per_app.抖音.F1_delta >= -0.01", False),       # 降幅超阈值
        ("overall.F1 == 0.94", True),
        ("overall.F1 < 1.0", True),
        ("overall.F1 > 0.94", False),
    ],
)
def test_evaluate_nested_facts(text: str, expected: bool) -> None:
    assert evaluate(parse_assertion(text), FACTS_NESTED) is expected


def test_evaluate_flat_dotted_keys() -> None:
    """facts 也可为扁平带点键的 dict（调用方直接提供路径键）。"""
    facts = {"per_app.bilibili.F1_delta": -0.005, "overall.F1": 0.9}
    assert evaluate(parse_assertion("per_app.bilibili.F1_delta >= -0.01"), facts) is True
    assert evaluate(parse_assertion("overall.F1 >= 0.935"), facts) is False


def test_evaluate_missing_metric_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        evaluate(parse_assertion("overall.F2 >= 0.9"), FACTS_NESTED)


def test_evaluate_non_numeric_fact_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        evaluate(parse_assertion("overall.label >= 0.9"), {"overall": {"label": "good"}})


def test_evaluate_rejects_non_assertion() -> None:
    with pytest.raises(TypeError):
        evaluate("overall.F1 >= 0.9", FACTS_NESTED)  # type: ignore[arg-type]
