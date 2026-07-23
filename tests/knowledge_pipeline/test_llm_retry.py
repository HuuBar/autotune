"""W-A1：OpenAICompatClient 网络层瞬态重试测试。

验收口径（交接包 §工作包 W-A1）：

a) mock 端点先 500 后 200 → 成功且重试发生；
b) 连续 500 → LLMResponseError（重试耗尽，共 3 次尝试）；
c) HTTP 400 → 立即失败零重试（内容/请求类错误不重试）；
d) 退避间隔由注入的 sleep 函数捕获（测试不许真 sleep），指数退避 1s → 4s。

另覆盖：连接失败（URLError）/ socket 超时视为瞬态；响应结构异常
（LLMResponseError 内容类）不重试；``max_retries=0`` 关闭重试。
所有用例 monkeypatch ``urllib.request.urlopen``，不发真实网络。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from knowledge_pipeline.compiler.llm_client import (
    LLMResponseError,
    OpenAICompatClient,
)

_ENDPOINT = "http://mock-llm.local/v1"


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        _ENDPOINT + "/chat/completions", code, f"HTTP {code}", {}, None
    )


def _ok_response(content: str = "hello") -> Any:
    """伪造 urlopen 返回值（200，OpenAI 兼容响应体，支持 with 语义）。"""
    body = json.dumps(
        {"choices": [{"message": {"content": content}}]}
    ).encode("utf-8")
    return io.BytesIO(body)


def _make_client(sleeps: list[float], **kwargs: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url=_ENDPOINT, model="mock-model", sleep=sleeps.append, **kwargs
    )


@pytest.fixture
def calls(monkeypatch):
    """替换 urlopen 为剧本驱动 fake；返回 (调用记录列表, 剧本队列 setter)。"""

    class FakeEndpoint:
        def __init__(self) -> None:
            self.script: list[Any] = []
            self.calls: list[urllib.request.Request] = []

        def __call__(self, req, timeout=None):
            self.calls.append(req)
            assert len(self.script) >= len(self.calls), "剧本耗尽仍被调用"
            item = self.script[len(self.calls) - 1]
            if isinstance(item, BaseException):
                raise item
            return item

    fake = FakeEndpoint()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def test_500_then_200_succeeds_with_retry(calls):
    """a) 先 500 后 200 → 成功且重试恰好发生一次。"""
    sleeps: list[float] = []
    calls.script = [_http_error(500), _ok_response("retry-ok")]
    client = _make_client(sleeps)

    assert client.complete("sys", "usr") == "retry-ok"
    assert len(calls.calls) == 2  # 首次 + 1 次重试
    assert sleeps == [1.0]  # 首次退避 1s


def test_continuous_500_raises_after_exhaustion(calls):
    """b) 连续 500 → LLMResponseError；共 3 次尝试（默认 max_retries=2）。"""
    sleeps: list[float] = []
    calls.script = [_http_error(500)] * 3
    client = _make_client(sleeps)

    with pytest.raises(LLMResponseError, match="瞬态错误已重试 2 次"):
        client.complete("sys", "usr")
    assert len(calls.calls) == 3
    assert sleeps == [1.0, 4.0]  # d) 指数退避 1s → 4s，且未真 sleep


def test_http_400_fails_immediately_without_retry(calls):
    """c) HTTP 400 → 立即失败零重试、零退避。"""
    sleeps: list[float] = []
    calls.script = [_http_error(400)]  # 剧本仅一条：若重试则 fake 断言失败
    client = _make_client(sleeps)

    with pytest.raises(LLMResponseError, match="HTTP 400"):
        client.complete("sys", "usr")
    assert len(calls.calls) == 1
    assert sleeps == []


def test_http_503_is_transient(calls):
    """HTTP 503 同属 5xx → 视为瞬态并触发重试。"""
    sleeps: list[float] = []
    calls.script = [_http_error(503), _ok_response()]
    client = _make_client(sleeps)

    assert client.complete("sys", "usr") == "hello"
    assert len(calls.calls) == 2


def test_connection_failure_and_timeout_are_transient(calls):
    """连接失败（URLError）与 socket 超时（TimeoutError）均为瞬态。"""
    sleeps: list[float] = []
    calls.script = [
        urllib.error.URLError("connection refused"),
        TimeoutError("socket timed out"),
        _ok_response("recovered"),
    ]
    client = _make_client(sleeps)

    assert client.complete("sys", "usr") == "recovered"
    assert len(calls.calls) == 3
    assert sleeps == [1.0, 4.0]


def test_content_error_not_retried(calls):
    """响应结构异常（LLMResponseError 内容类）不重试，首次即抛。"""
    sleeps: list[float] = []
    calls.script = [io.BytesIO(b'{"unexpected": true}')]  # 200 但缺 choices
    client = _make_client(sleeps)

    with pytest.raises(LLMResponseError, match="响应结构异常"):
        client.complete("sys", "usr")
    assert len(calls.calls) == 1
    assert sleeps == []


def test_max_retries_injectable(calls):
    """重试次数可构造参数注入：max_retries=0 → 单次尝试即抛。"""
    sleeps: list[float] = []
    calls.script = [_http_error(500)]
    client = _make_client(sleeps, max_retries=0)

    with pytest.raises(LLMResponseError, match="瞬态错误已重试 0 次"):
        client.complete("sys", "usr")
    assert len(calls.calls) == 1
    assert sleeps == []


def test_backoff_sequence_injectable(calls):
    """退避参数可注入：backoff_base/factor 改变退避序列。"""
    sleeps: list[float] = []
    calls.script = [_http_error(500)] * 4
    client = _make_client(sleeps, max_retries=3, backoff_base=0.5, backoff_factor=2.0)

    with pytest.raises(LLMResponseError):
        client.complete("sys", "usr")
    assert len(calls.calls) == 4
    assert sleeps == [0.5, 1.0, 2.0]
