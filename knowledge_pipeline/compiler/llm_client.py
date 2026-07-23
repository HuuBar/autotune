"""M4 可替换 LLM 客户端。

接口契约（SPEC §6）::

    class LLMClient(Protocol):
        def complete(self, system: str, user: str) -> str: ...

实现：

* :class:`OpenAICompatClient` —— OpenAI 兼容端点（``POST {base_url}/chat/completions``）。
  ``base_url`` / ``model`` / ``api_key`` 由构造参数注入，缺省回落到环境变量
  ``KP_LLM_BASE_URL`` / ``KP_LLM_MODEL`` / ``KP_LLM_API_KEY``。仅使用标准库
  ``urllib``，不引入第三方 HTTP 库。
* :class:`CachedClient` —— 离线重放：从 cache 目录读预录响应，供 e2e 与测试离线使用。
* :class:`FakeClient` —— 测试用注册表（子串匹配 → 响应/可调用/响应序列）。

CachedClient 寻址方案
----------------------
cache 键 = ``sha256((system + "\\x00" + user).encode("utf-8")).hexdigest()``，
响应文件为 ``<cache_dir>/<key>.txt``（UTF-8 原文，可含代码围栏，由编译器鲁棒解析）。
cache 目录可同时维护一份人工可读的 ``index.json``（key → 用途说明），
CachedClient 不依赖它，仅供维护者定位条目。预录条目用
``CachedClient.cache_key(system, user)`` 生成（见 fixtures/llm_cache/index.json）。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

__all__ = [
    "LLMClient",
    "LLMResponseError",
    "OpenAICompatClient",
    "CachedClient",
    "FakeClient",
]

#: 环境变量名（SPEC §6）
ENV_BASE_URL = "KP_LLM_BASE_URL"
ENV_API_KEY = "KP_LLM_API_KEY"
ENV_MODEL = "KP_LLM_MODEL"


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客户端协议：system + user 两段式补全，返回原始文本。"""

    def complete(self, system: str, user: str) -> str: ...


class LLMResponseError(RuntimeError):
    """端点返回异常（HTTP 错误、响应结构不符、缺 choices 等）。"""


class _TransientHTTPError(RuntimeError):
    """瞬态网络层错误（连接失败 / socket 超时 / HTTP 5xx），内部重试信号。"""


class OpenAICompatClient:
    """OpenAI 兼容聊天补全客户端（urllib 实现，网络层瞬态重试）。

    参数缺省依次回落：构造参数 → 环境变量 ``KP_LLM_BASE_URL`` /
    ``KP_LLM_MODEL`` / ``KP_LLM_API_KEY``。``base_url`` 与 ``model`` 必填，
    ``api_key`` 可空（本地自托管端点常见）。

    重试口径：仅**网络层瞬态错误**（连接失败 / socket 超时 / HTTP 5xx）
    触发重试，指数退避 1s → 4s（``backoff_base * backoff_factor**n``），
    最多重试 ``max_retries`` 次（默认 2，共 3 次尝试），耗尽后抛
    :class:`LLMResponseError`。HTTP 4xx 与响应结构类错误（同样是
    :class:`LLMResponseError`）**不重试**，首次即抛。``sleep`` 可注入
    便于测试（默认 ``time.sleep``，测试不得真 sleep）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        backoff_factor: float = 4.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL) or "").rstrip("/")
        self.model = model or os.environ.get(ENV_MODEL) or ""
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        self.timeout = timeout
        self.temperature = temperature
        if max_retries < 0:
            raise ValueError("max_retries 必须 >= 0")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self.sleep = sleep if sleep is not None else time.sleep
        if not self.base_url:
            raise ValueError(
                f"缺少 LLM base_url：请传构造参数或设置环境变量 {ENV_BASE_URL}"
            )
        if not self.model:
            raise ValueError(
                f"缺少 LLM model：请传构造参数或设置环境变量 {ENV_MODEL}"
            )

    @property
    def endpoint(self) -> str:
        """聊天补全端点 URL（``{base_url}/chat/completions``）。"""
        return f"{self.base_url}/chat/completions"

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def _build_request(self, system: str, user: str) -> urllib.request.Request:
        """构造 HTTP 请求（独立成方法便于离线测试 URL/头/体，不发真实网络）。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(self._payload(system, user), ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")

    def _send_once(self, req: urllib.request.Request) -> str:
        """单次 HTTP 尝试，返回响应原文。

        错误分类：HTTP 5xx / 连接失败 / socket 超时 → :class:`_TransientHTTPError`
        （交由重试层处理）；HTTP 4xx 等其余 HTTP 错误 → :class:`LLMResponseError`
        （内容/请求类错误，不重试）。
        """
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if 500 <= exc.code < 600:
                raise _TransientHTTPError(f"HTTP {exc.code} {exc.reason}") from exc
            raise LLMResponseError(
                f"LLM 端点返回 HTTP {exc.code}（非瞬态，不重试）: {exc.reason}"
            ) from exc
        except OSError as exc:  # URLError / ConnectionError / socket.timeout 等
            raise _TransientHTTPError(f"连接失败或超时: {exc}") from exc

    def _post_with_retry(self, req: urllib.request.Request) -> str:
        """网络层重试循环：仅瞬态错误重试，指数退避，耗尽抛 LLMResponseError。"""
        last: _TransientHTTPError | None = None
        for attempt in range(1, self.max_retries + 2):  # 共 max_retries+1 次尝试
            try:
                return self._send_once(req)
            except _TransientHTTPError as exc:
                last = exc
                if attempt > self.max_retries:
                    break
                delay = self.backoff_base * (self.backoff_factor ** (attempt - 1))
                self.sleep(delay)
        raise LLMResponseError(
            f"LLM 端点请求失败（瞬态错误已重试 {self.max_retries} 次）: {last}"
        ) from last

    def complete(self, system: str, user: str) -> str:
        req = self._build_request(system, user)
        raw = self._post_with_retry(req)
        try:
            payload = json.loads(raw)
            return str(payload["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"LLM 端点响应结构异常: {exc}: {raw[:200]!r}") from exc


class CachedClient:
    """预录响应重放客户端（e2e 离线用）。

    寻址方案：``key = sha256((system + "\\x00" + user).encode()).hexdigest()``，
    读取 ``<cache_dir>/<key>.txt``。命中失败抛 :class:`KeyError`（含提示，
    便于定位是哪一步 prompt 未预录）。
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise ValueError(f"cache 目录不存在: {self.cache_dir}")

    @staticmethod
    def cache_key(system: str, user: str) -> str:
        """按 prompt 内容寻址的稳定哈希（见模块 docstring）。"""
        h = hashlib.sha256()
        h.update(system.encode("utf-8"))
        h.update(b"\x00")
        h.update(user.encode("utf-8"))
        return h.hexdigest()

    def cache_path(self, system: str, user: str) -> Path:
        return self.cache_dir / f"{self.cache_key(system, user)}.txt"

    def complete(self, system: str, user: str) -> str:
        path = self.cache_path(system, user)
        if not path.is_file():
            raise KeyError(
                f"cache 未命中: {path.name}（该 prompt 未预录；"
                f"可用 CachedClient.cache_key 生成条目，参见 fixtures/llm_cache/index.json）"
            )
        return path.read_text(encoding="utf-8")


#: FakeClient 注册表项的响应类型：固定文本 / 可调用 / 文本序列（依次消费，用尽取末项）
FakeResponse = str | Callable[[str, str], str] | Sequence[str]


class FakeClient:
    """测试用客户端：按注册表（子串匹配 → 响应）返回。

    - ``register(match, response)``：当 ``match`` 是 ``system + user`` 的子串时命中，
      先注册者优先。
    - ``response`` 可为：固定字符串；``callable(system, user) -> str``；
      字符串序列（每次命中依次消费，用尽后重复末项）——用于"先坏后好"的重试剧本。
    - 全部调用记录在 ``calls``（``[(system, user), ...]``）。
    """

    def __init__(self, registry: dict[str, FakeResponse] | None = None) -> None:
        self._registry: list[list[Any]] = []
        self.calls: list[tuple[str, str]] = []
        for match, resp in (registry or {}).items():
            self.register(match, resp)

    def register(self, match: str, response: FakeResponse) -> None:
        self._registry.append([match, response, 0])

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        text = system + "\n" + user
        for entry in self._registry:
            match, resp, used = entry
            if match not in text:
                continue
            if callable(resp):
                return resp(system, user)
            if isinstance(resp, str):
                return resp
            # 序列：依次消费，用尽取末项
            idx = min(used, len(resp) - 1)
            entry[2] = used + 1
            return str(resp[idx])
        raise KeyError(
            f"FakeClient 无匹配注册项（已注册 match: {[e[0] for e in self._registry]}）"
        )
