"""M4 编译 CLI：``python -m knowledge_pipeline.compiler compile ...``。

用法：

.. code-block:: bash

    python -m knowledge_pipeline.compiler compile \\
        --cards <知识卡库.md> --brief <任务书.yaml> --out <产物目录> \\
        [--live] [--cache <目录>] [--max-retries N]

默认离线链：:class:`CachedClient` 重放预录缓存（``--cache``，缺省
``fixtures/llm_cache``）；``--live`` 切换 :class:`OpenAICompatClient`，
端点由 ``KP_LLM_BASE_URL`` / ``KP_LLM_MODEL`` / ``KP_LLM_API_KEY``
环境变量注入（缺失时报可读错误并非零退出）。

产物（``--out`` 目录）：

- ``graph.yaml``           编译产出的干净 YAML（过 M1 校验门，不绕过校验）
- ``card_decisions.json``  选卡论证表（每张卡 adopt/reject + rationale）

stdout 打印选卡数 / 领域数 / 校验门结果；编译失败 stderr 给人类可读原因且
退出码非 0。``--max-retries`` 为编译器每步（选卡/成图）LLM 最大尝试次数
（默认 3，与 ``knowledge_pipeline.e2e.run_pipeline`` 口径一致）；
``--live`` 客户端的网络层瞬态重试由 ``OpenAICompatClient`` 构造默认值承担
（重试 2 次、退避 1s→4s，见 llm_client 模块 docstring）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ..validator import validate_graph
from .compiler import CompileError, CompileResult, compile_graph
from .llm_client import CachedClient, LLMClient, LLMResponseError, OpenAICompatClient

__all__ = ["DEFAULT_CACHE", "build_parser", "main", "run_compile"]

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # knowledge_pipeline/

#: 离线模式默认 LLM 预录缓存目录
DEFAULT_CACHE = _PACKAGE_ROOT / "fixtures" / "llm_cache"

#: 产物文件名（--out 目录内）
GRAPH_YAML = "graph.yaml"
CARD_DECISIONS_JSON = "card_decisions.json"


class CliError(RuntimeError):
    """CLI 层失败（消息为人类可读原因，stderr 输出 + 非零退出）。"""


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m knowledge_pipeline.compiler",
        description="M4 编排编译器 CLI：选卡论证 → 成图 → M1 确定性校验门",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    compile_p = sub.add_parser(
        "compile", help="编译编排图（产出 graph.yaml + card_decisions.json）"
    )
    compile_p.add_argument("--cards", required=True, help="知识卡库 Markdown")
    compile_p.add_argument("--brief", required=True, help="任务书+地形 YAML")
    compile_p.add_argument("--out", required=True, help="产物目录")
    compile_p.add_argument(
        "--live",
        action="store_true",
        help="走真实 LLM 端点（OpenAICompatClient，KP_LLM_BASE_URL/KP_LLM_MODEL/"
        "KP_LLM_API_KEY 环境变量注入）；默认离线重放预录缓存",
    )
    compile_p.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="离线模式 LLM 预录缓存目录（默认 fixtures/llm_cache）",
    )
    compile_p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="编译器每步（选卡/成图）LLM 最大尝试次数（默认 3）",
    )
    return parser


# --------------------------------------------------------------------------
# 编译流程
# --------------------------------------------------------------------------


def _build_client(args: argparse.Namespace) -> tuple[LLMClient, str]:
    """按 --live/--cache 构造 LLM 客户端，返回 (client, 模式描述)。"""
    if args.live:
        try:
            client = OpenAICompatClient()  # 端点由 KP_LLM_* 环境变量注入
        except ValueError as exc:
            raise CliError(f"live 模式端点配置缺失: {exc}") from exc
        return client, f"live（{client.endpoint}，model={client.model}）"
    try:
        client = CachedClient(args.cache)
    except ValueError as exc:
        raise CliError(f"离线缓存不可用: {exc}") from exc
    return client, f"offline（缓存 {args.cache}）"


def run_compile(args: argparse.Namespace) -> CompileResult:
    """跑编译并落盘产物，返回 :class:`CompileResult`；失败抛 :class:`CliError`。"""
    cards, brief = Path(args.cards), Path(args.brief)
    for p, label in ((cards, "知识卡库"), (brief, "任务书")):
        if not p.is_file():
            raise CliError(f"{label}文件不存在: {p}")
    client, mode = _build_client(args)
    try:
        result = compile_graph(cards, brief, client, max_retries=args.max_retries)
    except KeyError as exc:
        # CachedClient 缓存未命中（含缓存损坏/条目缺失）
        raise CliError(f"LLM 预录缓存未命中（缓存缺失或损坏）: {exc}") from exc
    except LLMResponseError as exc:
        raise CliError(f"LLM 端点调用失败: {exc}") from exc
    except (CompileError, ValueError) as exc:
        raise CliError(str(exc)) from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / GRAPH_YAML).write_text(result.graph_yaml + "\n", encoding="utf-8")
    (out / CARD_DECISIONS_JSON).write_text(
        json.dumps(result.card_decisions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # stdout：选卡数 / 领域数 / 校验门结果
    adopted = sum(1 for d in result.card_decisions if d["decision"] == "adopt")
    domains = [
        n for n in (result.graph.get("nodes") or []) if n.get("layer") == "domain"
    ]
    print(f"[编译] 模式={mode}")
    print(f"[选卡] 共 {len(result.card_decisions)} 张"
          f"（adopt {adopted} / reject {len(result.card_decisions) - adopted}），"
          f"选卡调用 {result.select_attempts} 次、成图调用 {result.attempts} 次")
    print(f"[成图] 领域 {len(domains)} 个 / "
          f"节点 {len(result.graph.get('nodes') or [])} 个 / "
          f"边 {len(result.graph.get('edges') or [])} 条")
    errors = validate_graph(result.graph)
    if errors:  # 编译门已拦过，此处属防御性复查（不绕过校验）
        detail = "\n".join(f"  - {e}" for e in errors[:20])
        raise CliError(f"编译产物未通过 M1 校验（{len(errors)} 条）:\n{detail}")
    print("[校验门] graph.yaml 通过 M1 确定性校验（V-* 全绿）")
    print(f"[产物] {out}/{GRAPH_YAML} + {out}/{CARD_DECISIONS_JSON}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compile":
        try:
            run_compile(args)
        except CliError as exc:
            print(f"编译失败: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 — 兜底：任何未分类失败都要可读、非 0
            print(f"编译失败（未分类异常）: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
        return 0
    raise AssertionError(f"未知子命令: {args.command}")  # argparse required=True 拦截


if __name__ == "__main__":
    sys.exit(main())
