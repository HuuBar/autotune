"""W-A2：``python -m knowledge_pipeline.compiler compile`` CLI 测试。

验收口径（交接包 §工作包 W-A2 与 §2）：

a) 验收命令原样跑通：离线缓存模式 exit=0，graph.yaml + card_decisions.json
   两件产物齐全，graph.yaml 可解析且过 M1 校验（V-* 全绿），stdout 打印
   选卡数 / 领域数 / 校验门结果；
b) 输入文件缺失 / 缓存未命中 → 非零退出，stderr 可读；
c) ``--live`` 缺 KP_LLM_* 环境变量 → 非零退出，stderr 可读报错（指明缺哪个变量）；
d) main() 直接调用（不经子进程）同样返回非 0 且不抛栈。

所有用例离线运行（CachedClient 预录缓存 / 环境变量缺失），不发真实网络。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from knowledge_pipeline.compiler.__main__ import main
from knowledge_pipeline.validator import validate_graph

REPO_ROOT = Path(__file__).resolve().parents[2]
CARDS_MD = REPO_ROOT / "docs" / "knowledge-pipeline" / "自动化ML优化战术知识卡_v1.md"
BRIEF = REPO_ROOT / "knowledge_pipeline" / "fixtures" / "tbox_brief.yaml"
LLM_CACHE = REPO_ROOT / "knowledge_pipeline" / "fixtures" / "llm_cache"


def _run_cli(*argv: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """以子进程跑 ``python -m knowledge_pipeline.compiler``（隔离环境变量）。"""
    env = dict(os.environ)
    for var in ("KP_LLM_BASE_URL", "KP_LLM_MODEL", "KP_LLM_API_KEY"):
        env.pop(var, None)
    env.update(extra_env or {})
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "knowledge_pipeline.compiler", *argv],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_compile_acceptance_command(tmp_path):
    """a) 交接包 §2 验收命令（--out 换成 tmp_path）：exit=0 + 产物 + M1 全绿。"""
    out = tmp_path / "compile_out_test"
    proc = _run_cli(
        "compile",
        "--cards", str(CARDS_MD),
        "--brief", str(BRIEF),
        "--out", str(out),
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    # stdout 关键行：选卡数 / 领域数 / 校验门结果
    assert "[选卡]" in proc.stdout and "adopt" in proc.stdout
    assert "[成图] 领域" in proc.stdout
    assert "[校验门]" in proc.stdout and "V-* 全绿" in proc.stdout

    # 两件产物齐全且可解析；graph.yaml 过 M1 校验
    graph_path = out / "graph.yaml"
    decisions_path = out / "card_decisions.json"
    assert graph_path.is_file() and decisions_path.is_file()
    graph = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
    assert isinstance(graph, dict)
    assert validate_graph(graph) == []
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert isinstance(decisions, list) and decisions
    assert all(d["decision"] in ("adopt", "reject") for d in decisions)


def test_compile_explicit_cache_dir(tmp_path):
    """显式 --cache 指向预录缓存目录，同样跑通。"""
    out = tmp_path / "out"
    proc = _run_cli(
        "compile",
        "--cards", str(CARDS_MD),
        "--brief", str(BRIEF),
        "--out", str(out),
        "--cache", str(LLM_CACHE),
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert (out / "graph.yaml").is_file()


def test_missing_cards_file_readable_error(tmp_path):
    """b) 输入文件缺失 → 非零退出，stderr 指明缺失文件。"""
    proc = _run_cli(
        "compile",
        "--cards", str(tmp_path / "no_such_cards.md"),
        "--brief", str(BRIEF),
        "--out", str(tmp_path / "out"),
    )
    assert proc.returncode != 0
    assert "编译失败" in proc.stderr
    assert "知识卡库文件不存在" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cache_miss_readable_error(tmp_path):
    """b) 缓存目录为空（未预录）→ 非零退出，stderr 可读。"""
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    proc = _run_cli(
        "compile",
        "--cards", str(CARDS_MD),
        "--brief", str(BRIEF),
        "--out", str(tmp_path / "out"),
        "--cache", str(empty_cache),
    )
    assert proc.returncode != 0
    assert "编译失败" in proc.stderr
    assert "缓存未命中" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_live_missing_env_vars_readable_error(tmp_path):
    """c) --live 缺 KP_LLM_BASE_URL / KP_LLM_MODEL → 非零退出 + 可读报错。"""
    proc = _run_cli(
        "compile",
        "--cards", str(CARDS_MD),
        "--brief", str(BRIEF),
        "--out", str(tmp_path / "out"),
        "--live",
    )
    assert proc.returncode != 0
    assert "编译失败" in proc.stderr
    assert "KP_LLM_BASE_URL" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_main_returns_nonzero_on_failure(tmp_path, capsys):
    """d) main() 直接调用：失败返回 1、stderr 可读、不抛栈。"""
    rc = main([
        "compile",
        "--cards", str(tmp_path / "missing.md"),
        "--brief", str(BRIEF),
        "--out", str(tmp_path / "out"),
    ])
    assert rc == 1
    assert "编译失败" in capsys.readouterr().err


def test_main_success_returns_zero(tmp_path):
    """d) main() 直接调用成功路径返回 0。"""
    out = tmp_path / "out"
    rc = main([
        "compile",
        "--cards", str(CARDS_MD),
        "--brief", str(BRIEF),
        "--out", str(out),
    ])
    assert rc == 0
    assert (out / "graph.yaml").is_file()
    assert (out / "card_decisions.json").is_file()
