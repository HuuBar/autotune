"""W3-4：execute_sandbox 按命令签名统计连续失败次数。"""
import subprocess

import pytest

from tests.conftest import global_config
from tools.code import executor as executor_module
from tools.code.executor import SandboxTools


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """以 fake subprocess.run 模拟：容器存活 + docker exec 结果可编排。"""
    global_config["workspace"]["root_dir"] = str(tmp_path)
    state = {"exec_returncode": 1, "exec_stdout": "boom: something failed"}

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "ps"]:
            return _FakeCompleted(returncode=0, stdout="container123\n")  # 容器存活
        if cmd[:2] == ["docker", "inspect"]:
            return _FakeCompleted(returncode=0, stdout="autotune-env:latest\n")  # 镜像指纹一致
        if cmd[:2] == ["docker", "exec"]:
            return _FakeCompleted(
                returncode=state["exec_returncode"],
                stdout=state["exec_stdout"],
            )
        if cmd[:2] == ["docker", "rm"]:
            return _FakeCompleted(returncode=0)
        raise AssertionError(f"未预期的子进程调用: {cmd}")

    monkeypatch.setattr(executor_module.subprocess, "run", fake_run)
    tools = SandboxTools()
    return tools, state


def _fail_once(tools, command):
    return tools._execute_sandbox_impl(command)


class TestConsecutiveFailCounter:
    def test_same_command_counts_up(self, sandbox):
        tools, _ = sandbox
        res1 = _fail_once(tools, "python train.py")
        res2 = _fail_once(tools, "python train.py")
        assert "[连续失败 1/3]" in res1
        assert "[连续失败 2/3]" in res2

    def test_threshold_warning_on_third(self, sandbox):
        tools, _ = sandbox
        _fail_once(tools, "python train.py")
        _fail_once(tools, "python train.py")
        res3 = _fail_once(tools, "python train.py")
        assert "[连续失败 3/3]" in res3
        assert "已达重试上限" in res3

    def test_whitespace_variants_share_signature(self, sandbox):
        tools, _ = sandbox
        _fail_once(tools, "python   train.py")
        res = _fail_once(tools, "  python train.py  ")
        assert "[连续失败 2/3]" in res  # 归一化后视为同一命令

    def test_different_commands_counted_independently(self, sandbox):
        tools, _ = sandbox
        _fail_once(tools, "python a.py")
        res = _fail_once(tools, "python b.py")
        assert "[连续失败 1/3]" in res

    def test_success_resets_counter(self, sandbox):
        tools, state = sandbox
        _fail_once(tools, "python train.py")
        _fail_once(tools, "python train.py")
        state["exec_returncode"] = 0
        state["exec_stdout"] = "F1 = 0.90"
        ok = tools._execute_sandbox_impl("python train.py")
        assert "连续失败" not in ok  # 成功结果不带标记
        state["exec_returncode"] = 1
        res = _fail_once(tools, "python train.py")
        assert "[连续失败 1/3]" in res  # 计数已归零重新累计

    def test_tracking_disabled_suppresses_mark(self, sandbox, monkeypatch):
        """track_failures=False（内部程序化调用）不附带失败计数标记。"""
        tools, _ = sandbox
        tools._track_failures = False
        res = _fail_once(tools, "python train.py")
        assert "连续失败" not in res
        assert "⚠️ 运行失败" in res  # 失败事实本身仍如实返回

    def test_timeout_counts_as_failure(self, sandbox, monkeypatch):
        tools, _ = sandbox

        def timeout_run(cmd, **kwargs):
            if cmd[:2] == ["docker", "ps"]:
                return _FakeCompleted(returncode=0, stdout="container123\n")
            if cmd[:2] == ["docker", "exec"]:
                raise subprocess.TimeoutExpired(cmd, 300)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(executor_module.subprocess, "run", timeout_run)
        res = tools._execute_sandbox_impl("python hang.py")
        assert "运行超时" in res
        assert "[连续失败 1/3]" in res
