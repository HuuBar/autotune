"""W3-6：镜像 tag 混入 requirements md5 + manager.md 锚点规范。"""
import os

import pytest

from tests.conftest import global_config
from tools.code import executor as executor_module
from tools.code.executor import SandboxTools
from utils.docker_image import compute_requirements_hash, resolve_sandbox_image


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestResolveSandboxImage:
    def test_tag_contains_requirements_md5(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("pandas==2.3.2\n", encoding="utf-8")
        cfg = {"name": "autotune-env", "requirements_path": str(req)}
        image = resolve_sandbox_image(cfg)
        assert image == f"autotune-env:{compute_requirements_hash(str(req))}"
        assert len(image.split(":")[1]) == 8

    def test_tag_changes_when_requirements_change(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("pandas==2.3.2\n", encoding="utf-8")
        cfg = {"name": "autotune-env", "requirements_path": str(req)}
        before = resolve_sandbox_image(cfg)
        req.write_text("pandas==2.3.2\nxgboost==3.0.0\n", encoding="utf-8")  # 依赖变更
        after = resolve_sandbox_image(cfg)
        assert before != after  # → warmup 检测不到同名镜像 → 自动重建

    def test_fallback_latest_without_requirements(self):
        assert resolve_sandbox_image({"name": "autotune-env", "requirements_path": ""}) == "autotune-env:latest"
        assert resolve_sandbox_image({"name": "autotune-env", "requirements_path": "/no/such/file"}) == "autotune-env:latest"


class TestExecutorImageConsistency:
    """executor 使用的镜像必须与 warmup 构建口径一致，且旧指纹容器会被重建。"""

    @pytest.fixture()
    def env(self, tmp_path, monkeypatch):
        req = tmp_path / "requirements.txt"
        req.write_text("pandas==2.3.2\n", encoding="utf-8")
        global_config["workspace"]["root_dir"] = str(tmp_path)
        global_config["sandbox"]["docker"]["requirements_path"] = str(req)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["docker", "ps"]:
                return _FakeCompleted(returncode=0, stdout="")  # 无存活容器
            if cmd[:2] == ["docker", "exec"]:
                return _FakeCompleted(returncode=0, stdout="ok")
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(executor_module.subprocess, "run", fake_run)
        yield calls
        global_config["sandbox"]["docker"]["requirements_path"] = ""

    def test_executor_uses_hashed_tag(self, env):
        SandboxTools()._execute_sandbox_impl("python x.py")
        run_cmds = [c for c in env if c[:2] == ["docker", "run"]]
        assert run_cmds, "未触发容器启动"
        expected = resolve_sandbox_image(global_config["sandbox"]["docker"])
        assert expected in run_cmds[0]
        assert expected != "autotune-env:latest"

    def test_stale_container_recreated(self, tmp_path, monkeypatch):
        req = tmp_path / "requirements.txt"
        req.write_text("pandas==2.3.2\n", encoding="utf-8")
        global_config["workspace"]["root_dir"] = str(tmp_path)
        global_config["sandbox"]["docker"]["requirements_path"] = str(req)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["docker", "ps"]:
                return _FakeCompleted(returncode=0, stdout="abc123\n")  # 旧容器存活
            if cmd[:2] == ["docker", "inspect"]:
                return _FakeCompleted(returncode=0, stdout="autotune-env:deadbeef\n")  # 旧指纹
            if cmd[:2] == ["docker", "exec"]:
                return _FakeCompleted(returncode=0, stdout="ok")
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(executor_module.subprocess, "run", fake_run)
        try:
            SandboxTools()._execute_sandbox_impl("python x.py")
        finally:
            global_config["sandbox"]["docker"]["requirements_path"] = ""
        assert any(c[:2] == ["docker", "rm"] for c in calls), "旧指纹容器未被销毁"
        assert any(c[:2] == ["docker", "run"] for c in calls), "未按新指纹重建容器"


class TestManagerPromptAnchor:
    def test_plan_log_anchor_format(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        with open(os.path.join(repo_root, "prompt", "manager.md"), encoding="utf-8") as f:
            content = f.read()
        assert "## Plan[编号]: [方案名称]" in content
        assert "锚点" in content
