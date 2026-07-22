"""W3-2：Manager 归档核验门禁（run_verification 逻辑 + commit 前置校验）。"""
import json
import subprocess
import sys

import pytest

from tests.conftest import global_config
from tools.code.verify import (
    _SANDBOX_VERIFY_SCRIPT,
    evaluate_assertions,
    parse_assertion,
    verification_registry,
)
from tools.execute.execute_git import ExecuteGitTool


FAKE_METRICS = {
    "overall": {"F1": 0.9549, "recall": 0.978},
    "per_app": {
        "王者荣耀": {"F1": 0.9023, "precision": 0.88},
        "bilibili": {"F1": 0.9494},
    },
}


@pytest.fixture(autouse=True)
def _reset_registry():
    verification_registry.reset()
    yield
    verification_registry.reset()


class TestParseAssertion:
    def test_chinese_path(self):
        assert parse_assertion("per_app.王者荣耀.F1 >= 0.75") == ("per_app.王者荣耀.F1", ">=", 0.75)

    def test_operators(self):
        for op in [">=", "<=", ">", "<", "==", "!="]:
            path, parsed_op, val = parse_assertion(f"overall.F1 {op} 0.9")
            assert (path, parsed_op, val) == ("overall.F1", op, 0.9)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_assertion("F1 大约等于 0.9")


class TestEvaluateAssertions:
    def test_all_pass(self):
        report = evaluate_assertions(FAKE_METRICS, [
            "per_app.王者荣耀.F1 >= 0.75",
            "overall.F1 >= 0.935",
            "overall.recall >= 0.97",
        ])
        assert report["all_passed"] is True
        assert all(r["passed"] for r in report["results"])

    def test_failure_detected(self):
        report = evaluate_assertions(FAKE_METRICS, [
            "per_app.王者荣耀.F1 >= 0.75",
            "overall.F1 >= 0.99",  # 0.9549 < 0.99，应判失败
        ])
        assert report["all_passed"] is False
        failed = [r for r in report["results"] if not r["passed"]]
        assert len(failed) == 1 and failed[0]["actual"] == 0.9549

    def test_missing_path_is_failure(self):
        report = evaluate_assertions(FAKE_METRICS, ["per_app.不存在.F1 >= 0.5"])
        assert report["all_passed"] is False
        assert report["results"][0]["error"]

    def test_non_numeric_value_is_failure(self):
        report = evaluate_assertions({"a": {"b": "高分"}}, ["a.b >= 0.5"])
        assert report["all_passed"] is False
        assert "不是数值" in report["results"][0]["error"]


class TestSandboxScript:
    """内嵌沙箱脚本本身的端到端测试（本机 python3 代行，无 docker 环境）。"""

    def _run_script(self, metrics_path, assertions):
        return subprocess.run(
            [sys.executable, "-", str(metrics_path), json.dumps(assertions, ensure_ascii=False)],
            input=_SANDBOX_VERIFY_SCRIPT,
            capture_output=True, text=True, timeout=30,
        )

    def test_script_pass_and_fail(self, tmp_path):
        metrics_file = tmp_path / "result_stats.json"
        metrics_file.write_text(json.dumps(FAKE_METRICS, ensure_ascii=False), encoding="utf-8")

        res = self._run_script(metrics_file, ["per_app.王者荣耀.F1 >= 0.75", "overall.F1 >= 0.935"])
        assert res.returncode == 0
        report = json.loads(res.stdout.strip().splitlines()[-1])
        assert report["all_passed"] is True

        res = self._run_script(metrics_file, ["overall.F1 >= 0.99"])
        report = json.loads(res.stdout.strip().splitlines()[-1])
        assert report["all_passed"] is False
        assert report["results"][0]["passed"] is False

    def test_script_missing_file_reports_fatal(self, tmp_path):
        res = self._run_script(tmp_path / "nope.json", ["a.b >= 1"])
        report = json.loads(res.stdout.strip().splitlines()[-1])
        assert report["fatal"]


class TestCommitGate:
    """commit 前置校验：真实 temp git 仓库端到端。"""

    @pytest.fixture()
    def repo(self, tmp_path):
        global_config["workspace"]["root_dir"] = str(tmp_path)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
        subprocess.run(["git", "checkout", "-qb", "plan1-test"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("2", encoding="utf-8")
        return tmp_path

    def _record_verification(self, all_passed: bool):
        verification_registry.record({
            "all_passed": all_passed,
            "results": [{"assertion": "x >= 1", "actual": 1, "expected": 1,
                         "op": ">=", "passed": all_passed, "error": None}],
        })

    def test_commit_blocked_without_verification(self, repo):
        tool = ExecuteGitTool()
        tool._execute_git_command_impl("add .")
        res = tool._execute_git_command_impl("commit -m 'feat: x'")
        assert "归档门禁拦截" in res
        # 确认确实没有产生提交
        log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                             capture_output=True, text=True).stdout
        assert "feat: x" not in log

    def test_commit_allowed_after_passed_verification(self, repo):
        tool = ExecuteGitTool()
        self._record_verification(all_passed=True)
        tool._execute_git_command_impl("add .")
        res = tool._execute_git_command_impl("commit -m 'feat: x'")
        assert not res.startswith("❌")

    def test_passed_verification_consumed_by_one_commit(self, repo):
        tool = ExecuteGitTool()
        self._record_verification(all_passed=True)
        tool._execute_git_command_impl("add .")
        assert not tool._execute_git_command_impl("commit -m 'feat: x'").startswith("❌")
        # 许可已消费：第二次 commit 必须重新核验
        (repo / "a.txt").write_text("3", encoding="utf-8")
        tool._execute_git_command_impl("add .")
        res = tool._execute_git_command_impl("commit -m 'feat: y'")
        assert "归档门禁拦截" in res

    def test_failed_verification_blocks_commit(self, repo):
        tool = ExecuteGitTool()
        self._record_verification(all_passed=False)
        tool._execute_git_command_impl("add .")
        res = tool._execute_git_command_impl("commit -m 'feat: x'")
        assert "归档门禁拦截" in res

    def test_circuit_breaker_commit_exempt(self, repo):
        tool = ExecuteGitTool()
        tool._execute_git_command_impl("add .")
        res = tool._execute_git_command_impl("commit -m '存档：plan1 熔断失败现场'")
        assert not res.startswith("❌")
