"""
Manager 归档核验门禁（W3-2）

背景事故：commit 的触发条件曾是"Developer 汇报通过"，Manager 无独立核验手段，
出现过"说了查指标却直接 git add"的归档事故。

本模块提供：
1. `run_verification` 工具：在沙箱内用确定性脚本解析指标 JSON，与断言
   （如 `per_app.王者荣耀.F1 >= 0.75`）逐条比对，返回结构化通过/失败表；
2. `verification_registry`：进程内核验状态注册表，供 execute_git 的 commit
   前置校验查询——最近一次核验必须全部通过且尚未被一次成功 commit 消费。
"""
import json
import operator
import os
import re
import shlex
from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from tools.code.executor import SandboxTools
from utils.config import global_config
from utils.file_helper import get_secure_path
from utils.logger import logger


# ---------------------------------------------------------------------------
# 断言解析与求值（与沙箱脚本 _SANDBOX_VERIFY_SCRIPT 的逻辑保持一致）
# ---------------------------------------------------------------------------

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

# 点分路径（允许中文/字母/数字/下划线/连字符）+ 比较运算符 + 数值
_ASSERTION_RE = re.compile(
    r"^\s*([0-9A-Za-z_\-一-鿿]+(?:\.[0-9A-Za-z_\-一-鿿]+)*)\s*"
    r"(>=|<=|==|!=|>|<)\s*"
    r"(-?\d+(?:\.\d+)?)\s*$"
)


def parse_assertion(text: str) -> tuple[str, str, float]:
    """把 'per_app.王者荣耀.F1 >= 0.75' 解析为 (路径, 运算符, 期望值)。"""
    m = _ASSERTION_RE.match(text or "")
    if not m:
        raise ValueError(f"无法解析的断言: {text!r}（期望格式: 点分路径 运算符 数值，如 per_app.王者荣耀.F1 >= 0.75）")
    return m.group(1), m.group(2), float(m.group(3))


def get_by_path(metrics: dict, dotted_path: str) -> Any:
    """按点分路径从嵌套 dict 取值，缺失时抛 KeyError。"""
    node: Any = metrics
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(dotted_path)
        node = node[key]
    return node


def evaluate_assertions(metrics: dict, assertions: list[str]) -> dict:
    """
    确定性求值一组断言，返回结构化报告：
    {"all_passed": bool, "results": [{assertion, actual, expected, op, passed, error}]}
    """
    results = []
    for text in assertions:
        item = {"assertion": text, "actual": None, "expected": None,
                "op": None, "passed": False, "error": None}
        try:
            path, op, expected = parse_assertion(text)
            actual = get_by_path(metrics, path)
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                raise TypeError(f"路径 {path} 的值不是数值: {actual!r}")
            item.update({
                "actual": actual,
                "expected": expected,
                "op": op,
                "passed": bool(_OPS[op](actual, expected)),
            })
        except (ValueError, KeyError, TypeError) as e:
            item["error"] = str(e)
        results.append(item)
    return {
        "all_passed": bool(results) and all(r["passed"] for r in results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# 核验状态注册表（execute_git 的 commit 门禁查询）
# ---------------------------------------------------------------------------

class VerificationRegistry:
    """进程内单例：记录最近一次核验结果，控制 commit 放行。"""

    def __init__(self) -> None:
        self._last_report: Optional[dict] = None
        self._consumed: bool = True  # 通过核验后只允许消费一次 commit

    def record(self, report: dict) -> None:
        self._last_report = report
        # 只有全部断言通过才生成一枚"待消费"的 commit 许可
        self._consumed = not report.get("all_passed", False)

    def is_commit_allowed(self) -> bool:
        return (
            self._last_report is not None
            and self._last_report.get("all_passed", False)
            and not self._consumed
        )

    def consume(self) -> None:
        """一次成功的 commit 消费掉许可，下个 Plan 必须重新核验。"""
        self._consumed = True

    def reset(self) -> None:
        self._last_report = None
        self._consumed = True

    def status_hint(self) -> str:
        if self._last_report is None:
            return "从未执行过 run_verification"
        if not self._last_report.get("all_passed", False):
            return "最近一次 run_verification 存在未通过断言"
        if self._consumed:
            return "最近一次通过的核验已被一次 commit 消费"
        return "核验通过"


verification_registry = VerificationRegistry()


# ---------------------------------------------------------------------------
# 沙箱内执行的确定性核验脚本（纯标准库，经 docker exec stdin 送入容器）
# 注意：与 evaluate_assertions 逻辑保持一致；独立成脚本是因为容器内没有本仓库代码。
# 约定：脚本始终以退出码 0 结束，结果（含硬性错误）编码为最后一行的 JSON。
# ---------------------------------------------------------------------------

_SANDBOX_VERIFY_SCRIPT = r'''
import json, operator, re, sys

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt,
       "<": operator.lt, "==": operator.eq, "!=": operator.ne}
PATTERN = re.compile(
    r"^\s*([0-9A-Za-z_\-一-鿿]+(?:\.[0-9A-Za-z_\-一-鿿]+)*)\s*"
    r"(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")

def get_by_path(metrics, dotted):
    node = metrics
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(dotted)
        node = node[key]
    return node

def main():
    metrics_path, assertions_json = sys.argv[1], sys.argv[2]
    report = {"all_passed": False, "results": [], "fatal": None}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assertions = json.loads(assertions_json)
        if not isinstance(assertions, list) or not assertions:
            raise ValueError("assertions 必须是非空列表")
        for text in assertions:
            item = {"assertion": text, "actual": None, "expected": None,
                    "op": None, "passed": False, "error": None}
            try:
                m = PATTERN.match(text or "")
                if not m:
                    raise ValueError("无法解析的断言: %r" % (text,))
                path, op, expected = m.group(1), m.group(2), float(m.group(3))
                actual = get_by_path(metrics, path)
                if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                    raise TypeError("路径 %s 的值不是数值: %r" % (path, actual))
                item.update({"actual": actual, "expected": expected, "op": op,
                             "passed": bool(OPS[op](actual, expected))})
            except (ValueError, KeyError, TypeError) as e:
                item["error"] = str(e)
            report["results"].append(item)
        report["all_passed"] = all(r["passed"] for r in report["results"])
    except Exception as e:
        report["fatal"] = "%s: %s" % (type(e).__name__, e)
    print(json.dumps(report, ensure_ascii=False))

main()
'''


# ---------------------------------------------------------------------------
# run_verification 工具
# ---------------------------------------------------------------------------

RUN_VERIFICATION_DESC = """以确定性方式核验指标文件（JSON）是否满足验收断言，杜绝"凭汇报归档"。

【使用时机】：在结局归档（git add / commit）之前必须调用本工具，且全部断言通过才会放行 commit。
【参数说明】：
- metrics_file: 指标 JSON 文件的项目相对路径（如 outputs/xxx/result_stats.json）。
- assertions: 断言列表，每条格式为 `点分路径 运算符 数值`，运算符支持 >= <= > < == !=。
  示例: ["per_app.王者荣耀.F1 >= 0.75", "overall.F1 >= 0.935", "overall.recall >= 0.97"]
【返回】：逐条断言的 实际值/期望值/通过与否 对照表。断言全部通过 = true 时，才可执行 commit。"""


class RunVerificationSchema(BaseModel):
    """Input schema for `run_verification` tool"""
    metrics_file: str = Field(description="指标 JSON 文件的项目相对路径")
    assertions: list[str] = Field(description="断言列表，每条格式: 点分路径 运算符 数值")


class VerificationTools:
    def __init__(self) -> None:
        self.project_root = global_config.get("workspace", {}).get("root_dir", "")

    def _format_report(self, report: dict, metrics_file: str) -> str:
        """把结构化报告格式化为给 Manager 阅读的对照表。"""
        if report.get("fatal"):
            return (
                f"❌ 核验执行失败：{report['fatal']}\n"
                f"请确认指标文件路径正确且为合法 JSON：{metrics_file}"
            )
        lines = [f"指标核验表（文件: {metrics_file}）", "-" * 60]
        for r in report["results"]:
            if r["error"]:
                lines.append(f"❌ {r['assertion']}  | 求值失败: {r['error']}")
            else:
                mark = "✅" if r["passed"] else "❌"
                lines.append(
                    f"{mark} {r['assertion']}  | 实际值 {r['actual']} "
                    f"{'满足' if r['passed'] else '不满足'} {r['op']} {r['expected']}"
                )
        lines.append("-" * 60)
        if report["all_passed"]:
            lines.append("结论: 全部断言通过 ✅ 可以执行 git add / commit 归档。")
        else:
            lines.append("结论: 存在未通过断言 ❌ commit 将被门禁拦截，请先回到执行流程修复或按熔断流程处理。")
        return "\n".join(lines)

    def _run_verification_impl(self, metrics_file: str, assertions: list[str]) -> str:
        # 1. 安全路径解析 + 容器内路径换算
        try:
            abs_path = get_secure_path(self.project_root, metrics_file)
        except PermissionError as e:
            return f"❌ 安全错误: {str(e)}"
        if not os.path.isfile(abs_path):
            return f"❌ 错误: 指标文件不存在 ({metrics_file})"
        sandbox_cfg = global_config.get("sandbox", {}).get("docker", {})
        workdir = sandbox_cfg.get("workdir", "/workspace")
        rel_path = os.path.relpath(abs_path, os.path.abspath(self.project_root))
        container_metrics = f"{workdir.rstrip('/')}/{rel_path.replace(os.sep, '/')}"

        # 2. 在沙箱内执行确定性核验脚本（stdin 送入，无需往容器拷贝文件）
        assertions_json = json.dumps(assertions, ensure_ascii=False)
        command = (
            f"python3 - {shlex.quote(container_metrics)} {shlex.quote(assertions_json)} "
            f"<<'__VERIFY_EOF__'\n{_SANDBOX_VERIFY_SCRIPT}\n__VERIFY_EOF__"
        )
        raw_output = SandboxTools()._execute_sandbox_impl(command)

        # 3. 解析报告（脚本约定最后一行为 JSON；沙箱层错误直接透传，不记入注册表）
        report = None
        for line in reversed(raw_output.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict) and "results" in candidate:
                        report = candidate
                        break
                except json.JSONDecodeError:
                    continue
        if report is None:
            return (
                f"❌ 核验工具执行异常，未能取得结构化报告（本次不登记核验状态）。原始输出：\n{raw_output}"
            )

        # 4. 登记核验状态（commit 门禁依据）并格式化返回
        if not report.get("fatal"):
            verification_registry.record(report)
            logger.info(f"[run_verification] {metrics_file} all_passed={report['all_passed']}")
        return self._format_report(report, metrics_file)

    def create_run_verification_tool(self):
        return StructuredTool.from_function(
            name="run_verification",
            description=RUN_VERIFICATION_DESC,
            func=self._run_verification_impl,
            infer_schema=False,
            args_schema=RunVerificationSchema,
        )
