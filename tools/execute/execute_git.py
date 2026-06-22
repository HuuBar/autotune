import os
import shlex
import subprocess

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from utils.config import global_config


EXECUTE_GIT_COMMAND_DESC = """执行安全的 Git 版本管理操作。请直接传入 Git 参数，无需包含 'git' 前缀。
示例: "checkout -b plan1", "add .", "commit -m 'feat(task1): description'"。
特殊扩展命令: "diff-export master --output path/to/diff.diff"。"""

class ExecuteGitCommandSchema(BaseModel):
    """Input schema for `execute_git_command` tool"""
    command: str = Field(description="需要执行的 git 后续命令及参数字符串")


class ExecuteGitTool:
    def __init__(self) -> None:
        # Git 指令白名单
        self.ALLOWED_COMMANDS = ["branch", "checkout", "add", "commit", "diff", "log", "reset", "status"]
        # 危险写操作名单：触发这些操作时，必须检查是否在 master 分支
        self.DANGEROUS_WRITE_COMMANDS = ["add", "commit", "reset"]

    def _get_base_git_cmd(self) -> list:
        """获取标准化的 git 基础命令，处理中文和特殊格式"""
        return [
            "git",
            "--no-pager",                       # 禁用分页器，防止卡死
            "-c", "core.quotepath=false",       # 强制显示中文路径，不进行八进制转义
            "-c", "color.ui=false",             # 禁用颜色输出，防止 LLM 读到 ANSI 转义乱码
            "-c", "i18n.commitEncoding=utf-8",  # 强制提交编码为 utf-8
            "-c", "i18n.logOutputEncoding=utf-8"# 强制 log 输出为 utf-8
        ]

    def _get_current_branch(self, cwd: str) -> str:
        """获取指定目录下的当前 Git 分支名"""
        try:
            res = subprocess.run(
                self._get_base_git_cmd() + ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            return res.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def _run_git_command(self, args: list, cwd: str) -> str:
        """执行底层 Git 命令并捕获输出"""
        try:
            result = subprocess.run(
                self._get_base_git_cmd() + args,
                cwd=cwd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            out = result.stdout.strip()
            return out if out else f"✅ `git {' '.join(args)}` 执行成功 (无终端输出)"
        except subprocess.CalledProcessError as e:
            return f"❌ Git 执行失败 (退出码 {e.returncode}):\n{e.stderr.strip()}"

    def _handle_diff_export(self, args: list, cwd: str) -> None:
        """
        处理自定义的 diff-export 命令: 导出方案对比的物理文件
        期望格式: diff-export master --output path/to/diff.diff
        """
        if len(args) != 3 or args[1] != "--output":
            return "❌ 用法错误: diff-export <base_branch> --output <filepath>"

        base_branch = args[0]
        output_path = args[2]
        if not os.path.isabs(output_path):
            output_path = os.path.join(cwd, output_path)

        try:
            diff_result = subprocess.run(
                self._get_base_git_cmd() + ["diff", f"{base_branch}...HEAD"],
                cwd=cwd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            # 写入diff文件
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(diff_result.stdout)
            return f"✅ 全局 Diff 已成功导出至: {output_path}"
        except subprocess.CalledProcessError as e:
            return f"❌ 导出 Diff 失败:\n{e.stderr.strip()}"

    def _execute_git_command_impl(self, command: str) -> str:
        """
        执行安全的 Git 版本管理操作。请直接传入 Git 参数，无需包含 'git' 前缀。
        示例: "checkout -b plan1", "add .", "commit -m 'feat(task1): description'"
        特殊扩展命令: "diff-export master --output path/to/diff.diff"

        参数:
        - command: 需要执行的 git 后续命令及参数字符串。
        """
        args = command.strip()
        if not args:
            return "❌ 错误：参数不能为空"

        project_root = os.path.abspath(global_config.get("workspace", {}).get("root_dir", ""))
        # 解析命令字符串为列表
        try:
            parsed_args = shlex.split(args)
        except ValueError as e:
            return f"❌ 命令解析错误 (请检查引号是否配对闭合): {str(e)}"
        if not parsed_args:
            return "❌ 错误：未提供有效命令"

        cmd = parsed_args[0]
        cmd_args = parsed_args[1:]

        if cmd == "diff-export":
            return self._handle_diff_export(cmd_args, project_root)

        if cmd not in self.ALLOWED_COMMANDS:
            return f"❌ 越权拦截：命令 'git {cmd}' 不在安全白名单内！禁止执行任何网络同步或未授权操作。"

        if cmd in self.DANGEROUS_WRITE_COMMANDS:
            if self._get_current_branch(project_root) == "master":
                return f"❌ 保护拦截：禁止在 'master' 分支直接执行 {cmd}！请先 checkout -b 创建独立的方案分支。"

        return self._run_git_command(parsed_args, project_root)

    def create_execute_git_command_tool(self):
        return StructuredTool.from_function(
            name="execute_git_command",
            description=EXECUTE_GIT_COMMAND_DESC,
            func=self._execute_git_command_impl,
            infer_schema=False,
            args_schema=ExecuteGitCommandSchema,
        )
