import shlex
import subprocess

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from utils.config import global_config


EXECUTE_SAFE_BASH_DESC = """安全执行 Bash 命令，用于侦查项目目录、查看文件内容或排查环境依赖。
如果需要读取.py文件，优先使用 get_file_skeleton 或 read_code_block 工具，不要使用cat，head，tail。

【安全限制】
1. 这是一个纯只读工具。你只能使用以下基础命令: pwd, ls, cat, head, tail, glob, grep, wc, tree, printenv, find。
2. 支持使用管道符 '|' (例如 'cat logs.txt | grep Error')，但严禁使用其他任何重定向或拼接符。"""

class ExecuteSafeBashSchema(BaseModel):
    """Input schema for `execute_safe_bash` tool"""
    command: str = Field(description="需要执行的 bash 语句")


class ExecuteBashTool:
    def __init__(self) -> None:
        self.READ_ONLY_WHITELIST = [
            "pwd", "ls", "cat", "head", "tail", "glob", "grep", "wc", "tree", "printenv", "find"
        ]
        self.DANGEROUS_CHARS = [
            ";", "&", ">", "<", "$", "`", "\n", "\r", "-exec", "-execdir", "-ok", "-okdir", "-delete"
        ]

    def _execute_safe_bash_impl(self, command: str) -> str:
        """
        安全执行 Bash 命令，用于侦查项目目录、读取文件内容或排查环境依赖。

        【安全限制】
        1. 这是一个纯只读工具。你只能使用以下基础命令: pwd, ls, cat, head, tail, glob, grep, wc, tree, printenv, find。
        2. 支持使用管道符 "|" (例如 "cat logs.txt | grep Error")，但严禁使用其他任何重定向或拼接符。

        参数:
        - command: 需要执行的 bash 语句
        """
        command = command.strip()
        if not command:
            return "❌ 错误：命令不能为空。"
        if "/memories" in command:
            return "❌ 错误：无法作用于 /memories/ 路径"

        # 高危符号扫描
        for char in self.DANGEROUS_CHARS:
            if char in command:
                return f"❌ 安全拦截：命令包含被禁止的高危符号或参数 ('{char}')。本工具仅允许纯读取和 '|' 管道操作。"

        # 分段解析与白名单校验
        pipeline_stages = command.split('|')

        for stage in pipeline_stages:
            stage = stage.strip()
            if not stage:
                continue
            try:
                cmd_parts = shlex.split(stage)
            except ValueError as e:
                return f"❌ 语法错误：命令片段 '{stage}' 解析失败 ({str(e)})。"
            if not cmd_parts:
                continue

            base_exe = cmd_parts[0]
            # 拦截基础命令越权
            if base_exe not in self.READ_ONLY_WHITELIST:
                return (
                    f"❌ 越权拦截：严禁使用 '{base_exe}' 命令。\n"
                    f"你的全局只读白名单为: {', '.join(self.READ_ONLY_WHITELIST)}"
                )

        # 执行
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[标准错误输出 (stderr)]:\n{result.stderr}"
            if result.returncode != 0:
                final_output = f"⚠️ 命令执行失败 (退出码 {result.returncode}):\n{output.strip()}"
            else:
                final_output = output.strip() if output.strip() else "✅ (命令执行成功，无终端输出)"
            max_chars = global_config.get("agent", {}).get("max_chars")
            if max_chars and len(final_output) > max_chars:
                warning_msg = (
                    f"\n\n...[警告：输出过长已被强制截断！原输出长达 {len(final_output)} 字符]...\n"
                    f"请使用 grep 提取关键字，或使用 head/tail -n 100 限制行数后再试！"
                )
                return final_output[:max_chars] + warning_msg
            return final_output
        except subprocess.TimeoutExpired:
            return "❌ 执行超时：命令运行超过 10 秒被强制终止。请尝试缩小检索范围。"
        except Exception as e:
            return f"❌ 系统执行异常: {str(e)}"

    def create_execute_safe_bash_tool(self):
        return StructuredTool.from_function(
            name="execute_safe_bash",
            description=EXECUTE_SAFE_BASH_DESC,
            func=self._execute_safe_bash_impl,
            infer_schema=False,
            args_schema=ExecuteSafeBashSchema,
        )
