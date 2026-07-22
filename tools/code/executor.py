import os
import subprocess

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from utils.config import global_config
from utils.docker_image import resolve_sandbox_image


# execute_sandbox tool
EXECUTE_SANDBOX_DESC = """在高度隔离的沙箱中执行 Python 代码或 pip 命令。
注意：此工具仅用于执行代码并获取运行结果，修改代码请使用代码编辑工具。

【路径规则】:沙箱中已将项目的根目录挂载到 {workspace}。
因此 command 中的所有脚本路径必须以 {workspace} 开头，或使用相对路径。

【失败计数】:工具会按命令签名统计连续失败次数并在结果中附带 [连续失败 N/3] 标记。达到上限后请停止重复尝试同一命令。"""

class ExecuteSandboxSchema(BaseModel):
    """Input schema for `execute_sandbox` tool"""
    command: str = Field(description="需要执行的完整 bash 命令 (例如: \"python tests/test.py\")")

# reset_sandbox tool
RESET_SANDBOX_DESC = """强制重置当前的沙箱环境。此工具应用于新方案开始前的环境归零。"""


class SandboxTools:
    def __init__(self) -> None:
        # W3-4：按命令签名统计连续失败次数（替代模型数对话历史的脆弱做法）
        self._fail_counts: dict[str, int] = {}
        self._fail_mark_threshold = global_config.get("agent", {}).get("sandbox_fail_mark_threshold", 3)

    @staticmethod
    def _command_signature(command: str) -> str:
        """命令签名：归一化空白字符后的完整命令串。"""
        return " ".join(command.split())

    def _mark_failure(self, signature: str) -> str:
        """累计一次失败并返回附加给返回结果的标记文本。"""
        count = self._fail_counts.get(signature, 0) + 1
        self._fail_counts[signature] = count
        mark = f"\n[连续失败 {count}/{self._fail_mark_threshold}]"
        if count >= self._fail_mark_threshold:
            mark += " 同一命令已达重试上限：请停止重复尝试，改用不同命令/方案验证，或向 Manager 汇报请求 Debugger 支援。"
        return mark

    def _clear_failure(self, signature: str) -> None:
        self._fail_counts.pop(signature, None)

    def _execute_sandbox_impl(self, command: str) -> str:
        """
        在高度隔离的沙箱中执行 Python 代码或 pip 命令。
        注意：此工具仅用于执行代码并获取运行结果，修改代码请使用代码编辑工具。

        【路径规则】:沙箱中已将项目的根目录挂载到 /workspace。
        因此 command 中的所有脚本路径必须以 /workspace 开头，或使用相对路径。

        参数:
        - command: 需要执行的完整 bash 命令 (例如: "python tests/test.py")
        """
        command = command.strip()
        if not command:
            return "❌ 错误：命令不能为空。"

        # 1. 动态读取 Config 参数
        sandbox_cfg = global_config.get("sandbox", {}).get("docker", {})
        image_id = sandbox_cfg.get("name", "autotune-env")
        # tag 混入 requirements.txt 的 md5 前 8 位，与 warmup 构建的镜像保持一致
        target_image = resolve_sandbox_image(sandbox_cfg)
        container_name = f"{image_id}-runner"

        workdir = sandbox_cfg.get("workdir", "/workspace")
        memory_limit = sandbox_cfg.get("memory", "4g")
        cpus = str(sandbox_cfg.get("cpus", "4.0"))
        network_mode = sandbox_cfg.get("network", "bridge")
        timeout = sandbox_cfg.get("timeout", 300)
        host_root_dir = os.path.abspath(global_config.get("workspace", {}).get("root_dir", ""))

        # 2. 检查常驻后台容器是否存活（且镜像指纹与当前依赖一致，否则重建容器）
        check_cmd = ["docker", "ps", "-q", "-f", f"name=^{container_name}$"]
        try:
            res = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
            container_alive = bool(res.stdout.strip())
            if container_alive:
                inspect = subprocess.run(
                    ["docker", "inspect", "-f", "{{.Config.Image}}", container_name],
                    capture_output=True, text=True, check=False,
                )
                if inspect.stdout.strip() != target_image:
                    # 依赖已变更（tag 不同），旧容器环境过期，强制重建
                    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                    container_alive = False
            if not container_alive:
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                run_cmd = [
                    "docker", "run", "-d",
                    f"--name={container_name}",
                    f"--memory={memory_limit}",
                    f"--cpus={cpus}",
                    f"--network={network_mode}",
                    "-e", "PYTHONUNBUFFERED=1",
                    "-v", f"{host_root_dir}:{workdir}",
                    "-w", workdir,
                    target_image,
                    "tail", "-f", "/dev/null"
                ]
                subprocess.run(run_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return f"❌ 致命错误：沙箱守护进程启动失败。\n{e.stderr if e.stderr else str(e)}"
        except Exception as e:
            return f"❌ 沙箱启动异常: {str(e)}"

        # 3. 组装执行指令
        signature = self._command_signature(command)
        exec_cmd = [
            "docker", "exec",
            container_name,
            "bash", "-c", command
        ]

        # 4. 执行
        try:
            result = subprocess.run(
                exec_cmd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout
            )
            output = result.stdout

            if result.returncode != 0:
                max_chars = global_config.get("agent", {}).get("max_chars")
                truncated_out = output[-max_chars:] if (max_chars and len(output) > max_chars) else output
                final_report = (
                    f"⚠️ 运行失败 (退出码 {result.returncode})\n"
                    f"======== 原始报错尾部 ========\n"
                    f"{'(日志过长已截断) ... ' if (max_chars and len(output) > max_chars) else ''}"
                    f"{truncated_out}"
                )
                if "No such file" in truncated_out:
                    final_report += f"提示：{host_root_dir} 目录已被替换为 {workdir}"
                final_report += self._mark_failure(signature)
                return final_report
            self._clear_failure(signature)
            return output.strip() if output.strip() else "✅ (沙箱执行成功，无终端输出)"
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            return (
                f"❌ 运行超时：命令执行超过 {timeout} 秒，沙箱已被强制销毁。可能存在死循环。"
                f"{self._mark_failure(signature)}"
            )
        except Exception as e:
            return f"❌ 沙箱调用异常: {str(e)}"

    def _reset_sandbox_impl(self) -> str:
        """
        强制重置当前的沙箱环境。此工具应用于新方案开始前的环境归零。
        """
        sandbox_cfg = global_config.get("sandbox", {}).get("docker", {})
        image_id = sandbox_cfg.get("name", "autotune-env")
        container_name = f"{image_id}-runner"
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return "✅ 成功：沙箱环境已重置。"
            else:
                if "No such container" in result.stderr:
                    return "✅ 成功：沙箱环境本就处于纯净状态，无需重置。"
                return f"⚠️ 警告：重置指令已发送，但发生异常: {result.stderr}"
        except Exception as e:
            return f"❌ 错误：沙箱重置失败: {str(e)}"

    def create_execute_sandbox_tool(self):
        sandbox_cfg = global_config.get("sandbox", {}).get("docker", {})
        return StructuredTool.from_function(
            name="execute_sandbox",
            description=EXECUTE_SANDBOX_DESC.format(
                workspace=sandbox_cfg.get("workdir", "/workspace")
            ),
            func=self._execute_sandbox_impl,
            infer_schema=False,
            args_schema=ExecuteSandboxSchema,
        )

    def create_reset_sandbox_tool(self):
        return StructuredTool.from_function(
            name="reset_sandbox",
            description=RESET_SANDBOX_DESC,
            func=self._reset_sandbox_impl,
            infer_schema=True,
        )
