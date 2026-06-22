import os

from agent.agent import AutoTuneAgent
from tools.code.executor import SandboxTools
from utils.config import global_config


def sandbox_execute(command: str):
    execute_sandbox = SandboxTools().create_execute_sandbox_tool()
    exe_res = execute_sandbox.invoke({
        "command": command
    })
    print(exe_res)

def sandbox_reset():
    reset_sandbox = SandboxTools().create_reset_sandbox_tool()
    reset_res = reset_sandbox.invoke({})
    print(reset_res)


if __name__ == '__main__':
    # 自定义执行命令
    command = "python cluster/run.py"

    # 初始化 Dokcer
    AutoTuneAgent().warmup_docker_env()

    # 执行工具
    sandbox_execute(command)
    sandbox_reset()
