"""
pytest 全局基础设施。

说明：仓库自带的 config.yaml（GitHub 版）第 70 行存在引号未闭合的 YAML 语法错误，
导致 utils.config 在导入时即崩溃。测试不应依赖真实服务器配置，因此这里用 stub
模块替换 utils.config，global_config 为一个可变的普通 dict，各测试可按需覆写
workspace.root_dir 等键。
"""
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 既有手测脚本（非 pytest 用例，含硬编码服务器路径），全量运行时跳过
collect_ignore = ["unit/tools/code_test.py"]


def _install_config_stub() -> dict:
    stub = types.ModuleType("utils.config")
    stub.global_config = {
        "workspace": {
            "root_dir": "",
            "agents_md": "",
            "git_diff_dir": "git_patch",
            "structure_filepath": "",
            "grep_allowed_suffix": ["py", "ipynb", "json", "yaml"],
        },
        "skills": {"planner_skill_path": []},
        "sandbox": {
            "docker": {
                "name": "autotune-env",
                "image": "python:3.12-slim",
                "workdir": "/workspace",
                "cpus": "4.0",
                "memory": "4g",
                "network": "bridge",
                "timeout": 300,
                "requirements_path": "",
            }
        },
        "agent": {
            "max_chars": 5000,
            "max_retries": 5,
            "max_plan_num": 10,
            "recursion_limit": 30000,
        },
        "llm": {
            "dsv4": {
                "url": "http://127.0.0.1:1/v1",
                "api_key": "test",
                "model_name": "dsv4",
                "temperature": 0.7,
            }
        },
    }
    sys.modules["utils.config"] = stub
    return stub.global_config


# 在任何 autotune 模块被导入前安装 stub
global_config = _install_config_stub()
