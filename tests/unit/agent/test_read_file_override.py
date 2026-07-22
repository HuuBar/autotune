"""W3-3：read_file 工具描述按 agent 生效 + planner.md 绝对路径输出协议。"""
import os
import re

import pytest
from langgraph.prebuilt.tool_node import ToolNode

from agent.factory import (
    DEVELOPER_READ_FILE_DESCRIPTION,
    READ_FILE_REPLACED_DESCRIPTION,
    BaseAgentBuilder,
    DeveloperAgentBuilder,
    ManagerAgentBuilder,
    PlannerAgentBuilder,
    ReadFileDescriptionOverrideMiddleware,
)
from deepagents.middleware.filesystem import FilesystemMiddleware

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _make_builder(cls):
    return cls(llm_name="dsv4", store=None, checkpointer=None)


class TestOverrideMiddleware:
    def test_only_injects_read_file_with_custom_description(self):
        mw = ReadFileDescriptionOverrideMiddleware(read_file_description="定制描述XYZ")
        assert [t.name for t in mw.tools] == ["read_file"]
        assert mw.tools[0].description == "定制描述XYZ"

    def test_per_agent_descriptions_are_independent(self):
        mw_a = ReadFileDescriptionOverrideMiddleware(read_file_description="AgentA 描述")
        mw_b = ReadFileDescriptionOverrideMiddleware(read_file_description="AgentB 描述")
        assert mw_a.tools[0].description != mw_b.tools[0].description

    def test_last_wins_against_builtin_filesystem_middleware(self):
        """模拟真实装配顺序：内置 FS 中间件在前，覆盖中间件在后 → 定制描述生效。"""
        builtin = FilesystemMiddleware()  # 内置栈（默认描述）
        override = ReadFileDescriptionOverrideMiddleware(read_file_description="定制描述XYZ")
        ordered_tools = list(builtin.tools) + list(override.tools)  # 中间件链顺序
        node = ToolNode(ordered_tools)
        assert node.tools_by_name["read_file"].description == "定制描述XYZ"
        # 其余文件系统工具不受影响（仍来自内置栈）
        for name in ["write_file", "edit_file", "ls", "glob", "grep"]:
            assert node.tools_by_name[name].description != "定制描述XYZ"


class TestPerAgentDescription:
    def test_developer_gets_custom_description(self):
        builder = _make_builder(DeveloperAgentBuilder)
        assert builder.get_read_file_description() == DEVELOPER_READ_FILE_DESCRIPTION
        assert "不能" in builder.get_read_file_description()  # 明确禁止读项目代码

    @pytest.mark.parametrize("cls", [ManagerAgentBuilder, PlannerAgentBuilder])
    def test_other_agents_keep_default(self, cls):
        builder = _make_builder(cls)
        assert builder.get_read_file_description() == READ_FILE_REPLACED_DESCRIPTION

    def test_middleware_chain_contains_override(self):
        builder = _make_builder(DeveloperAgentBuilder)
        middleware = builder._get_middleware()
        overrides = [m for m in middleware if isinstance(m, ReadFileDescriptionOverrideMiddleware)]
        assert len(overrides) == 1
        assert overrides[0].tools[0].description == DEVELOPER_READ_FILE_DESCRIPTION


class TestPlannerPromptAbsolutePath:
    """输出协议"相关文件"字段必须要求绝对路径（静态校验 planner.md）。"""

    @pytest.fixture()
    def planner_prompt(self):
        with open(os.path.join(REPO_ROOT, "prompt", "planner.md"), encoding="utf-8") as f:
            return f.read()

    def test_related_files_requires_absolute_path(self, planner_prompt):
        related_lines = [l for l in planner_prompt.splitlines() if "相关文件" in l]
        assert related_lines, "planner.md 中未找到'相关文件'字段"
        for line in related_lines:
            assert "绝对路径" in line, f"该行未要求绝对路径: {line}"

    def test_no_relative_path_example_left(self, planner_prompt):
        # 协议示例中不得再出现相对路径示例（如 src/data.py）
        assert "如 src/data.py" not in planner_prompt

    def test_absolute_example_parseable(self, planner_prompt):
        """协议中的示例路径应为以 / 开头的绝对路径形态。"""
        m = re.search(r"相关文件: \[.*?如\s+(/\S+)\]", planner_prompt)
        assert m, "未找到相关文件的路径示例"
        assert m.group(1).startswith("/")
