import os
from abc import abstractmethod
from typing import List, Any, Dict

import deepagents.graph as da_gh
import deepagents.middleware.filesystem as fs_mw
import deepagents.middleware.subagents as sa_mw
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware.subagents import SubAgent
from deepagents.middleware.permissions import FilesystemPermission
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from langchain.agents.middleware import SummarizationMiddleware, TodoListMiddleware, ModelRetryMiddleware
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
# from langgraph.store.postgres import PostgresStore

from llm.request_llm import RequestLLM
from agent.middleware import StrictSubAgentMiddleware
from tools.code.editor import EditorTools
from tools.code.executor import SandboxTools
from tools.code.navigation import NavigationTools
from tools.code.verify import VerificationTools
from tools.execute.execute_bash import ExecuteBashTool
from tools.execute.execute_git import ExecuteGitTool
from tools.search.web_search import WebSearchTools
from utils.config import global_config
from utils.logger import logger


_FILESYSTEM_MANAGER_PROMPT_TEMPLATE = """## Following Conventions

- Read files before editing — understand existing content before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `read_file`, `write_file`, `edit_file`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with `/memories` Follow the tool docs for the available tools, and use pagination (offset/limit) when reading large files.

- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem

## Large Tool Results

When a tool result is too large, it may be offloaded into the filesystem instead of being returned inline. In those cases, use `read_file` to inspect the saved result in chunks. Offloaded tool results are stored under `{large_tool_results_prefix}/<tool_call_id>`."""

_FILESYSTEM_REPLACED_PROMPT_TEMPLATE = """## Following Conventions

- Read files before editing — understand existing content before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `read_file`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with `/memories` Follow the tool docs for the available tools, and use pagination (offset/limit) when reading large files.

- read_file: read a file from the filesystem

## Large Tool Results

When a tool result is too large, it may be offloaded into the filesystem instead of being returned inline. In those cases, use `read_file` to inspect the saved result in chunks. Offloaded tool results are stored under `{large_tool_results_prefix}/<tool_call_id>`."""

DEVELOPER_READ_FILE_DESCRIPTION = """Reads a file from the memory filesystem (虚拟文件系统).

【适用范围 — 严格限制】本工具只能读取虚拟文件系统中的 `/memories/`（如 /memories/todo.md）与 `/skills/` 目录。
它**不能**读取项目真实代码文件！项目源码、配置、数据文件一律使用代码侦察工具（get_file_skeleton / read_code_block / find_definition / grep_search），并传入以项目根目录开头的完整绝对路径。

Usage:
- By default, it reads up to 100 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.

- You should ALWAYS make sure a file has been read before editing it."""

READ_FILE_REPLACED_DESCRIPTION = """Reads a file from the memory filesystem.

Assume this tool is only able to read files in `/memories/` and `/skills/` directory. It is okay to read a file that does not exist; an error will be returned.

Usage:
- By default, it reads up to 100 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.

- You should ALWAYS make sure a file has been read before editing it."""

TASK_TOOL_REPLACED_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement."""


class ReadFileDescriptionOverrideMiddleware(fs_mw.FilesystemMiddleware):
    """
    按 agent 生效的 read_file 工具描述覆盖中间件（W3-3）。

    机制：只重新注入 read_file 一个工具。同一 Agent 的中间件链中，本中间件位于
    deepagents 内置 FilesystemMiddleware 之后，同名工具后注册者生效（ToolNode
    last-wins），因此 read_file 使用此处定制的描述，其余文件系统工具
    （write_file/edit_file/ls/glob/grep）仍由内置栈提供，行为完全不变。
    取代了原先"改写 fs_mw.READ_FILE_TOOL_DESCRIPTION 模块全局变量"的做法
    （该做法对所有 Agent 一刀切生效）。
    """
    def __init__(self, read_file_description: str, **kwargs):
        super().__init__(
            custom_tool_descriptions={"read_file": read_file_description},
            **kwargs,
        )
        self.tools = [t for t in self.tools if t.name == "read_file"]


class BaseAgentBuilder:
    """
    Agent 构建器的抽象基类
    """
    def __init__(
        self,
        llm_name: str,
        store: InMemoryStore,
        checkpointer: InMemorySaver,
        max_messages: int = 40,
        prompt_context: Dict[str, Any] = None
    ) -> None:
        self.llm_kwargs = global_config.get("llm", {}).get(llm_name)
        if not self.llm_kwargs:
            raise RuntimeError(f"LLM {llm_name} 配置不存在")
        self.store = store
        self.checkpointer = checkpointer
        self.max_messages = max_messages

        workspace_root = global_config.get("workspace", {}).get("root_dir")
        git_diff_dir = global_config.get("workspace", {}).get("git_diff_dir")
        structure_filepath = global_config.get("workspace", {}).get("structure_filepath")
        max_retries = global_config.get("agent", {}).get("max_retries")
        max_plan_num = global_config.get("agent", {}).get("max_plan_num")
        self.prompt_context = prompt_context or {
            "workspace_root": workspace_root,
            "git_diff_dir": git_diff_dir,
            "structure_filepath": structure_filepath or "无",
            "max_retries": max_retries or 5,
            "max_plan_num": max_plan_num or 5,
        }

        TodoListMiddleware.wrap_model_call = lambda self, request, handler: handler(request)
        TodoListMiddleware.awrap_model_call = lambda self, request, handler: handler(request)
        da_gh.BASE_AGENT_PROMPT = ""
        da_gh.SubAgentMiddleware = StrictSubAgentMiddleware
        # 注意：read_file 工具描述不再通过改写 fs_mw.READ_FILE_TOOL_DESCRIPTION
        # 模块全局变量注入（该方式对所有 Agent 一刀切生效），
        # 改由 ReadFileDescriptionOverrideMiddleware 按 agent 注入（见 _get_middleware）。
        sa_mw.TASK_TOOL_DESCRIPTION = TASK_TOOL_REPLACED_DESCRIPTION
        sa_mw.GENERAL_PURPOSE_SUBAGENT["name"] = "Planner"

    @abstractmethod
    def get_role(self) -> str:
        """返回Agent的角色名称"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """作为 SubAgent 的职责描述"""
        pass

    @abstractmethod
    def get_prompt_filepath(self) -> str:
        """返回Agent专属System Prompt 的文件路径"""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """返回Agent可用的工具列表"""
        pass

    @abstractmethod
    def get_exclude_tools(self) -> List[str]:
        """返回Agent需去除的工具列表"""
        pass

    def get_skill_mappings(self) -> Dict[str, str]:
        """返回skills的 {虚拟路径: 物理相对路径} 的映射"""
        return {}

    def get_read_file_description(self) -> str:
        """本 Agent 的 read_file 工具描述（按 agent 生效，子类可覆盖定制）"""
        return READ_FILE_REPLACED_DESCRIPTION

    def get_filesystem_prompt(self) -> str:
        """默认的 Filesystem Prompt (Subagents 使用)"""
        return _FILESYSTEM_REPLACED_PROMPT_TEMPLATE

    def _get_llm(self) -> RequestLLM:
        kwargs = {"role": self.get_role(), **self.llm_kwargs}
        return RequestLLM(**kwargs)

    def _load_prompt_template(self, filepath: str, **kwargs) -> str:
        """
        加载PromptTemplate并注入变量。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到 Prompt 文件: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        prompt_template = PromptTemplate.from_template(
            prompt_content, 
            template_format="jinja2"
        )
        return prompt_template.format(**kwargs)

    def _create_hybrid_backend(self) -> CompositeBackend:
        """
        创建混合存储后端：
        - 默认路由: 短期记忆存入 StoreBackend
        - 长期记忆路由: 存入 StoreBackend (/memories/) ,跨 Agent 继承
        """
        return CompositeBackend(
            default=StoreBackend(),
            routes={
                "/memories/": StoreBackend(
                    namespace=lambda _: ("filesystem",)
                )
            }
        )

    def _upload_skills(self):
        """
        将物理磁盘的 Skill 上传至 Backend 的 ("filesystem",) 命名空间
        """
        for virtual_base, physical_base in self.get_skill_mappings().items():
            if not os.path.exists(physical_base):
                logger.warning(f"⚠️ 未找到 skill 路径: {physical_base}")
                continue
            # 如果映射的是一个具体的文件
            if os.path.isfile(physical_base):
                self._upload_single_skill_file(physical_base, virtual_base)
            # 如果映射的是一个文件夹
            elif os.path.isdir(physical_base):
                for root, _, files in os.walk(physical_base):
                    for file in files:
                        physical_file_path = os.path.join(root, file)
                        # 计算相对路径
                        rel_path = os.path.relpath(physical_file_path, physical_base)
                        rel_path_posix = rel_path.replace(os.sep, '/')
                        # 组装最终的虚拟路径
                        virtual_key = f"{virtual_base.rstrip('/')}/{rel_path_posix}"
                        self._upload_single_skill_file(physical_file_path, virtual_key)

    def _upload_single_skill_file(self, physical_path: str, virtual_key: str):
        """
        文件上传器
        """
        try:
            with open(physical_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.store.put(
                namespace=("filesystem",),
                key=virtual_key,
                value=create_file_data(content) 
            )
            logger.debug(f"✅ 成功上传 Skill 文件: {physical_path} -> {virtual_key}")
        except UnicodeDecodeError:
            logger.warning(f"⏩ 跳过非纯文本格式的 Skill 资源文件: {physical_path}")
        except Exception as e:
            logger.error(f"❌ 上传 Skill 文件失败 [{physical_path}]: {str(e)}")

    def _get_system_prompt(self) -> str:
        return self._load_prompt_template(
            filepath=self.get_prompt_filepath(),
            **self.prompt_context
        )

    def _get_skills(self) -> List[str]:
        self._upload_skills()
        skill_mappings = self.get_skill_mappings()
        if skill_mappings:
            return [f"/skills/{self.get_role()}"]
        return None

    def _get_middleware(self):
        return [
            SummarizationMiddleware(model=self._get_llm(), trigger=("messages", self.max_messages)),
            ModelRetryMiddleware(max_retries=3, initial_delay=10),
            _ToolExclusionMiddleware(excluded=self.get_exclude_tools()),
            # 按 agent 生效的 read_file 描述注入（同名工具后注册者生效）
            ReadFileDescriptionOverrideMiddleware(
                read_file_description=self.get_read_file_description(),
                backend=self._create_hybrid_backend(),
            ),
        ]

    def to_subagent(self) -> SubAgent:
        """构建 SubAgent"""
        # fs_mw._FILESYSTEM_SYSTEM_PROMPT_TEMPLATE = self.get_filesystem_prompt()
        fs_mw._FILESYSTEM_SYSTEM_PROMPT_TEMPLATE = ""
        return SubAgent(
            name=self.get_role(),
            description=self.get_description(),
            system_prompt=self._get_system_prompt(),
            tools=self.get_tools(),
            model=self._get_llm(),
            middleware=self._get_middleware(),
            skills=self._get_skills(),
        )

    def build(self, subagents: List[SubAgent] = None) -> CompiledStateGraph:
        """构建 Agent"""
        # fs_mw._FILESYSTEM_SYSTEM_PROMPT_TEMPLATE = self.get_filesystem_prompt()
        fs_mw._FILESYSTEM_SYSTEM_PROMPT_TEMPLATE = ""
        return create_deep_agent(
            name=self.get_role(),
            model=self._get_llm(),
            tools=self.get_tools(),
            system_prompt=self._get_system_prompt(),
            middleware=self._get_middleware(),
            subagents=subagents,
            skills=self._get_skills(),
            # permissions=[
            #     FilesystemPermission(
            #         operations=["read"],
            #         paths=["/memories/**", "/skills/**"],
            #         mode="allow"
            #     ),
            #     FilesystemPermission(
            #         operations=["write"],
            #         paths=["/memories/**"],
            #         mode="allow"
            #     ),
            #     FilesystemPermission(
            #         operations=["read", "write"],
            #         paths=["/**"],
            #         mode="deny"
            #     )
            # ],
            checkpointer=self.checkpointer,
            store=self.store,
            backend=self._create_hybrid_backend(),
        )


class ManagerAgentBuilder(BaseAgentBuilder):
    def get_role(self) -> str:
        return "Manager"

    def get_description(self) -> str:
        return "系统总管，负责管理日志、版本与调度subagents。"

    def get_prompt_filepath(self) -> str:
        return "prompt/manager.md"

    def get_tools(self) -> List[Any]:
        reset_sandbox = SandboxTools().create_reset_sandbox_tool()
        execute_git_command = ExecuteGitTool().create_execute_git_command_tool()
        run_verification = VerificationTools().create_run_verification_tool()
        return [execute_git_command, reset_sandbox, run_verification]

    def get_exclude_tools(self) -> List[str]:
        return ["write_todos", "ls", "glob", "grep"]

    def get_skill_mappings(self) -> Dict[str, str]:
        return {f"/skills/{self.get_role()}/git-workflow": "skills/git-workflow"}

    def get_filesystem_prompt(self) -> str:
        return _FILESYSTEM_MANAGER_PROMPT_TEMPLATE


class PlannerAgentBuilder(BaseAgentBuilder):
    def get_role(self) -> str:
        return "Planner"

    def get_description(self) -> str:
        return "架构师，负责洞察需求，分析数据，制定开发方案。"

    def get_prompt_filepath(self) -> str:
        return "prompt/planner.md"

    def get_tools(self) -> List[Any]:
        navigation_tools = NavigationTools()
        get_file_skeleton = navigation_tools.create_get_file_skeleton_tool()
        find_definition = navigation_tools.create_find_definition_tool()
        read_code_block = navigation_tools.create_read_code_block_tool()
        inspect_dataframe = navigation_tools.create_inspect_dataframe_tool()
        execute_safe_bash = ExecuteBashTool().create_execute_safe_bash_tool()
        web_search = WebSearchTools().create_web_search_tool()
        return [
            get_file_skeleton, find_definition, read_code_block,
            execute_safe_bash, inspect_dataframe, web_search
        ]

    def get_exclude_tools(self) -> List[str]:
        return ["write_todos", "write_file", "edit_file", "ls", "glob", "grep"]

    def get_skill_mappings(self) -> Dict[str, str]:
        skill_paths = global_config.get("skills", {}).get("planner_skill_path", [])
        mappings = {}
        for physical_path in skill_paths:
            if not physical_path:
                continue
            normalized_path = os.path.normpath(physical_path.strip())
            skill_name = os.path.basename(normalized_path)
            if not skill_name:
                continue
            virtual_path = f"/skills/{self.get_role()}/{skill_name}"
            mappings[virtual_path] = normalized_path
        return mappings


class DeveloperAgentBuilder(BaseAgentBuilder):
    def get_role(self) -> str:
        return "Developer"

    def get_description(self) -> str:
        return "开发者，负责执行方案计划，阅读、编写业务代码，以及在独立沙箱内运行验证测试。"

    def get_prompt_filepath(self) -> str:
        return "prompt/developer.md"

    def get_read_file_description(self) -> str:
        # Developer 专属：明确 read_file 只读虚拟文件系统，项目代码须走
        # 代码侦察工具 + 绝对路径（针对"拿 read_file 读项目文件迷失"事故）
        return DEVELOPER_READ_FILE_DESCRIPTION

    def get_tools(self) -> List[Any]:
        navigation_tools = NavigationTools()
        grep_search = navigation_tools.create_grep_search_tool()
        get_file_skeleton = navigation_tools.create_get_file_skeleton_tool()
        find_definition = navigation_tools.create_find_definition_tool()
        read_code_block = navigation_tools.create_read_code_block_tool()
        editor_tools = EditorTools()
        edit_code_block = editor_tools.create_edit_code_block_tool()
        insert_code = editor_tools.create_insert_code_tool()
        delete_code = editor_tools.create_delete_code_tool()
        execute_sandbox = SandboxTools().create_execute_sandbox_tool()
        execute_safe_bash = ExecuteBashTool().create_execute_safe_bash_tool()
        return [
            grep_search, get_file_skeleton, find_definition, read_code_block,
            edit_code_block, insert_code, delete_code, execute_safe_bash, execute_sandbox
        ]

    def get_exclude_tools(self) -> List[str]:
        return ["write_todos", "write_file", "edit_file", "ls", "glob", "grep"]


class DebuggerAgentBuilder(BaseAgentBuilder):
    def get_role(self) -> str:
        return "Debugger"

    def get_description(self) -> str:
        return "诊断者，当沙箱测试报错时，负责分析报错日志并给出修复方案。"

    def get_prompt_filepath(self) -> str:
        return "prompt/debugger.md"

    def get_tools(self) -> List[Any]:
        navigation_tools = NavigationTools()
        grep_search = navigation_tools.create_grep_search_tool()
        get_file_skeleton = navigation_tools.create_get_file_skeleton_tool()
        find_definition = navigation_tools.create_find_definition_tool()
        read_code_block = navigation_tools.create_read_code_block_tool()
        execute_safe_bash = ExecuteBashTool().create_execute_safe_bash_tool()
        debug_search = WebSearchTools().create_debug_search_tool()
        return [
            grep_search, get_file_skeleton, find_definition, read_code_block,
            execute_safe_bash, debug_search
        ]

    def get_exclude_tools(self) -> List[str]:
        return ["write_todos", "read_file", "write_file", "edit_file", "ls", "glob", "grep"]
