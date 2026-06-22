import functools
from typing import Any

from langchain_core.tools import StructuredTool
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.subagents import SubAgentMiddleware


class StrictSubAgentMiddleware(SubAgentMiddleware):
    """
    继承自带的 SubAgentMiddleware，通过装饰器模式无痕拦截 task 工具，实现前置校验逻辑
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 提取 backend 实例用于检查文件
        backend_arg = kwargs.get("backend")
        actual_backend = backend_arg() if callable(backend_arg) else backend_arg
        # 提取父类创建好的默认 task_tool
        original_task_tool = self.tools[0]
        original_func = original_task_tool.func
        original_coroutine = original_task_tool.coroutine
        # 使用 functools.wraps 伪装成原函数 (保留所有参数签名，如 runtime)
        @functools.wraps(original_func)
        def wrapped_task(*task_args, **task_kwargs):
            # 获取大模型传入的目标 agent 类型
            sub_type = task_kwargs.get("subagent_type")
            if sub_type:
                err = self._check_preconditions(sub_type, actual_backend)
                if err:
                    return err # 触发拦截，直接将报错返回给大模型
            # 校验通过，放行！调用原始的 task 执行逻辑
            return original_func(*task_args, **task_kwargs)
        # 如果原工具有异步实现，也一并包装
        wrapped_atask = None
        if original_coroutine:
            @functools.wraps(original_coroutine)
            async def _wrapped_atask(*task_args, **task_kwargs):
                sub_type = task_kwargs.get("subagent_type")
                if sub_type:
                    err = self._check_preconditions(sub_type, actual_backend)
                    if err:
                        return err
                return await original_coroutine(*task_args, **task_kwargs)
            wrapped_atask = _wrapped_atask
        # 重新打包成 StructuredTool，替换掉原来的工具
        new_task_tool = StructuredTool.from_function(
            name=original_task_tool.name,
            func=wrapped_task,
            coroutine=wrapped_atask,
            description=original_task_tool.description,
            infer_schema=False,
            args_schema=original_task_tool.args_schema, 
        )
        self.tools = [new_task_tool]

    def _check_preconditions(self, sub_type: str, backend: BackendProtocol) -> str | None:
        if not sub_type:
            return None

        agent_target = sub_type.lower()
        check_file = None

        if agent_target == "planner":
            check_file = "/memories/plan_log.md"
        elif agent_target in ["developer", "debugger"]:
            check_file = "/memories/todo.md"

        if check_file:
            try:
                read_result = backend.read(check_file)
                error = None
                file_data = None

                if isinstance(read_result, dict):
                    error = read_result.get("error")
                    file_data = read_result.get("file_data")
                else:
                    error = getattr(read_result, "error", None)
                    file_data = getattr(read_result, "file_data", None)

                # 检查是否抛错
                if error:
                    return f"❌ 任务调度被拦截：调用 {sub_type} 失败！未创建 {check_file} 文件，请检测自己是否严格遵循工作流。"
                # 提取 FileData 内容
                content = ""
                if isinstance(file_data, dict):
                    content = file_data.get("content", "")
                elif file_data is not None:
                    content = getattr(file_data, "content", "")
                # 检查内容是否为空
                if not content or not str(content).strip():
                    return f"❌ 任务调度被拦截：调用 {sub_type} 失败！{check_file} 文件为空，请检测自己是否严格遵循工作流。"
            except Exception:
                return f"❌ 任务调度被拦截：调用 {sub_type} 失败！读取 {check_file} 文件发生异常，请先确认文件状态。"
        return None
