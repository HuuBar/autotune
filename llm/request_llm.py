import os
import json
import httpx

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from openai import OpenAI
from pydantic import PrivateAttr
from typing import Any, Sequence
from typing_extensions import override

from utils.logger import logger
from utils.token_tracker import global_token_tracker


class RequestLLM(BaseChatModel):
    role: str
    url: str
    api_key: str
    model_name: str
    temperature: float
    timeout: int = 600
    _client: OpenAI = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # os.environ["NO_PROXY"] = self.url
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.url,
            http_client=httpx.Client(verify=False, timeout=self.timeout)
        )
        self.disable_streaming = True

    @property
    @override
    def _llm_type(self) -> str:
        return "request_model"

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any
    ) -> ChatResult:
        if stop is not None:
            kwargs["stop"] = stop
        # recode token usage
        ai_msg = self._call(messages, **kwargs)
        token_usage = ai_msg.response_metadata.get("token_usage", {})
        if token_usage:
            global_token_tracker.add_tokens(
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0)
            )
        # return ChatResult
        generation = ChatGeneration(message=ai_msg)
        return ChatResult(
            generations=[generation],
            llm_output={"token_usage": token_usage, "model_name": self.model_name}
        )

    @override
    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Runnable:
        tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=tools, **kwargs)

    def _call(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        msg_list = [self._msg_to_dict(msg) for msg in messages]
        if "tool_choice" in kwargs and kwargs["tool_choice"] is None:
            kwargs.pop("tool_choice")

        payload = {"messages": msg_list}
        # payload.update(kwargs)
        formatted_payload = json.dumps(payload, indent=2, ensure_ascii=False)
        logger.debug(f"({self.role}) LLM input prompt:\n{formatted_payload}")
        # return AIMessage(content=".")
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=msg_list,
                temperature=self.temperature,
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                **kwargs,
            )
            choice = response.choices[0]
            # handle tool calls
            additional_kwargs = {}
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_calls_dict = [
                    {
                        "id": tc.id, 
                        "type": tc.type, 
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in choice.message.tool_calls
                ]
                additional_kwargs["tool_calls"] = tool_calls_dict
            if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content is not None:
                additional_kwargs["reasoning_content"] = choice.message.reasoning_content
            logger.info(f"({self.role}) LLM response:\n{choice.message}")
            # get token usage
            token_usage = {}
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return AIMessage(
                content=choice.message.content or "",
                additional_kwargs=additional_kwargs,
                response_metadata={"token_usage": token_usage, "model_name": self.model_name}
            )
        except Exception as e:
            logger.error(f"{self.model_name} request error: {str(e)}")
            return AIMessage(
                content="",
                response_metadata={"token_usage": {}, "error": str(e)}
            )

    @staticmethod
    def _msg_to_dict(message: BaseMessage) -> dict:
        """
        convert a langchain message to a dict
        return: dict{"role": message.role, "content": message.content}
        """
        if isinstance(message, ChatMessage):
            return {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            msg_dict = {"role": "assistant", "content": message.content}
            if message.additional_kwargs:
                msg_dict.update(message.additional_kwargs)
            return msg_dict
        elif isinstance(message, FunctionMessage):
            return {"role": "function", "content": message.content, "name": message.name}
        elif isinstance(message, ToolMessage):
            return {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
        else:
            raise TypeError(f"Unknown message type {message}")
