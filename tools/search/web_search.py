import os
import json
import requests
import urllib3

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from utils.config import global_config


# web_search tool
WEB_SEARCH_DESC = """通用网络搜索引擎。用于调研前沿技术方案、查阅官方文档或寻找算法灵感。当你需要规划方案时使用。"""

class WebSearchSchema(BaseModel):
    """Input schema for `web_search` tool"""
    query: str = Field(description="搜索关键词")

# debug_search tool
DEBUG_SEARCH_DESC = """通用报错解决方案搜索引擎。用于检索网络中的 Bug 修复方案。"""

class DebugSearchSchema(BaseModel):
    """Input schema for `debug_search` tool"""
    error_message: str = Field(description="报错的核心信息 (例如: \"ValueError: Expected 2D array, got 1D array instead\")")
    source: str = Field(default="all", description="""搜索来源策略，请严格按照以下规则选择:
1. "stackoverflow": 当报错是通用的 Python 语法错误、常见的逻辑 Bug，或基础环境配置问题时使用。
2. "github": 当报错明确指向某个开源库内部的 Bug，或者包含具体的依赖冲突时使用。
3. "all": 当你无法判断，或前两次专项搜索都找不到答案时作为兜底使用。""")


class WebSearchTools:
    def _tavily_search(self, query: str, include_domains: list = None, max_results: int = 3) -> str:
        """
        通用 Tavily 搜索执行引擎。
        """
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        proxy = global_config.get("proxy", "")
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy

        tavily_key = global_config.get("tavily", {}).get("api_key")
        if not tavily_key:
            return json.dumps({"status": "error", "message": "引擎未配置：缺少 Tavily API Key。"}, ensure_ascii=False)
        if not query:
            return json.dumps({"status": "error", "message": "无效搜索内容：query 不能为空"}, ensure_ascii=False)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {tavily_key}"
        }
        payload = {
            "api_key": tavily_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results
        }
        if include_domains:
            payload["include_domains"] = include_domains

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload,
                verify=False
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            if not results:
                return json.dumps({"status": "failed", "message": "未找到相关结果，请尝试更换搜索词。"}, ensure_ascii=False)

            formatted_results = [
                {
                    "title": res.get("title"),
                    "content": res.get("content")
                } for res in results
            ]
            return json.dumps({"status": "success", "data": formatted_results}, ensure_ascii=False, indent=2)
        except requests.exceptions.RequestException as e:
            return json.dumps({"status": "error", "message": f"Tavily 网络请求失败: {str(e)}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Tavily 解析异常: {str(e)}"}, ensure_ascii=False)
        finally:
            os.environ.pop("http_proxy")
            os.environ.pop("https_proxy")

    def _web_search_impl(self, query: str) -> str:
        """
        通用网络搜索引擎。用于调研前沿技术方案、查阅官方文档或寻找算法灵感。当你需要规划方案时使用。

        参数:
        - query: 搜索关键词
        """
        return self._tavily_search(query=query, max_results=5)

    def _debug_search_impl(self, error_message: str, source: str = "all") -> str:
        """
        通用报错解决方案搜索引擎。用于检索网络中的 Bug 修复方案。

        参数:
        - error_message: 报错的核心信息 (例如: "ValueError: Expected 2D array, got 1D array instead")
        - source: 搜索来源策略，请严格按照以下规则选择:
            1. "stackoverflow": 当报错是通用的 Python 语法错误、常见的逻辑 Bug，或基础环境配置问题时使用。
            2. "github": 当报错明确指向某个开源库内部的 Bug，或者包含具体的依赖冲突时使用。
            3. "all": 当你无法判断，或前两次专项搜索都找不到答案时作为兜底使用 (默认)。
        """
        domain_filter = []
        if source == "stackoverflow":
            domain_filter = ["stackoverflow.com"]
        elif source == "github":
            domain_filter = ["github.com"]
        return self._tavily_search(query=error_message, include_domains=domain_filter, max_results=3)

    def create_web_search_tool(self):
        return StructuredTool.from_function(
            name="web_search",
            description=WEB_SEARCH_DESC,
            func=self._web_search_impl,
            infer_schema=False,
            args_schema=WebSearchSchema,
        )

    def create_debug_search_tool(self):
        return StructuredTool.from_function(
            name="debug_search",
            description=DEBUG_SEARCH_DESC,
            func=self._debug_search_impl,
            infer_schema=False,
            args_schema=DebugSearchSchema,
        )
