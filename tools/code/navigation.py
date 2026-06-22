import os
import re
import ast
import json
import pandas as pd

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from tools.code.ast_helper import get_ast_context, get_definitions, generate_file_skeleton
from utils.config import global_config
from utils.file_helper import get_secure_path
from utils.logger import logger


# grep_search tool
GREP_SEARCH_DESC = """在项目工作区内全局搜索指定的字符串或正则表达式。
返回包含文件路径、行号、匹配内容以及所处类/函数的JSON列表。"""

class GrepSearchSchema(BaseModel):
    """Input schema for `grep_search` tool"""
    query: str = Field(description="需要搜索的字符串或正则规则")
    is_regex: bool = Field(default=False, description="query是否为正则表达式 (默认 False)")
    directory: str | None = Field(default=None, description="指定搜索的子目录 (默认为空:表示搜索整个目标项目根目录)")

# find_definition tool
FIND_DEFINITION_DESC = """在项目中查找指定的类、函数或方法的定义信息 (包含起止行号，签名和 docstring)。
本工具不返回具体的实现代码。当你遇到未知的函数调用，想知道它需要传什么参数时，优先使用此工具。"""

class FindDefinitionSchema(BaseModel):
    """Input schema for `find_definition` tool"""
    target_name: str = Field(description="目标名称 (如类名、函数名)")
    search_dir: str = Field(default=".", description="搜索的子目录范围，默认 \".\" (项目根目录)")
    exact_match: bool = Field(default=True, description="是否精确匹配名称，默认 True。设为 False 可进行模糊查找")

# get_file_skeleton tool
GET_FILE_SKELETON_DESC = """获取指定 Python 文件的结构大纲 (skeleton)。遇到陌生的代码文件时，必须先调用此工具了解文件的宏观结构，切忌盲目阅读全文件源码。
返回该文件内所有的类（及其第一层方法）、全局函数的签名、Docstring 以及它们所在的行号范围。"""

class GetFileSkeletonSchema(BaseModel):
    """Input schema for `get_file_skeleton` tool"""
    file_path: str = Field(description="文件相对路径")

# read_code_block tool
READ_CODE_BLOCK_DESC = """精确读取指定文件中特定行号范围内的源代码。需要了解详细代码时，优先使用此工具。
返回的代码会带有行号前缀 (例如 "  39 | def foo():")，方便你精确评估上下文并为后续的代码修改工具提供准确坐标。"""

class ReadCodeBlockSchema(BaseModel):
    """Input schema for `read_code_block` tool"""
    file_path: str = Field(description="需要读取的文件相对路径")
    start_line: int = Field(description="起始行号 (从 1 开始)")
    end_line: int = Field(description="结束行号 (包含此行)")

# inspect_dataframe tool
INSPECT_DATAFRAME_DESC = """结构化数据分析工具。当你需要了解一个 CSV、Excel 或 JSON 格式的数据集长什么样、包含哪些特征、数据分布如何时，优先使用此工具。"""

class InspectDataframeSchema(BaseModel):
    """Input schema for `inspect_dataframe` tool"""
    file_path: str = Field(description="数据文件相对路径")


class NavigationTools:
    def __init__(self) -> None:
        self.project_root = global_config.get("workspace", {}).get("root_dir", "")

    def _grep_search_impl(
        self,
        query: str,
        is_regex: bool = False,
        directory: Optional[str] = None
    ) -> str:
        """
        在项目工作区内全局搜索指定的字符串或正则表达式。
        返回包含文件路径、行号、匹配内容以及所处类/函数的JSON列表。

        参数:
        - query: 需要搜索的字符串或正则规则 (必填)
        - is_regex: query是否为正则表达式 (默认 False)
        - directory: 指定搜索的子目录 (默认为空:表示搜索整个目标项目根目录)
        """
        # 1. 加载配置并进行安全路径校验
        allowed_suffix = set(global_config.get("workspace", {}).get("grep_allowed_suffix", ["py"]))
        try:
            search_root = get_secure_path(self.project_root, directory)
        except PermissionError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        if not os.path.exists(search_root):
            return json.dumps({"error": f"❌ 目录不存在: {search_root}"}, ensure_ascii=False)

        # 2. 编译搜索模式
        try:
            pattern = re.compile(query) if is_regex else re.compile(re.escape(query))
        except re.error as e:
            return json.dumps({"error": f"❌ 无效的正则表达式: {e}"}, ensure_ascii=False)

        results = []
        # 3. 遍历搜索
        for root, _, files in os.walk(search_root):
            for file_name in files:
                # 校验白名单后缀
                suffix = file_name.split('.')[-1] if '.' in file_name else ""
                if suffix not in allowed_suffix:
                    continue
                # 读取文件内容
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    logger.warning(f"跳过无法读取的文件 {file_path}: {e}")
                    continue
                # 逐行匹配
                lines = content.split('\n')
                file_matches = [(i + 1, line) for i, line in enumerate(lines) if pattern.search(line)]
                if not file_matches:
                    continue

                # 4. AST 解析
                ast_tree = None
                if suffix == "py":
                    try:
                        ast_tree = ast.parse(content)
                    except Exception as e:
                        logger.warning(f"AST 解析异常 {file_path}: {e}")

                # 5. 组装结果
                for line_num, line_content in file_matches:
                    ast_context = get_ast_context(ast_tree, line_num)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    results.append({
                        "file_path": rel_path.replace("\\", "/"),
                        "line_number": line_num,
                        "matched_content": line_content.strip(),
                        "ast_context": ast_context
                    })

        if not results:
            return json.dumps({"message": f"未在项目中找到与 '{query}' 匹配的内容。"}, ensure_ascii=False)
        return json.dumps(results, ensure_ascii=False, indent=2)

    def _find_definition_impl(
        self,
        target_name: str,
        search_dir: str = ".",
        exact_match: bool = True
    ) -> str:
        """
        在项目中查找指定的类、函数或方法的定义信息 (包含起止行号，签名和 docstring)。
        本工具不返回具体的实现代码。当你遇到未知的函数调用，想知道它需要传什么参数时，优先使用此工具。

        参数:
        - target_name: 目标名称 (如类名、函数名，必填)
        - search_dir: 搜索的子目录范围，默认 "." (项目根目录)
        - exact_match: 是否精确匹配名称，默认 True。设为 False 可进行模糊查找。
        """
        # 1. 加载配置并进行安全路径校验
        try:
            search_root = get_secure_path(self.project_root, search_dir)
        except PermissionError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        if not os.path.exists(search_root):
            return json.dumps({"error": f"❌ 目录不存在: {search_root}"}, ensure_ascii=False)

        results = []
        # 2. 遍历搜索
        for root, _, files in os.walk(search_root):
            for file_name in files:
                suffix = file_name.split('.')[-1] if '.' in file_name else ""
                # 严格限制：AST 解析仅支持 .py 文件
                if suffix != "py":
                    continue
                file_path = os.path.join(root, file_name)

                # 3. 读取并解析 AST
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast_tree = ast.parse(content)
                except SyntaxError:
                    continue
                except Exception as e:
                    logger.warning(f"跳过异常文件 {file_path}: {e}")
                    continue

                # 4. 调用底层引擎提取定义
                defs = get_definitions(
                    file_path=os.path.relpath(file_path, self.project_root).replace("\\", "/"),
                    ast_tree=ast_tree,
                    source_code=content,
                    target_name=target_name,
                    exact_match=exact_match
                )
                if defs:
                    results.extend(defs)

        # 5. 返回结构化 JSON
        if not results:
            return json.dumps({"message": f"未找到名为 '{target_name}' 的定义。"}, ensure_ascii=False)
        return json.dumps(results, ensure_ascii=False, indent=2)

    def _get_file_skeleton_impl(self, file_path: str) -> str:
        """
        获取指定 Python 文件的结构大纲 (skeleton/outline)。
        返回该文件内所有的类（及其第一层方法）、全局函数的签名、Docstring 以及它们所在的行号范围。

        【黄金法则】：在修改一个陌生的文件之前，必须先调用此工具了解文件的宏观结构，切忌盲目阅读全文件源码！

        参数:
        - file_path: 文件相对路径 (必填)
        """
        # 1. 路径与安全校验
        try:
            abs_path = get_secure_path(self.project_root, file_path)
        except PermissionError as e:
            return f"❌ 安全错误: {str(e)}"

        if not os.path.exists(abs_path):
            return f"❌ 错误: 文件不存在 ({file_path})"
        if not abs_path.endswith('.py'):
            return f"❌ 错误: 结构提取仅支持 .py 文件。"

        # 2. 读取与解析
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_lines = content.splitlines()
            ast_tree = ast.parse(content)
        except SyntaxError as e:
            return f"❌ 语法错误: 该文件当前存在 SyntaxError ({e.lineno}行)，无法提取骨架。请使用 read_code_block 查看或修复。"
        except Exception as e:
            return f"❌ 解析异常: {str(e)}"

        # 3. 提取骨架
        return generate_file_skeleton(ast_tree, source_lines, file_path)

    def _read_code_block_impl(
        self,
        file_path: str,
        start_line: int,
        end_line: int
    ) -> str:
        """
        精确读取指定文件中特定行号范围内的源代码。
        返回的代码会带有行号前缀 (例如 "  39 | def foo():")，方便你精确评估上下文并为后续的 edit_code_block 工具提供准确坐标。

        参数:
        - file_path: 需要读取的文件相对路径 (必填)
        - start_line: 起始行号 (从 1 开始, 必填)
        - end_line: 结束行号 (包含此行, 必填)
        """
        # 1. 安全路径校验
        try:
            abs_path = get_secure_path(self.project_root, file_path)
        except PermissionError as e:
            return f"❌ 安全错误: {str(e)}"

        # 2. 基础校验
        if not os.path.exists(abs_path):
            return f"❌ 错误: 文件不存在 ({file_path})"
        if not os.path.isfile(abs_path):
            return f"❌ 错误: 目标路径不是一个文件 ({file_path})"
        if start_line < 1 or end_line < start_line:
            return f"❌ 错误: 无效的行号范围。start_line 必须 >= 1，且 end_line 必须 >= start_line。"

        # 3. 读取文件
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return f"❌ 错误: 无法读取文件内容 ({str(e)})"
        total_lines = len(lines)
        if start_line > total_lines:
            return f"❌ 错误: 起始行号 {start_line} 超出了文件总行数 ({total_lines})。"
        # 如果 end_line 超出了文件末尾，自动截断到最后一行
        actual_end_line = min(end_line, total_lines)

        # 4. 切片并拼接行号
        target_lines = lines[start_line - 1 : actual_end_line]
        # 构造输出头部
        result_str = f"--- {file_path} (Lines {start_line}-{actual_end_line}) ---\n"
        # 逐行打上行号
        for i, line in enumerate(target_lines):
            current_line_num = start_line + i
            clean_line = line.rstrip('\r\n')
            result_str += f"{current_line_num:4d} | {clean_line}\n"

        return result_str

    def _distill_dataframe_info(self, df: pd.DataFrame, file_path: str) -> str:
        """
        将 DataFrame 蒸馏为 Markdown 画像
        """
        try:
            # 基础结构
            schema_dict = df.dtypes.astype(str).to_dict()
            schema_str = "\n".join([f"- `{col}`: {dtype}" for col, dtype in schema_dict.items()])
            # 统计分布
            desc_df = df.describe(include='all')
            rows_to_drop = ['count', '25%', '50%', '75%']
            desc_df = desc_df.drop(index=[p for p in rows_to_drop if p in desc_df.index])
            distribution_str = desc_df.to_markdown()
            # 数据抽样
            head_str = df.head(5).to_markdown()
            if len(df) > 10:
                tail_str = df.tail(5).to_markdown()
                tail_body = "\n".join(tail_str.split('\n')[2:])
                dots_row = "| ..."
                samples_str = f"{head_str}\n{dots_row}\n{tail_body}"
            else:
                samples_str = head_str
            # 组装数据画像
            report = (
                f"**数据集画像: `{file_path}`**\n"
                f"- **规模**: {len(df)} 行 (Rows) × {len(df.columns)} 列 (Columns)\n\n"
                f"### 1. 特征结构 (Schema)\n"
                f"{schema_str}\n\n"
                f"### 2. 统计分布 (Distribution)\n"
                f"{distribution_str}\n\n"
                f"### 3. 数据切片 (Samples)\n"
                f"{samples_str}\n"
            )
            max_chars = global_config.get("agent", {}).get("max_chars")
            if max_chars and len(report) > max_chars:
                return report[:max_chars] + "\n\n... 数据过多，已自动截断 ..."
            return report
        except Exception as e:
            return f"❌ 生成数据画像时发生异常: {str(e)}"

    def _inspect_dataframe_impl(self, file_path: str) -> str:
        """
        结构化数据分析工具。当你需要了解一个 CSV、Excel 或 JSON 格式的数据集长什么样、包含哪些特征、数据分布如何时使用。

        参数:
        - file_path: 数据文件相对路径 (必填)
        """
        # 1. 安全路径校验
        try:
            abs_path = get_secure_path(self.project_root, file_path)
        except PermissionError as e:
            return f"❌ 安全错误: {str(e)}"
        if not os.path.exists(abs_path):
            return f"❌ 错误: 文件不存在 ({file_path})"
        if not os.path.isfile(abs_path):
            return f"❌ 错误: 目标路径不是一个文件 ({file_path})"

        # 2. 格式自动路由与数据加载
        ext = os.path.splitext(abs_path)[1].lower()
        try:
            if ext == '.csv':
                df = pd.read_csv(abs_path)
            elif ext == '.tsv':
                df = pd.read_csv(abs_path, sep='\t')
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(abs_path)
            elif ext == '.json':
                df = pd.read_json(abs_path)
            elif ext == '.jsonl':
                df = pd.read_json(abs_path, lines=True)
            else:
                return f"❌ 不支持的文件格式: {ext}。当前仅支持 csv, tsv, xlsx, xls, json, jsonl"
        except Exception as e:
            return f"❌ 数据解析失败，文件可能已损坏或格式不规范:\n{str(e)}"
        if df.empty:
            return "⚠️ 警告：成功读取文件，但数据集为空 (0 行数据)。"

        # 3. 信息蒸馏
        return self._distill_dataframe_info(df, file_path)

    def create_grep_search_tool(self):
        return StructuredTool.from_function(
            name="grep_search",
            description=GREP_SEARCH_DESC,
            func=self._grep_search_impl,
            infer_schema=False,
            args_schema=GrepSearchSchema,
        )

    def create_find_definition_tool(self):
        return StructuredTool.from_function(
            name="find_definition",
            description=FIND_DEFINITION_DESC,
            func=self._find_definition_impl,
            infer_schema=False,
            args_schema=FindDefinitionSchema,
        )

    def create_get_file_skeleton_tool(self):
        return StructuredTool.from_function(
            name="get_file_skeleton",
            description=GET_FILE_SKELETON_DESC,
            func=self._get_file_skeleton_impl,
            infer_schema=False,
            args_schema=GetFileSkeletonSchema,
        )

    def create_read_code_block_tool(self):
        return StructuredTool.from_function(
            name="read_code_block",
            description=READ_CODE_BLOCK_DESC,
            func=self._read_code_block_impl,
            infer_schema=False,
            args_schema=ReadCodeBlockSchema,
        )

    def create_inspect_dataframe_tool(self):
        return StructuredTool.from_function(
            name="inspect_dataframe",
            description=INSPECT_DATAFRAME_DESC,
            func=self._inspect_dataframe_impl,
            infer_schema=False,
            args_schema=InspectDataframeSchema,
        )
