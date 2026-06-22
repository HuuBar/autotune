import os
import ast

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from tools.code.ast_helper import format_syntax_error
from utils.config import global_config
from utils.file_helper import get_secure_path


# edit_code_block tool
EDIT_CODE_BLOCK_DESC = """用新代码精确替换指定文件中从 start_line 到 end_line (包含) 的代码块。
【强制约束】：
1. 你传入的 `new_code` 必须保持与原代码块完全一致的的**绝对前导空格**。
2. 严禁随意做去缩进(dedent)操作，如果有多行代码，尽量带上准确的前导空格。

【安全提示】：
工具自带语法预检。如果你的修改导致了 SyntaxError 或 IndentationError，修改将被自动拦截撤销，并返回报错行号供你修正。"""

class EditCodeBlockSchema(BaseModel):
    """Input schema for `edit_code_block` tool"""
    file_path: str = Field(description="文件相对路径")
    start_line: int = Field(description="起始行号 (从 1 开始)")
    end_line: int = Field(description="结束行号 (包含此行)")
    new_code: str = Field(description="用于替换的纯 Python 代码逻辑")

# insert_code tool
INSERT_CODE_DESC = """在指定文件的特定行之后插入新代码。
【进阶用法：创建新文件】
如果目标文件不存在，你可以使用此工具来创建新文件。
此时，必须严格设置 line_number=0。工具会自动为你创建目录结构并写入 new_code。

【注意】：请严格控制传入 new_code 的缩进级别，使其与上下文完美匹配。工具内置 AST 预检，缩进错误或语法错误将被拒绝写入。"""

class InsertCodeSchema(BaseModel):
    """Input schema for `insert_code` tool"""
    file_path: str = Field(description="文件相对路径")
    line_number: int = Field(description="锚点行号。代码将插入到这一行之后。传入 0 表示插入到文件最开头")
    new_code: str = Field(description="要插入的纯 Python 代码逻辑")

# delete_code tool
DELETE_CODE_DESC = """精确删除指定文件中从 start_line 到 end_line (包含) 的代码块。

【安全限制 1】：本工具被严禁用于删除整个文件。你必须保留文件的基本结构。
【安全限制 2】：工具内置 AST 预检。如果你删除了父级控制流（如 if/def/try）却遗留了其内部的缩进代码，或者破坏了括号闭合，操作将被自动撤销。"""

class DeleteCodeSchema(BaseModel):
    """Input schema for `delete_code` tool"""
    file_path: str = Field(description="文件相对路径")
    start_line: int = Field(description="起始行号 (从 1 开始)")
    end_line: int = Field(description="结束行号 (包含此行)")


class EditorTools:
    def __init__(self) -> None:
        self.project_root = global_config.get("workspace", {}).get("root_dir", "")

    def _clean_llm_code(self, code: str) -> str:
        """去除大模型习惯性附带的 Markdown 代码块标记"""
        code = code.strip()
        if code.startswith("```"):
            lines = code.split('\n')
            # 去除首行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 去除尾行 ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return code

    def _align_code_indentation(self, original_block: list[str], llm_code_lines: list[str]) -> str:
        """
        智能缩进对齐引擎
        """
        if not llm_code_lines:
            return ""
        # 获取原代码首行的完整前导空白字符
        original_first_line = original_block[0]
        orig_indent_str = original_first_line[:len(original_first_line) - len(original_first_line.lstrip('\t '))]
        orig_indent_len = len(orig_indent_str)
        llm_first_line = llm_code_lines[0]
        llm_indent_len = len(llm_first_line) - len(llm_first_line.lstrip('\t '))

        aligned_lines = []
        # 仅干预首行：如果 LLM 的首行缩进比原文件少，把缺失的部分补上
        if orig_indent_len > llm_indent_len:
            missing_indent_len = orig_indent_len - llm_indent_len
            indent_to_add = orig_indent_str[:missing_indent_len]
            aligned_lines.append(indent_to_add + llm_first_line)
        else:
            aligned_lines.append(llm_first_line)
        aligned_lines.extend(llm_code_lines[1:])
        return "\n".join(aligned_lines)

    def _edit_code_block_impl(self, file_path: str, start_line: int, end_line: int, new_code: str) -> str:
        """
        用新代码精确替换指定文件中从 start_line 到 end_line (包含) 的代码块。

        【强制约束】：
        1. 你传入的 `new_code` 必须保持与原代码块完全一致的的**绝对前导空格**。
        2. 严禁随意做去缩进(dedent)操作，如果有多行代码，尽量带上准确的前导空格。

        【安全提示】：
        工具自带语法预检。如果你的修改导致了 SyntaxError 或 IndentationError，修改将被自动拦截撤销，并返回报错行号供你修正。

        参数:
        - file_path: 文件相对路径 (必填)
        - start_line: 起始行号 (从 1 开始，必填)
        - end_line: 结束行号 (包含此行，必填)
        - new_code: 用于替换的纯 Python 代码逻辑 (必填)
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

        # 2. 读取原文件并校验行号
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return f"❌ 错误: 无法读取文件内容 ({str(e)})"
        total_lines = len(lines)
        if start_line < 1 or end_line < start_line:
            return f"❌ 错误: 无效的行号范围。start_line 必须 >= 1，且 end_line 必须 >= start_line。"
        if start_line > total_lines:
            return f"❌ 错误: 无效的行号范围。文件共 {total_lines} 行。"
        actual_end_line = min(end_line, total_lines)

        # 3. 智能缩进对齐
        original_block = lines[start_line - 1 : actual_end_line]
        cleaned_code = self._clean_llm_code(new_code).rstrip('\n')
        llm_lines = cleaned_code.split('\n') if cleaned_code else []
        aligned_code = self._align_code_indentation(original_block, llm_lines)

        # 4. 替换组装
        new_lines = lines[:start_line - 1]
        if aligned_code:
            new_lines.append(aligned_code + '\n')
        new_lines.extend(lines[actual_end_line:])
        new_file_content = "".join(new_lines)

        # 5. AST 预检
        if abs_path.endswith('.py'):
            try:
                ast.parse(new_file_content)
            except IndentationError as e:
                error_msg = (
                    f"❌ 修改被安全撤销：引发了 IndentationError (缩进异常)。\n"
                    f"报错行号: 第 {e.lineno} 行\n"
                    f"报错代码: {e.text.strip() if e.text else '未知'}\n"
                    f"系统提示：可能是你生成的代码首行缩进与上下文不匹配，或者代码块内部的相对缩进混乱导致。请重新查阅文件上下文，确保 `new_code` 包含严谨的缩进。"
                )
                return error_msg
            except SyntaxError as e:
                error_msg = format_syntax_error(e, "修改已被系统安全撤销，你提交的修改会导致文件出现 SyntaxError。")
                return error_msg
            except Exception as e:
                return f"❌ AST 解析异常: {str(e)}"

        # 6. 落盘保存
        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
        except Exception as e:
            return f"❌ 写入文件失败: {str(e)}"

        return f"✅ 成功: 文件 {file_path} 的第 {start_line} 到 {actual_end_line} 行已成功被替换。"

    def _insert_code_impl(self, file_path: str, line_number: int, new_code: str) -> str:
        """
        在指定文件的特定行之后插入新代码。
        【进阶用法：创建新文件】
        如果目标文件不存在，你可以使用此工具来创建新文件。
        此时，必须严格设置 line_number=0。工具会自动为你创建目录结构并写入 new_code。

        【注意】：请严格控制传入 new_code 的缩进级别，使其与上下文完美匹配。工具内置 AST 预检，缩进错误或语法错误将被拒绝写入。

        参数:
        - file_path: 文件相对路径 (必填)
        - line_number: 锚点行号。代码将插入到这一行之后。传入 0 表示插入到文件最开头 (必填)
        - new_code: 要插入的纯 Python 代码逻辑 (必填)
        """
        # 安全路径校验
        try:
            abs_path = get_secure_path(self.project_root, file_path)
        except PermissionError as e:
            return f"❌ 安全错误: {str(e)}"

        # 清洗 LLM 生成的残留 Markdown 符号
        cleaned_code = self._clean_llm_code(new_code)
        # 确保插入的代码自带独立换行
        if not cleaned_code.endswith('\n'):
            cleaned_code += '\n'

        # 分支 A: 文件不存在 -> 执行新建文件逻辑
        if not os.path.exists(abs_path):
            if line_number != 0:
                return f"❌ 错误: 文件 {file_path} 不存在。如果要创建新文件，请严格将 line_number 设为 0。"
            # AST 预检
            if abs_path.endswith('.py'):
                try:
                    ast.parse(cleaned_code)
                except SyntaxError as e:
                    error_msg = format_syntax_error(e, "新建文件失败，你提交的代码存在 SyntaxError。")
                    return error_msg
            # 创建多级父目录并写入文件
            try:
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_code)
                return f"✅ 成功: 已自动创建新文件 {file_path} 并写入代码。"
            except Exception as e:
                return f"❌ 创建文件失败: {str(e)}"

        # 分支 B: 文件存在 -> 执行精准插入逻辑
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return f"❌ 错误: 无法读取文件内容 ({str(e)})"
        total_lines = len(lines)
        if line_number < 0 or line_number > total_lines:
            return f"❌ 错误: 行号超出范围。目标文件共有 {total_lines} 行，line_number 必须在 0 到 {total_lines} 之间。"
        # 切片并插入
        new_lines = lines[:line_number]
        new_lines.append(cleaned_code)
        new_lines.extend(lines[line_number:])
        new_file_content = "".join(new_lines)
        # AST 预检
        if abs_path.endswith('.py'):
            try:
                ast.parse(new_file_content)
            except SyntaxError as e:
                error_msg = format_syntax_error(e, "插入操作已被安全撤销，合并后的代码出现 SyntaxError。")
                return error_msg
        # 落盘保存
        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
        except Exception as e:
            return f"❌ 写入文件失败: {str(e)}"

        return f"✅ 成功: 在 {file_path} 的第 {line_number} 行之后插入了新代码。"

    def _delete_code_impl(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        精确删除指定文件中从 start_line 到 end_line (包含) 的代码块。
        
        【安全限制 1】：本工具被严禁用于删除整个文件。你必须保留文件的基本结构。
        【安全限制 2】：工具内置 AST 预检。如果你删除了父级控制流（如 if/def/try）却遗留了其内部的缩进代码，或者破坏了括号闭合，操作将被自动撤销。
        
        参数:
        - file_path: 文件相对路径 (必填)
        - start_line: 起始行号 (从 1 开始，必填)
        - end_line: 结束行号 (包含此行，必填)
        """
        
        # 1. 安全路径校验
        try:
            abs_path = get_secure_path(self.project_root, file_path)
        except PermissionError as e:
            return f"❌ 安全错误: {str(e)}"
        if not os.path.exists(abs_path):
            return f"❌ 错误: 文件不存在 ({file_path})"

        # 2. 读取文件并校验行号
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return f"❌ 错误: 无法读取文件内容 ({str(e)})"
        total_lines = len(lines)
        if start_line < 1 or end_line < start_line:
            return f"❌ 错误: 无效的行号范围。start_line 必须 >= 1，且 end_line 必须 >= start_line。"
        if start_line > total_lines:
            return f"❌ 错误: 无效的行号范围。文件共 {total_lines} 行。"
        actual_end_line = min(end_line, total_lines)

        # 3. 护栏：拦截全文件删除企图
        if start_line == 1 and actual_end_line == total_lines:
            return (
                f"❌ 拒绝操作: 严禁使用此工具清空或删除整个文件！\n"
                f"如果你的目的是重构，请使用 edit_code_block 替换，或仅删除特定的函数/类。"
            )

        # 4. 切片删除
        new_lines = lines[:start_line - 1] + lines[actual_end_line:]
        new_file_content = "".join(new_lines)

        # 5. AST 预检
        if abs_path.endswith('.py'):
            try:
                ast.parse(new_file_content)
            except SyntaxError as e:
                # 复用底层格式化组件
                error_msg = format_syntax_error(e, context_msg="删除操作已被安全撤销，删除后的代码出现 SyntaxError。")
                error_msg += "\n通常是因为你遗留了部分代码未删除，或破坏了原有的括号/结构。请重新评估需要删除的行号范围。"
                return error_msg

        # 6. 落盘保存
        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
        except Exception as e:
            return f"❌ 写入文件失败: {str(e)}"

        return f"✅ 成功: 文件 {file_path} 的第 {start_line} 到 {actual_end_line} 行已成功被删除。"

    def create_edit_code_block_tool(self):
        return StructuredTool.from_function(
            name="edit_code_block",
            description=EDIT_CODE_BLOCK_DESC,
            func=self._edit_code_block_impl,
            infer_schema=False,
            args_schema=EditCodeBlockSchema,
        )

    def create_insert_code_tool(self):
        return StructuredTool.from_function(
            name="insert_code",
            description=INSERT_CODE_DESC,
            func=self._insert_code_impl,
            infer_schema=False,
            args_schema=InsertCodeSchema,
        )

    def create_delete_code_tool(self):
        return StructuredTool.from_function(
            name="delete_code",
            description=DELETE_CODE_DESC,
            func=self._delete_code_impl,
            infer_schema=False,
            args_schema=DeleteCodeSchema,
        )
