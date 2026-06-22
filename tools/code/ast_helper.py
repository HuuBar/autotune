import ast

from typing import Dict, Optional, List, Any


def _get_signature(node: ast.AST, source_lines: List[str]) -> str:
    """精准提取节点的方法签名"""
    start_idx = node.lineno - 1
    if hasattr(node, 'body') and node.body:
        body_start_idx = node.body[0].lineno - 1
        if start_idx == body_start_idx:
            return source_lines[start_idx].split(':', 1)[0] + ':'
        else:
            return "\n".join(source_lines[start_idx:body_start_idx]).strip()
    else:
        end_idx = getattr(node, "end_lineno", node.lineno)
        return "\n".join(source_lines[start_idx:end_idx]).strip()

def get_ast_context(ast_tree: ast.AST, line_num: int) -> Dict[str, Optional[str]]:
    """
    提取指定行号的AST上下文，支持闭包嵌套解析
    返回示例:
    {"class_name": "ClassName", "function_name": "outer_func.inner_func"}
    """
    enclosing_classes = []
    enclosing_funcs = []

    # 遍历语法树寻找包含该行的节点
    for node in ast.walk(ast_tree):
        # 确保节点有起止行号属性
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            end_line = getattr(node, "end_lineno", float('inf'))
            if node.lineno <= line_num <= end_line:
                if isinstance(node, ast.ClassDef):
                    enclosing_classes.append((node.lineno, node.name))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    enclosing_funcs.append((node.lineno, node.name))

    # 按行号正序排列，得到自外向内的链路
    enclosing_classes.sort(key=lambda x: x[0])
    enclosing_funcs.sort(key=lambda x: x[0])
    # 组装类名 (取最内层的类)
    class_name = enclosing_classes[-1][1] if enclosing_classes else None
    # 组装函数名 (支持闭包，拼接全链路)
    function_name = ".".join([name for _, name in enclosing_funcs]) if enclosing_funcs else None

    return {
        "class_name": class_name,
        "function_name": function_name
    }

def get_definitions(
    file_path: str,
    ast_tree: ast.AST, 
    source_code: str, 
    target_name: str, 
    exact_match: bool = True
) -> List[Dict[str, Any]]:
    """
    供 find_definition 使用: 精准提取类、函数或方法的定义信息(签名、docstring、行号等)。
    """
    results = []
    # 1. 为所有节点动态挂载 parent 指针
    for node in ast.walk(ast_tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    source_lines = source_code.splitlines()
    # 2. 遍历语法树寻找目标
    for node in ast.walk(ast_tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            # 匹配逻辑
            is_match = (name == target_name) if exact_match else (target_name in name)
            if not is_match:
                continue
            # 3. 确定实体类型 (class / function / method)
            node_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
            parent_class = "无父类"
            if node_type == 'function':
                parent = getattr(node, 'parent', None)
                if parent and isinstance(parent, ast.ClassDef):
                    node_type = 'method'
                    parent_class = parent.name
            # 4. 提取 docstring & signature
            docstring = ast.get_docstring(node) or "无注释"
            sig = _get_signature(node, source_lines)
            # 5. 提取子元素的框架
            children_skeletons = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    child_sig = _get_signature(child, source_lines)
                    compact_sig = " ".join(child_sig.split())   # 将签名压缩为单行
                    children_skeletons.append(compact_sig)
            # 6. 组装结构化字典
            results.append({
                "file_path": file_path,
                "name": name,
                "type": node_type,
                "parent_class": parent_class,
                "signature": sig,
                "docstring": docstring,
                "children": children_skeletons,
                "line_start": node.lineno,
                "line_end": getattr(node, "end_lineno", node.lineno)
            })

    return results

def generate_file_skeleton(ast_tree: ast.AST, source_lines: List[str], file_path: str) -> str:
    """
    核心骨架提取逻辑：遍历 AST 树，提取模块顶层的类、方法和全局函数，组装成结构化的大纲文本。
    """
    total_lines = len(source_lines)
    classes_info = []
    functions_info = []

    for node in ast_tree.body:
        # 处理类定义
        if isinstance(node, ast.ClassDef):
            class_sig = _get_signature(node, source_lines)
            class_line_range = f"(Line {node.lineno}-{getattr(node, 'end_lineno', node.lineno)})"
            class_header = f"{class_sig} {class_line_range}"
            class_doc = ast.get_docstring(node)
            if class_doc:
                first_line = class_doc.strip().split('\n')[0].strip()
                class_header += f' (desc: {first_line})'
            # 提取类的方法
            methods_str_list = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_sig = _get_signature(child, source_lines)
                    compact_method_sig = " ".join(method_sig.split())
                    method_line_range = f"(Line {child.lineno}-{getattr(child, 'end_lineno', child.lineno)})"
                    method_item = f"    |-- {compact_method_sig} {method_line_range}"
                    # 提取 docstring
                    method_doc = ast.get_docstring(child)
                    if method_doc:
                        m_first_line = method_doc.strip().split('\n')[0].strip()
                        if len(m_first_line) > 8:
                            method_item += f'\n        """{m_first_line}"""'
                    methods_str_list.append(method_item)
            methods_block = "\n".join(methods_str_list) if methods_str_list else "    |-- (无方法)"
            classes_info.append(f"{class_header}\n{methods_block}")
        # 处理全局函数定义
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_sig = _get_signature(node, source_lines)
            func_line_range = f"(Line {node.lineno}-{getattr(node, 'end_lineno', node.lineno)})"
            func_item = f"{func_sig} {func_line_range}"
            func_doc = ast.get_docstring(node)
            if func_doc:
                f_first_line = func_doc.strip().split('\n')[0].strip()
                if len(f_first_line) > 8:
                    func_item += f'\n    """{f_first_line}"""'
            functions_info.append(func_item)

    # 组装最终文本
    result = [f"文件: {file_path} (共 {total_lines} 行)\n"]
    if classes_info:
        result.append("[Classes]")
        result.append("\n\n".join(classes_info))
        result.append("\n")
    if functions_info:
        result.append("[Functions]")
        result.append("\n\n".join(functions_info))
    if not classes_info and not functions_info:
        result.append("（文件内未找到任何类或全局函数的定义）")

    return "\n".join(result)

def format_syntax_error(e: SyntaxError, context_msg: str) -> str:
    """
    将 SyntaxError 格式化为结构化的报错反馈。
    参数:
    - e: 捕获到的 SyntaxError 对象
    - context_msg: 触发异常的上下文说明
    """
    error_msg = (
        f"❌ 语法错误拦截: {context_msg}\n"
        f"【客观诊断信息】\n"
        f"- 报错详情: {e.msg}\n"
        f"- 报错行号: 在文件的第 {e.lineno} 行附近。\n"
    )
    if e.text:
        error_msg += f"- 出错代码: {e.text.strip()}\n"
    error_msg += f"请根据以上客观报错信息，重新审视你提交的内容，修复错误后重新调用本工具。"
    return error_msg
