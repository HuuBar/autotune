import os
import re
import json
import argparse

REASONING_CONTENT_KEY = "reasoning_content"

def decode_escaped_string(text: str) -> str:
    """安全地解码包含转义字符的文本，完美修复 utf-8 变 latin-1 的乱码"""
    if not text:
        return text
    try:
        text_bytes = text.encode('utf-8')
        decoded_str = text_bytes.decode('unicode_escape')
        try:
            return decoded_str.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return decoded_str
    except Exception:
        return text.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")

def parse_chat_message(text: str) -> dict:
    res = {"content": "", REASONING_CONTENT_KEY: "", "tool_calls": []}
    rc_match = re.search(fr"{REASONING_CONTENT_KEY}=(['\"])(.*?)\1(?:, |\)$)", text, re.DOTALL)
    if rc_match:
        res[REASONING_CONTENT_KEY] = decode_escaped_string(rc_match.group(2))

    content_match = re.search(r"content=(['\"])(.*?)\1, refusal=", text, re.DOTALL)
    if content_match:
        res["content"] = decode_escaped_string(content_match.group(2))

    tc_pattern = re.compile(
        r"ChatCompletionMessageFunctionToolCall\(id='(.*?)', function=Function\(arguments='(.*?)', name='(.*?)'\)"
    )
    for tc_match in tc_pattern.finditer(text):
        args_str = decode_escaped_string(tc_match.group(2))
        args_dict = {}
        try:
            args_dict = json.loads(args_str)
        except json.JSONDecodeError:
            args_dict = {"raw_args": args_str}

        res["tool_calls"].append({
            "id": tc_match.group(1),
            "name": tc_match.group(3),
            "arguments": args_dict,
            "result": None
        })
    return res

def generate_html_viewer(log_path: str, output_html_path: str, template_path: str = "trace_template.html"):
    if not os.path.exists(template_path):
        print(f"❌ 错误: 找不到HTML模板文件 {template_path}")
        return
    if not os.path.exists(log_path):
        print(f"❌ 错误: 找不到日志文件 {log_path}")
        return
    print(f"⏳ 正在加载模板: {template_path} ...")
    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()
    print(f"⏳ 正在解析日志文件: {log_path} ...")
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+(?:DEBUG|INFO).*?\((.*?)\)\s+LLM\s+(input prompt|response):\s*\n", 
        re.MULTILINE
    )

    matches = list(pattern.finditer(content))
    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(content)
        blocks.append({
            "time": match.group(1),
            "agent": match.group(2),
            "type": match.group(3),
            "text": content[start:end].strip()
        })

    trace_data = []
    current_session = None
    pending_tool_calls = {}
    mock_vfs = {"todo.md": "", "plan_log.md": ""}

    for block in blocks:
        if block['type'] == 'response':
            agent = block['agent']
            
            if not current_session or current_session['agent'] != agent:
                current_session = {
                    "id": f"session_{len(trace_data)}",
                    "agent": agent,
                    "start_time": block['time'],
                    "turns": []
                }
                trace_data.append(current_session)

            parsed_msg = parse_chat_message(block['text'])
            turn = {
                "id": f"turn_{len(trace_data)}_{len(current_session['turns'])}",
                "time": block['time'],
                "content": parsed_msg["content"],
                "reasoning_content": parsed_msg[REASONING_CONTENT_KEY],
                "tool_calls": parsed_msg["tool_calls"],
                "state_updates": {} 
            }

            for tc in turn["tool_calls"]:
                pending_tool_calls[tc["id"]] = tc
                args = tc["arguments"]

                file_path = args.get("file_path", "")
                filename = None
                if "todo.md" in file_path:
                    filename = "todo.md"
                elif "plan_log.md" in file_path:
                    filename = "plan_log.md"
                if not filename:
                    continue

                if tc["name"] == "write_file":
                    content_val = args.get("content", "")
                    mock_vfs[filename] = content_val
                    turn["state_updates"][filename] = mock_vfs[filename]
                elif tc["name"] == "edit_file":
                    try:
                        old_string = args.get("old_string", "")
                        new_string = args.get("new_string", "")
                        current_content = mock_vfs[filename]
                        if old_string and old_string in current_content:
                            parts = current_content.rsplit(old_string, 1)
                            mock_vfs[filename] = new_string.join(parts)
                            turn["state_updates"][filename] = mock_vfs[filename]
                    except Exception as e:
                        pass
            current_session['turns'].append(turn)

        elif block['type'] == 'input prompt':
            try:
                data = json.loads(block['text'])
                for msg in data.get('messages', []):
                    if msg.get('role') == 'tool' and msg.get('tool_call_id'):
                        tc_id = msg['tool_call_id']
                        if tc_id in pending_tool_calls:
                            result_val = msg.get('content')
                            if isinstance(result_val, list):
                                result_val = "\n".join(str(i) for i in result_val)
                            pending_tool_calls[tc_id]['result'] = str(result_val)
                            del pending_tool_calls[tc_id]
            except json.JSONDecodeError:
                pass

    # 将 Python 字典转为 JSON 字符串
    json_str = json.dumps(trace_data, ensure_ascii=False, indent=2)
    # 替换 HTML 模板中的占位符
    final_html = html_template.replace("__INJECT_TRACE_DATA_HERE__", json_str)

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"✅ 生成成功！共提取了 {len(trace_data)} 个 Agent 回合。")
    print(f"📂 可视化页面已保存至: {os.path.abspath(output_html_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTune Agent 日志可视化生成器")
    parser.add_argument("--log", type=str, default="autotune.log", help="纯文本日志文件路径")
    parser.add_argument("--out", type=str, default="trace_viewer.html", help="HTML可视化文件路径")
    parser.add_argument("--template", type=str, default="trace_template.html", help="HTML模板文件路径")

    args = parser.parse_args()
    generate_html_viewer(args.log, args.out, args.template)