"""W3-1：编辑器按后缀分发的格式校验（.py AST / .json / .yaml）。"""
import json
import os

import pytest

from tests.conftest import global_config
from tools.code.editor import EditorTools


@pytest.fixture()
def workspace(tmp_path):
    global_config["workspace"]["root_dir"] = str(tmp_path)
    return tmp_path


@pytest.fixture()
def editor(workspace):
    return EditorTools()


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestJsonValidation:
    def test_invalid_json_edit_blocked_and_file_untouched(self, editor, workspace):
        target = workspace / "configs" / "train.json"
        original = '{\n  "lr": 0.1,\n  "depth": 6\n}\n'
        _write(target, original)

        res = editor._edit_code_block_impl(
            file_path="configs/train.json",
            start_line=2,
            end_line=2,
            new_code='"lr": 0.1,,,',  # 非法 JSON：多余逗号
        )

        assert "JSON" in res and "撤销" in res
        assert "行" in res  # 返回错误位置
        assert target.read_text(encoding="utf-8") == original  # 文件未被污染

    def test_valid_json_edit_allowed(self, editor, workspace):
        target = workspace / "configs" / "train.json"
        _write(target, '{\n  "lr": 0.1,\n  "depth": 6\n}\n')

        res = editor._edit_code_block_impl(
            file_path="configs/train.json",
            start_line=2,
            end_line=2,
            new_code='"lr": 0.05,',
        )

        assert res.startswith("✅")
        json.loads(target.read_text(encoding="utf-8"))  # 结果仍是合法 JSON

    def test_invalid_json_insert_blocked(self, editor, workspace):
        target = workspace / "data.json"
        original = '{"a": 1}\n'
        _write(target, original)

        res = editor._insert_code_impl(
            file_path="data.json",
            line_number=1,
            new_code='{broken: true]',
        )

        assert "JSON" in res and "撤销" in res
        assert target.read_text(encoding="utf-8") == original

    def test_invalid_json_new_file_blocked(self, editor, workspace):
        target = workspace / "new.json"
        res = editor._insert_code_impl(
            file_path="new.json",
            line_number=0,
            new_code='{"a": }',
        )
        assert "JSON" in res
        assert not target.exists()

    def test_invalid_json_delete_blocked(self, editor, workspace):
        target = workspace / "d.json"
        original = '{\n  "a": 1\n}\n'
        _write(target, original)
        # 删掉第 1 行 '{' 会破坏 JSON 结构
        res = editor._delete_code_impl(file_path="d.json", start_line=1, end_line=1)
        assert "JSON" in res and "撤销" in res
        assert target.read_text(encoding="utf-8") == original


class TestYamlValidation:
    def test_invalid_yaml_edit_blocked_and_file_untouched(self, editor, workspace):
        target = workspace / "config.yaml"
        original = "a: 1\nb: 2\n"
        _write(target, original)

        res = editor._edit_code_block_impl(
            file_path="config.yaml",
            start_line=2,
            end_line=2,
            new_code='b: "unclosed',
        )

        assert "YAML" in res and "撤销" in res
        assert target.read_text(encoding="utf-8") == original

    def test_valid_yaml_edit_allowed(self, editor, workspace):
        target = workspace / "config.yml"
        _write(target, "a: 1\nb: 2\n")

        res = editor._edit_code_block_impl(
            file_path="config.yml",
            start_line=2,
            end_line=2,
            new_code="b: 3",
        )

        assert res.startswith("✅")
        assert "b: 3" in target.read_text(encoding="utf-8")


class TestPythonRegression:
    def test_invalid_python_still_blocked(self, editor, workspace):
        target = workspace / "mod.py"
        original = "x = 1\ny = 2\n"
        _write(target, original)

        res = editor._edit_code_block_impl(
            file_path="mod.py",
            start_line=2,
            end_line=2,
            new_code="def broken(:",
        )

        assert "撤销" in res
        assert target.read_text(encoding="utf-8") == original


class TestOtherSuffixPassthrough:
    def test_unlisted_suffix_not_validated(self, editor, workspace):
        target = workspace / "notes.txt"
        _write(target, "hello\nworld\n")

        res = editor._edit_code_block_impl(
            file_path="notes.txt",
            start_line=2,
            end_line=2,
            new_code="{not: json at all[",
        )

        assert res.startswith("✅")  # 非结构化后缀不做校验，保持原行为
